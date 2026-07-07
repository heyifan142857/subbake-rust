use crate::entities::{
    BatchPlanEntry, PipelineOptions, PipelineResult, SubtitleDocument, SubtitleSegment,
    TranslationLine, Usage,
};
use crate::error::{CoreError, CoreResult};
use crate::languages::normalize_language_name;
use crate::ports::{BackendPayload, ChatMessage, DashboardSink, LlmBackend};
use crate::validation::{validate_full_alignment, validate_translation_batch};

pub struct SubtitlePipeline<B, D> {
    backend: B,
    dashboard: D,
    options: PipelineOptions,
}

impl<B, D> SubtitlePipeline<B, D>
where
    B: LlmBackend,
    D: DashboardSink,
{
    pub fn new(backend: B, dashboard: D, mut options: PipelineOptions) -> Self {
        options.source_language = normalize_language_name(&options.source_language, true);
        options.target_language = normalize_language_name(&options.target_language, false);
        Self {
            backend,
            dashboard,
            options,
        }
    }

    pub fn run_document(&mut self, document: &SubtitleDocument) -> CoreResult<PipelineRun> {
        if self.options.batch_size == 0 {
            return Err(CoreError::InvalidTranslation(
                "batch size must be greater than zero".to_owned(),
            ));
        }

        let batches = chunk_segments(&document.segments, self.options.batch_size);
        let planned_batches = build_batch_plan(&batches);
        if self.options.dry_run {
            return Ok(PipelineRun {
                result: PipelineResult {
                    output_path: None,
                    batches_translated: 0,
                    review_batches: 0,
                    usage: Usage::default(),
                    dry_run: true,
                    planned_batches,
                    cache_hits: 0,
                    resumed_translation_batches: 0,
                    resumed_review_batches: 0,
                    translation_memory_hits: 0,
                    state_path: None,
                    glossary_path: None,
                    agent_repairs: Vec::new(),
                },
                translated_segments: Vec::new(),
            });
        }

        self.dashboard.set_total_steps(2 + batches.len());
        self.dashboard.mark_running("TRANSLATE");

        let mut translated_segments = Vec::with_capacity(document.segments.len());
        let mut usage = Usage::default();
        for (batch_index, batch) in batches.iter().enumerate() {
            let batch_result = self.translate_batch(batch_index + 1, batch)?;
            validate_translation_batch(batch, &batch_result.lines)?;
            usage.add(batch_result.usage);
            self.dashboard.add_usage(batch_result.usage);

            translated_segments.extend(apply_lines(batch, &batch_result.lines));
        }

        validate_full_alignment(&document.segments, &translated_segments)?;
        self.dashboard.mark_done("TRANSLATE");

        Ok(PipelineRun {
            result: PipelineResult {
                output_path: self.options.output_path.clone(),
                batches_translated: batches.len(),
                review_batches: 0,
                usage,
                dry_run: false,
                planned_batches,
                cache_hits: 0,
                resumed_translation_batches: 0,
                resumed_review_batches: 0,
                translation_memory_hits: 0,
                state_path: None,
                glossary_path: self.options.glossary_path.clone(),
                agent_repairs: Vec::new(),
            },
            translated_segments,
        })
    }

    fn translate_batch(
        &mut self,
        batch_index: usize,
        batch: &[SubtitleSegment],
    ) -> CoreResult<BatchWithUsage> {
        let messages = build_translation_messages(&self.options, batch_index, batch);
        let backend_result = self.backend.generate_json(&messages)?;
        let BackendPayload::Translation(result) = backend_result.payload;
        Ok(BatchWithUsage {
            lines: result.lines,
            usage: backend_result.usage,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineRun {
    pub result: PipelineResult,
    pub translated_segments: Vec<SubtitleSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BatchWithUsage {
    lines: Vec<TranslationLine>,
    usage: Usage,
}

fn build_translation_messages(
    options: &PipelineOptions,
    batch_index: usize,
    batch: &[SubtitleSegment],
) -> Vec<ChatMessage> {
    let mut user = String::new();
    user.push_str("CONTEXT_START\n");
    user.push_str(&format!(
        "source_language={}\n",
        escape_field(&options.source_language)
    ));
    user.push_str(&format!(
        "target_language={}\n",
        escape_field(&options.target_language)
    ));
    user.push_str(&format!("batch_index={batch_index}\n"));
    user.push_str(&format!("fast={}\n", options.fast_mode));
    user.push_str("CONTEXT_END\n");
    user.push_str("BATCH_LINES_START\n");
    for segment in batch {
        user.push_str(&escape_field(&segment.id));
        user.push('\t');
        user.push_str(&escape_field(&segment.text));
        user.push('\n');
    }
    user.push_str("BATCH_LINES_END");

    vec![
        ChatMessage::system("TASK_START\ntranslate_subtitles\nTASK_END"),
        ChatMessage::user(user),
    ]
}

pub fn escape_field(value: &str) -> String {
    let mut output = String::new();
    for ch in value.chars() {
        match ch {
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            ch => output.push(ch),
        }
    }
    output
}

pub fn unescape_field(value: &str) -> CoreResult<String> {
    let mut output = String::new();
    let mut chars = value.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            output.push(ch);
            continue;
        }
        match chars.next() {
            Some('\\') => output.push('\\'),
            Some('n') => output.push('\n'),
            Some('r') => output.push('\r'),
            Some('t') => output.push('\t'),
            Some(other) => {
                return Err(CoreError::Data(format!(
                    "unsupported escape sequence \\{other}"
                )));
            }
            None => return Err(CoreError::Data("trailing escape character".to_owned())),
        }
    }
    Ok(output)
}

fn chunk_segments(segments: &[SubtitleSegment], batch_size: usize) -> Vec<Vec<SubtitleSegment>> {
    segments
        .chunks(batch_size)
        .map(<[SubtitleSegment]>::to_vec)
        .collect()
}

fn build_batch_plan(batches: &[Vec<SubtitleSegment>]) -> Vec<BatchPlanEntry> {
    batches
        .iter()
        .enumerate()
        .filter_map(|(index, batch)| {
            let first = batch.first()?;
            let last = batch.last()?;
            Some(BatchPlanEntry {
                index: index + 1,
                size: batch.len(),
                first_id: first.id.clone(),
                last_id: last.id.clone(),
            })
        })
        .collect()
}

fn apply_lines(source: &[SubtitleSegment], lines: &[TranslationLine]) -> Vec<SubtitleSegment> {
    source
        .iter()
        .map(|segment| {
            let translation = lines
                .iter()
                .find(|line| line.id == segment.id)
                .map(|line| line.translation.clone())
                .unwrap_or_default();
            let mut translated = segment.clone();
            translated.text = translation;
            translated
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::entities::{BatchTranslationResult, GlossaryEntry};
    use crate::ports::{BackendJsonResult, NoopDashboard};

    use super::*;

    struct EchoBackend;

    impl LlmBackend for EchoBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "echo"
        }

        fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
            let prompt = messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let body = prompt
                .split("BATCH_LINES_START")
                .nth(1)
                .and_then(|value| value.split("BATCH_LINES_END").next())
                .ok_or_else(|| CoreError::Data("missing batch lines".to_owned()))?;
            let mut lines = Vec::new();
            for raw_line in body.lines().filter(|line| !line.trim().is_empty()) {
                let (id, text) = raw_line
                    .split_once('\t')
                    .ok_or_else(|| CoreError::Data("missing tab separator".to_owned()))?;
                lines.push(TranslationLine {
                    id: unescape_field(id)?,
                    translation: format!("[ECHO] {}", unescape_field(text)?),
                });
            }
            Ok(BackendJsonResult {
                payload: BackendPayload::Translation(BatchTranslationResult {
                    lines,
                    summary: "ok".to_owned(),
                    glossary_updates: Vec::<GlossaryEntry>::new(),
                }),
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    total_tokens: 2,
                },
            })
        }
    }

    #[test]
    fn pipeline_translates_document_batches() {
        let document = SubtitleDocument {
            path: "clip.txt".into(),
            format: "txt".to_owned(),
            segments: vec![SubtitleSegment {
                id: "1".to_owned(),
                text: "hello".to_owned(),
                start: None,
                end: None,
                identifier: None,
                settings: None,
            }],
            header: None,
            passthrough_blocks: Vec::new(),
        };
        let mut options = PipelineOptions::new("clip.txt".into());
        options.batch_size = 1;
        let mut pipeline = SubtitlePipeline::new(EchoBackend, NoopDashboard, options);
        let run = pipeline.run_document(&document).expect("run");

        assert_eq!(run.result.batches_translated, 1);
        assert_eq!(run.translated_segments[0].text, "[ECHO] hello");
    }
}
