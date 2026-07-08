use std::collections::HashMap;

use crate::entities::{
    BatchPlanEntry, GlossaryEntry, PipelineOptions, PipelineResult, SubtitleDocument,
    SubtitleSegment, TranslationLine, Usage,
};
use crate::memory::ContextMemory;
use crate::error::{CoreError, CoreResult};
use crate::languages::normalize_language_name;
use crate::ports::{BackendPayload, BatchShardKind, ChatMessage, DashboardSink, LlmBackend, RuntimeStore};
use crate::validation::{validate_full_alignment, validate_translation_batch};

pub struct SubtitlePipeline<B, D> {
    backend: B,
    dashboard: D,
    options: PipelineOptions,
    memory: ContextMemory,
    store: Option<Box<dyn RuntimeStore>>,
    /// Normalised-key → translation text cache loaded from the runtime store.
    translation_memory: HashMap<String, String>,
    translation_memory_hits: usize,
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
            memory: ContextMemory::new(),
            store: None,
            translation_memory: HashMap::new(),
            translation_memory_hits: 0,
        }
    }

    /// Attach a runtime store for glossary/TM persistence.
    pub fn with_store(mut self, store: Box<dyn RuntimeStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn run_document(&mut self, document: &SubtitleDocument) -> CoreResult<PipelineRun> {
        if self.options.batch_size == 0 {
            return Err(CoreError::InvalidTranslation(
                "batch size must be greater than zero".to_owned(),
            ));
        }

        // Load persisted glossary into context memory at start.
        if let Some(ref store) = self.store {
            let entries = store.load_glossary()?;
            self.memory.load_glossary(&entries);
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

        // Load translation memory from the runtime store at start.
        if let Some(ref store) = self.store {
            let tm_entries = store.load_translation_memory()?;
            for (key, text) in tm_entries {
                self.translation_memory.insert(key, text);
            }
        }

        let mut translated_segments = Vec::with_capacity(document.segments.len());
        let mut usage = Usage::default();
        for (batch_index, batch) in batches.iter().enumerate() {
            // TM lookup: split batch into TM-matched (by segment text) and pending.
            let (tm_hits, pending): (HashMap<String, String>, Vec<SubtitleSegment>) = {
                let mut hits = HashMap::new();
                let mut rest = Vec::new();
                for seg in batch.iter() {
                    let key = translation_memory_key(&seg.text);
                    if let Some(text) = self.translation_memory.get(&key) {
                        hits.insert(seg.id.clone(), text.clone());
                    } else {
                        rest.push(seg.clone());
                    }
                }
                (hits, rest)
            };

            let new_lines: Vec<TranslationLine> = if pending.is_empty() {
                // All matched — short-circuit, no LLM call.
                Vec::new()
            } else {
                let br = self.translate_batch(batch_index + 1, &pending)?;
                validate_translation_batch(&pending, &br.lines)?;
                usage.add(br.usage);
                self.dashboard.add_usage(br.usage);
                self.memory.update(&br.summary, &br.glossary_updates);
                br.lines
            };

            let merged: Vec<TranslationLine> = merge_translation_lines(batch, &tm_hits, &new_lines);
            self.translation_memory_hits += tm_hits.len();

            // Persist glossary, TM, batch segments after each batch.
            if let Some(ref store) = self.store {
                let glossary_entries: Vec<(String, String)> = self
                    .memory
                    .glossary
                    .iter()
                    .map(|(source, target)| (source.clone(), target.clone()))
                    .collect();
                store.save_glossary(&glossary_entries)?;
                store.save_batch_segments(
                    BatchShardKind::Translated,
                    batch_index + 1,
                    batch,
                )?;
                let tm_entries: Vec<(String, String)> = self
                    .translation_memory
                    .iter()
                    .map(|(key, text)| (key.clone(), text.clone()))
                    .collect();
                store.save_translation_memory(&tm_entries)?;
            }

            translated_segments.extend(apply_lines(batch, &merged));
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
                translation_memory_hits: self.translation_memory_hits,
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
        let messages = build_translation_messages(&self.options, batch_index, batch, &self.memory);
        let backend_result = self.backend.generate_json(&messages)?;
        let BackendPayload::Translation(result) = backend_result.payload;
        Ok(BatchWithUsage {
            lines: result.lines,
            summary: result.summary,
            glossary_updates: result.glossary_updates,
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
    summary: String,
    glossary_updates: Vec<GlossaryEntry>,
    usage: Usage,
}

fn build_translation_messages(
    options: &PipelineOptions,
    batch_index: usize,
    batch: &[SubtitleSegment],
    memory: &ContextMemory,
) -> Vec<ChatMessage> {
    let mut context = serde_json::json!({
        "src": options.source_language,
        "tgt": options.target_language,
        "batch_index": batch_index,
        "fast": options.fast_mode,
    });
    if !options.fast_mode {
        context["rules"] = serde_json::Value::Array(
            memory
                .style_rules
                .iter()
                .cloned()
                .map(serde_json::Value::String)
                .collect(),
        );
        let recent = memory.recent_summaries_for_prompt();
        if !recent.is_empty() {
            context["recent"] = serde_json::Value::Array(
                recent
                    .iter()
                    .cloned()
                    .map(serde_json::Value::String)
                    .collect(),
            );
        }
        let batch_texts: Vec<&str> = batch
            .iter()
            .map(|segment| segment.text.as_str())
            .filter(|text| !text.is_empty())
            .collect();
        let glossary = memory.select_relevant_glossary(&batch_texts);
        if !glossary.is_empty() {
            let map: serde_json::Map<String, serde_json::Value> = glossary
                .into_iter()
                .map(|(source, target)| (source, serde_json::Value::String(target)))
                .collect();
            context["glossary"] = serde_json::Value::Object(map);
        }
    }
    let lines: Vec<serde_json::Value> = batch
        .iter()
        .map(|segment| serde_json::json!({"id": segment.id, "text": segment.text}))
        .collect();
    let batch_payload = serde_json::json!({"lines": lines});

    let context_json = serde_json::to_string(&context).unwrap_or_default();
    let batch_json = serde_json::to_string(&batch_payload).unwrap_or_default();
    let user = format!(
        "CONTEXT_JSON_START{context_json}CONTEXT_JSON_END\nBATCH_JSON_START{batch_json}BATCH_JSON_END"
    );

    vec![
        ChatMessage::system("TASK_START\ntranslate_subtitles\nTASK_END"),
        ChatMessage::user(user),
    ]
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

/// Normalise a subtitle text for translation-memory lookup.
/// Mirrors Python `pipeline.py::_translation_memory_key`:
///   lower-case → collapse whitespace → attach punctuation.
pub fn translation_memory_key(text: &str) -> String {
    let lower = text.trim().to_lowercase();
    // Collapse whitespace runs to a single space.
    let mut collapsed = String::with_capacity(lower.len());
    let mut prev_was_space = false;
    for ch in lower.chars() {
        if ch.is_whitespace() {
            if !prev_was_space {
                collapsed.push(' ');
                prev_was_space = true;
            }
        } else {
            collapsed.push(ch);
            prev_was_space = false;
        }
    }
    // Remove spaces before punctuation (,.!?;:).
    let mut attached = String::with_capacity(collapsed.len());
    let mut chars = collapsed.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == ' ' && chars.peek().is_some_and(|&next| matches!(next, ',' | '.' | '!' | '?' | ';' | ':')) {
            continue; // skip space before punctuation
        }
        attached.push(ch);
    }
    attached
}

/// Interleave TM-matched lines with LLM-generated lines in source order.
/// For each segment in the batch, prefer a TM hit, fall back to the new line,
/// and log a `TranslationLine` with the segment's id either way.
fn merge_translation_lines(
    batch: &[SubtitleSegment],
    tm_hits: &HashMap<String, String>,
    new_lines: &[TranslationLine],
) -> Vec<TranslationLine> {
    batch
        .iter()
        .map(|seg| {
            if let Some(translation) = tm_hits.get(&seg.id) {
                TranslationLine {
                    id: seg.id.clone(),
                    translation: translation.clone(),
                }
            } else {
                new_lines
                    .iter()
                    .find(|line| line.id == seg.id)
                    .cloned()
                    .unwrap_or_else(|| TranslationLine {
                        id: seg.id.clone(),
                        translation: String::new(),
                    })
            }
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
                .split("BATCH_JSON_START")
                .nth(1)
                .and_then(|value| value.split("BATCH_JSON_END").next())
                .ok_or_else(|| CoreError::Data("missing batch json".to_owned()))?;
            let parsed: serde_json::Value = serde_json::from_str(body)
                .map_err(|err| CoreError::Data(format!("invalid batch json: {err}")))?;
            let lines = parsed["lines"]
                .as_array()
                .ok_or_else(|| CoreError::Data("missing lines array".to_owned()))?
                .iter()
                .map(|entry| {
                    let id = entry["id"].as_str().unwrap_or_default().to_owned();
                    let text = entry["text"].as_str().unwrap_or_default().to_owned();
                    let translation = if text.trim().is_empty() {
                        String::new()
                    } else {
                        format!("[ECHO] {text}")
                    };
                    TranslationLine { id, translation }
                })
                .collect();
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
    fn tm_key_normalizes_and_attaches_punctuation() {
        assert_eq!(translation_memory_key("Hello, world!"), "hello, world!");
        assert_eq!(translation_memory_key("  spaced   out  "), "spaced out");
        assert_eq!(translation_memory_key("A;B."), "a;b.");
        assert_eq!(translation_memory_key("\t\n  "), "");
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
