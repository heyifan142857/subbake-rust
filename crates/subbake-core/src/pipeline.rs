use std::collections::HashMap;

use crate::entities::{
    BatchPlanEntry, GlossaryEntry, PipelineOptions, PipelineResult, SubtitleDocument,
    SubtitleSegment, TranslationLine, Usage,
};
use crate::error::{CoreError, CoreResult};
use crate::languages::normalize_language_name;
use crate::memory::ContextMemory;
use crate::ports::{
    BackendPayload, BatchShardKind, CacheStage, ChatMessage, DashboardSink, LlmBackend,
    RuntimeStore,
};
use crate::storage::{
    InputSignature, JsonValue, ResumeSnapshot, RunState, build_request_hash,
    build_translation_fingerprint,
};
use crate::validation::{validate_full_alignment, validate_translation_batch};

pub struct SubtitlePipeline<B, D> {
    backend: B,
    dashboard: D,
    options: PipelineOptions,
    memory: ContextMemory,
    store: Option<Box<dyn RuntimeStore>>,
    input_signature: Option<InputSignature>,
    /// Normalised-key → translation text cache loaded from the runtime store.
    translation_memory: HashMap<String, String>,
    translation_memory_hits: usize,
    cache_hits: usize,
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
            input_signature: None,
            translation_memory: HashMap::new(),
            translation_memory_hits: 0,
            cache_hits: 0,
        }
    }

    /// Attach a runtime store for glossary/TM persistence.
    pub fn with_store(mut self, store: Box<dyn RuntimeStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn with_input_signature(mut self, input_signature: InputSignature) -> Self {
        self.input_signature = Some(input_signature);
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
        let state_path = self
            .store
            .as_ref()
            .map(|store| store.paths().state_path.clone());
        let glossary_path = self
            .store
            .as_ref()
            .map(|store| store.paths().glossary_path.clone())
            .or_else(|| self.options.glossary_path.clone());
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
                    state_path,
                    glossary_path,
                    agent_repairs: Vec::new(),
                },
                translated_segments: Vec::new(),
            });
        }

        self.dashboard.set_total_steps(2 + batches.len());
        self.dashboard.mark_running("TRANSLATE");

        // Load translation memory from the runtime store at start.
        if self.options.use_cache
            && let Some(ref store) = self.store
        {
            let tm_entries = store.load_translation_memory()?;
            for (key, text) in tm_entries {
                self.translation_memory.insert(key, text);
            }
        }

        let resume = self.load_resume_snapshot(&batches)?;
        let mut translated_segments = resume.translated_segments;
        translated_segments.reserve(
            document
                .segments
                .len()
                .saturating_sub(translated_segments.len()),
        );
        let mut usage = resume.usage;
        if usage != Usage::default() {
            self.dashboard.add_usage(usage);
        }
        for (batch_index, batch) in batches
            .iter()
            .enumerate()
            .skip(resume.translation_batches_completed)
        {
            // TM lookup: split batch into TM-matched (by segment text) and pending.
            let (tm_hits, pending): (HashMap<String, String>, Vec<SubtitleSegment>) = {
                let mut hits = HashMap::new();
                let mut rest = Vec::new();
                for seg in batch.iter() {
                    let key = translation_memory_key(&seg.text);
                    if !key.is_empty()
                        && self.options.use_cache
                        && let Some(text) = self.translation_memory.get(&key)
                    {
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
            let translated_batch = apply_lines(batch, &merged);
            self.translation_memory_hits += tm_hits.len();
            if self.options.use_cache {
                update_translation_memory(&mut self.translation_memory, batch, &translated_batch);
            }

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
                    &translated_batch,
                )?;
                if self.options.use_cache {
                    let tm_entries: Vec<(String, String)> = self
                        .translation_memory
                        .iter()
                        .map(|(key, text)| (key.clone(), text.clone()))
                        .collect();
                    store.save_translation_memory(&tm_entries)?;
                }
            }

            translated_segments.extend(translated_batch);
            self.save_run_state(
                batch_index + 1,
                resume.review_batches_completed,
                false,
                usage,
            )?;
        }

        validate_full_alignment(&document.segments, &translated_segments)?;
        self.save_run_state(batches.len(), resume.review_batches_completed, true, usage)?;
        self.dashboard.mark_done("TRANSLATE");

        Ok(PipelineRun {
            result: PipelineResult {
                output_path: self.options.output_path.clone(),
                batches_translated: batches.len(),
                review_batches: 0,
                usage,
                dry_run: false,
                planned_batches,
                cache_hits: self.cache_hits,
                resumed_translation_batches: resume.translation_batches_completed,
                resumed_review_batches: resume.review_batches_completed,
                translation_memory_hits: self.translation_memory_hits,
                state_path,
                glossary_path,
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
        let request_hash = build_request_hash(
            &self.options.provider,
            &self.options.model,
            CacheStage::Translate.as_str(),
            messages_json(&messages),
        );
        let cached_response = if self.options.use_cache {
            self.store
                .as_ref()
                .map(|store| store.load_cached_response(CacheStage::Translate, &request_hash))
                .transpose()?
                .flatten()
        } else {
            None
        };
        let cached = cached_response.is_some();
        let backend_result = match cached_response {
            Some(response) => {
                self.cache_hits += 1;
                response
            }
            None => self.backend.generate_json(&messages)?,
        };
        let BackendPayload::Translation(result) = &backend_result.payload;
        validate_translation_batch(batch, &result.lines)?;
        if self.options.use_cache
            && !cached
            && let Some(store) = self.store.as_ref()
        {
            store.save_cached_response(CacheStage::Translate, &request_hash, &backend_result)?;
        }
        let BackendPayload::Translation(result) = backend_result.payload;
        Ok(BatchWithUsage {
            lines: result.lines,
            summary: result.summary,
            glossary_updates: result.glossary_updates,
            usage: if cached {
                Usage::default()
            } else {
                backend_result.usage
            },
        })
    }

    fn load_resume_snapshot(
        &mut self,
        batches: &[Vec<SubtitleSegment>],
    ) -> CoreResult<ResumeSnapshot> {
        if !self.options.resume {
            return Ok(ResumeSnapshot::default());
        }
        let Some(input_signature) = self.input_signature.as_ref() else {
            return Ok(ResumeSnapshot::default());
        };
        let Some(store) = self.store.as_ref() else {
            return Ok(ResumeSnapshot::default());
        };
        let expected_fingerprint = build_translation_fingerprint(&self.options, input_signature);
        let Some(state) = store.load_run_state()? else {
            return Ok(ResumeSnapshot::default());
        };
        let Some(mut snapshot) = state.resume_snapshot(&expected_fingerprint) else {
            return Ok(ResumeSnapshot::default());
        };
        if snapshot.translation_batches_completed > batches.len() {
            return Err(CoreError::Data(format!(
                "resume state has {} translated batches, but the current input has only {}",
                snapshot.translation_batches_completed,
                batches.len()
            )));
        }

        let expected_segment_count: usize = batches
            .iter()
            .take(snapshot.translation_batches_completed)
            .map(Vec::len)
            .sum();
        if snapshot.translation_batches_completed > 0 && snapshot.translated_segments.is_empty() {
            snapshot.translated_segments = store.load_batch_segments(
                BatchShardKind::Translated,
                snapshot.translation_batches_completed,
            )?;
        }
        if snapshot.translated_segments.len() != expected_segment_count {
            return Err(CoreError::Data(format!(
                "resume state expected {expected_segment_count} translated segments across {} batches, but loaded {}",
                snapshot.translation_batches_completed,
                snapshot.translated_segments.len()
            )));
        }

        let resumed_source: Vec<SubtitleSegment> = batches
            .iter()
            .take(snapshot.translation_batches_completed)
            .flatten()
            .cloned()
            .collect();
        validate_full_alignment(&resumed_source, &snapshot.translated_segments)?;

        if snapshot.review_batches_completed > 0 && snapshot.reviewed_segments.is_empty() {
            snapshot.reviewed_segments = store
                .load_batch_segments(BatchShardKind::Reviewed, snapshot.review_batches_completed)?;
        }
        self.memory = snapshot.memory.clone();
        Ok(snapshot)
    }

    fn save_run_state(
        &self,
        translation_batches_completed: usize,
        review_batches_completed: usize,
        validation_completed: bool,
        usage: Usage,
    ) -> CoreResult<()> {
        let (Some(store), Some(input_signature)) =
            (self.store.as_ref(), self.input_signature.as_ref())
        else {
            return Ok(());
        };
        let state = RunState::new(
            &self.options,
            input_signature.clone(),
            usage,
            self.memory.clone(),
            translation_batches_completed,
            review_batches_completed,
            validation_completed,
        );
        store.save_run_state(&state)
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

fn messages_json(messages: &[ChatMessage]) -> JsonValue {
    JsonValue::Array(
        messages
            .iter()
            .map(|message| {
                JsonValue::Object(vec![
                    ("role".to_owned(), JsonValue::String(message.role.clone())),
                    (
                        "content".to_owned(),
                        JsonValue::String(message.content.clone()),
                    ),
                ])
            })
            .collect(),
    )
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
        if ch == ' '
            && chars
                .peek()
                .is_some_and(|&next| matches!(next, ',' | '.' | '!' | '?' | ';' | ':'))
        {
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

fn update_translation_memory(
    memory: &mut HashMap<String, String>,
    source: &[SubtitleSegment],
    translated: &[SubtitleSegment],
) {
    for (source, translated) in source.iter().zip(translated) {
        let key = translation_memory_key(&source.text);
        if key.is_empty() || translated.text.trim().is_empty() {
            continue;
        }
        memory.insert(key, translated.text.clone());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use crate::entities::{BatchTranslationResult, GlossaryEntry};
    use crate::ports::{BackendJsonResult, NoopDashboard};
    use crate::storage::{RunState, RuntimePaths, build_runtime_paths, input_signature_from_bytes};

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

    struct CountingBackend {
        calls: Arc<AtomicUsize>,
        fail_on_call: Option<usize>,
    }

    impl LlmBackend for CountingBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "echo"
        }

        fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_on_call == Some(call) {
                return Err(CoreError::Backend("scripted failure".to_owned()));
            }
            EchoBackend.generate_json(messages)
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

    #[test]
    fn pipeline_updates_translation_memory_and_saves_translated_shard() {
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

        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("clip.txt"),
                None,
                None,
                "Auto",
                "English",
                false,
            ),
            data: Arc::clone(&captured),
        };

        let mut pipeline =
            SubtitlePipeline::new(EchoBackend, NoopDashboard, options).with_store(Box::new(store));
        let run = pipeline.run_document(&document).expect("run");

        assert_eq!(run.translated_segments[0].text, "[ECHO] hello");
        let data = captured.lock().expect("capture lock");
        assert_eq!(
            data.saved_translation_memory,
            vec![("hello".to_owned(), "[ECHO] hello".to_owned())]
        );
        assert_eq!(data.saved_batches.len(), 1);
        assert_eq!(data.saved_batches[0].1[0].text, "[ECHO] hello");
    }

    #[test]
    fn pipeline_resumes_from_completed_batch_shards() {
        let document = SubtitleDocument {
            path: "resume.txt".into(),
            format: "txt".to_owned(),
            segments: ["one", "two", "three"]
                .into_iter()
                .enumerate()
                .map(|(index, text)| SubtitleSegment {
                    id: (index + 1).to_string(),
                    text: text.to_owned(),
                    start: None,
                    end: None,
                    identifier: None,
                    settings: None,
                })
                .collect(),
            header: None,
            passthrough_blocks: Vec::new(),
        };
        let mut options = PipelineOptions::new("resume.txt".into());
        options.batch_size = 1;
        let signature = input_signature_from_bytes(b"one\ntwo\nthree\n", Some(1));
        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("resume.txt"),
                None,
                None,
                "Auto",
                "Chinese",
                false,
            ),
            data: Arc::clone(&captured),
        };
        let first_calls = Arc::new(AtomicUsize::new(0));
        let mut first = SubtitlePipeline::new(
            CountingBackend {
                calls: Arc::clone(&first_calls),
                fail_on_call: Some(2),
            },
            NoopDashboard,
            options.clone(),
        )
        .with_store(Box::new(store.clone()))
        .with_input_signature(signature.clone());

        first
            .run_document(&document)
            .expect_err("second batch fails");
        assert_eq!(first_calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            captured
                .lock()
                .expect("capture lock")
                .saved_state
                .as_ref()
                .expect("checkpoint")
                .translation_batches_completed,
            1
        );

        let resumed_calls = Arc::new(AtomicUsize::new(0));
        let mut resumed = SubtitlePipeline::new(
            CountingBackend {
                calls: Arc::clone(&resumed_calls),
                fail_on_call: None,
            },
            NoopDashboard,
            options,
        )
        .with_store(Box::new(store))
        .with_input_signature(signature);
        let run = resumed.run_document(&document).expect("resume");

        assert_eq!(resumed_calls.load(Ordering::SeqCst), 2);
        assert_eq!(run.result.resumed_translation_batches, 1);
        assert_eq!(run.result.usage.total_tokens, 6);
        assert_eq!(
            run.translated_segments
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<Vec<_>>(),
            vec!["[ECHO] one", "[ECHO] two", "[ECHO] three"]
        );
    }

    #[test]
    fn pipeline_reuses_request_cache_without_backend_call() {
        let document = SubtitleDocument {
            path: "cache.txt".into(),
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
        let mut options = PipelineOptions::new("cache.txt".into());
        options.batch_size = 1;
        options.resume = false;
        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("cache.txt"),
                None,
                None,
                "Auto",
                "Chinese",
                false,
            ),
            data: Arc::clone(&captured),
        };

        let first_calls = Arc::new(AtomicUsize::new(0));
        let mut first = SubtitlePipeline::new(
            CountingBackend {
                calls: Arc::clone(&first_calls),
                fail_on_call: None,
            },
            NoopDashboard,
            options.clone(),
        )
        .with_store(Box::new(store.clone()));
        let first_run = first.run_document(&document).expect("first run");
        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(first_run.result.usage.total_tokens, 2);
        assert_eq!(first_run.result.cache_hits, 0);

        let second_calls = Arc::new(AtomicUsize::new(0));
        let mut second = SubtitlePipeline::new(
            CountingBackend {
                calls: Arc::clone(&second_calls),
                fail_on_call: Some(1),
            },
            NoopDashboard,
            options,
        )
        .with_store(Box::new(store));
        let second_run = second.run_document(&document).expect("cached run");

        assert_eq!(second_calls.load(Ordering::SeqCst), 0);
        assert_eq!(second_run.result.cache_hits, 1);
        assert_eq!(second_run.result.usage, Usage::default());
        assert_eq!(second_run.translated_segments[0].text, "[ECHO] hello");
    }

    #[derive(Debug, Default)]
    struct CapturedStoreData {
        saved_translation_memory: Vec<(String, String)>,
        saved_batches: Vec<(usize, Vec<SubtitleSegment>)>,
        saved_state: Option<RunState>,
        cached_responses: Vec<(CacheStage, String, BackendJsonResult)>,
    }

    #[derive(Debug, Clone)]
    struct CapturedStore {
        paths: RuntimePaths,
        data: Arc<Mutex<CapturedStoreData>>,
    }

    impl RuntimeStore for CapturedStore {
        fn paths(&self) -> &RuntimePaths {
            &self.paths
        }

        fn ensure_layout(&self) -> CoreResult<()> {
            Ok(())
        }

        fn save_glossary(&self, _entries: &[(String, String)]) -> CoreResult<()> {
            Ok(())
        }

        fn save_translation_memory(&self, entries: &[(String, String)]) -> CoreResult<()> {
            let mut data = self.data.lock().expect("capture lock");
            data.saved_translation_memory = entries.to_vec();
            data.saved_translation_memory.sort();
            Ok(())
        }

        fn save_batch_segments(
            &self,
            kind: BatchShardKind,
            batch_index: usize,
            segments: &[SubtitleSegment],
        ) -> CoreResult<()> {
            assert_eq!(kind, BatchShardKind::Translated);
            let mut data = self.data.lock().expect("capture lock");
            data.saved_batches.push((batch_index, segments.to_vec()));
            Ok(())
        }

        fn load_batch_segments(
            &self,
            kind: BatchShardKind,
            completed_batches: usize,
        ) -> CoreResult<Vec<SubtitleSegment>> {
            assert_eq!(kind, BatchShardKind::Translated);
            let data = self.data.lock().expect("capture lock");
            Ok(data
                .saved_batches
                .iter()
                .filter(|(index, _)| *index <= completed_batches)
                .flat_map(|(_, segments)| segments.clone())
                .collect())
        }

        fn save_run_state(&self, state: &RunState) -> CoreResult<()> {
            self.data.lock().expect("capture lock").saved_state = Some(state.clone());
            Ok(())
        }

        fn load_run_state(&self) -> CoreResult<Option<RunState>> {
            Ok(self.data.lock().expect("capture lock").saved_state.clone())
        }

        fn save_cached_response(
            &self,
            stage: CacheStage,
            request_hash: &str,
            response: &BackendJsonResult,
        ) -> CoreResult<()> {
            self.data
                .lock()
                .expect("capture lock")
                .cached_responses
                .push((stage, request_hash.to_owned(), response.clone()));
            Ok(())
        }

        fn load_cached_response(
            &self,
            stage: CacheStage,
            request_hash: &str,
        ) -> CoreResult<Option<BackendJsonResult>> {
            Ok(self
                .data
                .lock()
                .expect("capture lock")
                .cached_responses
                .iter()
                .find(|(cached_stage, cached_hash, _)| {
                    *cached_stage == stage && cached_hash == request_hash
                })
                .map(|(_, _, response)| response.clone()))
        }
    }
}
