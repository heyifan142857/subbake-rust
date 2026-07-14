use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::time::Instant;

use crate::CancellationGuard;
use crate::entities::{
    AgentLog, AgentRepairRecord, AttemptLog, FailureLog, GlossaryEntry, PipelineOptions,
    PipelineResult, ReviewPolicy, ReviewReport, ReviewStats, SplitRetryLog, SubtitleDocument,
    SubtitleSegment, TerminologyPreflightResult, TerminologyStats, TranslationLine, Usage,
};
use crate::error::{CoreError, CoreResult};
use crate::languages::normalize_language_name;
use crate::memory::{ContextMemory, english_possessive_base};
use crate::ports::{
    BackendJsonResult, BackendPayload, BatchShardKind, CacheStage, ChatMessage, DashboardSink,
    LlmBackend, RuntimeStore,
};
use crate::progress::{ProgressEvent, ProgressSink, ProgressUnit, TaskKind, TaskState};
use crate::recovery::{
    backend_payload_json, build_agent_repair_messages, combine_glossary, combine_summaries,
    parse_translation_payload, retry_correction_message, split_index,
};
use crate::review::{ReviewBatchPlan, build_review_messages, parse_review_payload};
use crate::storage::{
    InputSignature, JsonValue, ResumeSnapshot, RunState, build_request_hash, build_request_hash_v2,
    build_translation_fingerprint,
};
use crate::validation::{validate_full_alignment, validate_translation_batch};

mod planning;
mod review_stage;
mod translation_stage;

use planning::BatchPlanner;
use review_stage::ReviewStage;
use translation_stage::{PreparedBatch, TranslationStage};

pub struct SubtitlePipeline<B, D> {
    backend: B,
    dashboard: D,
    options: PipelineOptions,
    memory: ContextMemory,
    /// Only a user-supplied glossary is authoritative enough to reject a
    /// translation. Automatically extracted terminology remains advisory.
    required_glossary: BTreeMap<String, String>,
    store: Option<Box<dyn RuntimeStore>>,
    input_signature: Option<InputSignature>,
    /// Normalised-key → translation text cache loaded from the runtime store.
    translation_memory: HashMap<String, String>,
    translation_memory_hits: usize,
    cache_hits: usize,
    agent_repairs: Vec<AgentRepairRecord>,
    cancellation: CancellationGuard,
    progress: Option<Box<dyn ProgressSink>>,
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
            required_glossary: BTreeMap::new(),
            store: None,
            input_signature: None,
            translation_memory: HashMap::new(),
            translation_memory_hits: 0,
            cache_hits: 0,
            agent_repairs: Vec::new(),
            cancellation: CancellationGuard::never(),
            progress: None,
        }
    }

    pub fn with_progress(mut self, progress: Box<dyn ProgressSink>) -> Self {
        self.progress = Some(progress);
        self
    }

    fn report(
        &self,
        stage: &str,
        state: TaskState,
        current: usize,
        total: Option<usize>,
        resumed: usize,
        usage: Usage,
    ) {
        if let Some(sink) = &self.progress {
            sink.emit(ProgressEvent {
                task: TaskKind::Translation,
                stage: stage.to_owned(),
                state,
                current: current as u64,
                total: total.map(|v| v as u64),
                unit: ProgressUnit::Batches,
                resumed: resumed as u64,
                usage,
                message: None,
                translation: None,
            });
        }
    }

    fn report_translation_window(
        &self,
        batches: &[Vec<SubtitleSegment>],
        committed: usize,
        in_flight: usize,
        resumed: usize,
        usage: Usage,
    ) {
        let Some(sink) = &self.progress else { return };
        let completed_segments = batches.iter().take(committed).map(Vec::len).sum::<usize>();
        let total_segments = batches.iter().map(Vec::len).sum::<usize>();
        let mut event = ProgressEvent::running(
            TaskKind::Translation,
            "TRANSLATE",
            completed_segments as u64,
            Some(total_segments as u64),
            ProgressUnit::Lines,
        );
        event.resumed = batches.iter().take(resumed).map(Vec::len).sum::<usize>() as u64;
        event.usage = usage;
        event.translation = Some(crate::progress::TranslationProgress {
            segments_completed: completed_segments as u64,
            segments_total: total_segments as u64,
            batches_committed: committed as u64,
            batches_total: batches.len() as u64,
            requests_in_flight: in_flight as u64,
            cache_hits: self.cache_hits as u64,
            translation_memory_hits: self.translation_memory_hits as u64,
            window_index: committed.div_ceil(self.options.translation_concurrency.max(1)) as u64
                + 1,
            ..crate::progress::TranslationProgress::default()
        });
        sink.emit(event);
    }

    pub fn with_cancellation(mut self, cancellation: CancellationGuard) -> Self {
        self.cancellation = cancellation;
        self
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
        self.cancellation.check()?;
        if self.options.batch_size == 0 {
            return Err(CoreError::InvalidTranslation(
                "batch size must be greater than zero".to_owned(),
            ));
        }

        // Persisted auto terminology is useful prompt context, but only a
        // glossary explicitly supplied by the user is a hard requirement.
        self.required_glossary.clear();
        if let Some(ref store) = self.store {
            self.cancellation.check()?;
            let entries = store.load_glossary()?;
            self.memory.load_glossary(&entries);
            if self.options.glossary_path.is_some() {
                self.required_glossary.extend(entries);
            }
        }

        let batches = BatchPlanner::new(self.options.batch_size, self.options.batch_token_budget)
            .split(&document.segments);
        let planned_batches = BatchPlanner::describe(&batches);
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
                    agent_repairs: self.agent_repairs.clone(),
                    terminology: TerminologyStats::default(),
                    review: ReviewStats::default(),
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

        let terminology = self.run_terminology_preflight(document)?;

        let resume = self.load_resume_snapshot(&batches)?;
        self.report(
            "TRANSLATE",
            if resume.translation_batches_completed > 0 {
                TaskState::Resuming
            } else {
                TaskState::Running
            },
            resume.translation_batches_completed,
            Some(batches.len()),
            resume.translation_batches_completed,
            Usage::default(),
        );
        let mut translation_stage = TranslationStage::new(
            batches,
            resume.translation_batches_completed,
            resume.translated_segments,
        )?;
        let mut usage = resume.usage;
        if resume.translation_batches_completed == 0 {
            usage.add(terminology.usage);
        }
        if usage != Usage::default() {
            self.dashboard.add_usage(usage);
        }
        while !translation_stage.is_complete() {
            self.cancellation.check()?;
            let concurrency = if self.backend.supports_parallel_generation() {
                self.options.translation_concurrency.max(1)
            } else {
                1
            };
            let prepared = translation_stage.prepare_window(
                concurrency,
                self.options.use_cache,
                &self.translation_memory,
            );
            self.report(
                "TRANSLATE",
                TaskState::Running,
                translation_stage.next_batch(),
                Some(translation_stage.len()),
                resume.translation_batches_completed,
                usage,
            );
            let pending = prepared
                .iter()
                .filter(|batch| !batch.pending.is_empty())
                .map(|batch| (batch.index + 1, batch.pending.clone()))
                .collect::<Vec<_>>();
            self.report_translation_window(
                translation_stage.batches(),
                translation_stage.next_batch(),
                pending.len(),
                resume.translation_batches_completed,
                usage,
            );
            let mut generated = self.translate_window(&pending)?;
            validate_window_terminology(
                &prepared,
                &generated,
                &self.required_glossary,
                self.options.review_policy != ReviewPolicy::Off,
            )?;
            for prepared_batch in prepared {
                let result = if prepared_batch.pending.is_empty() {
                    None
                } else {
                    Some(
                        generated
                            .remove(&(prepared_batch.index + 1))
                            .ok_or_else(|| {
                                CoreError::Data(format!(
                                    "translation window omitted batch {}",
                                    prepared_batch.index + 1
                                ))
                            })?,
                    )
                };
                let applied = translation_stage.apply(prepared_batch, result)?;
                if let Some(result) = applied.result.as_ref() {
                    usage.add(result.usage);
                    self.dashboard.add_usage(result.usage);
                    self.memory
                        .update(&result.summary, &result.glossary_updates);
                }
                self.translation_memory_hits = translation_stage.memory_hits();
                if self.options.use_cache {
                    update_translation_memory(
                        &mut self.translation_memory,
                        &applied.source,
                        &applied.translated,
                    );
                }

                // Persist glossary, TM, batch segments after each batch.
                if let Some(ref store) = self.store {
                    self.cancellation.check()?;
                    let glossary_entries: Vec<(String, String)> = self
                        .memory
                        .glossary
                        .iter()
                        .map(|(source, target)| (source.clone(), target.clone()))
                        .collect();
                    store.save_glossary(&glossary_entries)?;
                    store.save_batch_segments(
                        BatchShardKind::Translated,
                        applied.index + 1,
                        &applied.translated,
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

                self.cancellation.check()?;
                self.save_run_state(
                    applied.index + 1,
                    resume.review_batches_completed,
                    false,
                    usage,
                )?;
                self.report(
                    "TRANSLATE",
                    TaskState::Running,
                    applied.index + 1,
                    Some(translation_stage.len()),
                    resume.translation_batches_completed,
                    usage,
                );
            }
            self.report_translation_window(
                translation_stage.batches(),
                translation_stage.next_batch(),
                0,
                resume.translation_batches_completed,
                usage,
            );
        }

        validate_full_alignment(&document.segments, translation_stage.output())?;
        self.cancellation.check()?;
        self.save_run_state(
            translation_stage.len(),
            resume.review_batches_completed,
            true,
            usage,
        )?;
        self.dashboard.mark_done("TRANSLATE");

        let batches = translation_stage.batches().to_vec();
        let translated_segments = translation_stage.finish();

        let mut review_stage = ReviewStage::new(
            &self.options,
            &batches,
            &translated_segments,
            &self.memory,
            resume.review_batches_completed,
            &resume.reviewed_segments,
            self.cache_hits,
        )?;
        self.dashboard
            .set_total_steps(3 + batches.len() + review_stage.len());
        let resumed_review_batches = review_stage.resumed();
        if !review_stage.is_empty() {
            self.dashboard.mark_running("FINAL_REVIEW");
            let mut next_review = resumed_review_batches;
            while next_review < review_stage.len() {
                self.cancellation.check()?;
                let concurrency = if self.backend.supports_parallel_generation() {
                    self.options.review_concurrency.max(1)
                } else {
                    1
                };
                self.report(
                    "FINAL_REVIEW",
                    TaskState::Running,
                    next_review,
                    Some(review_stage.len()),
                    resumed_review_batches,
                    usage,
                );
                let pending = review_stage.window(next_review, concurrency);
                let mut reviewed_window = self.review_window(&pending)?;
                for (review_position, _) in &pending {
                    let reviewed = reviewed_window.remove(review_position).ok_or_else(|| {
                        CoreError::Data(format!("review window omitted batch {review_position}"))
                    })?;
                    usage.add(reviewed.usage);
                    self.dashboard.add_usage(reviewed.usage);
                    let reviewed_segments =
                        review_stage.apply(*review_position, &reviewed.lines, reviewed.usage)?;
                    if let Some(ref store) = self.store {
                        self.cancellation.check()?;
                        store.save_batch_segments(
                            BatchShardKind::Reviewed,
                            *review_position,
                            &reviewed_segments,
                        )?;
                    }
                    self.save_run_state(batches.len(), *review_position, true, usage)?;
                    self.report(
                        "FINAL_REVIEW",
                        TaskState::Running,
                        *review_position,
                        Some(review_stage.len()),
                        resumed_review_batches,
                        usage,
                    );
                }
                next_review += pending.len();
            }
            validate_full_alignment(&document.segments, review_stage.output())?;
            self.dashboard.mark_done("FINAL_REVIEW");
        } else {
            self.report(
                "FINAL_REVIEW",
                TaskState::Skipped,
                0,
                Some(0),
                0,
                Usage::default(),
            );
        }
        let review_batches = review_stage.len();
        let review_outcome = review_stage.finish(self.cache_hits);
        let review = review_outcome.stats;
        if let Some(store) = self.store.as_ref() {
            store.save_review_report(&ReviewReport {
                terminology: terminology.clone(),
                review: review.clone(),
                changes: review_outcome.changes,
            })?;
        }
        self.cancellation.check()?;
        self.save_run_state(batches.len(), review_batches, true, usage)?;
        self.report(
            "WRITE_OUTPUT",
            TaskState::Running,
            batches.len(),
            Some(batches.len()),
            resume.translation_batches_completed,
            usage,
        );

        Ok(PipelineRun {
            result: PipelineResult {
                output_path: self.options.output_path.clone(),
                batches_translated: batches.len(),
                review_batches,
                usage,
                dry_run: false,
                planned_batches,
                cache_hits: self.cache_hits,
                resumed_translation_batches: resume.translation_batches_completed,
                resumed_review_batches,
                translation_memory_hits: self.translation_memory_hits,
                state_path,
                glossary_path,
                agent_repairs: self.agent_repairs.clone(),
                terminology,
                review,
            },
            translated_segments: review_outcome.output,
        })
    }

    fn run_terminology_preflight(
        &mut self,
        document: &SubtitleDocument,
    ) -> CoreResult<TerminologyStats> {
        let started = Instant::now();
        let candidates = extract_terminology_candidates(&document.segments);
        let mut stats = TerminologyStats {
            candidates: candidates.len(),
            ..TerminologyStats::default()
        };
        if !self.options.terminology_preflight
            || self.options.fast_mode
            || candidates.is_empty()
            || !self.backend.supports_terminology_preflight()
        {
            self.report(
                "TERMINOLOGY_PREFLIGHT",
                TaskState::Skipped,
                0,
                Some(candidates.len()),
                0,
                Usage::default(),
            );
            return Ok(stats);
        }

        self.report(
            "TERMINOLOGY_PREFLIGHT",
            TaskState::Running,
            0,
            Some(candidates.len()),
            0,
            Usage::default(),
        );
        let existing = self.memory.glossary.clone();
        let messages = build_terminology_messages(&self.options, &candidates);
        let hash = request_hash(&self.options, CacheStage::Terminology, &messages);
        let cached = if self.options.use_cache {
            self.store
                .as_ref()
                .map(|store| store.load_cached_response(CacheStage::Terminology, &hash))
                .transpose()?
                .flatten()
        } else {
            None
        };
        let result = if let Some(response) = cached {
            stats.cache_hits = 1;
            self.cache_hits += 1;
            Ok(response)
        } else {
            let mut last_error = None;
            let mut response = None;
            for _ in 0..=self.options.retries {
                self.cancellation.check()?;
                match self
                    .backend
                    .generate_raw_json_cancellable(&messages, &self.cancellation)
                    .and_then(|(json, usage)| {
                        let payload = parse_terminology_payload(&json, &candidates)?;
                        Ok(BackendJsonResult {
                            payload: BackendPayload::Terminology(payload),
                            usage,
                        })
                    }) {
                    Ok(value) => {
                        response = Some(value);
                        break;
                    }
                    Err(CoreError::Cancelled) => return Err(CoreError::Cancelled),
                    Err(error) => last_error = Some(error),
                }
            }
            response.ok_or_else(|| {
                last_error.unwrap_or_else(|| {
                    CoreError::Backend("terminology preflight failed".to_owned())
                })
            })
        };

        match result {
            Ok(response) => {
                let BackendPayload::Terminology(payload) = response.payload else {
                    return Err(CoreError::Data(
                        "terminology cache returned a different payload".to_owned(),
                    ));
                };
                stats.usage = if stats.cache_hits > 0 {
                    Usage::default()
                } else {
                    response.usage
                };
                let mut accepted = Vec::new();
                for entry in payload.entries {
                    if let Some(current) = self.memory.glossary.get(&entry.source) {
                        if !current.eq_ignore_ascii_case(&entry.target) {
                            stats.conflicts_omitted += 1;
                        }
                        continue;
                    }
                    if accepted.iter().any(|value: &GlossaryEntry| {
                        value.source.eq_ignore_ascii_case(&entry.source)
                            && !value.target.eq_ignore_ascii_case(&entry.target)
                    }) {
                        stats.conflicts_omitted += 1;
                        continue;
                    }
                    accepted.push(entry);
                }
                self.memory.update("", &accepted);
                stats.entries_added = self.memory.glossary.len().saturating_sub(existing.len());
                if self.options.use_cache
                    && stats.cache_hits == 0
                    && let Some(store) = self.store.as_ref()
                {
                    store.save_cached_response(
                        CacheStage::Terminology,
                        &hash,
                        &BackendJsonResult {
                            payload: BackendPayload::Terminology(TerminologyPreflightResult {
                                entries: accepted,
                            }),
                            usage: response.usage,
                        },
                    )?;
                }
                if let Some(store) = self.store.as_ref() {
                    let entries = self
                        .memory
                        .glossary
                        .iter()
                        .map(|(source, target)| (source.clone(), target.clone()))
                        .collect::<Vec<_>>();
                    store.save_glossary(&entries)?;
                }
            }
            Err(error) => {
                stats.degraded = true;
                stats.degraded_reason = Some(error.to_string());
            }
        }
        stats.duration_ms = duration_ms(started);
        self.dashboard.add_usage(stats.usage);
        self.report(
            "TERMINOLOGY_PREFLIGHT",
            TaskState::Completed,
            candidates.len(),
            Some(candidates.len()),
            0,
            stats.usage,
        );
        Ok(stats)
    }

    fn translate_batch(
        &mut self,
        batch_index: usize,
        batch: &[SubtitleSegment],
    ) -> CoreResult<BatchWithUsage> {
        self.translate_batch_impl(batch_index, batch, true, None)
    }

    fn translate_window(
        &mut self,
        batches: &[(usize, Vec<SubtitleSegment>)],
    ) -> CoreResult<HashMap<usize, BatchWithUsage>> {
        use crate::ports::{GenerationRequest, ResponseContract};

        if !self.backend.supports_parallel_generation() {
            let mut results = HashMap::new();
            for (batch_index, batch) in batches {
                results.insert(*batch_index, self.translate_batch(*batch_index, batch)?);
            }
            return Ok(results);
        }

        let mut results = HashMap::new();
        let mut pending = Vec::new();
        for (batch_index, batch) in batches {
            let messages = build_translation_messages(
                &self.options,
                *batch_index,
                batch,
                &self.memory,
                &self.required_glossary,
            );
            let hash = request_hash(&self.options, CacheStage::Translate, &messages);
            let cached = if self.options.use_cache {
                self.store
                    .as_ref()
                    .map(|store| store.load_cached_response(CacheStage::Translate, &hash))
                    .transpose()?
                    .flatten()
            } else {
                None
            };
            if let Some(response) = cached {
                let BackendPayload::Translation(payload) = response.payload else {
                    return Err(CoreError::Data(
                        "translation cache returned a review payload".to_owned(),
                    ));
                };
                validate_translation_batch(batch, &payload.lines)?;
                self.cache_hits += 1;
                results.insert(
                    *batch_index,
                    BatchWithUsage {
                        lines: payload.lines,
                        summary: payload.summary,
                        glossary_updates: payload.glossary_updates,
                        usage: Usage::default(),
                    },
                );
            } else {
                pending.push((*batch_index, batch.clone(), hash, messages));
            }
        }
        let requests = pending
            .iter()
            .map(|(_, _, _, messages)| GenerationRequest {
                messages: messages.clone(),
                response_contract: ResponseContract::JsonObject,
            })
            .collect();
        let responses = self.backend.generate_many_cancellable(
            requests,
            self.options.translation_concurrency,
            &self.cancellation,
        )?;
        if responses.len() != pending.len() {
            return Err(CoreError::Backend(format!(
                "backend returned {} responses for {} translation requests",
                responses.len(),
                pending.len()
            )));
        }
        for ((batch_index, batch, hash, _), response) in pending.into_iter().zip(responses) {
            match response.and_then(|response| {
                let payload = parse_translation_payload(&response.json)?;
                validate_translation_batch(&batch, &payload.lines)?;
                Ok((payload, response.usage))
            }) {
                Ok((payload, response_usage)) => {
                    let backend_result = BackendJsonResult {
                        payload: BackendPayload::Translation(payload.clone()),
                        usage: response_usage,
                    };
                    if self.options.use_cache
                        && let Some(store) = self.store.as_ref()
                    {
                        store.save_cached_response(
                            CacheStage::Translate,
                            &hash,
                            &backend_result,
                        )?;
                    }
                    results.insert(
                        batch_index,
                        BatchWithUsage {
                            lines: payload.lines,
                            summary: payload.summary,
                            glossary_updates: payload.glossary_updates,
                            usage: response_usage,
                        },
                    );
                }
                Err(error) => {
                    results.insert(
                        batch_index,
                        self.translate_batch_impl(batch_index, &batch, true, Some(error))?,
                    );
                }
            }
        }
        Ok(results)
    }

    fn translate_batch_impl(
        &mut self,
        batch_index: usize,
        batch: &[SubtitleSegment],
        record_failure: bool,
        initial_error: Option<CoreError>,
    ) -> CoreResult<BatchWithUsage> {
        let mut last_error = initial_error;
        let mut attempts = Vec::new();
        for attempt in 1..=self.options.retries + 1 {
            self.cancellation.check()?;
            let mut messages = build_translation_messages(
                &self.options,
                batch_index,
                batch,
                &self.memory,
                &self.required_glossary,
            );
            if let Some(error) = last_error.as_ref() {
                messages.push(retry_correction_message(error));
            }
            let request_hash = request_hash(&self.options, CacheStage::Translate, &messages);
            match self.translate_once(batch, &messages, &request_hash) {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if matches!(error, CoreError::Cancelled) {
                        return Err(error);
                    }
                    let failure_messages = messages.clone();
                    let mut attempt_log = AttemptLog {
                        attempt,
                        cached: false,
                        error: Some(error.to_string()),
                        payload: None,
                        messages,
                        split_retry: None,
                    };
                    if matches!(error, CoreError::InvalidTranslation(_))
                        && !self.options.fast_mode
                        && batch.len() > 1
                    {
                        let split = split_index(batch);
                        attempt_log.split_retry = Some(SplitRetryLog {
                            triggered: true,
                            sizes: vec![split, batch.len() - split],
                            resolved: false,
                            error: None,
                        });
                        match self.translate_split(batch_index, batch, split) {
                            Ok(result) => {
                                if let Some(split_log) = attempt_log.split_retry.as_mut() {
                                    split_log.resolved = true;
                                }
                                attempts.push(attempt_log);
                                return Ok(result);
                            }
                            Err(split_error) => {
                                if let Some(split_log) = attempt_log.split_retry.as_mut() {
                                    split_log.error = Some(split_error.to_string());
                                }
                                attempts.push(attempt_log);
                                if record_failure {
                                    return self.finish_translation_failure(
                                        batch_index,
                                        batch,
                                        split_error,
                                        request_hash,
                                        failure_messages,
                                        attempts,
                                    );
                                }
                                return Err(split_error);
                            }
                        }
                    }
                    last_error = Some(error.clone());
                    attempts.push(attempt_log);
                    if attempt == self.options.retries + 1 {
                        if record_failure {
                            return self.finish_translation_failure(
                                batch_index,
                                batch,
                                error,
                                request_hash,
                                failure_messages,
                                attempts,
                            );
                        }
                        return Err(error);
                    }
                }
            }
        }
        Err(CoreError::Data(
            "translation retry loop ended unexpectedly".to_owned(),
        ))
    }

    fn translate_once(
        &mut self,
        batch: &[SubtitleSegment],
        messages: &[ChatMessage],
        request_hash: &str,
    ) -> CoreResult<BatchWithUsage> {
        let cached_response = if self.options.use_cache {
            self.cancellation.check()?;
            self.store
                .as_ref()
                .map(|store| store.load_cached_response(CacheStage::Translate, request_hash))
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
            None => self
                .backend
                .generate_json_cancellable(messages, &self.cancellation)?,
        };
        let BackendPayload::Translation(result) = &backend_result.payload else {
            return Err(CoreError::Data(
                "translation cache returned a review payload".to_owned(),
            ));
        };
        validate_translation_batch(batch, &result.lines)?;
        if self.options.use_cache
            && !cached
            && let Some(store) = self.store.as_ref()
        {
            self.cancellation.check()?;
            store.save_cached_response(CacheStage::Translate, request_hash, &backend_result)?;
        }
        let BackendPayload::Translation(result) = backend_result.payload else {
            return Err(CoreError::Data(
                "translation backend returned a review payload".to_owned(),
            ));
        };
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

    fn translate_split(
        &mut self,
        batch_index: usize,
        batch: &[SubtitleSegment],
        split: usize,
    ) -> CoreResult<BatchWithUsage> {
        let left = self.translate_batch_impl(batch_index, &batch[..split], false, None)?;
        let right = self.translate_batch_impl(batch_index, &batch[split..], false, None)?;
        let mut usage = left.usage;
        usage.add(right.usage);
        Ok(BatchWithUsage {
            lines: left.lines.into_iter().chain(right.lines).collect(),
            summary: combine_summaries(&left.summary, &right.summary, self.memory.max_summaries),
            glossary_updates: combine_glossary(left.glossary_updates, right.glossary_updates),
            usage,
        })
    }

    fn review_batch(
        &mut self,
        batch_index: usize,
        batch: &ReviewBatchPlan,
    ) -> CoreResult<ReviewWithUsage> {
        self.review_batch_impl(batch_index, batch, None)
    }

    fn review_batch_after_error(
        &mut self,
        batch_index: usize,
        batch: &ReviewBatchPlan,
        error: CoreError,
    ) -> CoreResult<ReviewWithUsage> {
        self.review_batch_impl(batch_index, batch, Some(error))
    }

    fn review_batch_impl(
        &mut self,
        batch_index: usize,
        batch: &ReviewBatchPlan,
        initial_error: Option<CoreError>,
    ) -> CoreResult<ReviewWithUsage> {
        let mut last_error = initial_error;
        let mut attempts = Vec::new();
        for attempt in 1..=self.options.retries + 1 {
            self.cancellation.check()?;
            let mut messages = build_review_messages(
                &self.options,
                &batch.source,
                &batch.translated,
                &batch.reasons,
                &batch.candidate_reasons,
                &self.memory,
            );
            if let Some(error) = last_error.as_ref() {
                messages.push(retry_correction_message(error));
            }
            let request_hash = request_hash(&self.options, CacheStage::Review, &messages);
            match self.review_once(batch, &messages, &request_hash) {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if matches!(error, CoreError::Cancelled) {
                        return Err(error);
                    }
                    let failure_messages = messages.clone();
                    attempts.push(AttemptLog {
                        attempt,
                        cached: false,
                        error: Some(error.to_string()),
                        payload: None,
                        messages,
                        split_retry: None,
                    });
                    last_error = Some(error.clone());
                    if attempt == self.options.retries + 1 {
                        return self.finish_review_failure(
                            batch_index,
                            batch,
                            error,
                            request_hash,
                            failure_messages,
                            attempts,
                        );
                    }
                }
            }
        }
        Err(CoreError::Data(
            "review retry loop ended unexpectedly".to_owned(),
        ))
    }

    fn review_window(
        &mut self,
        batches: &[(usize, ReviewBatchPlan)],
    ) -> CoreResult<HashMap<usize, ReviewWithUsage>> {
        use crate::ports::{GenerationRequest, ResponseContract};
        if !self.backend.supports_parallel_generation() {
            let mut output = HashMap::new();
            for (index, batch) in batches {
                output.insert(*index, self.review_batch(*index, batch)?);
            }
            return Ok(output);
        }
        let mut output = HashMap::new();
        let mut pending = Vec::new();
        for (index, batch) in batches {
            let messages = build_review_messages(
                &self.options,
                &batch.source,
                &batch.translated,
                &batch.reasons,
                &batch.candidate_reasons,
                &self.memory,
            );
            let hash = request_hash(&self.options, CacheStage::Review, &messages);
            let cached = if self.options.use_cache {
                self.store
                    .as_ref()
                    .map(|store| store.load_cached_response(CacheStage::Review, &hash))
                    .transpose()?
                    .flatten()
            } else {
                None
            };
            if let Some(response) = cached {
                let BackendPayload::Review(result) = response.payload else {
                    return Err(CoreError::Data(
                        "review cache returned a translation payload".to_owned(),
                    ));
                };
                validate_translation_batch(&batch.source, &result.lines)?;
                self.cache_hits += 1;
                output.insert(
                    *index,
                    ReviewWithUsage {
                        lines: result.lines,
                        usage: Usage::default(),
                    },
                );
            } else {
                pending.push((*index, batch.clone(), hash, messages));
            }
        }
        let requests = pending
            .iter()
            .map(|(_, _, _, messages)| GenerationRequest {
                messages: messages.clone(),
                response_contract: ResponseContract::JsonObject,
            })
            .collect();
        let responses = self.backend.generate_many_cancellable(
            requests,
            self.options.review_concurrency,
            &self.cancellation,
        )?;
        if responses.len() != pending.len() {
            return Err(CoreError::Backend(format!(
                "backend returned {} responses for {} review requests",
                responses.len(),
                pending.len()
            )));
        }
        for ((index, batch, hash, _), response) in pending.into_iter().zip(responses) {
            match response.and_then(|response| {
                let mut result = parse_review_payload(&response.json)?;
                validate_review_candidate_ids(&batch, &result.lines)?;
                result.lines = merge_review_patch(&batch.translated, &result.lines)?;
                validate_translation_batch(&batch.source, &result.lines)?;
                Ok((result, response.usage))
            }) {
                Ok((result, response_usage)) => {
                    if self.options.use_cache
                        && let Some(store) = self.store.as_ref()
                    {
                        store.save_cached_response(
                            CacheStage::Review,
                            &hash,
                            &BackendJsonResult {
                                payload: BackendPayload::Review(result.clone()),
                                usage: response_usage,
                            },
                        )?;
                    }
                    output.insert(
                        index,
                        ReviewWithUsage {
                            lines: result.lines,
                            usage: response_usage,
                        },
                    );
                }
                Err(error) => {
                    output.insert(index, self.review_batch_after_error(index, &batch, error)?);
                }
            }
        }
        Ok(output)
    }

    fn review_once(
        &mut self,
        batch: &ReviewBatchPlan,
        messages: &[ChatMessage],
        request_hash: &str,
    ) -> CoreResult<ReviewWithUsage> {
        self.cancellation.check()?;
        let cached_response = if self.options.use_cache {
            self.store
                .as_ref()
                .map(|store| store.load_cached_response(CacheStage::Review, request_hash))
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
            None => {
                let (payload, usage) = self
                    .backend
                    .generate_raw_json_cancellable(messages, &self.cancellation)?;
                let mut review = parse_review_payload(&payload)?;
                validate_review_candidate_ids(batch, &review.lines)?;
                review.lines = merge_review_patch(&batch.translated, &review.lines)?;
                BackendJsonResult {
                    payload: BackendPayload::Review(review),
                    usage,
                }
            }
        };
        let BackendPayload::Review(result) = &backend_result.payload else {
            return Err(CoreError::Data(
                "review cache returned a translation payload".to_owned(),
            ));
        };
        validate_translation_batch(&batch.source, &result.lines)?;
        if self.options.use_cache
            && !cached
            && let Some(store) = self.store.as_ref()
        {
            self.cancellation.check()?;
            store.save_cached_response(CacheStage::Review, request_hash, &backend_result)?;
        }
        let BackendPayload::Review(result) = backend_result.payload else {
            return Err(CoreError::Data(
                "review backend returned a translation payload".to_owned(),
            ));
        };
        Ok(ReviewWithUsage {
            lines: result.lines,
            usage: if cached {
                Usage::default()
            } else {
                backend_result.usage
            },
        })
    }

    fn finish_translation_failure(
        &mut self,
        batch_index: usize,
        batch: &[SubtitleSegment],
        error: CoreError,
        request_hash: String,
        messages: Vec<ChatMessage>,
        attempts: Vec<AttemptLog>,
    ) -> CoreResult<BatchWithUsage> {
        if matches!(error, CoreError::Cancelled) {
            return Err(error);
        }
        self.cancellation.check()?;
        let repair =
            self.run_agent_repair("translate", batch_index, batch, None, &error, &attempts)?;
        if let Some(outcome) = repair.as_ref()
            && let Some(BackendPayload::Translation(result)) = outcome.payload.clone()
        {
            return Ok(BatchWithUsage {
                lines: result.lines,
                summary: result.summary,
                glossary_updates: result.glossary_updates,
                usage: outcome.usage,
            });
        }
        let agent_attempts = repair
            .as_ref()
            .map(|outcome| outcome.attempts.clone())
            .unwrap_or_default();
        let failure_path = self.save_failure(FailureLog {
            stage: "translate".to_owned(),
            batch_index,
            request_hash,
            batch_segments: batch.to_vec(),
            messages,
            translated_segments: Vec::new(),
            attempts,
            agent_attempts,
        })?;
        Err(failure_error(
            "Translation",
            batch_index,
            &error,
            failure_path.as_ref(),
            repair.as_ref(),
        ))
    }

    fn finish_review_failure(
        &mut self,
        batch_index: usize,
        batch: &ReviewBatchPlan,
        error: CoreError,
        request_hash: String,
        messages: Vec<ChatMessage>,
        attempts: Vec<AttemptLog>,
    ) -> CoreResult<ReviewWithUsage> {
        if matches!(error, CoreError::Cancelled) {
            return Err(error);
        }
        self.cancellation.check()?;
        let repair = self.run_agent_repair(
            "review",
            batch_index,
            &batch.source,
            Some(&batch.translated),
            &error,
            &attempts,
        )?;
        if let Some(outcome) = repair.as_ref()
            && let Some(BackendPayload::Review(result)) = outcome.payload.clone()
        {
            return Ok(ReviewWithUsage {
                lines: result.lines,
                usage: outcome.usage,
            });
        }
        let agent_attempts = repair
            .as_ref()
            .map(|outcome| outcome.attempts.clone())
            .unwrap_or_default();
        let failure_path = self.save_failure(FailureLog {
            stage: "review".to_owned(),
            batch_index,
            request_hash,
            batch_segments: batch.source.clone(),
            messages,
            translated_segments: batch.translated.clone(),
            attempts,
            agent_attempts,
        })?;
        Err(failure_error(
            "Final review",
            batch_index,
            &error,
            failure_path.as_ref(),
            repair.as_ref(),
        ))
    }

    fn run_agent_repair(
        &mut self,
        stage: &str,
        batch_index: usize,
        source: &[SubtitleSegment],
        translated: Option<&[SubtitleSegment]>,
        initial_error: &CoreError,
        failed_attempts: &[AttemptLog],
    ) -> CoreResult<Option<RepairOutcome>> {
        if !self.options.agent
            || self.options.agent_repair_attempts == 0
            || !is_agent_repairable(initial_error)
        {
            return Ok(None);
        }

        let cache_stage = if stage == "translate" {
            CacheStage::AgentTranslateRepair
        } else {
            CacheStage::AgentReviewRepair
        };
        let mut repair_error = initial_error.clone();
        let mut attempts = Vec::new();
        let mut total_usage = Usage::default();
        let mut log_path = None;
        for attempt in 1..=self.options.agent_repair_attempts {
            self.cancellation.check()?;
            let messages = build_agent_repair_messages(
                stage,
                source,
                translated,
                &self.options.target_language,
                &repair_error,
                failed_attempts,
                &attempts,
            );
            let request_hash = request_hash(&self.options, cache_stage, &messages);
            let cached_response = if self.options.use_cache {
                self.store
                    .as_ref()
                    .map(|store| store.load_cached_response(cache_stage, &request_hash))
                    .transpose()?
                    .flatten()
            } else {
                None
            };
            let cached = cached_response.is_some();
            let response_result = match cached_response {
                Some(response) => {
                    self.cache_hits += 1;
                    Ok(response)
                }
                None => self
                    .backend
                    .generate_raw_json_cancellable(&messages, &self.cancellation)
                    .and_then(|(payload, usage)| {
                        total_usage.add(usage);
                        let payload = if stage == "translate" {
                            BackendPayload::Translation(parse_translation_payload(&payload)?)
                        } else {
                            BackendPayload::Review(parse_review_payload(&payload)?)
                        };
                        Ok(BackendJsonResult { payload, usage })
                    }),
            };

            match response_result.and_then(|response| {
                let lines = match &response.payload {
                    BackendPayload::Translation(result) => &result.lines,
                    BackendPayload::Review(result) => &result.lines,
                    BackendPayload::Terminology(_) => {
                        return Err(CoreError::Data(
                            "repair cache returned a terminology payload".to_owned(),
                        ));
                    }
                };
                validate_translation_batch(source, lines)?;
                if self.options.use_cache
                    && !cached
                    && let Some(store) = self.store.as_ref()
                {
                    store.save_cached_response(cache_stage, &request_hash, &response)?;
                }
                Ok(response)
            }) {
                Ok(response) => {
                    attempts.push(AttemptLog {
                        attempt,
                        cached,
                        error: None,
                        payload: Some(backend_payload_json(&response.payload)?),
                        messages,
                        split_retry: None,
                    });
                    log_path = self.save_agent_log(AgentLog {
                        stage: stage.to_owned(),
                        batch_index,
                        success: true,
                        attempts: attempts.clone(),
                        final_error: None,
                    })?;
                    self.agent_repairs.push(AgentRepairRecord {
                        stage: stage.to_owned(),
                        batch_index,
                        attempts: attempt,
                        success: true,
                        log_path: log_path.clone(),
                        error: String::new(),
                    });
                    return Ok(Some(RepairOutcome {
                        payload: Some(response.payload),
                        usage: total_usage,
                        attempts,
                        log_path,
                        error: None,
                    }));
                }
                Err(error) => {
                    repair_error = error;
                    attempts.push(AttemptLog {
                        attempt,
                        cached,
                        error: Some(repair_error.to_string()),
                        payload: None,
                        messages,
                        split_retry: None,
                    });
                    log_path = self.save_agent_log(AgentLog {
                        stage: stage.to_owned(),
                        batch_index,
                        success: false,
                        attempts: attempts.clone(),
                        final_error: Some(repair_error.to_string()),
                    })?;
                }
            }
        }
        self.agent_repairs.push(AgentRepairRecord {
            stage: stage.to_owned(),
            batch_index,
            attempts: attempts.len(),
            success: false,
            log_path: log_path.clone(),
            error: repair_error.to_string(),
        });
        Ok(Some(RepairOutcome {
            payload: None,
            usage: total_usage,
            attempts,
            log_path,
            error: Some(repair_error.to_string()),
        }))
    }

    fn save_failure(&self, log: FailureLog) -> CoreResult<Option<PathBuf>> {
        self.store
            .as_ref()
            .map(|store| store.save_failure_log(&log))
            .transpose()
    }

    fn save_agent_log(&self, log: AgentLog) -> CoreResult<Option<PathBuf>> {
        self.store
            .as_ref()
            .map(|store| store.save_agent_log(&log))
            .transpose()
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReviewWithUsage {
    lines: Vec<TranslationLine>,
    usage: Usage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TerminologyCandidate {
    source: String,
    context: String,
}

fn extract_terminology_candidates(segments: &[SubtitleSegment]) -> Vec<TerminologyCandidate> {
    let mut candidates = std::collections::BTreeMap::new();
    for segment in segments {
        let words = segment
            .text
            .split_whitespace()
            .map(|word| {
                let word = word.trim_matches(|ch: char| {
                    !ch.is_alphanumeric() && ch != '-' && ch != '\'' && ch != '’'
                });
                english_possessive_base(word).unwrap_or(word)
            })
            .filter(|word| word.len() >= 2)
            .collect::<Vec<_>>();
        let mut index = 0;
        while index < words.len() {
            let word = words[index];
            let is_title = word
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_uppercase())
                && word.chars().skip(1).any(|ch| ch.is_ascii_lowercase());
            let is_acronym = word.chars().filter(|ch| ch.is_ascii_alphabetic()).count() >= 2
                && word
                    .chars()
                    .filter(|ch| ch.is_ascii_alphabetic())
                    .all(|ch| ch.is_ascii_uppercase());
            if !is_title && !is_acronym {
                index += 1;
                continue;
            }
            let mut end = index + 1;
            while end < words.len() && end - index < 4 {
                let next = words[end];
                if !next
                    .chars()
                    .next()
                    .is_some_and(|ch| ch.is_ascii_uppercase())
                {
                    break;
                }
                end += 1;
            }
            candidates
                .entry(word.to_ascii_lowercase())
                .or_insert_with(|| TerminologyCandidate {
                    source: word.to_owned(),
                    context: segment.text.chars().take(240).collect(),
                });
            let source = words[index..end].join(" ");
            if end > index + 1 {
                candidates
                    .entry(source.to_ascii_lowercase())
                    .or_insert_with(|| TerminologyCandidate {
                        source,
                        context: segment.text.chars().take(240).collect(),
                    });
            }
            index = end;
        }
    }
    candidates.into_values().take(256).collect()
}

fn build_terminology_messages(
    options: &PipelineOptions,
    candidates: &[TerminologyCandidate],
) -> Vec<ChatMessage> {
    let payload = serde_json::json!({
        "source_language": options.source_language,
        "target_language": options.target_language,
        "candidates": candidates.iter().map(|candidate| serde_json::json!({
            "source": candidate.source,
            "context": candidate.context,
        })).collect::<Vec<_>>(),
    });
    let payload = serde_json::to_string(&payload).unwrap_or_default();
    vec![
        ChatMessage::system(
            "TASK_START\nextract_terminology\nTASK_END\nReturn JSON only as {\"entries\":[{\"source\":\"exact candidate\",\"target\":\"canonical translation\"}]}. Include only names, titles, organizations, places, recurring objects, and domain terms whose translation should stay consistent. Copy source exactly from a candidate. Omit ordinary sentence-initial words and uncertain entries.",
        ),
        ChatMessage::user(format!(
            "TERMINOLOGY_JSON_START{payload}TERMINOLOGY_JSON_END"
        )),
    ]
}

fn parse_terminology_payload(
    payload: &serde_json::Value,
    candidates: &[TerminologyCandidate],
) -> CoreResult<TerminologyPreflightResult> {
    let entries = payload
        .get("entries")
        .or_else(|| payload.get("glossary"))
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            CoreError::InvalidTranslation("terminology response missing entries array".to_owned())
        })?;
    let mut parsed = Vec::new();
    for entry in entries {
        let source = entry["source"].as_str().unwrap_or_default().trim();
        let target = entry["target"].as_str().unwrap_or_default().trim();
        if source.is_empty() || target.is_empty() {
            return Err(CoreError::InvalidTranslation(
                "terminology entry contains an empty source or target".to_owned(),
            ));
        }
        let Some(candidate) = candidates
            .iter()
            .find(|candidate| candidate.source.eq_ignore_ascii_case(source))
        else {
            return Err(CoreError::InvalidTranslation(format!(
                "terminology response contains unknown source `{source}`"
            )));
        };
        parsed.push(GlossaryEntry {
            source: candidate.source.clone(),
            target: target.to_owned(),
        });
    }
    Ok(TerminologyPreflightResult { entries: parsed })
}

fn duration_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[derive(Debug, Clone)]
struct RepairOutcome {
    payload: Option<BackendPayload>,
    usage: Usage,
    attempts: Vec<AttemptLog>,
    log_path: Option<PathBuf>,
    error: Option<String>,
}

fn build_translation_messages(
    options: &PipelineOptions,
    batch_index: usize,
    batch: &[SubtitleSegment],
    memory: &ContextMemory,
    required_glossary: &BTreeMap<String, String>,
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
            let mut required = serde_json::Map::new();
            let mut advisory = serde_json::Map::new();
            for (source, target) in glossary {
                let entry = (source.clone(), serde_json::Value::String(target));
                if required_glossary.contains_key(&source) {
                    required.insert(entry.0, entry.1);
                } else {
                    advisory.insert(entry.0, entry.1);
                }
            }
            if !required.is_empty() {
                context["glossary"] = serde_json::Value::Object(required);
            }
            if !advisory.is_empty() {
                context["terminology_hints"] = serde_json::Value::Object(advisory);
            }
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
        ChatMessage::system(
            "TASK_START\ntranslate_subtitles\nTASK_END\n\
Return JSON only with this shape:\n\
{\"lines\":[{\"id\":\"<source id>\",\"translation\":\"<non-empty target-language text>\"}],\"summary\":\"\",\"glossary_updates\":[]}\n\
Return exactly one line for every input line, in the same order. Copy each id exactly.\n\
The translated text must be in the translation field; do not return it in text or translated_text.\n\
Every non-empty source line must have a non-empty translation. Do not include markdown or explanations.\n\
Entries in CONTEXT_JSON.glossary are user-required translations. Entries in \
CONTEXT_JSON.terminology_hints are automatically learned suggestions: use them \
only when they fit the meaning in the current context.",
        ),
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

fn request_hash(options: &PipelineOptions, stage: CacheStage, messages: &[ChatMessage]) -> String {
    if let Some(fingerprint) = &options.provider_fingerprint {
        return build_request_hash_v2(fingerprint, stage.as_str(), messages_json(messages));
    }
    build_request_hash(
        &options.provider,
        &options.model,
        stage.as_str(),
        messages_json(messages),
    )
}

fn failure_error(
    stage: &str,
    batch_index: usize,
    error: &CoreError,
    failure_path: Option<&PathBuf>,
    repair: Option<&RepairOutcome>,
) -> CoreError {
    let mut message = format!("{stage} batch {batch_index} failed: {error}");
    if let Some(repair) = repair
        && repair.payload.is_none()
    {
        message.push_str(&format!(
            "\nAgent repair failed after {} attempt(s).",
            repair.attempts.len()
        ));
        if let Some(log_path) = &repair.log_path {
            message.push_str(&format!("\nAgent log saved to:\n{}", log_path.display()));
        }
        if let Some(error) = &repair.error {
            message.push_str(&format!("\nLast agent error: {error}"));
        }
    }
    if let Some(path) = failure_path {
        message.push_str(&format!("\nFailure sample saved to:\n{}", path.display()));
    }
    CoreError::InvalidTranslation(message)
}

fn is_agent_repairable(error: &CoreError) -> bool {
    match error {
        CoreError::InvalidTranslation(_) => true,
        CoreError::Backend(message) => {
            message.contains("invalid JSON in response")
                || message.contains("response JSON object")
                || message.contains("response missing lines array")
        }
        _ => false,
    }
}

fn merge_review_patch(
    translated: &[SubtitleSegment],
    changes: &[TranslationLine],
) -> CoreResult<Vec<TranslationLine>> {
    let mut replacements = HashMap::new();
    for change in changes {
        if change.translation.trim().is_empty()
            || replacements
                .insert(&change.id, &change.translation)
                .is_some()
        {
            return Err(CoreError::InvalidTranslation(format!(
                "review patch contains an empty or duplicate change for `{}`",
                change.id
            )));
        }
    }
    if replacements
        .keys()
        .any(|id| !translated.iter().any(|segment| segment.id == ***id))
    {
        return Err(CoreError::InvalidTranslation(
            "review patch contains an unknown id".to_owned(),
        ));
    }
    Ok(translated
        .iter()
        .map(|segment| TranslationLine {
            id: segment.id.clone(),
            translation: replacements.get(&segment.id).map_or_else(
                || segment.text.clone(),
                |translation| (*translation).clone(),
            ),
        })
        .collect())
}

fn validate_review_candidate_ids(
    batch: &ReviewBatchPlan,
    changes: &[TranslationLine],
) -> CoreResult<()> {
    if let Some(line) = changes
        .iter()
        .find(|line| !batch.candidate_reasons.contains_key(&line.id))
    {
        return Err(CoreError::InvalidTranslation(format!(
            "review attempted to modify non-candidate id `{}`",
            line.id
        )));
    }
    Ok(())
}

fn validate_window_terminology(
    prepared: &[PreparedBatch],
    generated: &HashMap<usize, BatchWithUsage>,
    required_glossary: &BTreeMap<String, String>,
    defer_missing_to_review: bool,
) -> CoreResult<()> {
    for batch in prepared {
        let Some(result) = generated.get(&(batch.index + 1)) else {
            continue;
        };
        for (segment, line) in batch.pending.iter().zip(&result.lines) {
            let source_lower = segment.text.to_lowercase();
            let translation_lower = line.translation.to_lowercase();
            for (term, target) in required_glossary {
                if source_lower.contains(&term.to_lowercase())
                    && !translation_lower.contains(&target.to_lowercase())
                    && !defer_missing_to_review
                {
                    return Err(CoreError::InvalidTranslation(format!(
                        "line {} does not use required glossary translation `{term}` -> `{target}`",
                        segment.id
                    )));
                }
            }
        }
    }
    Ok(())
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
/// Builds the stable translation-memory key used by compatible runtime data:
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
    use crate::review::build_review_plan;
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

    struct ShortParallelBackend;

    impl LlmBackend for ShortParallelBackend {
        fn supports_parallel_generation(&self) -> bool {
            true
        }

        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "short-parallel"
        }

        fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
            EchoBackend.generate_json(messages)
        }

        fn generate_many_cancellable(
            &mut self,
            _requests: Vec<crate::ports::GenerationRequest>,
            _max_concurrency: usize,
            _cancellation: &CancellationGuard,
        ) -> CoreResult<Vec<CoreResult<crate::ports::GenerationResponse>>> {
            Ok(Vec::new())
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

    struct ReviewBackend {
        translation_calls: Arc<AtomicUsize>,
        review_calls: Arc<AtomicUsize>,
        fail_on_review_call: Option<usize>,
    }

    impl LlmBackend for ReviewBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "echo"
        }

        fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
            self.translation_calls.fetch_add(1, Ordering::SeqCst);
            EchoBackend.generate_json(messages)
        }

        fn generate_raw_json(
            &mut self,
            messages: &[ChatMessage],
        ) -> CoreResult<(serde_json::Value, Usage)> {
            let call = self.review_calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_on_review_call == Some(call) {
                return Err(CoreError::Backend("scripted review failure".to_owned()));
            }
            let prompt = messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let body = prompt
                .split("REVIEW_JSON_START")
                .nth(1)
                .and_then(|value| value.split("REVIEW_JSON_END").next())
                .ok_or_else(|| CoreError::Data("missing review json".to_owned()))?;
            let parsed: serde_json::Value = serde_json::from_str(body)
                .map_err(|error| CoreError::Data(format!("invalid review json: {error}")))?;
            let lines = parsed["lines"]
                .as_array()
                .ok_or_else(|| CoreError::Data("missing review lines".to_owned()))?
                .iter()
                .map(|line| {
                    serde_json::json!({
                        "id": line["id"],
                        "translation": format!(
                            "[REVIEWED] {}",
                            line["translation"].as_str().unwrap_or_default()
                        ),
                    })
                })
                .collect::<Vec<_>>();
            Ok((
                serde_json::json!({
                    "lines": lines,
                    "review_notes": "reviewed",
                }),
                Usage {
                    input_tokens: 1,
                    output_tokens: 2,
                    total_tokens: 3,
                },
            ))
        }
    }

    struct NoChangeReviewBackend;

    impl LlmBackend for NoChangeReviewBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "no-change"
        }

        fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
            EchoBackend.generate_json(messages)
        }

        fn generate_raw_json(
            &mut self,
            _messages: &[ChatMessage],
        ) -> CoreResult<(serde_json::Value, Usage)> {
            Ok((
                serde_json::json!({"changes": []}),
                Usage {
                    input_tokens: 5,
                    output_tokens: 1,
                    total_tokens: 6,
                },
            ))
        }
    }

    struct PreflightBackend {
        contexts: Arc<Mutex<Vec<serde_json::Value>>>,
    }

    impl LlmBackend for PreflightBackend {
        fn supports_terminology_preflight(&self) -> bool {
            true
        }

        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "preflight"
        }

        fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
            let prompt = messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let context = prompt
                .split("CONTEXT_JSON_START")
                .nth(1)
                .and_then(|value| value.split("CONTEXT_JSON_END").next())
                .ok_or_else(|| CoreError::Data("missing context json".to_owned()))?;
            let context: serde_json::Value = serde_json::from_str(context)
                .map_err(|error| CoreError::Data(format!("invalid context: {error}")))?;
            self.contexts
                .lock()
                .map_err(|_| CoreError::Data("context lock poisoned".to_owned()))?
                .push(context);
            let body = prompt
                .split("BATCH_JSON_START")
                .nth(1)
                .and_then(|value| value.split("BATCH_JSON_END").next())
                .ok_or_else(|| CoreError::Data("missing batch json".to_owned()))?;
            let parsed: serde_json::Value = serde_json::from_str(body)
                .map_err(|error| CoreError::Data(format!("invalid batch: {error}")))?;
            let batch_lines = parsed["lines"]
                .as_array()
                .ok_or_else(|| CoreError::Data("missing batch lines".to_owned()))?;
            let lines = batch_lines
                .iter()
                .map(|line| TranslationLine {
                    id: line["id"].as_str().unwrap_or_default().to_owned(),
                    translation: format!("统一译名 {}", line["text"].as_str().unwrap_or_default()),
                })
                .collect();
            Ok(BackendJsonResult {
                payload: BackendPayload::Translation(BatchTranslationResult {
                    lines,
                    summary: String::new(),
                    glossary_updates: Vec::new(),
                }),
                usage: Usage::default(),
            })
        }

        fn generate_raw_json(
            &mut self,
            messages: &[ChatMessage],
        ) -> CoreResult<(serde_json::Value, Usage)> {
            let prompt = messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let body = prompt
                .split("TERMINOLOGY_JSON_START")
                .nth(1)
                .and_then(|value| value.split("TERMINOLOGY_JSON_END").next())
                .ok_or_else(|| CoreError::Data("missing terminology json".to_owned()))?;
            let parsed: serde_json::Value = serde_json::from_str(body)
                .map_err(|error| CoreError::Data(format!("invalid terminology: {error}")))?;
            let candidates = parsed["candidates"]
                .as_array()
                .ok_or_else(|| CoreError::Data("missing terminology candidates".to_owned()))?;
            let entries = candidates
                .iter()
                .map(|candidate| {
                    serde_json::json!({
                        "source": candidate["source"],
                        "target": "统一译名",
                    })
                })
                .collect::<Vec<_>>();
            Ok((serde_json::json!({"entries": entries}), Usage::default()))
        }
    }

    struct StructuralFailureBackend {
        call_sizes: Arc<Mutex<Vec<usize>>>,
    }

    impl LlmBackend for StructuralFailureBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "structural"
        }

        fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
            let mut response = EchoBackend.generate_json(messages)?;
            let BackendPayload::Translation(result) = &mut response.payload else {
                unreachable!();
            };
            self.call_sizes
                .lock()
                .expect("call sizes lock")
                .push(result.lines.len());
            if result.lines.len() > 1 {
                result.lines.pop();
            } else {
                result.lines[0].translation =
                    result.lines[0].translation.replacen("[ECHO]", "[SPLIT]", 1);
            }
            Ok(response)
        }
    }

    struct AgentRepairBackend {
        regular_calls: Arc<AtomicUsize>,
        repair_calls: Arc<AtomicUsize>,
        repair_succeeds: bool,
    }

    impl LlmBackend for AgentRepairBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "repair"
        }

        fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
            self.regular_calls.fetch_add(1, Ordering::SeqCst);
            let mut response = EchoBackend.generate_json(messages)?;
            let BackendPayload::Translation(result) = &mut response.payload else {
                unreachable!();
            };
            for line in &mut result.lines {
                line.translation.clear();
            }
            Ok(response)
        }

        fn generate_raw_json(
            &mut self,
            messages: &[ChatMessage],
        ) -> CoreResult<(serde_json::Value, Usage)> {
            self.repair_calls.fetch_add(1, Ordering::SeqCst);
            let prompt = messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let body = prompt
                .split("AGENT_REPAIR_JSON_START")
                .nth(1)
                .and_then(|value| value.split("AGENT_REPAIR_JSON_END").next())
                .ok_or_else(|| CoreError::Data("missing agent repair json".to_owned()))?;
            let payload: serde_json::Value = serde_json::from_str(body)
                .map_err(|error| CoreError::Data(format!("invalid repair json: {error}")))?;
            let source_lines = payload["source_lines"]
                .as_array()
                .ok_or_else(|| CoreError::Data("missing repair source lines".to_owned()))?;
            let lines = source_lines
                .iter()
                .map(|line| {
                    serde_json::json!({
                        "id": line["id"],
                        "translation": if self.repair_succeeds {
                            format!("[AGENT] {}", line["text"].as_str().unwrap_or_default())
                        } else {
                            String::new()
                        },
                    })
                })
                .collect::<Vec<_>>();
            Ok((
                serde_json::json!({
                    "lines": lines,
                    "summary": "agent repaired",
                    "glossary_updates": [],
                }),
                Usage {
                    input_tokens: 3,
                    output_tokens: 4,
                    total_tokens: 7,
                },
            ))
        }
    }

    struct AgentReviewBackend {
        review_calls: Arc<AtomicUsize>,
        repair_calls: Arc<AtomicUsize>,
    }

    impl LlmBackend for AgentReviewBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "review-repair"
        }

        fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
            EchoBackend.generate_json(messages)
        }

        fn generate_raw_json(
            &mut self,
            messages: &[ChatMessage],
        ) -> CoreResult<(serde_json::Value, Usage)> {
            let prompt = messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            if prompt.contains("TASK_START\nreview_translations\nTASK_END") {
                self.review_calls.fetch_add(1, Ordering::SeqCst);
                let body = prompt
                    .split("REVIEW_JSON_START")
                    .nth(1)
                    .and_then(|value| value.split("REVIEW_JSON_END").next())
                    .ok_or_else(|| CoreError::Data("missing review json".to_owned()))?;
                let payload: serde_json::Value = serde_json::from_str(body)
                    .map_err(|error| CoreError::Data(format!("invalid review json: {error}")))?;
                let lines = payload["lines"]
                    .as_array()
                    .ok_or_else(|| CoreError::Data("missing review lines".to_owned()))?
                    .iter()
                    .map(|line| serde_json::json!({"id": line["id"], "translation": ""}))
                    .collect::<Vec<_>>();
                return Ok((
                    serde_json::json!({"lines": lines, "review_notes": "broken"}),
                    Usage::default(),
                ));
            }

            self.repair_calls.fetch_add(1, Ordering::SeqCst);
            let body = prompt
                .split("AGENT_REPAIR_JSON_START")
                .nth(1)
                .and_then(|value| value.split("AGENT_REPAIR_JSON_END").next())
                .ok_or_else(|| CoreError::Data("missing review repair json".to_owned()))?;
            let payload: serde_json::Value = serde_json::from_str(body)
                .map_err(|error| CoreError::Data(format!("invalid review repair json: {error}")))?;
            let current = payload["current_translations"]
                .as_array()
                .ok_or_else(|| CoreError::Data("missing current translations".to_owned()))?;
            let lines = current
                .iter()
                .map(|line| {
                    serde_json::json!({
                        "id": line["id"],
                        "translation": line["translation"],
                    })
                })
                .collect::<Vec<_>>();
            Ok((
                serde_json::json!({"lines": lines, "review_notes": "agent repaired"}),
                Usage {
                    input_tokens: 2,
                    output_tokens: 2,
                    total_tokens: 4,
                },
            ))
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
    fn parallel_backend_response_count_must_match_request_count() {
        let mut options = PipelineOptions::new("clip.txt".into());
        options.translation_concurrency = 2;
        options.batch_size = 1;
        let mut pipeline = SubtitlePipeline::new(ShortParallelBackend, NoopDashboard, options);

        let error = pipeline
            .run_document(&document("clip.txt", &["one", "two"]))
            .expect_err("short batch responses must be rejected");
        assert!(
            error
                .to_string()
                .contains("responses for 2 translation requests")
        );
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
                std::path::Path::new("/workspace/clip.txt"),
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
        options.retries = 0;
        let signature = input_signature_from_bytes(b"one\ntwo\nthree\n", Some(1));
        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("resume.txt"),
                std::path::Path::new("/workspace/resume.txt"),
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
                std::path::Path::new("/workspace/cache.txt"),
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

    #[test]
    fn pipeline_retries_transient_backend_failure() {
        let document = document("retry.txt", &["hello"]);
        let mut options = PipelineOptions::new("retry.txt".into());
        options.review_policy = ReviewPolicy::Off;
        options.retries = 1;
        let calls = Arc::new(AtomicUsize::new(0));
        let mut pipeline = SubtitlePipeline::new(
            CountingBackend {
                calls: Arc::clone(&calls),
                fail_on_call: Some(1),
            },
            NoopDashboard,
            options,
        );

        let run = pipeline.run_document(&document).expect("retry succeeds");

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(run.translated_segments[0].text, "[ECHO] hello");
    }

    #[test]
    fn structural_failures_recursively_split_translation_batch() {
        let document = document("split.txt", &["one", "two", "three", "four"]);
        let mut options = PipelineOptions::new("split.txt".into());
        options.batch_size = 8;
        options.review_policy = ReviewPolicy::Off;
        options.retries = 0;
        options.agent = false;
        let call_sizes = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = SubtitlePipeline::new(
            StructuralFailureBackend {
                call_sizes: Arc::clone(&call_sizes),
            },
            NoopDashboard,
            options,
        );

        let run = pipeline.run_document(&document).expect("split succeeds");

        assert_eq!(
            *call_sizes.lock().expect("call sizes lock"),
            vec![4, 2, 1, 1, 2, 1, 1]
        );
        assert!(
            run.translated_segments
                .iter()
                .all(|segment| segment.text.starts_with("[SPLIT]"))
        );
    }

    #[test]
    fn agent_repair_continues_pipeline_and_records_log() {
        let document = document("agent.txt", &["Alpha."]);
        let mut options = PipelineOptions::new("agent.txt".into());
        options.review_policy = ReviewPolicy::Off;
        options.retries = 0;
        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("agent.txt"),
                std::path::Path::new("/workspace/agent.txt"),
                None,
                None,
                "Auto",
                "Chinese",
                false,
            ),
            data: Arc::clone(&captured),
        };
        let regular_calls = Arc::new(AtomicUsize::new(0));
        let repair_calls = Arc::new(AtomicUsize::new(0));
        let mut pipeline = SubtitlePipeline::new(
            AgentRepairBackend {
                regular_calls: Arc::clone(&regular_calls),
                repair_calls: Arc::clone(&repair_calls),
                repair_succeeds: true,
            },
            NoopDashboard,
            options,
        )
        .with_store(Box::new(store));

        let run = pipeline.run_document(&document).expect("agent repairs");

        assert_eq!(regular_calls.load(Ordering::SeqCst), 1);
        assert_eq!(repair_calls.load(Ordering::SeqCst), 1);
        assert_eq!(run.translated_segments[0].text, "[AGENT] Alpha.");
        assert_eq!(run.result.agent_repairs.len(), 1);
        assert!(run.result.agent_repairs[0].success);
        let data = captured.lock().expect("capture lock");
        assert!(data.agent_logs.last().expect("agent log").success);
        assert!(data.failure_logs.is_empty());
    }

    #[test]
    fn agent_repair_reports_missing_log_path_without_a_runtime_store() {
        let document = document("agent-no-store.txt", &["Alpha."]);
        let mut options = PipelineOptions::new("agent-no-store.txt".into());
        options.review_policy = ReviewPolicy::Off;
        options.retries = 0;
        let mut pipeline = SubtitlePipeline::new(
            AgentRepairBackend {
                regular_calls: Arc::new(AtomicUsize::new(0)),
                repair_calls: Arc::new(AtomicUsize::new(0)),
                repair_succeeds: true,
            },
            NoopDashboard,
            options,
        );

        let run = pipeline.run_document(&document).expect("agent repairs");
        assert_eq!(run.result.agent_repairs.len(), 1);
        assert_eq!(run.result.agent_repairs[0].log_path, None);
    }

    #[test]
    fn agent_repair_cache_bypasses_second_repair_call() {
        let document = document("agent-cache.txt", &["Alpha."]);
        let mut options = PipelineOptions::new("agent-cache.txt".into());
        options.review_policy = ReviewPolicy::Off;
        options.retries = 0;
        options.resume = false;
        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("agent-cache.txt"),
                std::path::Path::new("/workspace/agent-cache.txt"),
                None,
                None,
                "Auto",
                "Chinese",
                false,
            ),
            data: Arc::clone(&captured),
        };
        let mut first = SubtitlePipeline::new(
            AgentRepairBackend {
                regular_calls: Arc::new(AtomicUsize::new(0)),
                repair_calls: Arc::new(AtomicUsize::new(0)),
                repair_succeeds: true,
            },
            NoopDashboard,
            options.clone(),
        )
        .with_store(Box::new(store.clone()));
        first.run_document(&document).expect("prime repair cache");

        let repair_calls = Arc::new(AtomicUsize::new(0));
        let mut second = SubtitlePipeline::new(
            AgentRepairBackend {
                regular_calls: Arc::new(AtomicUsize::new(0)),
                repair_calls: Arc::clone(&repair_calls),
                repair_succeeds: false,
            },
            NoopDashboard,
            options,
        )
        .with_store(Box::new(store));
        let run = second.run_document(&document).expect("cached repair");

        assert_eq!(repair_calls.load(Ordering::SeqCst), 0);
        assert_eq!(run.result.cache_hits, 1);
        assert_eq!(run.translated_segments[0].text, "[AGENT] Alpha.");
    }

    #[test]
    fn agent_can_repair_review_validation_failure() {
        let document = document("review-agent.txt", &["Meet Alice now."]);
        let mut options = PipelineOptions::new("review-agent.txt".into());
        options.batch_size = 1;
        options.retries = 0;
        options.review_policy = ReviewPolicy::Full;
        let review_calls = Arc::new(AtomicUsize::new(0));
        let repair_calls = Arc::new(AtomicUsize::new(0));
        let mut pipeline = SubtitlePipeline::new(
            AgentReviewBackend {
                review_calls: Arc::clone(&review_calls),
                repair_calls: Arc::clone(&repair_calls),
            },
            NoopDashboard,
            options,
        );

        let run = pipeline.run_document(&document).expect("review repaired");

        assert_eq!(review_calls.load(Ordering::SeqCst), 1);
        assert_eq!(repair_calls.load(Ordering::SeqCst), 1);
        assert_eq!(run.result.agent_repairs.len(), 1);
        assert_eq!(run.result.agent_repairs[0].stage, "review");
        assert_eq!(run.translated_segments[0].text, "[ECHO] Meet Alice now.");
    }

    #[test]
    fn failed_agent_repair_persists_failure_and_attempts() {
        let document = document("agent-fail.txt", &["Alpha."]);
        let mut options = PipelineOptions::new("agent-fail.txt".into());
        options.review_policy = ReviewPolicy::Off;
        options.retries = 0;
        options.agent_repair_attempts = 2;
        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("agent-fail.txt"),
                std::path::Path::new("/workspace/agent-fail.txt"),
                None,
                None,
                "Auto",
                "Chinese",
                false,
            ),
            data: Arc::clone(&captured),
        };
        let mut pipeline = SubtitlePipeline::new(
            AgentRepairBackend {
                regular_calls: Arc::new(AtomicUsize::new(0)),
                repair_calls: Arc::new(AtomicUsize::new(0)),
                repair_succeeds: false,
            },
            NoopDashboard,
            options,
        )
        .with_store(Box::new(store));

        let error = pipeline
            .run_document(&document)
            .expect_err("agent repair fails");

        assert!(error.to_string().contains("Agent repair failed after 2"));
        let data = captured.lock().expect("capture lock");
        assert_eq!(data.agent_logs.last().expect("agent log").attempts.len(), 2);
        assert_eq!(
            data.failure_logs
                .last()
                .expect("failure log")
                .agent_attempts
                .len(),
            2
        );
    }

    #[test]
    fn review_plan_selects_only_high_risk_batches() {
        let batches = vec![
            vec![segment("1", "Hello there.")],
            vec![segment("2", "Meet <i>Alice</i> now.")],
            vec![segment("3", &"long ".repeat(20))],
        ];
        let translated = vec![
            segment("1", "你好。"),
            segment("2", "现在去见爱丽丝。"),
            segment("3", "这是一条很长但有效的译文。"),
        ];

        let plan = build_review_plan(
            &batches,
            &translated,
            &ContextMemory::new(),
            "en",
            "zh-Hans",
        );

        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].source[0].id, "2");
        assert_eq!(plan[0].reasons, vec!["formatting mismatch"]);
    }

    #[test]
    fn targeted_review_ignores_names_speakers_and_multiline_text_by_themselves() {
        let batches = vec![vec![
            segment("1", "Meet Alice now.\nShe is waiting."),
            segment("2", "- Bob: Come here."),
        ]];
        let translated = vec![
            segment("1", "现在去见爱丽丝。\n她正在等。"),
            segment("2", "- 鲍勃：过来。"),
        ];

        let plan = build_review_plan(
            &batches,
            &translated,
            &ContextMemory::new(),
            "en",
            "zh-Hans",
        );

        assert!(plan.is_empty());
    }

    #[test]
    fn terminology_payload_accepts_only_known_nonempty_candidates() {
        let candidates = vec![TerminologyCandidate {
            source: "Axe Gang".to_owned(),
            context: "The Axe Gang is here.".to_owned(),
        }];
        let parsed = parse_terminology_payload(
            &serde_json::json!({
                "entries": [{"source": "Axe Gang", "target": "斧头帮"}]
            }),
            &candidates,
        )
        .expect("valid terminology");
        assert_eq!(parsed.entries[0].target, "斧头帮");

        let error = parse_terminology_payload(
            &serde_json::json!({
                "entries": [{"source": "Unknown", "target": "未知"}]
            }),
            &candidates,
        )
        .expect_err("unknown source rejected");
        assert!(error.to_string().contains("unknown source"));
    }

    #[test]
    fn terminology_candidates_normalize_english_possessives() {
        let segments = vec![
            segment("18", "MacAndrews'."),
            segment("19", "MacClannough's horse."),
            segment("20", "James’ horse."),
        ];

        let sources = extract_terminology_candidates(&segments)
            .into_iter()
            .map(|candidate| candidate.source)
            .collect::<Vec<_>>();

        assert!(sources.contains(&"MacAndrews".to_owned()));
        assert!(sources.contains(&"MacClannough".to_owned()));
        assert!(sources.contains(&"James".to_owned()));
        assert!(!sources.iter().any(|source| source.contains(['\'', '’'])));
    }

    #[test]
    fn auto_terminology_is_advisory_but_explicit_glossary_is_required() {
        let prepared = vec![translation_stage::PreparedBatch {
            index: 0,
            memory_hits: HashMap::new(),
            pending: vec![segment("69", "The Lord bless thee and keep thee.")],
        }];
        let generated = HashMap::from([(
            1,
            BatchWithUsage {
                lines: vec![TranslationLine {
                    id: "69".to_owned(),
                    translation: "愿主保佑你，保护你。".to_owned(),
                }],
                summary: String::new(),
                glossary_updates: vec![GlossaryEntry {
                    source: "Lord".to_owned(),
                    target: "勋爵".to_owned(),
                }],
                usage: Usage::default(),
            },
        )]);

        validate_window_terminology(&prepared, &generated, &BTreeMap::new(), false)
            .expect("automatically learned terminology must remain advisory");

        let mut memory = ContextMemory::new();
        memory.load_glossary(&[("Lord".to_owned(), "勋爵".to_owned())]);
        let options = PipelineOptions::new("terms.srt".into());
        let advisory_messages = build_translation_messages(
            &options,
            1,
            &prepared[0].pending,
            &memory,
            &BTreeMap::new(),
        );
        let advisory_context = translation_context(&advisory_messages);
        assert_eq!(advisory_context["terminology_hints"]["Lord"], "勋爵");
        assert!(advisory_context.get("glossary").is_none());

        let required = BTreeMap::from([("Lord".to_owned(), "勋爵".to_owned())]);
        let required_messages =
            build_translation_messages(&options, 1, &prepared[0].pending, &memory, &required);
        let required_context = translation_context(&required_messages);
        assert_eq!(required_context["glossary"]["Lord"], "勋爵");
        assert!(required_context.get("terminology_hints").is_none());

        let error = validate_window_terminology(&prepared, &generated, &required, false)
            .expect_err("an explicit user glossary must remain authoritative");
        assert!(error.to_string().contains("line 69"));
        assert!(error.to_string().contains("`Lord` -> `勋爵`"));
    }

    #[test]
    fn terminology_preflight_freezes_glossary_before_all_translation_batches() {
        let document = document("terms.txt", &["Meet Alice now.", "Meet Bob now."]);
        let mut options = PipelineOptions::new("terms.txt".into());
        options.batch_size = 1;
        options.resume = false;
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = SubtitlePipeline::new(
            PreflightBackend {
                contexts: Arc::clone(&contexts),
            },
            NoopDashboard,
            options,
        );

        let run = pipeline.run_document(&document).expect("translated");
        let contexts = contexts.lock().expect("contexts lock");

        assert!(run.result.terminology.entries_added >= 2);
        assert_eq!(contexts.len(), 2);
        assert!(contexts.iter().all(|context| {
            context["terminology_hints"]
                .as_object()
                .is_some_and(|map| !map.is_empty())
                && context.get("glossary").is_none()
        }));
    }

    #[test]
    fn token_budget_batches_short_and_long_segments_deterministically() {
        let segments = document("budget.txt", &["one", "two", &"x".repeat(80)]).segments;
        let batches = BatchPlanner::new(80, 20).split(&segments);
        assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 1]);
    }

    #[test]
    fn review_patch_preserves_unchanged_translations() {
        let translated = document("patch.txt", &["甲", "乙"]).segments;
        let merged = merge_review_patch(
            &translated,
            &[TranslationLine {
                id: "2".to_owned(),
                translation: "丙".to_owned(),
            }],
        )
        .expect("valid patch");
        assert_eq!(merged[0].translation, "甲");
        assert_eq!(merged[1].translation, "丙");
    }

    #[test]
    fn pipeline_reviews_high_risk_batches_and_replaces_output() {
        let document = document("review.txt", &["Meet Alice now.", "move now."]);
        let mut options = PipelineOptions::new("review.txt".into());
        options.batch_size = 2;
        options.resume = false;
        options.review_policy = ReviewPolicy::Full;
        let translation_calls = Arc::new(AtomicUsize::new(0));
        let review_calls = Arc::new(AtomicUsize::new(0));
        let mut pipeline = SubtitlePipeline::new(
            ReviewBackend {
                translation_calls: Arc::clone(&translation_calls),
                review_calls: Arc::clone(&review_calls),
                fail_on_review_call: None,
            },
            NoopDashboard,
            options,
        );

        let run = pipeline.run_document(&document).expect("reviewed run");

        assert_eq!(translation_calls.load(Ordering::SeqCst), 1);
        assert_eq!(review_calls.load(Ordering::SeqCst), 1);
        assert_eq!(run.result.review_batches, 1);
        assert_eq!(run.result.usage.total_tokens, 5);
        assert_eq!(
            run.translated_segments[0].text,
            "[REVIEWED] [ECHO] Meet Alice now."
        );
    }

    #[test]
    fn full_review_records_zero_changes_for_an_empty_patch() {
        let document = document("review-zero.txt", &["Meet Alice now."]);
        let mut options = PipelineOptions::new("review-zero.txt".into());
        options.review_policy = ReviewPolicy::Full;
        options.resume = false;
        let mut pipeline = SubtitlePipeline::new(NoChangeReviewBackend, NoopDashboard, options);

        let run = pipeline.run_document(&document).expect("reviewed run");

        assert_eq!(run.result.review.candidate_lines, 1);
        assert_eq!(run.result.review.reviewed_lines, 1);
        assert_eq!(run.result.review.changed_lines, 0);
        assert_eq!(run.result.review.usage.total_tokens, 6);
        assert_eq!(run.translated_segments[0].text, "[ECHO] Meet Alice now.");
    }

    #[test]
    fn pipeline_resumes_review_batches_from_shards() {
        let document = document("review-resume.txt", &["Meet Alice now.", "Meet Bob now."]);
        let mut options = PipelineOptions::new("review-resume.txt".into());
        options.batch_size = 1;
        options.retries = 0;
        options.review_policy = ReviewPolicy::Full;
        let signature = input_signature_from_bytes(b"Meet Alice now.\nMeet Bob now.\n", Some(1));
        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("review-resume.txt"),
                std::path::Path::new("/workspace/review-resume.txt"),
                None,
                None,
                "Auto",
                "Chinese",
                false,
            ),
            data: Arc::clone(&captured),
        };
        let first_review_calls = Arc::new(AtomicUsize::new(0));
        let mut first = SubtitlePipeline::new(
            ReviewBackend {
                translation_calls: Arc::new(AtomicUsize::new(0)),
                review_calls: Arc::clone(&first_review_calls),
                fail_on_review_call: Some(2),
            },
            NoopDashboard,
            options.clone(),
        )
        .with_store(Box::new(store.clone()))
        .with_input_signature(signature.clone());

        first
            .run_document(&document)
            .expect_err("second review fails");
        assert_eq!(first_review_calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            captured
                .lock()
                .expect("capture lock")
                .saved_state
                .as_ref()
                .expect("review checkpoint")
                .review_batches_completed,
            1
        );

        let resumed_translation_calls = Arc::new(AtomicUsize::new(0));
        let resumed_review_calls = Arc::new(AtomicUsize::new(0));
        let mut resumed = SubtitlePipeline::new(
            ReviewBackend {
                translation_calls: Arc::clone(&resumed_translation_calls),
                review_calls: Arc::clone(&resumed_review_calls),
                fail_on_review_call: None,
            },
            NoopDashboard,
            options,
        )
        .with_store(Box::new(store))
        .with_input_signature(signature);
        let run = resumed.run_document(&document).expect("resume review");

        assert_eq!(resumed_translation_calls.load(Ordering::SeqCst), 0);
        assert_eq!(resumed_review_calls.load(Ordering::SeqCst), 1);
        assert_eq!(run.result.resumed_translation_batches, 2);
        assert_eq!(run.result.resumed_review_batches, 1);
        assert_eq!(run.result.review_batches, 2);
        assert!(
            run.translated_segments
                .iter()
                .all(|segment| segment.text.starts_with("[REVIEWED]"))
        );
    }

    #[test]
    fn pipeline_reuses_review_request_cache() {
        let document = document("review-cache.txt", &["Meet Alice now."]);
        let mut options = PipelineOptions::new("review-cache.txt".into());
        options.batch_size = 1;
        options.resume = false;
        options.review_policy = ReviewPolicy::Full;
        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("review-cache.txt"),
                std::path::Path::new("/workspace/review-cache.txt"),
                None,
                None,
                "Auto",
                "Chinese",
                false,
            ),
            data: Arc::clone(&captured),
        };
        let mut first = SubtitlePipeline::new(
            ReviewBackend {
                translation_calls: Arc::new(AtomicUsize::new(0)),
                review_calls: Arc::new(AtomicUsize::new(0)),
                fail_on_review_call: None,
            },
            NoopDashboard,
            options.clone(),
        )
        .with_store(Box::new(store.clone()));
        first.run_document(&document).expect("prime cache");

        let translation_calls = Arc::new(AtomicUsize::new(0));
        let review_calls = Arc::new(AtomicUsize::new(0));
        let mut second = SubtitlePipeline::new(
            ReviewBackend {
                translation_calls: Arc::clone(&translation_calls),
                review_calls: Arc::clone(&review_calls),
                fail_on_review_call: Some(1),
            },
            NoopDashboard,
            options,
        )
        .with_store(Box::new(store));
        let run = second.run_document(&document).expect("cached review");

        assert_eq!(translation_calls.load(Ordering::SeqCst), 0);
        assert_eq!(review_calls.load(Ordering::SeqCst), 0);
        assert_eq!(run.result.cache_hits, 2);
        assert_eq!(run.result.usage, Usage::default());
        assert_eq!(
            run.translated_segments[0].text,
            "[REVIEWED] [ECHO] Meet Alice now."
        );
    }

    fn translation_context(messages: &[ChatMessage]) -> serde_json::Value {
        let context = messages
            .iter()
            .find(|message| message.role == "user")
            .and_then(|message| message.content.split("CONTEXT_JSON_START").nth(1))
            .and_then(|value| value.split("CONTEXT_JSON_END").next())
            .expect("translation context");
        serde_json::from_str(context).expect("valid translation context")
    }

    fn segment(id: &str, text: &str) -> SubtitleSegment {
        SubtitleSegment {
            id: id.to_owned(),
            text: text.to_owned(),
            start: None,
            end: None,
            identifier: None,
            settings: None,
        }
    }

    fn document(path: &str, texts: &[&str]) -> SubtitleDocument {
        SubtitleDocument {
            path: path.into(),
            format: "txt".to_owned(),
            segments: texts
                .iter()
                .enumerate()
                .map(|(index, text)| segment(&(index + 1).to_string(), text))
                .collect(),
            header: None,
            passthrough_blocks: Vec::new(),
        }
    }

    #[derive(Debug, Default)]
    struct CapturedStoreData {
        saved_translation_memory: Vec<(String, String)>,
        review_reports: Vec<ReviewReport>,
        saved_batches: Vec<(usize, Vec<SubtitleSegment>)>,
        saved_review_batches: Vec<(usize, Vec<SubtitleSegment>)>,
        saved_state: Option<RunState>,
        cached_responses: Vec<(CacheStage, String, BackendJsonResult)>,
        failure_logs: Vec<FailureLog>,
        agent_logs: Vec<AgentLog>,
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

        fn save_review_report(&self, report: &ReviewReport) -> CoreResult<()> {
            self.data
                .lock()
                .expect("capture lock")
                .review_reports
                .push(report.clone());
            Ok(())
        }

        fn save_batch_segments(
            &self,
            kind: BatchShardKind,
            batch_index: usize,
            segments: &[SubtitleSegment],
        ) -> CoreResult<()> {
            let mut data = self.data.lock().expect("capture lock");
            match kind {
                BatchShardKind::Translated => {
                    data.saved_batches.push((batch_index, segments.to_vec()))
                }
                BatchShardKind::Reviewed => data
                    .saved_review_batches
                    .push((batch_index, segments.to_vec())),
            }
            Ok(())
        }

        fn load_batch_segments(
            &self,
            kind: BatchShardKind,
            completed_batches: usize,
        ) -> CoreResult<Vec<SubtitleSegment>> {
            let data = self.data.lock().expect("capture lock");
            let batches = match kind {
                BatchShardKind::Translated => &data.saved_batches,
                BatchShardKind::Reviewed => &data.saved_review_batches,
            };
            Ok(batches
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

        fn save_failure_log(&self, log: &FailureLog) -> CoreResult<PathBuf> {
            self.data
                .lock()
                .expect("capture lock")
                .failure_logs
                .push(log.clone());
            Ok(self
                .paths
                .failures_dir
                .join(format!("{}_batch_{:04}.json", log.stage, log.batch_index)))
        }

        fn save_agent_log(&self, log: &AgentLog) -> CoreResult<PathBuf> {
            self.data
                .lock()
                .expect("capture lock")
                .agent_logs
                .push(log.clone());
            Ok(self
                .paths
                .agent_logs_dir
                .join(format!("{}_batch_{:04}.json", log.stage, log.batch_index)))
        }
    }
}
