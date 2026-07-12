use std::collections::HashMap;
use std::path::PathBuf;

use crate::CancellationGuard;
use crate::entities::{
    AgentLog, AgentRepairRecord, AttemptLog, BatchPlanEntry, FailureLog, GlossaryEntry,
    PipelineOptions, PipelineResult, SplitRetryLog, SubtitleDocument, SubtitleSegment,
    TranslationLine, Usage,
};
use crate::error::{CoreError, CoreResult};
use crate::languages::normalize_language_name;
use crate::memory::ContextMemory;
use crate::ports::{
    BackendJsonResult, BackendPayload, BatchShardKind, CacheStage, ChatMessage, DashboardSink,
    LlmBackend, RuntimeStore,
};
use crate::progress::{ProgressEvent, ProgressSink, ProgressUnit, TaskKind, TaskState};
use crate::recovery::{
    backend_payload_json, build_agent_repair_messages, combine_glossary, combine_summaries,
    parse_translation_payload, retry_correction_message, split_index,
};
use crate::review::{
    ReviewBatchPlan, build_review_messages, build_review_plan, parse_review_payload,
    restore_review_progress,
};
use crate::storage::{
    InputSignature, JsonValue, ResumeSnapshot, RunState, build_request_hash, build_request_hash_v2,
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
            });
        }
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

        // Load persisted glossary into context memory at start.
        if let Some(ref store) = self.store {
            self.cancellation.check()?;
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
                    agent_repairs: self.agent_repairs.clone(),
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
            self.cancellation.check()?;
            self.report(
                "TRANSLATE",
                TaskState::Running,
                batch_index,
                Some(batches.len()),
                resume.translation_batches_completed,
                usage,
            );
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
            self.cancellation.check()?;
            self.save_run_state(
                batch_index + 1,
                resume.review_batches_completed,
                false,
                usage,
            )?;
            self.report(
                "TRANSLATE",
                TaskState::Running,
                batch_index + 1,
                Some(batches.len()),
                resume.translation_batches_completed,
                usage,
            );
        }

        validate_full_alignment(&document.segments, &translated_segments)?;
        self.cancellation.check()?;
        self.save_run_state(batches.len(), resume.review_batches_completed, true, usage)?;
        self.dashboard.mark_done("TRANSLATE");

        let review_plan = if self.options.final_review
            && !self.options.fast_mode
            && !translated_segments.is_empty()
        {
            build_review_plan(&batches, &translated_segments, &self.memory)
        } else {
            Vec::new()
        };
        self.dashboard
            .set_total_steps(3 + batches.len() + review_plan.len());

        let resumed_review_batches = if review_plan.is_empty() {
            0
        } else {
            if resume.review_batches_completed > review_plan.len() {
                return Err(CoreError::Data(format!(
                    "resume state has {} reviewed batches, but the current review plan has only {}",
                    resume.review_batches_completed,
                    review_plan.len()
                )));
            }
            resume.review_batches_completed
        };
        let mut output_segments = translated_segments.clone();
        if !review_plan.is_empty() {
            restore_review_progress(
                &review_plan,
                resumed_review_batches,
                &resume.reviewed_segments,
                &mut output_segments,
            )?;
            self.dashboard.mark_running("FINAL_REVIEW");
            for (review_index, review_batch) in
                review_plan.iter().enumerate().skip(resumed_review_batches)
            {
                self.cancellation.check()?;
                self.report(
                    "FINAL_REVIEW",
                    TaskState::Running,
                    review_index,
                    Some(review_plan.len()),
                    resumed_review_batches,
                    usage,
                );
                let review_position = review_index + 1;
                let reviewed = self.review_batch(review_position, review_batch)?;
                usage.add(reviewed.usage);
                self.dashboard.add_usage(reviewed.usage);
                let reviewed_segments = apply_lines(&review_batch.source, &reviewed.lines);
                if let Some(ref store) = self.store {
                    self.cancellation.check()?;
                    store.save_batch_segments(
                        BatchShardKind::Reviewed,
                        review_position,
                        &reviewed_segments,
                    )?;
                }
                output_segments[review_batch.start_offset
                    ..review_batch.start_offset + reviewed_segments.len()]
                    .clone_from_slice(&reviewed_segments);
                self.save_run_state(batches.len(), review_position, true, usage)?;
                self.report(
                    "FINAL_REVIEW",
                    TaskState::Running,
                    review_position,
                    Some(review_plan.len()),
                    resumed_review_batches,
                    usage,
                );
            }
            validate_full_alignment(&document.segments, &output_segments)?;
            self.dashboard.mark_done("FINAL_REVIEW");
        }
        self.cancellation.check()?;
        self.save_run_state(batches.len(), review_plan.len(), true, usage)?;
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
                review_batches: review_plan.len(),
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
            },
            translated_segments: output_segments,
        })
    }

    fn translate_batch(
        &mut self,
        batch_index: usize,
        batch: &[SubtitleSegment],
    ) -> CoreResult<BatchWithUsage> {
        self.translate_batch_impl(batch_index, batch, true)
    }

    fn translate_batch_impl(
        &mut self,
        batch_index: usize,
        batch: &[SubtitleSegment],
        record_failure: bool,
    ) -> CoreResult<BatchWithUsage> {
        let mut last_error = None;
        let mut attempts = Vec::new();
        for attempt in 1..=self.options.retries + 1 {
            self.cancellation.check()?;
            let mut messages =
                build_translation_messages(&self.options, batch_index, batch, &self.memory);
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
        let left = self.translate_batch_impl(batch_index, &batch[..split], false)?;
        let right = self.translate_batch_impl(batch_index, &batch[split..], false)?;
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
        let mut last_error = None;
        let mut attempts = Vec::new();
        for attempt in 1..=self.options.retries + 1 {
            self.cancellation.check()?;
            let mut messages = build_review_messages(
                &self.options,
                &batch.source,
                &batch.translated,
                &batch.reasons,
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
                BackendJsonResult {
                    payload: BackendPayload::Review(parse_review_payload(&payload)?),
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
        let mut log_path = PathBuf::new();
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

    fn save_agent_log(&self, log: AgentLog) -> CoreResult<PathBuf> {
        self.store
            .as_ref()
            .map(|store| store.save_agent_log(&log))
            .transpose()
            .map(|path| path.unwrap_or_default())
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

#[derive(Debug, Clone)]
struct RepairOutcome {
    payload: Option<BackendPayload>,
    usage: Usage,
    attempts: Vec<AttemptLog>,
    log_path: PathBuf,
    error: Option<String>,
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
        ChatMessage::system(
            "TASK_START\ntranslate_subtitles\nTASK_END\n\
Return JSON only with this shape:\n\
{\"lines\":[{\"id\":\"<source id>\",\"translation\":\"<non-empty target-language text>\"}],\"summary\":\"\",\"glossary_updates\":[]}\n\
Return exactly one line for every input line, in the same order. Copy each id exactly.\n\
The translated text must be in the translation field; do not return it in text or translated_text.\n\
Every non-empty source line must have a non-empty translation. Do not include markdown or explanations.",
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
        if !repair.log_path.as_os_str().is_empty() {
            message.push_str(&format!(
                "\nAgent log saved to:\n{}",
                repair.log_path.display()
            ));
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
        options.retries = 0;
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

    #[test]
    fn pipeline_retries_transient_backend_failure() {
        let document = document("retry.txt", &["hello"]);
        let mut options = PipelineOptions::new("retry.txt".into());
        options.final_review = false;
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
        options.final_review = false;
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
        options.final_review = false;
        options.retries = 0;
        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("agent.txt"),
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
    fn agent_repair_cache_bypasses_second_repair_call() {
        let document = document("agent-cache.txt", &["Alpha."]);
        let mut options = PipelineOptions::new("agent-cache.txt".into());
        options.final_review = false;
        options.retries = 0;
        options.resume = false;
        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("agent-cache.txt"),
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
        options.final_review = false;
        options.retries = 0;
        options.agent_repair_attempts = 2;
        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("agent-fail.txt"),
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
            vec![segment("2", "Meet Alice now.")],
            vec![segment("3", &"long ".repeat(20))],
        ];
        let translated = batches
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<SubtitleSegment>>();

        let plan = build_review_plan(&batches, &translated, &ContextMemory::new());

        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].source[0].id, "2");
        assert_eq!(plan[0].reasons, vec!["names and terms"]);
    }

    #[test]
    fn pipeline_reviews_high_risk_batches_and_replaces_output() {
        let document = document("review.txt", &["Meet Alice now.", "move now."]);
        let mut options = PipelineOptions::new("review.txt".into());
        options.batch_size = 2;
        options.resume = false;
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
    fn pipeline_resumes_review_batches_from_shards() {
        let document = document("review-resume.txt", &["Meet Alice now.", "Meet Bob now."]);
        let mut options = PipelineOptions::new("review-resume.txt".into());
        options.batch_size = 1;
        options.retries = 0;
        let signature = input_signature_from_bytes(b"Meet Alice now.\nMeet Bob now.\n", Some(1));
        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("review-resume.txt"),
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
        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("review-cache.txt"),
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
