use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use crate::CancellationGuard;
use crate::entities::{
    AgentLog, AgentRepairRecord, AttemptLog, BatchTranslationResult, ConcurrencyStrategy,
    FailureLog, GlossaryEntry, PipelineOptions, PipelineResult, ReviewPolicy, ReviewStats,
    SplitRetryLog, StructuralRecoveryStrategy, SubtitleDocument, SubtitleSegment,
    TerminologyEntity, TerminologyStats, TerminologyStrategy, TranslationLine, Usage,
};
use crate::error::{CoreError, CoreResult};
use crate::formatting::restore_batch_formatting;
use crate::languages::normalize_language_name;
use crate::memory::ContextMemory;
use crate::ports::{
    BackendJsonResult, BackendPayload, BatchExecutionOptions, BatchShardKind, CacheStage,
    ChatMessage, DashboardSink, GenerationRequest, LlmBackend, RuntimeStore,
};
use crate::progress::{ProgressEvent, ProgressSink, ProgressUnit, TaskKind, TaskState};
use crate::recovery::{
    backend_payload_json, build_agent_repair_messages, combine_glossary, parse_translation_payload,
    retry_correction_message, split_index,
};
use crate::review::{ReviewBatchPlan, build_review_messages, parse_review_payload};
use crate::storage::{InputSignature, ResumeSnapshot, build_glossary_fingerprint};
use crate::validation::{
    FinalValidationIssue, FinalValidationPolicy, final_validation_error, final_validation_issues,
    validate_final_output, validate_full_alignment, validate_translation_batch,
};

mod name_alignment;
mod online_terminology;
mod persistence;
mod planning;
mod review_runner;
mod review_stage;
mod support;
mod terminology;
mod translation_runner;
mod translation_stage;

use persistence::PipelinePersistence;
use planning::{BatchPlanner, DeduplicationPlan};
use review_runner::{ReviewBatchInput, ReviewRun};
pub use support::translation_memory_key;
use support::{
    build_translation_messages, contextual_translation_memory_keys, estimated_request_tokens,
    is_agent_repairable, is_operational_llm_failure, merge_review_patch, request_hash,
    translation_memory_scope, update_translation_memory, validate_review_candidate_ids,
};
use terminology::TerminologyStage;
#[cfg(test)]
use terminology::{
    TerminologyCandidate, extract_candidates as extract_terminology_candidates,
    parse_payload as parse_terminology_payload,
};
use translation_runner::{TranslationRun, TranslationRunInput};
use translation_stage::{PreparedBatch, TranslationPromptContext};

#[cfg(test)]
use crate::entities::ReviewReport;
#[cfg(test)]
use support::validate_window_terminology;

pub struct SubtitlePipeline<B, D> {
    backend: B,
    reviewer: Option<B>,
    dashboard: D,
    options: PipelineOptions,
    memory: ContextMemory,
    /// User glossary entries and accepted proper-name entities are enforced;
    /// automatically learned domain terms remain advisory.
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
    /// Turbo starts conservatively and uses additive increase / multiplicative
    /// decrease when the provider signals pressure.
    adaptive_translation_concurrency: usize,
    translation_window_was_rate_limited: bool,
    provider_requests: usize,
    provider_tokens: usize,
}

impl<B, D> SubtitlePipeline<B, D>
where
    B: LlmBackend,
    D: DashboardSink,
{
    pub fn new(backend: B, dashboard: D, mut options: PipelineOptions) -> Self {
        options.source_language = normalize_language_name(&options.source_language, true);
        options.target_language = normalize_language_name(&options.target_language, false);
        let adaptive_translation_concurrency = options.translation_concurrency.clamp(1, 2);
        Self {
            backend,
            reviewer: None,
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
            adaptive_translation_concurrency,
            translation_window_was_rate_limited: false,
            provider_requests: 0,
            provider_tokens: 0,
        }
    }

    pub fn with_progress(mut self, progress: Box<dyn ProgressSink>) -> Self {
        self.progress = Some(progress);
        self
    }

    pub fn with_reviewer(mut self, reviewer: B) -> Self {
        self.reviewer = Some(reviewer);
        self
    }

    fn execute_json(&mut self, messages: &[ChatMessage]) -> CoreResult<(serde_json::Value, Usage)> {
        self.reserve_requests(1)?;
        let result = self
            .backend
            .execute(
                GenerationRequest::json(messages.to_vec()).without_reasoning(),
                &self.cancellation,
            )
            .map_err(CoreError::from)?
            .into_json()
            .map_err(CoreError::from)?;
        self.provider_tokens = self.provider_tokens.saturating_add(result.1.total_tokens);
        Ok(result)
    }

    fn execute_review_json(
        &mut self,
        messages: &[ChatMessage],
    ) -> CoreResult<(serde_json::Value, Usage)> {
        self.reserve_requests(1)?;
        let result = self
            .reviewer
            .as_mut()
            .unwrap_or(&mut self.backend)
            .execute(
                GenerationRequest::json(messages.to_vec()).without_reasoning(),
                &self.cancellation,
            )
            .map_err(CoreError::from)?
            .into_json()
            .map_err(CoreError::from)?;
        self.provider_tokens = self.provider_tokens.saturating_add(result.1.total_tokens);
        Ok(result)
    }

    pub(super) fn review_backend_supports_parallel_generation(&self) -> bool {
        self.reviewer
            .as_ref()
            .unwrap_or(&self.backend)
            .supports_parallel_generation()
    }

    fn reserve_requests(&mut self, additional: usize) -> CoreResult<()> {
        if additional == 0 {
            return Ok(());
        }
        if let Some(limit) = self.options.max_requests
            && self.provider_requests.saturating_add(additional) > limit
        {
            return Err(CoreError::ResourceBudgetExceeded(format!(
                "request limit is {limit}; {} request(s) already used and {additional} more required",
                self.provider_requests
            )));
        }
        if let Some(limit) = self.options.max_tokens
            && self.provider_tokens >= limit
        {
            return Err(CoreError::ResourceBudgetExceeded(format!(
                "token limit is {limit}; {} token(s) already used",
                self.provider_tokens
            )));
        }
        self.provider_requests = self.provider_requests.saturating_add(additional);
        Ok(())
    }

    fn record_response_tokens(
        &mut self,
        responses: &[Result<crate::ports::GenerationResponse, crate::LlmCallError>],
    ) {
        for response in responses
            .iter()
            .filter_map(|response| response.as_ref().ok())
        {
            self.provider_tokens = self
                .provider_tokens
                .saturating_add(response.usage.total_tokens);
        }
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

    pub(super) fn effective_translation_concurrency(&self) -> usize {
        if matches!(
            self.options.policy().concurrency_strategy,
            ConcurrencyStrategy::AdaptiveQueued { .. }
        ) {
            self.adaptive_translation_concurrency
        } else {
            self.options.translation_concurrency.max(1)
        }
    }

    pub(super) fn note_translation_window_success(&mut self) {
        if matches!(
            self.options.policy().concurrency_strategy,
            ConcurrencyStrategy::AdaptiveQueued { .. }
        ) {
            if std::mem::take(&mut self.translation_window_was_rate_limited) {
                return;
            }
            self.adaptive_translation_concurrency = self
                .adaptive_translation_concurrency
                .saturating_add(1)
                .min(self.options.translation_concurrency.max(1));
        }
    }

    fn note_translation_rate_limit(&mut self) {
        if matches!(
            self.options.policy().concurrency_strategy,
            ConcurrencyStrategy::AdaptiveQueued { .. }
        ) {
            self.translation_window_was_rate_limited = true;
            self.adaptive_translation_concurrency = self
                .adaptive_translation_concurrency
                .saturating_div(2)
                .max(1);
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

        let policy = self.options.policy();
        let deduplication = DeduplicationPlan::new(&document.segments, policy.deduplicate);
        let batches = BatchPlanner::new(self.options.batch_size, self.options.batch_token_budget)
            .scene_aware(policy.context_strategy.uses_scene_boundaries())
            .split(deduplication.canonical());
        let source_contexts = BatchPlanner::source_contexts(
            deduplication.canonical(),
            &batches,
            policy.context_strategy,
            self.options.batch_token_budget,
        );
        let translation_scene_groups = BatchPlanner::scene_group_ids(&batches);
        let original_batches =
            BatchPlanner::new(self.options.batch_size, self.options.batch_token_budget)
                .scene_aware(policy.context_strategy.uses_scene_boundaries())
                .split(&document.segments);
        let review_scene_groups = BatchPlanner::scene_group_ids(&original_batches);
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
                    mode: self.options.mode,
                    deduplicated_segments: deduplication.duplicates(),
                    reviewer_fallback: self.options.review_policy != ReviewPolicy::Off
                        && self.reviewer.is_none(),
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
        self.sync_enforced_entities();
        // Freeze the effective preflight glossary once for this run. Later
        // batches may learn advisory terms, but all Resume writes must retain
        // the same semantic-input fingerprint.
        self.options.glossary_fingerprint = Some(build_glossary_fingerprint(&self.memory.glossary));
        let translation_memory_scope = translation_memory_scope(&self.options);
        let translation_memory_keys =
            contextual_translation_memory_keys(&translation_memory_scope, &document.segments);

        let resume = self.load_resume_snapshot(&batches)?;
        self.sync_enforced_entities();
        let mut translation_document = document.clone();
        translation_document.segments = deduplication.canonical().to_vec();
        let TranslationRun {
            batches,
            segments: translated_segments,
            usage,
        } = translation_runner::run(
            self,
            &translation_document,
            TranslationRunInput {
                batches,
                resume: &resume,
                terminology: &terminology,
                memory_keys: &translation_memory_keys,
                source_contexts: &source_contexts,
                scene_groups: &translation_scene_groups,
            },
        )?;
        let translated_segments = deduplication.expand(&document.segments, &translated_segments)?;
        let ReviewRun {
            mut output,
            stats: review,
            batches: review_batches,
            resumed: resumed_review_batches,
            mut usage,
        } = review_runner::run(
            self,
            document,
            ReviewBatchInput {
                review_batches: &original_batches,
                translation_batches: batches.len(),
                scene_groups: &review_scene_groups,
            },
            &translated_segments,
            &resume,
            &terminology,
            usage,
        )?;
        let final_validation_policy = FinalValidationPolicy {
            max_characters_per_second: self.options.max_characters_per_second,
            max_characters_per_line: self.options.max_characters_per_line,
            max_lines: self.options.max_lines,
        };
        if resume.validation_completed {
            if resume.finalized_segments.is_empty() {
                return Err(CoreError::DataInvariant(
                    "resume state marks final validation complete but has no finalized output"
                        .to_owned(),
                ));
            }
            validate_full_alignment(&document.segments, &resume.finalized_segments)?;
            output = resume.finalized_segments.clone();
        }
        let issues = final_validation_issues(
            &document.segments,
            &output,
            &self.required_glossary,
            &self.options.source_language,
            &self.options.target_language,
            final_validation_policy,
        )?;
        if !issues.is_empty() {
            self.repair_final_validation(
                document,
                &mut output,
                &issues,
                final_validation_policy,
                &mut usage,
            )?;
        }
        validate_final_output(
            &document.segments,
            &output,
            &self.required_glossary,
            &self.options.source_language,
            &self.options.target_language,
            final_validation_policy,
        )?;
        if self.options.use_cache {
            update_translation_memory(
                &mut self.translation_memory,
                &translation_memory_keys,
                &document.segments,
                &output,
            );
            if let Some(store) = self.store.as_ref() {
                self.cancellation.check()?;
                store.save_translation_memory(
                    &self
                        .translation_memory
                        .iter()
                        .map(|(key, text)| (key.clone(), text.clone()))
                        .collect::<Vec<_>>(),
                )?;
            }
        }
        if let Some(store) = self.store.as_ref() {
            self.cancellation.check()?;
            store.save_batch_segments(BatchShardKind::Finalized, 1, &output)?;
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
                mode: self.options.mode,
                deduplicated_segments: deduplication.duplicates(),
                reviewer_fallback: self.options.review_policy != ReviewPolicy::Off
                    && self.reviewer.is_none(),
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
            translated_segments: output,
        })
    }

    fn run_terminology_preflight(
        &mut self,
        document: &SubtitleDocument,
    ) -> CoreResult<TerminologyStats> {
        let backend = self.reviewer.as_mut().unwrap_or(&mut self.backend);
        TerminologyStage {
            backend,
            dashboard: &mut self.dashboard,
            options: &self.options,
            memory: &mut self.memory,
            store: self.store.as_deref(),
            cancellation: &self.cancellation,
            progress: self.progress.as_deref(),
            cache_hits: &mut self.cache_hits,
            provider_requests: &mut self.provider_requests,
            provider_tokens: &mut self.provider_tokens,
        }
        .run(document)
    }

    fn sync_enforced_entities(&mut self) {
        for entity in &self.memory.terminology_entities {
            if entity.kind.is_enforced() {
                for variant in &entity.variants {
                    self.required_glossary
                        .entry(variant.source.to_lowercase())
                        .or_insert_with(|| variant.target.clone());
                }
            }
        }
    }

    fn reconcile_translation_window(
        &mut self,
        prepared: &[PreparedBatch],
        generated: &mut HashMap<usize, BatchWithUsage>,
    ) -> CoreResult<()> {
        if lightweight_name_alignment(&self.options) {
            let candidates = self.memory.name_candidates.clone();
            let target_language = self.options.target_language.clone();
            for batch in prepared.iter().filter(|batch| !batch.pending.is_empty()) {
                let index = batch.index + 1;
                let result = generated.get_mut(&index).ok_or_else(|| {
                    CoreError::DataInvariant(format!("translation window omitted batch {index}"))
                })?;
                if let Err(error) = name_alignment::reconcile_batch(
                    &batch.pending,
                    result,
                    &mut self.memory,
                    &self.required_glossary,
                    &candidates,
                    &target_language,
                ) {
                    let mut retried = self.translate_batch_impl(
                        index,
                        &batch.pending,
                        &batch.prompt_context,
                        true,
                        Some(error),
                    )?;
                    name_alignment::reconcile_batch(
                        &batch.pending,
                        &mut retried,
                        &mut self.memory,
                        &self.required_glossary,
                        &candidates,
                        &target_language,
                    )
                    .map_err(|retry_error| {
                        CoreError::InvalidTranslation(format!(
                            "batch {index} name alignment failed after retry: {retry_error}"
                        ))
                    })?;
                    *result = retried;
                }
            }
            return Ok(());
        }
        if !self.options.online_terminology {
            return Ok(());
        }
        let mut canonical = self
            .memory
            .glossary
            .iter()
            .map(|(source, target)| (source.to_lowercase(), target.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut enforced = self
            .required_glossary
            .iter()
            .map(|(source, target)| (source.to_lowercase(), target.clone()))
            .collect::<BTreeMap<_, _>>();
        let candidates = self.memory.terminology_candidates.clone();

        for batch in prepared.iter().filter(|batch| !batch.pending.is_empty()) {
            let index = batch.index + 1;
            let result = generated.get_mut(&index).ok_or_else(|| {
                CoreError::DataInvariant(format!("translation window omitted batch {index}"))
            })?;
            let expected_terminology = !result.terminology_updates.is_empty();
            if let Err(error) = online_terminology::reconcile_batch(
                &batch.pending,
                result,
                &mut canonical,
                &mut enforced,
                &candidates,
                &self.options.target_language,
                self.options.preserve_names,
            ) {
                for (source, target) in &enforced {
                    self.required_glossary
                        .entry(source.clone())
                        .or_insert_with(|| target.clone());
                }
                let mut retried = self.translate_batch_impl(
                    index,
                    &batch.pending,
                    &batch.prompt_context,
                    true,
                    Some(error.clone()),
                )?;
                if expected_terminology && retried.terminology_updates.is_empty() {
                    return Err(CoreError::InvalidTranslation(format!(
                        "batch {index} omitted terminology updates after correction"
                    )));
                }
                online_terminology::reconcile_batch(
                    &batch.pending,
                    &mut retried,
                    &mut canonical,
                    &mut enforced,
                    &candidates,
                    &self.options.target_language,
                    self.options.preserve_names,
                )
                .map_err(|retry_error| {
                    CoreError::InvalidTranslation(format!(
                        "batch {index} terminology reconciliation failed after retry: {retry_error}"
                    ))
                })?;
                *result = retried;
            }
        }
        self.required_glossary.extend(enforced);
        Ok(())
    }

    fn commit_terminology_updates(&mut self, entities: &[TerminologyEntity]) {
        for entity in entities {
            self.memory.update("", &entity.variants);
            self.memory.add_terminology_entity(entity.clone());
            if entity.kind.is_enforced() {
                for variant in &entity.variants {
                    self.required_glossary
                        .entry(variant.source.to_lowercase())
                        .or_insert_with(|| variant.target.clone());
                }
            }
        }
    }

    fn save_reconciled_translation_cache(&self, result: &BatchWithUsage) -> CoreResult<()> {
        let (Some(key), Some(store)) = (&result.cache_key, self.store.as_ref()) else {
            return Ok(());
        };
        self.cancellation.check()?;
        store.save_cached_response(
            CacheStage::Translate,
            key,
            &BackendJsonResult {
                payload: BackendPayload::Translation(crate::entities::BatchTranslationResult {
                    lines: result.lines.clone(),
                    summary: result.summary.clone(),
                    glossary_updates: result.glossary_updates.clone(),
                    terminology_updates: result.terminology_updates.clone(),
                }),
                usage: result.usage,
            },
        )
    }

    fn translate_batch(
        &mut self,
        batch_index: usize,
        batch: &[SubtitleSegment],
        prompt_context: &TranslationPromptContext,
    ) -> CoreResult<BatchWithUsage> {
        self.translate_batch_impl(batch_index, batch, prompt_context, true, None)
    }

    fn translate_window(
        &mut self,
        batches: &[(usize, Vec<SubtitleSegment>, TranslationPromptContext)],
    ) -> CoreResult<HashMap<usize, BatchWithUsage>> {
        if !self.backend.supports_parallel_generation() {
            let mut results = HashMap::new();
            for (batch_index, batch, prompt_context) in batches {
                results.insert(
                    *batch_index,
                    self.translate_batch(*batch_index, batch, prompt_context)?,
                );
            }
            return Ok(results);
        }

        let mut results = HashMap::new();
        let mut pending = Vec::new();
        for (batch_index, batch, prompt_context) in batches {
            let messages = build_translation_messages(
                &self.options,
                *batch_index,
                batch,
                prompt_context,
                &self.memory,
                &self.required_glossary,
                self.options.policy().compact_wire && self.backend.supports_compact_translation(),
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
                let BackendPayload::Translation(mut payload) = response.payload else {
                    return Err(CoreError::DataInvariant(
                        "translation cache returned a review payload".to_owned(),
                    ));
                };
                prepare_translation_result(
                    lightweight_name_alignment(&self.options),
                    self.options.online_terminology,
                    marker_candidates(&self.options, &self.memory),
                    batch,
                    &mut payload,
                    true,
                )?;
                validate_translation_batch(batch, &payload.lines)?;
                self.cache_hits += 1;
                results.insert(
                    *batch_index,
                    BatchWithUsage {
                        lines: payload.lines,
                        summary: payload.summary,
                        glossary_updates: payload.glossary_updates,
                        terminology_updates: payload.terminology_updates,
                        usage: Usage::default(),
                        cache_key: None,
                    },
                );
            } else {
                let estimated = estimated_request_tokens(&messages, batch);
                if estimated > self.options.request_token_budget {
                    if batch.len() == 1 {
                        return Err(CoreError::ResourceBudgetExceeded(format!(
                            "translation batch {batch_index} needs about {estimated} request tokens, exceeding the per-request limit of {}",
                            self.options.request_token_budget
                        )));
                    }
                    let split = split_index(batch);
                    results.insert(
                        *batch_index,
                        self.translate_split(*batch_index, batch, prompt_context, split)?,
                    );
                    continue;
                }
                pending.push((
                    *batch_index,
                    batch.clone(),
                    prompt_context.clone(),
                    hash,
                    messages,
                ));
            }
        }
        let requests = pending
            .iter()
            .map(|(_, _, _, _, messages)| {
                GenerationRequest::json(messages.clone()).without_reasoning()
            })
            .collect();
        self.reserve_requests(pending.len())?;
        let responses = self
            .backend
            .execute_many(
                requests,
                BatchExecutionOptions::new(self.effective_translation_concurrency()),
                &self.cancellation,
            )
            .map_err(CoreError::from)?;
        self.record_response_tokens(&responses);
        if responses.len() != pending.len() {
            return Err(CoreError::InvalidBackendResponse(format!(
                "backend returned {} responses for {} translation requests",
                responses.len(),
                pending.len()
            )));
        }
        for ((batch_index, batch, prompt_context, hash, _), response) in
            pending.into_iter().zip(responses)
        {
            match response.map_err(CoreError::from).and_then(|response| {
                let (json, usage) = response.into_json().map_err(CoreError::from)?;
                let mut payload = parse_translation_payload(&json)?;
                prepare_translation_result(
                    lightweight_name_alignment(&self.options),
                    self.options.online_terminology,
                    marker_candidates(&self.options, &self.memory),
                    &batch,
                    &mut payload,
                    false,
                )?;
                validate_translation_batch(&batch, &payload.lines)?;
                Ok((payload, usage))
            }) {
                Ok((payload, response_usage)) => {
                    results.insert(
                        batch_index,
                        BatchWithUsage {
                            lines: payload.lines,
                            summary: payload.summary,
                            glossary_updates: payload.glossary_updates,
                            terminology_updates: payload.terminology_updates,
                            usage: response_usage,
                            cache_key: self.options.use_cache.then_some(hash),
                        },
                    );
                }
                Err(CoreError::Llm(crate::error::LlmCallError::RateLimited { .. })) => {
                    // The adapter has already exhausted its transport retries. Preserve the
                    // successful items from this window, reduce Turbo's real in-flight limit,
                    // and requeue only the throttled batch once through the normal single-call
                    // path. A second rate limit remains an outer failure.
                    self.note_translation_rate_limit();
                    results.insert(
                        batch_index,
                        self.translate_batch_impl(
                            batch_index,
                            &batch,
                            &prompt_context,
                            true,
                            None,
                        )?,
                    );
                }
                Err(error) => {
                    results.insert(
                        batch_index,
                        self.translate_batch_impl(
                            batch_index,
                            &batch,
                            &prompt_context,
                            true,
                            Some(error),
                        )?,
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
        prompt_context: &TranslationPromptContext,
        record_failure: bool,
        initial_error: Option<CoreError>,
    ) -> CoreResult<BatchWithUsage> {
        if let Some(error) = initial_error.as_ref()
            && is_operational_llm_failure(error)
        {
            if matches!(
                error,
                CoreError::Llm(crate::error::LlmCallError::RateLimited { .. })
            ) {
                self.note_translation_rate_limit();
            }
            return Err(error.clone());
        }
        let mut last_error = initial_error;
        let mut attempts = Vec::new();
        for attempt in 1..=self.options.retries + 1 {
            self.cancellation.check()?;
            let mut messages = build_translation_messages(
                &self.options,
                batch_index,
                batch,
                prompt_context,
                &self.memory,
                &self.required_glossary,
                self.options.policy().compact_wire && self.backend.supports_compact_translation(),
            );
            if let Some(error) = last_error.as_ref() {
                messages.push(retry_correction_message(error));
            }
            let estimated = estimated_request_tokens(&messages, batch);
            if estimated > self.options.request_token_budget {
                if batch.len() > 1 {
                    return self.translate_split(
                        batch_index,
                        batch,
                        prompt_context,
                        split_index(batch),
                    );
                }
                return Err(CoreError::ResourceBudgetExceeded(format!(
                    "translation batch {batch_index} needs about {estimated} request tokens, exceeding the per-request limit of {}",
                    self.options.request_token_budget
                )));
            }
            let request_hash = request_hash(&self.options, CacheStage::Translate, &messages);
            match self.translate_once(batch, &messages, &request_hash) {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if matches!(error, CoreError::Cancelled) {
                        return Err(error);
                    }
                    if is_operational_llm_failure(&error) {
                        if matches!(
                            error,
                            CoreError::Llm(crate::error::LlmCallError::RateLimited { .. })
                        ) {
                            self.note_translation_rate_limit();
                        }
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
                    let correct_before_split = matches!(
                        self.options.policy().structural_recovery_strategy,
                        StructuralRecoveryStrategy::CorrectBeforeSplit
                    ) && record_failure
                        && attempt == 1
                        && self.options.retries > 0;
                    if matches!(error, CoreError::InvalidTranslation(_))
                        && batch.len() > 1
                        && !correct_before_split
                    {
                        let split = split_index(batch);
                        attempt_log.split_retry = Some(SplitRetryLog {
                            triggered: true,
                            sizes: vec![split, batch.len() - split],
                            resolved: false,
                            error: None,
                        });
                        match self.translate_split(batch_index, batch, prompt_context, split) {
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
        Err(CoreError::DataInvariant(
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
        let mut backend_result = match cached_response {
            Some(response) => {
                self.cache_hits += 1;
                response
            }
            None => {
                let (json, usage) = self.execute_json(messages)?;
                BackendJsonResult {
                    payload: BackendPayload::Translation(parse_translation_payload(&json)?),
                    usage,
                }
            }
        };
        let BackendPayload::Translation(result) = &mut backend_result.payload else {
            return Err(CoreError::DataInvariant(
                "translation cache returned a review payload".to_owned(),
            ));
        };
        prepare_translation_result(
            lightweight_name_alignment(&self.options),
            self.options.online_terminology,
            marker_candidates(&self.options, &self.memory),
            batch,
            result,
            cached,
        )?;
        validate_translation_batch(batch, &result.lines)?;
        let BackendPayload::Translation(result) = backend_result.payload else {
            return Err(CoreError::DataInvariant(
                "translation backend returned a review payload".to_owned(),
            ));
        };
        Ok(BatchWithUsage {
            lines: result.lines,
            summary: result.summary,
            glossary_updates: result.glossary_updates,
            terminology_updates: result.terminology_updates,
            usage: if cached {
                Usage::default()
            } else {
                backend_result.usage
            },
            cache_key: (!cached && self.options.use_cache).then(|| request_hash.to_owned()),
        })
    }

    fn translate_split(
        &mut self,
        batch_index: usize,
        batch: &[SubtitleSegment],
        prompt_context: &TranslationPromptContext,
        split: usize,
    ) -> CoreResult<BatchWithUsage> {
        let left_context = prompt_context.for_left_split(&batch[split..]);
        let right_context = prompt_context.for_right_split(&batch[..split]);
        let left =
            self.translate_batch_impl(batch_index, &batch[..split], &left_context, false, None)?;
        let right =
            self.translate_batch_impl(batch_index, &batch[split..], &right_context, false, None)?;
        let mut usage = left.usage;
        usage.add(right.usage);
        Ok(BatchWithUsage {
            lines: left.lines.into_iter().chain(right.lines).collect(),
            summary: String::new(),
            glossary_updates: combine_glossary(left.glossary_updates, right.glossary_updates),
            terminology_updates: left
                .terminology_updates
                .into_iter()
                .chain(right.terminology_updates)
                .collect(),
            usage,
            cache_key: None,
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
        if let Some(error) = initial_error.as_ref()
            && is_operational_llm_failure(error)
        {
            return Err(error.clone());
        }
        let mut last_error = initial_error;
        let mut attempts = Vec::new();
        for attempt in 1..=self.options.retries + 1 {
            self.cancellation.check()?;
            let mut messages = build_review_messages(&self.options, batch, &self.memory);
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
                    if is_operational_llm_failure(&error) {
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
        Err(CoreError::DataInvariant(
            "review retry loop ended unexpectedly".to_owned(),
        ))
    }

    fn review_window(
        &mut self,
        batches: &[(usize, ReviewBatchPlan)],
    ) -> CoreResult<HashMap<usize, ReviewWithUsage>> {
        if !self.review_backend_supports_parallel_generation() {
            let mut output = HashMap::new();
            for (index, batch) in batches {
                output.insert(*index, self.review_batch(*index, batch)?);
            }
            return Ok(output);
        }
        let mut output = HashMap::new();
        let mut pending = Vec::new();
        for (index, batch) in batches {
            let messages = build_review_messages(&self.options, batch, &self.memory);
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
                let BackendPayload::Review(mut result) = response.payload else {
                    return Err(CoreError::DataInvariant(
                        "review cache returned a translation payload".to_owned(),
                    ));
                };
                restore_batch_formatting(&batch.source, &mut result.lines);
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
            .map(|(_, _, _, messages)| {
                GenerationRequest::json(messages.clone()).without_reasoning()
            })
            .collect();
        self.reserve_requests(pending.len())?;
        let responses = match self.reviewer.as_mut() {
            Some(reviewer) => reviewer.execute_many(
                requests,
                BatchExecutionOptions::new(self.options.review_concurrency),
                &self.cancellation,
            ),
            None => self.backend.execute_many(
                requests,
                BatchExecutionOptions::new(self.options.review_concurrency),
                &self.cancellation,
            ),
        }
        .map_err(CoreError::from)?;
        self.record_response_tokens(&responses);
        if responses.len() != pending.len() {
            return Err(CoreError::InvalidBackendResponse(format!(
                "backend returned {} responses for {} review requests",
                responses.len(),
                pending.len()
            )));
        }
        for ((index, batch, hash, _), response) in pending.into_iter().zip(responses) {
            match response.map_err(CoreError::from).and_then(|response| {
                let (json, usage) = response.into_json().map_err(CoreError::from)?;
                let mut result = parse_review_payload(&json)?;
                validate_review_candidate_ids(&batch, &result.lines)?;
                result.lines = merge_review_patch(&batch.translated, &result.lines)?;
                restore_batch_formatting(&batch.source, &mut result.lines);
                validate_translation_batch(&batch.source, &result.lines)?;
                Ok((result, usage))
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
        let mut backend_result = match cached_response {
            Some(response) => {
                self.cache_hits += 1;
                response
            }
            None => {
                let (payload, usage) = self.execute_review_json(messages)?;
                let mut review = parse_review_payload(&payload)?;
                validate_review_candidate_ids(batch, &review.lines)?;
                review.lines = merge_review_patch(&batch.translated, &review.lines)?;
                BackendJsonResult {
                    payload: BackendPayload::Review(review),
                    usage,
                }
            }
        };
        let BackendPayload::Review(result) = &mut backend_result.payload else {
            return Err(CoreError::DataInvariant(
                "review cache returned a translation payload".to_owned(),
            ));
        };
        restore_batch_formatting(&batch.source, &mut result.lines);
        validate_translation_batch(&batch.source, &result.lines)?;
        if self.options.use_cache
            && !cached
            && let Some(store) = self.store.as_ref()
        {
            self.cancellation.check()?;
            store.save_cached_response(CacheStage::Review, request_hash, &backend_result)?;
        }
        let BackendPayload::Review(result) = backend_result.payload else {
            return Err(CoreError::DataInvariant(
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
                terminology_updates: result.terminology_updates,
                usage: outcome.usage,
                cache_key: None,
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

    fn repair_final_validation(
        &mut self,
        document: &SubtitleDocument,
        output: &mut [SubtitleSegment],
        issues: &[FinalValidationIssue],
        policy: FinalValidationPolicy,
        usage: &mut Usage,
    ) -> CoreResult<()> {
        let initial_error = final_validation_error(issues);
        let failing_ids = issues
            .iter()
            .map(|issue| issue.segment_id.as_str())
            .collect::<std::collections::HashSet<_>>();
        let source = document
            .segments
            .iter()
            .filter(|segment| failing_ids.contains(segment.id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let translated = output
            .iter()
            .filter(|segment| failing_ids.contains(segment.id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let Some(repair) = self.run_agent_repair(
            "final_validation",
            0,
            &source,
            Some(&translated),
            &initial_error,
            &[],
        )?
        else {
            return Err(initial_error);
        };
        usage.add(repair.usage);
        let Some(BackendPayload::Review(result)) = repair.payload else {
            let detail = repair
                .error
                .unwrap_or_else(|| "no corrected payload was returned".to_owned());
            return Err(CoreError::InvalidTranslation(format!(
                "{initial_error}; targeted final-validation repair failed: {detail}"
            )));
        };
        for line in result.lines {
            let segment = output
                .iter_mut()
                .find(|segment| segment.id == line.id)
                .ok_or_else(|| {
                    CoreError::DataInvariant(format!(
                        "final-validation repair returned unknown id `{}`",
                        line.id
                    ))
                })?;
            segment.text = line.translation;
        }
        validate_final_output(
            &document.segments,
            output,
            &self.required_glossary,
            &self.options.source_language,
            &self.options.target_language,
            policy,
        )
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
                None => (if stage == "translate" {
                    self.execute_json(&messages)
                } else {
                    self.execute_review_json(&messages)
                })
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

            match response_result.and_then(|mut response| {
                let lines = match &mut response.payload {
                    BackendPayload::Translation(result) => {
                        prepare_translation_result(
                            lightweight_name_alignment(&self.options),
                            self.options.online_terminology,
                            marker_candidates(&self.options, &self.memory),
                            source,
                            result,
                            cached,
                        )?;
                        &result.lines
                    }
                    BackendPayload::Review(result) => {
                        restore_batch_formatting(source, &mut result.lines);
                        &result.lines
                    }
                    BackendPayload::Terminology(_) => {
                        return Err(CoreError::DataInvariant(
                            "repair cache returned a terminology payload".to_owned(),
                        ));
                    }
                };
                validate_translation_batch(source, lines)?;
                if stage == "final_validation" {
                    let repaired = source
                        .iter()
                        .map(|segment| {
                            let mut repaired = segment.clone();
                            repaired.text = lines
                                .iter()
                                .find(|line| line.id == segment.id)
                                .map(|line| line.translation.clone())
                                .unwrap_or_default();
                            repaired
                        })
                        .collect::<Vec<_>>();
                    validate_final_output(
                        source,
                        &repaired,
                        &self.required_glossary,
                        &self.options.source_language,
                        &self.options.target_language,
                        FinalValidationPolicy {
                            max_characters_per_second: self.options.max_characters_per_second,
                            max_characters_per_line: self.options.max_characters_per_line,
                            max_lines: self.options.max_lines,
                        },
                    )?;
                }
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
                    let stop_after_attempt = is_operational_llm_failure(&error);
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
                    if stop_after_attempt {
                        break;
                    }
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
        PipelinePersistence {
            options: &self.options,
            store: self.store.as_deref(),
            input_signature: self.input_signature.as_ref(),
        }
        .load_resume_snapshot(batches, &mut self.memory)
    }

    fn save_run_state(
        &self,
        translation_batches_completed: usize,
        review_batches_completed: usize,
        validation_completed: bool,
        usage: Usage,
    ) -> CoreResult<()> {
        PipelinePersistence {
            options: &self.options,
            store: self.store.as_deref(),
            input_signature: self.input_signature.as_ref(),
        }
        .save_run_state(
            &self.memory,
            translation_batches_completed,
            review_batches_completed,
            validation_completed,
            usage,
        )
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
    terminology_updates: Vec<TerminologyEntity>,
    usage: Usage,
    cache_key: Option<String>,
}

fn prepare_translation_result(
    lightweight_names: bool,
    online_terminology: bool,
    candidates: &[String],
    source: &[SubtitleSegment],
    result: &mut BatchTranslationResult,
    cached: bool,
) -> CoreResult<()> {
    // `summary` remains readable in legacy cache payloads, but translation
    // context is now derived from subtitle lines and confirmed translations.
    result.summary.clear();
    if lightweight_names {
        let markers = name_alignment::select_markers(source, candidates);
        name_alignment::validate_markers(source, &result.lines, &markers)?;
        if !cached {
            // The lightweight contract learns names only from inline markers.
            // Ignore unrequested side-channel fields from a raw model response.
            result.glossary_updates.clear();
        }
    }
    if online_terminology {
        let markers = online_terminology::select_markers(source, candidates);
        let extracted = online_terminology::extract_terms(&mut result.lines, &markers);
        result.glossary_updates =
            combine_glossary(std::mem::take(&mut result.glossary_updates), extracted);
    }
    restore_batch_formatting(source, &mut result.lines);
    Ok(())
}

fn lightweight_name_alignment(options: &PipelineOptions) -> bool {
    options.policy().terminology_strategy == TerminologyStrategy::LightweightNames
        && !options.online_terminology
        && !options.preserve_names
}

fn marker_candidates<'a>(options: &PipelineOptions, memory: &'a ContextMemory) -> &'a [String] {
    if lightweight_name_alignment(options) {
        &memory.name_candidates
    } else {
        &memory.terminology_candidates
    }
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
    log_path: Option<PathBuf>,
    error: Option<String>,
}

fn failure_error(
    stage: &str,
    batch_index: usize,
    error: &CoreError,
    failure_path: Option<&PathBuf>,
    repair: Option<&RepairOutcome>,
) -> CoreError {
    if matches!(
        error,
        CoreError::Cancelled | CoreError::ResourceBudgetExceeded(_)
    ) {
        return error.clone();
    }
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use crate::entities::{BatchTranslationResult, GlossaryEntry, TerminologyKind};
    use crate::error::LlmCallError;
    use crate::ports::{GenerationInput, GenerationResponse, NoopDashboard};
    use crate::review::build_review_plan;
    use crate::storage::{RunState, RuntimePaths, build_runtime_paths, input_signature_from_bytes};

    use super::*;

    fn request_messages(request: GenerationRequest) -> Result<Vec<ChatMessage>, LlmCallError> {
        match request.input {
            GenerationInput::Messages(messages) => Ok(messages),
            GenerationInput::Continue { .. } => Err(LlmCallError::UnsupportedCapability(
                "test continuation".to_owned(),
            )),
        }
    }

    struct EchoBackend;

    impl LlmBackend for EchoBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "echo"
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            cancellation.check().map_err(LlmCallError::from)?;
            let messages = request_messages(request)?;
            let prompt = messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let body = prompt
                .split("BATCH_JSON_START")
                .nth(1)
                .and_then(|value| value.split("BATCH_JSON_END").next())
                .ok_or_else(|| CoreError::DataInvariant("missing batch json".to_owned()))?;
            let parsed: serde_json::Value = serde_json::from_str(body)
                .map_err(|err| CoreError::DataInvariant(format!("invalid batch json: {err}")))?;
            let lines = parsed["lines"]
                .as_array()
                .ok_or_else(|| CoreError::DataInvariant("missing lines array".to_owned()))?
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
            let payload = serde_json::to_value(BatchTranslationResult {
                lines,
                summary: "ok".to_owned(),
                glossary_updates: Vec::<GlossaryEntry>::new(),
                terminology_updates: Vec::new(),
            })
            .map_err(|error| LlmCallError::InvalidResponse(error.to_string()))?;
            Ok(GenerationResponse::json(
                payload,
                Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    total_tokens: 2,
                    ..Usage::default()
                },
            ))
        }
    }

    struct NonThinkingBackend;

    impl LlmBackend for NonThinkingBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "non-thinking"
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            assert_eq!(request.reasoning, crate::ports::ReasoningPolicy::Disabled);
            EchoBackend.execute(request, cancellation)
        }
    }

    #[test]
    fn translation_requests_disable_provider_reasoning() {
        let mut options = PipelineOptions::new(PathBuf::from("reasoning.srt"));
        options.terminology_preflight = false;
        options.agent = false;
        let mut pipeline = SubtitlePipeline::new(NonThinkingBackend, NoopDashboard, options);

        pipeline
            .run_document(&document("reasoning.srt", &["hello"]))
            .expect("translate without provider reasoning");
    }

    #[test]
    fn request_budget_stops_before_starting_a_provider_side_effect() {
        let mut options = PipelineOptions::new(PathBuf::from("budget.srt"));
        options.terminology_preflight = false;
        options.agent = false;
        options.max_requests = Some(0);
        let mut pipeline = SubtitlePipeline::new(EchoBackend, NoopDashboard, options);
        let error = pipeline
            .run_document(&document("budget.srt", &["hello"]))
            .expect_err("zero request budget must stop before generation");
        assert!(matches!(error, CoreError::ResourceBudgetExceeded(_)));
    }

    struct CorrectedTerminologyBackend {
        calls: Arc<AtomicUsize>,
    }

    struct NameMarkerBackend {
        calls: Arc<AtomicUsize>,
    }

    impl LlmBackend for NameMarkerBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "name-markers"
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            _cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let messages = request_messages(request)?;
            let prompt = messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let body = prompt
                .split("BATCH_JSON_START")
                .nth(1)
                .and_then(|value| value.split("BATCH_JSON_END").next())
                .ok_or_else(|| CoreError::DataInvariant("missing batch json".to_owned()))?;
            let parsed: serde_json::Value = serde_json::from_str(body)
                .map_err(|error| CoreError::DataInvariant(error.to_string()))?;
            let id = parsed["lines"][0]["id"]
                .as_str()
                .ok_or_else(|| CoreError::DataInvariant("missing line id".to_owned()))?;
            let target = if call == 0 { "玛丽" } else { "玛莉" };
            Ok(GenerationResponse::json(
                serde_json::json!({
                    "lines": [{
                        "id": id,
                        "translation": format!("⟦N0⟧{target}⟦/N0⟧来了。"),
                    }]
                }),
                Usage::default(),
            ))
        }
    }

    impl LlmBackend for CorrectedTerminologyBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "terminology-correction"
        }

        fn execute(
            &mut self,
            _request: GenerationRequest,
            _cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(GenerationResponse::json(
                serde_json::json!({
                    "lines": [{"id": "1", "translation": "扎萨来了。"}],
                    "terminology_updates": [{
                        "canonical_source": "Joey Zasa",
                        "kind": "person",
                        "variants": [{"source": "Zasa", "target": "扎萨"}]
                    }]
                }),
                Usage::default(),
            ))
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

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            EchoBackend.execute(request, cancellation)
        }

        fn execute_many(
            &mut self,
            _requests: Vec<GenerationRequest>,
            _options: BatchExecutionOptions,
            _cancellation: &CancellationGuard,
        ) -> Result<Vec<Result<GenerationResponse, LlmCallError>>, LlmCallError> {
            Ok(Vec::new())
        }
    }

    struct AdaptiveParallelBackend {
        limits: Arc<Mutex<Vec<usize>>>,
        batch_calls: usize,
        single_retries: Arc<AtomicUsize>,
    }

    impl LlmBackend for AdaptiveParallelBackend {
        fn supports_parallel_generation(&self) -> bool {
            true
        }

        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "adaptive-parallel"
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            self.single_retries.fetch_add(1, Ordering::SeqCst);
            EchoBackend.execute(request, cancellation)
        }

        fn execute_many(
            &mut self,
            requests: Vec<GenerationRequest>,
            options: BatchExecutionOptions,
            cancellation: &CancellationGuard,
        ) -> Result<Vec<Result<GenerationResponse, LlmCallError>>, LlmCallError> {
            self.limits
                .lock()
                .map_err(|_| LlmCallError::Transport("limits lock poisoned".to_owned()))?
                .push(options.max_concurrency);
            self.batch_calls += 1;
            Ok(requests
                .into_iter()
                .enumerate()
                .map(|(index, request)| {
                    if self.batch_calls == 1 && index == 0 {
                        Err(LlmCallError::RateLimited {
                            message: "scripted pressure".to_owned(),
                            retry_after_ms: Some(1),
                        })
                    } else {
                        EchoBackend.execute(request, cancellation)
                    }
                })
                .collect())
        }
    }

    struct CountingBackend {
        calls: Arc<AtomicUsize>,
        fail_on_call: Option<usize>,
    }

    struct ContextCaptureBackend {
        contexts: Arc<Mutex<Vec<serde_json::Value>>>,
    }

    impl LlmBackend for ContextCaptureBackend {
        fn supports_parallel_generation(&self) -> bool {
            true
        }

        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "context-capture"
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            let messages = request_messages(request)?;
            self.contexts
                .lock()
                .map_err(|_| CoreError::DataInvariant("context lock poisoned".to_owned()))?
                .push(translation_context(&messages));
            EchoBackend.execute(GenerationRequest::json(messages), cancellation)
        }
    }

    impl LlmBackend for CountingBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "echo"
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_on_call == Some(call) {
                return Err(LlmCallError::Rejected {
                    status: None,
                    message: "scripted failure".to_owned(),
                });
            }
            EchoBackend.execute(request, cancellation)
        }
    }

    struct ReviewBackend {
        translation_calls: Arc<AtomicUsize>,
        review_calls: Arc<AtomicUsize>,
        fail_on_review_call: Option<usize>,
    }

    struct RoutedParallelBackend {
        label: &'static str,
        translation_calls: Arc<AtomicUsize>,
        review_calls: Arc<AtomicUsize>,
    }

    impl LlmBackend for RoutedParallelBackend {
        fn supports_parallel_generation(&self) -> bool {
            true
        }

        fn provider_name(&self) -> &str {
            self.label
        }

        fn model_name(&self) -> &str {
            self.label
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            let messages = request_messages(request)?;
            let prompt = messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            if !prompt.contains("TASK_START\nreview_translations\nTASK_END") {
                self.translation_calls.fetch_add(1, Ordering::SeqCst);
                return EchoBackend.execute(GenerationRequest::json(messages), cancellation);
            }

            self.review_calls.fetch_add(1, Ordering::SeqCst);
            let body = prompt
                .split("REVIEW_JSON_START")
                .nth(1)
                .and_then(|value| value.split("REVIEW_JSON_END").next())
                .ok_or_else(|| CoreError::DataInvariant("missing review json".to_owned()))?;
            let parsed: serde_json::Value = serde_json::from_str(body)
                .map_err(|error| CoreError::DataInvariant(error.to_string()))?;
            let changes = parsed["lines"]
                .as_array()
                .ok_or_else(|| CoreError::DataInvariant("missing review lines".to_owned()))?
                .iter()
                .map(|line| {
                    serde_json::json!({
                        "id": line["id"],
                        "translation": format!(
                            "[{} REVIEW] {}",
                            self.label,
                            line["translation"].as_str().unwrap_or_default()
                        ),
                    })
                })
                .collect::<Vec<_>>();
            Ok(GenerationResponse::json(
                serde_json::json!({"changes": changes}),
                Usage {
                    input_tokens: 2,
                    output_tokens: 3,
                    total_tokens: 5,
                    ..Usage::default()
                },
            ))
        }
    }

    impl LlmBackend for ReviewBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "echo"
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            let messages = request_messages(request)?;
            if !messages.iter().any(|message| {
                message
                    .content
                    .contains("TASK_START\nreview_translations\nTASK_END")
            }) {
                self.translation_calls.fetch_add(1, Ordering::SeqCst);
                return EchoBackend.execute(GenerationRequest::json(messages), cancellation);
            }
            let call = self.review_calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_on_review_call == Some(call) {
                return Err(LlmCallError::Rejected {
                    status: None,
                    message: "scripted review failure".to_owned(),
                });
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
                .ok_or_else(|| CoreError::DataInvariant("missing review json".to_owned()))?;
            let parsed: serde_json::Value = serde_json::from_str(body).map_err(|error| {
                CoreError::DataInvariant(format!("invalid review json: {error}"))
            })?;
            let lines = parsed["lines"]
                .as_array()
                .ok_or_else(|| CoreError::DataInvariant("missing review lines".to_owned()))?
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
            Ok(GenerationResponse::json(
                serde_json::json!({
                    "lines": lines,
                    "review_notes": "reviewed",
                }),
                Usage {
                    input_tokens: 1,
                    output_tokens: 2,
                    total_tokens: 3,
                    ..Usage::default()
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

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            let messages = request_messages(request)?;
            if messages.iter().any(|message| {
                message
                    .content
                    .contains("TASK_START\nreview_translations\nTASK_END")
            }) {
                return Ok(GenerationResponse::json(
                    serde_json::json!({"changes": []}),
                    Usage {
                        input_tokens: 5,
                        output_tokens: 1,
                        total_tokens: 6,
                        ..Usage::default()
                    },
                ));
            }
            EchoBackend.execute(GenerationRequest::json(messages), cancellation)
        }
    }

    struct GlossaryRegressionBackend;

    impl LlmBackend for GlossaryRegressionBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "glossary-regression"
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            _cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            let messages = request_messages(request)?;
            let prompt = messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let (start, end, translation, reviewing) =
                if prompt.contains("TASK_START\nreview_translations\nTASK_END") {
                    ("REVIEW_JSON_START", "REVIEW_JSON_END", "他来了。", true)
                } else {
                    ("BATCH_JSON_START", "BATCH_JSON_END", "勋爵来了。", false)
                };
            let body = prompt
                .split(start)
                .nth(1)
                .and_then(|value| value.split(end).next())
                .ok_or_else(|| CoreError::DataInvariant(format!("missing {start}")))?;
            let parsed: serde_json::Value = serde_json::from_str(body)
                .map_err(|error| CoreError::DataInvariant(error.to_string()))?;
            let lines = parsed["lines"]
                .as_array()
                .ok_or_else(|| CoreError::DataInvariant("missing lines".to_owned()))?
                .iter()
                .map(|line| {
                    serde_json::json!({
                        "id": line["id"],
                        "translation": translation,
                    })
                })
                .collect::<Vec<_>>();
            let payload = if reviewing {
                serde_json::json!({"changes": lines})
            } else {
                serde_json::json!({
                    "lines": lines,
                    "summary": "",
                    "glossary_updates": [],
                    "terminology_updates": [],
                })
            };
            Ok(GenerationResponse::json(payload, Usage::default()))
        }
    }

    struct PreflightBackend {
        contexts: Arc<Mutex<Vec<serde_json::Value>>>,
        preflight_requests: Arc<AtomicUsize>,
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

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            let messages = request_messages(request)?;
            let prompt = messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            if prompt.contains("TERMINOLOGY_JSON_START") {
                self.preflight_requests.fetch_add(1, Ordering::SeqCst);
                let body = prompt
                    .split("TERMINOLOGY_JSON_START")
                    .nth(1)
                    .and_then(|value| value.split("TERMINOLOGY_JSON_END").next())
                    .ok_or_else(|| {
                        CoreError::DataInvariant("missing terminology json".to_owned())
                    })?;
                let parsed: serde_json::Value = serde_json::from_str(body).map_err(|error| {
                    CoreError::DataInvariant(format!("invalid terminology: {error}"))
                })?;
                let candidates = parsed["candidates"].as_array().ok_or_else(|| {
                    CoreError::DataInvariant("missing terminology candidates".to_owned())
                })?;
                let entries = candidates
                    .iter()
                    .map(|candidate| {
                        serde_json::json!({
                            "source": candidate["source"],
                            "target": "统一译名",
                        })
                    })
                    .collect::<Vec<_>>();
                return Ok(GenerationResponse::json(
                    serde_json::json!({"entries": entries}),
                    Usage::default(),
                ));
            }
            cancellation.check().map_err(LlmCallError::from)?;
            let context = prompt
                .split("CONTEXT_JSON_START")
                .nth(1)
                .and_then(|value| value.split("CONTEXT_JSON_END").next())
                .ok_or_else(|| CoreError::DataInvariant("missing context json".to_owned()))?;
            let context: serde_json::Value = serde_json::from_str(context)
                .map_err(|error| CoreError::DataInvariant(format!("invalid context: {error}")))?;
            self.contexts
                .lock()
                .map_err(|_| CoreError::DataInvariant("context lock poisoned".to_owned()))?
                .push(context);
            let body = prompt
                .split("BATCH_JSON_START")
                .nth(1)
                .and_then(|value| value.split("BATCH_JSON_END").next())
                .ok_or_else(|| CoreError::DataInvariant("missing batch json".to_owned()))?;
            let parsed: serde_json::Value = serde_json::from_str(body)
                .map_err(|error| CoreError::DataInvariant(format!("invalid batch: {error}")))?;
            let batch_lines = parsed["lines"]
                .as_array()
                .ok_or_else(|| CoreError::DataInvariant("missing batch lines".to_owned()))?;
            let lines = batch_lines
                .iter()
                .map(|line| TranslationLine {
                    id: line["id"].as_str().unwrap_or_default().to_owned(),
                    translation: format!("统一译名 {}", line["text"].as_str().unwrap_or_default()),
                })
                .collect();
            let payload = serde_json::to_value(BatchTranslationResult {
                lines,
                summary: String::new(),
                glossary_updates: Vec::new(),
                terminology_updates: Vec::new(),
            })
            .map_err(|error| LlmCallError::InvalidResponse(error.to_string()))?;
            Ok(GenerationResponse::json(payload, Usage::default()))
        }
    }

    struct StructuralFailureBackend {
        call_sizes: Arc<Mutex<Vec<usize>>>,
    }

    struct BatchSizeCaptureBackend {
        call_sizes: Arc<Mutex<Vec<usize>>>,
    }

    impl LlmBackend for BatchSizeCaptureBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "batch-size-capture"
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            let response = EchoBackend.execute(request, cancellation)?;
            let (payload, usage) = response.into_json()?;
            let size = payload["lines"].as_array().map(Vec::len).ok_or_else(|| {
                LlmCallError::InvalidResponse("missing translation lines".to_owned())
            })?;
            self.call_sizes
                .lock()
                .map_err(|_| LlmCallError::Transport("call sizes lock poisoned".to_owned()))?
                .push(size);
            Ok(GenerationResponse::json(payload, usage))
        }
    }

    impl LlmBackend for StructuralFailureBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "structural"
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            let response = EchoBackend.execute(request, cancellation)?;
            let (mut payload, usage) = response.into_json()?;
            let result = payload["lines"].as_array_mut().ok_or_else(|| {
                LlmCallError::InvalidResponse("missing translation lines".to_owned())
            })?;
            self.call_sizes
                .lock()
                .expect("call sizes lock")
                .push(result.len());
            if result.len() > 1 {
                result.pop();
            } else {
                let translation = result[0]["translation"]
                    .as_str()
                    .unwrap_or_default()
                    .replacen("[ECHO]", "[SPLIT]", 1);
                result[0]["translation"] = serde_json::Value::String(translation);
            }
            Ok(GenerationResponse::json(payload, usage))
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

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            let messages = request_messages(request)?;
            let prompt = messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            if !prompt.contains("AGENT_REPAIR_JSON_START") {
                self.regular_calls.fetch_add(1, Ordering::SeqCst);
                let response =
                    EchoBackend.execute(GenerationRequest::json(messages), cancellation)?;
                let (mut payload, usage) = response.into_json()?;
                let lines = payload["lines"].as_array_mut().ok_or_else(|| {
                    LlmCallError::InvalidResponse("missing translation lines".to_owned())
                })?;
                for line in lines {
                    line["translation"] = serde_json::Value::String(String::new());
                }
                return Ok(GenerationResponse::json(payload, usage));
            }
            self.repair_calls.fetch_add(1, Ordering::SeqCst);
            let body = prompt
                .split("AGENT_REPAIR_JSON_START")
                .nth(1)
                .and_then(|value| value.split("AGENT_REPAIR_JSON_END").next())
                .ok_or_else(|| CoreError::DataInvariant("missing agent repair json".to_owned()))?;
            let payload: serde_json::Value = serde_json::from_str(body).map_err(|error| {
                CoreError::DataInvariant(format!("invalid repair json: {error}"))
            })?;
            let source_lines = payload["source_lines"].as_array().ok_or_else(|| {
                CoreError::DataInvariant("missing repair source lines".to_owned())
            })?;
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
            Ok(GenerationResponse::json(
                serde_json::json!({
                    "lines": lines,
                    "summary": "agent repaired",
                    "glossary_updates": [],
                }),
                Usage {
                    input_tokens: 3,
                    output_tokens: 4,
                    total_tokens: 7,
                    ..Usage::default()
                },
            ))
        }
    }

    struct AgentReviewBackend {
        review_calls: Arc<AtomicUsize>,
        repair_calls: Arc<AtomicUsize>,
    }

    struct FinalValidationRepairBackend {
        repair_ids: Arc<Mutex<Vec<Vec<String>>>>,
    }

    impl LlmBackend for FinalValidationRepairBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "final-validation-repair"
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            cancellation.check().map_err(LlmCallError::from)?;
            let messages = request_messages(request)?;
            let prompt = messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            if prompt.contains("AGENT_REPAIR_JSON_START") {
                let body = prompt
                    .split("AGENT_REPAIR_JSON_START")
                    .nth(1)
                    .and_then(|value| value.split("AGENT_REPAIR_JSON_END").next())
                    .ok_or_else(|| {
                        LlmCallError::InvalidResponse("missing final repair payload".to_owned())
                    })?;
                let payload: serde_json::Value = serde_json::from_str(body)
                    .map_err(|error| LlmCallError::InvalidResponse(error.to_string()))?;
                let ids = payload["expected_ids"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(serde_json::Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>();
                self.repair_ids
                    .lock()
                    .map_err(|_| LlmCallError::InvalidResponse("repair lock poisoned".to_owned()))?
                    .push(ids.clone());
                let lines = ids
                    .into_iter()
                    .map(|id| serde_json::json!({"id": id, "translation": "费用是12美元。"}))
                    .collect::<Vec<_>>();
                return Ok(GenerationResponse::json(
                    serde_json::json!({"lines": lines, "review_notes": "fixed numeric fact"}),
                    Usage {
                        input_tokens: 3,
                        output_tokens: 2,
                        total_tokens: 5,
                        requests: 1,
                        ..Usage::default()
                    },
                ));
            }

            let body = prompt
                .split("BATCH_JSON_START")
                .nth(1)
                .and_then(|value| value.split("BATCH_JSON_END").next())
                .ok_or_else(|| {
                    LlmCallError::InvalidResponse("missing translation batch".to_owned())
                })?;
            let payload: serde_json::Value = serde_json::from_str(body)
                .map_err(|error| LlmCallError::InvalidResponse(error.to_string()))?;
            let lines = payload["lines"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|line| {
                    let id = line["id"].as_str().unwrap_or_default();
                    serde_json::json!({
                        "id": id,
                        "translation": if id == "1" { "费用是13美元。" } else { "这里没有数字。" },
                    })
                })
                .collect::<Vec<_>>();
            Ok(GenerationResponse::json(
                serde_json::json!({"lines": lines}),
                Usage {
                    input_tokens: 10,
                    output_tokens: 4,
                    total_tokens: 14,
                    requests: 1,
                    ..Usage::default()
                },
            ))
        }
    }

    impl LlmBackend for AgentReviewBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "review-repair"
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            let messages = request_messages(request)?;
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
                    .ok_or_else(|| CoreError::DataInvariant("missing review json".to_owned()))?;
                let payload: serde_json::Value = serde_json::from_str(body).map_err(|error| {
                    CoreError::DataInvariant(format!("invalid review json: {error}"))
                })?;
                let lines = payload["lines"]
                    .as_array()
                    .ok_or_else(|| CoreError::DataInvariant("missing review lines".to_owned()))?
                    .iter()
                    .map(|line| serde_json::json!({"id": line["id"], "translation": ""}))
                    .collect::<Vec<_>>();
                return Ok(GenerationResponse::json(
                    serde_json::json!({"lines": lines, "review_notes": "broken"}),
                    Usage::default(),
                ));
            }
            if !prompt.contains("AGENT_REPAIR_JSON_START") {
                return EchoBackend.execute(GenerationRequest::json(messages), cancellation);
            }

            self.repair_calls.fetch_add(1, Ordering::SeqCst);
            let body = prompt
                .split("AGENT_REPAIR_JSON_START")
                .nth(1)
                .and_then(|value| value.split("AGENT_REPAIR_JSON_END").next())
                .ok_or_else(|| CoreError::DataInvariant("missing review repair json".to_owned()))?;
            let payload: serde_json::Value = serde_json::from_str(body).map_err(|error| {
                CoreError::DataInvariant(format!("invalid review repair json: {error}"))
            })?;
            let current = payload["current_translations"].as_array().ok_or_else(|| {
                CoreError::DataInvariant("missing current translations".to_owned())
            })?;
            let lines = current
                .iter()
                .map(|line| {
                    serde_json::json!({
                        "id": line["id"],
                        "translation": line["translation"],
                    })
                })
                .collect::<Vec<_>>();
            Ok(GenerationResponse::json(
                serde_json::json!({"lines": lines, "review_notes": "agent repaired"}),
                Usage {
                    input_tokens: 2,
                    output_tokens: 2,
                    total_tokens: 4,
                    ..Usage::default()
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
    fn turbo_rate_limit_requeues_only_failed_batch_and_reduces_real_concurrency() {
        let limits = Arc::new(Mutex::new(Vec::new()));
        let single_retries = Arc::new(AtomicUsize::new(0));
        let mut options = PipelineOptions::new("adaptive.txt".into());
        options.batch_size = 1;
        options.translation_concurrency = 8;
        options.resume = false;
        options.use_cache = false;
        let mut pipeline = SubtitlePipeline::new(
            AdaptiveParallelBackend {
                limits: Arc::clone(&limits),
                batch_calls: 0,
                single_retries: Arc::clone(&single_retries),
            },
            NoopDashboard,
            options,
        );

        let run = pipeline
            .run_document(&document(
                "adaptive.txt",
                &["one", "two", "three", "four", "five"],
            ))
            .expect("adaptive retry succeeds");

        assert_eq!(run.translated_segments.len(), 5);
        assert_eq!(single_retries.load(Ordering::SeqCst), 1);
        assert_eq!(*limits.lock().expect("limits lock"), vec![2, 1]);
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
                semantic: Default::default(),
            }],
            header: None,
            passthrough_blocks: Vec::new(),
            metadata: Default::default(),
        };
        let mut options = PipelineOptions::new("clip.txt".into());
        options.batch_size = 1;
        let mut pipeline = SubtitlePipeline::new(EchoBackend, NoopDashboard, options);
        let run = pipeline.run_document(&document).expect("run");

        assert_eq!(run.result.batches_translated, 1);
        assert_eq!(run.translated_segments[0].text, "[ECHO] hello");
    }

    #[test]
    fn turbo_pipeline_reconciles_lightweight_names_in_subtitle_order() {
        let mut options = PipelineOptions::new("names.txt".into());
        options.batch_size = 1;
        options.translation_concurrency = 2;
        options.resume = false;
        options.use_cache = false;
        let calls = Arc::new(AtomicUsize::new(0));
        let mut pipeline = SubtitlePipeline::new(
            NameMarkerBackend {
                calls: Arc::clone(&calls),
            },
            NoopDashboard,
            options,
        );

        let run = pipeline
            .run_document(&document("names.txt", &["Mary arrived.", "Mary returned."]))
            .expect("translate names");

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(run.translated_segments[0].text, "玛丽来了。");
        assert_eq!(run.translated_segments[1].text, "玛丽来了。");
        assert_eq!(pipeline.memory.name_translations["mary"], "玛丽");
        assert_eq!(pipeline.memory.glossary["Mary"], "玛丽");
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
                semantic: Default::default(),
            }],
            header: None,
            passthrough_blocks: Vec::new(),
            metadata: Default::default(),
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
        assert_eq!(data.saved_translation_memory.len(), 1);
        assert!(data.saved_translation_memory[0].0.starts_with("ctx-v4:"));
        assert_eq!(data.saved_translation_memory[0].1, "[ECHO] hello");
        assert_eq!(data.saved_batches.len(), 1);
        assert_eq!(data.saved_batches[0].1[0].text, "[ECHO] hello");
    }

    #[test]
    fn pipeline_run_state_fingerprints_the_frozen_glossary() {
        let document = document("resume-glossary.txt", &["hello"]);
        let mut options = PipelineOptions::new("resume-glossary.txt".into());
        options.terminology_preflight = false;
        let signature = input_signature_from_bytes(b"hello\n", Some(123));
        let loaded_glossary = vec![("Lord".to_owned(), "勋爵".to_owned())];
        let captured = Arc::new(Mutex::new(CapturedStoreData {
            loaded_glossary: loaded_glossary.clone(),
            ..CapturedStoreData::default()
        }));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("resume-glossary.txt"),
                std::path::Path::new("/workspace/resume-glossary.txt"),
                None,
                None,
                "Auto",
                "Chinese",
                false,
            ),
            data: Arc::clone(&captured),
        };
        let mut pipeline = SubtitlePipeline::new(EchoBackend, NoopDashboard, options.clone())
            .with_store(Box::new(store))
            .with_input_signature(signature.clone());

        pipeline.run_document(&document).expect("run");

        let state = captured
            .lock()
            .expect("capture lock")
            .saved_state
            .clone()
            .expect("saved run state");
        assert!(state.validation_completed);
        assert_eq!(
            captured
                .lock()
                .expect("capture lock")
                .saved_finalized_batches
                .len(),
            1
        );
        let mut expected_options = options;
        expected_options.glossary_fingerprint = Some(build_glossary_fingerprint(
            &loaded_glossary.into_iter().collect(),
        ));
        let expected = crate::storage::build_translation_fingerprint(&expected_options, &signature);
        assert_eq!(state.translation_fingerprint, expected);

        expected_options.glossary_fingerprint = Some(build_glossary_fingerprint(
            &[("Lord".to_owned(), "领主".to_owned())]
                .into_iter()
                .collect(),
        ));
        let changed = crate::storage::build_translation_fingerprint(&expected_options, &signature);
        assert!(state.resume_snapshot(&changed).is_none());
    }

    #[test]
    fn contextual_translation_memory_is_scoped_and_ignores_legacy_keys() {
        let source = document("/shows/one/clip.txt", &["Fine."]);
        let mut same_scope_options = PipelineOptions::new("/shows/one/clip.txt".into());
        same_scope_options.resume = false;
        same_scope_options.glossary_fingerprint =
            Some(build_glossary_fingerprint(&BTreeMap::new()));
        let key = contextual_translation_memory_keys(
            &translation_memory_scope(&same_scope_options),
            &source.segments,
        )["1"]
            .clone();
        let captured = Arc::new(Mutex::new(CapturedStoreData {
            loaded_translation_memory: vec![
                (key, "好的。".to_owned()),
                (translation_memory_key("Fine."), "罚款。".to_owned()),
            ],
            ..CapturedStoreData::default()
        }));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("/shows/one/clip.txt"),
                std::path::Path::new("/shows/one/clip.txt"),
                None,
                None,
                "Auto",
                "Chinese",
                false,
            ),
            data: Arc::clone(&captured),
        };
        let same_scope_calls = Arc::new(AtomicUsize::new(0));
        let mut same_scope = SubtitlePipeline::new(
            CountingBackend {
                calls: Arc::clone(&same_scope_calls),
                fail_on_call: Some(1),
            },
            NoopDashboard,
            same_scope_options,
        )
        .with_store(Box::new(store.clone()));

        let reused = same_scope.run_document(&source).expect("contextual TM hit");

        assert_eq!(same_scope_calls.load(Ordering::SeqCst), 0);
        assert_eq!(reused.translated_segments[0].text, "好的。");
        assert_eq!(reused.result.translation_memory_hits, 1);

        let other_source = document("/shows/two/clip.txt", &["Fine."]);
        let mut other_scope_options = PipelineOptions::new("/shows/two/clip.txt".into());
        other_scope_options.resume = false;
        let other_scope_calls = Arc::new(AtomicUsize::new(0));
        let mut other_scope = SubtitlePipeline::new(
            CountingBackend {
                calls: Arc::clone(&other_scope_calls),
                fail_on_call: None,
            },
            NoopDashboard,
            other_scope_options,
        )
        .with_store(Box::new(store));

        let translated = other_scope
            .run_document(&other_source)
            .expect("different scope translates");

        assert_eq!(other_scope_calls.load(Ordering::SeqCst), 1);
        assert_eq!(translated.translated_segments[0].text, "[ECHO] Fine.");
        assert_eq!(translated.result.translation_memory_hits, 0);
    }

    #[test]
    fn changed_required_glossary_invalidates_translation_memory_before_lookup() {
        let source = document("/shows/one/terms.txt", &["The Lord is here."]);
        let mut options = PipelineOptions::new("/shows/one/terms.txt".into());
        options.glossary_path = Some("glossary.json".into());
        options.resume = false;
        options.agent = false;
        options.glossary_fingerprint = Some(build_glossary_fingerprint(&BTreeMap::from([(
            "Lord".to_owned(),
            "领主".to_owned(),
        )])));
        let key = contextual_translation_memory_keys(
            &translation_memory_scope(&options),
            &source.segments,
        )["1"]
            .clone();
        let captured = Arc::new(Mutex::new(CapturedStoreData {
            loaded_translation_memory: vec![(key, "领主来了。".to_owned())],
            loaded_glossary: vec![("Lord".to_owned(), "勋爵".to_owned())],
            ..CapturedStoreData::default()
        }));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("/shows/one/terms.txt"),
                std::path::Path::new("/shows/one/terms.txt"),
                None,
                None,
                "Auto",
                "Chinese",
                false,
            ),
            data: captured,
        };
        let mut pipeline = SubtitlePipeline::new(GlossaryRegressionBackend, NoopDashboard, options)
            .with_store(Box::new(store));

        let run = pipeline
            .run_document(&source)
            .expect("changed glossary retranslates instead of using stale memory");

        assert_eq!(run.result.translation_memory_hits, 0);
        assert_eq!(run.translated_segments[0].text, "勋爵来了。");
    }

    #[test]
    fn final_validation_checks_required_glossary_after_review() {
        let source = document("review-terms.txt", &["The Lord is here."]);
        let captured = Arc::new(Mutex::new(CapturedStoreData {
            loaded_glossary: vec![("Lord".to_owned(), "勋爵".to_owned())],
            ..CapturedStoreData::default()
        }));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("review-terms.txt"),
                std::path::Path::new("review-terms.txt"),
                None,
                None,
                "Auto",
                "Chinese",
                false,
            ),
            data: Arc::clone(&captured),
        };
        let mut options = PipelineOptions::new("review-terms.txt".into());
        options.glossary_path = Some("glossary.json".into());
        options.review_policy = ReviewPolicy::Full;
        options.terminology_preflight = false;
        options.resume = false;
        options.use_cache = false;
        let mut pipeline = SubtitlePipeline::new(GlossaryRegressionBackend, NoopDashboard, options)
            .with_store(Box::new(store))
            .with_input_signature(input_signature_from_bytes(b"The Lord is here.\n", Some(1)));

        let error = pipeline
            .run_document(&source)
            .expect_err("reviewer must not remove a required term");

        assert!(error.to_string().contains("final output validation failed"));
        assert!(error.to_string().contains("`Lord` -> `勋爵`"));
        let data = captured.lock().expect("capture lock");
        assert!(
            !data
                .saved_state
                .as_ref()
                .expect("pre-validation state")
                .validation_completed
        );
        assert!(data.saved_finalized_batches.is_empty());
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
                    semantic: Default::default(),
                })
                .collect(),
            header: None,
            passthrough_blocks: Vec::new(),
            metadata: Default::default(),
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
                semantic: Default::default(),
            }],
            header: None,
            passthrough_blocks: Vec::new(),
            metadata: Default::default(),
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
    fn pipeline_does_not_duplicate_adapter_level_llm_retries() {
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

        let error = pipeline
            .run_document(&document)
            .expect_err("operational LLM errors belong to the adapter retry policy");

        assert!(matches!(
            error,
            CoreError::Llm(crate::error::LlmCallError::Rejected { .. })
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
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
    fn economy_retries_one_corrected_large_batch_then_splits_structural_failure() {
        let document = document("economy-split.txt", &["one", "two", "three", "four"]);
        let mut options = PipelineOptions::new("economy-split.txt".into());
        options.mode = crate::entities::TranslationMode::Economy;
        options.batch_size = 8;
        options.batch_token_budget = 10_000;
        options.review_policy = ReviewPolicy::Off;
        options.retries = 1;
        options.agent = false;
        let call_sizes = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = SubtitlePipeline::new(
            StructuralFailureBackend {
                call_sizes: Arc::clone(&call_sizes),
            },
            NoopDashboard,
            options,
        );

        let run = pipeline
            .run_document(&document)
            .expect("economy split succeeds");

        assert_eq!(
            *call_sizes.lock().expect("call sizes lock"),
            vec![4, 4, 2, 1, 1, 2, 1, 1]
        );
        assert_eq!(run.translated_segments.len(), 4);
    }

    #[test]
    fn complete_request_budget_splits_before_provider_side_effect() {
        let source = [
            "a".repeat(400),
            "b".repeat(400),
            "c".repeat(400),
            "d".repeat(400),
        ];
        let source = source.iter().map(String::as_str).collect::<Vec<_>>();
        let document = document("request-budget.txt", &source);
        let mut options = PipelineOptions::new("request-budget.txt".into());
        options.mode = crate::entities::TranslationMode::Economy;
        options.batch_size = 8;
        options.batch_token_budget = 10_000;
        options.resume = false;
        options.use_cache = false;
        let full_messages = build_translation_messages(
            &options,
            1,
            &document.segments,
            &TranslationPromptContext::default(),
            &ContextMemory::default(),
            &BTreeMap::new(),
            false,
        );
        let one_messages = build_translation_messages(
            &options,
            1,
            &document.segments[..1],
            &TranslationPromptContext::default(),
            &ContextMemory::default(),
            &BTreeMap::new(),
            false,
        );
        let full_estimate = estimated_request_tokens(&full_messages, &document.segments);
        let one_estimate = estimated_request_tokens(&one_messages, &document.segments[..1]);
        assert!(full_estimate > one_estimate);
        options.request_token_budget = one_estimate.saturating_add(32);
        let call_sizes = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = SubtitlePipeline::new(
            BatchSizeCaptureBackend {
                call_sizes: Arc::clone(&call_sizes),
            },
            NoopDashboard,
            options,
        );

        let run = pipeline
            .run_document(&document)
            .expect("oversized request is split");

        let sizes = call_sizes.lock().expect("call sizes lock");
        assert!(!sizes.contains(&4));
        assert_eq!(sizes.iter().sum::<usize>(), 4);
        assert_eq!(run.translated_segments.len(), 4);
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
    fn final_validation_repairs_only_failing_segments() {
        let document = document(
            "final-repair.srt",
            &["The repair costs 12 dollars.", "There is no number here."],
        );
        let mut options = PipelineOptions::new("final-repair.srt".into());
        options.batch_size = 8;
        options.review_policy = ReviewPolicy::Off;
        options.terminology_preflight = false;
        options.online_terminology = false;
        let signature = input_signature_from_bytes(b"final repair\n", Some(7));
        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let paths = build_runtime_paths(
            std::path::Path::new("final-repair.srt"),
            std::path::Path::new("/workspace/final-repair.srt"),
            None,
            None,
            "Auto",
            "Chinese",
            false,
        );
        let repair_ids = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = SubtitlePipeline::new(
            FinalValidationRepairBackend {
                repair_ids: Arc::clone(&repair_ids),
            },
            NoopDashboard,
            options.clone(),
        )
        .with_store(Box::new(CapturedStore {
            paths: paths.clone(),
            data: Arc::clone(&captured),
        }))
        .with_input_signature(signature.clone());

        let run = pipeline
            .run_document(&document)
            .expect("targeted final validation repair");

        assert_eq!(
            *repair_ids.lock().expect("repair ids"),
            vec![vec!["1".to_owned()]]
        );
        assert_eq!(run.translated_segments[0].text, "费用是12美元。");
        assert_eq!(run.translated_segments[1].text, "这里没有数字。");
        assert_eq!(run.result.agent_repairs.len(), 1);
        assert_eq!(run.result.agent_repairs[0].stage, "final_validation");
        assert!(run.result.agent_repairs[0].success);
        assert_eq!(run.result.usage.total_tokens, 19);

        let data = captured.lock().expect("capture lock");
        assert!(
            data.saved_state
                .as_ref()
                .expect("state")
                .validation_completed
        );
        assert_eq!(data.saved_finalized_batches.len(), 1);
        drop(data);

        let resumed_repair_ids = Arc::new(Mutex::new(Vec::new()));
        let mut resumed = SubtitlePipeline::new(
            FinalValidationRepairBackend {
                repair_ids: Arc::clone(&resumed_repair_ids),
            },
            NoopDashboard,
            options,
        )
        .with_store(Box::new(CapturedStore {
            paths,
            data: Arc::clone(&captured),
        }))
        .with_input_signature(signature);
        let resumed_run = resumed
            .run_document(&document)
            .expect("resume finalized output");

        assert!(resumed_repair_ids.lock().expect("repair ids").is_empty());
        assert_eq!(resumed_run.translated_segments, run.translated_segments);
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
    fn targeted_review_uses_term_boundaries_and_inflections() {
        let batches = vec![vec![segment("1", "The actors left the theater.")]];
        let mut memory = ContextMemory::new();
        memory.glossary.extend([
            ("actor".to_owned(), "演员".to_owned()),
            ("he".to_owned(), "他".to_owned()),
        ]);

        let valid = build_review_plan(
            &batches,
            &[segment("1", "演员离开了剧院。")],
            &memory,
            "en",
            "zh-Hans",
        );
        assert!(valid.is_empty());

        let missing_inflected_term = build_review_plan(
            &batches,
            &[segment("1", "人们离开了剧院。")],
            &memory,
            "en",
            "zh-Hans",
        );
        assert_eq!(missing_inflected_term[0].reasons, vec!["glossary mismatch"]);
    }

    #[test]
    fn terminology_payload_accepts_only_known_nonempty_candidates() {
        let candidates = vec![TerminologyCandidate {
            source: "Axe Gang".to_owned(),
            context: "The Axe Gang is here.".to_owned(),
            align_as_name: false,
        }];
        let parsed = parse_terminology_payload(
            &serde_json::json!({
                "entries": [{"source": "Axe Gang", "target": "斧头帮"}]
            }),
            &candidates,
            &[segment("1", "The Axe Gang is here.")],
        )
        .expect("valid terminology");
        assert_eq!(parsed.entries[0].target, "斧头帮");

        let alias_candidates = vec![
            TerminologyCandidate {
                source: "Joey Zasa".to_owned(),
                context: "Joey Zasa arrived.".to_owned(),
                align_as_name: true,
            },
            TerminologyCandidate {
                source: "Joey".to_owned(),
                context: "Joey!".to_owned(),
                align_as_name: true,
            },
            TerminologyCandidate {
                source: "Zasa".to_owned(),
                context: "Zasa sent them.".to_owned(),
                align_as_name: true,
            },
        ];
        let aliases = parse_terminology_payload(
            &serde_json::json!({
                "entities": [{
                    "canonical_source": "Joey Zasa",
                    "kind": "person",
                    "variants": [
                        {"source": "Joey Zasa", "target": "乔伊·扎萨"},
                        {"source": "Joey", "target": "乔伊"},
                        {"source": "Zasa", "target": "扎萨"}
                    ]
                }]
            }),
            &alias_candidates,
            &[
                segment("1", "Joey Zasa arrived."),
                segment("2", "Joey left."),
                segment("3", "Zasa sent them."),
            ],
        )
        .expect("entity aliases");
        assert_eq!(aliases.entities[0].variants[1].target, "乔伊");
        assert_eq!(aliases.entities[0].variants[2].target, "扎萨");

        let error = parse_terminology_payload(
            &serde_json::json!({
                "entries": [{"source": "Unknown", "target": "未知"}]
            }),
            &candidates,
            &[segment("1", "The Axe Gang is here.")],
        )
        .expect_err("unknown source rejected");
        assert!(error.to_string().contains("unknown source"));
    }

    #[test]
    fn terminology_payload_accepts_exact_spans_from_non_latin_document_samples() {
        let segments = vec![
            segment("1", "量子航行系统已经启动。"),
            segment("2", "量子航行系统发生故障。"),
        ];
        let parsed = parse_terminology_payload(
            &serde_json::json!({
                "entries": [{"source": "量子航行系统", "target": "quantum navigation system"}]
            }),
            &[],
            &segments,
        )
        .expect("document sample term");

        assert_eq!(parsed.entries[0].source, "量子航行系统");
    }

    #[test]
    fn terminology_candidates_support_unicode_title_case() {
        let candidates = extract_terminology_candidates(&[
            segment("1", "Алиса вернулась."),
            segment("2", "Алиса ждала."),
        ]);

        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.source == "Алиса")
        );
    }

    #[test]
    fn terminology_candidates_normalize_english_possessives() {
        let segments = vec![
            segment("18", "MacAndrews'."),
            segment("18b", "MacAndrews returned."),
            segment("19", "MacClannough's horse."),
            segment("19b", "MacClannough waited."),
            segment("20", "James’ horse."),
            segment("20b", "James agreed."),
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
    fn terminology_candidates_include_japanese_names_before_honorifics() {
        let segments = vec![
            segment("1", "ヒムロ君がいない?"),
            segment("2", "待ってトキタ先生"),
            segment("3", "ただの文章です"),
        ];

        let candidates = extract_terminology_candidates(&segments)
            .into_iter()
            .map(|candidate| candidate.source)
            .collect::<Vec<_>>();

        assert!(candidates.contains(&"ヒムロ".to_owned()));
        assert!(candidates.contains(&"トキタ".to_owned()));
        assert!(!candidates.contains(&"文章".to_owned()));
    }

    #[test]
    fn lightweight_name_candidates_require_recurrence_or_japanese_honorific() {
        let candidates = extract_terminology_candidates(&[
            segment("1", "Meet Alice now."),
            segment("2", "Mary arrived."),
            segment("3", "Mary left."),
            segment("4", "トキタ君を待って"),
        ]);
        let aligned = candidates
            .into_iter()
            .filter(|candidate| candidate.align_as_name)
            .map(|candidate| candidate.source)
            .collect::<Vec<_>>();

        assert!(aligned.contains(&"Mary".to_owned()));
        assert!(aligned.contains(&"トキタ".to_owned()));
        assert!(!aligned.contains(&"Meet Alice".to_owned()));
    }

    #[test]
    fn terminology_candidate_limit_prioritizes_recurring_names() {
        let mut segments = (0..300)
            .map(|index| segment(&index.to_string(), &format!("Candidate{index} arrived.")))
            .collect::<Vec<_>>();
        segments.push(segment("301", "Zasa arrived."));
        segments.push(segment("302", "Zasa returned."));

        let sources = extract_terminology_candidates(&segments)
            .into_iter()
            .map(|candidate| candidate.source)
            .collect::<Vec<_>>();

        assert_eq!(sources, vec!["Zasa".to_owned()]);
    }

    #[test]
    fn terminology_candidates_ignore_sentence_initial_watermark_phrases() {
        let sources = extract_terminology_candidates(&[
            segment("1", "Downloaded from YTS.BZ"),
            segment("2", "Official YIFY movies site"),
        ])
        .into_iter()
        .map(|candidate| candidate.source)
        .collect::<Vec<_>>();

        assert!(!sources.contains(&"Official YIFY".to_owned()));
    }

    #[test]
    fn terminology_candidates_reject_recurring_dialogue_starters_and_contractions() {
        let candidates = extract_terminology_candidates(&[
            segment("1", "I'm ready."),
            segment("2", "I'm listening."),
            segment("3", "You're late."),
            segment("4", "You're impossible."),
            segment("5", "Can't you see?"),
            segment("6", "Can't we leave?"),
            segment("7", "Please wait."),
            segment("8", "Please stop."),
            segment("9", "Clark is here."),
            segment("10", "Where is Clark?"),
            segment("11", "All right."),
            segment("12", "All done."),
            segment("13", "Big mistake."),
            segment("14", "Big deal."),
            segment("15", "Yeah, sure."),
            segment("16", "Yeah, okay."),
            segment("17", "Y-you heard me."),
            segment("18", "Y-you know that."),
            segment("19", "I-I heard you."),
            segment("20", "I-I know that."),
            segment("21", "Ow, stop."),
            segment("22", "Ow, that hurts."),
            segment("23", "Dad, wait."),
            segment("24", "Dad, listen."),
        ]);
        let sources = candidates
            .iter()
            .map(|candidate| candidate.source.as_str())
            .collect::<Vec<_>>();
        let aligned = candidates
            .iter()
            .filter(|candidate| candidate.align_as_name)
            .map(|candidate| candidate.source.as_str())
            .collect::<Vec<_>>();

        assert_eq!(sources, vec!["Clark"]);
        assert_eq!(aligned, vec!["Clark"]);
    }

    #[test]
    fn terminology_candidates_normalize_stuttered_names_and_drop_stuttered_starters() {
        let sources = extract_terminology_candidates(&[
            segment("1", "M-Morty, wait."),
            segment("2", "Morty, listen."),
            segment("3", "Je-Jessica left."),
            segment("4", "Jessica returned."),
            segment("5", "W-what happened?"),
            segment("6", "W-what now?"),
        ])
        .into_iter()
        .map(|candidate| candidate.source)
        .collect::<Vec<_>>();

        assert!(sources.contains(&"Morty".to_owned()));
        assert!(sources.contains(&"Jessica".to_owned()));
        assert!(!sources.iter().any(|source| source.contains('-')));
        assert!(!sources.contains(&"What".to_owned()));
    }

    #[test]
    fn terminology_payload_omits_ordinary_and_stuttered_latin_spans() {
        let candidates = vec![TerminologyCandidate {
            source: "Morty".to_owned(),
            context: "Morty arrived.".to_owned(),
            align_as_name: true,
        }];
        let parsed = parse_terminology_payload(
            &serde_json::json!({
                "entries": [
                    {"source": "All", "target": "全"},
                    {"source": "Big", "target": "大"},
                    {"source": "Summer", "target": "夏茉"}
                ],
                "entities": [
                    {
                        "canonical_source": "Morty",
                        "kind": "person",
                        "variants": [
                            {"source": "Morty", "target": "莫蒂"},
                            {"source": "M-Morty", "target": "莫-莫蒂"},
                            {"source": "Morty's", "target": "莫蒂的"}
                        ]
                    },
                    {
                        "canonical_source": "All",
                        "kind": "person",
                        "variants": [{"source": "All", "target": "全"}]
                    }
                ]
            }),
            &candidates,
            &[
                segment("1", "All right, Morty."),
                segment("2", "Big mistake, M-Morty."),
                segment("3", "Summer left."),
                segment("4", "Morty's ready."),
            ],
        )
        .expect("ordinary response entries should be safely omitted");

        assert_eq!(parsed.entries.len(), 1);
        assert_eq!(parsed.entries[0].source, "Summer");
        assert_eq!(parsed.entities.len(), 1);
        assert_eq!(parsed.entities[0].canonical_source, "Morty");
        assert_eq!(parsed.entities[0].variants.len(), 1);
        assert_eq!(parsed.entities[0].variants[0].source, "Morty");
    }

    #[test]
    fn auto_terminology_is_advisory_but_explicit_glossary_is_required() {
        let prepared = vec![translation_stage::PreparedBatch {
            index: 0,
            memory_hits: HashMap::new(),
            pending: vec![segment("69", "The Lord bless thee and keep thee.")],
            prompt_context: TranslationPromptContext::default(),
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
                terminology_updates: Vec::new(),
                usage: Usage::default(),
                cache_key: None,
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
            &TranslationPromptContext::default(),
            &memory,
            &BTreeMap::new(),
            false,
        );
        let advisory_context = translation_context(&advisory_messages);
        assert_eq!(advisory_context["terminology_hints"]["Lord"], "勋爵");
        assert!(advisory_context.get("glossary").is_none());

        let required = BTreeMap::from([("Lord".to_owned(), "勋爵".to_owned())]);
        let required_messages = build_translation_messages(
            &options,
            1,
            &prepared[0].pending,
            &TranslationPromptContext::default(),
            &memory,
            &required,
            false,
        );
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
        let document = document(
            "terms.txt",
            &[
                "Meet Alice now.",
                "Alice returned.",
                "Meet Bob now.",
                "Bob returned.",
            ],
        );
        assert_eq!(
            extract_terminology_candidates(&document.segments)
                .into_iter()
                .map(|candidate| candidate.source)
                .collect::<Vec<_>>(),
            vec!["Alice".to_owned(), "Bob".to_owned()]
        );
        let mut options = PipelineOptions::new("terms.txt".into());
        options.batch_size = 1;
        options.resume = false;
        options.mode = crate::entities::TranslationMode::Cinema;
        options.terminology_preflight = true;
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = SubtitlePipeline::new(
            PreflightBackend {
                contexts: Arc::clone(&contexts),
                preflight_requests: Arc::new(AtomicUsize::new(0)),
            },
            NoopDashboard,
            options,
        );

        let run = pipeline.run_document(&document).expect("translated");
        let contexts = contexts.lock().expect("contexts lock");

        assert!(
            run.result.terminology.entries_added >= 2,
            "terminology stats: {:?}",
            run.result.terminology
        );
        assert_eq!(contexts.len(), 4);
        assert!(contexts.iter().all(|context| {
            context["terminology_hints"]
                .as_object()
                .is_some_and(|map| !map.is_empty())
                && context.get("glossary").is_none()
        }));
    }

    #[test]
    fn explicit_turbo_terminology_preflight_runs_before_translation() {
        let document = document("terms.txt", &["Meet Alice now.", "Alice returned."]);
        let mut options = PipelineOptions::new("terms.txt".into());
        options.batch_size = 1;
        options.resume = false;
        options.mode = crate::entities::TranslationMode::Turbo;
        options.terminology_preflight = true;
        options.preserve_names = true;
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let preflight_requests = Arc::new(AtomicUsize::new(0));
        let mut pipeline = SubtitlePipeline::new(
            PreflightBackend {
                contexts: Arc::clone(&contexts),
                preflight_requests: Arc::clone(&preflight_requests),
            },
            NoopDashboard,
            options,
        );

        let run = pipeline.run_document(&document).expect("translated");
        let contexts = contexts.lock().expect("contexts lock");

        assert!(run.result.terminology.entries_added >= 1);
        assert_eq!(preflight_requests.load(Ordering::SeqCst), 1);
        assert_eq!(contexts.len(), 2);
        assert!(contexts.iter().all(|context| {
            context["terminology_hints"]
                .as_object()
                .is_some_and(|map| !map.is_empty())
        }));
    }

    #[test]
    fn terminology_preflight_consumes_the_shared_request_budget() {
        let document = document("terms.txt", &["Meet Alice now.", "Alice returned."]);
        let mut options = PipelineOptions::new("terms.txt".into());
        options.resume = false;
        options.mode = crate::entities::TranslationMode::Cinema;
        options.terminology_preflight = true;
        options.preserve_names = true;
        options.max_requests = Some(1);
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = SubtitlePipeline::new(
            PreflightBackend {
                contexts: Arc::clone(&contexts),
                preflight_requests: Arc::new(AtomicUsize::new(0)),
            },
            NoopDashboard,
            options,
        );

        let error = pipeline
            .run_document(&document)
            .expect_err("translation must stop after the preflight request");

        assert!(error.to_string().contains("request limit is 1"));
        assert!(contexts.lock().expect("contexts lock").is_empty());
    }

    #[test]
    fn cinema_preflight_runs_for_non_latin_documents_without_heuristic_candidates() {
        let document = document("terms.txt", &["量子航行系统启动。", "推进器正常。"]);
        let mut options = PipelineOptions::new("terms.txt".into());
        options.resume = false;
        options.mode = crate::entities::TranslationMode::Cinema;
        options.terminology_preflight = true;
        options.preserve_names = true;
        options.source_language = "zh-Hans".to_owned();
        options.target_language = "zh-Hans".to_owned();
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let preflight_requests = Arc::new(AtomicUsize::new(0));
        let mut pipeline = SubtitlePipeline::new(
            PreflightBackend {
                contexts,
                preflight_requests: Arc::clone(&preflight_requests),
            },
            NoopDashboard,
            options,
        );

        let run = pipeline.run_document(&document).expect("translated");

        assert_eq!(run.result.terminology.candidates, 0);
        assert_eq!(preflight_requests.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cinema_strict_preflight_rejects_unsupported_backend_before_translation() {
        let mut options = PipelineOptions::new("strict-preflight.txt".into());
        options.mode = crate::entities::TranslationMode::Cinema;
        options.terminology_preflight = true;
        options.allow_degraded_preflight = false;
        options.resume = false;
        options.use_cache = false;
        let mut pipeline = SubtitlePipeline::new(EchoBackend, NoopDashboard, options);

        let error = pipeline
            .run_document(&document("strict-preflight.txt", &["hello"]))
            .expect_err("strict preflight must reject unsupported capability");

        assert!(matches!(error, CoreError::UnsupportedCapability(_)));
    }

    #[test]
    fn turbo_reconciles_parallel_terminology_in_subtitle_order() {
        let mut options = PipelineOptions::new("terms.srt".into());
        options.mode = crate::entities::TranslationMode::Turbo;
        options.online_terminology = true;
        let mut pipeline = SubtitlePipeline::new(EchoBackend, NoopDashboard, options);
        let prepared = vec![
            translation_stage::PreparedBatch {
                index: 0,
                memory_hits: HashMap::new(),
                pending: vec![segment("1", "Zasa arrived.")],
                prompt_context: TranslationPromptContext::default(),
            },
            translation_stage::PreparedBatch {
                index: 1,
                memory_hits: HashMap::new(),
                pending: vec![segment("2", "Zasa sent them.")],
                prompt_context: TranslationPromptContext::default(),
            },
        ];
        let entity = |canonical: &str, source: &str, target: &str| TerminologyEntity {
            canonical_source: canonical.to_owned(),
            kind: TerminologyKind::Person,
            variants: vec![GlossaryEntry {
                source: source.to_owned(),
                target: target.to_owned(),
            }],
        };
        let mut generated = HashMap::from([
            (
                2,
                BatchWithUsage {
                    lines: vec![TranslationLine {
                        id: "2".to_owned(),
                        translation: "萨萨派他们来的。".to_owned(),
                    }],
                    summary: String::new(),
                    glossary_updates: Vec::new(),
                    terminology_updates: vec![entity("Joey Zasa", "Zasa", "萨萨")],
                    usage: Usage::default(),
                    cache_key: None,
                },
            ),
            (
                1,
                BatchWithUsage {
                    lines: vec![TranslationLine {
                        id: "1".to_owned(),
                        translation: "扎萨到了。".to_owned(),
                    }],
                    summary: String::new(),
                    glossary_updates: Vec::new(),
                    terminology_updates: vec![entity("Joey Zasa", "Zasa", "扎萨")],
                    usage: Usage::default(),
                    cache_key: None,
                },
            ),
        ]);
        pipeline
            .reconcile_translation_window(&prepared, &mut generated)
            .expect("reconcile parallel window");

        assert_eq!(generated[&2].lines[0].translation, "扎萨派他们来的。");
        assert_eq!(
            generated[&1].terminology_updates[0].variants[0].target,
            "扎萨"
        );
    }

    #[test]
    fn unsafe_terminology_conflict_retries_only_that_batch() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut options = PipelineOptions::new("terms.srt".into());
        options.online_terminology = true;
        options.use_cache = false;
        let mut pipeline = SubtitlePipeline::new(
            CorrectedTerminologyBackend {
                calls: Arc::clone(&calls),
            },
            NoopDashboard,
            options,
        );
        pipeline
            .required_glossary
            .insert("zasa".to_owned(), "扎萨".to_owned());
        let prepared = vec![translation_stage::PreparedBatch {
            index: 0,
            memory_hits: HashMap::new(),
            pending: vec![segment("1", "Zasa arrived.")],
            prompt_context: TranslationPromptContext::default(),
        }];
        let mut generated = HashMap::from([(
            1,
            BatchWithUsage {
                lines: vec![TranslationLine {
                    id: "1".to_owned(),
                    translation: "那个家伙来了。".to_owned(),
                }],
                summary: String::new(),
                glossary_updates: Vec::new(),
                terminology_updates: vec![TerminologyEntity {
                    canonical_source: "Joey Zasa".to_owned(),
                    kind: TerminologyKind::Person,
                    variants: vec![GlossaryEntry {
                        source: "Zasa".to_owned(),
                        target: "萨萨".to_owned(),
                    }],
                }],
                usage: Usage::default(),
                cache_key: None,
            },
        )]);

        pipeline
            .reconcile_translation_window(&prepared, &mut generated)
            .expect("targeted retry");

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(generated[&1].lines[0].translation, "扎萨来了。");
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
        let captured = Arc::new(Mutex::new(CapturedStoreData::default()));
        let store = CapturedStore {
            paths: build_runtime_paths(
                std::path::Path::new("review.txt"),
                std::path::Path::new("/workspace/review.txt"),
                None,
                None,
                "Auto",
                "Chinese",
                false,
            ),
            data: Arc::clone(&captured),
        };
        let mut pipeline = SubtitlePipeline::new(
            ReviewBackend {
                translation_calls: Arc::clone(&translation_calls),
                review_calls: Arc::clone(&review_calls),
                fail_on_review_call: None,
            },
            NoopDashboard,
            options,
        )
        .with_store(Box::new(store));

        let run = pipeline.run_document(&document).expect("reviewed run");

        assert_eq!(translation_calls.load(Ordering::SeqCst), 1);
        assert_eq!(review_calls.load(Ordering::SeqCst), 1);
        assert_eq!(run.result.review_batches, 1);
        assert_eq!(run.result.usage.total_tokens, 5);
        assert_eq!(
            run.translated_segments[0].text,
            "[REVIEWED] [ECHO] Meet Alice now."
        );
        let data = captured.lock().expect("capture lock");
        assert_eq!(
            data.saved_translation_memory
                .iter()
                .find(|(key, _)| key.ends_with(":meet alice now."))
                .map(|(_, text)| text.as_str()),
            Some("[REVIEWED] [ECHO] Meet Alice now.")
        );
    }

    #[test]
    fn parallel_review_uses_the_explicit_reviewer_backend() {
        let document = document("parallel-review.txt", &["First.", "Second."]);
        let mut options = PipelineOptions::new("parallel-review.txt".into());
        options.batch_size = 1;
        options.translation_concurrency = 2;
        options.review_concurrency = 2;
        options.terminology_preflight = false;
        options.review_policy = ReviewPolicy::Full;
        options.resume = false;
        options.use_cache = false;
        let translator_translation_calls = Arc::new(AtomicUsize::new(0));
        let translator_review_calls = Arc::new(AtomicUsize::new(0));
        let reviewer_translation_calls = Arc::new(AtomicUsize::new(0));
        let reviewer_review_calls = Arc::new(AtomicUsize::new(0));
        let mut pipeline = SubtitlePipeline::new(
            RoutedParallelBackend {
                label: "translator",
                translation_calls: Arc::clone(&translator_translation_calls),
                review_calls: Arc::clone(&translator_review_calls),
            },
            NoopDashboard,
            options,
        )
        .with_reviewer(RoutedParallelBackend {
            label: "reviewer",
            translation_calls: Arc::clone(&reviewer_translation_calls),
            review_calls: Arc::clone(&reviewer_review_calls),
        });

        let run = pipeline.run_document(&document).expect("reviewed run");

        assert_eq!(translator_translation_calls.load(Ordering::SeqCst), 2);
        assert_eq!(translator_review_calls.load(Ordering::SeqCst), 0);
        assert_eq!(reviewer_translation_calls.load(Ordering::SeqCst), 0);
        assert_eq!(reviewer_review_calls.load(Ordering::SeqCst), 2);
        assert_eq!(run.result.review.usage.total_tokens, 10);
        assert!(
            run.translated_segments
                .iter()
                .all(|segment| segment.text.starts_with("[reviewer REVIEW]"))
        );
    }

    #[test]
    fn parallel_translation_uses_source_context_then_confirmed_previous_window() {
        let document = document(
            "parallel-context.txt",
            &["Line 1.", "Line 2.", "Line 3.", "Line 4.", "Line 5."],
        );
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let mut options = PipelineOptions::new("parallel-context.txt".into());
        options.batch_size = 1;
        options.translation_concurrency = 2;
        options.terminology_preflight = false;
        options.preserve_names = true;
        options.resume = false;
        options.use_cache = false;
        let mut pipeline = SubtitlePipeline::new(
            ContextCaptureBackend {
                contexts: Arc::clone(&contexts),
            },
            NoopDashboard,
            options,
        );

        pipeline.run_document(&document).expect("contextual run");

        let contexts = contexts.lock().expect("context lock");
        assert_eq!(contexts.len(), 5);
        assert_eq!(contexts[0]["editable_ids"], serde_json::json!(["1"]));
        assert_eq!(
            contexts[0]["readonly_source"]["after"],
            serde_json::json!([
                {"id": "2", "source": "Line 2."},
                {"id": "3", "source": "Line 3."},
                {"id": "4", "source": "Line 4."}
            ])
        );
        assert!(contexts[1].get("confirmed_previous").is_none());
        assert_eq!(contexts[4]["confirmed_previous"][0]["id"], "4");
        assert_eq!(
            contexts[4]["confirmed_previous"][0]["translation"],
            "[ECHO] Line 4."
        );
        assert!(
            contexts
                .iter()
                .all(|context| context.get("recent").is_none())
        );
    }

    #[test]
    fn composed_translation_uses_initial_confirmed_context_for_its_first_window() {
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let mut options = PipelineOptions::new("next-chunk.txt".into());
        options.batch_size = 1;
        options.terminology_preflight = false;
        options.preserve_names = true;
        options.resume = false;
        options.use_cache = false;
        options.initial_confirmed_context = vec![crate::ConfirmedTranslationContext {
            id: "previous".to_owned(),
            source: "Previous line.".to_owned(),
            translation: "上一句。".to_owned(),
        }];
        let mut pipeline = SubtitlePipeline::new(
            ContextCaptureBackend {
                contexts: Arc::clone(&contexts),
            },
            NoopDashboard,
            options,
        );

        pipeline
            .run_document(&document("next-chunk.txt", &["Next line."]))
            .expect("contextual run");

        let contexts = contexts.lock().expect("context lock");
        assert_eq!(contexts[0]["confirmed_previous"][0]["id"], "previous");
        assert_eq!(
            contexts[0]["confirmed_previous"][0]["translation"],
            "上一句。"
        );
    }

    #[test]
    fn parallel_review_falls_back_to_the_translator_without_a_reviewer() {
        let document = document("parallel-self-review.txt", &["First.", "Second."]);
        let mut options = PipelineOptions::new("parallel-self-review.txt".into());
        options.batch_size = 1;
        options.translation_concurrency = 2;
        options.review_concurrency = 2;
        options.terminology_preflight = false;
        options.review_policy = ReviewPolicy::Full;
        options.resume = false;
        options.use_cache = false;
        let translation_calls = Arc::new(AtomicUsize::new(0));
        let review_calls = Arc::new(AtomicUsize::new(0));
        let mut pipeline = SubtitlePipeline::new(
            RoutedParallelBackend {
                label: "translator",
                translation_calls: Arc::clone(&translation_calls),
                review_calls: Arc::clone(&review_calls),
            },
            NoopDashboard,
            options,
        );

        let run = pipeline.run_document(&document).expect("self-reviewed run");

        assert_eq!(translation_calls.load(Ordering::SeqCst), 2);
        assert_eq!(review_calls.load(Ordering::SeqCst), 2);
        assert!(
            run.translated_segments
                .iter()
                .all(|segment| segment.text.starts_with("[translator REVIEW]"))
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
            semantic: Default::default(),
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
            metadata: Default::default(),
        }
    }

    #[derive(Debug, Default)]
    struct CapturedStoreData {
        saved_translation_memory: Vec<(String, String)>,
        loaded_translation_memory: Vec<(String, String)>,
        loaded_glossary: Vec<(String, String)>,
        review_reports: Vec<ReviewReport>,
        saved_batches: Vec<(usize, Vec<SubtitleSegment>)>,
        saved_review_batches: Vec<(usize, Vec<SubtitleSegment>)>,
        saved_finalized_batches: Vec<(usize, Vec<SubtitleSegment>)>,
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

        fn load_glossary(&self) -> CoreResult<Vec<(String, String)>> {
            Ok(self
                .data
                .lock()
                .expect("capture lock")
                .loaded_glossary
                .clone())
        }

        fn save_translation_memory(&self, entries: &[(String, String)]) -> CoreResult<()> {
            let mut data = self.data.lock().expect("capture lock");
            data.saved_translation_memory = entries.to_vec();
            data.saved_translation_memory.sort();
            Ok(())
        }

        fn load_translation_memory(&self) -> CoreResult<Vec<(String, String)>> {
            Ok(self
                .data
                .lock()
                .expect("capture lock")
                .loaded_translation_memory
                .clone())
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
                BatchShardKind::Finalized => data
                    .saved_finalized_batches
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
                BatchShardKind::Finalized => &data.saved_finalized_batches,
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
