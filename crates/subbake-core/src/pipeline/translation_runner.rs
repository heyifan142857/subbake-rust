use crate::entities::{
    ConcurrencyStrategy, ContextStrategy, ReviewPolicy, SubtitleSegment, TerminologyStats, Usage,
};
use crate::error::{CoreError, CoreResult};
use crate::ports::{BatchShardKind, DashboardSink, LlmBackend};
use crate::progress::TaskState;
use crate::storage::ResumeSnapshot;
use crate::validation::validate_full_alignment;
use std::collections::{HashMap, HashSet};

use super::SubtitlePipeline;
use super::support::validate_window_terminology;
use super::translation_stage::{
    SourceBatchContext, TranslationPromptContext, TranslationStage, bounded_confirmed_context,
};

const CINEMA_RELEVANT_PREVIOUS_LINES: usize = 4;
const CINEMA_RELEVANT_PREVIOUS_TOKEN_BUDGET: usize = 600;

pub(super) struct TranslationRun {
    pub batches: Vec<Vec<SubtitleSegment>>,
    pub segments: Vec<SubtitleSegment>,
    pub usage: Usage,
}

pub(super) struct TranslationRunInput<'a> {
    pub batches: Vec<Vec<SubtitleSegment>>,
    pub resume: &'a ResumeSnapshot,
    pub terminology: &'a TerminologyStats,
    pub memory_keys: &'a HashMap<String, String>,
    pub source_contexts: &'a [SourceBatchContext],
    pub scene_groups: &'a [usize],
}

pub(super) fn run<B, D>(
    pipeline: &mut SubtitlePipeline<B, D>,
    document: &crate::entities::SubtitleDocument,
    input: TranslationRunInput<'_>,
) -> CoreResult<TranslationRun>
where
    B: LlmBackend,
    D: DashboardSink,
{
    let TranslationRunInput {
        batches,
        resume,
        terminology,
        memory_keys,
        source_contexts,
        scene_groups,
    } = input;
    pipeline.report(
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
    let mut stage = TranslationStage::new(
        batches,
        resume.translation_batches_completed,
        resume.translated_segments.clone(),
        memory_keys.clone(),
    )?;
    let mut usage = resume.usage;
    if resume.translation_batches_completed == 0 {
        usage.add(terminology.usage);
    }
    if usage != Usage::default() {
        pipeline.dashboard.add_usage(usage);
    }

    while !stage.is_complete() {
        pipeline.cancellation.check()?;
        let concurrency = if pipeline.backend.supports_parallel_generation() {
            pipeline.effective_translation_concurrency()
        } else {
            1
        };
        let window_size = match pipeline.options.policy().concurrency_strategy {
            ConcurrencyStrategy::AdaptiveQueued { window_multiplier }
                if pipeline.backend.supports_parallel_generation() =>
            {
                concurrency.saturating_mul(window_multiplier)
            }
            ConcurrencyStrategy::SceneAware => {
                cinema_window_size(stage.next_batch(), concurrency, scene_groups)
            }
            ConcurrencyStrategy::Fixed | ConcurrencyStrategy::AdaptiveQueued { .. } => concurrency,
        };
        let mut previous_confirmed = stage.previous_confirmed_context(
            pipeline.options.confirmed_context_lines,
            pipeline.options.confirmed_context_token_budget,
        );
        if previous_confirmed.is_empty() {
            previous_confirmed = bounded_confirmed_context(
                &pipeline.options.initial_confirmed_context,
                pipeline.options.confirmed_context_lines,
                pipeline.options.confirmed_context_token_budget,
            );
        }
        let mut prepared = stage.prepare_window(
            window_size,
            pipeline.options.use_cache,
            &pipeline.translation_memory,
        );
        for batch in &mut prepared {
            let source_context = source_contexts
                .get(batch.index)
                .cloned()
                .unwrap_or_default();
            let excluded_ids = previous_confirmed
                .iter()
                .map(|line| line.id.as_str())
                .chain(source_context.before.iter().map(|line| line.id.as_str()))
                .collect::<HashSet<_>>();
            let relevant_previous =
                if pipeline.options.policy().context_strategy == ContextStrategy::SceneAware {
                    stage.relevant_previous_context(
                        &batch.pending,
                        &excluded_ids,
                        CINEMA_RELEVANT_PREVIOUS_LINES,
                        CINEMA_RELEVANT_PREVIOUS_TOKEN_BUDGET,
                    )
                } else {
                    Vec::new()
                };
            batch.prompt_context = TranslationPromptContext {
                source: source_context,
                previous_confirmed: previous_confirmed.clone(),
                relevant_previous,
            };
        }
        pipeline.report(
            "TRANSLATE",
            TaskState::Running,
            stage.next_batch(),
            Some(stage.len()),
            resume.translation_batches_completed,
            usage,
        );
        let pending = prepared
            .iter()
            .filter(|batch| !batch.pending.is_empty())
            .map(|batch| {
                (
                    batch.index + 1,
                    batch.pending.clone(),
                    batch.prompt_context.clone(),
                )
            })
            .collect::<Vec<_>>();
        pipeline.report_translation_window(
            stage.batches(),
            stage.next_batch(),
            pending.len(),
            resume.translation_batches_completed,
            usage,
        );
        let mut generated = pipeline.translate_window(&pending)?;
        pipeline.reconcile_translation_window(&prepared, &mut generated)?;
        validate_window_terminology(
            &prepared,
            &generated,
            &pipeline.required_glossary,
            pipeline.options.review_policy != ReviewPolicy::Off,
        )?;
        for prepared_batch in prepared {
            let result = if prepared_batch.pending.is_empty() {
                None
            } else {
                Some(
                    generated
                        .remove(&(prepared_batch.index + 1))
                        .ok_or_else(|| {
                            CoreError::DataInvariant(format!(
                                "translation window omitted batch {}",
                                prepared_batch.index + 1
                            ))
                        })?,
                )
            };
            if let Some(result) = result.as_ref() {
                pipeline.save_reconciled_translation_cache(result)?;
            }
            let applied = stage.apply(prepared_batch, result)?;
            if let Some(result) = applied.result.as_ref() {
                usage.add(result.usage);
                pipeline.dashboard.add_usage(result.usage);
                pipeline.memory.update("", &result.glossary_updates);
                pipeline.commit_terminology_updates(&result.terminology_updates);
            }
            pipeline.translation_memory_hits = stage.memory_hits();
            if let Some(store) = pipeline.store.as_ref() {
                pipeline.cancellation.check()?;
                store.save_glossary(
                    &pipeline
                        .memory
                        .glossary
                        .iter()
                        .map(|(source, target)| (source.clone(), target.clone()))
                        .collect::<Vec<_>>(),
                )?;
                store.save_batch_segments(
                    BatchShardKind::Translated,
                    applied.index + 1,
                    &applied.translated,
                )?;
            }
            pipeline.cancellation.check()?;
            pipeline.save_run_state(
                applied.index + 1,
                resume.review_batches_completed,
                false,
                usage,
            )?;
            pipeline.report(
                "TRANSLATE",
                TaskState::Running,
                applied.index + 1,
                Some(stage.len()),
                resume.translation_batches_completed,
                usage,
            );
        }
        pipeline.report_translation_window(
            stage.batches(),
            stage.next_batch(),
            0,
            resume.translation_batches_completed,
            usage,
        );
        pipeline.note_translation_window_success();
    }

    validate_full_alignment(&document.segments, stage.output())?;
    pipeline.cancellation.check()?;
    pipeline.save_run_state(
        stage.len(),
        resume.review_batches_completed,
        resume.validation_completed,
        usage,
    )?;
    pipeline.dashboard.mark_done("TRANSLATE");
    let batches = stage.batches().to_vec();
    Ok(TranslationRun {
        batches,
        segments: stage.finish(),
        usage,
    })
}

fn cinema_window_size(start: usize, concurrency: usize, scene_groups: &[usize]) -> usize {
    let mut seen = HashSet::new();
    let mut count = 0usize;
    for group in scene_groups.iter().skip(start).take(concurrency.max(1)) {
        if !seen.insert(*group) {
            break;
        }
        count += 1;
    }
    count.max(1)
}

#[cfg(test)]
mod tests {
    use super::cinema_window_size;

    #[test]
    fn cinema_window_never_contains_two_batches_from_the_same_scene() {
        let groups = [0, 0, 1, 1, 2];

        assert_eq!(cinema_window_size(0, 4, &groups), 1);
        assert_eq!(cinema_window_size(1, 4, &groups), 2);
        assert_eq!(cinema_window_size(3, 4, &groups), 2);
    }
}
