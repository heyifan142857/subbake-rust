use crate::entities::{
    ConcurrencyStrategy, ReviewReport, ReviewRoute, ReviewRouteKind, ReviewStats, SubtitleDocument,
    SubtitleSegment, Usage,
};
use crate::error::{CoreError, CoreResult};
use crate::ports::{BatchShardKind, LlmBackend};
use crate::progress::TaskState;
use crate::storage::{REVIEW_REPORT_VERSION, ResumeSnapshot};
use crate::validation::validate_full_alignment;

use super::SubtitlePipeline;
use super::review_stage::{ReviewResumeInput, ReviewStage};

pub(super) struct ReviewRun {
    pub output: Vec<SubtitleSegment>,
    pub stats: ReviewStats,
    pub batches: usize,
    pub resumed: usize,
    pub usage: Usage,
}

pub(super) struct ReviewBatchInput<'a> {
    pub review_batches: &'a [Vec<SubtitleSegment>],
    pub translation_batches: usize,
    pub scene_groups: &'a [usize],
}

pub(super) fn run<B>(
    pipeline: &mut SubtitlePipeline<B>,
    document: &SubtitleDocument,
    batch_input: ReviewBatchInput<'_>,
    translated: &[SubtitleSegment],
    resume: &ResumeSnapshot,
    terminology: &crate::entities::TerminologyStats,
    mut usage: Usage,
) -> CoreResult<ReviewRun>
where
    B: LlmBackend,
{
    let batches = batch_input.review_batches;
    let translation_batches = batch_input.translation_batches;
    let scene_groups = batch_input.scene_groups;
    let restored_report = if resume.review_batches_completed > 0 {
        pipeline
            .store
            .as_ref()
            .map(|store| store.load_review_report())
            .transpose()?
            .flatten()
    } else {
        None
    };
    let route = (pipeline.options.execution.review_policy != crate::entities::ReviewPolicy::Off)
        .then(|| ReviewRoute {
            kind: if pipeline.reviewer.is_some() {
                ReviewRouteKind::ConfiguredReviewer
            } else {
                ReviewRouteKind::TranslatorFallback
            },
            backend_fingerprint: if pipeline.reviewer.is_some() {
                pipeline.options.identity.reviewer_fingerprint.clone()
            } else {
                pipeline
                    .options
                    .identity
                    .provider_fingerprint
                    .clone()
                    .or_else(|| {
                        Some(format!(
                            "{}:{}",
                            pipeline.options.identity.provider, pipeline.options.identity.model
                        ))
                    })
            },
        });
    let mut stage = ReviewStage::new_with_rules(
        &pipeline.options,
        &pipeline.language_rules,
        batches,
        translated,
        &pipeline.memory,
        &pipeline.required_glossary,
        ReviewResumeInput {
            completed_batches: resume.review_batches_completed,
            reviewed_segments: &resume.reviewed_segments,
            report: restored_report.as_ref(),
            cache_hits_before: pipeline.accounting.cache_hits(),
        },
    )?;
    let resumed = stage.resumed();
    if !stage.is_empty() {
        let mut next_review = resumed;
        while next_review < stage.len() {
            pipeline.cancellation.check()?;
            let concurrency = if pipeline.review_backend_supports_parallel_generation() {
                pipeline.options.execution.review_concurrency.max(1)
            } else {
                1
            };
            pipeline.report(
                "FINAL_REVIEW",
                TaskState::Running,
                next_review,
                Some(stage.len()),
                resumed,
                usage,
            );
            let window_size = match pipeline.options.policy().concurrency_strategy {
                ConcurrencyStrategy::AdaptiveQueued { window_multiplier }
                    if pipeline.review_backend_supports_parallel_generation() =>
                {
                    concurrency.saturating_mul(window_multiplier)
                }
                ConcurrencyStrategy::SceneAware => {
                    stage.scene_window_size(next_review, concurrency, scene_groups)
                }
                ConcurrencyStrategy::Fixed | ConcurrencyStrategy::AdaptiveQueued { .. } => {
                    concurrency
                }
            };
            let pending = stage.window(next_review, window_size);
            let mut reviewed_window = pipeline.review_window(&pending)?;
            for (review_position, _) in &pending {
                let reviewed = reviewed_window.remove(review_position).ok_or_else(|| {
                    CoreError::DataInvariant(format!(
                        "review window omitted batch {review_position}"
                    ))
                })?;
                usage.add(reviewed.usage);
                let reviewed_segments = stage.apply(
                    *review_position,
                    &reviewed.lines,
                    &reviewed.annotations,
                    reviewed.usage,
                )?;
                if let Some(store) = pipeline.store.as_ref() {
                    pipeline.cancellation.check()?;
                    let (review, changes) = stage.snapshot(pipeline.accounting.cache_hits());
                    store.save_review_report(&ReviewReport {
                        version: REVIEW_REPORT_VERSION,
                        terminology: terminology.clone(),
                        review,
                        changes,
                        route: route.clone(),
                    })?;
                }
                pipeline.commit_checkpoint(
                    translation_batches,
                    *review_position,
                    false,
                    usage,
                    vec![crate::ports::CheckpointShard {
                        kind: BatchShardKind::Reviewed,
                        batch_index: *review_position,
                        segments: reviewed_segments,
                    }],
                )?;
                pipeline.report(
                    "FINAL_REVIEW",
                    TaskState::Running,
                    *review_position,
                    Some(stage.len()),
                    resumed,
                    usage,
                );
            }
            next_review += pending.len();
        }
        validate_full_alignment(&document.segments, stage.output())?;
    } else {
        pipeline.report(
            "FINAL_REVIEW",
            TaskState::Skipped,
            0,
            Some(0),
            0,
            Usage::default(),
        );
    }

    let review_batches = stage.len();
    let outcome = stage.finish(pipeline.accounting.cache_hits());
    if let Some(store) = pipeline.store.as_ref() {
        store.save_review_report(&ReviewReport {
            version: REVIEW_REPORT_VERSION,
            terminology: terminology.clone(),
            review: outcome.stats.clone(),
            changes: outcome.changes,
            route,
        })?;
    }
    pipeline.cancellation.check()?;
    pipeline.save_run_state(
        translation_batches,
        review_batches,
        resume.validation_completed,
        usage,
    )?;
    if review_batches > 0 {
        pipeline.report(
            "FINAL_REVIEW",
            TaskState::Completed,
            review_batches,
            Some(review_batches),
            resumed,
            usage,
        );
    }
    Ok(ReviewRun {
        output: outcome.output,
        stats: outcome.stats,
        batches: review_batches,
        resumed,
        usage,
    })
}
