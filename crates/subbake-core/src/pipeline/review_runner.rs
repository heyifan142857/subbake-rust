use crate::entities::{ReviewReport, ReviewStats, SubtitleDocument, SubtitleSegment, Usage};
use crate::error::{CoreError, CoreResult};
use crate::ports::{BatchShardKind, DashboardSink, LlmBackend};
use crate::progress::TaskState;
use crate::storage::ResumeSnapshot;
use crate::validation::validate_full_alignment;

use super::SubtitlePipeline;
use super::review_stage::ReviewStage;

pub(super) struct ReviewRun {
    pub output: Vec<SubtitleSegment>,
    pub stats: ReviewStats,
    pub batches: usize,
    pub resumed: usize,
    pub usage: Usage,
}

pub(super) fn run<B, D>(
    pipeline: &mut SubtitlePipeline<B, D>,
    document: &SubtitleDocument,
    batches: &[Vec<SubtitleSegment>],
    translated: &[SubtitleSegment],
    resume: &ResumeSnapshot,
    terminology: &crate::entities::TerminologyStats,
    mut usage: Usage,
) -> CoreResult<ReviewRun>
where
    B: LlmBackend,
    D: DashboardSink,
{
    let mut stage = ReviewStage::new(
        &pipeline.options,
        batches,
        translated,
        &pipeline.memory,
        resume.review_batches_completed,
        &resume.reviewed_segments,
        pipeline.cache_hits,
    )?;
    pipeline
        .dashboard
        .set_total_steps(3 + batches.len() + stage.len());
    let resumed = stage.resumed();
    if !stage.is_empty() {
        pipeline.dashboard.mark_running("FINAL_REVIEW");
        let mut next_review = resumed;
        while next_review < stage.len() {
            pipeline.cancellation.check()?;
            let concurrency = if pipeline.backend.supports_parallel_generation() {
                pipeline.options.review_concurrency.max(1)
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
            let pending = stage.window(next_review, concurrency);
            let mut reviewed_window = pipeline.review_window(&pending)?;
            for (review_position, _) in &pending {
                let reviewed = reviewed_window.remove(review_position).ok_or_else(|| {
                    CoreError::DataInvariant(format!(
                        "review window omitted batch {review_position}"
                    ))
                })?;
                usage.add(reviewed.usage);
                pipeline.dashboard.add_usage(reviewed.usage);
                let reviewed_segments =
                    stage.apply(*review_position, &reviewed.lines, reviewed.usage)?;
                if let Some(store) = pipeline.store.as_ref() {
                    pipeline.cancellation.check()?;
                    store.save_batch_segments(
                        BatchShardKind::Reviewed,
                        *review_position,
                        &reviewed_segments,
                    )?;
                }
                pipeline.save_run_state(batches.len(), *review_position, true, usage)?;
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
        pipeline.dashboard.mark_done("FINAL_REVIEW");
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
    let outcome = stage.finish(pipeline.cache_hits);
    if let Some(store) = pipeline.store.as_ref() {
        store.save_review_report(&ReviewReport {
            terminology: terminology.clone(),
            review: outcome.stats.clone(),
            changes: outcome.changes,
        })?;
    }
    pipeline.cancellation.check()?;
    pipeline.save_run_state(batches.len(), review_batches, true, usage)?;
    Ok(ReviewRun {
        output: outcome.output,
        stats: outcome.stats,
        batches: review_batches,
        resumed,
        usage,
    })
}
