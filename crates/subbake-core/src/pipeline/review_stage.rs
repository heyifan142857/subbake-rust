use std::time::Instant;

use crate::entities::{
    PipelineOptions, ReviewChange, ReviewPolicy, ReviewStats, SubtitleSegment, TranslationLine,
    Usage,
};
use crate::error::{CoreError, CoreResult};
use crate::memory::ContextMemory;
use crate::review::{
    ReviewBatchPlan, build_full_review_plan, build_review_plan, restore_review_progress,
};

pub(super) struct ReviewStage {
    plan: Vec<ReviewBatchPlan>,
    output: Vec<SubtitleSegment>,
    resumed: usize,
    usage: Usage,
    cache_hits_before: usize,
    started: Instant,
}

pub(super) struct ReviewOutcome {
    pub output: Vec<SubtitleSegment>,
    pub stats: ReviewStats,
    pub changes: Vec<ReviewChange>,
}

impl ReviewStage {
    pub fn new(
        options: &PipelineOptions,
        batches: &[Vec<SubtitleSegment>],
        translated: &[SubtitleSegment],
        memory: &ContextMemory,
        resumed: usize,
        reviewed_segments: &[SubtitleSegment],
        cache_hits_before: usize,
    ) -> CoreResult<Self> {
        let plan = if options.fast_mode || translated.is_empty() {
            Vec::new()
        } else {
            match options.review_policy {
                ReviewPolicy::Off => Vec::new(),
                ReviewPolicy::Targeted => build_review_plan(
                    batches,
                    translated,
                    memory,
                    &options.source_language,
                    &options.target_language,
                ),
                ReviewPolicy::Full => build_full_review_plan(batches, translated),
            }
        };
        let resumed = if plan.is_empty() {
            0
        } else if resumed > plan.len() {
            return Err(CoreError::DataInvariant(format!(
                "resume state has {resumed} reviewed batches, but the current review plan has only {}",
                plan.len()
            )));
        } else {
            resumed
        };
        let mut output = translated.to_vec();
        if !plan.is_empty() {
            restore_review_progress(&plan, resumed, reviewed_segments, &mut output)?;
        }
        Ok(Self {
            plan,
            output,
            resumed,
            usage: Usage::default(),
            cache_hits_before,
            started: Instant::now(),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.plan.is_empty()
    }

    pub fn len(&self) -> usize {
        self.plan.len()
    }

    pub fn resumed(&self) -> usize {
        self.resumed
    }

    pub fn window(&self, start: usize, concurrency: usize) -> Vec<(usize, ReviewBatchPlan)> {
        self.plan
            .iter()
            .enumerate()
            .skip(start)
            .take(concurrency.max(1))
            .map(|(index, batch)| (index + 1, batch.clone()))
            .collect()
    }

    pub fn apply(
        &mut self,
        position: usize,
        lines: &[TranslationLine],
        usage: Usage,
    ) -> CoreResult<Vec<SubtitleSegment>> {
        let index = position.checked_sub(1).ok_or_else(|| {
            CoreError::DataInvariant("review result has invalid batch position 0".to_owned())
        })?;
        let batch = self.plan.get(index).ok_or_else(|| {
            CoreError::DataInvariant(format!(
                "review result has invalid batch position {position}"
            ))
        })?;
        let reviewed = super::apply_lines(&batch.source, lines);
        self.output[batch.start_offset..batch.start_offset + reviewed.len()]
            .clone_from_slice(&reviewed);
        self.usage.add(usage);
        Ok(reviewed)
    }

    pub fn output(&self) -> &[SubtitleSegment] {
        &self.output
    }

    pub fn finish(self, cache_hits: usize) -> ReviewOutcome {
        let changes = self
            .plan
            .iter()
            .flat_map(|batch| {
                let reviewed =
                    &self.output[batch.start_offset..batch.start_offset + batch.translated.len()];
                batch
                    .translated
                    .iter()
                    .zip(reviewed)
                    .filter(|(before, after)| before.text != after.text)
                    .map(|(before, after)| ReviewChange {
                        batch: batch.batch_index,
                        id: before.id.clone(),
                        reasons: batch
                            .candidate_reasons
                            .get(&before.id)
                            .cloned()
                            .unwrap_or_default(),
                        before: before.text.clone(),
                        after: after.text.clone(),
                    })
            })
            .collect::<Vec<_>>();
        let candidate_lines = self
            .plan
            .iter()
            .map(|batch| batch.candidate_reasons.len())
            .sum();
        ReviewOutcome {
            stats: ReviewStats {
                candidate_lines,
                reviewed_lines: candidate_lines,
                changed_lines: changes.len(),
                batches: self.plan.len(),
                cache_hits: cache_hits.saturating_sub(self.cache_hits_before),
                usage: self.usage,
                duration_ms: super::duration_ms(self.started),
            },
            output: self.output,
            changes,
        }
    }
}
