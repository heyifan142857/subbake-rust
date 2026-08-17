use std::collections::HashSet;
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
        let plan = if translated.is_empty() {
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
            .map(|(index, batch)| {
                let mut batch = batch.clone();
                for context in batch.before.iter_mut().chain(&mut batch.after) {
                    if let Some(current) = self
                        .output
                        .iter()
                        .find(|line| line.id == context.translated.id)
                    {
                        context.translated = current.clone();
                    }
                }
                for occurrence in batch
                    .consistency_groups
                    .iter_mut()
                    .flat_map(|group| &mut group.occurrences)
                {
                    if let Some(current) = self
                        .output
                        .iter()
                        .find(|line| line.id == occurrence.translated.id)
                    {
                        occurrence.translated = current.clone();
                    }
                }
                (index + 1, batch)
            })
            .collect()
    }

    pub fn scene_window_size(
        &self,
        start: usize,
        concurrency: usize,
        scene_groups: &[usize],
    ) -> usize {
        let mut seen = HashSet::new();
        let mut seen_consistency_groups = HashSet::new();
        let mut count = 0usize;
        for batch in self.plan.iter().skip(start).take(concurrency.max(1)) {
            let Some(group) = batch
                .batch_index
                .checked_sub(1)
                .and_then(|index| scene_groups.get(index))
            else {
                break;
            };
            if !seen.insert(*group) {
                break;
            }
            if batch
                .consistency_groups
                .iter()
                .any(|group| seen_consistency_groups.contains(&group.source_key))
            {
                break;
            }
            seen_consistency_groups.extend(
                batch
                    .consistency_groups
                    .iter()
                    .map(|group| group.source_key.clone()),
            );
            count += 1;
        }
        count.max(1)
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
        let reviewed = super::support::apply_lines(&batch.source, lines);
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
                duration_ms: super::support::duration_ms(self.started),
            },
            output: self.output,
            changes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::{TranslationLine, TranslationMode};

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

    #[test]
    fn cinema_serializes_shared_consistency_groups_and_refreshes_peer_translations() {
        let source = vec![
            segment("1", "Open"),
            segment("2", "Later"),
            segment("3", "Open"),
        ];
        let mut translated = source.clone();
        translated[0].text = "打开".to_owned();
        translated[1].text = "稍后".to_owned();
        translated[2].text = "营业中".to_owned();
        let batches = source
            .iter()
            .cloned()
            .map(|line| vec![line])
            .collect::<Vec<_>>();
        let mut options = PipelineOptions::new("review.ass".into());
        options.mode = TranslationMode::Cinema;
        options.review_policy = ReviewPolicy::Full;
        let mut stage = ReviewStage::new(
            &options,
            &batches,
            &translated,
            &ContextMemory::default(),
            0,
            &[],
            0,
        )
        .expect("review stage");

        assert_eq!(stage.scene_window_size(0, 3, &[0, 1, 2]), 2);
        stage
            .apply(
                1,
                &[TranslationLine {
                    id: "1".to_owned(),
                    translation: "开启".to_owned(),
                }],
                Usage::default(),
            )
            .expect("apply first review");
        let window = stage.window(2, 1);
        let peer = window[0]
            .1
            .consistency_groups
            .iter()
            .flat_map(|group| &group.occurrences)
            .find(|occurrence| occurrence.source.id == "1")
            .expect("refreshed peer");

        assert_eq!(peer.translated.text, "开启");
    }
}
