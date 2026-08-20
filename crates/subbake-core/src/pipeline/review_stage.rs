use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use crate::entities::{
    PipelineOptions, ReviewAnnotation, ReviewChange, ReviewIssueKind, ReviewPolicy, ReviewReport,
    ReviewStats, SubtitleSegment, TranslationLine, Usage,
};
use crate::error::{CoreError, CoreResult};
use crate::memory::ContextMemory;
use crate::review::{
    ReviewBatchPlan, build_full_review_plan, build_review_plan, restore_review_progress,
};
use crate::validation::{FinalValidationPolicy, final_validation_issues};

pub(super) struct ReviewStage {
    plan: Vec<ReviewBatchPlan>,
    output: Vec<SubtitleSegment>,
    resumed: usize,
    completed: usize,
    usage: Usage,
    restored_cache_hits: usize,
    restored_duration_ms: u64,
    review_policy: ReviewPolicy,
    required_glossary: BTreeMap<String, String>,
    source_language: String,
    target_language: String,
    final_validation_policy: FinalValidationPolicy,
    annotations: BTreeMap<String, ReviewAnnotation>,
    cache_hits_before: usize,
    started: Instant,
}

pub(super) struct ReviewOutcome {
    pub output: Vec<SubtitleSegment>,
    pub stats: ReviewStats,
    pub changes: Vec<ReviewChange>,
}

pub(super) struct ReviewResumeInput<'a> {
    pub completed_batches: usize,
    pub reviewed_segments: &'a [SubtitleSegment],
    pub report: Option<&'a ReviewReport>,
    pub cache_hits_before: usize,
}

impl ReviewStage {
    pub fn new(
        options: &PipelineOptions,
        batches: &[Vec<SubtitleSegment>],
        translated: &[SubtitleSegment],
        memory: &ContextMemory,
        required_glossary: &BTreeMap<String, String>,
        resume: ReviewResumeInput<'_>,
    ) -> CoreResult<Self> {
        let ReviewResumeInput {
            completed_batches: resumed,
            reviewed_segments,
            report: restored_report,
            cache_hits_before,
        } = resume;
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
        let annotations = restored_report
            .map(|report| report.changes.as_slice())
            .unwrap_or_default()
            .iter()
            .filter_map(|change| {
                Some(ReviewAnnotation {
                    id: change.id.clone(),
                    category: change.category?,
                    rationale: change.rationale.clone()?,
                })
            })
            .map(|annotation| (annotation.id.clone(), annotation))
            .collect();
        Ok(Self {
            plan,
            output,
            resumed,
            completed: resumed,
            usage: restored_report
                .map(|report| report.review.usage)
                .unwrap_or_default(),
            restored_cache_hits: restored_report
                .map(|report| report.review.cache_hits)
                .unwrap_or_default(),
            restored_duration_ms: restored_report
                .map(|report| report.review.duration_ms)
                .unwrap_or_default(),
            review_policy: options.review_policy,
            required_glossary: required_glossary.clone(),
            source_language: options.source_language.clone(),
            target_language: options.target_language.clone(),
            final_validation_policy: FinalValidationPolicy {
                max_characters_per_second: options.max_characters_per_second,
                max_characters_per_line: options.max_characters_per_line,
                max_lines: options.max_lines,
            },
            annotations,
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
        annotations: &[ReviewAnnotation],
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
        let mut reviewed = super::support::apply_lines(&batch.source, lines);
        self.reject_unsafe_changes(batch, &mut reviewed)?;
        let accepted_ids = batch
            .translated
            .iter()
            .zip(&reviewed)
            .filter(|(before, after)| before.text != after.text)
            .map(|(before, _)| before.id.as_str())
            .collect::<HashSet<_>>();
        for annotation in annotations {
            if accepted_ids.contains(annotation.id.as_str()) {
                self.annotations
                    .insert(annotation.id.clone(), annotation.clone());
            }
        }
        if self.review_policy == ReviewPolicy::Targeted {
            for (before, after) in batch.translated.iter().zip(&reviewed) {
                if before.text != after.text && !self.annotations.contains_key(&before.id) {
                    let rationale = batch
                        .candidate_reasons
                        .get(&before.id)
                        .cloned()
                        .unwrap_or_default()
                        .join("; ");
                    self.annotations.insert(
                        before.id.clone(),
                        ReviewAnnotation {
                            id: before.id.clone(),
                            category: ReviewIssueKind::DeterministicRepair,
                            rationale,
                        },
                    );
                }
            }
        }
        self.output[batch.start_offset..batch.start_offset + reviewed.len()]
            .clone_from_slice(&reviewed);
        self.usage.add(usage);
        self.completed = self.completed.max(position);
        Ok(reviewed)
    }

    fn reject_unsafe_changes(
        &self,
        batch: &ReviewBatchPlan,
        reviewed: &mut [SubtitleSegment],
    ) -> CoreResult<()> {
        for ((source, before), after) in batch.source.iter().zip(&batch.translated).zip(reviewed) {
            if before.text == after.text {
                continue;
            }
            let before_issues = final_validation_issues(
                std::slice::from_ref(source),
                std::slice::from_ref(before),
                &self.required_glossary,
                &self.source_language,
                &self.target_language,
                self.final_validation_policy,
            )?;
            let existing = before_issues
                .iter()
                .map(|issue| issue.message.as_str())
                .collect::<HashSet<_>>();
            let after_issues = final_validation_issues(
                std::slice::from_ref(source),
                std::slice::from_ref(after),
                &self.required_glossary,
                &self.source_language,
                &self.target_language,
                self.final_validation_policy,
            )?;
            if after_issues
                .iter()
                .any(|issue| !existing.contains(issue.message.as_str()))
            {
                *after = before.clone();
            }
        }
        Ok(())
    }

    pub fn output(&self) -> &[SubtitleSegment] {
        &self.output
    }

    fn changes(&self) -> Vec<ReviewChange> {
        self.plan
            .iter()
            .flat_map(|batch| {
                let reviewed =
                    &self.output[batch.start_offset..batch.start_offset + batch.translated.len()];
                batch
                    .translated
                    .iter()
                    .zip(reviewed)
                    .filter(|(before, after)| before.text != after.text)
                    .map(|(before, after)| {
                        let annotation = self.annotations.get(&before.id);
                        ReviewChange {
                            batch: batch.batch_index,
                            id: before.id.clone(),
                            reasons: batch
                                .candidate_reasons
                                .get(&before.id)
                                .cloned()
                                .unwrap_or_default(),
                            before: before.text.clone(),
                            after: after.text.clone(),
                            category: annotation.map(|value| value.category),
                            rationale: annotation.map(|value| value.rationale.clone()),
                        }
                    })
            })
            .collect()
    }

    fn stats(&self, cache_hits: usize, changed_lines: usize) -> ReviewStats {
        let candidate_lines = self
            .plan
            .iter()
            .map(|batch| batch.candidate_reasons.len())
            .sum();
        let reviewed_lines = self
            .plan
            .iter()
            .take(self.completed)
            .map(|batch| batch.candidate_reasons.len())
            .sum();
        ReviewStats {
            candidate_lines,
            reviewed_lines,
            changed_lines,
            batches: self.completed,
            cache_hits: self
                .restored_cache_hits
                .saturating_add(cache_hits.saturating_sub(self.cache_hits_before)),
            usage: self.usage,
            duration_ms: self
                .restored_duration_ms
                .saturating_add(super::support::duration_ms(self.started)),
        }
    }

    pub fn snapshot(&self, cache_hits: usize) -> (ReviewStats, Vec<ReviewChange>) {
        let changes = self.changes();
        (self.stats(cache_hits, changes.len()), changes)
    }

    pub fn finish(self, cache_hits: usize) -> ReviewOutcome {
        let changes = self.changes();
        ReviewOutcome {
            stats: self.stats(cache_hits, changes.len()),
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
            &BTreeMap::new(),
            ReviewResumeInput {
                completed_batches: 0,
                reviewed_segments: &[],
                report: None,
                cache_hits_before: 0,
            },
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
                &[],
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

    #[test]
    fn targeted_review_derives_auditable_deterministic_reason() {
        let source = vec![segment("1", "Version 12")];
        let mut translated = source.clone();
        translated[0].text = "版本十三".to_owned();
        let batches = vec![source];
        let mut options = PipelineOptions::new("review.txt".into());
        options.review_policy = ReviewPolicy::Targeted;
        let mut stage = ReviewStage::new(
            &options,
            &batches,
            &translated,
            &ContextMemory::default(),
            &BTreeMap::new(),
            ReviewResumeInput {
                completed_batches: 0,
                reviewed_segments: &[],
                report: None,
                cache_hits_before: 0,
            },
        )
        .expect("review stage");

        stage
            .apply(
                1,
                &[TranslationLine {
                    id: "1".to_owned(),
                    translation: "版本12".to_owned(),
                }],
                &[],
                Usage::default(),
            )
            .expect("apply targeted repair");
        let (_, changes) = stage.snapshot(0);

        assert_eq!(
            changes[0].category,
            Some(ReviewIssueKind::DeterministicRepair)
        );
        assert!(
            changes[0]
                .rationale
                .as_deref()
                .is_some_and(|value| value.contains("number mismatch"))
        );
    }

    #[test]
    fn review_rejects_a_new_required_glossary_violation() {
        let source = vec![segment("1", "Rick is here")];
        let mut translated = source.clone();
        translated[0].text = "Rick来了".to_owned();
        let batches = vec![source];
        let mut options = PipelineOptions::new("review.txt".into());
        options.review_policy = ReviewPolicy::Full;
        let required_glossary = BTreeMap::from([("Rick".to_owned(), "Rick".to_owned())]);
        let mut stage = ReviewStage::new(
            &options,
            &batches,
            &translated,
            &ContextMemory::default(),
            &required_glossary,
            ReviewResumeInput {
                completed_batches: 0,
                reviewed_segments: &[],
                report: None,
                cache_hits_before: 0,
            },
        )
        .expect("review stage");

        let reviewed = stage
            .apply(
                1,
                &[TranslationLine {
                    id: "1".to_owned(),
                    translation: "瑞克来了".to_owned(),
                }],
                &[ReviewAnnotation {
                    id: "1".to_owned(),
                    category: ReviewIssueKind::Terminology,
                    rationale: "localize the name".to_owned(),
                }],
                Usage::default(),
            )
            .expect("apply safe review filter");

        assert_eq!(reviewed[0].text, "Rick来了");
        assert!(stage.snapshot(0).1.is_empty());
    }

    #[test]
    fn review_rejects_a_new_factual_token_violation() {
        let source = vec![segment("1", "Wait 2 minutes")];
        let mut translated = source.clone();
        translated[0].text = "等2分钟".to_owned();
        let batches = vec![source];
        let mut options = PipelineOptions::new("review.txt".into());
        options.review_policy = ReviewPolicy::Full;
        let mut stage = ReviewStage::new(
            &options,
            &batches,
            &translated,
            &ContextMemory::default(),
            &BTreeMap::new(),
            ReviewResumeInput {
                completed_batches: 0,
                reviewed_segments: &[],
                report: None,
                cache_hits_before: 0,
            },
        )
        .expect("review stage");

        let reviewed = stage
            .apply(
                1,
                &[TranslationLine {
                    id: "1".to_owned(),
                    translation: "等3分钟".to_owned(),
                }],
                &[ReviewAnnotation {
                    id: "1".to_owned(),
                    category: ReviewIssueKind::Accuracy,
                    rationale: "scripted factual regression".to_owned(),
                }],
                Usage::default(),
            )
            .expect("apply safe review filter");

        assert_eq!(reviewed[0].text, "等2分钟");
        assert!(stage.snapshot(0).1.is_empty());
    }

    #[test]
    fn review_rejects_a_new_readability_violation() {
        let mut source = vec![segment("1", "Come here")];
        source[0].start = Some("00:00:00,000".to_owned());
        source[0].end = Some("00:00:01,000".to_owned());
        let mut translated = source.clone();
        translated[0].text = "过来".to_owned();
        let batches = vec![source];
        let mut options = PipelineOptions::new("review.srt".into());
        options.review_policy = ReviewPolicy::Full;
        options.max_characters_per_second = Some(4.0);
        let mut stage = ReviewStage::new(
            &options,
            &batches,
            &translated,
            &ContextMemory::default(),
            &BTreeMap::new(),
            ReviewResumeInput {
                completed_batches: 0,
                reviewed_segments: &[],
                report: None,
                cache_hits_before: 0,
            },
        )
        .expect("review stage");

        let reviewed = stage
            .apply(
                1,
                &[TranslationLine {
                    id: "1".to_owned(),
                    translation: "请马上过来".to_owned(),
                }],
                &[ReviewAnnotation {
                    id: "1".to_owned(),
                    category: ReviewIssueKind::Fluency,
                    rationale: "scripted readability regression".to_owned(),
                }],
                Usage::default(),
            )
            .expect("apply safe review filter");

        assert_eq!(reviewed[0].text, "过来");
        assert!(stage.snapshot(0).1.is_empty());
    }
}
