use std::collections::HashMap;

use crate::entities::{BatchPlanEntry, SubtitleSegment};
use crate::error::{CoreError, CoreResult};

pub(super) struct DeduplicationPlan {
    canonical: Vec<SubtitleSegment>,
    canonical_id_by_segment: Vec<String>,
    duplicates: usize,
}

impl DeduplicationPlan {
    pub fn new(segments: &[SubtitleSegment], enabled: bool) -> Self {
        if !enabled {
            return Self {
                canonical: segments.to_vec(),
                canonical_id_by_segment: segments.iter().map(|line| line.id.clone()).collect(),
                duplicates: 0,
            };
        }
        let mut first_id_by_text = HashMap::<String, String>::new();
        let mut canonical = Vec::new();
        let mut canonical_id_by_segment = Vec::with_capacity(segments.len());
        for segment in segments {
            let key = segment.text.trim().to_owned();
            if let Some(id) = first_id_by_text.get(&key) {
                canonical_id_by_segment.push(id.clone());
            } else {
                first_id_by_text.insert(key, segment.id.clone());
                canonical_id_by_segment.push(segment.id.clone());
                canonical.push(segment.clone());
            }
        }
        let duplicates = segments.len().saturating_sub(canonical.len());
        Self {
            canonical,
            canonical_id_by_segment,
            duplicates,
        }
    }

    pub fn canonical(&self) -> &[SubtitleSegment] {
        &self.canonical
    }

    pub fn duplicates(&self) -> usize {
        self.duplicates
    }

    pub fn expand(
        &self,
        source: &[SubtitleSegment],
        translated: &[SubtitleSegment],
    ) -> CoreResult<Vec<SubtitleSegment>> {
        let translations = translated
            .iter()
            .map(|line| (line.id.as_str(), line.text.as_str()))
            .collect::<HashMap<_, _>>();
        source
            .iter()
            .zip(&self.canonical_id_by_segment)
            .map(|(source, canonical_id)| {
                let text = translations.get(canonical_id.as_str()).ok_or_else(|| {
                    CoreError::DataInvariant(format!(
                        "deduplication result omitted canonical id `{canonical_id}`"
                    ))
                })?;
                let mut output = source.clone();
                output.text = (*text).to_owned();
                Ok(output)
            })
            .collect()
    }
}

pub(super) struct BatchPlanner {
    max_batch_size: usize,
    token_budget: usize,
    scene_aware: bool,
}

impl BatchPlanner {
    pub(super) fn new(max_batch_size: usize, token_budget: usize) -> Self {
        Self {
            max_batch_size,
            token_budget,
            scene_aware: false,
        }
    }

    pub(super) fn scene_aware(mut self, enabled: bool) -> Self {
        self.scene_aware = enabled;
        self
    }

    pub(super) fn split(&self, segments: &[SubtitleSegment]) -> Vec<Vec<SubtitleSegment>> {
        if self.token_budget == 0 {
            return segments
                .chunks(self.max_batch_size)
                .map(<[SubtitleSegment]>::to_vec)
                .collect();
        }

        let mut batches = Vec::new();
        let mut current = Vec::new();
        let mut tokens = 0usize;
        for segment in segments {
            let estimate = estimated_text_tokens(&segment.text).saturating_add(8);
            if !current.is_empty()
                && ((self.scene_aware && scene_boundary(current.last(), segment))
                    || current.len() >= self.max_batch_size
                    || tokens.saturating_add(estimate) > self.token_budget)
            {
                batches.push(std::mem::take(&mut current));
                tokens = 0;
            }
            current.push(segment.clone());
            tokens = tokens.saturating_add(estimate);
        }
        if !current.is_empty() {
            batches.push(current);
        }
        batches
    }

    pub(super) fn describe(batches: &[Vec<SubtitleSegment>]) -> Vec<BatchPlanEntry> {
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
}

fn scene_boundary(previous: Option<&SubtitleSegment>, next: &SubtitleSegment) -> bool {
    let Some(previous) = previous else {
        return false;
    };
    let (Some(end), Some(start)) = (previous.end.as_deref(), next.start.as_deref()) else {
        return false;
    };
    let (Some(end), Some(start)) = (subtitle_timestamp_ms(end), subtitle_timestamp_ms(start))
    else {
        return false;
    };
    start.saturating_sub(end) >= 1_500
}

fn subtitle_timestamp_ms(value: &str) -> Option<usize> {
    let value = value.trim().replace(',', ".");
    let (clock, milliseconds) = value.rsplit_once('.')?;
    let mut parts = clock.split(':').map(str::parse::<usize>);
    let hours = parts.next()?.ok()?;
    let minutes = parts.next()?.ok()?;
    let seconds = parts.next()?.ok()?;
    if parts.next().is_some() {
        return None;
    }
    let milliseconds = milliseconds.parse::<usize>().ok()?;
    Some((((hours * 60 + minutes) * 60 + seconds) * 1_000) + milliseconds)
}

fn estimated_text_tokens(text: &str) -> usize {
    let (ascii, non_ascii) = text.chars().fold((0usize, 0usize), |(ascii, other), ch| {
        if ch.is_ascii() {
            (ascii + 1, other)
        } else {
            (ascii, other + 1)
        }
    });
    ascii.div_ceil(4).saturating_add(non_ascii)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn planner_respects_size_and_token_boundaries() {
        let segments = vec![
            segment("1", "12345678"),
            segment("2", "12345678"),
            segment("3", "12345678"),
        ];
        let batches = BatchPlanner::new(2, 20).split(&segments);
        assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), [2, 1]);
        assert_eq!(BatchPlanner::describe(&batches)[1].first_id, "3");
    }

    #[test]
    fn deduplication_translates_once_and_restores_original_ids() {
        let source = vec![segment("1", "Again"), segment("2", "Again")];
        let plan = DeduplicationPlan::new(&source, true);
        assert_eq!(plan.canonical().len(), 1);
        assert_eq!(plan.duplicates(), 1);
        let translated = vec![segment("1", "再来一次")];
        let expanded = plan.expand(&source, &translated).expect("expand");
        assert_eq!(expanded[1].id, "2");
        assert_eq!(expanded[1].text, "再来一次");
    }

    #[test]
    fn scene_aware_planner_splits_on_large_timing_gaps() {
        let mut first = segment("1", "First.");
        first.end = Some("00:00:01,000".to_owned());
        let mut second = segment("2", "Second.");
        second.start = Some("00:00:03,000".to_owned());
        let batches = BatchPlanner::new(10, 1_000)
            .scene_aware(true)
            .split(&[first, second]);
        assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), [1, 1]);
    }
}
