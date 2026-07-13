use crate::entities::{BatchPlanEntry, SubtitleSegment};

pub(super) struct BatchPlanner {
    max_batch_size: usize,
    token_budget: usize,
}

impl BatchPlanner {
    pub(super) fn new(max_batch_size: usize, token_budget: usize) -> Self {
        Self {
            max_batch_size,
            token_budget,
        }
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
                && (current.len() >= self.max_batch_size
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
}
