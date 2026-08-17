use std::collections::HashMap;

use crate::entities::{BatchPlanEntry, SubtitleSegment, TranslationMode};
use crate::error::{CoreError, CoreResult};

use super::support::contextual_translation_memory_keys;
use super::translation_stage::SourceBatchContext;

const TURBO_CONTEXT_LINES: usize = 3;
const MIN_CINEMA_CONTEXT_TOKENS: usize = 128;
const MAX_CINEMA_CONTEXT_TOKENS: usize = 1_200;
const HARD_SCENE_GAP_MS: usize = 1_500;
const SOFT_SCENE_GAP_MS: usize = 900;

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
        let context_keys = contextual_translation_memory_keys("", segments);
        let mut first_id_by_context = HashMap::<String, String>::new();
        let mut canonical = Vec::new();
        let mut canonical_id_by_segment = Vec::with_capacity(segments.len());
        for segment in segments {
            let key = context_keys.get(&segment.id).cloned().unwrap_or_default();
            if let Some(id) = first_id_by_context.get(&key) {
                canonical_id_by_segment.push(id.clone());
            } else {
                first_id_by_context.insert(key, segment.id.clone());
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

    pub(super) fn scene_group_ids(batches: &[Vec<SubtitleSegment>]) -> Vec<usize> {
        let mut group = 0usize;
        batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                if index > 0
                    && let (Some(previous), Some(next)) = (batches[index - 1].last(), batch.first())
                    && scene_boundary(Some(previous), next)
                {
                    group = group.saturating_add(1);
                }
                group
            })
            .collect()
    }

    pub(super) fn source_contexts(
        segments: &[SubtitleSegment],
        batches: &[Vec<SubtitleSegment>],
        mode: TranslationMode,
        batch_token_budget: usize,
    ) -> Vec<SourceBatchContext> {
        let mut offset = 0usize;
        batches
            .iter()
            .map(|batch| {
                let start = offset;
                let end = start.saturating_add(batch.len()).min(segments.len());
                offset = end;
                match mode {
                    TranslationMode::Economy => SourceBatchContext::default(),
                    TranslationMode::Turbo => SourceBatchContext {
                        before: segments[start.saturating_sub(TURBO_CONTEXT_LINES)..start].to_vec(),
                        after: segments
                            [end..end.saturating_add(TURBO_CONTEXT_LINES).min(segments.len())]
                            .to_vec(),
                    },
                    TranslationMode::Cinema => cinema_source_context(
                        segments,
                        start,
                        end,
                        cinema_context_budget(batch_token_budget),
                    ),
                }
            })
            .collect()
    }
}

fn cinema_source_context(
    segments: &[SubtitleSegment],
    start: usize,
    end: usize,
    token_budget: usize,
) -> SourceBatchContext {
    if segments.is_empty() || start >= end {
        return SourceBatchContext::default();
    }
    let current_scene_start = scene_start(segments, start);
    let current_scene_end = scene_end(segments, end);
    let before_start = if current_scene_start < start {
        current_scene_start
    } else {
        scene_start(segments, current_scene_start.saturating_sub(1))
    };
    let after_end = if end < current_scene_end {
        current_scene_end
    } else if current_scene_end < segments.len() {
        scene_end(segments, current_scene_end + 1)
    } else {
        current_scene_end
    };
    let before_budget = token_budget.div_ceil(2);
    let after_budget = token_budget.saturating_sub(before_budget);
    SourceBatchContext {
        before: take_context_tail(&segments[before_start..start], before_budget),
        after: take_context_head(&segments[end..after_end], after_budget),
    }
}

fn scene_start(segments: &[SubtitleSegment], index: usize) -> usize {
    let mut start = index.min(segments.len().saturating_sub(1));
    while start > 0 && !scene_boundary(segments.get(start - 1), &segments[start]) {
        start -= 1;
    }
    start
}

fn scene_end(segments: &[SubtitleSegment], index: usize) -> usize {
    let mut end = index.min(segments.len());
    while end < segments.len() && !scene_boundary(segments.get(end - 1), &segments[end]) {
        end += 1;
    }
    end
}

fn cinema_context_budget(batch_token_budget: usize) -> usize {
    batch_token_budget
        .checked_div(2)
        .unwrap_or_default()
        .clamp(MIN_CINEMA_CONTEXT_TOKENS, MAX_CINEMA_CONTEXT_TOKENS)
}

fn take_context_tail(segments: &[SubtitleSegment], token_budget: usize) -> Vec<SubtitleSegment> {
    let mut selected = Vec::new();
    let mut tokens = 0usize;
    for segment in segments.iter().rev() {
        let estimate = estimated_text_tokens(&segment.text).saturating_add(4);
        if !selected.is_empty() && tokens.saturating_add(estimate) > token_budget {
            break;
        }
        selected.push(segment.clone());
        tokens = tokens.saturating_add(estimate);
    }
    selected.reverse();
    selected
}

fn take_context_head(segments: &[SubtitleSegment], token_budget: usize) -> Vec<SubtitleSegment> {
    let mut selected = Vec::new();
    let mut tokens = 0usize;
    for segment in segments {
        let estimate = estimated_text_tokens(&segment.text).saturating_add(4);
        if !selected.is_empty() && tokens.saturating_add(estimate) > token_budget {
            break;
        }
        selected.push(segment.clone());
        tokens = tokens.saturating_add(estimate);
    }
    selected
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
    let gap = start.saturating_sub(end);
    gap >= HARD_SCENE_GAP_MS
        || (gap >= SOFT_SCENE_GAP_MS
            && ends_complete_utterance(&previous.text)
            && starts_new_utterance(&next.text))
}

fn ends_complete_utterance(text: &str) -> bool {
    text.trim_end_matches(|character: char| {
        character.is_whitespace()
            || matches!(
                character,
                '"' | '\'' | '”' | '’' | '」' | '』' | ')' | ']' | '}'
            )
    })
    .chars()
    .last()
    .is_some_and(|character| matches!(character, '.' | '!' | '?' | '。' | '！' | '？' | '…'))
}

fn starts_new_utterance(text: &str) -> bool {
    let text = text.trim_start();
    let first = text
        .trim_start_matches(|character: char| {
            matches!(
                character,
                '-' | '—' | '"' | '\'' | '“' | '‘' | '(' | '[' | '{'
            )
        })
        .chars()
        .next();
    first.is_some_and(|character| {
        character.is_uppercase()
            || matches!(
                character,
                '\u{3040}'..='\u{30ff}'
                    | '\u{3400}'..='\u{9fff}'
                    | '\u{ac00}'..='\u{d7af}'
            )
    })
}

fn subtitle_timestamp_ms(value: &str) -> Option<usize> {
    let value = value.trim().replace(',', ".");
    let (clock, fraction) = value.rsplit_once('.')?;
    let mut parts = clock.split(':').map(str::parse::<usize>);
    let hours = parts.next()?.ok()?;
    let minutes = parts.next()?.ok()?;
    let seconds = parts.next()?.ok()?;
    if parts.next().is_some() {
        return None;
    }
    if fraction.is_empty()
        || fraction.len() > 3
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let scale = match fraction.len() {
        1 => 100,
        2 => 10,
        3 => 1,
        _ => return None,
    };
    let milliseconds = fraction.parse::<usize>().ok()?.saturating_mul(scale);
    Some((((hours * 60 + minutes) * 60 + seconds) * 1_000) + milliseconds)
}

pub(super) fn estimated_text_tokens(text: &str) -> usize {
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
            semantic: Default::default(),
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
        let source = vec![
            segment("1", "Start"),
            segment("2", "Again"),
            segment("3", "End"),
            segment("4", "Start"),
            segment("5", "Again"),
            segment("6", "End"),
        ];
        let plan = DeduplicationPlan::new(&source, true);
        assert_eq!(plan.canonical().len(), 5);
        assert_eq!(plan.duplicates(), 1);
        let translated = plan
            .canonical()
            .iter()
            .map(|item| {
                if item.id == "2" {
                    segment("2", "再来一次")
                } else {
                    item.clone()
                }
            })
            .collect::<Vec<_>>();
        let expanded = plan.expand(&source, &translated).expect("expand");
        assert_eq!(expanded[4].id, "5");
        assert_eq!(expanded[4].text, "再来一次");
    }

    #[test]
    fn deduplication_keeps_identical_text_in_different_contexts() {
        let source = vec![
            segment("1", "He paid the fee."),
            segment("2", "Fine."),
            segment("3", "We can leave."),
            segment("4", "The weather is clear."),
            segment("5", "Fine."),
            segment("6", "We should stay."),
        ];

        let plan = DeduplicationPlan::new(&source, true);

        assert_eq!(plan.canonical(), source);
        assert_eq!(plan.duplicates(), 0);
    }

    #[test]
    fn deduplication_keeps_identical_text_with_different_semantic_metadata() {
        let mut source = vec![
            segment("1", "Start"),
            segment("2", "Open"),
            segment("3", "End"),
            segment("4", "Start"),
            segment("5", "Open"),
            segment("6", "End"),
        ];
        source[1].semantic.speaker = Some("Alice".to_owned());
        source[1].semantic.style = Some("Default".to_owned());
        source[4].semantic.style = Some("Sign".to_owned());

        let plan = DeduplicationPlan::new(&source, true);

        assert_eq!(plan.canonical(), source);
        assert_eq!(plan.duplicates(), 0);
    }

    #[test]
    fn turbo_context_uses_fixed_neighboring_source_lines() {
        let source = (1..=8)
            .map(|id| segment(&id.to_string(), &format!("line {id}")))
            .collect::<Vec<_>>();
        let batches = source
            .chunks(2)
            .map(<[SubtitleSegment]>::to_vec)
            .collect::<Vec<_>>();

        let contexts =
            BatchPlanner::source_contexts(&source, &batches, TranslationMode::Turbo, 1_800);

        assert_eq!(ids(&contexts[1].before), ["1", "2"]);
        assert_eq!(ids(&contexts[1].after), ["5", "6", "7"]);
    }

    #[test]
    fn cinema_context_uses_adjacent_scene_blocks() {
        let source = vec![
            timed_segment("1", "scene one a", "00:00:00,000", "00:00:01,000"),
            timed_segment("2", "scene one b", "00:00:01,100", "00:00:02,000"),
            timed_segment("3", "scene two a", "00:00:04,000", "00:00:05,000"),
            timed_segment("4", "scene two b", "00:00:05,100", "00:00:06,000"),
            timed_segment("5", "scene three", "00:00:08,000", "00:00:09,000"),
        ];
        let batches = vec![
            source[0..2].to_vec(),
            source[2..4].to_vec(),
            source[4..].to_vec(),
        ];

        let contexts =
            BatchPlanner::source_contexts(&source, &batches, TranslationMode::Cinema, 1_600);

        assert_eq!(ids(&contexts[1].before), ["1", "2"]);
        assert_eq!(ids(&contexts[1].after), ["5"]);
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

    #[test]
    fn scene_groups_keep_token_splits_serial_and_separate_timed_scenes() {
        let batches = vec![
            vec![timed_segment(
                "1",
                "scene one a",
                "00:00:00,000",
                "00:00:01,000",
            )],
            vec![timed_segment(
                "2",
                "scene one b",
                "00:00:01,100",
                "00:00:02,000",
            )],
            vec![timed_segment(
                "3",
                "scene two",
                "00:00:04,000",
                "00:00:05,000",
            )],
        ];

        assert_eq!(BatchPlanner::scene_group_ids(&batches), [0, 0, 1]);
    }

    #[test]
    fn scene_aware_planner_uses_sentence_cues_for_medium_timing_gaps() {
        let source = vec![
            timed_segment(
                "1",
                "The conversation is over.",
                "00:00:00,000",
                "00:00:01,000",
            ),
            timed_segment("2", "A new morning begins.", "00:00:01,950", "00:00:03,000"),
        ];

        let batches = BatchPlanner::new(10, 1_000)
            .scene_aware(true)
            .split(&source);

        assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), [1, 1]);
    }

    #[test]
    fn scene_aware_planner_keeps_short_continuations_together() {
        let source = vec![
            timed_segment("1", "Wait", "00:00:00,000", "00:00:01,000"),
            timed_segment("2", "for me.", "00:00:01,950", "00:00:03,000"),
        ];

        let batches = BatchPlanner::new(10, 1_000)
            .scene_aware(true)
            .split(&source);

        assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), [2]);
    }

    #[test]
    fn scene_aware_planner_scales_ass_centiseconds_before_comparing_gaps() {
        let source = vec![
            timed_segment("1", "The thought is complete.", "0:00:02.00", "0:00:03.90"),
            timed_segment("2", "Another begins.", "0:00:04.00", "0:00:05.00"),
        ];

        let batches = BatchPlanner::new(10, 1_000)
            .scene_aware(true)
            .split(&source);

        assert_eq!(subtitle_timestamp_ms("0:00:03.9"), Some(3_900));
        assert_eq!(subtitle_timestamp_ms("0:00:03.90"), Some(3_900));
        assert_eq!(subtitle_timestamp_ms("00:00:03,900"), Some(3_900));
        assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), [2]);
    }

    fn timed_segment(id: &str, text: &str, start: &str, end: &str) -> SubtitleSegment {
        let mut segment = segment(id, text);
        segment.start = Some(start.to_owned());
        segment.end = Some(end.to_owned());
        segment
    }

    fn ids(segments: &[SubtitleSegment]) -> Vec<&str> {
        segments.iter().map(|segment| segment.id.as_str()).collect()
    }
}
