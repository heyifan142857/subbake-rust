//! Deterministic transcription metrics for subtitle-shaped ASR output.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::SubtitleDocument;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptionEvaluationReport {
    pub version: u64,
    pub reference_words: usize,
    pub word_errors: usize,
    pub wer: f64,
    pub reference_characters: usize,
    pub character_errors: usize,
    pub cer: f64,
    pub matched_segments: usize,
    pub missing_reference_segments: usize,
    pub mean_start_offset_ms: f64,
    pub mean_end_offset_ms: f64,
    pub max_boundary_offset_ms: f64,
    pub speech_coverage: f64,
    pub overlapping_segments: usize,
    pub overlap_seconds: f64,
    pub max_characters_per_second: f64,
    pub max_characters_per_line: usize,
}

/// Compare ASR output against a reference transcript and timeline.
///
/// WER uses Unicode words split on whitespace; CER removes whitespace so it
/// remains useful for languages such as Chinese where word segmentation is
/// not intrinsic. Timing offsets use stable segment IDs, while speech coverage
/// and overlap use interval unions and do not depend on IDs.
pub fn evaluate_transcription(
    candidate: &SubtitleDocument,
    reference: &SubtitleDocument,
) -> TranscriptionEvaluationReport {
    let reference_words = word_tokens(&document_text(reference));
    let candidate_words = word_tokens(&document_text(candidate));
    let reference_characters = character_tokens(&document_text(reference));
    let candidate_characters = character_tokens(&document_text(candidate));
    let word_errors = levenshtein(&candidate_words, &reference_words);
    let character_errors = levenshtein(&candidate_characters, &reference_characters);

    let candidate_by_id = candidate
        .segments
        .iter()
        .map(|segment| (segment.id.as_str(), segment))
        .collect::<BTreeMap<_, _>>();
    let mut start_offsets = Vec::new();
    let mut end_offsets = Vec::new();
    let mut missing_reference_segments = 0;
    for reference_segment in &reference.segments {
        let Some(candidate_segment) = candidate_by_id.get(reference_segment.id.as_str()) else {
            missing_reference_segments += 1;
            continue;
        };
        if let (Some((reference_start, reference_end)), Some((candidate_start, candidate_end))) = (
            segment_interval(reference_segment),
            segment_interval(candidate_segment),
        ) {
            start_offsets.push((candidate_start - reference_start).abs());
            end_offsets.push((candidate_end - reference_end).abs());
        }
    }

    let reference_intervals = merged_intervals(reference);
    let candidate_intervals = merged_intervals(candidate);
    let reference_duration = total_duration(&reference_intervals);
    let covered_duration = intersection_duration(&reference_intervals, &candidate_intervals);
    let (overlapping_segments, overlap_seconds) = overlap_metrics(candidate);
    let (max_characters_per_second, max_characters_per_line) = readability_metrics(candidate);
    let max_boundary_offset = start_offsets
        .iter()
        .chain(&end_offsets)
        .copied()
        .fold(0.0, f64::max);

    TranscriptionEvaluationReport {
        version: 1,
        reference_words: reference_words.len(),
        word_errors,
        wer: error_rate(word_errors, reference_words.len()),
        reference_characters: reference_characters.len(),
        character_errors,
        cer: error_rate(character_errors, reference_characters.len()),
        matched_segments: reference.segments.len() - missing_reference_segments,
        missing_reference_segments,
        mean_start_offset_ms: mean(&start_offsets) * 1_000.0,
        mean_end_offset_ms: mean(&end_offsets) * 1_000.0,
        max_boundary_offset_ms: max_boundary_offset * 1_000.0,
        speech_coverage: if reference_duration > 0.0 {
            covered_duration / reference_duration
        } else {
            1.0
        },
        overlapping_segments,
        overlap_seconds,
        max_characters_per_second,
        max_characters_per_line,
    }
}

fn document_text(document: &SubtitleDocument) -> String {
    document
        .segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn word_tokens(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.trim_matches(|character: char| !character.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn character_tokens(text: &str) -> Vec<char> {
    text.chars()
        .filter(|character| !character.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn levenshtein<T: Eq>(candidate: &[T], reference: &[T]) -> usize {
    let mut previous = (0..=reference.len()).collect::<Vec<_>>();
    let mut current = vec![0; reference.len() + 1];
    for (candidate_index, candidate_item) in candidate.iter().enumerate() {
        current[0] = candidate_index + 1;
        for (reference_index, reference_item) in reference.iter().enumerate() {
            let substitution =
                previous[reference_index] + usize::from(candidate_item != reference_item);
            current[reference_index + 1] = substitution
                .min(previous[reference_index + 1] + 1)
                .min(current[reference_index] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[reference.len()]
}

fn error_rate(errors: usize, reference_units: usize) -> f64 {
    if reference_units == 0 {
        f64::from(errors > 0)
    } else {
        errors as f64 / reference_units as f64
    }
}

fn parse_timestamp(value: &str) -> Option<f64> {
    let normalized = value.trim().replace(',', ".");
    let parts = normalized.split(':').collect::<Vec<_>>();
    match parts.as_slice() {
        [hours, minutes, seconds] => Some(
            hours.parse::<f64>().ok()? * 3_600.0
                + minutes.parse::<f64>().ok()? * 60.0
                + seconds.parse::<f64>().ok()?,
        ),
        [minutes, seconds] => {
            Some(minutes.parse::<f64>().ok()? * 60.0 + seconds.parse::<f64>().ok()?)
        }
        _ => None,
    }
}

fn segment_interval(segment: &crate::SubtitleSegment) -> Option<(f64, f64)> {
    let (start, end) = segment.start.as_deref().zip(segment.end.as_deref())?;
    let interval = (parse_timestamp(start)?, parse_timestamp(end)?);
    (interval.1 > interval.0).then_some(interval)
}

fn merged_intervals(document: &SubtitleDocument) -> Vec<(f64, f64)> {
    let mut intervals = document
        .segments
        .iter()
        .filter_map(segment_interval)
        .collect::<Vec<_>>();
    intervals.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (start, end) in intervals {
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

fn total_duration(intervals: &[(f64, f64)]) -> f64 {
    intervals.iter().map(|(start, end)| end - start).sum()
}

fn intersection_duration(left: &[(f64, f64)], right: &[(f64, f64)]) -> f64 {
    let mut left_index = 0;
    let mut right_index = 0;
    let mut duration = 0.0;
    while left_index < left.len() && right_index < right.len() {
        let start = left[left_index].0.max(right[right_index].0);
        let end = left[left_index].1.min(right[right_index].1);
        if end > start {
            duration += end - start;
        }
        if left[left_index].1 < right[right_index].1 {
            left_index += 1;
        } else {
            right_index += 1;
        }
    }
    duration
}

fn overlap_metrics(document: &SubtitleDocument) -> (usize, f64) {
    let mut intervals = document
        .segments
        .iter()
        .filter_map(segment_interval)
        .collect::<Vec<_>>();
    intervals.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut maximum_end: Option<f64> = None;
    let mut count = 0;
    let mut seconds = 0.0;
    for (start, end) in intervals {
        if let Some(previous_end) = maximum_end
            && start < previous_end
        {
            count += 1;
            seconds += previous_end.min(end) - start;
        }
        maximum_end = Some(maximum_end.map_or(end, |value| value.max(end)));
    }
    (count, seconds)
}

fn readability_metrics(document: &SubtitleDocument) -> (f64, usize) {
    let mut max_cps: f64 = 0.0;
    let mut max_line = 0;
    for segment in &document.segments {
        let characters = visible_characters(&segment.text);
        max_line = max_line.max(
            segment
                .text
                .lines()
                .map(visible_characters)
                .max()
                .unwrap_or(0),
        );
        if let Some((start, end)) = segment_interval(segment) {
            max_cps = max_cps.max(characters as f64 / (end - start));
        }
    }
    (max_cps, max_line)
}

fn visible_characters(text: &str) -> usize {
    text.chars()
        .filter(|character| !character.is_whitespace())
        .count()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{SubtitleDocumentMetadata, SubtitleSegment};

    use super::*;

    fn document(lines: &[(&str, &str, &str, &str)]) -> SubtitleDocument {
        SubtitleDocument {
            path: PathBuf::from("sample.srt"),
            format: "srt".to_owned(),
            segments: lines
                .iter()
                .map(|(id, text, start, end)| SubtitleSegment {
                    id: (*id).to_owned(),
                    text: (*text).to_owned(),
                    start: Some((*start).to_owned()),
                    end: Some((*end).to_owned()),
                    identifier: None,
                    settings: None,
                    semantic: Default::default(),
                })
                .collect(),
            header: None,
            passthrough_blocks: Vec::new(),
            metadata: SubtitleDocumentMetadata::None,
        }
    }

    #[test]
    fn reports_text_timing_coverage_overlap_and_readability_metrics() {
        let reference = document(&[
            ("1", "hello world", "00:00:00,000", "00:00:02,000"),
            ("2", "again", "00:00:02,000", "00:00:04,000"),
        ]);
        let candidate = document(&[
            ("1", "hello word", "00:00:00,100", "00:00:02,100"),
            ("2", "again", "00:00:01,900", "00:00:03,800"),
        ]);

        let report = evaluate_transcription(&candidate, &reference);
        assert_eq!(report.word_errors, 1);
        assert!((report.wer - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(report.character_errors, 1);
        assert!((report.cer - 1.0 / 15.0).abs() < 1e-9);
        assert_eq!(report.overlapping_segments, 1);
        assert!((report.overlap_seconds - 0.2).abs() < 1e-9);
        assert!((report.speech_coverage - 0.925).abs() < 1e-9);
        assert!((report.max_boundary_offset_ms - 200.0).abs() < 1e-9);
        assert_eq!(report.max_characters_per_line, 9);
    }
}
