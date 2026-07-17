//! Deterministic offline translation evaluation for regression tracking.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::entities::SubtitleDocument;
use crate::error::{CoreError, CoreResult};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub segments: usize,
    pub exact_matches: usize,
    pub chrf: f64,
    pub mqm: MqmCounts,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MqmCounts {
    pub critical: usize,
    pub major: usize,
    pub minor: usize,
}

/// Compare a produced subtitle against a reference using stable identifiers.
/// chrF uses character 1–6 gram F-score with beta=2; MQM counts are explicit
/// mechanical guards, not a claim of semantic human evaluation.
pub fn evaluate(
    candidate: &SubtitleDocument,
    reference: &SubtitleDocument,
) -> CoreResult<EvaluationReport> {
    let candidate_by_id = candidate
        .segments
        .iter()
        .map(|line| (line.id.as_str(), line.text.as_str()))
        .collect::<BTreeMap<_, _>>();
    if candidate_by_id.len() != candidate.segments.len() {
        return Err(CoreError::DataInvariant(
            "candidate subtitle has duplicate ids".to_owned(),
        ));
    }
    let mut exact_matches = 0;
    let mut candidate_text = String::new();
    let mut reference_text = String::new();
    let mut mqm = MqmCounts::default();
    for reference_line in &reference.segments {
        let Some(candidate_line) = candidate_by_id.get(reference_line.id.as_str()) else {
            mqm.critical += 1;
            continue;
        };
        if normalize(candidate_line) == normalize(&reference_line.text) {
            exact_matches += 1;
        }
        if candidate_line.trim().is_empty() {
            mqm.major += 1;
        }
        if number_tokens(candidate_line) != number_tokens(&reference_line.text) {
            mqm.major += 1;
        }
        if formatting_tokens(candidate_line) != formatting_tokens(&reference_line.text) {
            mqm.minor += 1;
        }
        candidate_text.push_str(candidate_line);
        candidate_text.push('\n');
        reference_text.push_str(&reference_line.text);
        reference_text.push('\n');
    }
    for candidate_line in &candidate.segments {
        if !reference
            .segments
            .iter()
            .any(|line| line.id == candidate_line.id)
        {
            mqm.critical += 1;
        }
    }
    Ok(EvaluationReport {
        segments: reference.segments.len(),
        exact_matches,
        chrf: chrf(&candidate_text, &reference_text),
        mqm,
    })
}

fn chrf(candidate: &str, reference: &str) -> f64 {
    let mut precision = 0.0;
    let mut recall = 0.0;
    let mut used = 0.0;
    for n in 1..=6 {
        let candidate = grams(candidate, n);
        let reference = grams(reference, n);
        if candidate.is_empty() || reference.is_empty() {
            continue;
        }
        let overlap = candidate
            .iter()
            .map(|(gram, count)| count.min(reference.get(gram).unwrap_or(&0)))
            .sum::<usize>() as f64;
        precision += overlap / candidate.values().sum::<usize>() as f64;
        recall += overlap / reference.values().sum::<usize>() as f64;
        used += 1.0;
    }
    if used == 0.0 {
        return 0.0;
    }
    let precision = precision / used;
    let recall = recall / used;
    if precision + recall == 0.0 {
        0.0
    } else {
        5.0 * precision * recall / (4.0 * precision + recall)
    }
}

fn grams(text: &str, n: usize) -> BTreeMap<String, usize> {
    let chars = text
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<Vec<_>>();
    chars
        .windows(n)
        .map(|window| window.iter().collect::<String>())
        .fold(BTreeMap::new(), |mut counts, gram| {
            *counts.entry(gram).or_default() += 1;
            counts
        })
}

fn normalize(value: &str) -> String {
    value.split_whitespace().collect::<String>()
}
fn number_tokens(value: &str) -> Vec<String> {
    value
        .split(|ch: char| !ch.is_ascii_digit())
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}
fn formatting_tokens(value: &str) -> Vec<char> {
    value
        .chars()
        .filter(|ch| matches!(ch, '<' | '>' | '{' | '}' | '[' | ']' | '\\'))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::SubtitleSegment;
    use std::path::PathBuf;
    fn document(text: &str) -> SubtitleDocument {
        SubtitleDocument {
            path: PathBuf::from("x.srt"),
            format: "srt".to_owned(),
            header: None,
            passthrough_blocks: Vec::new(),
            segments: vec![SubtitleSegment {
                id: "1".to_owned(),
                text: text.to_owned(),
                start: None,
                end: None,
                identifier: None,
                settings: None,
            }],
        }
    }
    #[test]
    fn identical_reference_scores_perfectly() {
        let report = evaluate(&document("你好，世界"), &document("你好，世界")).expect("evaluate");
        assert_eq!(report.exact_matches, 1);
        assert!((report.chrf - 1.0).abs() < f64::EPSILON);
        assert_eq!(report.mqm, MqmCounts::default());
    }
}
