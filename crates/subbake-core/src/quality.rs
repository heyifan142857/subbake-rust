//! Reference-free, deterministic subtitle quality checks.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::SubtitleDocument;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualitySeverity {
    Warning,
    Error,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityGate {
    #[default]
    Never,
    Error,
    Warning,
}

impl QualityGate {
    pub const fn fails(self, report: &QualityReport) -> bool {
        match self {
            Self::Never => false,
            Self::Error => report.errors > 0,
            Self::Warning => report.errors > 0 || report.warnings > 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityIssueKind {
    EmptyText,
    InvalidTiming,
    OverlappingTiming,
    ExcessiveReadingSpeed,
    ExcessiveLineLength,
    ExcessiveLineCount,
    RepeatedText,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityIssue {
    pub segment_id: String,
    pub kind: QualityIssueKind,
    pub severity: QualitySeverity,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measured: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityReport {
    pub version: u64,
    pub segments: usize,
    pub errors: usize,
    pub warnings: usize,
    pub issues: Vec<QualityIssue>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityPolicy {
    pub max_characters_per_second: f64,
    pub max_characters_per_line: usize,
    pub max_lines: usize,
}

impl Default for QualityPolicy {
    fn default() -> Self {
        Self {
            max_characters_per_second: 20.0,
            max_characters_per_line: 42,
            max_lines: 2,
        }
    }
}

pub fn inspect_quality(document: &SubtitleDocument, policy: QualityPolicy) -> QualityReport {
    let mut issues = Vec::new();
    let mut previous_end = None;
    let mut previous_text = String::new();

    for segment in &document.segments {
        let text = segment.text.trim();
        if text.is_empty() {
            push_issue(
                &mut issues,
                &segment.id,
                QualityIssueKind::EmptyText,
                QualitySeverity::Error,
                "subtitle text is empty",
                None,
                None,
            );
        }

        let lines = text.lines().collect::<Vec<_>>();
        if lines.len() > policy.max_lines {
            push_issue(
                &mut issues,
                &segment.id,
                QualityIssueKind::ExcessiveLineCount,
                QualitySeverity::Warning,
                format!("subtitle has {} lines", lines.len()),
                Some(lines.len() as f64),
                Some(policy.max_lines as f64),
            );
        }
        if let Some(longest) = lines.iter().map(|line| visible_characters(line)).max()
            && longest > policy.max_characters_per_line
        {
            push_issue(
                &mut issues,
                &segment.id,
                QualityIssueKind::ExcessiveLineLength,
                QualitySeverity::Warning,
                format!("longest line has {longest} visible characters"),
                Some(longest as f64),
                Some(policy.max_characters_per_line as f64),
            );
        }

        let timing = segment
            .start
            .as_deref()
            .zip(segment.end.as_deref())
            .and_then(|(start, end)| parse_timestamp(start).zip(parse_timestamp(end)));
        if segment.start.is_some() && segment.end.is_some() && timing.is_none() {
            push_issue(
                &mut issues,
                &segment.id,
                QualityIssueKind::InvalidTiming,
                QualitySeverity::Error,
                "subtitle timing could not be parsed",
                None,
                None,
            );
        }
        if let Some((start, end)) = timing {
            if end <= start {
                push_issue(
                    &mut issues,
                    &segment.id,
                    QualityIssueKind::InvalidTiming,
                    QualitySeverity::Error,
                    "subtitle end must be after its start",
                    None,
                    None,
                );
            } else {
                if previous_end.is_some_and(|value| start < value) {
                    push_issue(
                        &mut issues,
                        &segment.id,
                        QualityIssueKind::OverlappingTiming,
                        QualitySeverity::Warning,
                        "subtitle overlaps the previous segment",
                        None,
                        None,
                    );
                }
                let cps = visible_characters(text) as f64 / (end - start);
                if cps > policy.max_characters_per_second {
                    push_issue(
                        &mut issues,
                        &segment.id,
                        QualityIssueKind::ExcessiveReadingSpeed,
                        QualitySeverity::Warning,
                        format!("reading speed is {cps:.1} characters per second"),
                        Some(cps),
                        Some(policy.max_characters_per_second),
                    );
                }
            }
            previous_end = Some(end);
        }

        let normalized = text.split_whitespace().collect::<String>().to_lowercase();
        if !normalized.is_empty() && normalized == previous_text {
            push_issue(
                &mut issues,
                &segment.id,
                QualityIssueKind::RepeatedText,
                QualitySeverity::Warning,
                "subtitle repeats the previous segment exactly",
                None,
                None,
            );
        }
        previous_text = normalized;
    }

    let counts = issues.iter().fold(BTreeMap::new(), |mut counts, issue| {
        *counts.entry(issue.severity).or_insert(0usize) += 1;
        counts
    });
    QualityReport {
        version: 1,
        segments: document.segments.len(),
        errors: counts.get(&QualitySeverity::Error).copied().unwrap_or(0),
        warnings: counts.get(&QualitySeverity::Warning).copied().unwrap_or(0),
        issues,
    }
}

fn push_issue(
    issues: &mut Vec<QualityIssue>,
    segment_id: &str,
    kind: QualityIssueKind,
    severity: QualitySeverity,
    message: impl Into<String>,
    measured: Option<f64>,
    limit: Option<f64>,
) {
    issues.push(QualityIssue {
        segment_id: segment_id.to_owned(),
        kind,
        severity,
        message: message.into(),
        measured,
        limit,
    });
}

fn visible_characters(text: &str) -> usize {
    text.chars()
        .filter(|character| !character.is_whitespace())
        .count()
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{SubtitleDocumentMetadata, SubtitleSegment};

    use super::*;

    #[test]
    fn reports_timing_readability_and_repetition_without_a_reference() {
        let document = SubtitleDocument {
            path: PathBuf::from("sample.srt"),
            format: "srt".to_owned(),
            segments: vec![
                SubtitleSegment {
                    id: "1".to_owned(),
                    text: "This subtitle is deliberately much too long for one second".to_owned(),
                    start: Some("00:00:00,000".to_owned()),
                    end: Some("00:00:01,000".to_owned()),
                    identifier: None,
                    settings: None,
                    semantic: Default::default(),
                },
                SubtitleSegment {
                    id: "2".to_owned(),
                    text: "This subtitle is deliberately much too long for one second".to_owned(),
                    start: Some("00:00:00,900".to_owned()),
                    end: Some("00:00:02,000".to_owned()),
                    identifier: None,
                    settings: None,
                    semantic: Default::default(),
                },
            ],
            header: None,
            passthrough_blocks: Vec::new(),
            metadata: SubtitleDocumentMetadata::None,
        };
        let report = inspect_quality(&document, QualityPolicy::default());
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.kind == QualityIssueKind::OverlappingTiming)
        );
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.kind == QualityIssueKind::ExcessiveReadingSpeed)
        );
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.kind == QualityIssueKind::RepeatedText)
        );
    }

    #[test]
    fn accepts_srt_vtt_and_ass_timestamp_shapes() {
        assert_eq!(parse_timestamp("01:02:03,500"), Some(3_723.5));
        assert_eq!(parse_timestamp("01:02:03.500"), Some(3_723.5));
        assert_eq!(parse_timestamp("1:02:03.50"), Some(3_723.5));
    }
}
