//! Deterministic project-level subtitle inventory and consistency analysis.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{QualityPolicy, QualityReport, SubtitleDocument, inspect_quality};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectTranslationStatus {
    Pending,
    Translated,
    Bilingual,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectDocumentPair {
    pub source_path: String,
    pub source: SubtitleDocument,
    pub output_path: Option<String>,
    pub output: Option<SubtitleDocument>,
    pub bilingual: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectFileReport {
    pub source_path: String,
    pub output_path: Option<String>,
    pub format: String,
    pub segments: usize,
    pub translated_segments: usize,
    pub status: ProjectTranslationStatus,
    pub source_quality: QualityReport,
    pub output_quality: Option<QualityReport>,
    pub aligned: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectConsistencyOccurrence {
    pub source_path: String,
    pub output_path: String,
    pub segment_id: String,
    pub translation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProjectConsistencyIssue {
    SegmentAlignment {
        source_path: String,
        output_path: String,
        source_segments: usize,
        output_segments: usize,
    },
    DivergentTranslation {
        source_text: String,
        translations: Vec<String>,
        occurrences: Vec<ProjectConsistencyOccurrence>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectSummary {
    pub files: usize,
    pub pending: usize,
    pub translated: usize,
    pub bilingual: usize,
    pub segments: usize,
    pub qa_errors: usize,
    pub qa_warnings: usize,
    pub consistency_issues: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectReport {
    pub version: u64,
    pub root: String,
    pub summary: ProjectSummary,
    pub files: Vec<ProjectFileReport>,
    pub consistency_issues: Vec<ProjectConsistencyIssue>,
}

pub fn inspect_project(
    root: impl Into<String>,
    mut pairs: Vec<ProjectDocumentPair>,
    quality_policy: QualityPolicy,
) -> ProjectReport {
    pairs.sort_by(|left, right| left.source_path.cmp(&right.source_path));
    let mut files = Vec::with_capacity(pairs.len());
    let mut consistency_issues = Vec::new();
    let mut translations_by_source = BTreeMap::<String, Vec<ProjectConsistencyOccurrence>>::new();

    for pair in pairs {
        let source_quality = inspect_quality(&pair.source, quality_policy);
        let output_quality = pair
            .output
            .as_ref()
            .map(|document| inspect_quality(document, quality_policy));
        let status = if pair.output.is_none() {
            ProjectTranslationStatus::Pending
        } else if pair.bilingual {
            ProjectTranslationStatus::Bilingual
        } else {
            ProjectTranslationStatus::Translated
        };
        let translated_segments = pair
            .output
            .as_ref()
            .map_or(0, |document| document.segments.len());
        let aligned = pair
            .output
            .as_ref()
            .is_none_or(|output| segments_align(&pair.source, output));

        if let (Some(output_path), Some(output)) = (&pair.output_path, &pair.output) {
            if !aligned {
                consistency_issues.push(ProjectConsistencyIssue::SegmentAlignment {
                    source_path: pair.source_path.clone(),
                    output_path: output_path.clone(),
                    source_segments: pair.source.segments.len(),
                    output_segments: output.segments.len(),
                });
            } else if !pair.bilingual {
                for (source, translated) in pair.source.segments.iter().zip(&output.segments) {
                    let source_text = normalized_text(&source.text);
                    let translation = translated.text.trim().to_owned();
                    if !source_text.is_empty() && !translation.is_empty() {
                        translations_by_source.entry(source_text).or_default().push(
                            ProjectConsistencyOccurrence {
                                source_path: pair.source_path.clone(),
                                output_path: output_path.clone(),
                                segment_id: source.id.clone(),
                                translation,
                            },
                        );
                    }
                }
            }
        }

        files.push(ProjectFileReport {
            source_path: pair.source_path,
            output_path: pair.output_path,
            format: pair.source.format,
            segments: pair.source.segments.len(),
            translated_segments,
            status,
            source_quality,
            output_quality,
            aligned,
        });
    }

    for (source_text, occurrences) in translations_by_source {
        let translations = occurrences
            .iter()
            .map(|occurrence| occurrence.translation.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if translations.len() > 1 {
            consistency_issues.push(ProjectConsistencyIssue::DivergentTranslation {
                source_text,
                translations,
                occurrences,
            });
        }
    }

    let summary = summarize(&files, consistency_issues.len());
    ProjectReport {
        version: 1,
        root: root.into(),
        summary,
        files,
        consistency_issues,
    }
}

fn segments_align(source: &SubtitleDocument, output: &SubtitleDocument) -> bool {
    source.segments.len() == output.segments.len()
        && source
            .segments
            .iter()
            .zip(&output.segments)
            .all(|(source, output)| source.id == output.id)
}

fn normalized_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn summarize(files: &[ProjectFileReport], consistency_issues: usize) -> ProjectSummary {
    let mut summary = ProjectSummary {
        files: files.len(),
        pending: 0,
        translated: 0,
        bilingual: 0,
        segments: 0,
        qa_errors: 0,
        qa_warnings: 0,
        consistency_issues,
    };
    for file in files {
        match file.status {
            ProjectTranslationStatus::Pending => summary.pending += 1,
            ProjectTranslationStatus::Translated => summary.translated += 1,
            ProjectTranslationStatus::Bilingual => summary.bilingual += 1,
        }
        summary.segments += file.segments;
        summary.qa_errors += file.source_quality.errors
            + file
                .output_quality
                .as_ref()
                .map_or(0, |report| report.errors);
        summary.qa_warnings += file.source_quality.warnings
            + file
                .output_quality
                .as_ref()
                .map_or(0, |report| report.warnings);
    }
    summary
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::SubtitleSegment;

    use super::*;

    fn document(path: &str, texts: &[&str]) -> SubtitleDocument {
        SubtitleDocument {
            path: PathBuf::from(path),
            format: "srt".to_owned(),
            segments: texts
                .iter()
                .enumerate()
                .map(|(index, text)| SubtitleSegment {
                    id: (index + 1).to_string(),
                    text: (*text).to_owned(),
                    start: None,
                    end: None,
                    identifier: None,
                    settings: None,
                    semantic: Default::default(),
                })
                .collect(),
            header: None,
            passthrough_blocks: Vec::new(),
            metadata: Default::default(),
        }
    }

    #[test]
    fn reports_pending_files_and_divergent_translations() {
        let report = inspect_project(
            "season",
            vec![
                ProjectDocumentPair {
                    source_path: "e1.srt".to_owned(),
                    source: document("e1.srt", &["Hello"]),
                    output_path: Some("e1.translated.srt".to_owned()),
                    output: Some(document("e1.translated.srt", &["你好"])),
                    bilingual: false,
                },
                ProjectDocumentPair {
                    source_path: "e2.srt".to_owned(),
                    source: document("e2.srt", &["Hello"]),
                    output_path: Some("e2.translated.srt".to_owned()),
                    output: Some(document("e2.translated.srt", &["您好"])),
                    bilingual: false,
                },
                ProjectDocumentPair {
                    source_path: "e3.srt".to_owned(),
                    source: document("e3.srt", &["Later"]),
                    output_path: None,
                    output: None,
                    bilingual: false,
                },
            ],
            QualityPolicy::default(),
        );

        assert_eq!(report.summary.files, 3);
        assert_eq!(report.summary.pending, 1);
        assert_eq!(report.summary.translated, 2);
        assert_eq!(report.summary.consistency_issues, 1);
        assert!(matches!(
            report.consistency_issues[0],
            ProjectConsistencyIssue::DivergentTranslation { .. }
        ));
    }
}
