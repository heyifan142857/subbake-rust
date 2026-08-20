//! Deterministic offline translation evaluation for regression tracking.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::entities::SubtitleDocument;
use crate::error::{CoreError, CoreResult};
use crate::formatting::formatting_tokens;
use crate::number_facts::{NumberFactComparison, compare_number_facts};
use crate::quality::{QualityPolicy, QualityReport, inspect_quality};
use crate::term_matcher::TermMatcher;

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationQualityReport {
    pub version: u64,
    pub hard_constraints: HardConstraintReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<EvaluationReport>,
    pub document_consistency: DocumentConsistencyReport,
    pub readability: QualityReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardConstraintReport {
    pub passed: bool,
    pub checks: usize,
    pub violations: Vec<HardConstraintViolation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardConstraintKind {
    SegmentCount,
    DuplicateId,
    IdAlignment,
    Timing,
    Formatting,
    FactualToken,
    RequiredTerm,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardConstraintViolation {
    pub kind: HardConstraintKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment_id: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsistencyKind {
    PersonName,
    Terminology,
    Pronoun,
    Honorific,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsistencyRule {
    pub label: String,
    pub kind: ConsistencyKind,
    pub source_terms: Vec<String>,
    pub allowed_targets: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentConsistencyReport {
    pub passed: bool,
    pub rules_checked: usize,
    pub violations: Vec<DocumentConsistencyViolation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentConsistencyViolationKind {
    MissingTarget,
    VariantDrift,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentConsistencyViolation {
    pub rule: String,
    pub kind: ConsistencyKind,
    pub violation: DocumentConsistencyViolationKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment_id: Option<String>,
    pub message: String,
}

/// Evaluate translation-specific deterministic requirements separately from
/// reference or model-based semantic scores. `passed` is true only when every
/// ID, timing, formatting, factual-token, and required-term check passes.
pub fn evaluate_translation_quality(
    source: &SubtitleDocument,
    candidate: &SubtitleDocument,
    reference: Option<&SubtitleDocument>,
    required_glossary: &BTreeMap<String, String>,
    consistency_rules: &[ConsistencyRule],
    readability_policy: QualityPolicy,
) -> CoreResult<TranslationQualityReport> {
    let hard_constraints = evaluate_hard_constraints(source, candidate, required_glossary);
    let reference = if reference.is_some() && !has_duplicate_ids(candidate) {
        reference
            .map(|document| evaluate(candidate, document))
            .transpose()?
    } else {
        None
    };
    Ok(TranslationQualityReport {
        version: 1,
        hard_constraints,
        reference,
        document_consistency: evaluate_document_consistency(source, candidate, consistency_rules),
        readability: inspect_quality(candidate, readability_policy),
    })
}

fn evaluate_hard_constraints(
    source: &SubtitleDocument,
    candidate: &SubtitleDocument,
    required_glossary: &BTreeMap<String, String>,
) -> HardConstraintReport {
    let mut checks = 1;
    let mut violations = Vec::new();
    if source.segments.len() != candidate.segments.len() {
        push_hard_violation(
            &mut violations,
            HardConstraintKind::SegmentCount,
            None,
            format!(
                "expected {} segments, got {}",
                source.segments.len(),
                candidate.segments.len()
            ),
        );
    }

    report_duplicate_ids(source, "source", &mut checks, &mut violations);
    report_duplicate_ids(candidate, "candidate", &mut checks, &mut violations);
    let candidate_by_id = candidate
        .segments
        .iter()
        .map(|segment| (segment.id.as_str(), segment))
        .collect::<BTreeMap<_, _>>();
    let source_ids = source
        .segments
        .iter()
        .map(|segment| segment.id.as_str())
        .collect::<BTreeSet<_>>();
    let matcher = TermMatcher::case_insensitive();

    for (position, source_segment) in source.segments.iter().enumerate() {
        checks += 1;
        let Some(candidate_segment) = candidate_by_id.get(source_segment.id.as_str()) else {
            push_hard_violation(
                &mut violations,
                HardConstraintKind::IdAlignment,
                Some(&source_segment.id),
                "source ID is missing from candidate",
            );
            continue;
        };
        if candidate.segments.get(position).map(|segment| &segment.id) != Some(&source_segment.id) {
            push_hard_violation(
                &mut violations,
                HardConstraintKind::IdAlignment,
                Some(&source_segment.id),
                "candidate ID order differs from source",
            );
        }

        checks += 4;
        if source_segment.start != candidate_segment.start
            || source_segment.end != candidate_segment.end
        {
            push_hard_violation(
                &mut violations,
                HardConstraintKind::Timing,
                Some(&source_segment.id),
                "start or end timestamp changed",
            );
        }
        if formatting_tokens(&source_segment.text) != formatting_tokens(&candidate_segment.text) {
            push_hard_violation(
                &mut violations,
                HardConstraintKind::Formatting,
                Some(&source_segment.id),
                "subtitle formatting markers changed",
            );
        }
        if matches!(
            compare_number_facts(&source_segment.text, &candidate_segment.text),
            NumberFactComparison::HardMismatch { .. }
        ) {
            push_hard_violation(
                &mut violations,
                HardConstraintKind::FactualToken,
                Some(&source_segment.id),
                "numbers, dates, amounts, or percentages changed",
            );
        }
        let missing = matcher.missing_required(
            &source_segment.text,
            &candidate_segment.text,
            required_glossary,
        );
        if !missing.is_empty() {
            let pairs = missing
                .iter()
                .map(|(source, target)| format!("{source} -> {target}"))
                .collect::<Vec<_>>()
                .join(", ");
            push_hard_violation(
                &mut violations,
                HardConstraintKind::RequiredTerm,
                Some(&source_segment.id),
                format!("required glossary term is missing: {pairs}"),
            );
        }
    }
    for candidate_segment in &candidate.segments {
        if !source_ids.contains(candidate_segment.id.as_str()) {
            checks += 1;
            push_hard_violation(
                &mut violations,
                HardConstraintKind::IdAlignment,
                Some(&candidate_segment.id),
                "candidate contains an unexpected ID",
            );
        }
    }
    HardConstraintReport {
        passed: violations.is_empty(),
        checks,
        violations,
    }
}

fn report_duplicate_ids(
    document: &SubtitleDocument,
    role: &str,
    checks: &mut usize,
    violations: &mut Vec<HardConstraintViolation>,
) {
    let mut seen = BTreeSet::new();
    for segment in &document.segments {
        *checks += 1;
        if !seen.insert(segment.id.as_str()) {
            push_hard_violation(
                violations,
                HardConstraintKind::DuplicateId,
                Some(&segment.id),
                format!("{role} contains a duplicate ID"),
            );
        }
    }
}

fn push_hard_violation(
    violations: &mut Vec<HardConstraintViolation>,
    kind: HardConstraintKind,
    segment_id: Option<&str>,
    message: impl Into<String>,
) {
    violations.push(HardConstraintViolation {
        kind,
        segment_id: segment_id.map(ToOwned::to_owned),
        message: message.into(),
    });
}

fn evaluate_document_consistency(
    source: &SubtitleDocument,
    candidate: &SubtitleDocument,
    rules: &[ConsistencyRule],
) -> DocumentConsistencyReport {
    let candidate_by_id = candidate
        .segments
        .iter()
        .map(|segment| (segment.id.as_str(), segment.text.as_str()))
        .collect::<BTreeMap<_, _>>();
    let matcher = TermMatcher::case_insensitive();
    let mut violations = Vec::new();
    for rule in rules {
        let mut observed = BTreeSet::new();
        for source_segment in &source.segments {
            if !rule
                .source_terms
                .iter()
                .any(|term| matcher.contains(&source_segment.text, term))
            {
                continue;
            }
            let Some(candidate_text) = candidate_by_id.get(source_segment.id.as_str()) else {
                continue;
            };
            let matches = rule
                .allowed_targets
                .iter()
                .filter(|target| matcher.contains(candidate_text, target))
                .cloned()
                .collect::<Vec<_>>();
            if matches.is_empty() {
                violations.push(DocumentConsistencyViolation {
                    rule: rule.label.clone(),
                    kind: rule.kind,
                    violation: DocumentConsistencyViolationKind::MissingTarget,
                    segment_id: Some(source_segment.id.clone()),
                    message: "no allowed target form appears in the aligned segment".to_owned(),
                });
            } else {
                observed.extend(matches);
            }
        }
        if observed.len() > 1 {
            violations.push(DocumentConsistencyViolation {
                rule: rule.label.clone(),
                kind: rule.kind,
                violation: DocumentConsistencyViolationKind::VariantDrift,
                segment_id: None,
                message: format!(
                    "multiple target variants were used: {}",
                    observed.into_iter().collect::<Vec<_>>().join(", ")
                ),
            });
        }
    }
    DocumentConsistencyReport {
        passed: violations.is_empty(),
        rules_checked: rules.len(),
        violations,
    }
}

fn has_duplicate_ids(document: &SubtitleDocument) -> bool {
    let mut ids = BTreeSet::new();
    document
        .segments
        .iter()
        .any(|segment| !ids.insert(segment.id.as_str()))
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
        if matches!(
            compare_number_facts(candidate_line, &reference_line.text),
            NumberFactComparison::HardMismatch { .. }
        ) {
            mqm.major += 1;
        }
        if legacy_formatting_tokens(candidate_line)
            != legacy_formatting_tokens(&reference_line.text)
        {
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
fn legacy_formatting_tokens(value: &str) -> Vec<char> {
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
            metadata: Default::default(),
            segments: vec![SubtitleSegment {
                id: "1".to_owned(),
                text: text.to_owned(),
                start: None,
                end: None,
                identifier: None,
                settings: None,
                semantic: Default::default(),
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

    #[test]
    fn offline_evaluation_penalizes_only_hard_number_mismatches() {
        let uncertain = evaluate(
            &document("看看这些乱七八糟的东西。"),
            &document("Look at all this mess."),
        )
        .expect("evaluate ambiguous expression");
        assert_eq!(uncertain.mqm.major, 0);

        let hard = evaluate(&document("她十三岁。"), &document("She is 12 years old."))
            .expect("evaluate explicit number change");
        assert_eq!(hard.mqm.major, 1);
    }

    #[test]
    fn hard_constraint_evaluation_uses_the_same_number_fact_boundary() {
        let uncertain = evaluate_translation_quality(
            &document("Look at all this mess."),
            &document("看看这些乱七八糟的东西。"),
            None,
            &BTreeMap::new(),
            &[],
            QualityPolicy::default(),
        )
        .expect("evaluate ambiguous hard constraint");
        assert!(uncertain.hard_constraints.passed);

        let hard = evaluate_translation_quality(
            &document("She is 12 years old."),
            &document("她十三岁。"),
            None,
            &BTreeMap::new(),
            &[],
            QualityPolicy::default(),
        )
        .expect("evaluate explicit hard constraint");
        assert!(
            hard.hard_constraints
                .violations
                .iter()
                .any(|violation| { violation.kind == HardConstraintKind::FactualToken })
        );
    }
}
