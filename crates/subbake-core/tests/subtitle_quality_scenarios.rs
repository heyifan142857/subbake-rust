use std::collections::BTreeMap;
use std::path::Path;

use subbake_core::formats::parse_document_text;
use subbake_core::{
    ConsistencyKind, ConsistencyRule, DocumentConsistencyViolationKind, HardConstraintKind,
    QualityPolicy, evaluate_translation_quality,
};

fn srt(text: &str) -> subbake_core::SubtitleDocument {
    parse_document_text(Path::new("scenario.srt"), text, Some("srt")).expect("parse scenario")
}

#[test]
fn perfect_translation_passes_every_hard_constraint_and_reference_metric() {
    let source = srt(
        "1\n00:00:00,000 --> 00:00:02,000\n<i>Dr. Smith paid 1,200.</i>\n\n2\n00:00:02,000 --> 00:00:04,000\nHe returns in 3 days.\n",
    );
    let candidate = srt(
        "1\n00:00:00,000 --> 00:00:02,000\n<i>史密斯博士支付了 1,200。</i>\n\n2\n00:00:02,000 --> 00:00:04,000\n他将在 3 天后回来。\n",
    );
    let glossary = BTreeMap::from([("Dr. Smith".to_owned(), "史密斯博士".to_owned())]);
    let rules = vec![ConsistencyRule {
        label: "Dr. Smith".to_owned(),
        kind: ConsistencyKind::PersonName,
        source_terms: vec!["Dr. Smith".to_owned()],
        allowed_targets: vec!["史密斯博士".to_owned()],
    }];

    let report = evaluate_translation_quality(
        &source,
        &candidate,
        Some(&candidate),
        &glossary,
        &rules,
        QualityPolicy::default(),
    )
    .expect("evaluate translation");

    assert!(
        report.hard_constraints.passed,
        "{:?}",
        report.hard_constraints.violations
    );
    assert!(report.document_consistency.passed);
    let reference = report.reference.expect("reference metrics");
    assert_eq!(reference.exact_matches, 2);
    assert!((reference.chrf - 1.0).abs() < f64::EPSILON);
}

#[test]
fn adversarial_translation_fails_each_required_hard_constraint_category() {
    let source = srt(
        "1\n00:00:00,000 --> 00:00:02,000\n<i>Dr. Smith paid $1,200.</i>\n\n2\n00:00:02,000 --> 00:00:04,000\nHe returns in 3 days.\n",
    );
    let candidate = srt(
        "2\n00:00:02,000 --> 00:00:04,000\n她将在 4 天后回来。\n\n1\n00:00:00,100 --> 00:00:02,100\n<b>史密斯医生支付了 1,300 美元。</b>\n\n3\n00:00:04,000 --> 00:00:05,000\n额外字幕\n",
    );
    let glossary = BTreeMap::from([("Dr. Smith".to_owned(), "史密斯博士".to_owned())]);

    let report = evaluate_translation_quality(
        &source,
        &candidate,
        None,
        &glossary,
        &[],
        QualityPolicy::default(),
    )
    .expect("evaluate translation");
    let kinds = report
        .hard_constraints
        .violations
        .iter()
        .map(|violation| violation.kind)
        .collect::<Vec<_>>();

    assert!(!report.hard_constraints.passed);
    for required in [
        HardConstraintKind::SegmentCount,
        HardConstraintKind::IdAlignment,
        HardConstraintKind::Timing,
        HardConstraintKind::Formatting,
        HardConstraintKind::FactualToken,
        HardConstraintKind::RequiredTerm,
    ] {
        assert!(kinds.contains(&required), "missing {required:?}: {kinds:?}");
    }
}

#[test]
fn document_rules_detect_name_pronoun_and_honorific_drift() {
    let source = srt(
        "1\n00:00:00,000 --> 00:00:02,000\nBob said he was ready, sir.\n\n2\n00:00:02,000 --> 00:00:04,000\nBob said he would wait, sir.\n",
    );
    let candidate = srt(
        "1\n00:00:00,000 --> 00:00:02,000\n鲍勃说他准备好了，先生。\n\n2\n00:00:02,000 --> 00:00:04,000\n波布说她会等，阁下。\n",
    );
    let rules = vec![
        ConsistencyRule {
            label: "Bob".to_owned(),
            kind: ConsistencyKind::PersonName,
            source_terms: vec!["Bob".to_owned()],
            allowed_targets: vec!["鲍勃".to_owned(), "波布".to_owned()],
        },
        ConsistencyRule {
            label: "he".to_owned(),
            kind: ConsistencyKind::Pronoun,
            source_terms: vec!["he".to_owned()],
            allowed_targets: vec!["他".to_owned()],
        },
        ConsistencyRule {
            label: "sir".to_owned(),
            kind: ConsistencyKind::Honorific,
            source_terms: vec!["sir".to_owned()],
            allowed_targets: vec!["先生".to_owned()],
        },
    ];

    let report = evaluate_translation_quality(
        &source,
        &candidate,
        None,
        &BTreeMap::new(),
        &rules,
        QualityPolicy::default(),
    )
    .expect("evaluate translation");

    assert!(!report.document_consistency.passed);
    assert!(report.document_consistency.violations.iter().any(|issue| {
        issue.violation == DocumentConsistencyViolationKind::VariantDrift
            && issue.kind == ConsistencyKind::PersonName
    }));
    assert!(report.document_consistency.violations.iter().any(|issue| {
        issue.violation == DocumentConsistencyViolationKind::MissingTarget
            && issue.kind == ConsistencyKind::Pronoun
    }));
    assert!(report.document_consistency.violations.iter().any(|issue| {
        issue.violation == DocumentConsistencyViolationKind::MissingTarget
            && issue.kind == ConsistencyKind::Honorific
    }));
}
