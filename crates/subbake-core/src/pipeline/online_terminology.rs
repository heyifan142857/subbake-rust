use std::collections::{BTreeMap, HashMap};

use crate::entities::SubtitleSegment;
use crate::entities::TerminologyKind;
use crate::error::{CoreError, CoreResult};

use super::BatchWithUsage;

pub(super) fn reconcile_batch(
    source: &[SubtitleSegment],
    result: &mut BatchWithUsage,
    canonical: &mut BTreeMap<String, String>,
    enforced: &mut BTreeMap<String, String>,
    target_language: &str,
    preserve_names: bool,
) -> CoreResult<()> {
    let mut replacements: HashMap<String, Vec<(String, String)>> = HashMap::new();

    for entity in &mut result.terminology_updates {
        if entity.canonical_source.trim().is_empty() || entity.variants.is_empty() {
            return Err(CoreError::InvalidTranslation(
                "terminology update is missing a canonical source or variants".to_owned(),
            ));
        }
        for variant in &mut entity.variants {
            let source_form = variant.source.trim();
            let proposed = variant.target.trim();
            if source_form.is_empty() || proposed.is_empty() {
                return Err(CoreError::InvalidTranslation(
                    "terminology update contains an empty source or target".to_owned(),
                ));
            }
            let matching_ids = source
                .iter()
                .filter(|segment| contains_case_insensitive(&segment.text, source_form))
                .map(|segment| segment.id.as_str())
                .collect::<Vec<_>>();
            if matching_ids.is_empty() {
                return Err(CoreError::InvalidTranslation(format!(
                    "terminology update contains source form `{source_form}` absent from its batch"
                )));
            }

            let key = source_form.to_lowercase();
            let existing = canonical
                .get(&key)
                .or_else(|| canonical.get(source_form))
                .cloned();
            if existing.is_none()
                && entity.kind == TerminologyKind::Person
                && !preserve_names
                && requires_non_latin_name(target_language)
                && source_form.eq_ignore_ascii_case(proposed)
            {
                return Err(CoreError::InvalidTranslation(format!(
                    "personal name `{source_form}` was left in source spelling"
                )));
            }
            let chosen = existing.unwrap_or_else(|| proposed.to_owned());
            let target_is_present = result.lines.iter().any(|line| {
                matching_ids.contains(&line.id.as_str()) && line.translation.contains(proposed)
            });
            if !target_is_present {
                return Err(CoreError::InvalidTranslation(format!(
                    "terminology target `{proposed}` for `{source_form}` is absent from the corresponding translation"
                )));
            }

            if entity.kind.is_enforced() {
                enforced
                    .entry(key.clone())
                    .or_insert_with(|| chosen.clone());
                if !chosen.eq_ignore_ascii_case(proposed) {
                    for line in &result.lines {
                        if matching_ids.contains(&line.id.as_str())
                            && line.translation.contains(proposed)
                        {
                            replacements
                                .entry(line.id.clone())
                                .or_default()
                                .push((proposed.to_owned(), chosen.clone()));
                        }
                    }
                }
            }
            canonical.entry(key).or_insert_with(|| chosen.clone());
            variant.target = chosen;
        }
    }

    for line in &mut result.lines {
        if let Some(items) = replacements.get(&line.id) {
            line.translation = simultaneous_replace(&line.translation, items);
        }
    }
    Ok(())
}

fn requires_non_latin_name(target_language: &str) -> bool {
    matches!(
        target_language.split('-').next().unwrap_or_default(),
        "zh" | "ja" | "ko" | "ru" | "uk" | "ar" | "hi" | "th"
    )
}

fn contains_case_insensitive(text: &str, needle: &str) -> bool {
    let text = text.to_lowercase();
    let needle = needle.to_lowercase();
    if !needle.chars().any(|ch| ch.is_ascii_alphabetic()) {
        return text.contains(&needle);
    }
    text.match_indices(&needle).any(|(start, matched)| {
        let before = text[..start].chars().next_back();
        let after = text[start + matched.len()..].chars().next();
        !before.is_some_and(|ch| ch.is_ascii_alphanumeric())
            && !after.is_some_and(|ch| ch.is_ascii_alphanumeric())
    })
}

fn simultaneous_replace(text: &str, replacements: &[(String, String)]) -> String {
    let mut replacements = replacements.to_vec();
    replacements.sort_by_key(|(old, _)| std::cmp::Reverse(old.chars().count()));
    let mut output = text.to_owned();
    for (index, (old, _)) in replacements.iter().enumerate() {
        output = output.replace(old, &format!("\u{e000}{index}\u{e001}"));
    }
    for (index, (_, new)) in replacements.iter().enumerate() {
        output = output.replace(&format!("\u{e000}{index}\u{e001}"), new);
    }
    output
}

#[cfg(test)]
mod tests {
    use crate::entities::{
        GlossaryEntry, TerminologyEntity, TerminologyKind, TranslationLine, Usage,
    };

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
    fn earliest_term_wins_and_only_matching_source_line_is_rewritten() {
        let source = vec![
            segment("1", "Zasa arrived."),
            segment("2", "It was a plaza."),
        ];
        let mut result = BatchWithUsage {
            lines: vec![
                TranslationLine {
                    id: "1".to_owned(),
                    translation: "萨萨来了。".to_owned(),
                },
                TranslationLine {
                    id: "2".to_owned(),
                    translation: "那是萨萨广场。".to_owned(),
                },
            ],
            summary: String::new(),
            glossary_updates: Vec::new(),
            terminology_updates: vec![TerminologyEntity {
                canonical_source: "Joey Zasa".to_owned(),
                kind: TerminologyKind::Person,
                variants: vec![GlossaryEntry {
                    source: "Zasa".to_owned(),
                    target: "萨萨".to_owned(),
                }],
            }],
            usage: Usage::default(),
            cache_key: None,
        };
        let mut canonical = BTreeMap::from([("zasa".to_owned(), "扎萨".to_owned())]);
        let mut enforced = canonical.clone();

        reconcile_batch(
            &source,
            &mut result,
            &mut canonical,
            &mut enforced,
            "zh-Hans",
            false,
        )
        .expect("reconcile");

        assert_eq!(result.lines[0].translation, "扎萨来了。");
        assert_eq!(result.lines[1].translation, "那是萨萨广场。");
        assert_eq!(result.terminology_updates[0].variants[0].target, "扎萨");
    }

    #[test]
    fn source_matching_does_not_treat_joe_as_joey() {
        assert!(contains_case_insensitive("Mr. Joe Zasa", "Joe"));
        assert!(!contains_case_insensitive("Joey Zasa", "Joe"));
    }
}
