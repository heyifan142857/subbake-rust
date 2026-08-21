use std::collections::{BTreeMap, HashMap};

use crate::entities::TerminologyKind;
use crate::entities::{GlossaryEntry, SubtitleSegment, TranslationLine};
use crate::error::{CoreError, CoreResult};
#[cfg(test)]
use crate::language_rules::LanguageRuleRegistry;
use crate::language_rules::ResolvedLanguageRules;
use crate::term_matcher::TermMatcher;

use super::BatchWithUsage;

const TERM_MARKER_PREFIX: &str = "⟦T";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TermMarker {
    index: usize,
    source: String,
}

pub(super) fn select_markers(source: &[SubtitleSegment], candidates: &[String]) -> Vec<TermMarker> {
    let haystack = source
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let eligible = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| !is_source_stable_acronym(candidate))
        .collect::<Vec<_>>();
    let terms = eligible
        .iter()
        .map(|(_, candidate)| candidate.as_str())
        .collect::<Vec<_>>();
    TermMatcher::case_insensitive()
        .matching_indices(&haystack, &terms)
        .into_iter()
        .filter_map(|matched| eligible.get(matched))
        .take(12)
        .map(|(index, source)| TermMarker {
            index: *index,
            source: (*source).clone(),
        })
        .collect()
}

pub(super) fn protect_terms(text: &str, markers: &[TermMarker]) -> String {
    let terms = markers
        .iter()
        .map(|marker| marker.source.as_str())
        .collect::<Vec<_>>();
    TermMatcher::case_insensitive().replace_matches(text, &terms, |matched, text| {
        let index = markers[matched].index;
        format!("{TERM_MARKER_PREFIX}{index}⟧{text}⟦/T{index}⟧")
    })
}

pub(super) fn extract_terms(
    lines: &mut [TranslationLine],
    markers: &[TermMarker],
) -> Vec<GlossaryEntry> {
    let mut updates = Vec::new();
    for marker in markers {
        let start = format!("{}{index}⟧", TERM_MARKER_PREFIX, index = marker.index);
        let end = format!("⟦/T{index}⟧", index = marker.index);
        let mut accepted = None;
        for line in &mut *lines {
            while let Some(start_at) = line.translation.find(&start) {
                let content_at = start_at + start.len();
                let Some(relative_end) = line.translation[content_at..].find(&end) else {
                    line.translation.replace_range(start_at..content_at, "");
                    break;
                };
                let end_at = content_at + relative_end;
                let target = line.translation[content_at..end_at].trim().to_owned();
                line.translation
                    .replace_range(end_at..end_at + end.len(), "");
                line.translation.replace_range(start_at..content_at, "");
                if accepted.is_none() && !target.is_empty() {
                    accepted = Some(target);
                }
            }
            line.translation = line.translation.replace(&start, "").replace(&end, "");
        }
        if let Some(target) = accepted {
            updates.push(GlossaryEntry {
                source: marker.source.clone(),
                target,
            });
        }
    }
    updates
}

fn is_source_stable_acronym(candidate: &str) -> bool {
    let letters = candidate
        .chars()
        .filter(|character| character.is_ascii_alphabetic())
        .collect::<Vec<_>>();
    letters.len() >= 2
        && letters
            .iter()
            .all(|character| character.is_ascii_uppercase())
}

#[cfg(test)]
pub(super) fn reconcile_batch(
    source: &[SubtitleSegment],
    result: &mut BatchWithUsage,
    canonical: &mut BTreeMap<String, String>,
    enforced: &mut BTreeMap<String, String>,
    candidates: &[String],
    target_language: &str,
    preserve_names: bool,
) -> CoreResult<()> {
    let language_rules = LanguageRuleRegistry::resolve("Auto", target_language);
    reconcile_batch_with_rules(
        source,
        result,
        canonical,
        enforced,
        candidates,
        &language_rules,
        preserve_names,
    )
}

pub(super) fn reconcile_batch_with_rules(
    source: &[SubtitleSegment],
    result: &mut BatchWithUsage,
    canonical: &mut BTreeMap<String, String>,
    enforced: &mut BTreeMap<String, String>,
    candidates: &[String],
    language_rules: &ResolvedLanguageRules,
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
                && language_rules.target_requires_non_latin_name()
                && source_form.eq_ignore_ascii_case(proposed)
            {
                return Err(CoreError::InvalidTranslation(format!(
                    "personal name `{source_form}` was left in source spelling"
                )));
            }
            let chosen = existing.unwrap_or_else(|| proposed.to_owned());
            let target_is_present = result.lines.iter().any(|line| {
                matching_ids.contains(&line.id.as_str())
                    && TermMatcher::case_insensitive().contains(&line.translation, proposed)
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
                            && TermMatcher::case_insensitive().contains(&line.translation, proposed)
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

    let mut accepted = Vec::new();
    for mut entry in std::mem::take(&mut result.glossary_updates) {
        let source_form = entry.source.trim();
        let proposed = entry.target.trim();
        if source_form.is_empty()
            || proposed.is_empty()
            || !candidates
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(source_form))
        {
            continue;
        }
        let matching_ids = source
            .iter()
            .filter(|segment| contains_case_insensitive(&segment.text, source_form))
            .map(|segment| segment.id.as_str())
            .collect::<Vec<_>>();
        if matching_ids.is_empty()
            || !result.lines.iter().any(|line| {
                matching_ids.contains(&line.id.as_str())
                    && TermMatcher::case_insensitive().contains(&line.translation, proposed)
            })
        {
            continue;
        }
        let key = source_form.to_lowercase();
        let chosen = canonical
            .get(&key)
            .cloned()
            .unwrap_or_else(|| proposed.to_owned());
        if chosen != proposed {
            for line in &result.lines {
                if matching_ids.contains(&line.id.as_str())
                    && TermMatcher::case_insensitive().contains(&line.translation, proposed)
                {
                    replacements
                        .entry(line.id.clone())
                        .or_default()
                        .push((proposed.to_owned(), chosen.clone()));
                }
            }
        }
        canonical.entry(key).or_insert_with(|| chosen.clone());
        entry.source = source_form.to_owned();
        entry.target = chosen;
        accepted.push(entry);
    }
    result.glossary_updates = accepted;

    for line in &mut result.lines {
        if let Some(items) = replacements.get(&line.id) {
            line.translation = simultaneous_replace(&line.translation, items);
        }
    }
    Ok(())
}

fn contains_case_insensitive(text: &str, needle: &str) -> bool {
    TermMatcher::case_insensitive().contains(text, needle)
}

fn simultaneous_replace(text: &str, replacements: &[(String, String)]) -> String {
    let terms = replacements
        .iter()
        .map(|(source, _)| source.as_str())
        .collect::<Vec<_>>();
    TermMatcher::case_insensitive()
        .replace_matches(text, &terms, |matched, _| replacements[matched].1.clone())
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
            semantic: Default::default(),
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
            &["Zasa".to_owned()],
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
        assert_eq!(
            simultaneous_replace("Joey met Joe.", &[("Joe".to_owned(), "约瑟夫".to_owned())]),
            "Joey met 约瑟夫."
        );
    }

    #[test]
    fn lightweight_terms_accept_candidates_and_rewrite_later_conflicts() {
        let source = vec![segment("1", "Astrophage escaped.")];
        let mut result = BatchWithUsage {
            lines: vec![TranslationLine {
                id: "1".to_owned(),
                translation: "噬星体逃走了。".to_owned(),
            }],
            summary: String::new(),
            glossary_updates: vec![GlossaryEntry {
                source: "Astrophage".to_owned(),
                target: "噬星体".to_owned(),
            }],
            terminology_updates: Vec::new(),
            usage: Usage::default(),
            cache_key: None,
        };
        let mut canonical = BTreeMap::from([("astrophage".to_owned(), "星食体".to_owned())]);

        reconcile_batch(
            &source,
            &mut result,
            &mut canonical,
            &mut BTreeMap::new(),
            &["Astrophage".to_owned()],
            "zh-Hans",
            false,
        )
        .expect("lightweight reconcile");

        assert_eq!(result.lines[0].translation, "星食体逃走了。");
        assert_eq!(result.glossary_updates[0].target, "星食体");
    }

    #[test]
    fn lightweight_terms_silently_drop_unrequested_or_unverifiable_entries() {
        let source = vec![segment("1", "Astrophage escaped.")];
        let mut result = BatchWithUsage {
            lines: vec![TranslationLine {
                id: "1".to_owned(),
                translation: "它逃走了。".to_owned(),
            }],
            summary: String::new(),
            glossary_updates: vec![GlossaryEntry {
                source: "Ordinary".to_owned(),
                target: "普通".to_owned(),
            }],
            terminology_updates: Vec::new(),
            usage: Usage::default(),
            cache_key: None,
        };

        reconcile_batch(
            &source,
            &mut result,
            &mut BTreeMap::new(),
            &mut BTreeMap::new(),
            &["Astrophage".to_owned()],
            "zh-Hans",
            false,
        )
        .expect("optional terms never fail translation");

        assert!(result.glossary_updates.is_empty());
    }

    #[test]
    fn marker_round_trip_extracts_term_without_a_separate_response_field() {
        let source = vec![segment("1", "Astrophage escaped.")];
        let markers = select_markers(&source, &["Astrophage".to_owned()]);
        assert_eq!(
            protect_terms(&source[0].text, &markers),
            "⟦T0⟧Astrophage⟦/T0⟧ escaped."
        );
        let mut lines = vec![TranslationLine {
            id: "1".to_owned(),
            translation: "⟦T0⟧星食体⟦/T0⟧逃走了。".to_owned(),
        }];

        let updates = extract_terms(&mut lines, &markers);

        assert_eq!(lines[0].translation, "星食体逃走了。");
        assert_eq!(
            updates,
            vec![GlossaryEntry {
                source: "Astrophage".to_owned(),
                target: "星食体".to_owned(),
            }]
        );
    }

    #[test]
    fn lightweight_markers_skip_source_stable_acronyms() {
        let source = vec![segment("1", "YIFY links to YTS.BZ.")];
        let markers = select_markers(&source, &["YIFY".to_owned(), "YTS.BZ".to_owned()]);
        assert!(markers.is_empty());
    }
}
