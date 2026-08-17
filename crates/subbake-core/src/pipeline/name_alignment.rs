use std::collections::BTreeMap;

use crate::entities::{GlossaryEntry, SubtitleSegment, TranslationLine};
use crate::error::{CoreError, CoreResult};
use crate::memory::ContextMemory;
use crate::term_matcher::TermMatcher;

use super::BatchWithUsage;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NameMarker {
    index: usize,
    source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TranslatedNameMarker {
    start: usize,
    end: usize,
    source: String,
    target: String,
}

pub(super) fn select_markers(source: &[SubtitleSegment], candidates: &[String]) -> Vec<NameMarker> {
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
        .map(|(index, source)| NameMarker {
            index: *index,
            source: (*source).clone(),
        })
        .collect()
}

pub(super) fn protect_names(text: &str, markers: &[NameMarker]) -> String {
    let terms = markers
        .iter()
        .map(|marker| marker.source.as_str())
        .collect::<Vec<_>>();
    TermMatcher::case_insensitive().replace_matches(text, &terms, |matched, text| {
        let index = markers[matched].index;
        format!("⟦N{index}⟧{text}⟦/N{index}⟧")
    })
}

pub(super) fn validate_markers(
    source: &[SubtitleSegment],
    lines: &[TranslationLine],
    markers: &[NameMarker],
) -> CoreResult<()> {
    for line in lines {
        let Some(source_line) = source.iter().find(|segment| segment.id == line.id) else {
            continue;
        };
        parse_markers(&source_line.text, &line.translation, markers)?;
    }
    Ok(())
}

pub(super) fn reconcile_batch(
    source: &[SubtitleSegment],
    result: &mut BatchWithUsage,
    memory: &mut ContextMemory,
    required_glossary: &BTreeMap<String, String>,
    candidates: &[String],
    target_language: &str,
) -> CoreResult<()> {
    let required = required_glossary
        .iter()
        .map(|(source, target)| (source.to_lowercase(), target.clone()))
        .collect::<BTreeMap<_, _>>();
    let markers = select_markers(source, candidates);
    let mut updates = Vec::new();
    let mut found_markers = false;

    for source_line in source {
        let Some(line) = result
            .lines
            .iter_mut()
            .find(|line| line.id == source_line.id)
        else {
            continue;
        };
        let translated = parse_markers(&source_line.text, &line.translation, &markers)?;
        if translated.is_empty() {
            continue;
        }
        found_markers = true;
        let mut clean = String::with_capacity(line.translation.len());
        let mut cursor = 0;
        for marker in translated {
            clean.push_str(&line.translation[cursor..marker.start]);
            if marker.source == marker.target
                && requires_target_script_change(&marker.source, target_language)
            {
                return Err(CoreError::InvalidTranslation(format!(
                    "personal name `{}` was left in source spelling",
                    marker.source
                )));
            }
            let key = marker.source.to_lowercase();
            let chosen = required
                .get(&key)
                .or_else(|| memory.name_translations.get(&key))
                .or_else(|| memory.glossary.get(&marker.source))
                .cloned()
                .unwrap_or(marker.target);
            memory
                .name_translations
                .entry(key)
                .or_insert_with(|| chosen.clone());
            push_update(&mut updates, marker.source, chosen.clone());
            clean.push_str(&chosen);
            cursor = marker.end;
        }
        clean.push_str(&line.translation[cursor..]);
        line.translation = clean;
    }

    if found_markers {
        result.glossary_updates = updates;
        return Ok(());
    }

    // Reconciled cache entries no longer contain markers. Rehydrate the
    // per-run canonical map from their persisted glossary updates.
    for mut entry in std::mem::take(&mut result.glossary_updates) {
        let source_form = entry.source.trim();
        let proposed = entry.target.trim();
        if source_form.is_empty()
            || proposed.is_empty()
            || !source
                .iter()
                .any(|segment| TermMatcher::case_insensitive().contains(&segment.text, source_form))
        {
            return Err(CoreError::InvalidTranslation(format!(
                "cached name alignment contains invalid entry `{source_form}` -> `{proposed}`"
            )));
        }
        let key = source_form.to_lowercase();
        let chosen = required
            .get(&key)
            .or_else(|| memory.name_translations.get(&key))
            .or_else(|| memory.glossary.get(source_form))
            .cloned()
            .unwrap_or_else(|| proposed.to_owned());
        memory
            .name_translations
            .entry(key)
            .or_insert_with(|| chosen.clone());
        entry.source = source_form.to_owned();
        entry.target = chosen;
        push_update(&mut updates, entry.source, entry.target);
    }
    result.glossary_updates = updates;
    Ok(())
}

fn parse_markers(
    source_line: &str,
    translation: &str,
    markers: &[NameMarker],
) -> CoreResult<Vec<TranslatedNameMarker>> {
    let mut translated = Vec::new();
    let mut cursor = 0;
    while let Some(relative_start) = translation[cursor..].find("⟦N") {
        let start = cursor + relative_start;
        if translation[cursor..start].contains("⟦/N") {
            return Err(invalid_marker(
                "closing marker appears before its opening marker",
            ));
        }
        let number_start = start + "⟦N".len();
        let Some(relative_open_end) = translation[number_start..].find('⟧') else {
            return Err(invalid_marker("missing opening-marker terminator `⟧`"));
        };
        let open_end = number_start + relative_open_end;
        let index = translation[number_start..open_end]
            .parse::<usize>()
            .map_err(|_| invalid_marker("opening marker has a non-numeric id"))?;
        let Some(marker) = markers.iter().find(|marker| marker.index == index) else {
            return Err(invalid_marker(&format!("unknown marker id `{index}`")));
        };
        if !TermMatcher::case_insensitive().contains(source_line, &marker.source) {
            return Err(invalid_marker(&format!(
                "marker `{index}` moved to a line without `{}`",
                marker.source
            )));
        }
        let target_start = open_end + '⟧'.len_utf8();
        let closing = format!("⟦/N{index}⟧");
        let Some(relative_end) = translation[target_start..].find(&closing) else {
            return Err(invalid_marker(&format!(
                "missing closing marker `{closing}`"
            )));
        };
        let target_end = target_start + relative_end;
        let target = &translation[target_start..target_end];
        if target.is_empty() || target != target.trim() || target.contains("⟦N") {
            return Err(invalid_marker(
                "translated name must be non-empty, trimmed text",
            ));
        }
        let end = target_end + closing.len();
        translated.push(TranslatedNameMarker {
            start,
            end,
            source: marker.source.clone(),
            target: target.to_owned(),
        });
        cursor = end;
    }
    if translation[cursor..].contains("⟦/N") {
        return Err(invalid_marker("closing marker has no opening marker"));
    }
    Ok(translated)
}

fn push_update(updates: &mut Vec<GlossaryEntry>, source: String, target: String) {
    if updates.iter().any(|entry| entry.source == source) {
        return;
    }
    updates.push(GlossaryEntry { source, target });
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

fn requires_target_script_change(source: &str, target_language: &str) -> bool {
    let target = target_language
        .split('-')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    match target.as_str() {
        "zh" => source.chars().any(|character| {
            character.is_ascii_alphabetic()
                || matches!(character, '\u{3040}'..='\u{30ff}' | '\u{ff66}'..='\u{ff9f}')
        }),
        "ja" => source
            .chars()
            .any(|character| character.is_ascii_alphabetic()),
        "ko" => source.chars().any(|character| {
            character.is_ascii_alphabetic()
                || matches!(character, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{9fff}')
        }),
        "ru" | "uk" | "ar" | "hi" | "th" => source
            .chars()
            .any(|character| character.is_ascii_alphabetic()),
        _ => false,
    }
}

fn invalid_marker(reason: &str) -> CoreError {
    CoreError::InvalidTranslation(format!("invalid Turbo personal-name marker: {reason}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::Usage;

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

    fn result(lines: &[(&str, &str)]) -> BatchWithUsage {
        BatchWithUsage {
            lines: lines
                .iter()
                .map(|(id, translation)| TranslationLine {
                    id: (*id).to_owned(),
                    translation: (*translation).to_owned(),
                })
                .collect(),
            summary: String::new(),
            glossary_updates: Vec::new(),
            terminology_updates: Vec::new(),
            usage: Usage::default(),
            cache_key: None,
        }
    }

    fn candidates() -> Vec<String> {
        vec!["Mary".to_owned(), "Marie".to_owned()]
    }

    #[test]
    fn protects_source_names_with_indexed_markers() {
        let source = vec![segment("1", "Hi Mary.")];
        let markers = select_markers(&source, &candidates());

        assert_eq!(
            protect_names(&source[0].text, &markers),
            "Hi ⟦N0⟧Mary⟦/N0⟧."
        );
    }

    #[test]
    fn earliest_translation_wins_in_subtitle_order() {
        let source = vec![segment("1", "Hi Mary."), segment("2", "Mary, wait.")];
        let mut translated = result(&[
            ("1", "你好，⟦N0⟧玛丽⟦/N0⟧。"),
            ("2", "⟦N0⟧玛莉⟦/N0⟧，等等。"),
        ]);
        let mut memory = ContextMemory::new();

        reconcile_batch(
            &source,
            &mut translated,
            &mut memory,
            &BTreeMap::new(),
            &candidates(),
            "zh-Hans",
        )
        .expect("reconcile");

        assert_eq!(translated.lines[0].translation, "你好，玛丽。");
        assert_eq!(translated.lines[1].translation, "玛丽，等等。");
        assert_eq!(translated.glossary_updates[0].target, "玛丽");
        assert_eq!(memory.name_translations["mary"], "玛丽");
    }

    #[test]
    fn explicit_glossary_overrides_first_model_translation() {
        let source = vec![segment("1", "Hi Mary.")];
        let mut translated = result(&[("1", "你好，⟦N0⟧玛丽⟦/N0⟧。")]);
        let mut memory = ContextMemory::new();
        let required = BTreeMap::from([("Mary".to_owned(), "玛莉".to_owned())]);

        reconcile_batch(
            &source,
            &mut translated,
            &mut memory,
            &required,
            &candidates(),
            "zh-Hans",
        )
        .expect("reconcile");

        assert_eq!(translated.lines[0].translation, "你好，玛莉。");
        assert_eq!(memory.name_translations["mary"], "玛莉");
    }

    #[test]
    fn separately_marked_names_are_replaced_without_collateral_changes() {
        let source = vec![segment("1", "Mary met Marie.")];
        let mut translated = result(&[("1", "⟦N0⟧玛丽⟦/N0⟧见到了⟦N1⟧玛丽⟦/N1⟧。")]);
        let mut memory = ContextMemory::new();
        memory
            .name_translations
            .insert("mary".to_owned(), "玛莉".to_owned());

        reconcile_batch(
            &source,
            &mut translated,
            &mut memory,
            &BTreeMap::new(),
            &candidates(),
            "zh-Hans",
        )
        .expect("reconcile");

        assert_eq!(translated.lines[0].translation, "玛莉见到了玛丽。");
    }

    #[test]
    fn later_batch_reuses_the_first_batch_translation() {
        let mut memory = ContextMemory::new();
        let mut first = result(&[("1", "⟦N0⟧玛丽⟦/N0⟧来了。")]);
        reconcile_batch(
            &[segment("1", "Mary arrived.")],
            &mut first,
            &mut memory,
            &BTreeMap::new(),
            &candidates(),
            "zh-Hans",
        )
        .expect("first batch");
        let mut second = result(&[("2", "⟦N0⟧玛莉⟦/N0⟧走了。")]);

        reconcile_batch(
            &[segment("2", "Mary left.")],
            &mut second,
            &mut memory,
            &BTreeMap::new(),
            &candidates(),
            "zh-Hans",
        )
        .expect("second batch");

        assert_eq!(second.lines[0].translation, "玛丽走了。");
    }

    #[test]
    fn persisted_advisory_glossary_rehydrates_name_alignment() {
        let mut memory = ContextMemory::new();
        memory.load_glossary(&[("Mary".to_owned(), "玛丽".to_owned())]);
        let mut translated = result(&[("1", "⟦N0⟧玛莉⟦/N0⟧走了。")]);

        reconcile_batch(
            &[segment("1", "Mary left.")],
            &mut translated,
            &mut memory,
            &BTreeMap::new(),
            &candidates(),
            "zh-Hans",
        )
        .expect("persisted name alignment");

        assert_eq!(translated.lines[0].translation, "玛丽走了。");
    }

    #[test]
    fn reconciled_cache_rehydrates_run_name_memory() {
        let source = vec![segment("1", "Hi Mary.")];
        let mut translated = result(&[("1", "你好，玛丽。")]);
        translated.glossary_updates.push(GlossaryEntry {
            source: "Mary".to_owned(),
            target: "玛丽".to_owned(),
        });
        let mut memory = ContextMemory::new();

        reconcile_batch(
            &source,
            &mut translated,
            &mut memory,
            &BTreeMap::new(),
            &candidates(),
            "zh-Hans",
        )
        .expect("cached batch");

        assert_eq!(memory.name_translations["mary"], "玛丽");
        assert_eq!(translated.lines[0].translation, "你好，玛丽。");
    }

    #[test]
    fn rejects_malformed_or_moved_markers() {
        let source = vec![segment("1", "Hi Mary.")];
        let markers = select_markers(&source, &candidates());
        let malformed = vec![TranslationLine {
            id: "1".to_owned(),
            translation: "⟦N0⟧玛丽".to_owned(),
        }];
        let moved = vec![TranslationLine {
            id: "1".to_owned(),
            translation: "⟦N1⟧玛丽⟦/N1⟧".to_owned(),
        }];

        assert!(validate_markers(&source, &malformed, &markers).is_err());
        assert!(validate_markers(&source, &moved, &markers).is_err());
    }

    #[test]
    fn rejects_untranslated_kana_name_for_chinese_target() {
        let source = vec![segment("1", "ヒムロ君がいない?")];
        let candidates = vec!["ヒムロ".to_owned()];
        let mut translated = result(&[("1", "⟦N0⟧ヒムロ⟦/N0⟧君不在吗？")]);

        let error = reconcile_batch(
            &source,
            &mut translated,
            &mut ContextMemory::new(),
            &BTreeMap::new(),
            &candidates,
            "zh-Hans",
        )
        .expect_err("kana name must use the Chinese target script");

        assert!(error.to_string().contains("left in source spelling"));
    }
}
