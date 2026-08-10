use std::collections::{BTreeMap, HashSet};

use crate::entities::{SubtitleSegment, TranslationLine};
use crate::error::{CoreError, CoreResult};
use crate::formatting::formatting_tokens;
use crate::term_matcher::TermMatcher;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FinalValidationPolicy {
    pub max_characters_per_second: Option<f64>,
    pub max_characters_per_line: Option<usize>,
    pub max_lines: Option<usize>,
}

pub fn validate_translation_batch(
    source: &[SubtitleSegment],
    lines: &[TranslationLine],
) -> CoreResult<()> {
    if source.len() != lines.len() {
        return Err(CoreError::InvalidTranslation(format!(
            "expected {} translated line(s), got {}",
            source.len(),
            lines.len()
        )));
    }

    let source_ids = source
        .iter()
        .map(|segment| segment.id.as_str())
        .collect::<HashSet<_>>();
    for line in lines {
        if !source_ids.contains(line.id.as_str()) {
            return Err(CoreError::InvalidTranslation(format!(
                "unexpected translated id `{}`",
                line.id
            )));
        }
    }

    for segment in source {
        if segment.text.trim().is_empty() {
            continue;
        }
        let translation = lines
            .iter()
            .find(|line| line.id == segment.id)
            .ok_or_else(|| CoreError::InvalidTranslation(format!("missing id `{}`", segment.id)))?;
        if translation.translation.trim().is_empty() {
            return Err(CoreError::InvalidTranslation(format!(
                "empty translation for id `{}`",
                segment.id
            )));
        }
        if formatting_tokens(&segment.text) != formatting_tokens(&translation.translation) {
            return Err(CoreError::InvalidTranslation(format!(
                "formatting mismatch for id `{}`",
                segment.id
            )));
        }
    }

    Ok(())
}

pub fn validate_full_alignment(
    source: &[SubtitleSegment],
    translated: &[SubtitleSegment],
) -> CoreResult<()> {
    if source.len() != translated.len() {
        return Err(CoreError::InvalidTranslation(format!(
            "source has {} segment(s), translated has {}",
            source.len(),
            translated.len()
        )));
    }

    for (source, translated) in source.iter().zip(translated) {
        if source.id != translated.id {
            return Err(CoreError::InvalidTranslation(format!(
                "id mismatch: expected `{}`, got `{}`",
                source.id, translated.id
            )));
        }
    }

    Ok(())
}

pub fn validate_final_output(
    source: &[SubtitleSegment],
    translated: &[SubtitleSegment],
    required_glossary: &BTreeMap<String, String>,
    source_language: &str,
    target_language: &str,
    policy: FinalValidationPolicy,
) -> CoreResult<()> {
    validate_full_alignment(source, translated)?;

    let mut issues = Vec::new();
    let cross_language = !source_language.eq_ignore_ascii_case(target_language);
    for (source, translated) in source.iter().zip(translated) {
        validate_final_segment(
            source,
            translated,
            required_glossary,
            cross_language,
            policy,
            &mut issues,
        );
    }

    if issues.is_empty() {
        return Ok(());
    }
    let visible = issues
        .iter()
        .take(8)
        .cloned()
        .collect::<Vec<_>>()
        .join("; ");
    let remainder = issues.len().saturating_sub(8);
    Err(CoreError::InvalidTranslation(format!(
        "final output validation failed with {} issue(s): {visible}{}",
        issues.len(),
        if remainder == 0 {
            String::new()
        } else {
            format!("; and {remainder} more")
        }
    )))
}

fn validate_final_segment(
    source: &SubtitleSegment,
    translated: &SubtitleSegment,
    required_glossary: &BTreeMap<String, String>,
    cross_language: bool,
    policy: FinalValidationPolicy,
    issues: &mut Vec<String>,
) {
    let source_text = source.text.trim();
    let translated_text = translated.text.trim();
    if !source_text.is_empty() && translated_text.is_empty() {
        issues.push(format!("line {} is empty", source.id));
        return;
    }
    if source_text.is_empty() {
        return;
    }

    if formatting_tokens(&source.text) != formatting_tokens(&translated.text) {
        issues.push(format!("line {} has a formatting mismatch", source.id));
    }

    let source_facts = factual_tokens(&source.text);
    let translated_facts = factual_tokens(&translated.text);
    if source_facts != translated_facts {
        issues.push(format!(
            "line {} changes numbers, dates, amounts, or percentages (expected {}, got {})",
            source.id,
            display_tokens(&source_facts),
            display_tokens(&translated_facts),
        ));
    }

    for (term, target) in TermMatcher::case_insensitive().missing_required(
        &source.text,
        &translated.text,
        required_glossary,
    ) {
        issues.push(format!(
            "line {} does not use required glossary translation `{term}` -> `{target}`",
            source.id
        ));
    }

    if cross_language && is_suspiciously_untranslated(source_text, translated_text) {
        issues.push(format!("line {} appears untranslated", source.id));
    }

    let lines = translated_text.lines().collect::<Vec<_>>();
    if let Some(limit) = policy.max_lines
        && lines.len() > limit
    {
        issues.push(format!(
            "line {} has {} subtitle lines (limit {limit})",
            source.id,
            lines.len()
        ));
    }
    if let Some(limit) = policy.max_characters_per_line
        && let Some(longest) = lines.iter().map(|line| visible_characters(line)).max()
        && longest > limit
    {
        issues.push(format!(
            "line {} has {longest} visible characters on one line (limit {limit})",
            source.id
        ));
    }
    if let Some(limit) = policy.max_characters_per_second
        && let Some(duration) = segment_duration_seconds(translated)
    {
        let speed = visible_characters(translated_text) as f64 / duration;
        if speed > limit {
            issues.push(format!(
                "line {} has a reading speed of {speed:.1} characters per second (limit {limit:.1})",
                source.id
            ));
        }
    }
}

fn factual_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let characters = text.chars().collect::<Vec<_>>();
    let mut index = 0;
    while index < characters.len() {
        let character = characters[index];
        if character.is_ascii_digit() {
            let mut digits = String::new();
            while index < characters.len() && characters[index].is_ascii_digit() {
                digits.push(characters[index]);
                index += 1;
            }
            while index < characters.len() && matches!(characters[index], ',' | '.') {
                let group_start = index + 1;
                let mut group_end = group_start;
                while group_end < characters.len() && characters[group_end].is_ascii_digit() {
                    group_end += 1;
                }
                if group_end.saturating_sub(group_start) != 3 {
                    break;
                }
                digits.extend(characters[group_start..group_end].iter());
                index = group_end;
            }
            push_digits(&mut tokens, &digits);
            continue;
        }
        let marker = match character {
            '%' | '％' => Some("%"),
            '$' => Some("$"),
            '€' => Some("€"),
            '£' => Some("£"),
            '¥' | '￥' => Some("¥"),
            '₩' => Some("₩"),
            '₹' => Some("₹"),
            _ => None,
        };
        if let Some(marker) = marker {
            tokens.push(marker.to_owned());
        }
        index += 1;
    }
    tokens.sort();
    tokens
}

fn push_digits(tokens: &mut Vec<String>, digits: &str) {
    if digits.is_empty() {
        return;
    }
    let normalized = digits.trim_start_matches('0');
    tokens.push(if normalized.is_empty() {
        "0".to_owned()
    } else {
        normalized.to_owned()
    });
}

fn display_tokens(tokens: &[String]) -> String {
    if tokens.is_empty() {
        "none".to_owned()
    } else {
        format!("[{}]", tokens.join(", "))
    }
}

fn is_suspiciously_untranslated(source: &str, translated: &str) -> bool {
    let source = normalize_visible_text(source);
    let translated = normalize_visible_text(translated);
    source == translated
        && source.chars().count() >= 4
        && source.chars().any(char::is_alphabetic)
        && (source.chars().any(char::is_whitespace)
            || source
                .chars()
                .any(|character| character.is_ascii_punctuation())
            || source
                .chars()
                .any(|character| character.is_alphabetic() && !character.is_ascii()))
}

fn normalize_visible_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn visible_characters(text: &str) -> usize {
    let total = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .count();
    formatting_tokens(text).iter().fold(total, |count, token| {
        count.saturating_sub(
            token
                .chars()
                .filter(|character| !character.is_whitespace())
                .count(),
        )
    })
}

fn segment_duration_seconds(segment: &SubtitleSegment) -> Option<f64> {
    let (start, end) = segment
        .start
        .as_deref()
        .zip(segment.end.as_deref())
        .and_then(|(start, end)| parse_timestamp(start).zip(parse_timestamp(end)))?;
    (end > start).then_some(end - start)
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
    use super::*;

    #[test]
    fn final_validation_accepts_required_terms_facts_and_formatting() {
        let source = vec![segment(
            "1",
            "<i>The Lord paid $1,000 on 2024-05-01 (50%).</i>",
            Some(("00:00:00,000", "00:00:04,000")),
        )];
        let translated = vec![segment(
            "1",
            "<i>勋爵在2024年5月1日支付了$1000（50%）。</i>",
            Some(("00:00:00,000", "00:00:04,000")),
        )];

        validate_final_output(
            &source,
            &translated,
            &BTreeMap::from([("Lord".to_owned(), "勋爵".to_owned())]),
            "English",
            "Chinese",
            FinalValidationPolicy::default(),
        )
        .expect("valid final output");
    }

    #[test]
    fn final_validation_rejects_glossary_facts_formatting_and_omissions() {
        let source = vec![
            segment("1", "<i>The Lord paid $20 (50%).</i>", None),
            segment("2", "Translate this sentence.", None),
            segment("3", "Do not leave this empty.", None),
        ];
        let translated = vec![
            segment("1", "领主支付了$21（40%）。", None),
            segment("2", "Translate this sentence.", None),
            segment("3", "", None),
        ];

        let error = validate_final_output(
            &source,
            &translated,
            &BTreeMap::from([("Lord".to_owned(), "勋爵".to_owned())]),
            "English",
            "Chinese",
            FinalValidationPolicy::default(),
        )
        .expect_err("invalid final output");
        let message = error.to_string();

        assert!(message.contains("formatting mismatch"));
        assert!(message.contains("numbers, dates, amounts, or percentages"));
        assert!(message.contains("`Lord` -> `勋爵`"));
        assert!(message.contains("appears untranslated"));
        assert!(message.contains("line 3 is empty"));
    }

    #[test]
    fn final_validation_uses_term_boundaries_and_inflections() {
        let source = vec![segment("1", "The actors left the theater.", None)];
        let translated = vec![segment("1", "演员离开了剧院。", None)];
        let glossary = BTreeMap::from([
            ("actor".to_owned(), "演员".to_owned()),
            ("he".to_owned(), "他".to_owned()),
        ]);

        validate_final_output(
            &source,
            &translated,
            &glossary,
            "English",
            "Chinese",
            FinalValidationPolicy::default(),
        )
        .expect("an inflected whole-word match should pass without a substring false positive");
    }

    #[test]
    fn final_validation_enforces_configured_readability_limits() {
        let source = vec![segment(
            "1",
            "A readable source sentence.",
            Some(("00:00:00,000", "00:00:01,000")),
        )];
        let translated = vec![segment(
            "1",
            "第一行太长了\n第二行\n第三行",
            Some(("00:00:00,000", "00:00:01,000")),
        )];

        let error = validate_final_output(
            &source,
            &translated,
            &BTreeMap::new(),
            "English",
            "Chinese",
            FinalValidationPolicy {
                max_characters_per_second: Some(5.0),
                max_characters_per_line: Some(4),
                max_lines: Some(2),
            },
        )
        .expect_err("readability limits should be enforced");
        let message = error.to_string();

        assert!(message.contains("subtitle lines"));
        assert!(message.contains("visible characters"));
        assert!(message.contains("reading speed"));
    }

    fn segment(id: &str, text: &str, timing: Option<(&str, &str)>) -> SubtitleSegment {
        SubtitleSegment {
            id: id.to_owned(),
            text: text.to_owned(),
            start: timing.map(|(start, _)| start.to_owned()),
            end: timing.map(|(_, end)| end.to_owned()),
            identifier: None,
            settings: None,
        }
    }
}
