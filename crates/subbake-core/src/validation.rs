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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FinalValidationIssue {
    pub segment_id: String,
    pub message: String,
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
    let issues = final_validation_issues(
        source,
        translated,
        required_glossary,
        source_language,
        target_language,
        policy,
    )?;
    if issues.is_empty() {
        return Ok(());
    }
    Err(final_validation_error(&issues))
}

pub(crate) fn final_validation_issues(
    source: &[SubtitleSegment],
    translated: &[SubtitleSegment],
    required_glossary: &BTreeMap<String, String>,
    source_language: &str,
    target_language: &str,
    policy: FinalValidationPolicy,
) -> CoreResult<Vec<FinalValidationIssue>> {
    validate_full_alignment(source, translated)?;

    let mut issues = Vec::new();
    let cross_language = !source_language.eq_ignore_ascii_case(target_language);
    for (source, translated) in source.iter().zip(translated) {
        let mut segment_issues = Vec::new();
        validate_final_segment(
            source,
            translated,
            required_glossary,
            cross_language,
            policy,
            &mut segment_issues,
        );
        issues.extend(
            segment_issues
                .into_iter()
                .map(|message| FinalValidationIssue {
                    segment_id: source.id.clone(),
                    message,
                }),
        );
    }
    Ok(issues)
}

pub(crate) fn final_validation_error(issues: &[FinalValidationIssue]) -> CoreError {
    let visible = issues
        .iter()
        .take(8)
        .map(|issue| issue.message.clone())
        .collect::<Vec<_>>()
        .join("; ");
    let remainder = issues.len().saturating_sub(8);
    CoreError::InvalidTranslation(format!(
        "final output validation failed with {} issue(s): {visible}{}",
        issues.len(),
        if remainder == 0 {
            String::new()
        } else {
            format!("; and {remainder} more")
        }
    ))
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

    let (source_facts, translated_facts) =
        comparable_factual_tokens(&source.text, &translated.text);
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

#[derive(Debug, Default)]
struct FactualTokens {
    strong: Vec<String>,
    weak: Vec<String>,
}

fn comparable_factual_tokens(source: &str, translated: &str) -> (Vec<String>, Vec<String>) {
    let source = factual_tokens(source);
    let translated = factual_tokens(translated);
    let mut source_strong = source.strong;
    let mut translated_strong = translated.strong;
    promote_matching_weak(&mut source_strong, &source.weak, &translated_strong);
    promote_matching_weak(&mut translated_strong, &translated.weak, &source_strong);
    source_strong.sort();
    translated_strong.sort();
    (source_strong, translated_strong)
}

fn promote_matching_weak(strong: &mut Vec<String>, weak: &[String], opposite: &[String]) {
    for token in weak {
        let expected = opposite.iter().filter(|value| *value == token).count();
        let present = strong.iter().filter(|value| *value == token).count();
        if present < expected {
            strong.push(token.clone());
        }
    }
}

fn factual_tokens(text: &str) -> FactualTokens {
    let mut tokens = FactualTokens::default();
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
            if digits.len() == 1 {
                let mut sequence = digits.clone();
                let mut sequence_end = index;
                while sequence_end + 1 < characters.len()
                    && characters[sequence_end] == '-'
                    && characters[sequence_end + 1].is_ascii_digit()
                    && characters
                        .get(sequence_end + 2)
                        .is_none_or(|character| !character.is_ascii_digit())
                {
                    sequence.push(characters[sequence_end + 1]);
                    sequence_end += 2;
                }
                if sequence.len() > 1 {
                    push_scaled_digits(&mut tokens.strong, &sequence, 1);
                    index = sequence_end;
                    continue;
                }
            }
            let mut multiplier = 1_u128;
            let mut suffix_start = index;
            while suffix_start < characters.len() && characters[suffix_start].is_whitespace() {
                suffix_start += 1;
            }
            if let Some((scale, suffix_end)) = numeric_scale(&characters, suffix_start) {
                multiplier = scale;
                index = suffix_end;
            }
            push_scaled_digits(&mut tokens.strong, &digits, multiplier);
            continue;
        }
        if is_cjk_numeral(character) {
            let start = index;
            while index < characters.len() && is_cjk_numeral(characters[index]) {
                index += 1;
            }
            if start > 0 && characters[start - 1] == '第' {
                continue;
            }
            // A single CJK digit is commonly part of an article, pronoun, or
            // idiom (一个, 之一, 一天) rather than a hard numeric fact. Treat
            // it like a standalone English zero-to-nine word and leave exact
            // digits plus compound/scaled number expressions as strong facts.
            if index - start == 1 && cjk_digit(characters[start]).is_some() {
                if let Some(value) = parse_cjk_number(&characters[start..index]) {
                    tokens.weak.push(value.to_string());
                }
                continue;
            }
            if index - start == 1
                && matches!(characters[start], '十' | '百' | '千' | '万' | '亿')
                && start > 0
                && matches!(characters[start - 1], '数' | '几')
            {
                continue;
            }
            if let Some(value) = parse_cjk_number(&characters[start..index]) {
                let pure_repeated_digits = characters[start..index]
                    .iter()
                    .all(|character| cjk_digit(*character).is_some())
                    && characters[start..index]
                        .windows(2)
                        .all(|pair| pair[0] == pair[1]);
                if pure_repeated_digits {
                    tokens.weak.push(value.to_string());
                } else {
                    tokens.strong.push(value.to_string());
                }
            }
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
            tokens.strong.push(marker.to_owned());
        }
        index += 1;
    }
    let english = english_number_tokens(text);
    tokens.strong.extend(english.strong);
    tokens.weak.extend(english.weak);
    tokens
}

fn push_scaled_digits(tokens: &mut Vec<String>, digits: &str, multiplier: u128) {
    if digits.is_empty() {
        return;
    }
    let normalized = digits.trim_start_matches('0');
    let normalized = if normalized.is_empty() {
        "0"
    } else {
        normalized
    };
    let value = normalized.parse::<u128>().ok();
    tokens.push(
        value
            .and_then(|value| value.checked_mul(multiplier))
            .map(|value| value.to_string())
            .unwrap_or_else(|| normalized.to_owned()),
    );
}

fn numeric_scale(characters: &[char], start: usize) -> Option<(u128, usize)> {
    let cjk = characters.get(start).copied().and_then(|character| {
        let scale = match character {
            '百' => 100,
            '千' => 1_000,
            '万' => 10_000,
            '亿' => 100_000_000,
            _ => return None,
        };
        Some((scale, start + 1))
    });
    if cjk.is_some() {
        return cjk;
    }
    let mut end = start;
    while end < characters.len() && characters[end].is_ascii_alphabetic() {
        end += 1;
    }
    let word = characters[start..end]
        .iter()
        .collect::<String>()
        .to_ascii_lowercase();
    let scale = match word.as_str() {
        "hundred" => 100,
        "thousand" => 1_000,
        "million" => 1_000_000,
        "billion" => 1_000_000_000,
        "trillion" => 1_000_000_000_000,
        _ => return None,
    };
    Some((scale, end))
}

fn is_cjk_numeral(character: char) -> bool {
    matches!(
        character,
        '零' | '〇'
            | '一'
            | '二'
            | '两'
            | '三'
            | '四'
            | '五'
            | '六'
            | '七'
            | '八'
            | '九'
            | '十'
            | '百'
            | '千'
            | '万'
            | '亿'
    )
}

fn parse_cjk_number(characters: &[char]) -> Option<u128> {
    if characters.is_empty() {
        return None;
    }
    let has_unit = characters
        .iter()
        .any(|character| matches!(character, '十' | '百' | '千' | '万' | '亿'));
    if !has_unit {
        return characters.iter().try_fold(0_u128, |value, character| {
            value.checked_mul(10)?.checked_add(cjk_digit(*character)?)
        });
    }

    let mut total = 0_u128;
    let mut section = 0_u128;
    let mut number = 0_u128;
    for character in characters {
        if let Some(digit) = cjk_digit(*character) {
            number = digit;
            continue;
        }
        let unit = match character {
            '十' => 10,
            '百' => 100,
            '千' => 1_000,
            '万' => 10_000,
            '亿' => 100_000_000,
            _ => return None,
        };
        if unit < 10_000 {
            section = section.checked_add(number.max(1).checked_mul(unit)?)?;
        } else {
            let scaled = section.checked_add(number)?.max(1).checked_mul(unit)?;
            total = total.checked_add(scaled)?;
            section = 0;
        }
        number = 0;
    }
    total.checked_add(section)?.checked_add(number)
}

fn cjk_digit(character: char) -> Option<u128> {
    match character {
        '零' | '〇' => Some(0),
        '一' => Some(1),
        '二' | '两' => Some(2),
        '三' => Some(3),
        '四' => Some(4),
        '五' => Some(5),
        '六' => Some(6),
        '七' => Some(7),
        '八' => Some(8),
        '九' => Some(9),
        _ => None,
    }
}

fn english_number_tokens(text: &str) -> FactualTokens {
    let words = text
        // Keep digit-only tokens as boundaries so `a 12 million` does not
        // look like the article-plus-scale expression `a million`.
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    let mut tokens = FactualTokens::default();
    let mut index = 0;
    while index < words.len() {
        let Some(_) = english_number_value(&words[index]) else {
            index += 1;
            continue;
        };
        let start = index;
        while index < words.len() && english_number_value(&words[index]).is_some() {
            index += 1;
        }
        let phrase = &words[start..index];
        if phrase.len() == 1 {
            if let Some(scale) = english_scale(&phrase[0]) {
                if start > 0 && matches!(words[start - 1].as_str(), "a" | "an") {
                    tokens.strong.push(scale.to_string());
                }
                continue;
            }
            if english_number_value(&phrase[0]).is_some_and(|value| value < 10) {
                // Standalone small number words are too ambiguous in natural
                // dialogue (one more, one of, split in two, "oh"). Numeric
                // sequences such as Zero-One remain strong facts.
                if let Some(value) = parse_english_number(phrase) {
                    tokens.weak.push(value.to_string());
                }
                continue;
            }
        }
        if let Some(value) = parse_english_number(phrase) {
            tokens.strong.push(value.to_string());
        }
    }
    tokens
}

fn parse_english_number(words: &[String]) -> Option<u128> {
    if words.len() > 1 && words.iter().all(|word| english_digit(word).is_some()) {
        return words.iter().try_fold(0_u128, |value, word| {
            value.checked_mul(10)?.checked_add(english_digit(word)?)
        });
    }
    let mut total = 0_u128;
    let mut current = 0_u128;
    for word in words {
        if let Some(digit) = english_digit(word) {
            current = current.checked_add(digit)?;
        } else if let Some(value) = english_small_number(word) {
            current = current.checked_add(value)?;
        } else {
            let scale = english_scale(word)?;
            if scale == 100 {
                current = current.max(1).checked_mul(scale)?;
            } else {
                total = total.checked_add(current.max(1).checked_mul(scale)?)?;
                current = 0;
            }
        }
    }
    total.checked_add(current)
}

fn english_number_value(word: &str) -> Option<u128> {
    english_digit(word)
        .or_else(|| english_small_number(word))
        .or_else(|| english_scale(word))
}

fn english_digit(word: &str) -> Option<u128> {
    match word {
        "zero" | "oh" => Some(0),
        "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        _ => None,
    }
}

fn english_small_number(word: &str) -> Option<u128> {
    match word {
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "thirteen" => Some(13),
        "fourteen" => Some(14),
        "fifteen" => Some(15),
        "sixteen" => Some(16),
        "seventeen" => Some(17),
        "eighteen" => Some(18),
        "nineteen" => Some(19),
        "twenty" => Some(20),
        "thirty" => Some(30),
        "forty" => Some(40),
        "fifty" => Some(50),
        "sixty" => Some(60),
        "seventy" => Some(70),
        "eighty" => Some(80),
        "ninety" => Some(90),
        _ => None,
    }
}

fn english_scale(word: &str) -> Option<u128> {
    match word {
        "hundred" => Some(100),
        "thousand" => Some(1_000),
        "million" => Some(1_000_000),
        "billion" => Some(1_000_000_000),
        "trillion" => Some(1_000_000_000_000),
        _ => None,
    }
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
    fn final_validation_normalizes_english_and_chinese_number_expressions() {
        let pairs = [
            ("it's nearly 20,000 years old.", "将近有两万年的历史了。"),
            (
                "For 100,000 years, our civilization flourished.",
                "十万年来，我们的文明繁荣昌盛。",
            ),
            ("For 33 years, we prepared.", "三十三年间，我们一直在准备。"),
            (
                "We'll make a second gun run on a heading of two, one-two degrees.",
                "我们将在航向212度进行第二轮扫射。",
            ),
            ("This is Badger Zero-One.", "这里是獾01。"),
            (
                "one of your surveillance drones is a 12 million dollar piece of equipment",
                "你的监视无人机之一是价值1200万美元的设备",
            ),
        ];
        let source = pairs
            .iter()
            .enumerate()
            .map(|(index, (source, _))| segment(&(index + 1).to_string(), source, None))
            .collect::<Vec<_>>();
        let translated = pairs
            .iter()
            .enumerate()
            .map(|(index, (_, translated))| segment(&(index + 1).to_string(), translated, None))
            .collect::<Vec<_>>();

        validate_final_output(
            &source,
            &translated,
            &BTreeMap::new(),
            "English",
            "Chinese",
            FinalValidationPolicy::default(),
        )
        .expect("semantically equal number expressions should pass");
    }

    #[test]
    fn final_validation_ignores_ambiguous_small_number_words_and_cjk_articles() {
        let pairs = [
            (
                "This is Coast Guard 6510. We're gonna make one more pass.",
                "这里是海岸警卫队6510号。我们再飞一圈。",
            ),
            (
                "He'll be an outcast. A freak.",
                "他会成为被排挤的人，一个异类。",
            ),
            (
                "one of thousands launched into the void",
                "发射到虚空中的数千艘之一",
            ),
            ("We split in two.", "我们一分为二。"),
            ("Oh, what are you doing?", "哦，你在做什么？"),
            ("Two weeks leave.", "停职两周。"),
        ];
        let source = pairs
            .iter()
            .enumerate()
            .map(|(index, (source, _))| segment(&(index + 1).to_string(), source, None))
            .collect::<Vec<_>>();
        let translated = pairs
            .iter()
            .enumerate()
            .map(|(index, (_, translated))| segment(&(index + 1).to_string(), translated, None))
            .collect::<Vec<_>>();

        validate_final_output(
            &source,
            &translated,
            &BTreeMap::new(),
            "English",
            "Chinese",
            FinalValidationPolicy::default(),
        )
        .expect("ambiguous small-number grammar should not be treated as changed facts");
    }

    #[test]
    fn final_validation_treats_article_plus_scale_as_a_strong_number() {
        let source = vec![segment("1", "the DNA of a billion people", None)];
        let translated = vec![segment("1", "一亿人的DNA", None)];

        let error = validate_final_output(
            &source,
            &translated,
            &BTreeMap::new(),
            "English",
            "Chinese",
            FinalValidationPolicy::default(),
        )
        .expect_err("one billion must not equal one hundred million");

        assert!(error.to_string().contains("expected [1000000000]"));
        assert!(error.to_string().contains("got [100000000]"));
    }

    #[test]
    fn final_validation_normalizes_call_signs_but_rejects_dropped_components() {
        let source = vec![
            segment("1", "Under-1-2 calling Guardian.", None),
            segment("2", "Tally-3 target.", None),
        ];
        let translated = vec![
            segment("1", "水下一二呼叫守护者。", None),
            segment("2", "发现三个目标。", None),
        ];
        validate_final_output(
            &source,
            &translated,
            &BTreeMap::new(),
            "English",
            "Chinese",
            FinalValidationPolicy::default(),
        )
        .expect("equivalent call signs and explicit small quantities should pass");

        let error = validate_final_output(
            &[segment("1", "Copy, 1-1.", None)],
            &[segment("1", "收到，一号机。", None)],
            &BTreeMap::new(),
            "English",
            "Chinese",
            FinalValidationPolicy::default(),
        )
        .expect_err("dropping half of a call sign must still fail");
        assert!(error.to_string().contains("expected [11]"));
    }

    #[test]
    fn final_validation_still_rejects_a_real_number_change_after_normalization() {
        let source = vec![segment("1", "The repair costs 12 million dollars.", None)];
        let translated = vec![segment("1", "维修费用为一千三百万美元。", None)];

        let error = validate_final_output(
            &source,
            &translated,
            &BTreeMap::new(),
            "English",
            "Chinese",
            FinalValidationPolicy::default(),
        )
        .expect_err("12 million must not equal 13 million");

        assert!(error.to_string().contains("expected [12000000]"));
        assert!(error.to_string().contains("got [13000000]"));
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
