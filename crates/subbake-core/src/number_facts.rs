use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NumberFactComparison {
    Match,
    HardMismatch {
        left: Vec<String>,
        right: Vec<String>,
    },
    Uncertain,
}

#[derive(Debug, Default)]
struct NumberFacts {
    definite: Vec<String>,
    ambiguous: Vec<AmbiguousFact>,
}

#[derive(Debug, Clone)]
struct AmbiguousFact {
    identity: String,
    comparable_value: Option<String>,
}

impl AmbiguousFact {
    fn numeric(value: impl Into<String>) -> Self {
        let value = value.into();
        Self {
            identity: value.clone(),
            comparable_value: Some(value),
        }
    }

    fn raw(value: impl Into<String>) -> Self {
        Self {
            identity: value.into(),
            comparable_value: None,
        }
    }
}

pub(crate) fn compare_number_facts(left: &str, right: &str) -> NumberFactComparison {
    let left = number_facts(left);
    let right = number_facts(right);
    let mut left_ambiguous = left.ambiguous;
    let mut right_ambiguous = right.ambiguous;
    let left_counts = counts(&left.definite);
    let right_counts = counts(&right.definite);
    let keys = left_counts
        .keys()
        .chain(right_counts.keys())
        .cloned()
        .collect::<BTreeSet<_>>();

    for key in keys {
        let left_count = left_counts.get(&key).copied().unwrap_or_default();
        let right_count = right_counts.get(&key).copied().unwrap_or_default();
        if left_count > right_count
            && !consume_ambiguous(&mut right_ambiguous, &key, left_count - right_count)
        {
            return NumberFactComparison::HardMismatch {
                left: sorted(left.definite),
                right: sorted(right.definite),
            };
        }
        if right_count > left_count
            && !consume_ambiguous(&mut left_ambiguous, &key, right_count - left_count)
        {
            return NumberFactComparison::HardMismatch {
                left: sorted(left.definite),
                right: sorted(right.definite),
            };
        }
    }

    let mut left_remaining = left_ambiguous
        .into_iter()
        .map(|fact| fact.identity)
        .collect::<Vec<_>>();
    let mut right_remaining = right_ambiguous
        .into_iter()
        .map(|fact| fact.identity)
        .collect::<Vec<_>>();
    left_remaining.sort();
    right_remaining.sort();
    if left_remaining == right_remaining {
        NumberFactComparison::Match
    } else {
        NumberFactComparison::Uncertain
    }
}

fn counts(values: &[String]) -> BTreeMap<String, usize> {
    values
        .iter()
        .cloned()
        .fold(BTreeMap::new(), |mut map, value| {
            *map.entry(value).or_default() += 1;
            map
        })
}

fn consume_ambiguous(facts: &mut Vec<AmbiguousFact>, value: &str, count: usize) -> bool {
    for _ in 0..count {
        let Some(index) = facts
            .iter()
            .position(|fact| fact.comparable_value.as_deref() == Some(value))
        else {
            return false;
        };
        facts.remove(index);
    }
    true
}

fn sorted(mut values: Vec<String>) -> Vec<String> {
    values.sort();
    values
}

fn number_facts(text: &str) -> NumberFacts {
    let mut facts = NumberFacts::default();
    let characters = text.chars().collect::<Vec<_>>();
    let mut index = 0;
    while index < characters.len() {
        if let Some((value, end)) = cjk_percentage(&characters, index) {
            facts.definite.push("%".to_owned());
            facts.definite.push(value);
            index = end;
            continue;
        }
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
            if let Some(clock_end) = zero_minute_clock_end(&characters, index) {
                push_scaled_digits(&mut facts.definite, &digits, 1);
                index = clock_end;
                continue;
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
                    push_scaled_digits(&mut facts.definite, &sequence, 1);
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
            push_scaled_digits(&mut facts.definite, &digits, multiplier);
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
            classify_cjk_span(&characters, start, index, &mut facts);
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
            facts.definite.push(marker.to_owned());
        }
        index += 1;
    }
    let english = english_number_facts(text);
    facts.definite.extend(english.definite);
    facts.ambiguous.extend(english.ambiguous);
    facts
}

fn classify_cjk_span(characters: &[char], start: usize, end: usize, facts: &mut NumberFacts) {
    let full_span = &characters[start..end];
    let raw = full_span.iter().collect::<String>();
    let following_context = longest_quantity_context(characters, end);
    let trailing_liang_unit = following_context.is_none()
        && full_span.len() > 1
        && full_span.last() == Some(&'两')
        && parse_cjk_cardinal(&full_span[..full_span.len() - 1]).is_some();
    let (span, context) = if trailing_liang_unit {
        (&full_span[..full_span.len() - 1], Some("两"))
    } else {
        (full_span, following_context)
    };
    let pure_digits = span.iter().all(|character| cjk_digit(*character).is_some());

    if pure_digits && span.len() > 1 {
        let value = parse_cjk_digit_sequence(span).map(|value| value.to_string());
        if context.is_some_and(|suffix| DIGIT_SEQUENCE_CONTEXTS.contains(&suffix)) {
            if let Some(value) = value {
                facts.definite.push(value);
            } else {
                facts
                    .ambiguous
                    .push(AmbiguousFact::raw(format!("cjk:{raw}")));
            }
        } else if context.is_some() {
            // Adjacent digits before an ordinary classifier normally denote an
            // approximate range (七八个人), not the integer 78.
            facts
                .ambiguous
                .push(AmbiguousFact::raw(format!("cjk:{raw}")));
        } else if let Some(value) = value {
            facts.ambiguous.push(AmbiguousFact::numeric(value));
        }
        return;
    }

    let Some(value) = parse_cjk_cardinal(span).map(|value| value.to_string()) else {
        facts
            .ambiguous
            .push(AmbiguousFact::raw(format!("cjk:{raw}")));
        return;
    };
    if span.len() == 1 || context.is_none() {
        facts.ambiguous.push(AmbiguousFact::numeric(value));
    } else {
        facts.definite.push(value);
    }
}

const DIGIT_SEQUENCE_CONTEXTS: &[&str] = &["年", "编号", "代码", "号码", "号"];

// Ordered only for readability; longest_quantity_context still selects by
// character count so additions cannot accidentally shadow a longer suffix.
const QUANTITY_CONTEXTS: &[&str] = &[
    "摄氏度",
    "华氏度",
    "个百分点",
    "个小时",
    "个星期",
    "个季度",
    "个世纪",
    "人民币",
    "分钟",
    "秒钟",
    "小时",
    "钟头",
    "星期",
    "季度",
    "世纪",
    "年代",
    "个月",
    "公里",
    "千米",
    "厘米",
    "毫米",
    "英里",
    "英尺",
    "英寸",
    "公斤",
    "千克",
    "毫升",
    "加仑",
    "美元",
    "美金",
    "欧元",
    "英镑",
    "日元",
    "韩元",
    "块钱",
    "等级",
    "编号",
    "代码",
    "号码",
    "个人",
    "点钟",
    "年",
    "岁",
    "月",
    "周",
    "天",
    "日",
    "时",
    "点",
    "刻",
    "分",
    "秒",
    "人",
    "个",
    "位",
    "名",
    "只",
    "条",
    "本",
    "件",
    "张",
    "辆",
    "艘",
    "架",
    "台",
    "部",
    "枚",
    "颗",
    "块",
    "份",
    "家",
    "间",
    "所",
    "场",
    "次",
    "遍",
    "回",
    "趟",
    "轮",
    "集",
    "季",
    "章",
    "页",
    "行",
    "句",
    "字",
    "层",
    "级",
    "号",
    "届",
    "期",
    "队",
    "组",
    "对",
    "双",
    "套",
    "种",
    "米",
    "码",
    "吨",
    "磅",
    "斤",
    "升",
    "克",
    "元",
    "档",
    "阶",
    "星",
    "倍",
    "度",
];

fn longest_quantity_context(characters: &[char], start: usize) -> Option<&'static str> {
    QUANTITY_CONTEXTS
        .iter()
        .copied()
        .filter(|suffix| {
            let suffix = suffix.chars().collect::<Vec<_>>();
            characters[start..].starts_with(&suffix)
        })
        .max_by_key(|suffix| suffix.chars().count())
}

fn zero_minute_clock_end(characters: &[char], colon: usize) -> Option<usize> {
    if characters.get(colon..colon + 3) != Some(&[':', '0', '0']) {
        return None;
    }
    let end = colon + 3;
    characters
        .get(end)
        .is_none_or(|character| !character.is_ascii_digit() && *character != ':')
        .then_some(end)
}

fn cjk_percentage(characters: &[char], start: usize) -> Option<(String, usize)> {
    if characters[start..].starts_with(&['百', '分', '百']) {
        return Some(("100".to_owned(), start + 3));
    }
    if !characters[start..].starts_with(&['百', '分', '之']) {
        return None;
    }
    let number_start = start + 3;
    let mut end = number_start;
    if characters.get(end).is_some_and(char::is_ascii_digit) {
        while characters.get(end).is_some_and(char::is_ascii_digit) {
            end += 1;
        }
        let digits = characters[number_start..end].iter().collect::<String>();
        return Some((normalize_digits(&digits).to_owned(), end));
    }
    while characters
        .get(end)
        .is_some_and(|character| is_cjk_numeral(*character))
    {
        end += 1;
    }
    if characters.get(number_start..end) == Some(&['百']) {
        return Some(("100".to_owned(), end));
    }
    (end > number_start)
        .then(|| parse_cjk_cardinal(&characters[number_start..end]))
        .flatten()
        .map(|value| (value.to_string(), end))
}

fn push_scaled_digits(tokens: &mut Vec<String>, digits: &str, multiplier: u128) {
    if digits.is_empty() {
        return;
    }
    let normalized = normalize_digits(digits);
    let value = normalized.parse::<u128>().ok();
    tokens.push(
        value
            .and_then(|value| value.checked_mul(multiplier))
            .map(|value| value.to_string())
            .unwrap_or_else(|| normalized.to_owned()),
    );
}

fn normalize_digits(digits: &str) -> &str {
    let normalized = digits.trim_start_matches('0');
    if normalized.is_empty() {
        "0"
    } else {
        normalized
    }
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

fn parse_cjk_digit_sequence(characters: &[char]) -> Option<u128> {
    characters.iter().try_fold(0_u128, |value, character| {
        value.checked_mul(10)?.checked_add(cjk_digit(*character)?)
    })
}

fn parse_cjk_cardinal(characters: &[char]) -> Option<u128> {
    if characters.is_empty() {
        return None;
    }
    if characters
        .iter()
        .all(|character| cjk_digit(*character).is_some())
    {
        return (characters.len() == 1)
            .then(|| cjk_digit(characters[0]))
            .flatten();
    }

    let mut total = 0_u128;
    let mut section = 0_u128;
    let mut pending_digit = None;
    let mut last_small_unit = 10_000_u128;
    let mut last_large_unit = u128::MAX;
    let mut section_has_value = false;
    let mut zero_pending = false;

    for (index, character) in characters.iter().copied().enumerate() {
        if let Some(digit) = cjk_digit(character) {
            if digit == 0 {
                if index == 0
                    || index + 1 == characters.len()
                    || zero_pending
                    || pending_digit.is_some()
                {
                    return None;
                }
                zero_pending = true;
            } else {
                if pending_digit.is_some() {
                    return None;
                }
                pending_digit = Some(digit);
                zero_pending = false;
            }
            continue;
        }

        let unit = cjk_unit_value(character)?;
        if unit < 10_000 {
            if zero_pending || unit >= last_small_unit {
                return None;
            }
            let digit = pending_digit
                .take()
                .or_else(|| (unit == 10 && !section_has_value && section == 0).then_some(1))?;
            section = section.checked_add(digit.checked_mul(unit)?)?;
            section_has_value = true;
            last_small_unit = unit;
        } else {
            if zero_pending || unit >= last_large_unit {
                return None;
            }
            if let Some(digit) = pending_digit.take() {
                section = section.checked_add(digit)?;
            }
            if section == 0 {
                return None;
            }
            total = total.checked_add(section.checked_mul(unit)?)?;
            section = 0;
            section_has_value = false;
            last_small_unit = 10_000;
            last_large_unit = unit;
        }
    }
    if zero_pending {
        return None;
    }
    if let Some(digit) = pending_digit {
        section = section.checked_add(digit)?;
    }
    total.checked_add(section)
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

fn cjk_unit_value(character: char) -> Option<u128> {
    match character {
        '十' => Some(10),
        '百' => Some(100),
        '千' => Some(1_000),
        '万' => Some(10_000),
        '亿' => Some(100_000_000),
        _ => None,
    }
}

fn english_number_facts(text: &str) -> NumberFacts {
    let words = text
        // Keep digit-only tokens as boundaries so `a 12 million` does not
        // look like the article-plus-scale expression `a million`.
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    let mut facts = NumberFacts::default();
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
                    facts.definite.push(scale.to_string());
                }
                continue;
            }
            if english_number_value(&phrase[0]).is_some_and(|value| value < 10) {
                if let Some(value) = parse_english_number(phrase) {
                    facts
                        .ambiguous
                        .push(AmbiguousFact::numeric(value.to_string()));
                }
                continue;
            }
        }
        if let Some(value) = parse_english_number(phrase) {
            facts.definite.push(value.to_string());
        }
    }
    facts
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_ambiguous_cjk_number_glyph_expressions_as_uncertain() {
        for text in ["乱七八糟", "十万火急", "万一", "三三两两", "七八个人"] {
            assert_eq!(
                compare_number_facts("This has no numeric fact.", text),
                NumberFactComparison::Uncertain,
                "{text}"
            );
        }
    }

    #[test]
    fn recognizes_valid_cjk_cardinals_only_in_quantity_context() {
        for (left, right) in [
            ("20,000", "两万"),
            ("12 years", "十二年"),
            ("the 80s", "八十年代"),
            ("70 miles", "七十英里"),
            ("25 times", "二十五次"),
            ("25 grams", "二十五克"),
            ("8 liang", "八两"),
            ("13 million dollars", "一千三百万美元"),
        ] {
            assert_eq!(
                compare_number_facts(left, right),
                NumberFactComparison::Match,
                "{left} / {right}"
            );
        }
    }

    #[test]
    fn rejects_real_numeric_changes_and_approximate_digit_sequences() {
        for (left, right) in [
            ("12 years old", "十三岁"),
            ("$12", "$13"),
            ("80%", "百分之七十"),
            ("Meet at 3:30", "三点见"),
            ("78 people", "七八个人"),
        ] {
            assert!(
                matches!(
                    compare_number_facts(left, right),
                    NumberFactComparison::HardMismatch { .. }
                ),
                "{left} / {right}"
            );
        }
    }

    #[test]
    fn rejects_invalid_cjk_unit_order_and_zero_placement() {
        for value in ["万一", "十十个", "一百百个", "零十个", "一百零个"] {
            assert_eq!(
                compare_number_facts("nothing", value),
                NumberFactComparison::Uncertain,
                "{value}"
            );
        }
    }
}
