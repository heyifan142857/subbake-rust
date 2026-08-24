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
    validate_unique_segment_ids(source, "source")?;
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
    let mut translated_ids = HashSet::with_capacity(lines.len());
    for line in lines {
        if !translated_ids.insert(line.id.as_str()) {
            return Err(CoreError::InvalidTranslation(format!(
                "duplicate translated id `{}`",
                line.id
            )));
        }
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
    validate_unique_segment_ids(source, "source")?;
    validate_unique_segment_ids(translated, "translated")?;
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

pub fn validate_unique_segment_ids(
    segments: &[SubtitleSegment],
    collection: &str,
) -> CoreResult<()> {
    let mut seen = HashSet::with_capacity(segments.len());
    for segment in segments {
        if !seen.insert(segment.id.as_str()) {
            return Err(CoreError::InvalidTranslation(format!(
                "duplicate {collection} segment id `{}`",
                segment.id
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
    fn full_alignment_rejects_duplicate_internal_ids() {
        let source = vec![segment("1", "first", None), segment("1", "second", None)];
        let translated = source.clone();

        let error = validate_full_alignment(&source, &translated)
            .expect_err("duplicate internal ids must fail before alignment");
        assert!(
            error
                .to_string()
                .contains("duplicate source segment id `1`")
        );
    }

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
    fn final_validation_accepts_english_and_chinese_number_expressions() {
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
    fn final_validation_accepts_clock_hours_cjk_percentages_and_numeric_idioms() {
        let pairs = [
            ("Some of y'all ain't gonna see 3:00.", "有些人活不到三点。"),
            (
                "It will be 100% honesty and clear communication.",
                "我们会百分之百坦诚，清楚沟通。",
            ),
            ("Takes one to know one.", "半斤八两。"),
            ("The battery is at 80%.", "电量为百分之八十。"),
            ("We need 100% honesty.", "我们要百分百坦诚。"),
            ("It has existed for millennia.", "它已经存在千百年了。"),
            ("Tenssica.", "十西卡。"),
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
        .expect("equivalent time, percentage, and idiom expressions should pass");
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
    fn final_validation_does_not_block_ambiguous_cjk_number_glyphs() {
        for (index, text) in ["乱七八糟", "十万火急", "万一", "三三两两", "七八个人"]
            .into_iter()
            .enumerate()
        {
            validate_final_output(
                &[segment(
                    &(index + 1).to_string(),
                    "There is no explicit numeric fact here.",
                    None,
                )],
                &[segment(&(index + 1).to_string(), text, None)],
                &BTreeMap::new(),
                "English",
                "Chinese",
                FinalValidationPolicy::default(),
            )
            .unwrap_or_else(|error| panic!("{text} should be uncertain, not hard: {error}"));
        }
    }

    #[test]
    fn final_validation_leaves_number_consistency_to_cinema_review() {
        let pairs = [
            ("Two. One.", "二。一。"),
            ("Oh. Oh.", "哦。哦。"),
            (
                "2400 hours and 2 minutes. Subject declining rapidly.",
                "24时零2分。对象状态急剧恶化。",
            ),
            ("ANNOUNCER 1: 58 fo nothing.", "播音员1：58比0，毫无收获。"),
            (
                "The repair costs 12 million dollars.",
                "维修费用为一千三百万美元。",
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
        .expect("number differences are advisory cinema-review candidates, not hard failures");
    }

    #[test]
    fn final_validation_rejects_glossary_formatting_and_omissions() {
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
            semantic: Default::default(),
        }
    }
}
