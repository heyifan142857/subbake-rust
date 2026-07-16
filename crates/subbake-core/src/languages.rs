use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguageNormalizationError {
    value: String,
    allow_auto: bool,
}

impl LanguageNormalizationError {
    pub fn examples(&self) -> &'static str {
        if self.allow_auto {
            "Auto, English, en, Japanese, ja, Chinese, zh-Hans"
        } else {
            "English, en, Japanese, ja, Chinese, zh-Hans"
        }
    }
}

impl Display for LanguageNormalizationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "unsupported or ambiguous language `{}`; use a common language name or BCP-47 tag such as {}",
            self.value,
            self.examples()
        )
    }
}

impl Error for LanguageNormalizationError {}

pub fn normalize_language_name(value: &str, allow_auto: bool) -> String {
    let key = normalize_key(value);
    if allow_auto
        && matches!(
            key.as_str(),
            "" | "auto" | "detect" | "auto-detect" | "autodetect"
        )
    {
        return "Auto".to_owned();
    }

    let tag = match key.as_str() {
        // Bare Chinese is kept deterministic for translation and cache keys.
        "zh" | "zho" | "chi" | "cn" | "中文" | "汉语" | "漢語" | "简体" | "简体中文" | "簡體"
        | "簡體中文" => "zh-Hans",
        "zh-hans" | "zh-cn" | "zh-sg" => "zh-Hans",
        "繁体" | "繁体中文" | "繁體" | "繁體中文" => "zh-Hant",
        "zh-hant" | "zh-tw" | "zh-hk" | "zh-mo" => "zh-Hant",
        "en" | "eng" | "english" | "英语" | "英語" => "en",
        "ja" | "jp" | "jpn" | "japanese" | "日本語" | "日语" | "日語" => "ja",
        "ko" | "kor" | "korean" | "한국어" | "韩语" | "韓語" => "ko",
        "fr" | "fra" | "fre" | "french" | "français" => "fr",
        "es" | "spa" | "spanish" | "español" => "es",
        "de" | "deu" | "ger" | "german" | "deutsch" => "de",
        "pt" | "por" | "portuguese" | "português" => "pt",
        "pt-br" => "pt-BR",
        "ru" | "rus" | "russian" | "русский" => "ru",
        "it" | "ita" | "italian" | "italiano" => "it",
        "ar" | "ara" | "arabic" | "العربية" => "ar",
        "hi" | "hin" | "hindi" | "हिन्दी" => "hi",
        "nl" | "nld" | "dut" | "dutch" | "nederlands" => "nl",
        "pl" | "pol" | "polish" | "polski" => "pl",
        "tr" | "tur" | "turkish" | "türkçe" => "tr",
        "uk" | "ukr" | "ukrainian" | "українська" => "uk",
        "vi" | "vie" | "vietnamese" | "tiếng việt" => "vi",
        "th" | "tha" | "thai" | "ไทย" => "th",
        "id" | "ind" | "indonesian" | "bahasa indonesia" => "id",
        _ => return canonicalize_unknown(value),
    };
    tag.to_owned()
}

pub fn normalize_language(
    value: &str,
    allow_auto: bool,
) -> Result<String, LanguageNormalizationError> {
    let normalized = normalize_language_name(value, allow_auto);
    if normalized == "und" || (!allow_auto && normalized == "Auto") {
        Err(LanguageNormalizationError {
            value: value.to_owned(),
            allow_auto,
        })
    } else {
        Ok(normalized)
    }
}

pub fn is_language_tag(value: &str) -> bool {
    if value.eq_ignore_ascii_case("auto") || value.eq_ignore_ascii_case("und") {
        return false;
    }
    let parts = value.split('-').collect::<Vec<_>>();
    !parts.is_empty()
        && parts[0].len() >= 2
        && parts[0].len() <= 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.chars().all(|ch| ch.is_ascii_alphanumeric()))
}

pub fn language_short_code(value: &str) -> String {
    let normalized = normalize_language_name(value, true);
    if normalized == "Auto" {
        return "AUTO".to_owned();
    }
    let slug = slugify(&normalized);
    if slug.is_empty() {
        "LANG".to_owned()
    } else {
        slug.chars().take(8).collect::<String>().to_uppercase()
    }
}

pub fn language_pair_slug(source_language: &str, target_language: &str) -> String {
    format!(
        "{}-{}",
        slugify(&normalize_language_name(source_language, true)),
        slugify(&normalize_language_name(target_language, false))
    )
}

fn normalize_key(value: &str) -> String {
    value.trim().to_lowercase().replace('_', "-")
}

fn canonicalize_unknown(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return "und".to_owned();
    }

    // Preserve free-form language names, but canonicalize tag-looking input.
    let parts = trimmed.split(['-', '_']).collect::<Vec<_>>();
    if !parts[0].chars().all(|ch| ch.is_ascii_alphabetic()) || parts[0].len() > 3 {
        return "und".to_owned();
    }
    parts
        .iter()
        .enumerate()
        .map(|(index, part)| match (index, part.len()) {
            (0, _) => part.to_ascii_lowercase(),
            (_, 2) if part.chars().all(|ch| ch.is_ascii_alphabetic()) => part.to_ascii_uppercase(),
            (_, 4) if part.chars().all(|ch| ch.is_ascii_alphabetic()) => {
                let mut chars = part.chars();
                let first = chars
                    .next()
                    .map_or_else(String::new, |ch| ch.to_uppercase().collect());
                format!("{first}{}", chars.as_str().to_ascii_lowercase())
            }
            _ => part.to_ascii_lowercase(),
        })
        .collect::<Vec<_>>()
        .join("-")
}

fn slugify(value: &str) -> String {
    let mut output = String::new();
    let mut previous_dash = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            output.push(ch.to_ascii_lowercase());
            previous_dash = false;
        } else if !previous_dash {
            output.push('-');
            previous_dash = true;
        }
    }
    output.trim_matches('-').to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_common_languages_to_bcp_47() {
        assert_eq!(normalize_language_name("zh", false), "zh-Hans");
        assert_eq!(normalize_language_name("繁體中文", false), "zh-Hant");
        assert_eq!(normalize_language_name("zh_TW", false), "zh-Hant");
        assert_eq!(normalize_language_name("jp", false), "ja");
        assert_eq!(normalize_language_name("EN", false), "en");
        assert_eq!(normalize_language_name("pt_br", false), "pt-BR");
        assert_eq!(normalize_language_name("auto", true), "Auto");
        assert_eq!(normalize_language_name("Japanese", false), "ja");
        assert_eq!(normalize_language_name("English", false), "en");
        assert_eq!(normalize_language_name("Italian", false), "it");
    }

    #[test]
    fn canonicalizes_unknown_bcp_47_tags() {
        assert_eq!(normalize_language_name("sr_latn_rs", false), "sr-Latn-RS");
        assert_eq!(normalize_language_name("Klingon", false), "und");
        assert_eq!(normalize_language_name("", false), "und");
    }

    #[test]
    fn strict_normalization_rejects_empty_und_and_ambiguous_names() {
        assert_eq!(
            normalize_language("Japanese", false).expect("Japanese"),
            "ja"
        );
        assert!(normalize_language("", false).is_err());
        assert!(normalize_language("und", false).is_err());
        assert!(normalize_language("Klingon", false).is_err());
    }

    #[test]
    fn recognizes_language_tags_used_in_generated_output_names() {
        assert!(is_language_tag("ja"));
        assert!(is_language_tag("zh-Hans"));
        assert!(!is_language_tag("translated"));
        assert!(!is_language_tag("und"));
    }

    #[test]
    fn builds_pair_slug() {
        assert_eq!(language_pair_slug("auto", "zh"), "auto-zh-hans");
        assert_eq!(language_pair_slug("en", "繁體中文"), "en-zh-hant");
    }
}
