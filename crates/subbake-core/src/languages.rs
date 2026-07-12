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
        "en" | "eng" => "en",
        "ja" | "jp" | "jpn" | "日本語" => "ja",
        "ko" | "kor" | "한국어" => "ko",
        "fr" | "fra" | "fre" => "fr",
        "es" | "spa" => "es",
        "de" | "deu" | "ger" => "de",
        "pt" | "por" | "português" => "pt",
        "pt-br" => "pt-BR",
        "ru" | "rus" => "ru",
        _ => return canonicalize_unknown(value),
    };
    tag.to_owned()
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
    }

    #[test]
    fn canonicalizes_unknown_bcp_47_tags() {
        assert_eq!(normalize_language_name("sr_latn_rs", false), "sr-Latn-RS");
        assert_eq!(normalize_language_name("Italian", false), "und");
        assert_eq!(normalize_language_name("English", false), "und");
        assert_eq!(normalize_language_name("", false), "und");
    }

    #[test]
    fn builds_pair_slug() {
        assert_eq!(language_pair_slug("auto", "zh"), "auto-zh-hans");
        assert_eq!(language_pair_slug("en", "繁體中文"), "en-zh-hant");
    }
}
