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

    let normalized = match key.as_str() {
        "zh" | "zho" | "chi" | "cn" | "chinese" | "中文" | "汉语" | "漢語" => "Chinese",
        "en" | "eng" | "english" => "English",
        "ja" | "jp" | "jpn" | "japanese" | "日本語" => "Japanese",
        "ko" | "kor" | "korean" | "한국어" => "Korean",
        "fr" | "fra" | "fre" | "french" => "French",
        "es" | "spa" | "spanish" => "Spanish",
        "de" | "deu" | "ger" | "german" => "German",
        _ => return beautify_language_name(value),
    };
    normalized.to_owned()
}

pub fn language_short_code(value: &str) -> String {
    let normalized = normalize_language_name(value, true);
    match normalized.as_str() {
        "Auto" => "AUTO".to_owned(),
        "Chinese" => "ZH".to_owned(),
        "English" => "EN".to_owned(),
        "Japanese" => "JA".to_owned(),
        "Korean" => "KO".to_owned(),
        "French" => "FR".to_owned(),
        "Spanish" => "ES".to_owned(),
        "German" => "DE".to_owned(),
        other => {
            let slug = slugify(other);
            if slug.is_empty() {
                "LANG".to_owned()
            } else {
                slug.chars().take(8).collect::<String>().to_uppercase()
            }
        }
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

fn beautify_language_name(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return "Unknown".to_owned();
    }
    let mut chars = trimmed.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
        None => "Unknown".to_owned(),
    }
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
    fn normalizes_common_languages() {
        assert_eq!(normalize_language_name("zh", false), "Chinese");
        assert_eq!(normalize_language_name("EN", false), "English");
        assert_eq!(normalize_language_name("auto", true), "Auto");
    }

    #[test]
    fn builds_pair_slug() {
        assert_eq!(language_pair_slug("auto", "zh"), "auto-chinese");
    }
}
