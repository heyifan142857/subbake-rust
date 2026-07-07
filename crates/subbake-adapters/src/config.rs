use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::settings::TranslationSettingsPatch;

pub fn load_translation_settings_patch(path: &Path) -> io::Result<TranslationSettingsPatch> {
    let content = fs::read_to_string(path)?;
    parse_translation_settings_patch(&content).map_err(io::Error::other)
}

pub fn parse_translation_settings_patch(content: &str) -> Result<TranslationSettingsPatch, String> {
    let mut patch = TranslationSettingsPatch::default();

    for (line_number, raw_line) in content.lines().enumerate() {
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            let table = line[1..line.len() - 1].trim();
            if table != "defaults" {
                return Err(format!(
                    "unsupported config table `[{}]` on line {}",
                    table,
                    line_number + 1
                ));
            }
            continue;
        }

        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("expected `key = value` on line {}", line_number + 1))?;
        apply_key_value(
            &mut patch,
            key.trim(),
            parse_value(value.trim())
                .map_err(|error| format!("{error} on line {}", line_number + 1))?,
        )
        .map_err(|error| format!("{error} on line {}", line_number + 1))?;
    }

    Ok(patch)
}

fn apply_key_value(
    patch: &mut TranslationSettingsPatch,
    key: &str,
    value: ConfigValue,
) -> Result<(), String> {
    match key {
        "output_format" => patch.output_format = Some(value.into_string(key)?),
        "provider" => patch.provider = Some(value.into_string(key)?),
        "model" => patch.model = Some(value.into_string(key)?),
        "source_language" | "source_lang" => patch.source_language = Some(value.into_string(key)?),
        "target_language" | "target_lang" => patch.target_language = Some(value.into_string(key)?),
        "batch_size" => patch.batch_size = Some(value.into_usize(key)?),
        "bilingual" => patch.bilingual = Some(value.into_bool(key)?),
        "fast" | "fast_mode" => patch.fast_mode = Some(value.into_bool(key)?),
        "final_review" | "review" => patch.final_review = Some(value.into_bool(key)?),
        "dry_run" => patch.dry_run = Some(value.into_bool(key)?),
        "runtime_dir" => patch.runtime_dir = Some(PathBuf::from(value.into_string(key)?)),
        "glossary" | "glossary_path" => {
            patch.glossary_path = Some(PathBuf::from(value.into_string(key)?));
        }
        other => return Err(format!("unsupported config key `{other}`")),
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
enum ConfigValue {
    String(String),
    Bool(bool),
    Integer(usize),
}

impl ConfigValue {
    fn into_string(self, key: &str) -> Result<String, String> {
        match self {
            ConfigValue::String(value) => Ok(value),
            ConfigValue::Bool(_) | ConfigValue::Integer(_) => {
                Err(format!("config key `{key}` expects a string"))
            }
        }
    }

    fn into_bool(self, key: &str) -> Result<bool, String> {
        match self {
            ConfigValue::Bool(value) => Ok(value),
            ConfigValue::String(_) | ConfigValue::Integer(_) => {
                Err(format!("config key `{key}` expects a boolean"))
            }
        }
    }

    fn into_usize(self, key: &str) -> Result<usize, String> {
        match self {
            ConfigValue::Integer(value) if value > 0 => Ok(value),
            ConfigValue::Integer(_) => Err(format!("config key `{key}` must be greater than zero")),
            ConfigValue::String(_) | ConfigValue::Bool(_) => {
                Err(format!("config key `{key}` expects an integer"))
            }
        }
    }
}

fn parse_value(raw: &str) -> Result<ConfigValue, String> {
    if let Some(value) = parse_quoted_string(raw)? {
        return Ok(ConfigValue::String(value));
    }
    match raw {
        "true" => return Ok(ConfigValue::Bool(true)),
        "false" => return Ok(ConfigValue::Bool(false)),
        _ => {}
    }
    raw.parse::<usize>()
        .map(ConfigValue::Integer)
        .map_err(|_| format!("unsupported config value `{raw}`"))
}

fn parse_quoted_string(raw: &str) -> Result<Option<String>, String> {
    if !raw.starts_with('"') {
        return Ok(None);
    }
    if !raw.ends_with('"') || raw.len() < 2 {
        return Err("unterminated string".to_owned());
    }
    let inner = &raw[1..raw.len() - 1];
    let mut output = String::new();
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            output.push(ch);
            continue;
        }
        match chars.next() {
            Some('"') => output.push('"'),
            Some('\\') => output.push('\\'),
            Some('n') => output.push('\n'),
            Some('r') => output.push('\r'),
            Some('t') => output.push('\t'),
            Some(other) => return Err(format!("unsupported escape sequence `\\{other}`")),
            None => return Err("trailing escape character".to_owned()),
        }
    }
    Ok(Some(output))
}

fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    let mut escaped = false;
    for (index, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            '#' if !in_string => return &line[..index],
            _ => {}
        }
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_defaults_table() {
        let patch = parse_translation_settings_patch(
            r#"
            [defaults]
            provider = "mock"
            model = "mock-en"
            target_language = "English"
            batch_size = 8
            bilingual = true
            final_review = false
            "#,
        )
        .expect("config should parse");

        assert_eq!(patch.provider.as_deref(), Some("mock"));
        assert_eq!(patch.model.as_deref(), Some("mock-en"));
        assert_eq!(patch.target_language.as_deref(), Some("English"));
        assert_eq!(patch.batch_size, Some(8));
        assert_eq!(patch.bilingual, Some(true));
        assert_eq!(patch.final_review, Some(false));
    }

    #[test]
    fn rejects_unknown_keys() {
        let error =
            parse_translation_settings_patch("unknown = true").expect_err("unknown key fails");
        assert!(error.contains("unsupported config key"));
    }

    #[test]
    fn preserves_hash_inside_quoted_string() {
        let patch = parse_translation_settings_patch(r#"model = "mock#tag" # comment"#)
            .expect("config should parse");
        assert_eq!(patch.model.as_deref(), Some("mock#tag"));
    }
}
