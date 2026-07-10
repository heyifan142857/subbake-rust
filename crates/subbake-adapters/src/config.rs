use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::providers::ApiFormat;
use crate::settings::TranslationSettingsPatch;

pub fn load_translation_settings_patch(path: &Path) -> io::Result<TranslationSettingsPatch> {
    let content = fs::read_to_string(path)?;
    parse_translation_settings_patch(&content).map_err(io::Error::other)
}

/// Discover config file from XDG paths or project root.
pub fn discover_config_path() -> Option<PathBuf> {
    // 1. XDG_CONFIG_HOME / default ~/.config
    let candidates = vec![
        std::env::var("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| dirs_or_default())
            .join("subbake/config.toml"),
        dirs_or_default().join("subbake/config.toml"),
        PathBuf::from(".subbake.toml"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

fn dirs_or_default() -> PathBuf {
    std::env::var("HOME")
        .map(|home| PathBuf::from(home).join(".config"))
        .unwrap_or_else(|_| PathBuf::from("."))
}

/// Load a config file (full format) and resolve the named profile
/// (or `default_profile`). Returns `None` if the file doesn't exist.
pub fn load_and_resolve(
    path: &Path,
    profile: Option<&str>,
) -> io::Result<Option<TranslationSettingsPatch>> {
    if !path.exists() {
        return Ok(None);
    }
    let config = ConfigFile::load(path)?;
    let resolved = config.resolve(profile);
    if resolved == TranslationSettingsPatch::default() {
        // All fields are None → nothing to apply.
        return Ok(None);
    }
    Ok(Some(resolved))
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

// ---------------------------------------------------------------------------
// Full config file parser (multi-profile)
// ---------------------------------------------------------------------------

/// Parsed representation of a `subbake.toml` / `config.toml`.
#[derive(Debug, Clone)]
pub struct ConfigFile {
    pub default_profile: Option<String>,
    pub defaults: TranslationSettingsPatch,
    pub profiles: HashMap<String, TranslationSettingsPatch>,
}

impl ConfigFile {
    /// Load and parse a config file from disk.
    pub fn load(path: &Path) -> io::Result<Self> {
        let content = fs::read_to_string(path)?;
        Self::parse(&content).map_err(io::Error::other)
    }

    /// Parse TOML-like config content supporting `default_profile`,
    /// `[defaults]`, and `[profiles.<name>]` sections.
    pub fn parse(content: &str) -> Result<Self, String> {
        let mut default_profile = None;
        let mut defaults = TranslationSettingsPatch::default();
        let mut profiles = HashMap::new();
        let mut current_table: Option<String> = None;

        for (line_number, raw_line) in content.lines().enumerate() {
            let line = strip_comment(raw_line).trim();
            if line.is_empty() {
                continue;
            }

            // Section header
            if line.starts_with('[') && line.ends_with(']') {
                let table = line[1..line.len() - 1].trim().to_owned();
                current_table = Some(table.clone());

                if table == "defaults" {
                    // valid
                } else if let Some(profile_name) = table.strip_prefix("profiles.") {
                    profiles.entry(profile_name.to_owned()).or_default();
                } else {
                    return Err(format!(
                        "unsupported config table `[{table}]` on line {}",
                        line_number + 1
                    ));
                }
                continue;
            }

            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("expected `key = value` on line {}", line_number + 1))?;
            let key = key.trim();
            let cv = parse_value(value.trim())
                .map_err(|e| format!("{e} on line {}", line_number + 1))?;

            // Top-level keys (outside any table)
            let table = current_table.as_deref().unwrap_or("");
            if table.is_empty() {
                if key == "default_profile" {
                    default_profile = Some(cv.into_string(key)?);
                    continue;
                }
                return Err(format!(
                    "unsupported top-level key `{key}` on line {}",
                    line_number + 1
                ));
            }

            let target = if table == "defaults" {
                &mut defaults
            } else if let Some(name) = table.strip_prefix("profiles.") {
                profiles.get_mut(name).ok_or_else(|| {
                    format!(
                        "internal: missing profile `{name}` on line {}",
                        line_number + 1
                    )
                })?
            } else {
                continue; // unreachable
            };

            apply_key_value(target, key, cv)
                .map_err(|e| format!("{e} on line {}", line_number + 1))?;
        }

        Ok(Self {
            default_profile,
            defaults,
            profiles,
        })
    }

    /// Resolve settings for `profile_name` (falls back to default_profile,
    /// then to `defaults`, then to built-in defaults).
    pub fn resolve(&self, profile_name: Option<&str>) -> TranslationSettingsPatch {
        let name = profile_name
            .or(self.default_profile.as_deref())
            .unwrap_or("");
        let mut patch = self.defaults.clone();
        if !name.is_empty()
            && let Some(profile) = self.profiles.get(name)
        {
            patch.merge(profile.clone());
        }
        patch
    }
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
        "api_key" => patch.api_key = Some(value.into_string(key)?),
        "base_url" => patch.base_url = Some(value.into_string(key)?),
        "api_format" => {
            patch.api_format =
                Some(ApiFormat::parse(&value.into_string(key)?).map_err(|e| e.to_string())?)
        }
        "endpoint_url" => patch.endpoint_url = Some(value.into_string(key)?),
        "api_key_env" => patch.api_key_env = Some(value.into_string(key)?),
        "auth_header" => patch.auth_header = Some(value.into_string(key)?),
        "auth_prefix" => patch.auth_prefix = Some(value.into_string(key)?),
        "source_language" | "source_lang" => patch.source_language = Some(value.into_string(key)?),
        "target_language" | "target_lang" => patch.target_language = Some(value.into_string(key)?),
        "batch_size" => patch.batch_size = Some(value.into_usize(key)?),
        "bilingual" => patch.bilingual = Some(value.into_bool(key)?),
        "fast" | "fast_mode" => patch.fast_mode = Some(value.into_bool(key)?),
        "final_review" | "review" => patch.final_review = Some(value.into_bool(key)?),
        "dry_run" => patch.dry_run = Some(value.into_bool(key)?),
        "resume" => patch.resume = Some(value.into_bool(key)?),
        "cache" | "use_cache" => patch.use_cache = Some(value.into_bool(key)?),
        "retries" => patch.retries = Some(value.into_nonnegative_usize(key)?),
        "agent" => patch.agent = Some(value.into_bool(key)?),
        "agent_repair_attempts" => {
            patch.agent_repair_attempts = Some(value.into_nonnegative_usize(key)?);
        }
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

    fn into_nonnegative_usize(self, key: &str) -> Result<usize, String> {
        match self {
            ConfigValue::Integer(value) => Ok(value),
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
            base_url = "https://example.test/v1"
            target_language = "English"
            batch_size = 8
            bilingual = true
            final_review = false
            resume = false
            cache = false
            retries = 0
            agent = false
            agent_repair_attempts = 3
            "#,
        )
        .expect("config should parse");

        assert_eq!(patch.provider.as_deref(), Some("mock"));
        assert_eq!(patch.model.as_deref(), Some("mock-en"));
        assert_eq!(patch.base_url.as_deref(), Some("https://example.test/v1"));
        assert_eq!(patch.target_language.as_deref(), Some("English"));
        assert_eq!(patch.batch_size, Some(8));
        assert_eq!(patch.bilingual, Some(true));
        assert_eq!(patch.final_review, Some(false));
        assert_eq!(patch.resume, Some(false));
        assert_eq!(patch.use_cache, Some(false));
        assert_eq!(patch.retries, Some(0));
        assert_eq!(patch.agent, Some(false));
        assert_eq!(patch.agent_repair_attempts, Some(3));
    }

    #[test]
    fn rejects_unknown_keys() {
        let error =
            parse_translation_settings_patch("unknown = true").expect_err("unknown key fails");
        assert!(error.contains("unsupported config key"));
    }

    #[test]
    fn parses_user_config_file() {
        let config = ConfigFile::parse(
            r#"
            default_profile = "deepseek"

            [defaults]
            target_language = "zh"
            batch_size = 30

            [profiles.deepseek]
            provider = "openai"
            model = "deepseek-v4-flash"
            base_url = "https://api.deepseek.com"
            api_key = "sk-test123"
            "#,
        )
        .expect("config should parse");

        assert_eq!(config.default_profile.as_deref(), Some("deepseek"));
        assert_eq!(
            config
                .profiles
                .get("deepseek")
                .and_then(|p| p.provider.as_deref()),
            Some("openai")
        );
        assert_eq!(
            config
                .profiles
                .get("deepseek")
                .and_then(|p| p.model.as_deref()),
            Some("deepseek-v4-flash")
        );
    }

    #[test]
    fn resolve_merges_defaults_and_profile() {
        let config = ConfigFile::parse(
            r#"
            [defaults]
            target_language = "Chinese"
            batch_size = 20

            [profiles.custom]
            provider = "anthropic"
            model = "claude-sonnet-4"
            "#,
        )
        .expect("config should parse");

        let resolved = config.resolve(Some("custom"));
        assert_eq!(resolved.target_language.as_deref(), Some("Chinese"));
        assert_eq!(resolved.batch_size, Some(20));
        assert_eq!(resolved.provider.as_deref(), Some("anthropic"));
        assert_eq!(resolved.model.as_deref(), Some("claude-sonnet-4"));
    }

    #[test]
    fn preserves_hash_inside_quoted_string() {
        let patch = parse_translation_settings_patch(r#"model = "mock#tag" # comment"#)
            .expect("config should parse");
        assert_eq!(patch.model.as_deref(), Some("mock#tag"));
    }
}
