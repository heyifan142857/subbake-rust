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

/// Append a profile snapshot without rewriting existing configuration, comments,
/// or inline credentials. The caller can keep using its current profile until
/// the newly created profile has been reviewed.
pub fn append_profile_snapshot(
    path: &Path,
    name: &str,
    mut settings: TranslationSettingsPatch,
) -> io::Result<()> {
    if name.is_empty()
        || !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "profile name may contain only letters, numbers, '-' and '_'",
        ));
    }

    let config = ConfigFile::load(path)?;
    if config.profiles.contains_key(name) {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("profile `{name}` already exists"),
        ));
    }

    // Do not proliferate inline credentials when creating a profile. An
    // environment-variable reference remains safe to reuse.
    settings.api_key = None;
    settings.auth_header = None;

    let mut content = fs::read_to_string(path)?;
    if !content.ends_with('\n') {
        content.push('\n');
    }
    content.push('\n');
    content.push_str(&format!("[profiles.{name}]\n"));
    write_profile_settings(&mut content, &settings);

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("config");
    let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    let permissions = fs::metadata(path)?.permissions();
    fs::write(&temporary, content)?;
    fs::set_permissions(&temporary, permissions)?;
    let rename_result = fs::rename(&temporary, path);
    if rename_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    rename_result
}

fn write_profile_settings(content: &mut String, settings: &TranslationSettingsPatch) {
    macro_rules! string_setting {
        ($key:literal, $value:expr) => {
            if let Some(value) = &$value {
                content.push_str(&format!("{} = {}\n", $key, quote_string(value)));
            }
        };
    }
    macro_rules! bool_setting {
        ($key:literal, $value:expr) => {
            if let Some(value) = $value {
                content.push_str(&format!("{} = {}\n", $key, value));
            }
        };
    }
    macro_rules! usize_setting {
        ($key:literal, $value:expr) => {
            if let Some(value) = $value {
                content.push_str(&format!("{} = {}\n", $key, value));
            }
        };
    }
    string_setting!("output_format", settings.output_format);
    string_setting!("provider", settings.provider);
    string_setting!("model", settings.model);
    string_setting!("base_url", settings.base_url);
    if let Some(value) = settings.api_format {
        string_setting!("api_format", Some(value.as_str().to_owned()));
    }
    string_setting!("endpoint_url", settings.endpoint_url);
    string_setting!("api_key_env", settings.api_key_env);
    string_setting!("auth_prefix", settings.auth_prefix);
    string_setting!("source_language", settings.source_language);
    string_setting!("target_language", settings.target_language);
    usize_setting!("batch_size", settings.batch_size);
    usize_setting!("batch_token_budget", settings.batch_token_budget);
    usize_setting!("translation_concurrency", settings.translation_concurrency);
    usize_setting!("review_concurrency", settings.review_concurrency);
    bool_setting!("bilingual", settings.bilingual);
    bool_setting!("fast_mode", settings.fast_mode);
    bool_setting!("final_review", settings.final_review);
    if let Some(value) = settings.review_policy {
        string_setting!("review_policy", Some(value.as_str().to_owned()));
    }
    bool_setting!("terminology_preflight", settings.terminology_preflight);
    bool_setting!("dry_run", settings.dry_run);
    bool_setting!("resume", settings.resume);
    bool_setting!("use_cache", settings.use_cache);
    usize_setting!("retries", settings.retries);
    bool_setting!("agent", settings.agent);
    usize_setting!("agent_repair_attempts", settings.agent_repair_attempts);
    if let Some(value) = &settings.runtime_dir {
        string_setting!("runtime_dir", Some(value.to_string_lossy().into_owned()));
    }
    if let Some(value) = &settings.glossary_path {
        string_setting!("glossary_path", Some(value.to_string_lossy().into_owned()));
    }
}

fn quote_string(value: &str) -> String {
    format!(
        "\"{}\"",
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    )
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
    let config = match ConfigFile::load(path) {
        Ok(config) => config,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
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
        "batch_token_budget" => patch.batch_token_budget = Some(value.into_usize(key)?),
        "translation_concurrency" => patch.translation_concurrency = Some(value.into_usize(key)?),
        "review_concurrency" => patch.review_concurrency = Some(value.into_usize(key)?),
        "bilingual" => patch.bilingual = Some(value.into_bool(key)?),
        "fast" | "fast_mode" => patch.fast_mode = Some(value.into_bool(key)?),
        "final_review" => patch.final_review = Some(value.into_bool(key)?),
        "review" | "review_policy" => {
            patch.review_policy = Some(subbake_core::ReviewPolicy::parse(&value.into_string(key)?)?)
        }
        "terminology_preflight" => patch.terminology_preflight = Some(value.into_bool(key)?),
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

    fn temporary_config(name: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-{name}-{unique}.toml"))
    }

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

    #[test]
    fn appends_profile_snapshot_without_rewriting_comments_or_credentials() {
        let path = temporary_config("new-profile");
        let original = "# keep this comment\n[defaults]\nprovider = \"mock\"\n";
        fs::write(&path, original).expect("write config");
        let patch = TranslationSettingsPatch {
            provider: Some("custom".to_owned()),
            model: Some("model\\\"name".to_owned()),
            api_key: Some("secret".to_owned()),
            api_key_env: Some("CUSTOM_API_KEY".to_owned()),
            auth_header: Some("also-secret".to_owned()),
            batch_size: Some(12),
            bilingual: Some(true),
            ..TranslationSettingsPatch::default()
        };

        append_profile_snapshot(&path, "review_copy", patch).expect("append profile");
        let content = fs::read_to_string(&path).expect("read config");
        assert!(content.starts_with(original));
        assert!(content.contains("[profiles.review_copy]"));
        assert!(content.contains("model = \"model\\\\\\\"name\""));
        assert!(content.contains("api_key_env = \"CUSTOM_API_KEY\""));
        assert!(!content.contains("secret"));

        let parsed = ConfigFile::load(&path).expect("parse appended config");
        let profile = parsed.profiles.get("review_copy").expect("new profile");
        assert_eq!(profile.batch_size, Some(12));
        assert_eq!(profile.bilingual, Some(true));
        fs::remove_file(path).expect("remove config");
    }

    #[test]
    fn profile_snapshot_rejects_unsafe_and_duplicate_names() {
        let path = temporary_config("profile-validation");
        fs::write(&path, "[profiles.existing]\nmodel = \"mock\"\n").expect("write config");
        let patch = TranslationSettingsPatch::default();
        assert_eq!(
            append_profile_snapshot(&path, "bad.name", patch.clone())
                .expect_err("unsafe name")
                .kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            append_profile_snapshot(&path, "existing", patch)
                .expect_err("duplicate name")
                .kind(),
            io::ErrorKind::AlreadyExists
        );
        fs::remove_file(path).expect("remove config");
    }
}
