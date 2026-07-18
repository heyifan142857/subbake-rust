use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AdapterError, AdapterResult, ConfigError};
use crate::settings::{BackendOverrides, ResolvedSettings, SettingsOverrides};

pub const CONFIG_VERSION: u64 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigFile {
    pub version: u64,
    #[serde(default)]
    pub default_profile: Option<String>,
    #[serde(default)]
    pub defaults: SettingsOverrides,
    #[serde(default)]
    pub backends: HashMap<String, BackendOverrides>,
    #[serde(default)]
    pub profiles: HashMap<String, SettingsOverrides>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct ResolveRequest {
    pub explicit_path: Option<PathBuf>,
    pub pinned_path: Option<PathBuf>,
    pub profile: Option<String>,
    pub cli_overrides: SettingsOverrides,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedConfiguration {
    pub settings: ResolvedSettings,
    pub config_path: Option<PathBuf>,
    pub profile: Option<String>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ConfigurationResolver;

impl ConfigFile {
    pub fn load(path: &Path) -> AdapterResult<Self> {
        let content = fs::read_to_string(path).map_err(|source| {
            AdapterError::external_io("read configuration", Some(path.to_path_buf()), source)
        })?;
        Self::parse(&content).map_err(|source| AdapterError::ConfigurationFile {
            path: path.to_path_buf(),
            source,
        })
    }

    pub fn parse(content: &str) -> Result<Self, ConfigError> {
        let config: Self =
            toml::from_str(content).map_err(|error| ConfigError::invalid(error.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if !matches!(self.version, 1 | CONFIG_VERSION) {
            return Err(ConfigError::invalid(format!(
                "unsupported configuration version `{}`; expected 1 or `{CONFIG_VERSION}`",
                self.version
            )));
        }
        for name in self.profiles.keys() {
            validate_profile_name(name)?;
        }
        for name in self.backends.keys() {
            validate_profile_name(name)?;
        }
        if let Some(name) = self.default_profile.as_deref()
            && !self.profiles.contains_key(name)
        {
            return Err(ConfigError::invalid(format!(
                "default profile `{name}` is not defined"
            )));
        }
        for (profile, settings) in &self.profiles {
            for (role, reference) in [
                ("translator", settings.translator.as_deref()),
                ("reviewer", settings.reviewer.as_deref()),
            ] {
                if let Some(reference) = reference
                    && !self.backends.contains_key(reference)
                {
                    return Err(ConfigError::invalid(format!(
                        "profile `{profile}` {role} references unknown backend `{reference}`"
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn selected_profile<'a>(
        &'a self,
        requested: Option<&'a str>,
    ) -> Result<Option<&'a str>, ConfigError> {
        let selected = requested.or(self.default_profile.as_deref());
        if let Some(name) = selected
            && !self.profiles.contains_key(name)
        {
            return Err(ConfigError::invalid(format!(
                "profile `{name}` is not defined"
            )));
        }
        Ok(selected)
    }

    pub fn resolve(
        &self,
        requested_profile: Option<&str>,
        cli_overrides: SettingsOverrides,
    ) -> Result<(ResolvedSettings, Option<String>), ConfigError> {
        let selected = self.selected_profile(requested_profile)?;
        let mut overrides = self.defaults.clone();
        if let Some(name) = selected {
            let profile = self
                .profiles
                .get(name)
                .ok_or_else(|| ConfigError::invalid(format!("profile `{name}` is not defined")))?;
            overrides.merge(profile.clone());
        }
        overrides.merge(cli_overrides);
        if let Some(name) = overrides.translator.as_deref() {
            let mut backend = self.backends.get(name).cloned().ok_or_else(|| {
                ConfigError::invalid(format!("unknown translator backend `{name}`"))
            })?;
            backend.merge(overrides.backend);
            overrides.backend = backend;
        }
        if let Some(name) = overrides.reviewer.as_deref() {
            let mut backend = self.backends.get(name).cloned().ok_or_else(|| {
                ConfigError::invalid(format!("unknown reviewer backend `{name}`"))
            })?;
            if let Some(role_overrides) = overrides.reviewer_backend.take() {
                backend.merge(role_overrides);
            }
            overrides.reviewer_backend = Some(backend);
        }
        let settings = ResolvedSettings::default()
            .with_overrides(overrides)
            .map_err(|error| ConfigError::invalid(error.to_string()))?;
        Ok((settings, selected.map(str::to_owned)))
    }
}

impl ConfigurationResolver {
    pub fn resolve(&self, request: ResolveRequest) -> AdapterResult<ResolvedConfiguration> {
        let requested_path = request.pinned_path.or(request.explicit_path);
        let path = requested_path.clone().or_else(discover_config_path);
        let Some(path) = path else {
            if let Some(profile) = request.profile {
                return Err(ConfigError::invalid(format!(
                    "profile `{profile}` requires a configuration file"
                ))
                .into());
            }
            return Ok(ResolvedConfiguration {
                settings: ResolvedSettings::default().with_overrides(request.cli_overrides)?,
                config_path: None,
                profile: None,
            });
        };

        if !path.exists() {
            if requested_path.is_some() {
                return Err(AdapterError::ConfigurationFile {
                    path,
                    source: ConfigError::invalid("configuration file does not exist"),
                });
            }
            return Ok(ResolvedConfiguration {
                settings: ResolvedSettings::default().with_overrides(request.cli_overrides)?,
                config_path: None,
                profile: None,
            });
        }

        let config = ConfigFile::load(&path)?;
        let (settings, profile) = config
            .resolve(request.profile.as_deref(), request.cli_overrides)
            .map_err(|source| AdapterError::ConfigurationFile {
                path: path.clone(),
                source,
            })?;
        Ok(ResolvedConfiguration {
            settings,
            config_path: Some(path),
            profile,
        })
    }
}

pub fn discover_config_path() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("XDG_CONFIG_HOME") {
        candidates.push(PathBuf::from(path).join("subbake/config.toml"));
    } else if let Some(home) = std::env::var_os("HOME") {
        candidates.push(PathBuf::from(home).join(".config/subbake/config.toml"));
    }
    candidates.push(PathBuf::from("subbake.toml"));
    candidates.push(PathBuf::from(".subbake.toml"));
    candidates.into_iter().find(|path| path.exists())
}

pub fn append_profile_snapshot(
    path: &Path,
    name: &str,
    settings: &ResolvedSettings,
) -> AdapterResult<()> {
    validate_profile_name(name)?;
    let config = ConfigFile::load(path)?;
    if config.profiles.contains_key(name) {
        return Err(ConfigError::DuplicateProfile {
            name: name.to_owned(),
        }
        .into());
    }

    let mut snapshot = SettingsOverrides::from_resolved(settings);
    snapshot.backend.api_key = None;
    snapshot.backend.auth_header = None;
    if let Some(reviewer) = &mut snapshot.reviewer_backend {
        reviewer.api_key = None;
        reviewer.auth_header = None;
    }

    #[derive(Serialize)]
    struct ProfileAppend {
        profiles: BTreeMap<String, SettingsOverrides>,
    }

    let mut profiles = BTreeMap::new();
    profiles.insert(name.to_owned(), snapshot);
    let rendered = toml::to_string(&ProfileAppend { profiles })
        .map_err(|error| ConfigError::invalid(error.to_string()))?;

    let mut content = fs::read_to_string(path).map_err(|source| {
        AdapterError::external_io("read configuration", Some(path.to_path_buf()), source)
    })?;
    if !content.ends_with('\n') {
        content.push('\n');
    }
    content.push('\n');
    content.push_str(&rendered);

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("config");
    let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    let permissions = fs::metadata(path)
        .map_err(|source| {
            AdapterError::external_io(
                "read configuration metadata",
                Some(path.to_path_buf()),
                source,
            )
        })?
        .permissions();
    fs::write(&temporary, content).map_err(|source| {
        AdapterError::external_io(
            "write temporary configuration",
            Some(temporary.clone()),
            source,
        )
    })?;
    fs::set_permissions(&temporary, permissions).map_err(|source| {
        AdapterError::external_io(
            "preserve configuration permissions",
            Some(temporary.clone()),
            source,
        )
    })?;
    let rename_result = fs::rename(&temporary, path);
    if rename_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    rename_result.map_err(|source| {
        AdapterError::external_io("replace configuration", Some(path.to_path_buf()), source)
    })
}

fn validate_profile_name(name: &str) -> Result<(), ConfigError> {
    if name.is_empty()
        || !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(ConfigError::InvalidProfileName);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{
        BackendOverrides, OutputOverrides, StorageOverrides, TranslationOverrides,
    };

    fn temporary_config(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-{label}-{nonce}.toml"))
    }

    #[test]
    fn resolves_builtin_defaults_then_defaults_profile_and_cli() {
        let config = ConfigFile::parse(
            r#"
            version = 1
            default_profile = "quality"

            [defaults.translation]
            target_language = "Japanese"
            batch_size = 20

            [defaults.output]
            bilingual = true

            [defaults.storage]
            runtime_dir = "defaults-runtime"

            [profiles.quality.backend]
            id = "openai"
            model = "gpt-profile"
            api_format = "openai_chat"

            [profiles.quality.translation]
            batch_size = 10

            [profiles.quality.output]
            bilingual = false

            [profiles.quality.storage]
            runtime_dir = "profile-runtime"
            "#,
        )
        .expect("v1 config");

        let (settings, profile) = config
            .resolve(
                None,
                SettingsOverrides {
                    translation: TranslationOverrides {
                        target_language: Some("English".to_owned()),
                        ..TranslationOverrides::default()
                    },
                    output: OutputOverrides {
                        bilingual: Some(true),
                        ..OutputOverrides::default()
                    },
                    storage: StorageOverrides {
                        runtime_dir: Some("cli-runtime".into()),
                        ..StorageOverrides::default()
                    },
                    ..SettingsOverrides::default()
                },
            )
            .expect("resolved config");

        assert_eq!(profile.as_deref(), Some("quality"));
        assert_eq!(settings.backend.id, "openai");
        assert_eq!(settings.backend.model, "gpt-profile");
        assert_eq!(settings.translation.target_language, "English");
        assert_eq!(settings.translation.batch_size, 10);
        assert!(settings.output.bilingual);
        assert_eq!(settings.storage.runtime_dir, Some("cli-runtime".into()));
        assert_eq!(settings.translation.source_language, "Auto");
    }

    #[test]
    fn rejects_old_flat_configuration() {
        let error = ConfigFile::parse(
            r#"
            version = 1
            [profiles.openai]
            provider = "openai"
            model = "gpt"
            "#,
        )
        .expect_err("flat fields are invalid");
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn rejects_missing_or_invalid_versions_and_profiles() {
        assert!(ConfigFile::parse("[defaults.output]\nbilingual = true").is_err());
        assert!(ConfigFile::parse("version = 3").is_err());
        assert!(ConfigFile::parse("version = 1\ndefault_profile = \"missing\"").is_err());
        let config = ConfigFile::parse("version = 1").expect("empty v1 config");
        assert!(
            config
                .resolve(Some("missing"), SettingsOverrides::default())
                .is_err()
        );
    }

    #[test]
    fn grouped_types_round_trip() {
        let config = ConfigFile {
            version: CONFIG_VERSION,
            default_profile: Some("test".to_owned()),
            defaults: SettingsOverrides {
                output: OutputOverrides {
                    format: Some("vtt".to_owned()),
                    ..OutputOverrides::default()
                },
                storage: StorageOverrides {
                    runtime_dir: Some(".runtime".into()),
                    whisper_binary_path: Some("tools/whisper-cli".into()),
                    whisper_models_dir: Some("models".into()),
                    ..StorageOverrides::default()
                },
                ..SettingsOverrides::default()
            },
            backends: HashMap::new(),
            profiles: HashMap::from([(
                "test".to_owned(),
                SettingsOverrides {
                    backend: BackendOverrides {
                        id: Some("mock".to_owned()),
                        model: Some("mock-en".to_owned()),
                        ..BackendOverrides::default()
                    },
                    ..SettingsOverrides::default()
                },
            )]),
        };
        let text = toml::to_string(&config).expect("serialize config");
        assert_eq!(ConfigFile::parse(&text).expect("parse config"), config);
    }

    #[test]
    fn v2_resolves_reusable_translator_and_reviewer_backends() {
        let config = ConfigFile::parse(
            r#"
            version = 2
            default_profile = "cinema"

            [backends.fast]
            id = "openai"
            model = "fast-model"
            api_format = "openai_chat"

            [backends.judge]
            id = "anthropic"
            model = "judge-model"
            api_format = "anthropic_messages"

            [profiles.cinema]
            translator = "fast"
            reviewer = "judge"

            [profiles.cinema.translation]
            mode = "cinema"
            "#,
        )
        .expect("v2 config");
        let (settings, profile) = config
            .resolve(None, SettingsOverrides::default())
            .expect("resolve");
        assert_eq!(profile.as_deref(), Some("cinema"));
        assert_eq!(settings.backend.model, "fast-model");
        assert_eq!(
            settings
                .reviewer_backend
                .as_ref()
                .map(|backend| backend.model.as_str()),
            Some("judge-model")
        );
        assert_eq!(
            settings.translation.mode,
            subbake_core::TranslationMode::Cinema
        );
        assert_eq!(
            settings.translation.review_policy,
            subbake_core::ReviewPolicy::Full
        );
    }

    #[test]
    fn profile_snapshot_uses_grouped_v1_shape_without_inline_secrets() {
        let path = temporary_config("profile-snapshot");
        fs::write(
            &path,
            "# preserve this comment\nversion = 1\n\n[defaults.output]\nbilingual = true\n",
        )
        .expect("write config");
        let settings = ResolvedSettings::default()
            .with_overrides(SettingsOverrides {
                backend: BackendOverrides {
                    id: Some("openai".to_owned()),
                    model: Some("gpt-test".to_owned()),
                    api_format: Some(crate::ApiFormat::OpenaiChat),
                    api_key: Some("inline-secret".to_owned()),
                    api_key_env: Some("OPENAI_API_KEY".to_owned()),
                    auth_header: Some("X-Secret".to_owned()),
                    ..BackendOverrides::default()
                },
                output: OutputOverrides {
                    bilingual: Some(true),
                    ..OutputOverrides::default()
                },
                ..SettingsOverrides::default()
            })
            .expect("settings");

        append_profile_snapshot(&path, "copy", &settings).expect("append profile");

        let content = fs::read_to_string(&path).expect("read config");
        assert!(content.starts_with("# preserve this comment"));
        assert!(content.contains("[profiles.copy.backend]"));
        assert!(content.contains("id = \"openai\""));
        assert!(content.contains("api_key_env = \"OPENAI_API_KEY\""));
        assert!(!content.contains("inline-secret"));
        assert!(!content.contains("X-Secret"));

        let parsed = ConfigFile::load(&path).expect("parse updated config");
        let (copied, _) = parsed
            .resolve(Some("copy"), SettingsOverrides::default())
            .expect("resolve copied profile");
        assert_eq!(copied.backend.id, "openai");
        assert_eq!(copied.backend.model, "gpt-test");
        assert!(copied.output.bilingual);
        let _ = fs::remove_file(path);
    }
}
