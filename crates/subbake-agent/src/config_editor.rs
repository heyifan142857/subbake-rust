use std::fmt::{Debug, Formatter};
use std::path::PathBuf;

use subbake_adapters::{
    ApiFormat, ConfigEditTarget, ConfigFieldUpdate, ConfigFile, ConfigScalar, ResolvedSettings,
    SettingsOverrides,
};

use crate::error::{AgentError, AgentResult};
use crate::presentation::ProfileChoice;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConfigSection {
    Profile,
    Provider,
    Reviewer,
    Agent,
    Translation,
    Transcription,
    Output,
    Storage,
}

impl ConfigSection {
    pub const ALL: [Self; 8] = [
        Self::Profile,
        Self::Provider,
        Self::Reviewer,
        Self::Agent,
        Self::Translation,
        Self::Transcription,
        Self::Output,
        Self::Storage,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Profile => "Profile",
            Self::Provider => "Provider",
            Self::Reviewer => "Reviewer",
            Self::Agent => "Agent",
            Self::Translation => "Translation",
            Self::Transcription => "Transcription",
            Self::Output => "Output",
            Self::Storage => "Storage",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFieldKind {
    Text,
    Secret,
    Integer,
    Float,
    Boolean,
    Choice(&'static [&'static str]),
    Profile,
}

const API_FORMATS: &[&str] = &[
    "anthropic_messages",
    "openai_chat",
    "openai_responses",
    "gemini_generate_content",
];
const TRANSLATION_MODES: &[&str] = &["economy", "turbo", "cinema"];
const REVIEW_POLICIES: &[&str] = &["off", "targeted", "full"];
const BILINGUAL_ORDERS: &[&str] = &["source_first", "target_first"];

macro_rules! config_fields {
    ($($variant:ident => ($section:ident, $label:literal, $kind:expr, [$($path:literal),+])),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum ConfigFieldId { $($variant),+ }

        impl ConfigFieldId {
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            pub const fn section(self) -> ConfigSection {
                match self { $(Self::$variant => ConfigSection::$section),+ }
            }

            pub const fn label(self) -> &'static str {
                match self { $(Self::$variant => $label),+ }
            }

            pub const fn kind(self) -> ConfigFieldKind {
                match self { $(Self::$variant => $kind),+ }
            }

            pub const fn path(self) -> &'static [&'static str] {
                match self { $(Self::$variant => &[$($path),+]),+ }
            }

            pub fn toml_key(self) -> String {
                self.path().join(".")
            }

            pub const fn hint(self) -> Option<&'static str> {
                match self {
                    Self::ProviderModel => Some("Model used for translation requests"),
                    Self::TranscriptionModel => Some("Whisper model name, for example large-v3-turbo"),
                    Self::WhisperBinaryPath => Some("Path to an existing whisper-cli executable; the installer is optional"),
                    Self::WhisperModelsDir => Some("Directory containing ggml-*.bin or ggml-*.gguf models"),
                    _ => None,
                }
            }
        }
    };
}

config_fields! {
    ActiveProfile => (Profile, "Active profile", ConfigFieldKind::Profile, ["profile"]),
    Translator => (Provider, "Backend reference", ConfigFieldKind::Text, ["translator"]),
    ProviderId => (Provider, "Provider id", ConfigFieldKind::Text, ["backend", "id"]),
    ProviderModel => (Provider, "Translation model", ConfigFieldKind::Text, ["backend", "model"]),
    ProviderApiFormat => (Provider, "API format", ConfigFieldKind::Choice(API_FORMATS), ["backend", "api_format"]),
    ProviderBaseUrl => (Provider, "Base URL", ConfigFieldKind::Text, ["backend", "base_url"]),
    ProviderEndpointUrl => (Provider, "Endpoint URL", ConfigFieldKind::Text, ["backend", "endpoint_url"]),
    ProviderApiKey => (Provider, "API key", ConfigFieldKind::Secret, ["backend", "api_key"]),
    ProviderApiKeyEnv => (Provider, "API key env", ConfigFieldKind::Text, ["backend", "api_key_env"]),
    ProviderAuthHeader => (Provider, "Authorization header", ConfigFieldKind::Secret, ["backend", "auth_header"]),
    ProviderAuthPrefix => (Provider, "Authorization prefix", ConfigFieldKind::Text, ["backend", "auth_prefix"]),
    ProviderTimeout => (Provider, "Timeout seconds", ConfigFieldKind::Float, ["backend", "timeout_seconds"]),
    Reviewer => (Reviewer, "Backend reference", ConfigFieldKind::Text, ["reviewer"]),
    ReviewerId => (Reviewer, "Provider id", ConfigFieldKind::Text, ["reviewer_backend", "id"]),
    ReviewerModel => (Reviewer, "Model", ConfigFieldKind::Text, ["reviewer_backend", "model"]),
    ReviewerApiFormat => (Reviewer, "API format", ConfigFieldKind::Choice(API_FORMATS), ["reviewer_backend", "api_format"]),
    ReviewerBaseUrl => (Reviewer, "Base URL", ConfigFieldKind::Text, ["reviewer_backend", "base_url"]),
    ReviewerEndpointUrl => (Reviewer, "Endpoint URL", ConfigFieldKind::Text, ["reviewer_backend", "endpoint_url"]),
    ReviewerApiKey => (Reviewer, "API key", ConfigFieldKind::Secret, ["reviewer_backend", "api_key"]),
    ReviewerApiKeyEnv => (Reviewer, "API key env", ConfigFieldKind::Text, ["reviewer_backend", "api_key_env"]),
    ReviewerAuthHeader => (Reviewer, "Authorization header", ConfigFieldKind::Secret, ["reviewer_backend", "auth_header"]),
    ReviewerAuthPrefix => (Reviewer, "Authorization prefix", ConfigFieldKind::Text, ["reviewer_backend", "auth_prefix"]),
    ReviewerTimeout => (Reviewer, "Timeout seconds", ConfigFieldKind::Float, ["reviewer_backend", "timeout_seconds"]),
    AgentMaxSteps => (Agent, "Maximum steps", ConfigFieldKind::Integer, ["agent", "max_steps"]),
    AgentAutoApprove => (Agent, "Auto-approve commands", ConfigFieldKind::Boolean, ["agent", "auto_approve_commands"]),
    SourceLanguage => (Translation, "Source language", ConfigFieldKind::Text, ["translation", "source_language"]),
    TargetLanguage => (Translation, "Target language", ConfigFieldKind::Text, ["translation", "target_language"]),
    SubtitleStreamIndex => (Translation, "Subtitle stream index", ConfigFieldKind::Integer, ["translation", "subtitle_stream_index"]),
    BatchSize => (Translation, "Batch size", ConfigFieldKind::Integer, ["translation", "batch_size"]),
    BatchTokenBudget => (Translation, "Batch token budget", ConfigFieldKind::Integer, ["translation", "batch_token_budget"]),
    TranslationConcurrency => (Translation, "Translation concurrency", ConfigFieldKind::Integer, ["translation", "translation_concurrency"]),
    ReviewConcurrency => (Translation, "Review concurrency", ConfigFieldKind::Integer, ["translation", "review_concurrency"]),
    TranslationMode => (Translation, "Translation mode", ConfigFieldKind::Choice(TRANSLATION_MODES), ["translation", "mode"]),
    ReviewPolicy => (Translation, "Review policy", ConfigFieldKind::Choice(REVIEW_POLICIES), ["translation", "review_policy"]),
    TerminologyPreflight => (Translation, "Terminology preflight", ConfigFieldKind::Boolean, ["translation", "terminology_preflight"]),
    OnlineTerminology => (Translation, "Online terminology", ConfigFieldKind::Boolean, ["translation", "online_terminology"]),
    PreserveNames => (Translation, "Preserve names", ConfigFieldKind::Boolean, ["translation", "preserve_names"]),
    DryRun => (Translation, "Dry run", ConfigFieldKind::Boolean, ["translation", "dry_run"]),
    Resume => (Translation, "Resume", ConfigFieldKind::Boolean, ["translation", "resume"]),
    UseCache => (Translation, "Use cache", ConfigFieldKind::Boolean, ["translation", "use_cache"]),
    Retries => (Translation, "Retries", ConfigFieldKind::Integer, ["translation", "retries"]),
    TranslationAgent => (Translation, "Agent repair", ConfigFieldKind::Boolean, ["translation", "agent"]),
    AgentRepairAttempts => (Translation, "Repair attempts", ConfigFieldKind::Integer, ["translation", "agent_repair_attempts"]),
    MaxRequests => (Translation, "Maximum requests", ConfigFieldKind::Integer, ["translation", "max_requests"]),
    MaxTokens => (Translation, "Maximum tokens", ConfigFieldKind::Integer, ["translation", "max_tokens"]),
    TranscriptionModel => (Transcription, "Whisper model", ConfigFieldKind::Text, ["transcription", "model"]),
    OutputFormat => (Output, "Output format", ConfigFieldKind::Text, ["output", "format"]),
    Bilingual => (Output, "Bilingual output", ConfigFieldKind::Boolean, ["output", "bilingual"]),
    BilingualOrder => (Output, "Bilingual order", ConfigFieldKind::Choice(BILINGUAL_ORDERS), ["output", "bilingual_order"]),
    BilingualFontScale => (Output, "Bilingual font scale", ConfigFieldKind::Float, ["output", "bilingual_font_scale"]),
    PreserveSourceContainer => (Output, "Preserve source container", ConfigFieldKind::Boolean, ["output", "preserve_source_container"]),
    RuntimeDir => (Storage, "Runtime directory", ConfigFieldKind::Text, ["storage", "runtime_dir"]),
    GlossaryPath => (Storage, "Glossary path", ConfigFieldKind::Text, ["storage", "glossary_path"]),
    WhisperBinaryPath => (Transcription, "whisper-cli path", ConfigFieldKind::Text, ["storage", "whisper_binary_path"]),
    WhisperModelsDir => (Transcription, "Whisper models directory", ConfigFieldKind::Text, ["storage", "whisper_models_dir"]),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigFieldView {
    pub id: ConfigFieldId,
    pub value: String,
    pub inherited: bool,
    pub configured: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigEditorSnapshot {
    pub path: PathBuf,
    pub target: ConfigEditTarget,
    pub active_profile: Option<String>,
    pub profiles: Vec<ProfileChoice>,
    pub fields: Vec<ConfigFieldView>,
}

#[derive(Clone, PartialEq, Eq)]
pub struct ConfigChange {
    pub id: ConfigFieldId,
    pub value: Option<String>,
}

impl Debug for ConfigChange {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("ConfigChange");
        debug.field("id", &self.id);
        if matches!(self.id.kind(), ConfigFieldKind::Secret) && self.value.is_some() {
            debug.field("value", &"[REDACTED]");
        } else {
            debug.field("value", &self.value);
        }
        debug.finish()
    }
}

impl ConfigChange {
    pub(crate) fn into_update(self) -> AgentResult<ConfigFieldUpdate> {
        if self.id == ConfigFieldId::ActiveProfile {
            return Err(AgentError::invalid_input(
                "the active profile is changed through the profile picker",
            ));
        }
        let secret = matches!(self.id.kind(), ConfigFieldKind::Secret);
        let value = self
            .value
            .map(|value| parse_scalar(self.id.kind(), value))
            .transpose()?;
        Ok(ConfigFieldUpdate {
            path: self.id.path().to_vec(),
            value,
            secret,
        })
    }
}

fn parse_scalar(kind: ConfigFieldKind, value: String) -> AgentResult<ConfigScalar> {
    match kind {
        ConfigFieldKind::Integer => value
            .parse::<i64>()
            .map(ConfigScalar::Integer)
            .map_err(|_| AgentError::invalid_input(format!("`{value}` is not an integer"))),
        ConfigFieldKind::Float => value
            .parse::<f64>()
            .map(ConfigScalar::Float)
            .map_err(|_| AgentError::invalid_input(format!("`{value}` is not a number"))),
        ConfigFieldKind::Boolean => value
            .parse::<bool>()
            .map(ConfigScalar::Boolean)
            .map_err(|_| AgentError::invalid_input(format!("`{value}` is not a boolean"))),
        ConfigFieldKind::Choice(choices) if choices.contains(&value.as_str()) => {
            Ok(ConfigScalar::String(value))
        }
        ConfigFieldKind::Choice(choices) => Err(AgentError::invalid_input(format!(
            "`{value}` must be one of: {}",
            choices.join(", ")
        ))),
        ConfigFieldKind::Text | ConfigFieldKind::Secret => Ok(ConfigScalar::String(value)),
        ConfigFieldKind::Profile => Err(AgentError::invalid_input(
            "profile values cannot be written as configuration fields",
        )),
    }
}

pub(crate) fn build_snapshot(
    path: PathBuf,
    config: &ConfigFile,
    active_profile: Option<&str>,
    profiles: Vec<ProfileChoice>,
) -> AgentResult<ConfigEditorSnapshot> {
    let selected = config
        .selected_profile(active_profile)
        .map_err(subbake_adapters::AdapterError::from)?;
    let target = selected.map_or(ConfigEditTarget::Defaults, |name| {
        ConfigEditTarget::Profile(name.to_owned())
    });
    let layer = selected
        .and_then(|name| config.profiles.get(name))
        .unwrap_or(&config.defaults);
    let mut effective_overrides = config.defaults.clone();
    if let Some(name) = selected
        && let Some(profile) = config.profiles.get(name)
    {
        effective_overrides.merge(profile.clone());
    }
    let (resolved, _) = config
        .resolve(selected, SettingsOverrides::default())
        .map_err(subbake_adapters::AdapterError::from)?;
    let fields = ConfigFieldId::ALL
        .iter()
        .copied()
        .map(|id| field_view(id, layer, &effective_overrides, &resolved, selected))
        .collect();
    Ok(ConfigEditorSnapshot {
        path,
        target,
        active_profile: selected.map(str::to_owned),
        profiles,
        fields,
    })
}

fn field_view(
    id: ConfigFieldId,
    layer: &SettingsOverrides,
    effective: &SettingsOverrides,
    resolved: &ResolvedSettings,
    selected: Option<&str>,
) -> ConfigFieldView {
    let inherited = !has_override(id, layer);
    let (value, configured) = effective_value(id, effective, resolved, selected);
    ConfigFieldView {
        id,
        value: if matches!(id.kind(), ConfigFieldKind::Secret) {
            String::new()
        } else {
            value
        },
        inherited,
        configured,
    }
}

macro_rules! some_string {
    ($value:expr) => {
        $value.as_ref().map(ToString::to_string).unwrap_or_default()
    };
}

fn effective_value(
    id: ConfigFieldId,
    effective: &SettingsOverrides,
    resolved: &ResolvedSettings,
    selected: Option<&str>,
) -> (String, bool) {
    let reviewer = resolved.reviewer_backend.as_ref();
    let value = match id {
        ConfigFieldId::ActiveProfile => selected.unwrap_or("defaults").to_owned(),
        ConfigFieldId::Translator => effective.translator.clone().unwrap_or_default(),
        ConfigFieldId::ProviderId => resolved.backend.id.clone(),
        ConfigFieldId::ProviderModel => resolved.backend.model.clone(),
        ConfigFieldId::ProviderApiFormat => resolved
            .backend
            .api_format
            .map(ApiFormat::as_str)
            .unwrap_or_default()
            .to_owned(),
        ConfigFieldId::ProviderBaseUrl => some_string!(resolved.backend.base_url),
        ConfigFieldId::ProviderEndpointUrl => some_string!(resolved.backend.endpoint_url),
        ConfigFieldId::ProviderApiKey => some_string!(resolved.backend.api_key),
        ConfigFieldId::ProviderApiKeyEnv => some_string!(resolved.backend.api_key_env),
        ConfigFieldId::ProviderAuthHeader => some_string!(resolved.backend.auth_header),
        ConfigFieldId::ProviderAuthPrefix => some_string!(resolved.backend.auth_prefix),
        ConfigFieldId::ProviderTimeout => resolved.backend.timeout_seconds.to_string(),
        ConfigFieldId::Reviewer => effective.reviewer.clone().unwrap_or_default(),
        ConfigFieldId::ReviewerId => reviewer.map(|value| value.id.clone()).unwrap_or_default(),
        ConfigFieldId::ReviewerModel => reviewer
            .map(|value| value.model.clone())
            .unwrap_or_default(),
        ConfigFieldId::ReviewerApiFormat => reviewer
            .and_then(|value| value.api_format)
            .map(ApiFormat::as_str)
            .unwrap_or_default()
            .to_owned(),
        ConfigFieldId::ReviewerBaseUrl => reviewer
            .and_then(|value| value.base_url.clone())
            .unwrap_or_default(),
        ConfigFieldId::ReviewerEndpointUrl => reviewer
            .and_then(|value| value.endpoint_url.clone())
            .unwrap_or_default(),
        ConfigFieldId::ReviewerApiKey => reviewer
            .and_then(|value| value.api_key.clone())
            .unwrap_or_default(),
        ConfigFieldId::ReviewerApiKeyEnv => reviewer
            .and_then(|value| value.api_key_env.clone())
            .unwrap_or_default(),
        ConfigFieldId::ReviewerAuthHeader => reviewer
            .and_then(|value| value.auth_header.clone())
            .unwrap_or_default(),
        ConfigFieldId::ReviewerAuthPrefix => reviewer
            .and_then(|value| value.auth_prefix.clone())
            .unwrap_or_default(),
        ConfigFieldId::ReviewerTimeout => reviewer
            .map(|value| value.timeout_seconds.to_string())
            .unwrap_or_default(),
        ConfigFieldId::AgentMaxSteps => resolved.agent.max_steps.to_string(),
        ConfigFieldId::AgentAutoApprove => resolved.agent.auto_approve_commands.to_string(),
        ConfigFieldId::SourceLanguage => resolved.translation.source_language.clone(),
        ConfigFieldId::TargetLanguage => resolved.translation.target_language.clone(),
        ConfigFieldId::SubtitleStreamIndex => {
            some_string!(resolved.translation.subtitle_stream_index)
        }
        ConfigFieldId::BatchSize => resolved.translation.batch_size.to_string(),
        ConfigFieldId::BatchTokenBudget => resolved.translation.batch_token_budget.to_string(),
        ConfigFieldId::TranslationConcurrency => {
            resolved.translation.translation_concurrency.to_string()
        }
        ConfigFieldId::ReviewConcurrency => resolved.translation.review_concurrency.to_string(),
        ConfigFieldId::TranslationMode => resolved.translation.mode.as_str().to_owned(),
        ConfigFieldId::ReviewPolicy => resolved.translation.review_policy.as_str().to_owned(),
        ConfigFieldId::TerminologyPreflight => {
            resolved.translation.terminology_preflight.to_string()
        }
        ConfigFieldId::OnlineTerminology => resolved.translation.online_terminology.to_string(),
        ConfigFieldId::PreserveNames => resolved.translation.preserve_names.to_string(),
        ConfigFieldId::DryRun => resolved.translation.dry_run.to_string(),
        ConfigFieldId::Resume => resolved.translation.resume.to_string(),
        ConfigFieldId::UseCache => resolved.translation.use_cache.to_string(),
        ConfigFieldId::Retries => resolved.translation.retries.to_string(),
        ConfigFieldId::TranslationAgent => resolved.translation.agent.to_string(),
        ConfigFieldId::AgentRepairAttempts => {
            resolved.translation.agent_repair_attempts.to_string()
        }
        ConfigFieldId::MaxRequests => some_string!(resolved.translation.max_requests),
        ConfigFieldId::MaxTokens => some_string!(resolved.translation.max_tokens),
        ConfigFieldId::TranscriptionModel => some_string!(resolved.transcription.model),
        ConfigFieldId::OutputFormat => some_string!(resolved.output.format),
        ConfigFieldId::Bilingual => resolved.output.bilingual.to_string(),
        ConfigFieldId::BilingualOrder => resolved.output.bilingual_order.as_str().to_owned(),
        ConfigFieldId::BilingualFontScale => resolved.output.bilingual_font_scale.to_string(),
        ConfigFieldId::PreserveSourceContainer => {
            resolved.output.preserve_source_container.to_string()
        }
        ConfigFieldId::RuntimeDir => resolved
            .storage
            .runtime_dir
            .as_ref()
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_default(),
        ConfigFieldId::GlossaryPath => resolved
            .storage
            .glossary_path
            .as_ref()
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_default(),
        ConfigFieldId::WhisperBinaryPath => resolved
            .storage
            .whisper_binary_path
            .as_ref()
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_default(),
        ConfigFieldId::WhisperModelsDir => resolved
            .storage
            .whisper_models_dir
            .as_ref()
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_default(),
    };
    let configured = !value.is_empty();
    (value, configured)
}

fn has_override(id: ConfigFieldId, settings: &SettingsOverrides) -> bool {
    let backend = &settings.backend;
    let reviewer = settings.reviewer_backend.as_ref();
    match id {
        ConfigFieldId::ActiveProfile => true,
        ConfigFieldId::Translator => settings.translator.is_some(),
        ConfigFieldId::ProviderId => backend.id.is_some(),
        ConfigFieldId::ProviderModel => backend.model.is_some(),
        ConfigFieldId::ProviderApiFormat => backend.api_format.is_some(),
        ConfigFieldId::ProviderBaseUrl => backend.base_url.is_some(),
        ConfigFieldId::ProviderEndpointUrl => backend.endpoint_url.is_some(),
        ConfigFieldId::ProviderApiKey => backend.api_key.is_some(),
        ConfigFieldId::ProviderApiKeyEnv => backend.api_key_env.is_some(),
        ConfigFieldId::ProviderAuthHeader => backend.auth_header.is_some(),
        ConfigFieldId::ProviderAuthPrefix => backend.auth_prefix.is_some(),
        ConfigFieldId::ProviderTimeout => backend.timeout_seconds.is_some(),
        ConfigFieldId::Reviewer => settings.reviewer.is_some(),
        ConfigFieldId::ReviewerId => reviewer.is_some_and(|value| value.id.is_some()),
        ConfigFieldId::ReviewerModel => reviewer.is_some_and(|value| value.model.is_some()),
        ConfigFieldId::ReviewerApiFormat => {
            reviewer.is_some_and(|value| value.api_format.is_some())
        }
        ConfigFieldId::ReviewerBaseUrl => reviewer.is_some_and(|value| value.base_url.is_some()),
        ConfigFieldId::ReviewerEndpointUrl => {
            reviewer.is_some_and(|value| value.endpoint_url.is_some())
        }
        ConfigFieldId::ReviewerApiKey => reviewer.is_some_and(|value| value.api_key.is_some()),
        ConfigFieldId::ReviewerApiKeyEnv => {
            reviewer.is_some_and(|value| value.api_key_env.is_some())
        }
        ConfigFieldId::ReviewerAuthHeader => {
            reviewer.is_some_and(|value| value.auth_header.is_some())
        }
        ConfigFieldId::ReviewerAuthPrefix => {
            reviewer.is_some_and(|value| value.auth_prefix.is_some())
        }
        ConfigFieldId::ReviewerTimeout => {
            reviewer.is_some_and(|value| value.timeout_seconds.is_some())
        }
        ConfigFieldId::AgentMaxSteps => settings.agent.max_steps.is_some(),
        ConfigFieldId::AgentAutoApprove => settings.agent.auto_approve_commands.is_some(),
        ConfigFieldId::SourceLanguage => settings.translation.source_language.is_some(),
        ConfigFieldId::TargetLanguage => settings.translation.target_language.is_some(),
        ConfigFieldId::SubtitleStreamIndex => settings.translation.subtitle_stream_index.is_some(),
        ConfigFieldId::BatchSize => settings.translation.batch_size.is_some(),
        ConfigFieldId::BatchTokenBudget => settings.translation.batch_token_budget.is_some(),
        ConfigFieldId::TranslationConcurrency => {
            settings.translation.translation_concurrency.is_some()
        }
        ConfigFieldId::ReviewConcurrency => settings.translation.review_concurrency.is_some(),
        ConfigFieldId::TranslationMode => settings.translation.mode.is_some(),
        ConfigFieldId::ReviewPolicy => settings.translation.review_policy.is_some(),
        ConfigFieldId::TerminologyPreflight => settings.translation.terminology_preflight.is_some(),
        ConfigFieldId::OnlineTerminology => settings.translation.online_terminology.is_some(),
        ConfigFieldId::PreserveNames => settings.translation.preserve_names.is_some(),
        ConfigFieldId::DryRun => settings.translation.dry_run.is_some(),
        ConfigFieldId::Resume => settings.translation.resume.is_some(),
        ConfigFieldId::UseCache => settings.translation.use_cache.is_some(),
        ConfigFieldId::Retries => settings.translation.retries.is_some(),
        ConfigFieldId::TranslationAgent => settings.translation.agent.is_some(),
        ConfigFieldId::AgentRepairAttempts => settings.translation.agent_repair_attempts.is_some(),
        ConfigFieldId::MaxRequests => settings.translation.max_requests.is_some(),
        ConfigFieldId::MaxTokens => settings.translation.max_tokens.is_some(),
        ConfigFieldId::TranscriptionModel => settings.transcription.model.is_some(),
        ConfigFieldId::OutputFormat => settings.output.format.is_some(),
        ConfigFieldId::Bilingual => settings.output.bilingual.is_some(),
        ConfigFieldId::BilingualOrder => settings.output.bilingual_order.is_some(),
        ConfigFieldId::BilingualFontScale => settings.output.bilingual_font_scale.is_some(),
        ConfigFieldId::PreserveSourceContainer => {
            settings.output.preserve_source_container.is_some()
        }
        ConfigFieldId::RuntimeDir => settings.storage.runtime_dir.is_some(),
        ConfigFieldId::GlossaryPath => settings.storage.glossary_path.is_some(),
        ConfigFieldId::WhisperBinaryPath => settings.storage.whisper_binary_path.is_some(),
        ConfigFieldId::WhisperModelsDir => settings.storage.whisper_models_dir.is_some(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_changes_are_redacted_from_debug_output() {
        let change = ConfigChange {
            id: ConfigFieldId::ProviderApiKey,
            value: Some("do-not-print-me".to_owned()),
        };
        let debug = format!("{change:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("do-not-print-me"));
        let adapter_debug = format!("{:?}", change.into_update().expect("convert change"));
        assert!(adapter_debug.contains("[REDACTED]"));
        assert!(!adapter_debug.contains("do-not-print-me"));
    }

    #[test]
    fn snapshot_marks_profile_values_as_inherited_until_overridden() {
        let config = ConfigFile::parse(
            r#"
            version = 2
            default_profile = "work"

            [defaults.translation]
            target_language = "French"

            [profiles.work.agent]
            max_steps = 24
            "#,
        )
        .expect("parse config");
        let snapshot = build_snapshot(PathBuf::from("subbake.toml"), &config, None, Vec::new())
            .expect("build snapshot");
        let target = snapshot
            .fields
            .iter()
            .find(|field| field.id == ConfigFieldId::TargetLanguage)
            .expect("target language");
        let steps = snapshot
            .fields
            .iter()
            .find(|field| field.id == ConfigFieldId::AgentMaxSteps)
            .expect("agent steps");
        assert_eq!(target.value, "French");
        assert!(target.inherited);
        assert_eq!(steps.value, "24");
        assert!(!steps.inherited);
    }

    #[test]
    fn registry_paths_are_unique_except_for_the_profile_action() {
        let mut paths = std::collections::HashSet::new();
        for id in ConfigFieldId::ALL {
            if *id == ConfigFieldId::ActiveProfile {
                continue;
            }
            assert!(
                paths.insert(id.path()),
                "duplicate field path: {:?}",
                id.path()
            );
        }
    }

    #[test]
    fn snapshot_exposes_translation_and_external_whisper_configuration() {
        let config = ConfigFile::parse(
            r#"
            version = 2
            default_profile = "work"

            [backends.remote]
            id = "openai"
            model = "translator-v1"
            api_format = "openai_chat"

            [profiles.work]
            translator = "remote"

            [profiles.work.transcription]
            model = "large-v3-turbo"

            [profiles.work.storage]
            whisper_binary_path = "/opt/whisper/bin/whisper-cli"
            whisper_models_dir = "/opt/whisper/models"
            "#,
        )
        .expect("parse config");
        let snapshot = build_snapshot(PathBuf::from("subbake.toml"), &config, None, Vec::new())
            .expect("build snapshot");

        let field = |id| {
            snapshot
                .fields
                .iter()
                .find(|field| field.id == id)
                .expect("registered field")
        };
        assert_eq!(field(ConfigFieldId::ProviderModel).value, "translator-v1");
        assert_eq!(
            field(ConfigFieldId::TranscriptionModel).value,
            "large-v3-turbo"
        );
        assert_eq!(
            field(ConfigFieldId::WhisperBinaryPath).value,
            "/opt/whisper/bin/whisper-cli"
        );
        assert_eq!(
            field(ConfigFieldId::WhisperModelsDir).value,
            "/opt/whisper/models"
        );
        assert_eq!(
            ConfigFieldId::WhisperBinaryPath.section(),
            ConfigSection::Transcription
        );
        assert_eq!(
            ConfigFieldId::WhisperBinaryPath.toml_key(),
            "storage.whisper_binary_path"
        );
    }

    #[test]
    fn tui_changes_write_translation_model_and_external_whisper_paths() {
        let cases = [
            (
                ConfigFieldId::ProviderModel,
                "translator-v2",
                vec!["backend", "model"],
            ),
            (
                ConfigFieldId::TranscriptionModel,
                "large-v3-turbo-q8_0",
                vec!["transcription", "model"],
            ),
            (
                ConfigFieldId::WhisperBinaryPath,
                "/usr/local/bin/whisper-cli",
                vec!["storage", "whisper_binary_path"],
            ),
            (
                ConfigFieldId::WhisperModelsDir,
                "/srv/whisper-models",
                vec!["storage", "whisper_models_dir"],
            ),
        ];

        for (id, value, expected_path) in cases {
            let update = ConfigChange {
                id,
                value: Some(value.to_owned()),
            }
            .into_update()
            .expect("convert TUI change");
            assert_eq!(update.path, expected_path);
            assert_eq!(update.value, Some(ConfigScalar::String(value.to_owned())));
        }
    }
}
