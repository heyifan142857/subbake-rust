use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use subbake_core::entities::{
    BilingualOrder, DEFAULT_AGENT_REPAIR_ATTEMPTS, DEFAULT_BATCH_SIZE, DEFAULT_BATCH_TOKEN_BUDGET,
    DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_RETRIES, DEFAULT_REVIEW_CONCURRENCY,
    DEFAULT_SOURCE_LANGUAGE, DEFAULT_TARGET_LANGUAGE, DEFAULT_TRANSLATION_CONCURRENCY,
    PipelineOptions, ReviewPolicy, TranslationMode,
};

use crate::error::{AdapterError, AdapterResult};
use crate::providers::{ApiFormat, BackendConfig};

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedSettings {
    pub output: OutputSettings,
    pub backend: BackendSettings,
    pub reviewer_backend: Option<BackendSettings>,
    pub translation: TranslationDomainSettings,
    pub transcription: TranscriptionDomainSettings,
    pub storage: StorageSettings,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionDomainSettings {
    pub model: Option<String>,
}

/// Compatibility alias for service request types. New configuration code
/// should name the complete resolved value `ResolvedSettings`.
pub type TranslationSettings = ResolvedSettings;

#[derive(Debug, Clone, PartialEq)]
pub struct OutputSettings {
    pub format: Option<String>,
    pub bilingual: bool,
    pub bilingual_order: BilingualOrder,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackendSettings {
    pub id: String,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub api_format: Option<ApiFormat>,
    pub endpoint_url: Option<String>,
    pub api_key_env: Option<String>,
    pub auth_header: Option<String>,
    pub auth_prefix: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationDomainSettings {
    pub source_language: String,
    pub target_language: String,
    pub batch_size: usize,
    pub batch_token_budget: usize,
    pub translation_concurrency: usize,
    pub review_concurrency: usize,
    pub mode: TranslationMode,
    pub review_policy: ReviewPolicy,
    pub terminology_preflight: bool,
    pub dry_run: bool,
    pub resume: bool,
    pub use_cache: bool,
    pub retries: usize,
    pub agent: bool,
    pub agent_repair_attempts: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StorageSettings {
    pub runtime_dir: Option<PathBuf>,
    pub glossary_path: Option<PathBuf>,
    pub whisper_binary_path: Option<PathBuf>,
    pub whisper_models_dir: Option<PathBuf>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SettingsOverrides {
    /// Optional v2 references into the configuration's `[backends]` table.
    pub translator: Option<String>,
    pub reviewer: Option<String>,
    pub backend: BackendOverrides,
    pub reviewer_backend: Option<BackendOverrides>,
    pub translation: TranslationOverrides,
    pub transcription: TranscriptionOverrides,
    pub output: OutputOverrides,
    pub storage: StorageOverrides,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendOverrides {
    pub id: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub api_format: Option<ApiFormat>,
    pub endpoint_url: Option<String>,
    pub api_key_env: Option<String>,
    pub auth_header: Option<String>,
    pub auth_prefix: Option<String>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TranslationOverrides {
    pub source_language: Option<String>,
    pub target_language: Option<String>,
    pub batch_size: Option<usize>,
    pub batch_token_budget: Option<usize>,
    pub translation_concurrency: Option<usize>,
    pub review_concurrency: Option<usize>,
    pub mode: Option<TranslationMode>,
    /// Legacy v1 input. `true` maps to turbo; `false` keeps the selected mode.
    pub fast_mode: Option<bool>,
    pub review_policy: Option<ReviewPolicy>,
    pub terminology_preflight: Option<bool>,
    pub dry_run: Option<bool>,
    pub resume: Option<bool>,
    pub use_cache: Option<bool>,
    pub retries: Option<usize>,
    pub agent: Option<bool>,
    pub agent_repair_attempts: Option<usize>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TranscriptionOverrides {
    pub model: Option<String>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputOverrides {
    pub format: Option<String>,
    pub bilingual: Option<bool>,
    pub bilingual_order: Option<BilingualOrder>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StorageOverrides {
    pub runtime_dir: Option<PathBuf>,
    pub glossary_path: Option<PathBuf>,
    pub whisper_binary_path: Option<PathBuf>,
    pub whisper_models_dir: Option<PathBuf>,
}

impl SettingsOverrides {
    pub fn merge(&mut self, other: Self) {
        if other.translator.is_some() {
            self.translator = other.translator;
        }
        if other.reviewer.is_some() {
            self.reviewer = other.reviewer;
        }
        self.backend.merge(other.backend);
        if let Some(other) = other.reviewer_backend {
            if let Some(current) = &mut self.reviewer_backend {
                current.merge(other);
            } else {
                self.reviewer_backend = Some(other);
            }
        }
        self.translation.merge(other.translation);
        self.transcription.merge(other.transcription);
        self.output.merge(other.output);
        self.storage.merge(other.storage);
    }

    pub fn from_resolved(settings: &ResolvedSettings) -> Self {
        Self {
            translator: None,
            reviewer: None,
            backend: BackendOverrides {
                id: Some(settings.backend.id.clone()),
                model: Some(settings.backend.model.clone()),
                api_key: settings.backend.api_key.clone(),
                base_url: settings.backend.base_url.clone(),
                api_format: settings.backend.api_format,
                endpoint_url: settings.backend.endpoint_url.clone(),
                api_key_env: settings.backend.api_key_env.clone(),
                auth_header: settings.backend.auth_header.clone(),
                auth_prefix: settings.backend.auth_prefix.clone(),
            },
            reviewer_backend: settings.reviewer_backend.as_ref().map(backend_overrides),
            translation: TranslationOverrides {
                source_language: Some(settings.translation.source_language.clone()),
                target_language: Some(settings.translation.target_language.clone()),
                batch_size: Some(settings.translation.batch_size),
                batch_token_budget: Some(settings.translation.batch_token_budget),
                translation_concurrency: Some(settings.translation.translation_concurrency),
                review_concurrency: Some(settings.translation.review_concurrency),
                mode: Some(settings.translation.mode),
                fast_mode: None,
                review_policy: Some(settings.translation.review_policy),
                terminology_preflight: Some(settings.translation.terminology_preflight),
                dry_run: Some(settings.translation.dry_run),
                resume: Some(settings.translation.resume),
                use_cache: Some(settings.translation.use_cache),
                retries: Some(settings.translation.retries),
                agent: Some(settings.translation.agent),
                agent_repair_attempts: Some(settings.translation.agent_repair_attempts),
            },
            transcription: TranscriptionOverrides {
                model: settings.transcription.model.clone(),
            },
            output: OutputOverrides {
                format: settings.output.format.clone(),
                bilingual: Some(settings.output.bilingual),
                bilingual_order: Some(settings.output.bilingual_order),
            },
            storage: StorageOverrides {
                runtime_dir: settings.storage.runtime_dir.clone(),
                glossary_path: settings.storage.glossary_path.clone(),
                whisper_binary_path: settings.storage.whisper_binary_path.clone(),
                whisper_models_dir: settings.storage.whisper_models_dir.clone(),
            },
        }
    }
}

macro_rules! merge_optional_fields {
    ($self:expr, $other:expr, $($field:ident),+ $(,)?) => {
        $(
            if $other.$field.is_some() {
                $self.$field = $other.$field;
            }
        )+
    };
}

impl BackendOverrides {
    pub(crate) fn merge(&mut self, other: Self) {
        merge_optional_fields!(
            self,
            other,
            id,
            model,
            api_key,
            base_url,
            api_format,
            endpoint_url,
            api_key_env,
            auth_header,
            auth_prefix
        );
    }
}

impl TranslationOverrides {
    fn merge(&mut self, other: Self) {
        merge_optional_fields!(
            self,
            other,
            source_language,
            target_language,
            batch_size,
            batch_token_budget,
            translation_concurrency,
            review_concurrency,
            mode,
            fast_mode,
            review_policy,
            terminology_preflight,
            dry_run,
            resume,
            use_cache,
            retries,
            agent,
            agent_repair_attempts
        );
    }
}

impl TranscriptionOverrides {
    fn merge(&mut self, other: Self) {
        merge_optional_fields!(self, other, model);
    }
}

fn backend_overrides(settings: &BackendSettings) -> BackendOverrides {
    BackendOverrides {
        id: Some(settings.id.clone()),
        model: Some(settings.model.clone()),
        api_key: settings.api_key.clone(),
        base_url: settings.base_url.clone(),
        api_format: settings.api_format,
        endpoint_url: settings.endpoint_url.clone(),
        api_key_env: settings.api_key_env.clone(),
        auth_header: settings.auth_header.clone(),
        auth_prefix: settings.auth_prefix.clone(),
    }
}

impl OutputOverrides {
    fn merge(&mut self, other: Self) {
        merge_optional_fields!(self, other, format, bilingual, bilingual_order);
    }
}

impl StorageOverrides {
    fn merge(&mut self, other: Self) {
        merge_optional_fields!(
            self,
            other,
            runtime_dir,
            glossary_path,
            whisper_binary_path,
            whisper_models_dir
        );
    }
}

impl Default for ResolvedSettings {
    fn default() -> Self {
        Self {
            output: OutputSettings {
                format: None,
                bilingual: false,
                bilingual_order: BilingualOrder::default(),
            },
            backend: BackendSettings {
                id: DEFAULT_PROVIDER.to_owned(),
                model: DEFAULT_MODEL.to_owned(),
                api_key: None,
                base_url: None,
                api_format: None,
                endpoint_url: None,
                api_key_env: None,
                auth_header: None,
                auth_prefix: None,
            },
            reviewer_backend: None,
            translation: TranslationDomainSettings {
                source_language: DEFAULT_SOURCE_LANGUAGE.to_owned(),
                target_language: DEFAULT_TARGET_LANGUAGE.to_owned(),
                batch_size: DEFAULT_BATCH_SIZE,
                batch_token_budget: DEFAULT_BATCH_TOKEN_BUDGET,
                translation_concurrency: DEFAULT_TRANSLATION_CONCURRENCY,
                review_concurrency: DEFAULT_REVIEW_CONCURRENCY,
                mode: TranslationMode::Turbo,
                review_policy: ReviewPolicy::Off,
                terminology_preflight: true,
                dry_run: false,
                resume: true,
                use_cache: true,
                retries: DEFAULT_RETRIES,
                agent: true,
                agent_repair_attempts: DEFAULT_AGENT_REPAIR_ATTEMPTS,
            },
            transcription: TranscriptionDomainSettings { model: None },
            storage: StorageSettings {
                runtime_dir: None,
                glossary_path: None,
                whisper_binary_path: None,
                whisper_models_dir: None,
            },
        }
    }
}

impl ResolvedSettings {
    pub fn with_overrides(mut self, overrides: SettingsOverrides) -> AdapterResult<Self> {
        self.apply_overrides(overrides);
        self.validate()?;
        Ok(self)
    }

    pub fn apply_overrides(&mut self, overrides: SettingsOverrides) {
        let BackendOverrides {
            id,
            model,
            api_key,
            base_url,
            api_format,
            endpoint_url,
            api_key_env,
            auth_header,
            auth_prefix,
        } = overrides.backend;
        if let Some(value) = id {
            self.backend.id = value;
        }
        if let Some(value) = model {
            self.backend.model = value;
        }
        if let Some(value) = api_key {
            self.backend.api_key = Some(value);
        }
        if let Some(value) = base_url {
            self.backend.base_url = Some(value);
        }
        if let Some(value) = api_format {
            self.backend.api_format = Some(value);
        }
        if let Some(value) = endpoint_url {
            self.backend.endpoint_url = Some(value);
        }
        if let Some(value) = api_key_env {
            self.backend.api_key_env = Some(value);
        }
        if let Some(value) = auth_header {
            self.backend.auth_header = Some(value);
        }
        if let Some(value) = auth_prefix {
            self.backend.auth_prefix = Some(value);
        }
        if let Some(reviewer) = overrides.reviewer_backend {
            let mut settings = self
                .reviewer_backend
                .take()
                .unwrap_or_else(|| self.backend.clone());
            apply_backend_overrides(&mut settings, reviewer);
            self.reviewer_backend = Some(settings);
        }

        let TranslationOverrides {
            source_language,
            target_language,
            batch_size,
            batch_token_budget,
            translation_concurrency,
            review_concurrency,
            mode,
            fast_mode,
            review_policy,
            terminology_preflight,
            dry_run,
            resume,
            use_cache,
            retries,
            agent,
            agent_repair_attempts,
        } = overrides.translation;
        if let Some(model) = overrides.transcription.model {
            self.transcription.model = Some(model);
        }
        if let Some(value) = mode {
            self.apply_mode_defaults(value);
        }
        if fast_mode == Some(true) {
            self.apply_mode_defaults(TranslationMode::Turbo);
        }
        if let Some(value) = source_language {
            self.translation.source_language = value;
        }
        if let Some(value) = target_language {
            self.translation.target_language = value;
        }
        if let Some(value) = batch_size {
            self.translation.batch_size = value;
        }
        if let Some(value) = batch_token_budget {
            self.translation.batch_token_budget = value;
        }
        if let Some(value) = translation_concurrency {
            self.translation.translation_concurrency = value;
        }
        if let Some(value) = review_concurrency {
            self.translation.review_concurrency = value;
        }
        if let Some(value) = review_policy {
            self.translation.review_policy = value;
        }
        if let Some(value) = terminology_preflight {
            self.translation.terminology_preflight = value;
        }
        if let Some(value) = dry_run {
            self.translation.dry_run = value;
        }
        if let Some(value) = resume {
            self.translation.resume = value;
        }
        if let Some(value) = use_cache {
            self.translation.use_cache = value;
        }
        if let Some(value) = retries {
            self.translation.retries = value;
        }
        if let Some(value) = agent {
            self.translation.agent = value;
        }
        if let Some(value) = agent_repair_attempts {
            self.translation.agent_repair_attempts = value;
        }

        let OutputOverrides {
            format,
            bilingual,
            bilingual_order,
        } = overrides.output;
        if let Some(value) = format {
            self.output.format = Some(value);
        }
        if let Some(value) = bilingual {
            self.output.bilingual = value;
        }
        if let Some(value) = bilingual_order {
            self.output.bilingual_order = value;
        }

        let StorageOverrides {
            runtime_dir,
            glossary_path,
            whisper_binary_path,
            whisper_models_dir,
        } = overrides.storage;
        if let Some(value) = runtime_dir {
            self.storage.runtime_dir = Some(value);
        }
        if let Some(value) = glossary_path {
            self.storage.glossary_path = Some(value);
        }
        if let Some(value) = whisper_binary_path {
            self.storage.whisper_binary_path = Some(value);
        }
        if let Some(value) = whisper_models_dir {
            self.storage.whisper_models_dir = Some(value);
        }
    }

    pub fn validate(&self) -> AdapterResult<()> {
        for (name, value) in [
            ("backend.id", self.backend.id.as_str()),
            ("backend.model", self.backend.model.as_str()),
            (
                "translation.source_language",
                self.translation.source_language.as_str(),
            ),
            (
                "translation.target_language",
                self.translation.target_language.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                return Err(AdapterError::invalid_input(format!(
                    "configuration field `{name}` must not be empty"
                )));
            }
        }
        if self
            .transcription
            .model
            .as_deref()
            .is_some_and(|model| model.trim().is_empty())
        {
            return Err(AdapterError::invalid_input(
                "configuration field `transcription.model` must not be empty",
            ));
        }
        for (name, value) in [
            ("translation.batch_size", self.translation.batch_size),
            (
                "translation.batch_token_budget",
                self.translation.batch_token_budget,
            ),
            (
                "translation.translation_concurrency",
                self.translation.translation_concurrency,
            ),
            (
                "translation.review_concurrency",
                self.translation.review_concurrency,
            ),
        ] {
            if value == 0 {
                return Err(AdapterError::invalid_input(format!(
                    "configuration field `{name}` must be greater than zero"
                )));
            }
        }
        self.backend_config().validate()?;
        if let Some(config) = self.reviewer_backend_config() {
            config.validate()?;
        }
        Ok(())
    }

    pub fn backend_config(&self) -> BackendConfig {
        BackendConfig {
            id: self.backend.id.clone(),
            display_name: self.backend.id.clone(),
            api_format: self.backend.api_format,
            model: self.backend.model.clone(),
            api_key: self.backend.api_key.clone(),
            api_key_env: self.backend.api_key_env.clone(),
            base_url: self.backend.base_url.clone(),
            endpoint_url: self.backend.endpoint_url.clone(),
            auth_header: self.backend.auth_header.clone(),
            auth_prefix: self.backend.auth_prefix.clone(),
        }
    }

    pub fn reviewer_backend_config(&self) -> Option<BackendConfig> {
        self.reviewer_backend.as_ref().map(backend_config)
    }

    pub fn to_pipeline_options(
        &self,
        input_path: impl Into<PathBuf>,
        output_path: Option<PathBuf>,
    ) -> PipelineOptions {
        let mut options = PipelineOptions::new(input_path.into());
        options.output_path = output_path;
        options.output_format = self.output.format.clone();
        options.provider = self.backend.id.clone();
        options.model = self.backend.model.clone();
        options.provider_fingerprint = self.provider_fingerprint();
        options.reviewer_fingerprint = self.reviewer_backend.as_ref().and_then(backend_fingerprint);
        options.source_language = self.translation.source_language.clone();
        options.target_language = self.translation.target_language.clone();
        options.batch_size = self.translation.batch_size;
        options.batch_token_budget = self.translation.batch_token_budget;
        options.translation_concurrency = self.translation.translation_concurrency;
        options.review_concurrency = self.translation.review_concurrency;
        options.bilingual = self.output.bilingual;
        options.bilingual_order = self.output.bilingual_order;
        options.mode = self.translation.mode;
        options.review_policy = self.translation.review_policy;
        options.terminology_preflight = self.translation.terminology_preflight;
        options.dry_run = self.translation.dry_run;
        options.resume = self.translation.resume;
        options.use_cache = self.translation.use_cache;
        options.retries = self.translation.retries;
        options.agent = self.translation.agent;
        options.agent_repair_attempts = self.translation.agent_repair_attempts;
        options.runtime_dir = self.storage.runtime_dir.clone();
        options.glossary_path = self.storage.glossary_path.clone();
        options
    }

    pub fn output_format(&self) -> Option<&str> {
        self.output.format.as_deref()
    }

    pub fn runtime_dir(&self) -> Option<&Path> {
        self.storage.runtime_dir.as_deref()
    }

    pub fn glossary_path(&self) -> Option<&Path> {
        self.storage.glossary_path.as_deref()
    }

    fn provider_fingerprint(&self) -> Option<String> {
        backend_fingerprint(&self.backend)
    }

    fn apply_mode_defaults(&mut self, mode: TranslationMode) {
        self.translation.mode = mode;
        match mode {
            TranslationMode::Economy => {
                self.translation.batch_size = 160;
                self.translation.batch_token_budget = 6_000;
                self.translation.translation_concurrency = 3;
                self.translation.review_concurrency = 1;
                self.translation.review_policy = ReviewPolicy::Off;
                self.translation.terminology_preflight = false;
            }
            TranslationMode::Turbo => {
                self.translation.batch_size = 96;
                self.translation.batch_token_budget = 2_400;
                self.translation.translation_concurrency = 8;
                self.translation.review_concurrency = 4;
                self.translation.review_policy = ReviewPolicy::Off;
                self.translation.terminology_preflight = false;
            }
            TranslationMode::Cinema => {
                self.translation.batch_size = 48;
                self.translation.batch_token_budget = 1_600;
                self.translation.translation_concurrency = 4;
                self.translation.review_concurrency = 3;
                self.translation.review_policy = ReviewPolicy::Full;
                self.translation.terminology_preflight = true;
            }
        }
    }
}

fn apply_backend_overrides(settings: &mut BackendSettings, overrides: BackendOverrides) {
    if let Some(value) = overrides.id {
        settings.id = value;
    }
    if let Some(value) = overrides.model {
        settings.model = value;
    }
    if let Some(value) = overrides.api_key {
        settings.api_key = Some(value);
    }
    if let Some(value) = overrides.base_url {
        settings.base_url = Some(value);
    }
    if let Some(value) = overrides.api_format {
        settings.api_format = Some(value);
    }
    if let Some(value) = overrides.endpoint_url {
        settings.endpoint_url = Some(value);
    }
    if let Some(value) = overrides.api_key_env {
        settings.api_key_env = Some(value);
    }
    if let Some(value) = overrides.auth_header {
        settings.auth_header = Some(value);
    }
    if let Some(value) = overrides.auth_prefix {
        settings.auth_prefix = Some(value);
    }
}

fn backend_config(settings: &BackendSettings) -> BackendConfig {
    BackendConfig {
        id: settings.id.clone(),
        display_name: settings.id.clone(),
        api_format: settings.api_format,
        model: settings.model.clone(),
        api_key: settings.api_key.clone(),
        api_key_env: settings.api_key_env.clone(),
        base_url: settings.base_url.clone(),
        endpoint_url: settings.endpoint_url.clone(),
        auth_header: settings.auth_header.clone(),
        auth_prefix: settings.auth_prefix.clone(),
    }
}

fn backend_fingerprint(settings: &BackendSettings) -> Option<String> {
    if settings.id.eq_ignore_ascii_case("mock") {
        return None;
    }
    let config = backend_config(settings);
    let format = config.api_format?.as_str();
    let endpoint = config.endpoint_url.or(config.base_url).unwrap_or_default();
    Some(format!(
        "{}|{}|{}|{}",
        config.id,
        format,
        endpoint.trim_end_matches('/'),
        config.model
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_overrides_apply_to_their_owner() {
        let settings = ResolvedSettings::default()
            .with_overrides(SettingsOverrides {
                backend: BackendOverrides {
                    id: Some("openai".to_owned()),
                    model: Some("gpt-test".to_owned()),
                    api_format: Some(ApiFormat::OpenaiChat),
                    ..BackendOverrides::default()
                },
                translation: TranslationOverrides {
                    batch_size: Some(12),
                    ..TranslationOverrides::default()
                },
                output: OutputOverrides {
                    bilingual: Some(true),
                    ..OutputOverrides::default()
                },
                storage: StorageOverrides {
                    runtime_dir: Some(".runtime".into()),
                    ..StorageOverrides::default()
                },
                ..SettingsOverrides::default()
            })
            .expect("valid overrides");

        assert_eq!(settings.backend.id, "openai");
        assert_eq!(settings.backend.model, "gpt-test");
        assert_eq!(settings.translation.batch_size, 12);
        assert!(settings.output.bilingual);
        assert_eq!(settings.storage.runtime_dir, Some(".runtime".into()));
    }

    #[test]
    fn cinema_mode_applies_defaults_before_explicit_overrides() {
        let settings = ResolvedSettings::default()
            .with_overrides(SettingsOverrides {
                translation: TranslationOverrides {
                    mode: Some(TranslationMode::Cinema),
                    translation_concurrency: Some(7),
                    ..TranslationOverrides::default()
                },
                ..SettingsOverrides::default()
            })
            .expect("valid settings");
        assert_eq!(settings.translation.mode, TranslationMode::Cinema);
        assert_eq!(settings.translation.batch_size, 48);
        assert_eq!(settings.translation.translation_concurrency, 7);
        assert_eq!(settings.translation.review_policy, ReviewPolicy::Full);
    }

    #[test]
    fn validation_rejects_zero_work_limits() {
        let error = ResolvedSettings::default()
            .with_overrides(SettingsOverrides {
                translation: TranslationOverrides {
                    batch_size: Some(0),
                    ..TranslationOverrides::default()
                },
                ..SettingsOverrides::default()
            })
            .expect_err("zero batch size");
        assert!(error.to_string().contains("batch_size"));
    }
}
