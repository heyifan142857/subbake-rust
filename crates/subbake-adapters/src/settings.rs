use std::path::{Path, PathBuf};

use subbake_core::entities::{
    DEFAULT_AGENT_REPAIR_ATTEMPTS, DEFAULT_BATCH_SIZE, DEFAULT_BATCH_TOKEN_BUDGET, DEFAULT_MODEL,
    DEFAULT_PROVIDER, DEFAULT_RETRIES, DEFAULT_REVIEW_CONCURRENCY, DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE, DEFAULT_TRANSLATION_CONCURRENCY, PipelineOptions, ReviewPolicy,
};

use crate::providers::{ApiFormat, BackendConfig, legacy_api_format};

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationSettings {
    pub output: OutputSettings,
    pub backend: BackendSettings,
    pub translation: TranslationDomainSettings,
    pub runtime: RuntimeSettings,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OutputSettings {
    pub format: Option<String>,
    pub bilingual: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackendSettings {
    pub provider: String,
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
    pub fast_mode: bool,
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
pub struct RuntimeSettings {
    pub runtime_dir: Option<PathBuf>,
    pub glossary_path: Option<PathBuf>,
}

impl TranslationSettingsPatch {
    /// Overlay `other` onto `self` — `Some` fields in `other` replace `self`.
    pub fn merge(&mut self, other: TranslationSettingsPatch) {
        if let Some(value) = other.output_format {
            self.output_format = Some(value);
        }
        if let Some(value) = other.provider {
            self.provider = Some(value);
        }
        if let Some(value) = other.model {
            self.model = Some(value);
        }
        if let Some(value) = other.api_key {
            self.api_key = Some(value);
        }
        if let Some(value) = other.base_url {
            self.base_url = Some(value);
        }
        if let Some(value) = other.api_format {
            self.api_format = Some(value);
        }
        if let Some(value) = other.endpoint_url {
            self.endpoint_url = Some(value);
        }
        if let Some(value) = other.api_key_env {
            self.api_key_env = Some(value);
        }
        if let Some(value) = other.auth_header {
            self.auth_header = Some(value);
        }
        if let Some(value) = other.auth_prefix {
            self.auth_prefix = Some(value);
        }
        if let Some(value) = other.source_language {
            self.source_language = Some(value);
        }
        if let Some(value) = other.target_language {
            self.target_language = Some(value);
        }
        if let Some(value) = other.batch_size {
            self.batch_size = Some(value);
        }
        if let Some(value) = other.batch_token_budget {
            self.batch_token_budget = Some(value);
        }
        if let Some(value) = other.translation_concurrency {
            self.translation_concurrency = Some(value);
        }
        if let Some(value) = other.review_concurrency {
            self.review_concurrency = Some(value);
        }
        if let Some(value) = other.bilingual {
            self.bilingual = Some(value);
        }
        if let Some(value) = other.fast_mode {
            self.fast_mode = Some(value);
        }
        if let Some(value) = other.review_policy {
            self.review_policy = Some(value);
        }
        if let Some(value) = other.terminology_preflight {
            self.terminology_preflight = Some(value);
        }
        if let Some(value) = other.dry_run {
            self.dry_run = Some(value);
        }
        if let Some(value) = other.resume {
            self.resume = Some(value);
        }
        if let Some(value) = other.use_cache {
            self.use_cache = Some(value);
        }
        if let Some(value) = other.retries {
            self.retries = Some(value);
        }
        if let Some(value) = other.agent {
            self.agent = Some(value);
        }
        if let Some(value) = other.agent_repair_attempts {
            self.agent_repair_attempts = Some(value);
        }
        if let Some(value) = other.runtime_dir {
            self.runtime_dir = Some(value);
        }
        if let Some(value) = other.glossary_path {
            self.glossary_path = Some(value);
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct TranslationSettingsPatch {
    pub output_format: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub api_format: Option<ApiFormat>,
    pub endpoint_url: Option<String>,
    pub api_key_env: Option<String>,
    pub auth_header: Option<String>,
    pub auth_prefix: Option<String>,
    pub source_language: Option<String>,
    pub target_language: Option<String>,
    pub batch_size: Option<usize>,
    pub batch_token_budget: Option<usize>,
    pub translation_concurrency: Option<usize>,
    pub review_concurrency: Option<usize>,
    pub bilingual: Option<bool>,
    pub fast_mode: Option<bool>,
    pub review_policy: Option<ReviewPolicy>,
    pub terminology_preflight: Option<bool>,
    pub dry_run: Option<bool>,
    pub resume: Option<bool>,
    pub use_cache: Option<bool>,
    pub retries: Option<usize>,
    pub agent: Option<bool>,
    pub agent_repair_attempts: Option<usize>,
    pub runtime_dir: Option<PathBuf>,
    pub glossary_path: Option<PathBuf>,
}

impl Default for TranslationSettings {
    fn default() -> Self {
        Self {
            output: OutputSettings {
                format: None,
                bilingual: false,
            },
            backend: BackendSettings {
                provider: DEFAULT_PROVIDER.to_owned(),
                model: DEFAULT_MODEL.to_owned(),
                api_key: None,
                base_url: None,
                api_format: None,
                endpoint_url: None,
                api_key_env: None,
                auth_header: None,
                auth_prefix: None,
            },
            translation: TranslationDomainSettings {
                source_language: DEFAULT_SOURCE_LANGUAGE.to_owned(),
                target_language: DEFAULT_TARGET_LANGUAGE.to_owned(),
                batch_size: DEFAULT_BATCH_SIZE,
                batch_token_budget: DEFAULT_BATCH_TOKEN_BUDGET,
                translation_concurrency: DEFAULT_TRANSLATION_CONCURRENCY,
                review_concurrency: DEFAULT_REVIEW_CONCURRENCY,
                fast_mode: false,
                review_policy: ReviewPolicy::Off,
                terminology_preflight: true,
                dry_run: false,
                resume: true,
                use_cache: true,
                retries: DEFAULT_RETRIES,
                agent: true,
                agent_repair_attempts: DEFAULT_AGENT_REPAIR_ATTEMPTS,
            },
            runtime: RuntimeSettings {
                runtime_dir: None,
                glossary_path: None,
            },
        }
    }
}

impl TranslationSettings {
    pub fn with_patch(mut self, patch: TranslationSettingsPatch) -> Self {
        self.apply_patch(patch);
        self
    }

    pub fn apply_patch(&mut self, patch: TranslationSettingsPatch) {
        if let Some(value) = patch.output_format {
            self.output.format = Some(value);
        }
        if let Some(value) = patch.provider {
            self.backend.provider = value;
        }
        if let Some(value) = patch.model {
            self.backend.model = value;
        }
        if let Some(value) = patch.api_key {
            self.backend.api_key = Some(value);
        }
        if let Some(value) = patch.base_url {
            self.backend.base_url = Some(value);
        }
        if let Some(value) = patch.api_format {
            self.backend.api_format = Some(value);
        }
        if let Some(value) = patch.endpoint_url {
            self.backend.endpoint_url = Some(value);
        }
        if let Some(value) = patch.api_key_env {
            self.backend.api_key_env = Some(value);
        }
        if let Some(value) = patch.auth_header {
            self.backend.auth_header = Some(value);
        }
        if let Some(value) = patch.auth_prefix {
            self.backend.auth_prefix = Some(value);
        }
        if let Some(value) = patch.source_language {
            self.translation.source_language = value;
        }
        if let Some(value) = patch.target_language {
            self.translation.target_language = value;
        }
        if let Some(value) = patch.batch_size {
            self.translation.batch_size = value;
        }
        if let Some(value) = patch.batch_token_budget {
            self.translation.batch_token_budget = value;
        }
        if let Some(value) = patch.translation_concurrency {
            self.translation.translation_concurrency = value;
        }
        if let Some(value) = patch.review_concurrency {
            self.translation.review_concurrency = value;
        }
        if let Some(value) = patch.bilingual {
            self.output.bilingual = value;
        }
        if let Some(value) = patch.fast_mode {
            self.translation.fast_mode = value;
        }
        if let Some(value) = patch.review_policy {
            self.translation.review_policy = value;
        }
        if let Some(value) = patch.terminology_preflight {
            self.translation.terminology_preflight = value;
        }
        if let Some(value) = patch.dry_run {
            self.translation.dry_run = value;
        }
        if let Some(value) = patch.resume {
            self.translation.resume = value;
        }
        if let Some(value) = patch.use_cache {
            self.translation.use_cache = value;
        }
        if let Some(value) = patch.retries {
            self.translation.retries = value;
        }
        if let Some(value) = patch.agent {
            self.translation.agent = value;
        }
        if let Some(value) = patch.agent_repair_attempts {
            self.translation.agent_repair_attempts = value;
        }
        if let Some(value) = patch.runtime_dir {
            self.runtime.runtime_dir = Some(value);
        }
        if let Some(value) = patch.glossary_path {
            self.runtime.glossary_path = Some(value);
        }
    }

    pub fn backend_config(&self) -> BackendConfig {
        BackendConfig {
            id: self.backend.provider.clone(),
            provider: self.backend.provider.clone(),
            display_name: self.backend.provider.clone(),
            api_format: self
                .backend
                .api_format
                .or_else(|| legacy_api_format(&self.backend.provider)),
            model: self.backend.model.clone(),
            api_key: self.backend.api_key.clone(),
            api_key_env: self.backend.api_key_env.clone(),
            base_url: self.backend.base_url.clone(),
            endpoint_url: self.backend.endpoint_url.clone(),
            auth_header: self.backend.auth_header.clone(),
            auth_prefix: self.backend.auth_prefix.clone(),
        }
    }

    pub fn to_pipeline_options(
        &self,
        input_path: impl Into<PathBuf>,
        output_path: Option<PathBuf>,
    ) -> PipelineOptions {
        let mut options = PipelineOptions::new(input_path.into());
        options.output_path = output_path;
        options.output_format = self.output.format.clone();
        options.provider = self.backend.provider.clone();
        options.model = self.backend.model.clone();
        options.provider_fingerprint = self.provider_fingerprint();
        options.source_language = self.translation.source_language.clone();
        options.target_language = self.translation.target_language.clone();
        options.batch_size = self.translation.batch_size;
        options.batch_token_budget = self.translation.batch_token_budget;
        options.translation_concurrency = self.translation.translation_concurrency;
        options.review_concurrency = self.translation.review_concurrency;
        options.bilingual = self.output.bilingual;
        options.fast_mode = self.translation.fast_mode;
        options.review_policy = self.translation.review_policy;
        options.terminology_preflight = self.translation.terminology_preflight;
        options.dry_run = self.translation.dry_run;
        options.resume = self.translation.resume;
        options.use_cache = self.translation.use_cache;
        options.retries = self.translation.retries;
        options.agent = self.translation.agent;
        options.agent_repair_attempts = self.translation.agent_repair_attempts;
        options.runtime_dir = self.runtime.runtime_dir.clone();
        options.glossary_path = self.runtime.glossary_path.clone();
        options
    }

    pub fn output_format(&self) -> Option<&str> {
        self.output.format.as_deref()
    }

    pub fn runtime_dir(&self) -> Option<&Path> {
        self.runtime.runtime_dir.as_deref()
    }

    pub fn glossary_path(&self) -> Option<&Path> {
        self.runtime.glossary_path.as_deref()
    }

    fn provider_fingerprint(&self) -> Option<String> {
        if self.backend.provider.eq_ignore_ascii_case("mock") {
            return None;
        }
        let config = self.backend_config();
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_current_cli_behavior() {
        let settings = TranslationSettings::default();

        assert_eq!(settings.backend.provider, "mock");
        assert_eq!(settings.backend.model, "mock-zh");
        assert_eq!(settings.translation.source_language, "Auto");
        assert_eq!(settings.translation.target_language, "zh-Hans");
        assert_eq!(settings.translation.batch_size, DEFAULT_BATCH_SIZE);
        assert_eq!(settings.translation.review_policy, ReviewPolicy::Off);
        assert!(settings.translation.resume);
        assert!(settings.translation.use_cache);
        assert_eq!(settings.translation.retries, 2);
        assert!(settings.translation.agent);
        assert_eq!(settings.translation.agent_repair_attempts, 2);
    }

    #[test]
    fn builds_pipeline_options_from_settings() {
        let mut settings = TranslationSettings::default();
        settings.translation.target_language = "English".to_owned();
        settings.output.bilingual = true;
        settings.output.format = Some("txt".to_owned());

        let options = settings.to_pipeline_options("clip.srt", Some("out.txt".into()));

        assert_eq!(options.target_language, "English");
        assert!(options.bilingual);
        assert_eq!(options.output_format.as_deref(), Some("txt"));
        assert!(options.resume);
        assert!(options.use_cache);
        assert_eq!(options.retries, 2);
        assert!(options.agent);
        assert_eq!(options.agent_repair_attempts, 2);
    }

    #[test]
    fn applies_patch_over_defaults() {
        let settings = TranslationSettings::default().with_patch(TranslationSettingsPatch {
            provider: Some("openai".to_owned()),
            batch_size: Some(12),
            review_policy: Some(ReviewPolicy::Off),
            ..TranslationSettingsPatch::default()
        });

        assert_eq!(settings.backend.provider, "openai");
        assert_eq!(settings.translation.batch_size, 12);
        assert_eq!(settings.translation.review_policy, ReviewPolicy::Off);
    }

    #[test]
    fn builds_backend_config_with_api_key_sources() {
        let mut settings = TranslationSettings::default();
        settings.backend.provider = "openai".to_owned();
        settings.backend.model = "gpt".to_owned();
        settings.backend.api_key = Some("direct-key".to_owned());
        settings.backend.base_url = Some("https://example.test/v1".to_owned());

        let config = settings.backend_config();

        assert_eq!(config.api_key.as_deref(), Some("direct-key"));
        assert_eq!(config.base_url.as_deref(), Some("https://example.test/v1"));
    }
}
