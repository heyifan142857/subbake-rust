use std::path::{Path, PathBuf};

use subbake_core::entities::{DEFAULT_BATCH_SIZE, PipelineOptions};

use crate::providers::{BackendConfig, default_api_key_env, resolve_env_var};

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationSettings {
    pub output_format: Option<String>,
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub source_language: String,
    pub target_language: String,
    pub batch_size: usize,
    pub bilingual: bool,
    pub fast_mode: bool,
    pub final_review: bool,
    pub dry_run: bool,
    pub resume: bool,
    pub use_cache: bool,
    pub retries: usize,
    pub agent: bool,
    pub agent_repair_attempts: usize,
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
        if let Some(value) = other.source_language {
            self.source_language = Some(value);
        }
        if let Some(value) = other.target_language {
            self.target_language = Some(value);
        }
        if let Some(value) = other.batch_size {
            self.batch_size = Some(value);
        }
        if let Some(value) = other.bilingual {
            self.bilingual = Some(value);
        }
        if let Some(value) = other.fast_mode {
            self.fast_mode = Some(value);
        }
        if let Some(value) = other.final_review {
            self.final_review = Some(value);
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
    pub source_language: Option<String>,
    pub target_language: Option<String>,
    pub batch_size: Option<usize>,
    pub bilingual: Option<bool>,
    pub fast_mode: Option<bool>,
    pub final_review: Option<bool>,
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
            output_format: None,
            provider: "mock".to_owned(),
            model: "mock-zh".to_owned(),
            api_key: None,
            base_url: None,
            source_language: "Auto".to_owned(),
            target_language: "Chinese".to_owned(),
            batch_size: DEFAULT_BATCH_SIZE,
            bilingual: false,
            fast_mode: false,
            final_review: true,
            dry_run: false,
            resume: true,
            use_cache: true,
            retries: 2,
            agent: true,
            agent_repair_attempts: 2,
            runtime_dir: None,
            glossary_path: None,
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
            self.output_format = Some(value);
        }
        if let Some(value) = patch.provider {
            self.provider = value;
        }
        if let Some(value) = patch.model {
            self.model = value;
        }
        if let Some(value) = patch.api_key {
            self.api_key = Some(value);
        }
        if let Some(value) = patch.base_url {
            self.base_url = Some(value);
        }
        if let Some(value) = patch.source_language {
            self.source_language = value;
        }
        if let Some(value) = patch.target_language {
            self.target_language = value;
        }
        if let Some(value) = patch.batch_size {
            self.batch_size = value;
        }
        if let Some(value) = patch.bilingual {
            self.bilingual = value;
        }
        if let Some(value) = patch.fast_mode {
            self.fast_mode = value;
        }
        if let Some(value) = patch.final_review {
            self.final_review = value;
        }
        if let Some(value) = patch.dry_run {
            self.dry_run = value;
        }
        if let Some(value) = patch.resume {
            self.resume = value;
        }
        if let Some(value) = patch.use_cache {
            self.use_cache = value;
        }
        if let Some(value) = patch.retries {
            self.retries = value;
        }
        if let Some(value) = patch.agent {
            self.agent = value;
        }
        if let Some(value) = patch.agent_repair_attempts {
            self.agent_repair_attempts = value;
        }
        if let Some(value) = patch.runtime_dir {
            self.runtime_dir = Some(value);
        }
        if let Some(value) = patch.glossary_path {
            self.glossary_path = Some(value);
        }
    }

    pub fn backend_config(&self) -> BackendConfig {
        BackendConfig {
            provider: self.provider.clone(),
            model: self.model.clone(),
            api_key: self.resolve_api_key(),
            base_url: self.base_url.clone(),
        }
    }

    /// Resolve the API key: use the already-resolved direct/env value, or fall back
    /// to the provider-default environment variable (e.g. OPENAI_API_KEY for "openai").
    fn resolve_api_key(&self) -> Option<String> {
        self.api_key
            .clone()
            .or_else(|| resolve_env_var(default_api_key_env(&self.provider)))
    }

    pub fn to_pipeline_options(
        &self,
        input_path: impl Into<PathBuf>,
        output_path: Option<PathBuf>,
    ) -> PipelineOptions {
        let mut options = PipelineOptions::new(input_path.into());
        options.output_path = output_path;
        options.output_format = self.output_format.clone();
        options.provider = self.provider.clone();
        options.model = self.model.clone();
        options.api_key = self.backend_config().api_key;
        options.base_url = self.base_url.clone();
        options.source_language = self.source_language.clone();
        options.target_language = self.target_language.clone();
        options.batch_size = self.batch_size;
        options.bilingual = self.bilingual;
        options.fast_mode = self.fast_mode;
        options.final_review = self.final_review;
        options.dry_run = self.dry_run;
        options.resume = self.resume;
        options.use_cache = self.use_cache;
        options.retries = self.retries;
        options.agent = self.agent;
        options.agent_repair_attempts = self.agent_repair_attempts;
        options.runtime_dir = self.runtime_dir.clone();
        options.glossary_path = self.glossary_path.clone();
        options
    }

    pub fn output_format(&self) -> Option<&str> {
        self.output_format.as_deref()
    }

    pub fn runtime_dir(&self) -> Option<&Path> {
        self.runtime_dir.as_deref()
    }

    pub fn glossary_path(&self) -> Option<&Path> {
        self.glossary_path.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_current_cli_behavior() {
        let settings = TranslationSettings::default();

        assert_eq!(settings.provider, "mock");
        assert_eq!(settings.model, "mock-zh");
        assert_eq!(settings.source_language, "Auto");
        assert_eq!(settings.target_language, "Chinese");
        assert_eq!(settings.batch_size, DEFAULT_BATCH_SIZE);
        assert!(settings.final_review);
        assert!(settings.resume);
        assert!(settings.use_cache);
        assert_eq!(settings.retries, 2);
        assert!(settings.agent);
        assert_eq!(settings.agent_repair_attempts, 2);
    }

    #[test]
    fn builds_pipeline_options_from_settings() {
        let mut settings = TranslationSettings {
            target_language: "English".to_owned(),
            bilingual: true,
            ..TranslationSettings::default()
        };
        settings.output_format = Some("txt".to_owned());

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
            final_review: Some(false),
            ..TranslationSettingsPatch::default()
        });

        assert_eq!(settings.provider, "openai");
        assert_eq!(settings.batch_size, 12);
        assert!(!settings.final_review);
    }

    #[test]
    fn builds_backend_config_with_api_key_sources() {
        let settings = TranslationSettings {
            provider: "openai".to_owned(),
            model: "gpt".to_owned(),
            api_key: Some("direct-key".to_owned()),
            base_url: Some("https://example.test/v1".to_owned()),
            ..TranslationSettings::default()
        };

        let config = settings.backend_config();

        assert_eq!(config.api_key.as_deref(), Some("direct-key"));
        assert_eq!(config.base_url.as_deref(), Some("https://example.test/v1"));
    }
}
