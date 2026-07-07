use std::path::{Path, PathBuf};

use subbake_core::entities::{DEFAULT_BATCH_SIZE, PipelineOptions};

use crate::providers::BackendConfig;

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationSettings {
    pub output_format: Option<String>,
    pub provider: String,
    pub model: String,
    pub source_language: String,
    pub target_language: String,
    pub batch_size: usize,
    pub bilingual: bool,
    pub fast_mode: bool,
    pub final_review: bool,
    pub dry_run: bool,
    pub runtime_dir: Option<PathBuf>,
    pub glossary_path: Option<PathBuf>,
}

impl Default for TranslationSettings {
    fn default() -> Self {
        Self {
            output_format: None,
            provider: "mock".to_owned(),
            model: "mock-zh".to_owned(),
            source_language: "Auto".to_owned(),
            target_language: "Chinese".to_owned(),
            batch_size: DEFAULT_BATCH_SIZE,
            bilingual: false,
            fast_mode: false,
            final_review: true,
            dry_run: false,
            runtime_dir: None,
            glossary_path: None,
        }
    }
}

impl TranslationSettings {
    pub fn backend_config(&self) -> BackendConfig {
        BackendConfig::new(self.provider.clone(), self.model.clone())
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
        options.source_language = self.source_language.clone();
        options.target_language = self.target_language.clone();
        options.batch_size = self.batch_size;
        options.bilingual = self.bilingual;
        options.fast_mode = self.fast_mode;
        options.final_review = self.final_review;
        options.dry_run = self.dry_run;
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
    }
}
