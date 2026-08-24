use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use subbake_core::entities::{
    BilingualOrder, DEFAULT_AGENT_REPAIR_ATTEMPTS, DEFAULT_MODEL, DEFAULT_PROVIDER,
    DEFAULT_RETRIES, DEFAULT_SOURCE_LANGUAGE, DEFAULT_TARGET_LANGUAGE, PipelineOptions,
    ReviewPolicy, TranslationMode,
};

use crate::error::{AdapterError, AdapterResult};
use crate::providers::{ApiFormat, BackendConfig};
use crate::whisper::DEFAULT_WHISPER_VAD_MODEL;

pub const DEFAULT_VAD_THRESHOLD: f32 = 0.5;
pub const DEFAULT_VAD_MIN_SPEECH_DURATION_MS: u64 = 250;
pub const DEFAULT_VAD_MIN_SILENCE_DURATION_MS: u64 = 100;
pub const DEFAULT_VAD_SPEECH_PAD_MS: u64 = 30;

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedSettings {
    pub output: OutputSettings,
    pub backend: BackendSettings,
    pub reviewer_backend: Option<BackendSettings>,
    pub agent: AgentDomainSettings,
    pub translation: TranslationDomainSettings,
    pub transcription: TranscriptionDomainSettings,
    pub storage: StorageSettings,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgentDomainSettings {
    pub max_steps: usize,
    pub auto_approve_commands: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionDomainSettings {
    pub model: Option<String>,
    pub normalize_text: bool,
    pub speaker_labels: bool,
    pub vad_enabled: bool,
    pub vad_model: String,
    pub vad_threshold: f32,
    pub vad_min_speech_duration_ms: u64,
    pub vad_min_silence_duration_ms: u64,
    pub vad_speech_pad_ms: u64,
}

/// Compatibility alias for service request types. New configuration code
/// should name the complete resolved value `ResolvedSettings`.
pub type TranslationSettings = ResolvedSettings;

#[derive(Debug, Clone, PartialEq)]
pub struct OutputSettings {
    pub format: Option<String>,
    pub bilingual: bool,
    pub bilingual_order: BilingualOrder,
    pub bilingual_font_scale: f64,
    pub preserve_source_container: bool,
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
    pub timeout_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationDomainSettings {
    pub source_language: String,
    pub target_language: String,
    /// Explicit ffprobe stream index for embedded subtitle containers.
    pub subtitle_stream_index: Option<usize>,
    pub batch_size: usize,
    pub batch_token_budget: usize,
    pub request_token_budget: usize,
    pub confirmed_context_lines: usize,
    pub confirmed_context_token_budget: usize,
    pub translation_concurrency: usize,
    pub review_concurrency: usize,
    pub mode: TranslationMode,
    pub review_policy: ReviewPolicy,
    pub terminology_preflight: bool,
    pub online_terminology: bool,
    pub allow_degraded_preflight: bool,
    pub preserve_names: bool,
    pub max_characters_per_second: Option<f64>,
    pub max_characters_per_line: Option<usize>,
    pub max_lines: Option<usize>,
    pub dry_run: bool,
    pub resume: bool,
    pub use_cache: bool,
    pub retries: usize,
    pub agent: bool,
    pub agent_repair_attempts: usize,
    pub max_requests: Option<usize>,
    pub max_tokens: Option<usize>,
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
    pub agent: AgentOverrides,
    pub translation: TranslationOverrides,
    pub transcription: TranscriptionOverrides,
    pub output: OutputOverrides,
    pub storage: StorageOverrides,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentOverrides {
    pub max_steps: Option<usize>,
    pub auto_approve_commands: Option<bool>,
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
    pub timeout_seconds: Option<f64>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TranslationOverrides {
    pub source_language: Option<String>,
    pub target_language: Option<String>,
    pub subtitle_stream_index: Option<usize>,
    pub batch_size: Option<usize>,
    pub batch_token_budget: Option<usize>,
    pub request_token_budget: Option<usize>,
    pub confirmed_context_lines: Option<usize>,
    pub confirmed_context_token_budget: Option<usize>,
    pub translation_concurrency: Option<usize>,
    pub review_concurrency: Option<usize>,
    pub mode: Option<TranslationMode>,
    /// Legacy v1 input. `true` maps to turbo; `false` keeps the selected mode.
    pub fast_mode: Option<bool>,
    pub review_policy: Option<ReviewPolicy>,
    pub terminology_preflight: Option<bool>,
    pub online_terminology: Option<bool>,
    pub allow_degraded_preflight: Option<bool>,
    pub preserve_names: Option<bool>,
    pub max_characters_per_second: Option<f64>,
    pub max_characters_per_line: Option<usize>,
    pub max_lines: Option<usize>,
    pub dry_run: Option<bool>,
    pub resume: Option<bool>,
    pub use_cache: Option<bool>,
    pub retries: Option<usize>,
    pub agent: Option<bool>,
    pub agent_repair_attempts: Option<usize>,
    pub max_requests: Option<usize>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TranscriptionOverrides {
    pub model: Option<String>,
    pub normalize_text: Option<bool>,
    pub speaker_labels: Option<bool>,
    pub vad_enabled: Option<bool>,
    pub vad_model: Option<String>,
    pub vad_threshold: Option<f32>,
    pub vad_min_speech_duration_ms: Option<u64>,
    pub vad_min_silence_duration_ms: Option<u64>,
    pub vad_speech_pad_ms: Option<u64>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputOverrides {
    pub format: Option<String>,
    pub bilingual: Option<bool>,
    pub bilingual_order: Option<BilingualOrder>,
    pub bilingual_font_scale: Option<f64>,
    pub preserve_source_container: Option<bool>,
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
        self.agent.merge(other.agent);
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
                timeout_seconds: Some(settings.backend.timeout_seconds),
            },
            reviewer_backend: settings.reviewer_backend.as_ref().map(backend_overrides),
            agent: AgentOverrides {
                max_steps: Some(settings.agent.max_steps),
                auto_approve_commands: Some(settings.agent.auto_approve_commands),
            },
            translation: TranslationOverrides {
                source_language: Some(settings.translation.source_language.clone()),
                target_language: Some(settings.translation.target_language.clone()),
                subtitle_stream_index: settings.translation.subtitle_stream_index,
                batch_size: Some(settings.translation.batch_size),
                batch_token_budget: Some(settings.translation.batch_token_budget),
                request_token_budget: Some(settings.translation.request_token_budget),
                confirmed_context_lines: Some(settings.translation.confirmed_context_lines),
                confirmed_context_token_budget: Some(
                    settings.translation.confirmed_context_token_budget,
                ),
                translation_concurrency: Some(settings.translation.translation_concurrency),
                review_concurrency: Some(settings.translation.review_concurrency),
                mode: Some(settings.translation.mode),
                fast_mode: None,
                review_policy: Some(settings.translation.review_policy),
                terminology_preflight: Some(settings.translation.terminology_preflight),
                online_terminology: Some(settings.translation.online_terminology),
                allow_degraded_preflight: Some(settings.translation.allow_degraded_preflight),
                preserve_names: Some(settings.translation.preserve_names),
                max_characters_per_second: settings.translation.max_characters_per_second,
                max_characters_per_line: settings.translation.max_characters_per_line,
                max_lines: settings.translation.max_lines,
                dry_run: Some(settings.translation.dry_run),
                resume: Some(settings.translation.resume),
                use_cache: Some(settings.translation.use_cache),
                retries: Some(settings.translation.retries),
                agent: Some(settings.translation.agent),
                agent_repair_attempts: Some(settings.translation.agent_repair_attempts),
                max_requests: settings.translation.max_requests,
                max_tokens: settings.translation.max_tokens,
            },
            transcription: TranscriptionOverrides {
                model: settings.transcription.model.clone(),
                normalize_text: Some(settings.transcription.normalize_text),
                speaker_labels: Some(settings.transcription.speaker_labels),
                vad_enabled: Some(settings.transcription.vad_enabled),
                vad_model: Some(settings.transcription.vad_model.clone()),
                vad_threshold: Some(settings.transcription.vad_threshold),
                vad_min_speech_duration_ms: Some(settings.transcription.vad_min_speech_duration_ms),
                vad_min_silence_duration_ms: Some(
                    settings.transcription.vad_min_silence_duration_ms,
                ),
                vad_speech_pad_ms: Some(settings.transcription.vad_speech_pad_ms),
            },
            output: OutputOverrides {
                format: settings.output.format.clone(),
                bilingual: Some(settings.output.bilingual),
                bilingual_order: Some(settings.output.bilingual_order),
                bilingual_font_scale: Some(settings.output.bilingual_font_scale),
                preserve_source_container: Some(settings.output.preserve_source_container),
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
            auth_prefix,
            timeout_seconds
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
            subtitle_stream_index,
            batch_size,
            batch_token_budget,
            request_token_budget,
            confirmed_context_lines,
            confirmed_context_token_budget,
            translation_concurrency,
            review_concurrency,
            mode,
            fast_mode,
            review_policy,
            terminology_preflight,
            online_terminology,
            allow_degraded_preflight,
            preserve_names,
            max_characters_per_second,
            max_characters_per_line,
            max_lines,
            dry_run,
            resume,
            use_cache,
            retries,
            agent,
            agent_repair_attempts,
            max_requests,
            max_tokens
        );
    }
}

impl AgentOverrides {
    fn merge(&mut self, other: Self) {
        merge_optional_fields!(self, other, max_steps, auto_approve_commands);
    }
}

impl TranscriptionOverrides {
    fn merge(&mut self, other: Self) {
        merge_optional_fields!(
            self,
            other,
            model,
            normalize_text,
            speaker_labels,
            vad_enabled,
            vad_model,
            vad_threshold,
            vad_min_speech_duration_ms,
            vad_min_silence_duration_ms,
            vad_speech_pad_ms
        );
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
        timeout_seconds: Some(settings.timeout_seconds),
    }
}

impl OutputOverrides {
    fn merge(&mut self, other: Self) {
        merge_optional_fields!(
            self,
            other,
            format,
            bilingual,
            bilingual_order,
            bilingual_font_scale,
            preserve_source_container
        );
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
        let policy = subbake_core::TranslationPolicy::for_mode(TranslationMode::Turbo);
        Self {
            output: OutputSettings {
                format: None,
                bilingual: false,
                bilingual_order: BilingualOrder::default(),
                bilingual_font_scale: 1.0,
                preserve_source_container: false,
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
                timeout_seconds: crate::llm_backends::default_timeout_seconds(),
            },
            reviewer_backend: None,
            agent: AgentDomainSettings {
                max_steps: 64,
                auto_approve_commands: false,
            },
            translation: TranslationDomainSettings {
                source_language: DEFAULT_SOURCE_LANGUAGE.to_owned(),
                target_language: DEFAULT_TARGET_LANGUAGE.to_owned(),
                subtitle_stream_index: None,
                batch_size: policy.batch_size,
                batch_token_budget: policy.batch_token_budget,
                request_token_budget: policy.request_token_budget,
                confirmed_context_lines: policy.confirmed_context_lines,
                confirmed_context_token_budget: policy.confirmed_context_token_budget,
                translation_concurrency: policy.translation_concurrency,
                review_concurrency: policy.review_concurrency,
                mode: TranslationMode::Turbo,
                review_policy: policy.review_policy,
                terminology_preflight: policy.terminology_strategy.preflight_default(),
                online_terminology: policy.terminology_strategy.online_default(),
                allow_degraded_preflight: policy.preflight_failure_policy.allows_degraded(),
                preserve_names: false,
                max_characters_per_second: None,
                max_characters_per_line: None,
                max_lines: None,
                dry_run: false,
                resume: true,
                use_cache: true,
                retries: DEFAULT_RETRIES,
                agent: true,
                agent_repair_attempts: DEFAULT_AGENT_REPAIR_ATTEMPTS,
                max_requests: None,
                max_tokens: None,
            },
            transcription: TranscriptionDomainSettings {
                model: None,
                normalize_text: true,
                speaker_labels: true,
                vad_enabled: true,
                vad_model: DEFAULT_WHISPER_VAD_MODEL.to_owned(),
                vad_threshold: DEFAULT_VAD_THRESHOLD,
                vad_min_speech_duration_ms: DEFAULT_VAD_MIN_SPEECH_DURATION_MS,
                vad_min_silence_duration_ms: DEFAULT_VAD_MIN_SILENCE_DURATION_MS,
                vad_speech_pad_ms: DEFAULT_VAD_SPEECH_PAD_MS,
            },
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
            timeout_seconds,
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
        if let Some(value) = timeout_seconds {
            self.backend.timeout_seconds = value;
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
            subtitle_stream_index,
            batch_size,
            batch_token_budget,
            request_token_budget,
            confirmed_context_lines,
            confirmed_context_token_budget,
            translation_concurrency,
            review_concurrency,
            mode,
            fast_mode,
            review_policy,
            terminology_preflight,
            online_terminology,
            allow_degraded_preflight,
            preserve_names,
            max_characters_per_second,
            max_characters_per_line,
            max_lines,
            dry_run,
            resume,
            use_cache,
            retries,
            agent,
            agent_repair_attempts,
            max_requests,
            max_tokens,
        } = overrides.translation;
        if let Some(model) = overrides.transcription.model {
            self.transcription.model = Some(model);
        }
        if let Some(value) = overrides.transcription.normalize_text {
            self.transcription.normalize_text = value;
        }
        if let Some(value) = overrides.transcription.speaker_labels {
            self.transcription.speaker_labels = value;
        }
        if let Some(value) = overrides.transcription.vad_enabled {
            self.transcription.vad_enabled = value;
        }
        if let Some(value) = overrides.transcription.vad_model {
            self.transcription.vad_model = value;
        }
        if let Some(value) = overrides.transcription.vad_threshold {
            self.transcription.vad_threshold = value;
        }
        if let Some(value) = overrides.transcription.vad_min_speech_duration_ms {
            self.transcription.vad_min_speech_duration_ms = value;
        }
        if let Some(value) = overrides.transcription.vad_min_silence_duration_ms {
            self.transcription.vad_min_silence_duration_ms = value;
        }
        if let Some(value) = overrides.transcription.vad_speech_pad_ms {
            self.transcription.vad_speech_pad_ms = value;
        }
        let AgentOverrides {
            max_steps,
            auto_approve_commands,
        } = overrides.agent;
        if let Some(value) = max_steps {
            self.agent.max_steps = value;
        }
        if let Some(value) = auto_approve_commands {
            self.agent.auto_approve_commands = value;
        }
        let selected_mode = mode;
        if let Some(value) = selected_mode {
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
        if selected_mode.is_some() {
            self.apply_readability_defaults();
        }
        if let Some(value) = subtitle_stream_index {
            self.translation.subtitle_stream_index = Some(value);
        }
        if let Some(value) = batch_size {
            self.translation.batch_size = value;
        }
        if let Some(value) = batch_token_budget {
            self.translation.batch_token_budget = value;
        }
        if let Some(value) = request_token_budget {
            self.translation.request_token_budget = value;
        }
        if let Some(value) = confirmed_context_lines {
            self.translation.confirmed_context_lines = value;
        }
        if let Some(value) = confirmed_context_token_budget {
            self.translation.confirmed_context_token_budget = value;
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
        if let Some(value) = online_terminology {
            self.translation.online_terminology = value;
        }
        if let Some(value) = allow_degraded_preflight {
            self.translation.allow_degraded_preflight = value;
        }
        if let Some(value) = preserve_names {
            self.translation.preserve_names = value;
        }
        if let Some(value) = max_characters_per_second {
            self.translation.max_characters_per_second = Some(value);
        }
        if let Some(value) = max_characters_per_line {
            self.translation.max_characters_per_line = Some(value);
        }
        if let Some(value) = max_lines {
            self.translation.max_lines = Some(value);
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
        if let Some(value) = max_requests {
            self.translation.max_requests = Some(value);
        }
        if let Some(value) = max_tokens {
            self.translation.max_tokens = Some(value);
        }

        let OutputOverrides {
            format,
            bilingual,
            bilingual_order,
            bilingual_font_scale,
            preserve_source_container,
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
        if let Some(value) = bilingual_font_scale {
            self.output.bilingual_font_scale = value;
        }
        if let Some(value) = preserve_source_container {
            self.output.preserve_source_container = value;
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
        for (name, value) in [
            ("translation.max_requests", self.translation.max_requests),
            ("translation.max_tokens", self.translation.max_tokens),
            (
                "translation.max_characters_per_line",
                self.translation.max_characters_per_line,
            ),
            ("translation.max_lines", self.translation.max_lines),
        ] {
            if value == Some(0) {
                return Err(AdapterError::invalid_input(format!(
                    "configuration field `{name}` must be greater than zero when set"
                )));
            }
        }
        if self
            .translation
            .max_characters_per_second
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(AdapterError::invalid_input(
                "configuration field `translation.max_characters_per_second` must be a finite number greater than zero when set",
            ));
        }
        if !(1..=128).contains(&self.agent.max_steps) {
            return Err(AdapterError::invalid_input(
                "configuration field `agent.max_steps` must be from 1 through 128",
            ));
        }
        if !self.output.bilingual_font_scale.is_finite()
            || !(0.1..=2.0).contains(&self.output.bilingual_font_scale)
        {
            return Err(AdapterError::invalid_input(
                "configuration field `output.bilingual_font_scale` must be from 0.1 through 2.0",
            ));
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
        if self.transcription.vad_enabled && self.transcription.vad_model.trim().is_empty() {
            return Err(AdapterError::invalid_input(
                "configuration field `transcription.vad_model` must not be empty when VAD is enabled",
            ));
        }
        if !self.transcription.vad_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.transcription.vad_threshold)
        {
            return Err(AdapterError::invalid_input(
                "configuration field `transcription.vad_threshold` must be from 0 through 1",
            ));
        }
        for (name, value) in [
            (
                "transcription.vad_min_speech_duration_ms",
                self.transcription.vad_min_speech_duration_ms,
            ),
            (
                "transcription.vad_min_silence_duration_ms",
                self.transcription.vad_min_silence_duration_ms,
            ),
        ] {
            if value == 0 {
                return Err(AdapterError::invalid_input(format!(
                    "configuration field `{name}` must be greater than zero"
                )));
            }
        }
        for (name, value) in [
            ("translation.batch_size", self.translation.batch_size),
            (
                "translation.batch_token_budget",
                self.translation.batch_token_budget,
            ),
            (
                "translation.request_token_budget",
                self.translation.request_token_budget,
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
            timeout_seconds: self.backend.timeout_seconds,
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
        options.rendering.output_path = output_path;
        options.rendering.output_format = self.output.format.clone();
        options.identity.provider = self.backend.id.clone();
        options.identity.model = self.backend.model.clone();
        options.identity.provider_fingerprint = self.provider_fingerprint();
        options.identity.reviewer_fingerprint =
            self.reviewer_backend.as_ref().and_then(backend_fingerprint);
        options.validation.source_language = self.translation.source_language.clone();
        options.validation.target_language = self.translation.target_language.clone();
        options.execution.batch_size = self.translation.batch_size;
        options.execution.batch_token_budget = self.translation.batch_token_budget;
        options.execution.request_token_budget = self.translation.request_token_budget;
        options.execution.confirmed_context_lines = self.translation.confirmed_context_lines;
        options.execution.confirmed_context_token_budget =
            self.translation.confirmed_context_token_budget;
        options.execution.translation_concurrency = self.translation.translation_concurrency;
        options.execution.review_concurrency = self.translation.review_concurrency;
        options.rendering.bilingual = self.output.bilingual;
        options.rendering.bilingual_order = self.output.bilingual_order;
        options.rendering.bilingual_font_scale = self.output.bilingual_font_scale;
        options.execution.mode = self.translation.mode;
        options.execution.review_policy = self.translation.review_policy;
        options.execution.terminology_preflight = self.translation.terminology_preflight;
        options.execution.online_terminology = self.translation.online_terminology;
        options.execution.allow_degraded_preflight = self.translation.allow_degraded_preflight;
        options.validation.preserve_names = self.translation.preserve_names;
        options.validation.max_characters_per_second = self.translation.max_characters_per_second;
        options.validation.max_characters_per_line = self.translation.max_characters_per_line;
        options.validation.max_lines = self.translation.max_lines;
        options.execution.dry_run = self.translation.dry_run;
        options.execution.resume = self.translation.resume;
        options.execution.use_cache = self.translation.use_cache;
        options.execution.retries = self.translation.retries;
        options.execution.agent = self.translation.agent;
        options.execution.agent_repair_attempts = self.translation.agent_repair_attempts;
        options.execution.max_requests = self.translation.max_requests;
        options.execution.max_tokens = self.translation.max_tokens;
        options.identity.runtime_dir = self.storage.runtime_dir.clone();
        options.identity.glossary_path = self.storage.glossary_path.clone();
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
        let policy = subbake_core::TranslationPolicy::for_mode(mode);
        self.translation.mode = mode;
        self.translation.batch_size = policy.batch_size;
        self.translation.batch_token_budget = policy.batch_token_budget;
        self.translation.request_token_budget = policy.request_token_budget;
        self.translation.confirmed_context_lines = policy.confirmed_context_lines;
        self.translation.confirmed_context_token_budget = policy.confirmed_context_token_budget;
        self.translation.translation_concurrency = policy.translation_concurrency;
        self.translation.review_concurrency = policy.review_concurrency;
        self.translation.review_policy = policy.review_policy;
        self.translation.terminology_preflight = policy.terminology_strategy.preflight_default();
        self.translation.online_terminology = policy.terminology_strategy.online_default();
        self.translation.allow_degraded_preflight =
            policy.preflight_failure_policy.allows_degraded();
        self.apply_readability_defaults();
    }

    fn apply_readability_defaults(&mut self) {
        let defaults = subbake_core::translation_readability_defaults(
            self.translation.mode,
            &self.translation.target_language,
        );
        self.translation.max_characters_per_second = defaults.max_characters_per_second;
        self.translation.max_characters_per_line = defaults.max_characters_per_line;
        self.translation.max_lines = defaults.max_lines;
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
    if let Some(value) = overrides.timeout_seconds {
        settings.timeout_seconds = value;
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
        timeout_seconds: settings.timeout_seconds,
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
                    bilingual_font_scale: Some(0.9),
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
        assert_eq!(settings.agent.max_steps, 64);
        assert!(!settings.agent.auto_approve_commands);
        assert!(settings.output.bilingual);
        assert_eq!(settings.output.bilingual_font_scale, 0.9);
        assert_eq!(settings.storage.runtime_dir, Some(".runtime".into()));
        assert_eq!(settings.backend.timeout_seconds, 120.0);
    }

    #[test]
    fn backend_timeout_is_overridable_and_validated() {
        let settings = ResolvedSettings::default()
            .with_overrides(SettingsOverrides {
                backend: BackendOverrides {
                    timeout_seconds: Some(300.0),
                    ..BackendOverrides::default()
                },
                ..SettingsOverrides::default()
            })
            .expect("valid timeout");
        assert_eq!(settings.backend.timeout_seconds, 300.0);
        assert_eq!(settings.backend_config().timeout_seconds, 300.0);

        let error = ResolvedSettings::default()
            .with_overrides(SettingsOverrides {
                backend: BackendOverrides {
                    timeout_seconds: Some(0.5),
                    ..BackendOverrides::default()
                },
                ..SettingsOverrides::default()
            })
            .expect_err("sub-second timeout must fail");
        assert!(error.to_string().contains("timeout_seconds"));
    }

    #[test]
    fn cinema_mode_applies_defaults_before_explicit_overrides() {
        let settings = ResolvedSettings::default()
            .with_overrides(SettingsOverrides {
                translation: TranslationOverrides {
                    mode: Some(TranslationMode::Cinema),
                    translation_concurrency: Some(7),
                    target_language: Some("zh-Hans".to_owned()),
                    max_characters_per_line: Some(26),
                    ..TranslationOverrides::default()
                },
                ..SettingsOverrides::default()
            })
            .expect("valid settings");
        assert_eq!(settings.translation.mode, TranslationMode::Cinema);
        assert_eq!(settings.translation.batch_size, 48);
        assert_eq!(settings.translation.translation_concurrency, 7);
        assert_eq!(settings.translation.review_policy, ReviewPolicy::Full);
        assert!(settings.translation.online_terminology);
        assert!(!settings.translation.allow_degraded_preflight);
        assert_eq!(settings.translation.max_characters_per_second, Some(23.0));
        assert_eq!(settings.translation.max_characters_per_line, Some(26));
        assert_eq!(settings.translation.max_lines, Some(2));
    }

    #[test]
    fn turbo_disables_online_terminology_unless_explicitly_enabled() {
        let defaults = ResolvedSettings::default()
            .with_overrides(SettingsOverrides {
                translation: TranslationOverrides {
                    mode: Some(TranslationMode::Turbo),
                    ..TranslationOverrides::default()
                },
                ..SettingsOverrides::default()
            })
            .expect("turbo defaults");
        assert!(!defaults.translation.online_terminology);

        let enabled = defaults
            .with_overrides(SettingsOverrides {
                translation: TranslationOverrides {
                    online_terminology: Some(true),
                    ..TranslationOverrides::default()
                },
                ..SettingsOverrides::default()
            })
            .expect("explicit online terminology");
        assert!(enabled.translation.online_terminology);
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

        let error = ResolvedSettings::default()
            .with_overrides(SettingsOverrides {
                agent: AgentOverrides {
                    max_steps: Some(129),
                    ..AgentOverrides::default()
                },
                ..SettingsOverrides::default()
            })
            .expect_err("excessive agent max steps");
        assert!(error.to_string().contains("agent.max_steps"));
    }

    #[test]
    fn validation_rejects_invalid_bilingual_font_scale() {
        let mut settings = ResolvedSettings::default();
        settings.output.bilingual_font_scale = 0.0;

        let error = settings.validate().expect_err("invalid scale must fail");

        assert!(error.to_string().contains("bilingual_font_scale"));
    }

    #[test]
    fn resolved_defaults_match_the_turbo_policy() {
        let settings = ResolvedSettings::default();
        let policy = subbake_core::TranslationPolicy::for_mode(TranslationMode::Turbo);

        assert_eq!(settings.translation.mode, TranslationMode::Turbo);
        assert_eq!(settings.translation.batch_size, policy.batch_size);
        assert_eq!(
            settings.translation.batch_token_budget,
            policy.batch_token_budget
        );
        assert_eq!(
            settings.translation.request_token_budget,
            policy.request_token_budget
        );
        assert_eq!(
            settings.translation.confirmed_context_lines,
            policy.confirmed_context_lines
        );
        assert_eq!(
            settings.translation.confirmed_context_token_budget,
            policy.confirmed_context_token_budget
        );
        assert_eq!(
            settings.translation.translation_concurrency,
            policy.translation_concurrency
        );
        assert_eq!(
            settings.translation.review_concurrency,
            policy.review_concurrency
        );
        assert_eq!(settings.translation.review_policy, policy.review_policy);
        assert_eq!(
            settings.translation.terminology_preflight,
            policy.terminology_strategy.preflight_default()
        );
        assert_eq!(
            settings.translation.online_terminology,
            policy.terminology_strategy.online_default()
        );
        assert_eq!(
            settings.translation.allow_degraded_preflight,
            policy.preflight_failure_policy.allows_degraded()
        );
    }
}
