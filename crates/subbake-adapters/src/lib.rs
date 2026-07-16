pub mod config;
pub mod diagnostics;
pub mod editing;
pub mod error;
pub mod fs;
pub mod llm_backends;
pub mod mock;
pub mod pipeline;
pub mod providers;
pub mod runtime;
pub mod runtime_store;
pub mod settings;
pub mod transcription;
pub mod translation;
pub mod whisper;

pub use config::{
    CONFIG_VERSION, ConfigFile, ConfigurationResolver, ResolveRequest, ResolvedConfiguration,
    append_profile_snapshot, discover_config_path,
};
pub use diagnostics::{diagnose_failure_path, format_diagnostic_report, load_diagnostic_reports};
pub use editing::{
    SubtitleEditOutcome, SubtitleEditRequest, edit_subtitle, edit_subtitle_cancellable,
};
pub use error::{AdapterError, AdapterResult, ConfigError};
pub use fs::{
    default_output_path, is_supported_subtitle_path, read_document, render_and_write_document,
    stable_runtime_input_path,
};
pub use mock::MockBackend;
pub use pipeline::{
    PipelineOutcome, PipelineRequest, run_pipeline, run_pipeline_cancellable_with_progress,
};
pub use providers::{
    ApiFormat, BackendConfig, ProviderCheckOutcome, ProviderCheckRequest, build_backend,
    check_provider, default_api_key_env, resolve_env_var,
};
pub use runtime::{
    RuntimeAction, RuntimeCleanOutcome, RuntimeInspection, RuntimeOutcome, RuntimeRequest,
    run_runtime,
};
pub use runtime_store::FileRuntimeStore;
pub use settings::{
    BackendOverrides, BackendSettings, OutputOverrides, OutputSettings, ResolvedSettings,
    SettingsOverrides, StorageOverrides, StorageSettings, TranslationDomainSettings,
    TranslationOverrides, TranslationSettings,
};
pub use subbake_core::ports::BatchShardKind;
pub use transcription::{
    TranscriptionFormat, TranscriptionOutcome, TranscriptionRequest, TranscriptionSettings,
    transcribe_media, transcribe_media_cancellable, transcribe_media_cancellable_with_progress,
};
pub use translation::{
    BatchTranslationOutcome, BatchTranslationRequest, TranslationOutcome, TranslationRequest,
    translate_subtitle, translate_subtitle_batch, translate_subtitle_batch_cancellable,
    translate_subtitle_batch_with_progress, translate_subtitle_cancellable,
    translate_subtitle_cancellable_with_progress,
};
pub use whisper::{
    WhisperAction, WhisperModel, WhisperModelList, WhisperOutcome, WhisperRequest, WhisperStatus,
    default_whisper_binary_path, run_whisper, run_whisper_cancellable,
    run_whisper_cancellable_with_progress,
};
