pub mod config;
pub mod diagnostics;
pub mod editing;
pub mod embedded_subtitles;
pub mod error;
pub mod fs;
pub mod llm_backends;
pub mod mock;
pub mod overnight;
pub mod pipeline;
mod process;
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
pub use embedded_subtitles::{
    default_embedded_translation_output_path, is_supported_subtitle_container_path,
    remove_embedded_subtitle_by_title, restore_embedded_subtitle_from_srt,
    translate_embedded_subtitle, translate_embedded_subtitle_cancellable,
    translate_embedded_subtitle_cancellable_with_progress,
};
pub use error::{AdapterError, AdapterResult, ConfigError};
pub use fs::{
    default_output_path, default_output_path_with_language, is_supported_subtitle_path,
    read_document, render_and_write_document, stable_runtime_input_path,
};
pub use mock::MockBackend;
pub use overnight::{
    OvernightCollectOutcome, OvernightCollectRequest, OvernightStatusOutcome,
    OvernightStatusRequest, OvernightSubmitOutcome, OvernightSubmitRequest, collect_overnight,
    overnight_status, submit_overnight,
};
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
    SettingsOverrides, StorageOverrides, StorageSettings, TranscriptionDomainSettings,
    TranscriptionOverrides, TranslationDomainSettings, TranslationOverrides, TranslationSettings,
};
pub use subbake_core::ports::BatchShardKind;
pub use transcription::{
    MultipleModelPolicy, TranscriptionFormat, TranscriptionOutcome, TranscriptionRequest,
    TranscriptionSettings, apply_whisper_configuration, apply_whisper_storage, transcribe_media,
    transcribe_media_cancellable, transcribe_media_cancellable_with_progress,
};
pub use translation::{
    BatchTranslationOutcome, BatchTranslationRequest, ContainerTranslationChange,
    TranslationOutcome, TranslationRequest, batch_translation_output_path,
    default_translation_output_path, translate_input, translate_input_cancellable,
    translate_input_cancellable_with_progress, translate_subtitle, translate_subtitle_batch,
    translate_subtitle_batch_cancellable, translate_subtitle_batch_with_progress,
    translate_subtitle_cancellable, translate_subtitle_cancellable_with_progress,
};
pub use whisper::{
    WhisperAction, WhisperBuildVariant, WhisperModel, WhisperModelList, WhisperOutcome,
    WhisperRequest, WhisperStatus, WhisperVersion, WhisperVersionList, default_whisper_binary_path,
    default_whisper_binary_path_for, default_whisper_models_dir, default_whisper_models_dir_for,
    run_whisper, run_whisper_cancellable, run_whisper_cancellable_with_progress,
};
