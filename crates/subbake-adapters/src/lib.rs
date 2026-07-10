pub mod config;
pub mod diagnostics;
pub mod editing;
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
    ConfigFile, discover_config_path, load_and_resolve, load_translation_settings_patch,
    parse_translation_settings_patch,
};
pub use diagnostics::{diagnose_failure_path, load_diagnostic_reports};
pub use editing::{SubtitleEditOutcome, SubtitleEditRequest, edit_subtitle};
pub use fs::{
    default_output_path, is_supported_subtitle_path, read_document, render_and_write_document,
};
pub use mock::MockBackend;
pub use pipeline::{PipelineOutcome, PipelineRequest, run_pipeline};
pub use providers::{
    BackendConfig, ProviderCheckOutcome, ProviderCheckRequest, build_backend, check_provider,
    default_api_key_env, resolve_env_var,
};
pub use runtime::{
    RuntimeAction, RuntimeCleanOutcome, RuntimeInspection, RuntimeOutcome, RuntimeRequest,
    run_runtime,
};
pub use runtime_store::FileRuntimeStore;
pub use settings::{TranslationSettings, TranslationSettingsPatch};
pub use subbake_core::ports::BatchShardKind;
pub use transcription::{
    TranscriptionFormat, TranscriptionOutcome, TranscriptionRequest, TranscriptionSettings,
    transcribe_media,
};
pub use translation::{
    BatchTranslationOutcome, BatchTranslationRequest, TranslationOutcome, TranslationRequest,
    translate_subtitle, translate_subtitle_batch,
};
pub use whisper::{
    WhisperAction, WhisperModel, WhisperModelList, WhisperOutcome, WhisperRequest, WhisperStatus,
    default_whisper_binary_path, run_whisper,
};
