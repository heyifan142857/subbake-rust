pub mod config;
pub mod fs;
pub mod mock;
pub mod providers;
pub mod runtime_store;
pub mod settings;
pub mod transcription;
pub mod translation;
pub mod whisper;

pub use config::{load_translation_settings_patch, parse_translation_settings_patch};
pub use fs::{
    default_output_path, is_supported_subtitle_path, read_document, render_and_write_document,
};
pub use mock::MockBackend;
pub use providers::{BackendConfig, build_backend};
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
pub use whisper::{WhisperAction, WhisperOutcome, WhisperRequest, WhisperStatus, run_whisper};
