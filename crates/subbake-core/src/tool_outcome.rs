use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::entities::BilingualOrder;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionStatus {
    Written,
    DryRun,
    Skipped,
    Unchanged,
    Observed,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedPath {
    pub path: PathBuf,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranslationToolOutcome {
    pub status: ToolExecutionStatus,
    pub source_language: String,
    pub target_language: String,
    pub provider: String,
    pub model: String,
    pub output_format: String,
    pub bilingual: bool,
    pub bilingual_order: BilingualOrder,
    pub inputs: Vec<PathBuf>,
    pub outputs: Vec<PathBuf>,
    pub processed_files: usize,
    pub skipped: Vec<SkippedPath>,
    pub subtitle_entries: usize,
    pub dry_run: bool,
    pub cache_hits: usize,
    pub resumed_translation_batches: usize,
    pub resumed_review_batches: usize,
    pub translation_memory_hits: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptionToolOutcome {
    pub status: ToolExecutionStatus,
    pub input: PathBuf,
    pub output: PathBuf,
    pub language: String,
    pub provider: String,
    pub model: String,
    pub output_format: String,
    pub subtitle_entries: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubtitleEditToolOutcome {
    pub status: ToolExecutionStatus,
    pub target_path: PathBuf,
    pub target_language: String,
    pub modified_entries: usize,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub edit_notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhisperToolOutcome {
    pub status: ToolExecutionStatus,
    pub action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_exists: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models_dir: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models_dir_exists: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<WhisperModelFact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub available_models: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub available_versions: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_warning: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhisperModelFact {
    pub name: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileToolOutcome {
    pub status: ToolExecutionStatus,
    pub action: String,
    pub paths: Vec<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub destination_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileToolOutcome {
    pub status: ToolExecutionStatus,
    pub action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_profile: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub available_profiles: Vec<String>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationToolOutcome {
    pub status: ToolExecutionStatus,
    pub observation: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", content = "facts", rename_all = "snake_case")]
pub enum AgentToolOutcome {
    Translation(TranslationToolOutcome),
    Transcription(TranscriptionToolOutcome),
    SubtitleEdit(SubtitleEditToolOutcome),
    Whisper(WhisperToolOutcome),
    File(FileToolOutcome),
    Profile(ProfileToolOutcome),
    Observation(ObservationToolOutcome),
}
