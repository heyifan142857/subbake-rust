use std::sync::Arc;

use crate::Usage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    Translation,
    BatchTranslation,
    Pipeline,
    Transcription,
    Download,
    Installation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    Pending,
    Running,
    Cancelling,
    Cancelled,
    Resuming,
    Completed,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressUnit {
    Steps,
    Files,
    Batches,
    Lines,
    Bytes,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TranslationProgress {
    pub segments_completed: u64,
    pub segments_total: u64,
    pub batches_committed: u64,
    pub batches_total: u64,
    pub requests_in_flight: u64,
    pub requests_buffered: u64,
    pub requests_retrying: u64,
    pub cache_hits: u64,
    pub translation_memory_hits: u64,
    pub terminology_candidates: u64,
    pub terminology_conflicts: u64,
    pub terminology_patched: u64,
    pub window_index: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressEvent {
    pub task: TaskKind,
    pub stage: String,
    pub state: TaskState,
    pub current: u64,
    pub total: Option<u64>,
    pub unit: ProgressUnit,
    pub resumed: u64,
    pub usage: Usage,
    pub message: Option<String>,
    pub translation: Option<TranslationProgress>,
}

impl ProgressEvent {
    pub fn running(
        task: TaskKind,
        stage: impl Into<String>,
        current: u64,
        total: Option<u64>,
        unit: ProgressUnit,
    ) -> Self {
        Self {
            task,
            stage: stage.into(),
            state: TaskState::Running,
            current,
            total,
            unit,
            resumed: 0,
            usage: Usage::default(),
            message: None,
            translation: None,
        }
    }
}

pub trait ProgressSink: Send + Sync {
    fn emit(&self, event: ProgressEvent);
}
impl<T: ProgressSink + ?Sized> ProgressSink for Arc<T> {
    fn emit(&self, event: ProgressEvent) {
        (**self).emit(event);
    }
}

#[derive(Debug, Default)]
pub struct NoopProgress;
impl ProgressSink for NoopProgress {
    fn emit(&self, _event: ProgressEvent) {}
}

pub type SharedProgress = Arc<dyn ProgressSink>;
