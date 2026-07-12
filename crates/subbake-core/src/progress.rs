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
    Bytes,
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
