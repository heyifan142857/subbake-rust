use crate::entities::{BatchTranslationResult, SubtitleSegment, Usage};
use crate::error::CoreResult;
use crate::storage::{RunState, RuntimePaths};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_owned(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_owned(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendJsonResult {
    pub payload: BackendPayload,
    pub usage: Usage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendPayload {
    Translation(BatchTranslationResult),
}

pub trait LlmBackend {
    fn provider_name(&self) -> &str;
    fn model_name(&self) -> &str;
    fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult>;
    fn generate_raw_json(
        &mut self,
        _messages: &[ChatMessage],
    ) -> CoreResult<(serde_json::Value, Usage)> {
        Err(crate::error::CoreError::Backend(format!(
            "{} backend does not support raw JSON generation",
            self.provider_name()
        )))
    }

    fn check_credentials(&self) -> CoreResult<(bool, String)> {
        Ok((
            true,
            format!("{} provider is configured.", self.provider_name()),
        ))
    }
}

impl<T> LlmBackend for Box<T>
where
    T: LlmBackend + ?Sized,
{
    fn provider_name(&self) -> &str {
        (**self).provider_name()
    }

    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
        (**self).generate_json(messages)
    }

    fn generate_raw_json(
        &mut self,
        messages: &[ChatMessage],
    ) -> CoreResult<(serde_json::Value, Usage)> {
        (**self).generate_raw_json(messages)
    }

    fn check_credentials(&self) -> CoreResult<(bool, String)> {
        (**self).check_credentials()
    }
}

pub trait DashboardSink {
    fn set_total_steps(&mut self, _total_steps: usize) {}
    fn mark_running(&mut self, _stage: &str) {}
    fn mark_done(&mut self, _stage: &str) {}
    fn add_usage(&mut self, _usage: Usage) {}
}

#[derive(Debug, Default)]
pub struct NoopDashboard;

impl DashboardSink for NoopDashboard {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchShardKind {
    Translated,
    Reviewed,
}

pub trait RuntimeStore {
    fn paths(&self) -> &RuntimePaths;
    fn ensure_layout(&self) -> CoreResult<()>;

    fn save_glossary(&self, entries: &[(String, String)]) -> CoreResult<()>;
    fn load_glossary(&self) -> CoreResult<Vec<(String, String)>> {
        let _ = self;
        Ok(Vec::new())
    }

    fn save_translation_memory(&self, entries: &[(String, String)]) -> CoreResult<()>;
    fn load_translation_memory(&self) -> CoreResult<Vec<(String, String)>> {
        let _ = self;
        Ok(Vec::new())
    }

    fn save_batch_segments(
        &self,
        kind: BatchShardKind,
        batch_index: usize,
        segments: &[SubtitleSegment],
    ) -> CoreResult<()>;
    fn load_batch_segments(
        &self,
        _kind: BatchShardKind,
        _completed_batches: usize,
    ) -> CoreResult<Vec<SubtitleSegment>> {
        Ok(Vec::new())
    }

    fn save_run_state(&self, _state: &RunState) -> CoreResult<()> {
        Ok(())
    }

    fn load_run_state(&self) -> CoreResult<Option<RunState>> {
        Ok(None)
    }
}
