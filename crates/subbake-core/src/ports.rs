use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::CancellationGuard;
use crate::entities::{
    AgentLog, BatchTranslationResult, FailureLog, ReviewReport, ReviewResult, SubtitleSegment,
    TerminologyPreflightResult, Usage,
};
use crate::error::CoreResult;
use crate::storage::{RunState, RuntimePaths};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// A protocol-neutral request for a model generation.  Business code owns the
/// JSON schema it asks for; HTTP adapters only transport and normalize it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub messages: Vec<ChatMessage>,
    pub response_contract: ResponseContract,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseContract {
    JsonObject,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationResponse {
    pub json: serde_json::Value,
    pub usage: Usage,
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
    Review(ReviewResult),
    Terminology(TerminologyPreflightResult),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStage {
    Translate,
    Review,
    Terminology,
    AgentTranslateRepair,
    AgentReviewRepair,
}

impl CacheStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Translate => "translate",
            Self::Review => "review",
            Self::Terminology => "terminology",
            Self::AgentTranslateRepair => "agent_translate_repair",
            Self::AgentReviewRepair => "agent_review_repair",
        }
    }
}

pub trait LlmBackend: Send {
    fn supports_terminology_preflight(&self) -> bool {
        false
    }

    fn supports_parallel_generation(&self) -> bool {
        false
    }
    fn provider_name(&self) -> &str;
    fn model_name(&self) -> &str;
    fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult>;
    fn generate_json_cancellable(
        &mut self,
        messages: &[ChatMessage],
        cancellation: &CancellationGuard,
    ) -> CoreResult<BackendJsonResult> {
        cancellation.check()?;
        self.generate_json(messages)
    }
    fn generate_raw_json(
        &mut self,
        _messages: &[ChatMessage],
    ) -> CoreResult<(serde_json::Value, Usage)> {
        Err(crate::error::CoreError::Backend(format!(
            "{} backend does not support raw JSON generation",
            self.provider_name()
        )))
    }
    fn generate_raw_json_cancellable(
        &mut self,
        messages: &[ChatMessage],
        cancellation: &CancellationGuard,
    ) -> CoreResult<(serde_json::Value, Usage)> {
        cancellation.check()?;
        self.generate_raw_json(messages)
    }

    /// The protocol-neutral generation API.  The legacy raw methods remain as
    /// compatibility shims while callers migrate their business contracts.
    fn generate(&mut self, request: GenerationRequest) -> CoreResult<GenerationResponse> {
        let (json, usage) = self.generate_raw_json(&request.messages)?;
        Ok(GenerationResponse { json, usage })
    }

    fn generate_cancellable(
        &mut self,
        request: GenerationRequest,
        cancellation: &CancellationGuard,
    ) -> CoreResult<GenerationResponse> {
        cancellation.check()?;
        let (json, usage) = self.generate_raw_json_cancellable(&request.messages, cancellation)?;
        Ok(GenerationResponse { json, usage })
    }

    fn generate_many_cancellable(
        &mut self,
        requests: Vec<GenerationRequest>,
        _max_concurrency: usize,
        cancellation: &CancellationGuard,
    ) -> Vec<CoreResult<GenerationResponse>> {
        requests
            .into_iter()
            .map(|request| {
                let result = self.generate_json_cancellable(&request.messages, cancellation)?;
                let json = match result.payload {
                    BackendPayload::Translation(payload) => serde_json::to_value(payload),
                    BackendPayload::Review(payload) => serde_json::to_value(payload),
                    BackendPayload::Terminology(payload) => serde_json::to_value(payload),
                }
                .map_err(|error| {
                    crate::error::CoreError::Data(format!(
                        "backend payload serialization failed: {error}"
                    ))
                })?;
                Ok(GenerationResponse {
                    json,
                    usage: result.usage,
                })
            })
            .collect()
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
    fn supports_terminology_preflight(&self) -> bool {
        (**self).supports_terminology_preflight()
    }

    fn supports_parallel_generation(&self) -> bool {
        (**self).supports_parallel_generation()
    }
    fn provider_name(&self) -> &str {
        (**self).provider_name()
    }

    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
        (**self).generate_json(messages)
    }
    fn generate_json_cancellable(
        &mut self,
        messages: &[ChatMessage],
        cancellation: &CancellationGuard,
    ) -> CoreResult<BackendJsonResult> {
        (**self).generate_json_cancellable(messages, cancellation)
    }

    fn generate_raw_json(
        &mut self,
        messages: &[ChatMessage],
    ) -> CoreResult<(serde_json::Value, Usage)> {
        (**self).generate_raw_json(messages)
    }
    fn generate_raw_json_cancellable(
        &mut self,
        messages: &[ChatMessage],
        cancellation: &CancellationGuard,
    ) -> CoreResult<(serde_json::Value, Usage)> {
        (**self).generate_raw_json_cancellable(messages, cancellation)
    }

    fn generate(&mut self, request: GenerationRequest) -> CoreResult<GenerationResponse> {
        (**self).generate(request)
    }
    fn generate_cancellable(
        &mut self,
        request: GenerationRequest,
        cancellation: &CancellationGuard,
    ) -> CoreResult<GenerationResponse> {
        (**self).generate_cancellable(request, cancellation)
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

    fn save_review_report(&self, _report: &ReviewReport) -> CoreResult<()> {
        Ok(())
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

    fn save_cached_response(
        &self,
        _stage: CacheStage,
        _request_hash: &str,
        _response: &BackendJsonResult,
    ) -> CoreResult<()> {
        Ok(())
    }

    fn load_cached_response(
        &self,
        _stage: CacheStage,
        _request_hash: &str,
    ) -> CoreResult<Option<BackendJsonResult>> {
        Ok(None)
    }

    fn save_failure_log(&self, log: &FailureLog) -> CoreResult<PathBuf> {
        Ok(self
            .paths()
            .failures_dir
            .join(format!("{}_batch_{:04}.json", log.stage, log.batch_index)))
    }

    fn save_agent_log(&self, log: &AgentLog) -> CoreResult<PathBuf> {
        Ok(self
            .paths()
            .agent_logs_dir
            .join(format!("{}_batch_{:04}.json", log.stage, log.batch_index)))
    }
}
