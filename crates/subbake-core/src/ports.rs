use std::path::PathBuf;
use std::time::Duration;
use std::{any::Any, fmt};

use serde::{Deserialize, Serialize};

use crate::CancellationGuard;
use crate::entities::{
    AgentLog, BatchTranslationResult, FailureLog, ReviewReport, ReviewResult, SubtitleSegment,
    TerminologyPreflightResult, Usage,
};
use crate::error::{CoreResult, LlmCallError};
use crate::storage::{RunState, RuntimePaths};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// A stable shared prefix that adapters may map to provider prompt caching.
    #[serde(default)]
    pub cacheable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseContract {
    JsonObject,
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeToolSupport {
    Unknown,
    Supported,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    Required,
    Specific(String),
    None,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelToolResult {
    pub id: String,
    pub name: String,
    pub output: String,
    pub is_error: bool,
}

/// Provider-owned, in-memory state required to continue a native tool turn.
/// The domain layer intentionally cannot inspect or persist its contents.
pub struct ToolContinuation(Box<dyn Any + Send>);

impl ToolContinuation {
    pub fn new<T: Any + Send>(backend_id: impl Into<String>, value: T) -> Self {
        Self(Box::new((backend_id.into(), value)))
    }

    pub fn downcast_for<T: Any + Send>(self, backend_id: &str) -> Result<T, LlmCallError> {
        match self.0.downcast::<(String, T)>() {
            Ok(value) if value.0 == backend_id => Ok(value.1),
            Ok(value) => Err(LlmCallError::ContinuationMismatch(format!(
                "continuation belongs to backend `{}`, not `{backend_id}`",
                value.0
            ))),
            Err(_) => Err(LlmCallError::ContinuationMismatch(
                "continuation state has an incompatible provider type".to_owned(),
            )),
        }
    }
}

impl fmt::Debug for ToolContinuation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ToolContinuation(..)")
    }
}

#[derive(Debug)]
pub enum GenerationInput {
    Messages(Vec<ChatMessage>),
    Continue {
        continuation: ToolContinuation,
        tool_results: Vec<ModelToolResult>,
    },
}

#[derive(Debug)]
pub struct ToolConfiguration {
    pub definitions: Vec<ToolDefinition>,
    pub choice: ToolChoice,
}

#[derive(Debug)]
pub struct GenerationRequest {
    pub input: GenerationInput,
    pub response_contract: ResponseContract,
    pub tools: Option<ToolConfiguration>,
}

impl GenerationRequest {
    pub fn json(messages: Vec<ChatMessage>) -> Self {
        Self {
            input: GenerationInput::Messages(messages),
            response_contract: ResponseContract::JsonObject,
            tools: None,
        }
    }

    pub fn text(messages: Vec<ChatMessage>) -> Self {
        Self {
            input: GenerationInput::Messages(messages),
            response_contract: ResponseContract::Text,
            tools: None,
        }
    }

    pub fn with_tools(mut self, definitions: Vec<ToolDefinition>, choice: ToolChoice) -> Self {
        self.tools = Some(ToolConfiguration {
            definitions,
            choice,
        });
        self
    }

    pub fn continue_with_tools(
        continuation: ToolContinuation,
        tool_results: Vec<ModelToolResult>,
        definitions: Vec<ToolDefinition>,
        choice: ToolChoice,
        response_contract: ResponseContract,
    ) -> Self {
        Self {
            input: GenerationInput::Continue {
                continuation,
                tool_results,
            },
            response_contract,
            tools: Some(ToolConfiguration {
                definitions,
                choice,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationContent {
    Empty,
    Json(serde_json::Value),
    Text(String),
}

#[derive(Debug)]
pub struct GenerationResponse {
    pub content: GenerationContent,
    pub tool_calls: Vec<ModelToolCall>,
    pub continuation: Option<ToolContinuation>,
    pub usage: Usage,
}

impl GenerationResponse {
    pub fn json(json: serde_json::Value, usage: Usage) -> Self {
        Self {
            content: GenerationContent::Json(json),
            tool_calls: Vec::new(),
            continuation: None,
            usage,
        }
    }

    pub fn into_json(self) -> Result<(serde_json::Value, Usage), LlmCallError> {
        match self.content {
            GenerationContent::Json(json) => Ok((json, self.usage)),
            GenerationContent::Empty | GenerationContent::Text(_) => {
                Err(LlmCallError::InvalidResponse(
                    "backend returned text for a JSON response contract".to_owned(),
                ))
            }
        }
    }

    pub fn into_text(self) -> Result<(String, Usage), LlmCallError> {
        match self.content {
            GenerationContent::Text(text) => Ok((text, self.usage)),
            GenerationContent::Empty | GenerationContent::Json(_) => {
                Err(LlmCallError::InvalidResponse(
                    "backend returned JSON for a text response contract".to_owned(),
                ))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchExecutionOptions {
    pub max_concurrency: usize,
    pub deadline: Option<Duration>,
}

impl BatchExecutionOptions {
    pub fn new(max_concurrency: usize) -> Self {
        Self {
            max_concurrency: max_concurrency.max(1),
            deadline: None,
        }
    }
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_owned(),
            content: content.into(),
            cacheable: false,
        }
    }

    pub fn cacheable_system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_owned(),
            content: content.into(),
            cacheable: true,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_owned(),
            content: content.into(),
            cacheable: false,
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
    fn native_tool_support(&self) -> NativeToolSupport {
        NativeToolSupport::Unsupported
    }
    fn provider_name(&self) -> &str;
    fn model_name(&self) -> &str;
    fn execute(
        &mut self,
        request: GenerationRequest,
        cancellation: &CancellationGuard,
    ) -> Result<GenerationResponse, LlmCallError>;

    fn execute_many(
        &mut self,
        requests: Vec<GenerationRequest>,
        _options: BatchExecutionOptions,
        cancellation: &CancellationGuard,
    ) -> Result<Vec<Result<GenerationResponse, LlmCallError>>, LlmCallError> {
        let mut responses = Vec::with_capacity(requests.len());
        for request in requests {
            if cancellation.is_cancelled() {
                return Err(LlmCallError::Cancelled);
            }
            match self.execute(request, cancellation) {
                Err(LlmCallError::Cancelled) => return Err(LlmCallError::Cancelled),
                result => responses.push(result),
            }
        }
        Ok(responses)
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
    fn native_tool_support(&self) -> NativeToolSupport {
        (**self).native_tool_support()
    }
    fn provider_name(&self) -> &str {
        (**self).provider_name()
    }

    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    fn execute(
        &mut self,
        request: GenerationRequest,
        cancellation: &CancellationGuard,
    ) -> Result<GenerationResponse, LlmCallError> {
        (**self).execute(request, cancellation)
    }

    fn execute_many(
        &mut self,
        requests: Vec<GenerationRequest>,
        options: BatchExecutionOptions,
        cancellation: &CancellationGuard,
    ) -> Result<Vec<Result<GenerationResponse, LlmCallError>>, LlmCallError> {
        (**self).execute_many(requests, options, cancellation)
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

    fn save_review_report(&self, report: &ReviewReport) -> CoreResult<()>;

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

    fn save_run_state(&self, state: &RunState) -> CoreResult<()>;

    fn load_run_state(&self) -> CoreResult<Option<RunState>> {
        Ok(None)
    }

    fn save_cached_response(
        &self,
        stage: CacheStage,
        request_hash: &str,
        response: &BackendJsonResult,
    ) -> CoreResult<()>;

    fn load_cached_response(
        &self,
        _stage: CacheStage,
        _request_hash: &str,
    ) -> CoreResult<Option<BackendJsonResult>> {
        Ok(None)
    }

    fn save_failure_log(&self, log: &FailureLog) -> CoreResult<PathBuf>;

    fn save_agent_log(&self, log: &AgentLog) -> CoreResult<PathBuf>;
}

#[cfg(test)]
mod tool_tests {
    use super::{
        BatchExecutionOptions, ChatMessage, GenerationInput, GenerationRequest, GenerationResponse,
        LlmBackend, ToolContinuation,
    };
    use crate::CancellationToken;
    use crate::entities::Usage;
    use crate::error::LlmCallError;

    struct BatchBackend;

    impl LlmBackend for BatchBackend {
        fn provider_name(&self) -> &str {
            "batch-test"
        }

        fn model_name(&self) -> &str {
            "batch-test"
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            _cancellation: &crate::CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            let GenerationInput::Messages(messages) = request.input else {
                return Err(LlmCallError::ContinuationMismatch(
                    "unexpected continuation".to_owned(),
                ));
            };
            if messages
                .iter()
                .any(|message| message.content == "item failure")
            {
                return Err(LlmCallError::Authentication("bad key".to_owned()));
            }
            Ok(GenerationResponse::json(
                serde_json::json!({"ok": true}),
                Usage::default(),
            ))
        }
    }

    #[test]
    fn continuation_is_opaque_but_returns_to_its_provider() {
        let continuation = ToolContinuation::new("provider-a", vec!["wire state".to_owned()]);
        let state = continuation
            .downcast_for::<Vec<String>>("provider-a")
            .expect("provider continuation type");
        assert_eq!(state, vec!["wire state"]);
    }

    #[test]
    fn continuation_rejects_a_different_provider() {
        let continuation = ToolContinuation::new("provider-a", 42usize);
        assert!(matches!(
            continuation.downcast_for::<usize>("provider-b"),
            Err(crate::error::LlmCallError::ContinuationMismatch(_))
        ));
    }

    #[test]
    fn batch_keeps_item_failures_inside_the_ordered_result() {
        let requests = vec![
            GenerationRequest::json(vec![ChatMessage::user("item failure")]),
            GenerationRequest::json(vec![ChatMessage::user("ok")]),
        ];
        let results = BatchBackend
            .execute_many(
                requests,
                BatchExecutionOptions::new(2),
                &crate::CancellationGuard::never(),
            )
            .expect("batch scheduling");

        assert!(matches!(results[0], Err(LlmCallError::Authentication(_))));
        assert!(results[1].is_ok());
    }

    #[test]
    fn batch_cancellation_is_an_outer_failure() {
        let token = CancellationToken::default();
        let guard = token.guard();
        token.cancel();
        let error = BatchBackend
            .execute_many(
                vec![GenerationRequest::json(vec![ChatMessage::user("ok")])],
                BatchExecutionOptions::new(1),
                &guard,
            )
            .expect_err("shared cancellation must stop the whole batch");

        assert_eq!(error, LlmCallError::Cancelled);
    }
}
