use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use subbake_core::CancellationGuard;
#[cfg(test)]
use subbake_core::entities::{BatchTranslationResult, GlossaryEntry, TranslationLine};
use subbake_core::error::{CoreError, CoreResult, LlmCallError};
use subbake_core::ports::{
    BatchExecutionOptions, ChatMessage, GenerationContent, GenerationInput, GenerationRequest,
    GenerationResponse, LlmBackend, NativeToolSupport, ResponseContract, ToolContinuation,
};
use tokio::runtime::Runtime;

use crate::error::{AdapterError, AdapterResult};
use crate::providers::{ApiFormat, BackendConfig};
mod native;
mod protocols;
mod transport;

use native::{
    ProtocolContinuation, append_results as append_native_results,
    parse_response as parse_native_response, request_body as native_request_body,
    start as native_start,
};
use protocols::{authenticate, endpoint, request_body, response_text, usage};
#[cfg(test)]
use transport::native_tools_unsupported;
use transport::{await_http, classify_status_error, client, send_json};
const TIMEOUT: f64 = 120.0;
static BACKEND_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// All protocol implementations share transport, auth validation and response
/// normalization. Public aliases make the registry's four adapters explicit.
#[derive(Clone)]
pub struct ProtocolAdapter {
    config: BackendConfig,
    format: ApiFormat,
    key: String,
    client: Client,
    runtime: Arc<Runtime>,
    backend_id: String,
    native_tool_support: NativeToolSupport,
}

pub type OpenAiChatAdapter = ProtocolAdapter;
pub type OpenAiResponsesAdapter = ProtocolAdapter;
pub type AnthropicMessagesAdapter = ProtocolAdapter;
pub type GeminiGenerateContentAdapter = ProtocolAdapter;

/// Adapter for OpenAI-compatible Batch endpoints. It intentionally supports
/// only the two OpenAI request formats because the batch JSONL wire contract
/// is provider-specific.
pub struct OpenAiBatchClient {
    config: BackendConfig,
    format: ApiFormat,
    key: String,
    client: Client,
    runtime: Runtime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiBatchStatus {
    pub id: String,
    pub status: String,
    pub input_file_id: String,
    pub output_file_id: Option<String>,
    pub error_file_id: Option<String>,
    pub total: Option<usize>,
    pub completed: Option<usize>,
    pub failed: Option<usize>,
}

pub fn build_openai_batch_client(
    config: &BackendConfig,
    timeout: f64,
) -> AdapterResult<OpenAiBatchClient> {
    config.validate()?;
    let format = config
        .api_format
        .ok_or_else(|| AdapterError::invalid_input("api_format is required"))?;
    if !matches!(format, ApiFormat::OpenaiChat | ApiFormat::OpenaiResponses) {
        return Err(AdapterError::invalid_input(
            "overnight batches currently require api_format openai_chat or openai_responses",
        ));
    }
    if config.endpoint_url.is_some() {
        return Err(AdapterError::invalid_input(
            "overnight batches require base_url rather than endpoint_url so /files and /batches can be addressed safely",
        ));
    }
    let key = config
        .resolved_api_key()
        .ok_or_else(|| AdapterError::Authentication {
            message: format!(
                "Missing API key for {} provider. Set --api-key, api_key, or api_key_env.",
                config.display_name
            ),
        })?;
    if key.contains(['\r', '\n']) {
        return Err(AdapterError::invalid_input(
            "authentication header value must not contain CR/LF",
        ));
    }
    Ok(OpenAiBatchClient {
        config: config.clone(),
        format,
        key,
        client: client(timeout)?,
        runtime: Runtime::new().map_err(|source| AdapterError::ExternalIo {
            operation: "start overnight batch runtime",
            path: None,
            source,
        })?,
    })
}

impl OpenAiBatchClient {
    pub fn endpoint_path(&self) -> &'static str {
        match self.format {
            ApiFormat::OpenaiChat => "/v1/chat/completions",
            ApiFormat::OpenaiResponses => "/v1/responses",
            _ => unreachable!(),
        }
    }

    pub fn request_body(&self, messages: &[ChatMessage]) -> Value {
        request_body(
            self.format,
            &self.config.model,
            messages,
            ResponseContract::JsonObject,
        )
    }

    pub fn submit_jsonl(
        &self,
        jsonl: &str,
        metadata: Value,
        cancellation: &CancellationGuard,
    ) -> AdapterResult<OpenAiBatchStatus> {
        let input_file_id = self.upload(jsonl, cancellation)?;
        let payload = json!({
            "input_file_id": input_file_id,
            "endpoint": self.endpoint_path(),
            "completion_window": "24h",
            "metadata": metadata,
        });
        self.json_request(
            self.authenticated(self.client.post(self.api_url("batches")).json(&payload)),
            cancellation,
        )
        .and_then(parse_batch_status)
    }

    pub fn status(
        &self,
        job_id: &str,
        cancellation: &CancellationGuard,
    ) -> AdapterResult<OpenAiBatchStatus> {
        self.json_request(
            self.authenticated(self.client.get(self.api_url(&format!("batches/{job_id}")))),
            cancellation,
        )
        .and_then(parse_batch_status)
    }

    pub fn download_output(
        &self,
        file_id: &str,
        cancellation: &CancellationGuard,
    ) -> AdapterResult<String> {
        cancellation.check().map_err(AdapterError::from)?;
        let request = self.authenticated(
            self.client
                .get(self.api_url(&format!("files/{file_id}/content"))),
        );
        self.runtime
            .block_on(async {
                let response = await_http(
                    request.send(),
                    cancellation,
                    "overnight batch download failed",
                )
                .await?;
                let status = response.status();
                let text = await_http(
                    response.text(),
                    cancellation,
                    "overnight batch output read failed",
                )
                .await?;
                if status.is_success() {
                    Ok(text)
                } else {
                    Err(classify_status_error(status.as_u16(), text, None))
                }
            })
            .map_err(AdapterError::from)
    }

    pub fn parse_output_json(&self, body: &Value) -> AdapterResult<Value> {
        let text = response_text(self.format, body)
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| AdapterError::invalid_input("batch response missing text output"))?;
        serde_json::from_str(&text).map_err(|error| {
            AdapterError::invalid_input(format!(
                "batch response did not contain a JSON translation payload: {error}"
            ))
        })
    }

    fn upload(&self, jsonl: &str, cancellation: &CancellationGuard) -> AdapterResult<String> {
        cancellation.check().map_err(AdapterError::from)?;
        let part = reqwest::multipart::Part::text(jsonl.to_owned())
            .file_name("subbake-overnight.jsonl")
            .mime_str("application/jsonl")
            .map_err(|error| {
                AdapterError::invalid_input(format!("invalid batch mime type: {error}"))
            })?;
        let form = reqwest::multipart::Form::new()
            .text("purpose", "batch")
            .part("file", part);
        let value = self.json_request(
            self.authenticated(self.client.post(self.api_url("files")).multipart(form)),
            cancellation,
        )?;
        value["id"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| AdapterError::invalid_input("batch upload response missing file id"))
    }

    fn api_url(&self, suffix: &str) -> String {
        format!(
            "{}/{}",
            self.config
                .base_url
                .as_deref()
                .unwrap_or("https://api.openai.com/v1")
                .trim_end_matches('/'),
            suffix
        )
    }

    fn authenticated(&self, request: RequestBuilder) -> RequestBuilder {
        authenticate(self.format, &self.config, &self.key, request)
    }

    fn json_request(
        &self,
        request: RequestBuilder,
        cancellation: &CancellationGuard,
    ) -> AdapterResult<Value> {
        cancellation.check().map_err(AdapterError::from)?;
        self.runtime
            .block_on(async {
                let response = await_http(
                    request.send(),
                    cancellation,
                    "overnight batch request failed",
                )
                .await?;
                let status = response.status();
                let text = await_http(
                    response.text(),
                    cancellation,
                    "overnight batch response read failed",
                )
                .await?;
                if status.is_success() {
                    serde_json::from_str(&text).map_err(|error| {
                        LlmCallError::InvalidResponse(format!(
                            "batch response decode failed: {error}"
                        ))
                    })
                } else {
                    Err(classify_status_error(status.as_u16(), text, None))
                }
            })
            .map_err(AdapterError::from)
    }
}

fn parse_batch_status(value: Value) -> AdapterResult<OpenAiBatchStatus> {
    let id = value["id"]
        .as_str()
        .ok_or_else(|| AdapterError::invalid_input("batch response missing id"))?
        .to_owned();
    let status = value["status"].as_str().unwrap_or("unknown").to_owned();
    let input_file_id = value["input_file_id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    Ok(OpenAiBatchStatus {
        id,
        status,
        input_file_id,
        output_file_id: value["output_file_id"].as_str().map(str::to_owned),
        error_file_id: value["error_file_id"].as_str().map(str::to_owned),
        total: value["request_counts"]["total"]
            .as_u64()
            .map(|v| v as usize),
        completed: value["request_counts"]["completed"]
            .as_u64()
            .map(|v| v as usize),
        failed: value["request_counts"]["failed"]
            .as_u64()
            .map(|v| v as usize),
    })
}

pub fn build_protocol_backend(
    config: &BackendConfig,
    timeout: f64,
) -> AdapterResult<Box<dyn LlmBackend>> {
    let format = config
        .api_format
        .ok_or_else(|| AdapterError::invalid_input("api_format is required"))?;
    let key = config
        .resolved_api_key()
        .ok_or_else(|| AdapterError::Authentication {
            message: format!(
                "Missing API key for {} provider. Set --api-key, api_key, or api_key_env.",
                config.display_name
            ),
        })?;
    if key.contains(['\r', '\n']) {
        return Err(AdapterError::invalid_input(
            "authentication header value must not contain CR/LF",
        ));
    }
    Ok(Box::new(ProtocolAdapter {
        config: config.clone(),
        format,
        key,
        client: client(timeout)?,
        runtime: Arc::new(Runtime::new().map_err(|source| AdapterError::ExternalIo {
            operation: "start LLM runtime",
            path: None,
            source,
        })?),
        backend_id: format!(
            "{}:{}:{}",
            config.display_name,
            config.model,
            BACKEND_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ),
        native_tool_support: NativeToolSupport::Unknown,
    }))
}
impl ProtocolAdapter {
    fn endpoint(&self) -> String {
        endpoint(self.format, &self.config)
    }
    fn authenticated(&self, request: RequestBuilder) -> RequestBuilder {
        authenticate(self.format, &self.config, &self.key, request)
    }
    fn body(&self, messages: &[ChatMessage], contract: ResponseContract) -> Value {
        request_body(self.format, &self.config.model, messages, contract)
    }

    async fn send_payload(
        &self,
        payload: &Value,
        cancellation: &CancellationGuard,
        native_tools: bool,
    ) -> Result<Value, LlmCallError> {
        let endpoint = self.endpoint();
        send_json(
            || self.authenticated(self.client.post(&endpoint).json(payload)),
            &self.config.display_name,
            cancellation,
            native_tools,
        )
        .await
    }

    fn native_payload(
        &self,
        request: GenerationRequest,
    ) -> Result<(Value, ProtocolContinuation, ResponseContract), LlmCallError> {
        let tool_config = request.tools.ok_or_else(|| {
            LlmCallError::UnsupportedCapability("native tools were not configured".to_owned())
        })?;
        let tools = tool_config.definitions;
        let choice = tool_config.choice;
        let mut continuation = match request.input {
            GenerationInput::Messages(messages) => native_start(self.format, &messages),
            GenerationInput::Continue {
                continuation,
                tool_results,
            } => {
                let mut continuation =
                    continuation.downcast_for::<ProtocolContinuation>(&self.backend_id)?;
                if continuation.format != self.format {
                    return Err(LlmCallError::ContinuationMismatch(
                        "native tool continuation protocol changed".to_owned(),
                    ));
                }
                append_native_results(&mut continuation, &tool_results)
                    .map_err(|error| LlmCallError::InvalidResponse(error.to_string()))?;
                continuation
            }
        };
        continuation.call_ids.clear();
        let payload = native_request_body(
            self.format,
            &self.config.model,
            &continuation,
            &tools,
            &choice,
        );
        Ok((payload, continuation, request.response_contract))
    }

    async fn run_native(
        &self,
        request: GenerationRequest,
        cancel: &CancellationGuard,
    ) -> Result<GenerationResponse, LlmCallError> {
        let (payload, mut continuation, contract) = self.native_payload(request)?;
        let body = self.send_payload(&payload, cancel, true).await?;
        let (response_text, tool_calls) =
            parse_native_response(self.format, &body, &mut continuation)
                .map_err(|error| LlmCallError::InvalidResponse(error.to_string()))?;
        let usage = usage(
            self.format,
            &body,
            response_text.as_deref().unwrap_or(""),
            &payload,
        );
        let continuation = (!tool_calls.is_empty())
            .then(|| ToolContinuation::new(self.backend_id.clone(), continuation));
        let content = if !tool_calls.is_empty() && response_text.is_none() {
            GenerationContent::Empty
        } else {
            match contract {
                ResponseContract::JsonObject => GenerationContent::Json(
                    extract_json_object(response_text.as_deref().unwrap_or_default())
                        .map_err(|error| LlmCallError::InvalidResponse(error.to_string()))?,
                ),
                ResponseContract::Text => {
                    GenerationContent::Text(response_text.unwrap_or_default())
                }
            }
        };
        Ok(GenerationResponse {
            content,
            tool_calls,
            continuation,
            usage,
        })
    }

    async fn run(
        &self,
        messages: &[ChatMessage],
        contract: ResponseContract,
        cancel: &CancellationGuard,
    ) -> Result<GenerationResponse, LlmCallError> {
        let payload = self.body(messages, contract);
        let body = self.send_payload(&payload, cancel, false).await?;
        let text = self
            .content(&body)
            .map_err(|error| LlmCallError::InvalidResponse(error.to_string()))?;
        let content = match contract {
            ResponseContract::JsonObject => GenerationContent::Json(
                extract_json_object(&text)
                    .map_err(|error| LlmCallError::InvalidResponse(error.to_string()))?,
            ),
            ResponseContract::Text => GenerationContent::Text(text.clone()),
        };
        Ok(GenerationResponse {
            usage: usage(self.format, &body, &text, &payload),
            content,
            tool_calls: Vec::new(),
            continuation: None,
        })
    }
    fn content(&self, body: &Value) -> CoreResult<String> {
        let text = response_text(self.format, body);
        text.filter(|s| !s.is_empty()).ok_or_else(|| {
            CoreError::InvalidBackendResponse(format!(
                "{} response missing text output",
                self.config.display_name
            ))
        })
    }
}

#[cfg(test)]
fn apply_response_contract(format: ApiFormat, contract: ResponseContract, body: &mut Value) {
    protocols::apply_response_contract(format, contract, body);
}

impl LlmBackend for ProtocolAdapter {
    fn supports_terminology_preflight(&self) -> bool {
        true
    }

    fn supports_parallel_generation(&self) -> bool {
        true
    }
    fn supports_compact_translation(&self) -> bool {
        true
    }
    fn native_tool_support(&self) -> NativeToolSupport {
        self.native_tool_support
    }
    fn provider_name(&self) -> &str {
        &self.config.display_name
    }
    fn model_name(&self) -> &str {
        &self.config.model
    }
    fn execute(
        &mut self,
        request: GenerationRequest,
        cancellation: &CancellationGuard,
    ) -> Result<GenerationResponse, LlmCallError> {
        if request.tools.is_some() {
            if self.native_tool_support == NativeToolSupport::Unsupported {
                return Err(LlmCallError::UnsupportedCapability(
                    "native tools".to_owned(),
                ));
            }
            let result = self
                .runtime
                .block_on(self.run_native(request, cancellation));
            match &result {
                Ok(_) => self.native_tool_support = NativeToolSupport::Supported,
                Err(LlmCallError::UnsupportedCapability(_)) => {
                    self.native_tool_support = NativeToolSupport::Unsupported;
                }
                Err(_) => {}
            }
            return result;
        }
        match request.input {
            GenerationInput::Messages(messages) => {
                self.runtime
                    .block_on(self.run(&messages, request.response_contract, cancellation))
            }
            GenerationInput::Continue { .. } => Err(LlmCallError::ContinuationMismatch(
                "continuation request is missing native tool configuration".to_owned(),
            )),
        }
    }

    fn execute_many(
        &mut self,
        requests: Vec<GenerationRequest>,
        options: BatchExecutionOptions,
        cancellation: &CancellationGuard,
    ) -> Result<Vec<Result<GenerationResponse, LlmCallError>>, LlmCallError> {
        let adapter = self.clone();
        let cancellation = cancellation.clone();
        let deadline = options.deadline;
        let batch = async move {
            let semaphore =
                std::sync::Arc::new(tokio::sync::Semaphore::new(options.max_concurrency));
            let mut set = tokio::task::JoinSet::new();
            for (index, request) in requests.into_iter().enumerate() {
                let adapter = adapter.clone();
                let cancellation = cancellation.clone();
                let semaphore = semaphore.clone();
                set.spawn(async move {
                    let permit = semaphore.acquire_owned().await.map_err(|error| {
                        LlmCallError::Transport(format!("concurrency limiter closed: {error}"))
                    })?;
                    cancellation.check().map_err(LlmCallError::from)?;
                    let result = if request.tools.is_some() {
                        adapter.run_native(request, &cancellation).await
                    } else {
                        match request.input {
                            GenerationInput::Messages(messages) => {
                                adapter
                                    .run(&messages, request.response_contract, &cancellation)
                                    .await
                            }
                            GenerationInput::Continue { .. } => {
                                Err(LlmCallError::ContinuationMismatch(
                                    "continuation request is missing native tool configuration"
                                        .to_owned(),
                                ))
                            }
                        }
                    };
                    drop(permit);
                    Ok::<_, LlmCallError>((index, result))
                });
            }
            let mut ordered = Vec::new();
            while let Some(joined) = set.join_next().await {
                match joined {
                    Ok(Ok((_, Err(LlmCallError::Cancelled)))) => {
                        return Err(LlmCallError::Cancelled);
                    }
                    Ok(Ok(item)) => ordered.push(item),
                    Ok(Err(error)) => return Err(error),
                    Err(error) => {
                        return Err(LlmCallError::Transport(format!(
                            "provider task failed: {error}"
                        )));
                    }
                }
            }
            ordered.sort_by_key(|(index, _)| *index);
            Ok(ordered.into_iter().map(|(_, result)| result).collect())
        };
        self.runtime.block_on(async move {
            if let Some(deadline) = deadline {
                tokio::time::timeout(deadline, batch)
                    .await
                    .map_err(|_| LlmCallError::Timeout("batch deadline elapsed".to_owned()))?
            } else {
                batch.await
            }
        })
    }
}
pub fn default_timeout_seconds() -> f64 {
    TIMEOUT
}
pub(crate) fn extract_json_object(text: &str) -> CoreResult<Value> {
    let t = text.trim();
    let start = t.find('{').ok_or_else(|| {
        CoreError::InvalidBackendResponse("response is missing a JSON object".to_owned())
    })?;
    let (mut d, mut s, mut e) = (0i32, false, false);
    for (i, c) in t.char_indices().skip_while(|(i, _)| *i < start) {
        if s {
            if e {
                e = false
            } else if c == '\\' {
                e = true
            } else if c == '"' {
                s = false
            };
            continue;
        }
        match c {
            '"' => s = true,
            '{' => d += 1,
            '}' => {
                d -= 1;
                if d == 0 {
                    return serde_json::from_str(&t[start..i + 1]).map_err(|x| {
                        CoreError::InvalidBackendResponse(format!("invalid JSON in response: {x}"))
                    });
                }
            }
            _ => {}
        }
    }
    Err(CoreError::InvalidBackendResponse(
        "response JSON object is unbalanced".to_owned(),
    ))
}
#[cfg(test)]
pub(crate) fn parse_translation_payload(v: &Value) -> CoreResult<BatchTranslationResult> {
    let lines = v["lines"]
        .as_array()
        .ok_or_else(|| CoreError::InvalidTranslation("response missing lines array".to_owned()))?
        .iter()
        .enumerate()
        .map(|(index, line)| parse_translation_line(line, index))
        .collect::<CoreResult<Vec<_>>>()?;
    let glossary_updates = match &v["glossary_updates"] {
        Value::Array(a) => a
            .iter()
            .map(|x| GlossaryEntry {
                source: x["source"].as_str().unwrap_or_default().to_owned(),
                target: x["target"].as_str().unwrap_or_default().to_owned(),
            })
            .collect(),
        Value::Object(m) => m
            .iter()
            .map(|(s, t)| GlossaryEntry {
                source: s.clone(),
                target: t.as_str().unwrap_or_default().to_owned(),
            })
            .collect(),
        _ => Vec::new(),
    };
    Ok(BatchTranslationResult {
        lines,
        summary: v["summary"].as_str().unwrap_or_default().to_owned(),
        glossary_updates,
    })
}
#[cfg(test)]
fn parse_translation_line(v: &Value, index: usize) -> CoreResult<TranslationLine> {
    let id = v["id"].as_str().ok_or_else(|| {
        CoreError::InvalidTranslation(format!("line {} is missing string field `id`", index + 1))
    })?;
    let translation = ["translation", "translated_text", "text"]
        .into_iter()
        .find_map(|field| v[field].as_str())
        .ok_or_else(|| {
            CoreError::InvalidTranslation(format!(
                "translation for id `{id}` is missing string field `translation`"
            ))
        })?;
    Ok(TranslationLine {
        id: id.to_owned(),
        translation: translation.to_owned(),
    })
}
#[cfg(test)]
mod tests {
    use serde_json::json;
    use subbake_core::error::LlmCallError;
    use subbake_core::ports::{ModelToolResult, ResponseContract, ToolChoice, ToolDefinition};

    use super::{
        ApiFormat, append_native_results, apply_response_contract, classify_status_error,
        native_request_body, native_start, native_tools_unsupported, parse_native_response,
        parse_translation_payload,
    };

    fn tool() -> ToolDefinition {
        ToolDefinition {
            name: "list_files".to_owned(),
            description: "List files".to_owned(),
            input_schema: json!({"type":"object","properties":{}}),
        }
    }

    #[test]
    fn translation_payload_accepts_migration_field_aliases() {
        let payload = parse_translation_payload(&json!({
            "lines": [
                {"id": "1", "translated_text": "你好"},
                {"id": "2", "text": "再见"}
            ]
        }))
        .expect("migration aliases should parse");

        assert_eq!(payload.lines[0].translation, "你好");
        assert_eq!(payload.lines[1].translation, "再见");
    }

    #[test]
    fn translation_payload_reports_a_missing_translation_field() {
        let error = parse_translation_payload(&json!({
            "lines": [{"id": "1", "content": "你好"}]
        }))
        .expect_err("unknown fields must not become an empty translation");

        assert!(
            error
                .to_string()
                .contains("missing string field `translation`")
        );
    }

    #[test]
    fn response_contract_is_encoded_only_by_protocol_adapters_that_support_it() {
        let mut chat = json!({"model": "x"});
        apply_response_contract(
            ApiFormat::OpenaiChat,
            ResponseContract::JsonObject,
            &mut chat,
        );
        assert_eq!(chat["response_format"]["type"], "json_object");

        let mut responses = json!({"model": "x"});
        apply_response_contract(
            ApiFormat::OpenaiResponses,
            ResponseContract::JsonObject,
            &mut responses,
        );
        assert_eq!(responses["text"]["format"]["type"], "json_object");

        let mut anthropic = json!({"model": "x"});
        apply_response_contract(
            ApiFormat::AnthropicMessages,
            ResponseContract::JsonObject,
            &mut anthropic,
        );
        assert!(anthropic.get("response_format").is_none());

        let mut text = json!({"model": "x"});
        apply_response_contract(ApiFormat::OpenaiChat, ResponseContract::Text, &mut text);
        assert!(text.get("response_format").is_none());
    }

    #[test]
    fn status_errors_keep_authentication_rate_limit_and_retry_categories() {
        assert!(matches!(
            classify_status_error(401, "bad key".to_owned(), None),
            LlmCallError::Authentication(_)
        ));
        let limited = classify_status_error(429, "slow down".to_owned(), Some(2_000));
        assert!(matches!(
            &limited,
            LlmCallError::RateLimited {
                retry_after_ms: Some(2_000),
                ..
            }
        ));
        assert!(limited.is_retryable());
        assert!(classify_status_error(503, "unavailable".to_owned(), None).is_retryable());
        assert!(!classify_status_error(400, "invalid request".to_owned(), None).is_retryable());
    }

    #[test]
    fn openai_chat_native_turn_round_trips_tool_results() {
        let mut continuation = native_start(
            ApiFormat::OpenaiChat,
            &[subbake_core::ports::ChatMessage::user("list")],
        );
        let (_, calls) = parse_native_response(
            ApiFormat::OpenaiChat,
            &json!({"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"list_files","arguments":"{\"path\":\".\"}"}}]}}]}),
            &mut continuation,
        )
        .expect("parse native response");
        assert_eq!(calls[0].arguments["path"], ".");
        append_native_results(
            &mut continuation,
            &[ModelToolResult {
                id: "call_1".to_owned(),
                name: "list_files".to_owned(),
                output: "clip.srt".to_owned(),
                is_error: false,
            }],
        )
        .expect("append result");
        let body = native_request_body(
            ApiFormat::OpenaiChat,
            "model",
            &continuation,
            &[tool()],
            &ToolChoice::Auto,
        );
        assert_eq!(body["messages"][2]["role"], "tool");
        assert_eq!(body["messages"][2]["tool_call_id"], "call_1");
    }

    #[test]
    fn openai_chat_parses_deepseek_dsml_tool_calls() {
        let mut continuation = native_start(ApiFormat::OpenaiChat, &[]);
        let content = concat!(
            "<｜｜DSML｜｜tool_calls>\n",
            "<｜｜DSML｜｜invoke name=\"translate_file\">\n",
            "<｜｜DSML｜｜parameter name=\"file_path\" string=\"true\">/tmp/a&amp;b.srt</｜｜DSML｜｜parameter>\n",
            "<｜｜DSML｜｜parameter name=\"bilingual\" string=\"false\">false</｜｜DSML｜｜parameter>\n",
            "</｜｜DSML｜｜invoke>\n",
            "</｜｜DSML｜｜tool_calls>"
        );
        let (text, calls) = parse_native_response(
            ApiFormat::OpenaiChat,
            &json!({"choices":[{"message":{"role":"assistant","content":content}}]}),
            &mut continuation,
        )
        .expect("parse DSML response");

        assert_eq!(text, None);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "translate_file");
        assert_eq!(calls[0].arguments["path"], "/tmp/a&b.srt");
        assert_eq!(calls[0].arguments["bilingual"], false);
        assert_eq!(continuation.history[0]["content"], json!(null));
        assert_eq!(
            continuation.history[0]["tool_calls"][0]["function"]["name"],
            "translate_file"
        );
    }

    #[test]
    fn malformed_dsml_is_reported_instead_of_rendered() {
        let mut continuation = native_start(ApiFormat::OpenaiChat, &[]);
        let error = parse_native_response(
            ApiFormat::OpenaiChat,
            &json!({"choices":[{"message":{"role":"assistant","content":"<｜｜DSML｜｜tool_calls></｜｜DSML｜｜tool_calls>"}}]}),
            &mut continuation,
        )
        .expect_err("empty DSML should fail");

        assert!(error.to_string().contains("contains no invocation"));
    }

    #[test]
    fn responses_native_turn_preserves_output_items() {
        let mut continuation = native_start(ApiFormat::OpenaiResponses, &[]);
        let (_, calls) = parse_native_response(
            ApiFormat::OpenaiResponses,
            &json!({"output":[
                {"type":"reasoning","id":"reason_1","summary":[]},
                {"type":"function_call","id":"fc_1","call_id":"call_1","name":"list_files","arguments":"{}"}
            ]}),
            &mut continuation,
        )
        .expect("parse native response");
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(continuation.history[0]["type"], "reasoning");
        assert_eq!(continuation.history[1]["id"], "fc_1");
    }

    #[test]
    fn anthropic_native_turn_places_error_result_first() {
        let mut continuation = native_start(ApiFormat::AnthropicMessages, &[]);
        parse_native_response(
            ApiFormat::AnthropicMessages,
            &json!({"content":[{"type":"tool_use","id":"toolu_1","name":"list_files","input":{}}]}),
            &mut continuation,
        )
        .expect("parse native response");
        append_native_results(
            &mut continuation,
            &[ModelToolResult {
                id: "toolu_1".to_owned(),
                name: "list_files".to_owned(),
                output: "denied".to_owned(),
                is_error: true,
            }],
        )
        .expect("append result");
        let last = continuation.history.last().expect("result message");
        assert_eq!(last["content"][0]["type"], "tool_result");
        assert_eq!(last["content"][0]["is_error"], true);
    }

    #[test]
    fn gemini_native_turn_preserves_content_and_maps_missing_id() {
        let mut continuation = native_start(ApiFormat::GeminiGenerateContent, &[]);
        let (_, calls) = parse_native_response(
            ApiFormat::GeminiGenerateContent,
            &json!({"candidates":[{"content":{"role":"model","parts":[
                {"thoughtSignature":"opaque"},
                {"functionCall":{"name":"list_files","args":{}}}
            ]}}]}),
            &mut continuation,
        )
        .expect("parse native response");
        assert_eq!(calls[0].id, "gemini_call_1");
        assert_eq!(
            continuation.history[0]["parts"][0]["thoughtSignature"],
            "opaque"
        );
        append_native_results(
            &mut continuation,
            &[ModelToolResult {
                id: calls[0].id.clone(),
                name: calls[0].name.clone(),
                output: "ok".to_owned(),
                is_error: false,
            }],
        )
        .expect("append result");
        assert!(continuation.history[1]["parts"][0]["functionResponse"]["id"].is_null());
    }

    #[test]
    fn native_fallback_classifier_is_narrow() {
        assert!(native_tools_unsupported(400, "unknown parameter: tools"));
        assert!(!native_tools_unsupported(401, "tools unauthorized"));
        assert!(!native_tools_unsupported(400, "invalid api key"));
        assert!(!native_tools_unsupported(500, "tools unsupported"));
    }

    #[test]
    fn every_protocol_maps_the_shared_tool_schema_and_forced_choice() {
        let chat = native_request_body(
            ApiFormat::OpenaiChat,
            "model",
            &native_start(ApiFormat::OpenaiChat, &[]),
            &[tool()],
            &ToolChoice::Specific("list_files".to_owned()),
        );
        assert_eq!(chat["tools"][0]["function"]["name"], "list_files");
        assert_eq!(chat["tool_choice"]["function"]["name"], "list_files");

        let responses = native_request_body(
            ApiFormat::OpenaiResponses,
            "model",
            &native_start(ApiFormat::OpenaiResponses, &[]),
            &[tool()],
            &ToolChoice::Specific("list_files".to_owned()),
        );
        assert_eq!(responses["tools"][0]["name"], "list_files");
        assert_eq!(responses["tool_choice"]["name"], "list_files");

        let anthropic = native_request_body(
            ApiFormat::AnthropicMessages,
            "model",
            &native_start(ApiFormat::AnthropicMessages, &[]),
            &[tool()],
            &ToolChoice::Specific("list_files".to_owned()),
        );
        assert_eq!(anthropic["tools"][0]["name"], "list_files");
        assert_eq!(anthropic["tool_choice"]["type"], "tool");

        let gemini = native_request_body(
            ApiFormat::GeminiGenerateContent,
            "model",
            &native_start(ApiFormat::GeminiGenerateContent, &[]),
            &[tool()],
            &ToolChoice::Specific("list_files".to_owned()),
        );
        assert_eq!(
            gemini["tools"][0]["functionDeclarations"][0]["name"],
            "list_files"
        );
        assert_eq!(
            gemini["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"][0],
            "list_files"
        );
    }
}
