use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reqwest::{Client, RequestBuilder};
use serde_json::{Value, json};
use subbake_core::CancellationGuard;
use subbake_core::entities::Usage;
#[cfg(test)]
use subbake_core::entities::{BatchTranslationResult, GlossaryEntry, TranslationLine};
use subbake_core::error::{CoreError, CoreResult, LlmCallError};
use subbake_core::ports::{
    BatchExecutionOptions, ChatMessage, GenerationContent, GenerationInput, GenerationRequest,
    GenerationResponse, LlmBackend, ModelToolCall, ModelToolResult, NativeToolSupport,
    ResponseContract, ToolChoice, ToolContinuation, ToolDefinition,
};
use tokio::runtime::Runtime;

use crate::error::{AdapterError, AdapterResult};
use crate::providers::{ApiFormat, BackendConfig};
const TIMEOUT: f64 = 120.0;
const ANTHROPIC_VERSION: &str = "2023-06-01";
const MAX_TRANSPORT_RETRIES: usize = 2;
static BACKEND_SEQUENCE: AtomicU64 = AtomicU64::new(1);
fn client(timeout: f64) -> AdapterResult<Client> {
    Client::builder()
        .timeout(Duration::from_secs_f64(timeout.max(1.0)))
        .build()
        .map_err(|error| AdapterError::from_http("LLM HTTP client", error))
}
async fn await_http<F, T>(
    future: F,
    cancellation: &CancellationGuard,
    context: &str,
) -> Result<T, LlmCallError>
where
    F: Future<Output = Result<T, reqwest::Error>>,
{
    cancellation.check().map_err(LlmCallError::from)?;
    tokio::pin!(future);
    let mut tick = tokio::time::interval(Duration::from_millis(25));
    loop {
        tokio::select! {
            out=&mut future => {
                return out.map_err(|error| {
                    if error.is_timeout() {
                        LlmCallError::Timeout(format!("{context}: {error}"))
                    } else {
                        LlmCallError::Transport(format!("{context}: {error}"))
                    }
                });
            },
            _=tick.tick()=>cancellation.check().map_err(LlmCallError::from)?,
        }
    }
}

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

#[derive(Debug)]
struct ProtocolContinuation {
    format: ApiFormat,
    system: Option<String>,
    history: Vec<Value>,
    call_ids: HashMap<String, Option<String>>,
}
pub type OpenAiChatAdapter = ProtocolAdapter;
pub type OpenAiResponsesAdapter = ProtocolAdapter;
pub type AnthropicMessagesAdapter = ProtocolAdapter;
pub type GeminiGenerateContentAdapter = ProtocolAdapter;

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
        if let Some(url) = &self.config.endpoint_url {
            return url.trim().to_owned();
        }
        let base = self
            .config
            .base_url
            .as_deref()
            .unwrap_or(match self.format {
                ApiFormat::OpenaiChat | ApiFormat::OpenaiResponses => "https://api.openai.com/v1",
                ApiFormat::AnthropicMessages => "https://api.anthropic.com/v1",
                ApiFormat::GeminiGenerateContent => {
                    "https://generativelanguage.googleapis.com/v1beta"
                }
            })
            .trim_end_matches('/');
        match self.format {
            ApiFormat::OpenaiChat => format!("{base}/chat/completions"),
            ApiFormat::OpenaiResponses => format!("{base}/responses"),
            ApiFormat::AnthropicMessages => format!("{base}/messages"),
            ApiFormat::GeminiGenerateContent => {
                format!("{base}/models/{}:generateContent", self.config.model)
            }
        }
    }
    fn authenticated(&self, request: RequestBuilder) -> RequestBuilder {
        if let Some(header) = &self.config.auth_header {
            let value = format!(
                "{}{}",
                self.config.auth_prefix.as_deref().unwrap_or(""),
                self.key
            );
            return request.header(header, value);
        }
        match self.format {
            ApiFormat::OpenaiChat | ApiFormat::OpenaiResponses => request.bearer_auth(&self.key),
            ApiFormat::AnthropicMessages => request
                .header("x-api-key", &self.key)
                .header("anthropic-version", ANTHROPIC_VERSION),
            ApiFormat::GeminiGenerateContent => request.header("x-goog-api-key", &self.key),
        }
    }
    fn body(&self, messages: &[ChatMessage], contract: ResponseContract) -> Value {
        let mut body = match self.format {
            ApiFormat::OpenaiChat => {
                json!({"model":self.config.model,"messages":messages.iter().map(openai_message).collect::<Vec<_>>() })
            }
            ApiFormat::OpenaiResponses => {
                json!({"model":self.config.model,"input":messages.iter().map(|m| json!({"role":m.role,"content":[{"type":"input_text","text":m.content}]})).collect::<Vec<_>>() })
            }
            ApiFormat::AnthropicMessages => {
                let system = messages
                    .iter()
                    .filter(|m| m.role == "system")
                    .map(|m| m.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                json!({"model":self.config.model,"max_tokens":4096,"system":system,"messages":messages.iter().filter(|m|m.role!="system").map(|m|json!({"role":m.role,"content":[{"type":"text","text":m.content}]})).collect::<Vec<_>>()})
            }
            ApiFormat::GeminiGenerateContent => {
                let system = messages
                    .iter()
                    .filter(|m| m.role == "system")
                    .map(|m| m.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                json!({"systemInstruction":{"parts":[{"text":system}]},"contents":messages.iter().filter(|m|m.role!="system").map(|m|json!({"role":if m.role=="assistant" {"model"} else {"user"},"parts":[{"text":m.content}]})).collect::<Vec<_>>()})
            }
        };
        apply_response_contract(self.format, contract, &mut body);
        body
    }

    async fn send_payload(
        &self,
        payload: &Value,
        cancellation: &CancellationGuard,
        native_tools: bool,
    ) -> Result<Value, LlmCallError> {
        let mut attempt = 0;
        loop {
            cancellation.check().map_err(LlmCallError::from)?;
            let response = await_http(
                self.authenticated(self.client.post(self.endpoint()).json(payload))
                    .send(),
                cancellation,
                "provider request failed",
            )
            .await;
            let result = match response {
                Ok(response) => {
                    let status = response.status();
                    let retry_after_ms = response
                        .headers()
                        .get(reqwest::header::RETRY_AFTER)
                        .and_then(|value| value.to_str().ok())
                        .and_then(|value| value.parse::<u64>().ok())
                        .map(|seconds| seconds.saturating_mul(1_000));
                    let text = await_http(
                        response.text(),
                        cancellation,
                        "provider response read failed",
                    )
                    .await?;
                    if status.is_success() {
                        serde_json::from_str(&text).map_err(|error| {
                            LlmCallError::InvalidResponse(format!(
                                "{} response decode failed: {error}",
                                self.config.display_name
                            ))
                        })
                    } else if native_tools && native_tools_unsupported(status.as_u16(), &text) {
                        Err(LlmCallError::UnsupportedCapability(format!(
                            "native tools: {text}"
                        )))
                    } else {
                        Err(classify_status_error(status.as_u16(), text, retry_after_ms))
                    }
                }
                Err(error) => Err(error),
            };

            match result {
                Err(error) if error.is_retryable() && attempt < MAX_TRANSPORT_RETRIES => {
                    attempt += 1;
                    let delay_ms = match &error {
                        LlmCallError::RateLimited {
                            retry_after_ms: Some(delay),
                            ..
                        } => (*delay).min(5_000),
                        _ => 100 * (1_u64 << (attempt - 1)),
                    };
                    wait_retry(Duration::from_millis(delay_ms), cancellation).await?;
                }
                other => return other,
            }
        }
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
        let text = match self.format {
            ApiFormat::OpenaiChat => body["choices"][0]["message"]["content"]
                .as_str()
                .map(str::to_owned),
            ApiFormat::OpenaiResponses => {
                body["output_text"].as_str().map(str::to_owned).or_else(|| {
                    body["output"].as_array().map(|out| {
                        out.iter()
                            .flat_map(|o| o["content"].as_array().into_iter().flatten())
                            .filter_map(|c| c["text"].as_str())
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                })
            }
            ApiFormat::AnthropicMessages => body["content"].as_array().map(|a| {
                a.iter()
                    .filter_map(|v| v["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            }),
            ApiFormat::GeminiGenerateContent => body["candidates"][0]["content"]["parts"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v["text"].as_str())
                        .collect::<Vec<_>>()
                        .join("\n")
                }),
        };
        text.filter(|s| !s.is_empty()).ok_or_else(|| {
            CoreError::InvalidBackendResponse(format!(
                "{} response missing text output",
                self.config.display_name
            ))
        })
    }
}

fn native_start(format: ApiFormat, messages: &[ChatMessage]) -> ProtocolContinuation {
    let (system, history) = match format {
        ApiFormat::OpenaiChat => (
            None,
            messages.iter().map(openai_message).collect::<Vec<_>>(),
        ),
        ApiFormat::OpenaiResponses => (
            None,
            messages
                .iter()
                .map(|message| {
                    json!({"role":message.role,"content":[{"type":"input_text","text":message.content}]})
                })
                .collect(),
        ),
        ApiFormat::AnthropicMessages => (
            Some(
                messages
                    .iter()
                    .filter(|message| message.role == "system")
                    .map(|message| message.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n"),
            ),
            messages
                .iter()
                .filter(|message| message.role != "system")
                .map(|message| {
                    json!({"role":message.role,"content":[{"type":"text","text":message.content}]})
                })
                .collect(),
        ),
        ApiFormat::GeminiGenerateContent => (
            Some(
                messages
                    .iter()
                    .filter(|message| message.role == "system")
                    .map(|message| message.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n"),
            ),
            messages
                .iter()
                .filter(|message| message.role != "system")
                .map(|message| {
                    json!({"role":if message.role == "assistant" {"model"} else {"user"},"parts":[{"text":message.content}]})
                })
                .collect(),
        ),
    };
    ProtocolContinuation {
        format,
        system,
        history,
        call_ids: HashMap::new(),
    }
}

fn append_native_results(
    continuation: &mut ProtocolContinuation,
    results: &[ModelToolResult],
) -> CoreResult<()> {
    for result in results {
        if !continuation.call_ids.contains_key(&result.id) {
            return Err(CoreError::DataInvariant(format!(
                "native tool result references unknown call `{}`",
                result.id
            )));
        }
    }
    match continuation.format {
        ApiFormat::OpenaiChat => {
            for result in results {
                let wire_id = continuation
                    .call_ids
                    .get(&result.id)
                    .and_then(|value| value.as_deref())
                    .unwrap_or(&result.id);
                continuation.history.push(json!({
                    "role":"tool",
                    "tool_call_id":wire_id,
                    "content":result.output,
                }));
            }
        }
        ApiFormat::OpenaiResponses => {
            for result in results {
                let wire_id = continuation
                    .call_ids
                    .get(&result.id)
                    .and_then(|value| value.as_deref())
                    .unwrap_or(&result.id);
                continuation.history.push(json!({
                    "type":"function_call_output",
                    "call_id":wire_id,
                    "output":result.output,
                }));
            }
        }
        ApiFormat::AnthropicMessages => {
            let content = results
                .iter()
                .map(|result| {
                    let wire_id = continuation
                        .call_ids
                        .get(&result.id)
                        .and_then(|value| value.as_deref())
                        .unwrap_or(&result.id);
                    json!({
                        "type":"tool_result",
                        "tool_use_id":wire_id,
                        "content":result.output,
                        "is_error":result.is_error,
                    })
                })
                .collect::<Vec<_>>();
            continuation
                .history
                .push(json!({"role":"user","content":content}));
        }
        ApiFormat::GeminiGenerateContent => {
            let parts = results
                .iter()
                .map(|result| {
                    let wire_id = continuation
                        .call_ids
                        .get(&result.id)
                        .and_then(|value| value.as_deref());
                    let response = if result.is_error {
                        json!({"error":result.output})
                    } else {
                        json!({"result":result.output})
                    };
                    let mut function_response = json!({
                        "name":result.name,
                        "response":response,
                    });
                    if let Some(id) = wire_id {
                        function_response["id"] = Value::String(id.to_owned());
                    }
                    json!({"functionResponse":function_response})
                })
                .collect::<Vec<_>>();
            continuation
                .history
                .push(json!({"role":"user","parts":parts}));
        }
    }
    Ok(())
}

fn native_request_body(
    format: ApiFormat,
    model: &str,
    continuation: &ProtocolContinuation,
    tools: &[ToolDefinition],
    choice: &ToolChoice,
) -> Value {
    match format {
        ApiFormat::OpenaiChat => json!({
            "model":model,
            "messages":continuation.history,
            "tools":tools.iter().map(openai_chat_tool).collect::<Vec<_>>(),
            "tool_choice":openai_chat_tool_choice(choice),
            "parallel_tool_calls":false,
        }),
        ApiFormat::OpenaiResponses => json!({
            "model":model,
            "input":continuation.history,
            "tools":tools.iter().map(openai_responses_tool).collect::<Vec<_>>(),
            "tool_choice":openai_responses_tool_choice(choice),
            "parallel_tool_calls":false,
        }),
        ApiFormat::AnthropicMessages => json!({
            "model":model,
            "max_tokens":4096,
            "system":continuation.system.as_deref().unwrap_or(""),
            "messages":continuation.history,
            "tools":tools.iter().map(anthropic_tool).collect::<Vec<_>>(),
            "tool_choice":anthropic_tool_choice(choice),
        }),
        ApiFormat::GeminiGenerateContent => json!({
            "systemInstruction":{"parts":[{"text":continuation.system.as_deref().unwrap_or("")}]},
            "contents":continuation.history,
            "tools":[{"functionDeclarations":tools.iter().map(gemini_tool).collect::<Vec<_>>()}],
            "toolConfig":{"functionCallingConfig":gemini_tool_choice(choice)},
        }),
    }
}

fn openai_chat_tool(tool: &ToolDefinition) -> Value {
    json!({"type":"function","function":{"name":tool.name,"description":tool.description,"parameters":tool.input_schema}})
}

fn openai_responses_tool(tool: &ToolDefinition) -> Value {
    json!({"type":"function","name":tool.name,"description":tool.description,"parameters":tool.input_schema})
}

fn anthropic_tool(tool: &ToolDefinition) -> Value {
    json!({"name":tool.name,"description":tool.description,"input_schema":tool.input_schema})
}

fn gemini_tool(tool: &ToolDefinition) -> Value {
    json!({"name":tool.name,"description":tool.description,"parameters":tool.input_schema})
}

fn openai_chat_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Specific(name) => json!({"type":"function","function":{"name":name}}),
        ToolChoice::None => json!("none"),
    }
}

fn openai_responses_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Specific(name) => json!({"type":"function","name":name}),
        ToolChoice::None => json!("none"),
    }
}

fn anthropic_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({"type":"auto"}),
        ToolChoice::Required => json!({"type":"any"}),
        ToolChoice::Specific(name) => json!({"type":"tool","name":name}),
        ToolChoice::None => json!({"type":"none"}),
    }
}

fn gemini_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({"mode":"AUTO"}),
        ToolChoice::Required => json!({"mode":"ANY"}),
        ToolChoice::Specific(name) => json!({"mode":"ANY","allowedFunctionNames":[name]}),
        ToolChoice::None => json!({"mode":"NONE"}),
    }
}

fn parse_native_response(
    format: ApiFormat,
    body: &Value,
    continuation: &mut ProtocolContinuation,
) -> CoreResult<(Option<String>, Vec<ModelToolCall>)> {
    let mut calls = Vec::new();
    let text = match format {
        ApiFormat::OpenaiChat => {
            let mut message = body["choices"][0]["message"].clone();
            for (index, call) in message["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
                .enumerate()
            {
                let id = call["id"]
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("openai_chat_call_{index}"));
                continuation.call_ids.insert(id.clone(), Some(id.clone()));
                calls.push(ModelToolCall {
                    id,
                    name: required_wire_string(call, &["function", "name"])?.to_owned(),
                    arguments: parse_wire_arguments(&call["function"]["arguments"])?,
                });
            }
            let mut text = message["content"].as_str().map(str::to_owned);
            if calls.is_empty()
                && let Some(content) = text.as_deref()
                && let Some(dsml_calls) = parse_dsml_tool_calls(content)?
            {
                for call in &dsml_calls {
                    continuation
                        .call_ids
                        .insert(call.id.clone(), Some(call.id.clone()));
                }
                calls = dsml_calls;
                text = None;
                message["content"] = Value::Null;
                message["tool_calls"] = Value::Array(
                    calls
                        .iter()
                        .map(|call| {
                            json!({
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments.to_string(),
                                }
                            })
                        })
                        .collect(),
                );
            }
            continuation.history.push(message.clone());
            text
        }
        ApiFormat::OpenaiResponses => {
            let output = body["output"].as_array().ok_or_else(|| {
                CoreError::InvalidBackendResponse("OpenAI Responses output is missing".to_owned())
            })?;
            for (index, item) in output.iter().enumerate() {
                continuation.history.push(item.clone());
                if item["type"].as_str() != Some("function_call") {
                    continue;
                }
                let wire_id = item["call_id"].as_str().ok_or_else(|| {
                    CoreError::InvalidBackendResponse("function call is missing call_id".to_owned())
                })?;
                let id = wire_id.to_owned();
                continuation
                    .call_ids
                    .insert(id.clone(), Some(wire_id.to_owned()));
                calls.push(ModelToolCall {
                    id,
                    name: item["name"]
                        .as_str()
                        .ok_or_else(|| {
                            CoreError::InvalidBackendResponse(format!(
                                "function call {} is missing name",
                                index + 1
                            ))
                        })?
                        .to_owned(),
                    arguments: parse_wire_arguments(&item["arguments"])?,
                });
            }
            responses_text(body)
        }
        ApiFormat::AnthropicMessages => {
            let content = body["content"].as_array().ok_or_else(|| {
                CoreError::InvalidBackendResponse(
                    "Anthropic response content is missing".to_owned(),
                )
            })?;
            let mut texts = Vec::new();
            for (index, block) in content.iter().enumerate() {
                match block["type"].as_str() {
                    Some("text") => {
                        if let Some(value) = block["text"].as_str() {
                            texts.push(value.to_owned());
                        }
                    }
                    Some("tool_use") => {
                        let id = block["id"]
                            .as_str()
                            .ok_or_else(|| {
                                CoreError::InvalidBackendResponse(format!(
                                    "tool use {} is missing id",
                                    index + 1
                                ))
                            })?
                            .to_owned();
                        continuation.call_ids.insert(id.clone(), Some(id.clone()));
                        calls.push(ModelToolCall {
                            id,
                            name: block["name"]
                                .as_str()
                                .ok_or_else(|| {
                                    CoreError::InvalidBackendResponse(format!(
                                        "tool use {} is missing name",
                                        index + 1
                                    ))
                                })?
                                .to_owned(),
                            arguments: block["input"].clone(),
                        });
                    }
                    _ => {}
                }
            }
            continuation
                .history
                .push(json!({"role":"assistant","content":content}));
            (!texts.is_empty()).then(|| texts.join("\n"))
        }
        ApiFormat::GeminiGenerateContent => {
            let content = body["candidates"][0]["content"].clone();
            let parts = content["parts"].as_array().ok_or_else(|| {
                CoreError::InvalidBackendResponse("Gemini response parts are missing".to_owned())
            })?;
            let mut texts = Vec::new();
            for (index, part) in parts.iter().enumerate() {
                if let Some(value) = part["text"].as_str() {
                    texts.push(value.to_owned());
                }
                let Some(function_call) = part.get("functionCall") else {
                    continue;
                };
                let wire_id = function_call["id"].as_str().map(str::to_owned);
                let id = wire_id
                    .clone()
                    .unwrap_or_else(|| format!("gemini_call_{index}"));
                continuation.call_ids.insert(id.clone(), wire_id);
                calls.push(ModelToolCall {
                    id,
                    name: function_call["name"]
                        .as_str()
                        .ok_or_else(|| {
                            CoreError::InvalidBackendResponse(format!(
                                "function call {} is missing name",
                                index + 1
                            ))
                        })?
                        .to_owned(),
                    arguments: function_call["args"].clone(),
                });
            }
            continuation.history.push(content);
            (!texts.is_empty()).then(|| texts.join("\n"))
        }
    };
    Ok((text.filter(|value| !value.is_empty()), calls))
}

fn parse_dsml_tool_calls(content: &str) -> CoreResult<Option<Vec<ModelToolCall>>> {
    const PREFIXES: [&str; 2] = ["｜｜DSML｜｜", "||DSML||"];
    let Some(prefix) = PREFIXES
        .iter()
        .find(|prefix| content.contains(&format!("<{prefix}tool_calls>")))
    else {
        return Ok(None);
    };
    let invoke_open = format!("<{prefix}invoke ");
    let invoke_close = format!("</{prefix}invoke>");
    let parameter_open = format!("<{prefix}parameter ");
    let parameter_close = format!("</{prefix}parameter>");
    let mut remainder = content;
    let mut calls = Vec::new();
    while let Some(start) = remainder.find(&invoke_open) {
        remainder = &remainder[start + invoke_open.len()..];
        let header_end = remainder.find('>').ok_or_else(|| {
            CoreError::InvalidBackendResponse(
                "DSML tool invocation has an unterminated header".to_owned(),
            )
        })?;
        let name = dsml_attribute(&remainder[..header_end], "name").ok_or_else(|| {
            CoreError::InvalidBackendResponse("DSML tool invocation is missing name".to_owned())
        })?;
        remainder = &remainder[header_end + 1..];
        let body_end = remainder.find(&invoke_close).ok_or_else(|| {
            CoreError::InvalidBackendResponse(format!(
                "DSML tool invocation `{name}` is unterminated"
            ))
        })?;
        let body = &remainder[..body_end];
        let mut parameters = serde_json::Map::new();
        let mut parameter_remainder = body;
        while let Some(parameter_start) = parameter_remainder.find(&parameter_open) {
            parameter_remainder = &parameter_remainder[parameter_start + parameter_open.len()..];
            let parameter_header_end = parameter_remainder.find('>').ok_or_else(|| {
                CoreError::InvalidBackendResponse(format!(
                    "DSML parameter for `{name}` has an unterminated header"
                ))
            })?;
            let header = &parameter_remainder[..parameter_header_end];
            let parameter_name = dsml_attribute(header, "name").ok_or_else(|| {
                CoreError::InvalidBackendResponse(format!(
                    "DSML parameter for `{name}` is missing name"
                ))
            })?;
            let is_string = dsml_attribute(header, "string").as_deref() == Some("true");
            parameter_remainder = &parameter_remainder[parameter_header_end + 1..];
            let value_end = parameter_remainder.find(&parameter_close).ok_or_else(|| {
                CoreError::InvalidBackendResponse(format!(
                    "DSML parameter `{parameter_name}` is unterminated"
                ))
            })?;
            let raw_value = decode_dsml_entities(parameter_remainder[..value_end].trim());
            let value = if is_string {
                Value::String(raw_value)
            } else {
                serde_json::from_str(&raw_value).unwrap_or(Value::String(raw_value))
            };
            let parameter_name = if name == "translate_file" && parameter_name == "file_path" {
                "path".to_owned()
            } else {
                parameter_name
            };
            parameters.insert(parameter_name, value);
            parameter_remainder = &parameter_remainder[value_end + parameter_close.len()..];
        }
        let index = calls.len() + 1;
        calls.push(ModelToolCall {
            id: format!("dsml_call_{index}"),
            name,
            arguments: Value::Object(parameters),
        });
        remainder = &remainder[body_end + invoke_close.len()..];
    }
    if calls.is_empty() {
        return Err(CoreError::InvalidBackendResponse(
            "DSML tool_calls block contains no invocation".to_owned(),
        ));
    }
    Ok(Some(calls))
}

fn dsml_attribute(header: &str, name: &str) -> Option<String> {
    let marker = format!(r#"{name}="#);
    let value = header.split_once(&marker)?.1.trim_start();
    let quote = value.chars().next()?;
    if !matches!(quote, '\'' | '"') {
        return None;
    }
    let value = &value[quote.len_utf8()..];
    let end = value.find(quote)?;
    Some(decode_dsml_entities(&value[..end]))
}

fn decode_dsml_entities(value: &str) -> String {
    value
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

fn required_wire_string<'a>(value: &'a Value, path: &[&str]) -> CoreResult<&'a str> {
    let mut current = value;
    for part in path {
        current = &current[*part];
    }
    current.as_str().ok_or_else(|| {
        CoreError::InvalidBackendResponse(format!("native tool call is missing {}", path.join(".")))
    })
}

fn parse_wire_arguments(value: &Value) -> CoreResult<Value> {
    if let Some(text) = value.as_str() {
        serde_json::from_str(text).map_err(|error| {
            CoreError::InvalidBackendResponse(format!("invalid native tool arguments: {error}"))
        })
    } else {
        Ok(value.clone())
    }
}

fn responses_text(body: &Value) -> Option<String> {
    body["output_text"].as_str().map(str::to_owned).or_else(|| {
        let parts = body["output"]
            .as_array()?
            .iter()
            .flat_map(|item| item["content"].as_array().into_iter().flatten())
            .filter_map(|content| content["text"].as_str())
            .collect::<Vec<_>>();
        (!parts.is_empty()).then(|| parts.join("\n"))
    })
}

async fn wait_retry(
    duration: Duration,
    cancellation: &CancellationGuard,
) -> Result<(), LlmCallError> {
    let sleep = tokio::time::sleep(duration);
    tokio::pin!(sleep);
    let mut tick = tokio::time::interval(Duration::from_millis(25));
    loop {
        tokio::select! {
            _ = &mut sleep => return Ok(()),
            _ = tick.tick() => cancellation.check().map_err(LlmCallError::from)?,
        }
    }
}

fn classify_status_error(
    status: u16,
    message: String,
    retry_after_ms: Option<u64>,
) -> LlmCallError {
    match status {
        401 | 403 => LlmCallError::Authentication(message),
        429 => LlmCallError::RateLimited {
            message,
            retry_after_ms,
        },
        _ => LlmCallError::Rejected {
            status: Some(status),
            message,
        },
    }
}

fn apply_response_contract(format: ApiFormat, contract: ResponseContract, body: &mut Value) {
    if contract != ResponseContract::JsonObject {
        return;
    }
    match format {
        ApiFormat::OpenaiChat => {
            body["response_format"] = json!({"type": "json_object"});
        }
        ApiFormat::OpenaiResponses => {
            body["text"] = json!({"format": {"type": "json_object"}});
        }
        ApiFormat::AnthropicMessages | ApiFormat::GeminiGenerateContent => {}
    }
}

fn native_tools_unsupported(status: u16, body: &str) -> bool {
    if !matches!(status, 400 | 404 | 422) {
        return false;
    }
    let body = body.to_lowercase();
    let mentions_tools = [
        "tools",
        "tool_choice",
        "function_declarations",
        "function calling",
    ]
    .iter()
    .any(|needle| body.contains(needle));
    let rejects_feature = [
        "unsupported",
        "unknown field",
        "unknown parameter",
        "unrecognized",
        "not supported",
        "not allowed",
    ]
    .iter()
    .any(|needle| body.contains(needle));
    mentions_tools && rejects_feature
}

impl LlmBackend for ProtocolAdapter {
    fn supports_terminology_preflight(&self) -> bool {
        true
    }

    fn supports_parallel_generation(&self) -> bool {
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
fn openai_message(m: &ChatMessage) -> Value {
    json!({"role":m.role,"content":m.content})
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
fn usage(format: ApiFormat, b: &Value, text: &str, p: &Value) -> Usage {
    let u = &b["usage"];
    let input = match format {
        ApiFormat::AnthropicMessages => u["input_tokens"].as_u64(),
        ApiFormat::GeminiGenerateContent => b["usageMetadata"]["promptTokenCount"].as_u64(),
        _ => u["prompt_tokens"]
            .as_u64()
            .or_else(|| u["input_tokens"].as_u64()),
    }
    .map(|n| n as usize)
    .unwrap_or_else(|| estimate(&p.to_string()));
    let output = match format {
        ApiFormat::AnthropicMessages => u["output_tokens"].as_u64(),
        ApiFormat::GeminiGenerateContent => b["usageMetadata"]["candidatesTokenCount"].as_u64(),
        _ => u["completion_tokens"]
            .as_u64()
            .or_else(|| u["output_tokens"].as_u64()),
    }
    .map(|n| n as usize)
    .unwrap_or_else(|| estimate(text));
    let total = u["total_tokens"]
        .as_u64()
        .or_else(|| b["usageMetadata"]["totalTokenCount"].as_u64())
        .map(|n| n as usize)
        .unwrap_or(input + output);
    Usage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: total,
    }
}
fn estimate(s: &str) -> usize {
    s.chars().count().div_ceil(4).max(1)
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
