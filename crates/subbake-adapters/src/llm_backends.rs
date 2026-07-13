use std::collections::HashMap;
use std::future::Future;
use std::sync::OnceLock;
use std::time::Duration;

use reqwest::{Client, RequestBuilder};
use serde_json::{Value, json};
use subbake_core::CancellationGuard;
use subbake_core::entities::{BatchTranslationResult, GlossaryEntry, TranslationLine, Usage};
use subbake_core::error::{CoreError, CoreResult};
use subbake_core::ports::{
    BackendJsonResult, BackendPayload, ChatMessage, GenerationRequest, GenerationResponse,
    LlmBackend, ModelToolCall, ModelToolResult, NativeToolSupport, ToolChoice, ToolContinuation,
    ToolDefinition, ToolGenerationInput, ToolGenerationRequest, ToolGenerationResponse,
};
use tokio::runtime::Runtime;

use crate::providers::{ApiFormat, BackendConfig};
const TIMEOUT: f64 = 120.0;
const ANTHROPIC_VERSION: &str = "2023-06-01";
fn runtime() -> &'static Runtime {
    static R: OnceLock<Runtime> = OnceLock::new();
    R.get_or_init(|| Runtime::new().expect("tokio runtime"))
}
fn client(timeout: f64) -> CoreResult<Client> {
    Client::builder()
        .timeout(Duration::from_secs_f64(timeout.max(1.0)))
        .build()
        .map_err(|e| CoreError::Backend(format!("http client build failed: {e}")))
}
async fn await_http<F, T>(
    future: F,
    cancellation: &CancellationGuard,
    context: &str,
) -> CoreResult<T>
where
    F: Future<Output = Result<T, reqwest::Error>>,
{
    cancellation.check()?;
    tokio::pin!(future);
    let mut tick = tokio::time::interval(Duration::from_millis(25));
    loop {
        tokio::select! { out=&mut future => return out.map_err(|e| CoreError::Backend(format!("{context}: {e}"))), _=tick.tick()=>cancellation.check()? }
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
) -> CoreResult<Box<dyn LlmBackend>> {
    let format = config
        .api_format
        .ok_or_else(|| CoreError::Backend("api_format is required".to_owned()))?;
    let key = config.resolved_api_key().ok_or_else(|| {
        CoreError::Backend(format!(
            "Missing API key for {} provider. Set --api-key, api_key, or api_key_env.",
            config.display_name
        ))
    })?;
    if key.contains(['\r', '\n']) {
        return Err(CoreError::Backend(
            "authentication header value must not contain CR/LF".to_owned(),
        ));
    }
    Ok(Box::new(ProtocolAdapter {
        config: config.clone(),
        format,
        key,
        client: client(timeout)?,
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
    fn body(&self, messages: &[ChatMessage]) -> Value {
        match self.format {
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
        }
    }

    fn native_payload(
        &self,
        request: ToolGenerationRequest,
    ) -> CoreResult<(Value, ProtocolContinuation)> {
        let tools = request.tools;
        let choice = request.tool_choice;
        let mut continuation = match request.input {
            ToolGenerationInput::Start { messages } => native_start(self.format, &messages),
            ToolGenerationInput::Continue {
                continuation,
                results,
            } => {
                let mut continuation =
                    continuation
                        .downcast::<ProtocolContinuation>()
                        .map_err(|_| {
                            CoreError::Data(
                                "native tool continuation belongs to a different backend"
                                    .to_owned(),
                            )
                        })?;
                if continuation.format != self.format {
                    return Err(CoreError::Data(
                        "native tool continuation protocol changed".to_owned(),
                    ));
                }
                append_native_results(&mut continuation, &results)?;
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
        Ok((payload, continuation))
    }

    async fn run_native(
        &self,
        request: ToolGenerationRequest,
        cancel: &CancellationGuard,
    ) -> CoreResult<ToolGenerationResponse> {
        let (payload, mut continuation) = self.native_payload(request)?;
        let url = self.endpoint();
        let response = await_http(
            self.authenticated(self.client.post(url).json(&payload))
                .send(),
            cancel,
            "provider request failed",
        )
        .await?;
        let status = response.status();
        let text = await_http(response.text(), cancel, "provider response read failed").await?;
        if !status.is_success() {
            if native_tools_unsupported(status.as_u16(), &text) {
                return Err(CoreError::UnsupportedCapability(format!(
                    "native tools: {text}"
                )));
            }
            return Err(CoreError::Backend(format!(
                "{} rejected request ({status}): {text}",
                self.config.display_name
            )));
        }
        let body: Value = serde_json::from_str(&text).map_err(|error| {
            CoreError::Backend(format!(
                "{} response decode failed: {error}",
                self.config.display_name
            ))
        })?;
        let (response_text, tool_calls) =
            parse_native_response(self.format, &body, &mut continuation)?;
        let usage = usage(
            self.format,
            &body,
            response_text.as_deref().unwrap_or(""),
            &payload,
        );
        let continuation = (!tool_calls.is_empty()).then(|| ToolContinuation::new(continuation));
        Ok(ToolGenerationResponse {
            text: response_text,
            tool_calls,
            continuation,
            usage,
        })
    }

    async fn run(
        &self,
        messages: &[ChatMessage],
        cancel: &CancellationGuard,
    ) -> CoreResult<GenerationResponse> {
        let payload = self.body(messages);
        let url = self.endpoint();
        let response = await_http(
            self.authenticated(self.client.post(url).json(&payload))
                .send(),
            cancel,
            "provider request failed",
        )
        .await?;
        let status = response.status();
        let text = await_http(response.text(), cancel, "provider response read failed").await?;
        if !status.is_success() {
            return Err(CoreError::Backend(format!(
                "{} rejected request ({status}): {text}",
                self.config.display_name
            )));
        }
        let body: Value = serde_json::from_str(&text).map_err(|e| {
            CoreError::Backend(format!(
                "{} response decode failed: {e}",
                self.config.display_name
            ))
        })?;
        let content = self.content(&body)?;
        let json = extract_json_object(&content)?;
        Ok(GenerationResponse {
            usage: usage(self.format, &body, &content, &payload),
            json,
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
            CoreError::Backend(format!(
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
            return Err(CoreError::Data(format!(
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
            let message = body["choices"][0]["message"].clone();
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
            continuation.history.push(message.clone());
            message["content"].as_str().map(str::to_owned)
        }
        ApiFormat::OpenaiResponses => {
            let output = body["output"].as_array().ok_or_else(|| {
                CoreError::Backend("OpenAI Responses output is missing".to_owned())
            })?;
            for (index, item) in output.iter().enumerate() {
                continuation.history.push(item.clone());
                if item["type"].as_str() != Some("function_call") {
                    continue;
                }
                let wire_id = item["call_id"].as_str().ok_or_else(|| {
                    CoreError::Backend("function call is missing call_id".to_owned())
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
                            CoreError::Backend(format!(
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
                CoreError::Backend("Anthropic response content is missing".to_owned())
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
                                CoreError::Backend(format!("tool use {} is missing id", index + 1))
                            })?
                            .to_owned();
                        continuation.call_ids.insert(id.clone(), Some(id.clone()));
                        calls.push(ModelToolCall {
                            id,
                            name: block["name"]
                                .as_str()
                                .ok_or_else(|| {
                                    CoreError::Backend(format!(
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
                CoreError::Backend("Gemini response parts are missing".to_owned())
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
                            CoreError::Backend(format!(
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

fn required_wire_string<'a>(value: &'a Value, path: &[&str]) -> CoreResult<&'a str> {
    let mut current = value;
    for part in path {
        current = &current[*part];
    }
    current.as_str().ok_or_else(|| {
        CoreError::Backend(format!("native tool call is missing {}", path.join(".")))
    })
}

fn parse_wire_arguments(value: &Value) -> CoreResult<Value> {
    if let Some(text) = value.as_str() {
        serde_json::from_str(text)
            .map_err(|error| CoreError::Backend(format!("invalid native tool arguments: {error}")))
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
    fn generate_json(&mut self, m: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
        let r = self.generate_raw_json(m)?;
        Ok(BackendJsonResult {
            payload: BackendPayload::Translation(parse_translation_payload(&r.0)?),
            usage: r.1,
        })
    }
    fn generate_raw_json(&mut self, m: &[ChatMessage]) -> CoreResult<(Value, Usage)> {
        self.generate_raw_json_cancellable(m, &CancellationGuard::never())
    }
    fn generate_raw_json_cancellable(
        &mut self,
        m: &[ChatMessage],
        c: &CancellationGuard,
    ) -> CoreResult<(Value, Usage)> {
        let r = runtime().block_on(self.run(m, c))?;
        Ok((r.json, r.usage))
    }
    fn generate_with_tools_cancellable(
        &mut self,
        request: ToolGenerationRequest,
        cancellation: &CancellationGuard,
    ) -> CoreResult<ToolGenerationResponse> {
        if self.native_tool_support == NativeToolSupport::Unsupported {
            return Err(CoreError::UnsupportedCapability("native tools".to_owned()));
        }
        let result = runtime().block_on(self.run_native(request, cancellation));
        match &result {
            Ok(_) => self.native_tool_support = NativeToolSupport::Supported,
            Err(CoreError::UnsupportedCapability(_)) => {
                self.native_tool_support = NativeToolSupport::Unsupported;
            }
            Err(_) => {}
        }
        result
    }
    fn generate_cancellable(
        &mut self,
        request: GenerationRequest,
        c: &CancellationGuard,
    ) -> CoreResult<GenerationResponse> {
        runtime().block_on(self.run(&request.messages, c))
    }
    fn generate_many_cancellable(
        &mut self,
        requests: Vec<GenerationRequest>,
        max_concurrency: usize,
        cancellation: &CancellationGuard,
    ) -> Vec<CoreResult<GenerationResponse>> {
        let adapter = self.clone();
        let cancellation = cancellation.clone();
        runtime().block_on(async move {
            let semaphore =
                std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrency.max(1)));
            let mut set = tokio::task::JoinSet::new();
            for (index, request) in requests.into_iter().enumerate() {
                let adapter = adapter.clone();
                let cancellation = cancellation.clone();
                let semaphore = semaphore.clone();
                set.spawn(async move {
                    let permit = semaphore.acquire_owned().await.map_err(|error| {
                        CoreError::Backend(format!("concurrency limiter closed: {error}"))
                    })?;
                    let result = adapter.run(&request.messages, &cancellation).await;
                    drop(permit);
                    Ok::<_, CoreError>((index, result))
                });
            }
            let mut ordered = Vec::new();
            while let Some(joined) = set.join_next().await {
                match joined {
                    Ok(Ok(item)) => ordered.push(item),
                    Ok(Err(error)) => ordered.push((usize::MAX, Err(error))),
                    Err(error) => ordered.push((
                        usize::MAX,
                        Err(CoreError::Backend(format!("provider task failed: {error}"))),
                    )),
                }
            }
            ordered.sort_by_key(|(index, _)| *index);
            ordered.into_iter().map(|(_, result)| result).collect()
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
    let start = t
        .find('{')
        .ok_or_else(|| CoreError::Backend("response is missing a JSON object".to_owned()))?;
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
                    return serde_json::from_str(&t[start..i + 1])
                        .map_err(|x| CoreError::Backend(format!("invalid JSON in response: {x}")));
                }
            }
            _ => {}
        }
    }
    Err(CoreError::Backend(
        "response JSON object is unbalanced".to_owned(),
    ))
}
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
    use subbake_core::ports::{ModelToolResult, ToolChoice, ToolDefinition};

    use super::{
        ApiFormat, append_native_results, native_request_body, native_start,
        native_tools_unsupported, parse_native_response, parse_translation_payload,
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
