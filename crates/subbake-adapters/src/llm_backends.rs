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
    LlmBackend,
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
pub struct ProtocolAdapter {
    config: BackendConfig,
    format: ApiFormat,
    key: String,
    client: Client,
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
impl LlmBackend for ProtocolAdapter {
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
    fn generate_cancellable(
        &mut self,
        request: GenerationRequest,
        c: &CancellationGuard,
    ) -> CoreResult<GenerationResponse> {
        runtime().block_on(self.run(&request.messages, c))
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
        .map(|x| TranslationLine {
            id: x["id"].as_str().unwrap_or_default().to_owned(),
            translation: x["translation"].as_str().unwrap_or_default().to_owned(),
        })
        .collect();
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
