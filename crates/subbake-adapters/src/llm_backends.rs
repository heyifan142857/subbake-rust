// Real LLM provider backends (OpenAI-compatible, Gemini, Anthropic).
//
// `LlmBackend::generate_json` is synchronous because the translation pipeline
// is synchronous. Each backend issues an async `reqwest` call and drives it to
// completion on a shared tokio runtime via `block_on`. This is safe on the CLI
// translation path (no ambient runtime); the agent path (stage 5) will reach
// the pipeline through `spawn_blocking` so a runtime is never nested.

use std::sync::OnceLock;
use std::time::Duration;

use reqwest::Client;
use serde_json::{Value as JsonValue, json};
use subbake_core::entities::{BatchTranslationResult, GlossaryEntry, TranslationLine, Usage};
use subbake_core::error::{CoreError, CoreResult};
use subbake_core::ports::{BackendJsonResult, BackendPayload, ChatMessage, LlmBackend};
use tokio::runtime::Runtime;

use crate::providers::BackendConfig;

const DEFAULT_TIMEOUT_SECONDS: f64 = 120.0;

/// A shared multi-thread tokio runtime used to drive async `reqwest` calls
/// from the synchronous `LlmBackend` trait methods.
fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        Runtime::new().unwrap_or_else(|_| panic!("unable to start subbake tokio runtime"))
    })
}

fn build_client(timeout: Duration) -> CoreResult<Client> {
    Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|error| CoreError::Backend(format!("http client build failed: {error}")))
}

fn missing_api_key(provider_label: &str) -> CoreError {
    CoreError::Backend(format!(
        "Missing API key for {provider_label} provider. Set the provider's environment variable or use --api-key."
    ))
}

// ---------------------------------------------------------------------------
// OpenAI-compatible backend (also backs Gemini via its OpenAI-compatible endpoint)
// ---------------------------------------------------------------------------

pub struct OpenAiCompatibleBackend {
    model: String,
    provider_label: String,
    api_key: String,
    base_url: String,
    client: Client,
}

impl OpenAiCompatibleBackend {
    pub fn new(
        config: &BackendConfig,
        provider_label: &str,
        default_base_url: &str,
        timeout_seconds: f64,
    ) -> CoreResult<Self> {
        let api_key = config
            .api_key
            .as_ref()
            .ok_or_else(|| missing_api_key(provider_label))?
            .clone();
        let base_url = config
            .base_url
            .as_deref()
            .map(trim_trailing_slash)
            .unwrap_or_else(|| default_base_url.to_owned());
        let client = build_client(duration_from_seconds(timeout_seconds))?;
        Ok(Self {
            model: config.model.clone(),
            provider_label: provider_label.to_owned(),
            api_key,
            base_url,
            client,
        })
    }

    async fn generate_once(&self, payload: &JsonValue) -> CoreResult<(JsonValue, Usage)> {
        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(payload)
            .send()
            .await
            .map_err(|error| {
                CoreError::Backend(format!("{} request failed: {error}", self.provider_label))
            })?;
        let status = response.status();
        let text = response.text().await.map_err(|error| {
            CoreError::Backend(format!(
                "{} response read failed: {error}",
                self.provider_label
            ))
        })?;
        if !status.is_success() {
            return Err(CoreError::Backend(format!(
                "{} rejected request ({status}): {text}",
                self.provider_label
            )));
        }
        let body: JsonValue = serde_json::from_str(&text).map_err(|error| {
            CoreError::Backend(format!(
                "{} response decode failed: {error}",
                self.provider_label
            ))
        })?;
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| {
                CoreError::Backend(format!(
                    "{} response missing message content",
                    self.provider_label
                ))
            })?;
        let parsed = extract_json_object(content)?;
        Ok((parsed, openai_usage(&body, content, payload)))
    }

    async fn check_once(&self) -> CoreResult<(bool, String)> {
        let url = format!("{}/models", self.base_url);
        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|error| {
                CoreError::Backend(format!(
                    "{} credential check failed: {error}",
                    self.provider_label
                ))
            })?;
        let status = response.status();
        let text = response.text().await.map_err(|error| {
            CoreError::Backend(format!(
                "{} response read failed: {error}",
                self.provider_label
            ))
        })?;
        if !status.is_success() {
            return Ok((
                false,
                format!(
                    "{} rejected credentials ({status}): {text}",
                    self.provider_label
                ),
            ));
        }
        let body: JsonValue = serde_json::from_str(&text).unwrap_or(JsonValue::Null);
        let model_count = body["data"].as_array().map(Vec::len).unwrap_or(0);
        let message = if model_count > 0 {
            format!(
                "Credentials look valid. {model_count} model(s) visible from {}.",
                self.provider_label
            )
        } else {
            format!(
                "Credentials look valid. Successfully reached {}.",
                self.provider_label
            )
        };
        Ok((true, message))
    }
}

impl LlmBackend for OpenAiCompatibleBackend {
    fn provider_name(&self) -> &str {
        &self.provider_label
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
        let (parsed, usage) = self.generate_raw_json(messages)?;
        let result = parse_translation_payload(&parsed)?;
        Ok(BackendJsonResult {
            payload: BackendPayload::Translation(result),
            usage,
        })
    }

    fn generate_raw_json(&mut self, messages: &[ChatMessage]) -> CoreResult<(JsonValue, Usage)> {
        let messages_value: Vec<JsonValue> = messages.iter().map(message_json).collect();
        let payload_with_format = json!({
            "model": self.model,
            "messages": &messages_value,
            "response_format": {"type": "json_object"},
        });
        let result = runtime().block_on(async { self.generate_once(&payload_with_format).await });
        let (parsed, usage) = match result {
            Ok(value) => value,
            Err(error) => {
                let message = error.to_string();
                // Endpoints that don't understand `response_format` (status 400
                // whose body mentions the field) retry with a plain chat request,
                // mirroring the Python client.
                if message.contains("(400") && message.contains("response_format") {
                    let payload = json!({"model": self.model, "messages": &messages_value});
                    runtime().block_on(async { self.generate_once(&payload).await })?
                } else {
                    return Err(error);
                }
            }
        };
        Ok((parsed, usage))
    }

    fn check_credentials(&self) -> CoreResult<(bool, String)> {
        runtime().block_on(async { self.check_once().await })
    }
}

// ---------------------------------------------------------------------------
// Anthropic backend (native /v1/messages API)
// ---------------------------------------------------------------------------

const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";
const ANTHROPIC_API_VERSION: &str = "2023-06-01";

pub struct AnthropicBackend {
    model: String,
    api_key: String,
    client: Client,
}

impl AnthropicBackend {
    pub fn new(config: &BackendConfig, timeout_seconds: f64) -> CoreResult<Self> {
        let api_key = config
            .api_key
            .as_ref()
            .ok_or_else(|| missing_api_key("Anthropic"))?
            .clone();
        let client = build_client(duration_from_seconds(timeout_seconds))?;
        Ok(Self {
            model: config.model.clone(),
            api_key,
            client,
        })
    }

    async fn generate_once(&self, messages: &[ChatMessage]) -> CoreResult<(JsonValue, Usage)> {
        let system = messages
            .iter()
            .filter(|message| message.role == "system")
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let body_messages: Vec<JsonValue> = messages
            .iter()
            .filter(|message| message.role != "system")
            .map(|message| {
                json!({
                    "role": message.role,
                    "content": [{"type": "text", "text": message.content}],
                })
            })
            .collect();
        let payload = json!({
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": body_messages,
        });

        let url = format!("{ANTHROPIC_BASE_URL}/messages");
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .json(&payload)
            .send()
            .await
            .map_err(|error| CoreError::Backend(format!("Anthropic request failed: {error}")))?;
        let status = response.status();
        let text = response.text().await.map_err(|error| {
            CoreError::Backend(format!("Anthropic response read failed: {error}"))
        })?;
        if !status.is_success() {
            return Err(CoreError::Backend(format!(
                "Anthropic rejected request ({status}): {text}"
            )));
        }
        let body: JsonValue = serde_json::from_str(&text).map_err(|error| {
            CoreError::Backend(format!("Anthropic response decode failed: {error}"))
        })?;
        let content = body["content"]
            .as_array()
            .map(|entries| {
                entries
                    .iter()
                    .filter(|entry| entry["type"].as_str() == Some("text"))
                    .filter_map(|entry| entry["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        let parsed = extract_json_object(&content)?;
        Ok((parsed, anthropic_usage(&body, &content, &payload)))
    }

    async fn check_once(&self) -> CoreResult<(bool, String)> {
        let url = format!("{ANTHROPIC_BASE_URL}/models");
        let response = self
            .client
            .get(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .send()
            .await
            .map_err(|error| {
                CoreError::Backend(format!("Anthropic credential check failed: {error}"))
            })?;
        let status = response.status();
        let text = response.text().await.map_err(|error| {
            CoreError::Backend(format!("Anthropic response read failed: {error}"))
        })?;
        if !status.is_success() {
            return Ok((
                false,
                format!("Anthropic rejected credentials ({status}): {text}"),
            ));
        }
        let body: JsonValue = serde_json::from_str(&text).unwrap_or(JsonValue::Null);
        let model_count = body["data"].as_array().map(Vec::len).unwrap_or(0);
        let message = if model_count > 0 {
            format!("Credentials look valid. {model_count} model(s) visible from Anthropic.")
        } else {
            "Credentials look valid. Successfully reached Anthropic.".to_owned()
        };
        Ok((true, message))
    }
}

impl LlmBackend for AnthropicBackend {
    fn provider_name(&self) -> &str {
        "Anthropic"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
        let (parsed, usage) = self.generate_raw_json(messages)?;
        let result = parse_translation_payload(&parsed)?;
        Ok(BackendJsonResult {
            payload: BackendPayload::Translation(result),
            usage,
        })
    }

    fn generate_raw_json(&mut self, messages: &[ChatMessage]) -> CoreResult<(JsonValue, Usage)> {
        let (parsed, usage) = runtime().block_on(async { self.generate_once(messages).await })?;
        Ok((parsed, usage))
    }

    fn check_credentials(&self) -> CoreResult<(bool, String)> {
        runtime().block_on(async { self.check_once().await })
    }
}

// ---------------------------------------------------------------------------
// Constructors used by `providers::build_backend`
// ---------------------------------------------------------------------------

pub fn openai_backend(
    config: &BackendConfig,
    timeout_seconds: f64,
) -> CoreResult<Box<dyn LlmBackend>> {
    Ok(Box::new(OpenAiCompatibleBackend::new(
        config,
        "OpenAI",
        "https://api.openai.com/v1",
        timeout_seconds,
    )?))
}

pub fn gemini_backend(
    config: &BackendConfig,
    timeout_seconds: f64,
) -> CoreResult<Box<dyn LlmBackend>> {
    Ok(Box::new(OpenAiCompatibleBackend::new(
        config,
        "Gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        timeout_seconds,
    )?))
}

pub fn anthropic_backend(
    config: &BackendConfig,
    timeout_seconds: f64,
) -> CoreResult<Box<dyn LlmBackend>> {
    Ok(Box::new(AnthropicBackend::new(config, timeout_seconds)?))
}

pub fn default_timeout_seconds() -> f64 {
    DEFAULT_TIMEOUT_SECONDS
}

// ---------------------------------------------------------------------------
// Response parsing + token estimation
// ---------------------------------------------------------------------------

fn message_json(message: &ChatMessage) -> JsonValue {
    json!({"role": message.role, "content": message.content})
}

/// Extract the first balanced `{...}` JSON object embedded in `text`. The LLM
/// may wrap the payload in markdown fences or surrounding prose, so a naive
/// `serde_json::from_str` is not enough.
pub(crate) fn extract_json_object(text: &str) -> CoreResult<JsonValue> {
    let trimmed = text.trim();
    let start = trimmed
        .find('{')
        .ok_or_else(|| CoreError::Backend("response is missing a JSON object".to_owned()))?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    let mut end = None;
    for (index, character) in trimmed.char_indices().skip_while(|(i, _)| *i < start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }
        match character {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(index + character.len_utf8());
                    break;
                }
            }
            _ => {}
        }
    }
    let end =
        end.ok_or_else(|| CoreError::Backend("response JSON object is unbalanced".to_owned()))?;
    let slice = trimmed.get(start..end).ok_or_else(|| {
        CoreError::Backend("response JSON object slice is out of bounds".to_owned())
    })?;
    serde_json::from_str(slice)
        .map_err(|error| CoreError::Backend(format!("invalid JSON in response: {error}")))
}

/// Parse the LLM JSON contract (`lines`/`summary`/`glossary_updates`) into the
/// domain type. Mirrors Python `parse_translation_lines` +
/// `parse_glossary_entries`, which accept `\[{source,target}\]` *or* a
/// `{source:target}` map for glossary updates.
pub(crate) fn parse_translation_payload(value: &JsonValue) -> CoreResult<BatchTranslationResult> {
    let lines = value["lines"]
        .as_array()
        .ok_or_else(|| CoreError::Backend("response missing lines array".to_owned()))?
        .iter()
        .map(|entry| TranslationLine {
            id: entry["id"].as_str().unwrap_or_default().to_owned(),
            translation: entry["translation"].as_str().unwrap_or_default().to_owned(),
        })
        .collect();
    let summary = value["summary"].as_str().unwrap_or_default().to_owned();
    let glossary_updates = parse_glossary_updates(&value["glossary_updates"]);
    Ok(BatchTranslationResult {
        lines,
        summary,
        glossary_updates,
    })
}

fn parse_glossary_updates(value: &JsonValue) -> Vec<GlossaryEntry> {
    let mut entries = Vec::new();
    match value {
        JsonValue::Array(items) => {
            for item in items {
                let source = item["source"].as_str().unwrap_or_default().to_owned();
                let target = item["target"].as_str().unwrap_or_default().to_owned();
                if !source.is_empty() || !target.is_empty() {
                    entries.push(GlossaryEntry { source, target });
                }
            }
        }
        JsonValue::Object(map) => {
            for (source, target) in map {
                entries.push(GlossaryEntry {
                    source: source.clone(),
                    target: target.as_str().unwrap_or_default().to_owned(),
                });
            }
        }
        _ => {}
    }
    entries
}

fn openai_usage(body: &JsonValue, content: &str, payload: &JsonValue) -> Usage {
    let usage = &body["usage"];
    let input_tokens = usage["prompt_tokens"]
        .as_u64()
        .map(|value| value as usize)
        .unwrap_or_else(|| estimate_tokens_json(payload));
    let output_tokens = usage["completion_tokens"]
        .as_u64()
        .map(|value| value as usize)
        .unwrap_or_else(|| estimate_tokens(content));
    let total_tokens = usage["total_tokens"]
        .as_u64()
        .map(|value| value as usize)
        .filter(|value| *value > 0)
        .unwrap_or(input_tokens + output_tokens);
    Usage {
        input_tokens,
        output_tokens,
        total_tokens,
    }
}

fn anthropic_usage(body: &JsonValue, content: &str, payload: &JsonValue) -> Usage {
    let usage = &body["usage"];
    let input_tokens = usage["input_tokens"]
        .as_u64()
        .map(|value| value as usize)
        .unwrap_or_else(|| estimate_tokens_json(payload));
    let output_tokens = usage["output_tokens"]
        .as_u64()
        .map(|value| value as usize)
        .unwrap_or_else(|| estimate_tokens(content));
    Usage {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens + output_tokens,
    }
}

fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4).max(1)
}

fn estimate_tokens_json(value: &JsonValue) -> usize {
    estimate_tokens(&value.to_string())
}

fn duration_from_seconds(seconds: f64) -> Duration {
    Duration::from_secs_f64(seconds.max(1.0))
}

fn trim_trailing_slash(value: &str) -> String {
    let trimmed = value.trim();
    trimmed
        .strip_suffix('/')
        .map(|inner| inner.to_owned())
        .unwrap_or_else(|| trimmed.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_object_handles_markdown_fences() {
        let value = extract_json_object("```json\n{\"lines\":[],\"summary\":\"ok\"}\n```")
            .expect("extract from fenced block");
        assert_eq!(value["summary"].as_str(), Some("ok"));
    }

    #[test]
    fn extract_json_object_balances_braces_in_strings() {
        let payload = "{\"lines\":[{\"id\":\"1\",\"translation\":\"a {b} c\"}],\"summary\":\"s\"}";
        let value = extract_json_object(payload).expect("extract nested");
        let result = parse_translation_payload(&value).expect("parse");
        assert_eq!(result.lines.len(), 1);
        assert_eq!(result.lines[0].translation, "a {b} c");
    }

    #[test]
    fn parse_translation_payload_accepts_glossary_list() {
        let value = json!({
            "lines": [{"id": "1", "translation": "你好"}],
            "summary": "batch done",
            "glossary_updates": [{"source": "alice", "target": "爱丽丝"}],
        });
        let result = parse_translation_payload(&value).expect("parse list");
        assert_eq!(result.lines[0].translation, "你好");
        assert_eq!(result.glossary_updates.len(), 1);
        assert_eq!(result.glossary_updates[0].target, "爱丽丝");
    }

    #[test]
    fn parse_translation_payload_accepts_glossary_map() {
        let value = json!({
            "lines": [],
            "glossary_updates": {"bob": "鲍勃"},
        });
        let result = parse_translation_payload(&value).expect("parse map");
        assert_eq!(result.glossary_updates.len(), 1);
        assert_eq!(result.glossary_updates[0].source, "bob");
    }

    #[test]
    fn extract_json_object_rejects_text_without_object() {
        assert!(extract_json_object("no json here").is_err());
    }

    #[test]
    fn trim_trailing_slash_strips_only_trailing() {
        assert_eq!(trim_trailing_slash("https://x/v1/"), "https://x/v1");
        assert_eq!(trim_trailing_slash("https://x/v1"), "https://x/v1");
    }
}
