use reqwest::RequestBuilder;
use serde_json::{Value, json};
use subbake_core::ports::ChatMessage;

use crate::providers::BackendConfig;

const ANTHROPIC_VERSION: &str = "2023-06-01";

pub(super) fn endpoint(config: &BackendConfig) -> String {
    format!(
        "{}/messages",
        config
            .base_url
            .as_deref()
            .unwrap_or("https://api.anthropic.com/v1")
            .trim_end_matches('/')
    )
}

pub(super) fn authenticate(key: &str, request: RequestBuilder) -> RequestBuilder {
    request
        .header("x-api-key", key)
        .header("anthropic-version", ANTHROPIC_VERSION)
}

pub(super) fn request_body(model: &str, messages: &[ChatMessage]) -> Value {
    let system = messages
        .iter()
        .filter(|message| message.role == "system")
        .collect::<Vec<_>>();
    let mut body = json!({
        "model": model,
        "max_tokens": 4096,
        "messages": messages.iter()
            .filter(|message| message.role != "system")
            .map(|message| json!({
                "role":message.role,
                "content":[{"type":"text","text":message.content}]
            }))
            .collect::<Vec<_>>()
    });
    body["system"] = if system.iter().any(|message| message.cacheable) {
        json!(
            system
                .iter()
                .map(|message| {
                    let mut block = json!({"type":"text", "text": message.content});
                    if message.cacheable {
                        block["cache_control"] = json!({"type":"ephemeral"});
                    }
                    block
                })
                .collect::<Vec<_>>()
        )
    } else {
        json!(
            system
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n")
        )
    };
    body
}

pub(super) fn response_text(body: &Value) -> Option<String> {
    body["content"].as_array().map(|content| {
        content
            .iter()
            .filter_map(|value| value["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n")
    })
}
