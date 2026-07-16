use reqwest::RequestBuilder;
use serde_json::{Value, json};
use subbake_core::ports::ChatMessage;

use crate::providers::BackendConfig;

pub(super) fn endpoint(config: &BackendConfig) -> String {
    format!(
        "{}/models/{}:generateContent",
        config
            .base_url
            .as_deref()
            .unwrap_or("https://generativelanguage.googleapis.com/v1beta")
            .trim_end_matches('/'),
        config.model
    )
}

pub(super) fn authenticate(key: &str, request: RequestBuilder) -> RequestBuilder {
    request.header("x-goog-api-key", key)
}

pub(super) fn request_body(messages: &[ChatMessage]) -> Value {
    let system = messages
        .iter()
        .filter(|message| message.role == "system")
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    json!({
        "systemInstruction":{"parts":[{"text":system}]},
        "contents":messages.iter()
            .filter(|message|message.role!="system")
            .map(|message|json!({
                "role":if message.role=="assistant" {"model"} else {"user"},
                "parts":[{"text":message.content}]
            }))
            .collect::<Vec<_>>()
    })
}

pub(super) fn response_text(body: &Value) -> Option<String> {
    body["candidates"][0]["content"]["parts"]
        .as_array()
        .map(|parts| {
            parts
                .iter()
                .filter_map(|value| value["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n")
        })
}
