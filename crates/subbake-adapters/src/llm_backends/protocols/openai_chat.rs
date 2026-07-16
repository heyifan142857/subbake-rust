use reqwest::RequestBuilder;
use serde_json::{Value, json};
use subbake_core::ports::{ChatMessage, ResponseContract};

use crate::providers::BackendConfig;

pub(super) fn endpoint(config: &BackendConfig) -> String {
    format!(
        "{}/chat/completions",
        config
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1")
            .trim_end_matches('/')
    )
}

pub(super) fn authenticate(key: &str, request: RequestBuilder) -> RequestBuilder {
    request.bearer_auth(key)
}

pub(super) fn request_body(
    model: &str,
    messages: &[ChatMessage],
    contract: ResponseContract,
) -> Value {
    let mut body = json!({
        "model": model,
        "messages": messages.iter().map(message).collect::<Vec<_>>()
    });
    apply_contract(contract, &mut body);
    body
}

pub(super) fn response_text(body: &Value) -> Option<String> {
    body["choices"][0]["message"]["content"]
        .as_str()
        .map(str::to_owned)
}

pub(super) fn message(message: &ChatMessage) -> Value {
    json!({"role":message.role,"content":message.content})
}

pub(super) fn apply_contract(contract: ResponseContract, body: &mut Value) {
    if contract == ResponseContract::JsonObject {
        body["response_format"] = json!({"type": "json_object"});
    }
}
