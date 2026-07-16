use reqwest::RequestBuilder;
use serde_json::{Value, json};
use subbake_core::ports::{ChatMessage, ResponseContract};

use crate::providers::BackendConfig;

pub(super) fn endpoint(config: &BackendConfig) -> String {
    format!(
        "{}/responses",
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
        "input": messages.iter().map(|message| json!({
            "role": message.role,
            "content": [{"type":"input_text","text":message.content}]
        })).collect::<Vec<_>>()
    });
    apply_contract(contract, &mut body);
    body
}

pub(super) fn response_text(body: &Value) -> Option<String> {
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

pub(super) fn apply_contract(contract: ResponseContract, body: &mut Value) {
    if contract == ResponseContract::JsonObject {
        body["text"] = json!({"format": {"type": "json_object"}});
    }
}
