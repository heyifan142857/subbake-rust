mod anthropic;
mod gemini;
mod openai_chat;
mod openai_responses;

use reqwest::RequestBuilder;
use serde_json::Value;
use subbake_core::entities::Usage;
use subbake_core::ports::{ChatMessage, ResponseContract};

use crate::providers::{ApiFormat, BackendConfig};

pub(super) fn endpoint(format: ApiFormat, config: &BackendConfig) -> String {
    if let Some(url) = &config.endpoint_url {
        return url.trim().to_owned();
    }
    match format {
        ApiFormat::OpenaiChat => openai_chat::endpoint(config),
        ApiFormat::OpenaiResponses => openai_responses::endpoint(config),
        ApiFormat::AnthropicMessages => anthropic::endpoint(config),
        ApiFormat::GeminiGenerateContent => gemini::endpoint(config),
    }
}

pub(super) fn authenticate(
    format: ApiFormat,
    config: &BackendConfig,
    key: &str,
    request: RequestBuilder,
) -> RequestBuilder {
    if let Some(header) = &config.auth_header {
        return request.header(
            header,
            format!("{}{}", config.auth_prefix.as_deref().unwrap_or(""), key),
        );
    }
    match format {
        ApiFormat::OpenaiChat => openai_chat::authenticate(key, request),
        ApiFormat::OpenaiResponses => openai_responses::authenticate(key, request),
        ApiFormat::AnthropicMessages => anthropic::authenticate(key, request),
        ApiFormat::GeminiGenerateContent => gemini::authenticate(key, request),
    }
}

pub(super) fn request_body(
    format: ApiFormat,
    model: &str,
    messages: &[ChatMessage],
    contract: ResponseContract,
) -> Value {
    match format {
        ApiFormat::OpenaiChat => openai_chat::request_body(model, messages, contract),
        ApiFormat::OpenaiResponses => openai_responses::request_body(model, messages, contract),
        ApiFormat::AnthropicMessages => anthropic::request_body(model, messages),
        ApiFormat::GeminiGenerateContent => gemini::request_body(messages),
    }
}

pub(super) fn response_text(format: ApiFormat, body: &Value) -> Option<String> {
    match format {
        ApiFormat::OpenaiChat => openai_chat::response_text(body),
        ApiFormat::OpenaiResponses => openai_responses::response_text(body),
        ApiFormat::AnthropicMessages => anthropic::response_text(body),
        ApiFormat::GeminiGenerateContent => gemini::response_text(body),
    }
}

pub(super) fn usage(format: ApiFormat, body: &Value, text: &str, payload: &Value) -> Usage {
    let usage = &body["usage"];
    let input = match format {
        ApiFormat::AnthropicMessages => usage["input_tokens"].as_u64(),
        ApiFormat::GeminiGenerateContent => body["usageMetadata"]["promptTokenCount"].as_u64(),
        ApiFormat::OpenaiChat | ApiFormat::OpenaiResponses => usage["prompt_tokens"]
            .as_u64()
            .or_else(|| usage["input_tokens"].as_u64()),
    }
    .map(|tokens| tokens as usize)
    .unwrap_or_else(|| estimate(&payload.to_string()));
    let output = match format {
        ApiFormat::AnthropicMessages => usage["output_tokens"].as_u64(),
        ApiFormat::GeminiGenerateContent => body["usageMetadata"]["candidatesTokenCount"].as_u64(),
        ApiFormat::OpenaiChat | ApiFormat::OpenaiResponses => usage["completion_tokens"]
            .as_u64()
            .or_else(|| usage["output_tokens"].as_u64()),
    }
    .map(|tokens| tokens as usize)
    .unwrap_or_else(|| estimate(text));
    let total = usage["total_tokens"]
        .as_u64()
        .or_else(|| body["usageMetadata"]["totalTokenCount"].as_u64())
        .map(|tokens| tokens as usize)
        .unwrap_or(input + output);
    Usage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: total,
    }
}

pub(super) fn openai_message(message: &ChatMessage) -> Value {
    openai_chat::message(message)
}

#[cfg(test)]
pub(super) fn apply_response_contract(
    format: ApiFormat,
    contract: ResponseContract,
    body: &mut Value,
) {
    match format {
        ApiFormat::OpenaiChat => openai_chat::apply_contract(contract, body),
        ApiFormat::OpenaiResponses => openai_responses::apply_contract(contract, body),
        ApiFormat::AnthropicMessages | ApiFormat::GeminiGenerateContent => {}
    }
}

fn estimate(value: &str) -> usize {
    value.chars().count().div_ceil(4).max(1)
}
