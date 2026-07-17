use std::future::Future;
use std::time::Duration;

use reqwest::{Client, RequestBuilder};
use serde_json::Value;
use subbake_core::CancellationGuard;
use subbake_core::error::LlmCallError;

use crate::error::{AdapterError, AdapterResult};

const MAX_TRANSPORT_RETRIES: usize = 2;

pub(super) fn client(timeout: f64) -> AdapterResult<Client> {
    Client::builder()
        .timeout(Duration::from_secs_f64(timeout.max(1.0)))
        .build()
        .map_err(|error| AdapterError::from_http("LLM HTTP client", error))
}

pub(super) async fn send_json<F>(
    request: F,
    display_name: &str,
    cancellation: &CancellationGuard,
    native_tools: bool,
) -> Result<Value, LlmCallError>
where
    F: Fn() -> RequestBuilder,
{
    let mut attempt = 0;
    loop {
        cancellation.check().map_err(LlmCallError::from)?;
        let response = await_http(request().send(), cancellation, "provider request failed").await;
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
                            "{display_name} response decode failed: {error}"
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

pub(super) async fn await_http<F, T>(
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
            output = &mut future => {
                return output.map_err(|error| {
                    if error.is_timeout() {
                        LlmCallError::Timeout(format!("{context}: {error}"))
                    } else {
                        LlmCallError::Transport(format!("{context}: {error}"))
                    }
                });
            },
            _ = tick.tick() => cancellation.check().map_err(LlmCallError::from)?,
        }
    }
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

pub(super) fn classify_status_error(
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

pub(super) fn native_tools_unsupported(status: u16, body: &str) -> bool {
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
