use serde::{Deserialize, Serialize};
use subbake_core::CancellationGuard;
use subbake_core::error::CoreError;
use subbake_core::ports::{ChatMessage, GenerationRequest, LlmBackend};

use crate::error::{AdapterError, AdapterResult};
use crate::mock::MockBackend;

/// Wire protocol, deliberately independent of a user-facing profile name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiFormat {
    AnthropicMessages,
    OpenaiChat,
    OpenaiResponses,
    GeminiGenerateContent,
}

impl ApiFormat {
    pub fn parse(value: &str) -> AdapterResult<Self> {
        match value {
            "anthropic_messages" => Ok(Self::AnthropicMessages),
            "openai_chat" => Ok(Self::OpenaiChat),
            "openai_responses" => Ok(Self::OpenaiResponses),
            "gemini_generate_content" => Ok(Self::GeminiGenerateContent),
            _ => Err(AdapterError::invalid_input(format!(
                "invalid api_format `{value}`; expected anthropic_messages, openai_chat, openai_responses, or gemini_generate_content"
            ))),
        }
    }
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AnthropicMessages => "anthropic_messages",
            Self::OpenaiChat => "openai_chat",
            Self::OpenaiResponses => "openai_responses",
            Self::GeminiGenerateContent => "gemini_generate_content",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendConfig {
    /// Stable profile identifier. It never selects a protocol.
    pub id: String,
    pub display_name: String,
    pub api_format: Option<ApiFormat>,
    pub model: String,
    pub base_url: Option<String>,
    pub endpoint_url: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub auth_header: Option<String>,
    pub auth_prefix: Option<String>,
}

impl BackendConfig {
    /// Construct a backend identity without selecting a wire protocol.
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        let id = provider.into();
        Self {
            display_name: id.clone(),
            id,
            api_format: None,
            model: model.into(),
            base_url: None,
            endpoint_url: None,
            api_key: None,
            api_key_env: None,
            auth_header: None,
            auth_prefix: None,
        }
    }
    pub fn resolved_api_key(&self) -> Option<String> {
        self.api_key
            .clone()
            .or_else(|| resolve_env_var(self.api_key_env.as_deref()))
            .or_else(|| resolve_env_var(default_api_key_env(&self.id)))
    }
    pub fn validate(&self) -> AdapterResult<()> {
        if self.id.trim().is_empty() || self.model.trim().is_empty() {
            return Err(AdapterError::invalid_input(
                "provider id and model are required",
            ));
        }
        for value in [self.auth_header.as_deref(), self.auth_prefix.as_deref()] {
            if value.is_some_and(|v| v.contains(['\r', '\n'])) {
                return Err(AdapterError::invalid_input(
                    "authentication header must not contain CR/LF",
                ));
            }
        }
        if self.id != "mock" && self.api_format.is_none() {
            return Err(AdapterError::invalid_input(
                "api_format is required for custom provider profiles",
            ));
        }
        Ok(())
    }
}

pub fn default_api_key_env(provider: &str) -> Option<&'static str> {
    match provider.to_ascii_lowercase().as_str() {
        "openai" => Some("OPENAI_API_KEY"),
        "anthropic" => Some("ANTHROPIC_API_KEY"),
        "gemini" => Some("GEMINI_API_KEY"),
        "deepseek" => Some("DEEPSEEK_API_KEY"),
        _ => None,
    }
}
pub fn resolve_env_var(env_var: Option<&str>) -> Option<String> {
    env_var
        .and_then(non_empty_string)
        .and_then(|name| std::env::var(name).ok())
        .and_then(|v| non_empty_string(&v).map(ToOwned::to_owned))
}
fn non_empty_string(value: &str) -> Option<&str> {
    (!value.trim().is_empty()).then_some(value.trim())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderCheckRequest {
    pub config: BackendConfig,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderCheckOutcome {
    pub provider: String,
    pub model: String,
    pub message: String,
}

pub fn build_backend(config: &BackendConfig) -> AdapterResult<Box<dyn LlmBackend>> {
    build_backend_with_timeout(config, crate::llm_backends::default_timeout_seconds())
}
pub fn build_backend_with_timeout(
    config: &BackendConfig,
    timeout: f64,
) -> AdapterResult<Box<dyn LlmBackend>> {
    config.validate()?;
    if config.id.eq_ignore_ascii_case("mock") {
        return Ok(Box::new(MockBackend::new(config.model.clone())));
    }
    crate::llm_backends::build_protocol_backend(config, timeout)
}

pub fn check_provider(request: ProviderCheckRequest) -> AdapterResult<ProviderCheckOutcome> {
    let mut backend = build_backend(&request.config)?;
    if request.config.id.eq_ignore_ascii_case("mock") {
        let (_, message) = backend.check_credentials().map_err(AdapterError::from)?;
        return Ok(ProviderCheckOutcome {
            provider: backend.provider_name().to_owned(),
            model: backend.model_name().to_owned(),
            message,
        });
    }
    // Deliberately a real minimal generation, rather than a model-list request:
    // it validates the configured endpoint, model, auth and protocol together.
    let response = backend
        .execute(
            GenerationRequest::json(vec![ChatMessage::user(
                "Return exactly this JSON object: {\"ok\":true}",
            )]),
            &CancellationGuard::never(),
        )
        .map_err(AdapterError::from)?;
    let (json, _) = response.into_json().map_err(AdapterError::from)?;
    if json["ok"].as_bool() != Some(true) {
        return Err(AdapterError::Core(CoreError::InvalidBackendResponse(
            "provider check response did not satisfy the JSON probe".to_owned(),
        )));
    }
    Ok(ProviderCheckOutcome {
        provider: backend.provider_name().to_owned(),
        model: backend.model_name().to_owned(),
        message:
            "Provider accepted a minimal JSON generation probe (this call may incur model usage)."
                .to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_ids_do_not_select_a_wire_protocol() {
        assert_eq!(BackendConfig::new("openai", "x").api_format, None);
        assert_eq!(BackendConfig::new("anthropic", "x").api_format, None);
        assert_eq!(BackendConfig::new("gemini", "x").api_format, None);
    }

    #[test]
    fn custom_profiles_must_name_a_protocol() {
        let error = BackendConfig::new("company-relay", "x")
            .validate()
            .expect_err("format required");
        assert!(error.to_string().contains("api_format"));
    }

    #[test]
    fn rejects_header_injection() {
        let mut config = BackendConfig::new("mock", "x");
        config.auth_header = Some("X-Key\r\nInjected: yes".to_owned());
        assert!(config.validate().is_err());
    }
}
