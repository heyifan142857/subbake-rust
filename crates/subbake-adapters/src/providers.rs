use subbake_core::error::{CoreError, CoreResult};
use subbake_core::ports::LlmBackend;

use crate::mock::MockBackend;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendConfig {
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
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

impl BackendConfig {
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            api_key: None,
            base_url: None,
        }
    }
}

pub fn default_api_key_env(provider: &str) -> Option<&'static str> {
    match provider.to_lowercase().as_str() {
        "openai" => Some("OPENAI_API_KEY"),
        "anthropic" => Some("ANTHROPIC_API_KEY"),
        "gemini" => Some("GEMINI_API_KEY"),
        _ => None,
    }
}

pub fn resolve_env_var(env_var: Option<&str>) -> Option<String> {
    env_var
        .and_then(non_empty_string)
        .and_then(|name| std::env::var(name).ok())
        .and_then(|value| non_empty_string(&value).map(ToOwned::to_owned))
}

pub fn build_backend(config: &BackendConfig) -> CoreResult<Box<dyn LlmBackend>> {
    build_backend_with_timeout(config, crate::llm_backends::default_timeout_seconds())
}

/// Build a backend with an explicit request timeout (seconds). The translation
/// paths carry a configured timeout; credential checks use the default.
pub fn build_backend_with_timeout(
    config: &BackendConfig,
    timeout_seconds: f64,
) -> CoreResult<Box<dyn LlmBackend>> {
    match config.provider.to_lowercase().as_str() {
        "mock" => Ok(Box::new(MockBackend::new(config.model.clone()))),
        "openai" => crate::llm_backends::openai_backend(config, timeout_seconds),
        "gemini" => crate::llm_backends::gemini_backend(config, timeout_seconds),
        "anthropic" => crate::llm_backends::anthropic_backend(config, timeout_seconds),
        provider => Err(CoreError::Backend(format!(
            "unsupported provider `{provider}`"
        ))),
    }
}

fn non_empty_string(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

pub fn check_provider(request: ProviderCheckRequest) -> CoreResult<ProviderCheckOutcome> {
    let backend = build_backend(&request.config)?;
    let (ok, message) = backend.check_credentials()?;
    if !ok {
        return Err(CoreError::Backend(message));
    }

    Ok(ProviderCheckOutcome {
        provider: backend.provider_name().to_owned(),
        model: backend.model_name().to_owned(),
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_mock_backend() {
        let backend = build_backend(&BackendConfig::new("mock", "mock-zh"))
            .expect("mock backend should build");
        assert_eq!(backend.provider_name(), "mock");
        assert_eq!(backend.model_name(), "mock-zh");
    }

    #[test]
    fn rejects_unknown_backend() {
        let error = match build_backend(&BackendConfig::new("zzz-unknown", "model")) {
            Ok(_) => panic!("unknown provider should not build"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("unsupported provider"));
    }

    #[test]
    fn openai_backend_requires_api_key() {
        let error = match build_backend(&BackendConfig::new("openai", "gpt")) {
            Ok(_) => panic!("missing api key should fail build"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("API key"));
    }

    #[test]
    fn default_api_key_env_depends_on_provider() {
        assert_eq!(default_api_key_env("openai"), Some("OPENAI_API_KEY"));
        assert_eq!(default_api_key_env("anthropic"), Some("ANTHROPIC_API_KEY"));
        assert_eq!(default_api_key_env("gemini"), Some("GEMINI_API_KEY"));
        assert_eq!(default_api_key_env("mock"), None);
    }

    #[test]
    fn checks_mock_provider() {
        let outcome = check_provider(ProviderCheckRequest {
            config: BackendConfig::new("mock", "mock-zh"),
        })
        .expect("mock provider should check");

        assert_eq!(outcome.provider, "mock");
        assert_eq!(outcome.model, "mock-zh");
        assert!(!outcome.message.is_empty());
    }
}
