use subbake_core::error::{CoreError, CoreResult};
use subbake_core::ports::LlmBackend;

use crate::mock::MockBackend;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendConfig {
    pub provider: String,
    pub model: String,
}

impl BackendConfig {
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }
}

pub fn build_backend(config: &BackendConfig) -> CoreResult<Box<dyn LlmBackend>> {
    match config.provider.to_lowercase().as_str() {
        "mock" => Ok(Box::new(MockBackend::new(config.model.clone()))),
        provider => Err(CoreError::Backend(format!(
            "provider `{provider}` adapter is pending migration"
        ))),
    }
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
        let error = match build_backend(&BackendConfig::new("openai", "gpt")) {
            Ok(_) => panic!("openai should be pending"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("pending migration"));
    }
}
