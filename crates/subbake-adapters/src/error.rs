use std::io;
use std::path::PathBuf;

use subbake_core::{CoreError, LlmCallError};
use thiserror::Error;

pub type AdapterResult<T> = Result<T, AdapterError>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ConfigError {
    #[error("profile name may contain only letters, numbers, '-' and '_'")]
    InvalidProfileName,
    #[error("profile `{name}` already exists")]
    DuplicateProfile { name: String },
    #[error("{reason}")]
    Invalid { reason: String },
    #[error("{reason} on line {line}")]
    Line { line: usize, reason: String },
    #[error("config key `{key}` {reason}")]
    Key { key: String, reason: String },
    #[error("config key `{key}` {reason} on line {line}")]
    KeyAtLine {
        key: String,
        line: usize,
        reason: String,
    },
}

impl ConfigError {
    pub fn invalid(reason: impl Into<String>) -> Self {
        Self::Invalid {
            reason: reason.into(),
        }
    }

    pub fn at_line(self, line: usize) -> Self {
        match self {
            Self::Invalid { reason } => Self::Line { line, reason },
            Self::Key { key, reason } => Self::KeyAtLine { key, line, reason },
            Self::Line { .. }
            | Self::KeyAtLine { .. }
            | Self::InvalidProfileName
            | Self::DuplicateProfile { .. } => self,
        }
    }

    pub fn for_key(key: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Key {
            key: key.into(),
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("operation cancelled")]
    Cancelled,
    #[error("{message}")]
    InvalidInput { message: String },
    #[error(transparent)]
    Configuration(#[from] ConfigError),
    #[error("failed to load config `{path}`: {source}")]
    ConfigurationFile {
        path: PathBuf,
        #[source]
        source: ConfigError,
    },
    #[error("authentication failed: {message}")]
    Authentication { message: String },
    #[error("request was rate limited: {message}")]
    RateLimited {
        message: String,
        retry_after_ms: Option<u64>,
    },
    #[error("operation timed out: {message}")]
    Timeout { message: String },
    #[error("transport failed: {message}")]
    Transport { message: String },
    #[error("{service} rejected the request{status_suffix}: {message}", status_suffix = status.map(|value| format!(" ({value})")).unwrap_or_default())]
    ServiceRejected {
        service: &'static str,
        status: Option<u16>,
        message: String,
    },
    #[error("{operation}{path_suffix}: {source}", path_suffix = path.as_ref().map(|value| format!(" `{}`", value.display())).unwrap_or_default())]
    ExternalIo {
        operation: &'static str,
        path: Option<PathBuf>,
        #[source]
        source: io::Error,
    },
    #[error("{context}: {source}")]
    Serialization {
        context: &'static str,
        #[source]
        source: serde_json::Error,
    },
    #[error("{program} failed: {message}")]
    ChildProcess {
        program: &'static str,
        status: Option<i32>,
        message: String,
    },
    #[error(transparent)]
    Core(CoreError),
    #[error("{operation}{path_suffix}: {source}", path_suffix = path.as_ref().map(|value| format!(" `{}`", value.display())).unwrap_or_default())]
    CoreContext {
        operation: &'static str,
        path: Option<PathBuf>,
        #[source]
        source: CoreError,
    },
}

impl AdapterError {
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    pub fn external_io(
        operation: &'static str,
        path: impl Into<Option<PathBuf>>,
        source: io::Error,
    ) -> Self {
        Self::ExternalIo {
            operation,
            path: path.into(),
            source,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }

    pub fn is_not_found(&self) -> bool {
        matches!(
            self,
            Self::ExternalIo { source, .. } if source.kind() == io::ErrorKind::NotFound
        )
    }

    pub fn from_http(service: &'static str, error: reqwest::Error) -> Self {
        if error.is_timeout() {
            Self::Timeout {
                message: format!("{service}: {error}"),
            }
        } else {
            Self::Transport {
                message: format!("{service}: {error}"),
            }
        }
    }

    pub fn from_http_status(
        service: &'static str,
        status: u16,
        message: impl Into<String>,
        retry_after_ms: Option<u64>,
    ) -> Self {
        let message = message.into();
        match status {
            401 | 403 => Self::Authentication { message },
            429 => Self::RateLimited {
                message,
                retry_after_ms,
            },
            _ => Self::ServiceRejected {
                service,
                status: Some(status),
                message,
            },
        }
    }
}

impl From<CoreError> for AdapterError {
    fn from(error: CoreError) -> Self {
        match error {
            CoreError::Cancelled => Self::Cancelled,
            CoreError::Llm(error) => Self::from(error),
            other => Self::Core(other),
        }
    }
}

impl From<LlmCallError> for AdapterError {
    fn from(error: LlmCallError) -> Self {
        match error {
            LlmCallError::Cancelled => Self::Cancelled,
            LlmCallError::Timeout(message) => Self::Timeout { message },
            LlmCallError::Authentication(message) => Self::Authentication { message },
            LlmCallError::RateLimited {
                message,
                retry_after_ms,
            } => Self::RateLimited {
                message,
                retry_after_ms,
            },
            LlmCallError::Transport(message) => Self::Transport { message },
            LlmCallError::Rejected { status, message } => Self::ServiceRejected {
                service: "LLM provider",
                status,
                message,
            },
            other => Self::Core(CoreError::from(other)),
        }
    }
}

impl From<io::Error> for AdapterError {
    fn from(source: io::Error) -> Self {
        if source.kind() == io::ErrorKind::Interrupted {
            Self::Cancelled
        } else {
            Self::ExternalIo {
                operation: "external I/O operation failed",
                path: None,
                source,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_categories_survive_the_adapter_boundary() {
        let authentication =
            AdapterError::from(LlmCallError::Authentication("bad token".to_owned()));
        assert!(matches!(
            authentication,
            AdapterError::Authentication { .. }
        ));

        let rate_limited = AdapterError::from(LlmCallError::RateLimited {
            message: "slow down".to_owned(),
            retry_after_ms: Some(2_000),
        });
        assert!(matches!(
            rate_limited,
            AdapterError::RateLimited {
                retry_after_ms: Some(2_000),
                ..
            }
        ));
        assert!(AdapterError::from(LlmCallError::Cancelled).is_cancelled());
    }

    #[test]
    fn http_statuses_keep_authentication_and_rate_limit_categories() {
        assert!(matches!(
            AdapterError::from_http_status("test", 401, "denied", None),
            AdapterError::Authentication { .. }
        ));
        assert!(matches!(
            AdapterError::from_http_status("test", 429, "limited", Some(500)),
            AdapterError::RateLimited {
                retry_after_ms: Some(500),
                ..
            }
        ));
        assert!(matches!(
            AdapterError::from_http_status("test", 503, "down", None),
            AdapterError::ServiceRejected {
                status: Some(503),
                ..
            }
        ));
    }

    #[test]
    fn external_io_keeps_the_original_error_source() {
        let error = AdapterError::external_io(
            "read test file",
            Some(PathBuf::from("missing.txt")),
            io::Error::new(io::ErrorKind::NotFound, "missing"),
        );
        assert!(error.is_not_found());
        let source = std::error::Error::source(&error).expect("I/O source");
        assert_eq!(source.to_string(), "missing");
    }
}
