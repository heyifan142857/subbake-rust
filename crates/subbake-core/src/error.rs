use std::error::Error;
use std::fmt::{Display, Formatter};

pub type CoreResult<T> = Result<T, CoreError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageIoKind {
    NotFound,
    PermissionDenied,
    AlreadyExists,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    Io {
        operation: String,
        path: Option<String>,
        kind: StorageIoKind,
        message: String,
    },
    Serialization {
        operation: String,
        message: String,
    },
    CorruptData {
        data_kind: String,
        path: Option<String>,
        message: String,
    },
}

impl Display for StorageError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io {
                operation,
                path,
                message,
                ..
            } => match path {
                Some(path) => write!(formatter, "{operation} `{path}`: {message}"),
                None => write!(formatter, "{operation}: {message}"),
            },
            Self::Serialization { operation, message } => {
                write!(formatter, "{operation}: {message}")
            }
            Self::CorruptData {
                data_kind,
                path,
                message,
            } => match path {
                Some(path) => write!(formatter, "invalid {data_kind} in `{path}`: {message}"),
                None => write!(formatter, "invalid {data_kind}: {message}"),
            },
        }
    }
}

impl Error for StorageError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmCallError {
    Cancelled,
    Timeout(String),
    Authentication(String),
    RateLimited {
        message: String,
        retry_after_ms: Option<u64>,
    },
    Transport(String),
    Rejected {
        status: Option<u16>,
        message: String,
    },
    InvalidResponse(String),
    UnsupportedCapability(String),
    ContinuationMismatch(String),
}

impl LlmCallError {
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Timeout(_) | Self::RateLimited { .. } | Self::Transport(_)
        ) || matches!(
            self,
            Self::Rejected {
                status: Some(500..=599),
                ..
            }
        )
    }
}

impl Display for LlmCallError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => write!(formatter, "cancelled"),
            Self::Timeout(message) => write!(formatter, "LLM request timed out: {message}"),
            Self::Authentication(message) => {
                write!(formatter, "LLM authentication failed: {message}")
            }
            Self::RateLimited { message, .. } => {
                write!(formatter, "LLM request was rate limited: {message}")
            }
            Self::Transport(message) => write!(formatter, "LLM transport failed: {message}"),
            Self::Rejected { status, message } => match status {
                Some(status) => write!(formatter, "LLM request was rejected ({status}): {message}"),
                None => write!(formatter, "LLM request was rejected: {message}"),
            },
            Self::InvalidResponse(message) => {
                write!(formatter, "LLM response was invalid: {message}")
            }
            Self::UnsupportedCapability(capability) => {
                write!(formatter, "unsupported LLM capability: {capability}")
            }
            Self::ContinuationMismatch(message) => {
                write!(formatter, "LLM continuation mismatch: {message}")
            }
        }
    }
}

impl Error for LlmCallError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    Cancelled,
    Llm(LlmCallError),
    UnsupportedFormat(String),
    MalformedSubtitle(String),
    InvalidTranslation(String),
    UnsupportedCapability(String),
    InvalidBackendResponse(String),
    DataInvariant(String),
    Storage(StorageError),
}

impl Display for CoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CoreError::Cancelled => write!(formatter, "cancelled"),
            CoreError::Llm(error) => Display::fmt(error, formatter),
            CoreError::UnsupportedFormat(value) => {
                write!(formatter, "unsupported subtitle format: {value}")
            }
            CoreError::MalformedSubtitle(value) => {
                write!(formatter, "malformed subtitle document: {value}")
            }
            CoreError::InvalidTranslation(value) => {
                write!(formatter, "invalid translation result: {value}")
            }
            CoreError::UnsupportedCapability(value) => {
                write!(formatter, "unsupported backend capability: {value}")
            }
            CoreError::InvalidBackendResponse(value) => {
                write!(formatter, "invalid backend response: {value}")
            }
            CoreError::DataInvariant(value) => write!(formatter, "data invariant failed: {value}"),
            CoreError::Storage(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for CoreError {}

impl From<LlmCallError> for CoreError {
    fn from(error: LlmCallError) -> Self {
        match error {
            LlmCallError::Cancelled => Self::Cancelled,
            LlmCallError::UnsupportedCapability(capability) => {
                Self::UnsupportedCapability(capability)
            }
            other => Self::Llm(other),
        }
    }
}

impl From<CoreError> for LlmCallError {
    fn from(error: CoreError) -> Self {
        match error {
            CoreError::Cancelled => Self::Cancelled,
            CoreError::UnsupportedCapability(capability) => Self::UnsupportedCapability(capability),
            CoreError::Llm(error) => error,
            CoreError::InvalidBackendResponse(message) => Self::InvalidResponse(message),
            CoreError::UnsupportedFormat(value) => {
                Self::InvalidResponse(format!("unsupported subtitle format: {value}"))
            }
            CoreError::MalformedSubtitle(value) => {
                Self::InvalidResponse(format!("malformed subtitle document: {value}"))
            }
            CoreError::InvalidTranslation(value) => {
                Self::InvalidResponse(format!("invalid translation result: {value}"))
            }
            CoreError::DataInvariant(value) => {
                Self::InvalidResponse(format!("data error: {value}"))
            }
            CoreError::Storage(error) => {
                Self::InvalidResponse(format!("runtime storage error: {error}"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_to_llm_conversion_is_explicit_and_category_preserving() {
        assert_eq!(
            LlmCallError::from(CoreError::Cancelled),
            LlmCallError::Cancelled
        );
        assert_eq!(
            LlmCallError::from(CoreError::UnsupportedCapability("tools".to_owned())),
            LlmCallError::UnsupportedCapability("tools".to_owned())
        );
        assert_eq!(
            LlmCallError::from(CoreError::InvalidBackendResponse("bad JSON".to_owned())),
            LlmCallError::InvalidResponse("bad JSON".to_owned())
        );
    }

    #[test]
    fn storage_io_remains_distinguishable_from_domain_invariants() {
        let error = CoreError::Storage(StorageError::Io {
            operation: "read cache".to_owned(),
            path: Some("cache.json".to_owned()),
            kind: StorageIoKind::PermissionDenied,
            message: "denied".to_owned(),
        });
        assert!(matches!(
            error,
            CoreError::Storage(StorageError::Io {
                kind: StorageIoKind::PermissionDenied,
                ..
            })
        ));
    }
}
