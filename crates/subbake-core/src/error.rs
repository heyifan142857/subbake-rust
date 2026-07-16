use std::error::Error;
use std::fmt::{Display, Formatter};

pub type CoreResult<T> = Result<T, CoreError>;

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
    Backend(String),
    Data(String),
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
            CoreError::Backend(value) => write!(formatter, "backend error: {value}"),
            CoreError::Data(value) => write!(formatter, "data error: {value}"),
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
            CoreError::Backend(message) => Self::Rejected {
                status: None,
                message,
            },
            other => Self::InvalidResponse(other.to_string()),
        }
    }
}
