use std::io;

use subbake_adapters::AdapterError;
use subbake_core::CoreError;
use thiserror::Error;

pub type AgentResult<T> = Result<T, AgentError>;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("operation cancelled")]
    Cancelled,
    #[error("{message}")]
    InvalidInput { message: String },
    #[error("{message}")]
    InvalidDecision { message: String },
    #[error("{message}")]
    ToolArguments { message: String },
    #[error("{message}")]
    ToolPolicy { message: String },
    #[error("{message}")]
    InvalidState { message: String },
    #[error(transparent)]
    FileGuard(#[from] crate::guard::FileGuardError),
    #[error("{operation}{path_suffix}: {source}", path_suffix = path.as_ref().map(|value| format!(" `{}`", value.display())).unwrap_or_default())]
    SessionStorage {
        operation: &'static str,
        path: Option<std::path::PathBuf>,
        #[source]
        source: io::Error,
    },
    #[error("{operation} `{path}`: {source}")]
    SessionData {
        operation: &'static str,
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(transparent)]
    Adapter(Box<AdapterError>),
    #[error("{operation}: {source}")]
    AdapterContext {
        operation: &'static str,
        #[source]
        source: Box<AdapterError>,
    },
    #[error(transparent)]
    Core(CoreError),
    #[error("agent worker stopped")]
    WorkerStopped,
    #[error("agent worker panicked")]
    WorkerPanicked,
    #[error("{message}")]
    Reported {
        message: String,
        #[source]
        source: Box<AgentError>,
    },
    #[error("{context}: {source}")]
    Io {
        context: &'static str,
        #[source]
        source: io::Error,
    },
}

impl AgentError {
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    pub fn invalid_state(message: impl Into<String>) -> Self {
        Self::InvalidState {
            message: message.into(),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        match self {
            Self::Cancelled => true,
            Self::Adapter(source) => source.is_cancelled(),
            Self::AdapterContext { source, .. } => source.is_cancelled(),
            Self::Reported { source, .. } => source.is_cancelled(),
            _ => false,
        }
    }
}

impl From<CoreError> for AgentError {
    fn from(error: CoreError) -> Self {
        match error {
            CoreError::Cancelled => Self::Cancelled,
            other => Self::Core(other),
        }
    }
}

impl From<AdapterError> for AgentError {
    fn from(error: AdapterError) -> Self {
        Self::Adapter(Box::new(error))
    }
}

impl From<io::Error> for AgentError {
    fn from(source: io::Error) -> Self {
        if source.kind() == io::ErrorKind::Interrupted {
            Self::Cancelled
        } else {
            Self::Io {
                context: "agent I/O failed",
                source,
            }
        }
    }
}
