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

    pub(crate) fn tool_failure_category(&self) -> String {
        match self {
            Self::Cancelled => "cancelled".to_owned(),
            Self::InvalidInput { .. } => "invalid_input".to_owned(),
            Self::InvalidDecision { .. } => "invalid_decision".to_owned(),
            Self::ToolArguments { .. } => "invalid_arguments".to_owned(),
            Self::ToolPolicy { .. } => "tool_policy".to_owned(),
            Self::InvalidState { .. } => "invalid_state".to_owned(),
            Self::FileGuard(_) => "file_guard".to_owned(),
            Self::SessionStorage { .. } | Self::SessionData { .. } => "session_storage".to_owned(),
            Self::Adapter(source) => adapter_failure_category(source),
            Self::AdapterContext { source, .. } => adapter_failure_category(source),
            Self::Core(_) => "core".to_owned(),
            Self::WorkerStopped | Self::WorkerPanicked => "worker".to_owned(),
            Self::Reported { source, .. } => source.tool_failure_category(),
            Self::Io { .. } => "external_io".to_owned(),
        }
    }
}

fn adapter_failure_category(error: &AdapterError) -> String {
    match error {
        AdapterError::Cancelled => "cancelled".to_owned(),
        AdapterError::InvalidInput { .. } => "invalid_input".to_owned(),
        AdapterError::Configuration(_) | AdapterError::ConfigurationFile { .. } => {
            "configuration".to_owned()
        }
        AdapterError::Authentication { .. } => "authentication".to_owned(),
        AdapterError::RateLimited { .. } => "rate_limited".to_owned(),
        AdapterError::Timeout { .. } => "timeout".to_owned(),
        AdapterError::Transport { .. } => "transport".to_owned(),
        AdapterError::ServiceRejected { .. } => "service_rejected".to_owned(),
        AdapterError::ExternalIo { .. } => "external_io".to_owned(),
        AdapterError::Serialization { .. } => "serialization".to_owned(),
        AdapterError::ChildProcess { program, .. } => format!("child_process:{program}"),
        AdapterError::Core(_) | AdapterError::CoreContext { .. } => "core".to_owned(),
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
