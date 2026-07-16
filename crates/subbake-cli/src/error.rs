use std::io;

use subbake_adapters::AdapterError;
use subbake_agent::AgentError;
use thiserror::Error;

pub type CliResult<T> = Result<T, CliError>;

#[derive(Debug, Error)]
pub enum CliError {
    #[error("{message}")]
    Usage { message: String },
    #[error("operation cancelled")]
    Cancelled,
    #[error(transparent)]
    Adapter(Box<AdapterError>),
    #[error(transparent)]
    Agent(Box<AgentError>),
    #[error("{context}: {source}")]
    Io {
        context: &'static str,
        #[source]
        source: io::Error,
    },
}

impl CliError {
    pub fn usage(message: impl Into<String>) -> Self {
        Self::Usage {
            message: message.into(),
        }
    }

    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Usage { .. } => 2,
            Self::Cancelled => 130,
            Self::Adapter(error) if error.is_cancelled() => 130,
            Self::Agent(error) if error.is_cancelled() => 130,
            Self::Adapter(error)
                if matches!(
                    error.as_ref(),
                    AdapterError::Configuration(_)
                        | AdapterError::ConfigurationFile { .. }
                        | AdapterError::InvalidInput { .. }
                ) =>
            {
                2
            }
            _ => 1,
        }
    }
}

impl From<AgentError> for CliError {
    fn from(error: AgentError) -> Self {
        if error.is_cancelled() {
            Self::Cancelled
        } else {
            Self::Agent(Box::new(error))
        }
    }
}

impl From<AdapterError> for CliError {
    fn from(error: AdapterError) -> Self {
        Self::Adapter(Box::new(error))
    }
}

impl From<io::Error> for CliError {
    fn from(source: io::Error) -> Self {
        match source.kind() {
            io::ErrorKind::Interrupted => Self::Cancelled,
            io::ErrorKind::InvalidInput => Self::usage(source.to_string()),
            _ => Self::Io {
                context: "I/O operation failed",
                source,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use subbake_adapters::ConfigError;

    #[test]
    fn exit_codes_follow_structured_error_categories() {
        assert_eq!(CliError::usage("bad flag").exit_code(), 2);
        assert_eq!(CliError::Cancelled.exit_code(), 130);
        assert_eq!(
            CliError::from(AdapterError::Authentication {
                message: "denied".to_owned(),
            })
            .exit_code(),
            1
        );
        assert_eq!(
            CliError::from(AdapterError::Configuration(ConfigError::invalid(
                "bad config",
            )))
            .exit_code(),
            2
        );
    }
}
