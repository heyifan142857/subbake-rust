use std::error::Error;
use std::fmt::{Display, Formatter};

pub type CoreResult<T> = Result<T, CoreError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    UnsupportedFormat(String),
    MalformedSubtitle(String),
    InvalidTranslation(String),
    Backend(String),
    Data(String),
}

impl Display for CoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CoreError::UnsupportedFormat(value) => write!(formatter, "unsupported subtitle format: {value}"),
            CoreError::MalformedSubtitle(value) => write!(formatter, "malformed subtitle document: {value}"),
            CoreError::InvalidTranslation(value) => write!(formatter, "invalid translation result: {value}"),
            CoreError::Backend(value) => write!(formatter, "backend error: {value}"),
            CoreError::Data(value) => write!(formatter, "data error: {value}"),
        }
    }
}

impl Error for CoreError {}
