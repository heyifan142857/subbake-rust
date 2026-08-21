pub mod args;
mod cancellation;
pub mod commands;
pub mod error;
pub mod output;
mod progress;
mod version;

pub use error::{CliError, CliResult};

pub fn command_names() -> &'static [&'static str] {
    commands::COMMAND_NAMES
}

pub fn run(args: Vec<String>) -> CliResult<()> {
    commands::dispatch(args)
}
