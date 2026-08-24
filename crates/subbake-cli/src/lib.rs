pub mod args;
mod cancellation;
pub mod commands;
pub mod error;
pub mod output;
mod progress;
mod version;

pub use error::{CliError, CliResult};

pub fn command_names() -> Vec<&'static str> {
    commands::command_specs()
        .iter()
        .map(|spec| spec.name)
        .collect()
}

pub fn run(args: Vec<String>) -> CliResult<()> {
    commands::dispatch(args)
}
