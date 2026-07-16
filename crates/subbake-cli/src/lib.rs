pub mod args;
pub mod commands;
pub mod error;
pub mod output;
mod progress;

pub use error::{CliError, CliResult};

pub fn command_names() -> &'static [&'static str] {
    &[
        "agent",
        "resume",
        "translate",
        "batch",
        "transcribe",
        "pipeline",
        "provider",
        "runtime",
        "whisper",
    ]
}

pub fn run(args: Vec<String>) -> CliResult<()> {
    commands::dispatch(args)
}
