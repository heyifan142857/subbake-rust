use std::io;

pub mod args;
pub mod commands;
pub mod output;

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

pub fn run(args: Vec<String>) -> io::Result<()> {
    commands::dispatch(args)
}
