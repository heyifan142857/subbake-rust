use std::io;

use subbake_adapters::{WhisperRequest, run_whisper};

use crate::args::WhisperArgs;
use crate::output::print_whisper_outcome;

pub fn run(args: WhisperArgs) -> io::Result<()> {
    let outcome = run_whisper(WhisperRequest {
        action: args.action,
        binary_path: args.binary_path,
        models_dir: args.models_dir,
    })?;
    print_whisper_outcome(&outcome);
    Ok(())
}
