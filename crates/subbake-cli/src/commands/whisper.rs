use subbake_adapters::{WhisperRequest, run_whisper_cancellable_with_progress};
use subbake_core::CancellationGuard;

use crate::CliResult;
use crate::args::WhisperArgs;
use crate::output::print_whisper_outcome;

pub fn run(args: WhisperArgs) -> CliResult<()> {
    let outcome = run_whisper_cancellable_with_progress(
        WhisperRequest {
            action: args.action,
            binary_path: args.binary_path,
            models_dir: args.models_dir,
            build_variant: args.build_variant,
        },
        &CancellationGuard::never(),
        std::sync::Arc::new(crate::progress::CliProgress::new()),
    )?;
    print_whisper_outcome(&outcome);
    Ok(())
}
