use crate::CliResult;
use crate::args::WhisperArgs;
use crate::output::print_whisper_outcome;
use subbake_adapters::{WhisperRequest, run_whisper_cancellable_with_progress};

pub fn run(args: WhisperArgs) -> CliResult<()> {
    let cancellation = crate::cancellation::CliCancellation::new()?;
    let outcome = run_whisper_cancellable_with_progress(
        WhisperRequest {
            action: args.action,
            binary_path: args.binary_path,
            models_dir: args.models_dir,
            build_variant: args.build_variant,
        },
        cancellation.guard(),
        std::sync::Arc::new(crate::progress::CliProgress::new()),
    )?;
    print_whisper_outcome(&outcome, args.json);
    Ok(())
}
