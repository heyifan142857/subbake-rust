use subbake_adapters::{PipelineRequest, run_pipeline_cancellable_with_progress};
use subbake_core::CancellationGuard;

use crate::CliResult;
use crate::args::TranslateArgs;
use crate::output::print_pipeline_outcome;

pub fn run(args: TranslateArgs) -> CliResult<()> {
    let outcome = run_pipeline_cancellable_with_progress(
        PipelineRequest {
            input_path: args.input_path,
            output_path: args.output,
            settings: args.settings,
            transcription_settings: args.transcription_settings,
        },
        &CancellationGuard::never(),
        std::sync::Arc::new(crate::progress::CliProgress::new()),
    )?;
    print_pipeline_outcome(&outcome, args.json)?;
    Ok(())
}
