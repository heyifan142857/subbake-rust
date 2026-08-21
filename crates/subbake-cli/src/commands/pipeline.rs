use crate::CliResult;
use crate::args::TranslateArgs;
use crate::output::print_pipeline_outcome;
use subbake_adapters::{PipelineRequest, run_pipeline_cancellable_with_progress};

pub fn run(args: TranslateArgs) -> CliResult<()> {
    let cancellation = crate::cancellation::CliCancellation::new()?;
    let outcome = run_pipeline_cancellable_with_progress(
        PipelineRequest {
            input_path: args.input_path,
            output_path: args.output,
            settings: args.settings,
            transcription_settings: args.transcription_settings,
        },
        cancellation.guard(),
        std::sync::Arc::new(crate::progress::CliProgress::new()),
    )?;
    print_pipeline_outcome(&outcome, args.json)?;
    Ok(())
}
