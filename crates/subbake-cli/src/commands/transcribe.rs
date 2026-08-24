use crate::CliResult;
use crate::args::TranscribeArgs;
use crate::output::print_transcription_outcome;
use subbake_adapters::{TranscriptionRequest, transcribe_media_cancellable_with_progress};

pub fn run(args: TranscribeArgs) -> CliResult<()> {
    let cancellation = crate::cancellation::CliCancellation::new()?;
    let outcome = transcribe_media_cancellable_with_progress(
        TranscriptionRequest {
            media_path: args.media_path,
            output_path: args.output,
            overwrite: args.overwrite,
            settings: args.settings,
        },
        cancellation.guard(),
        std::sync::Arc::new(crate::progress::CliProgress::new()),
    )?;
    print_transcription_outcome(&outcome);
    Ok(())
}
