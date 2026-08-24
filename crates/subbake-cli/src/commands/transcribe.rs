use crate::CliResult;
use crate::args::TranscribeArgs;
use crate::output::print_transcription_outcome;
use subbake_adapters::{
    TranscriptionRequest, transcribe_media_cancellable_with_progress_and_quality,
};

pub fn run(args: TranscribeArgs) -> CliResult<()> {
    let cancellation = crate::cancellation::CliCancellation::new()?;
    let outcome = transcribe_media_cancellable_with_progress_and_quality(
        TranscriptionRequest {
            media_path: args.media_path,
            output_path: args.output,
            overwrite: args.overwrite,
            settings: args.settings,
        },
        cancellation.guard(),
        std::sync::Arc::new(crate::progress::CliProgress::new()),
        args.qa_fail_on,
    )?;
    print_transcription_outcome(&outcome, args.json);
    Ok(())
}
