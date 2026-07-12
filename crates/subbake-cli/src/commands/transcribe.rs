use std::io;

use subbake_adapters::{TranscriptionRequest, transcribe_media_cancellable_with_progress};
use subbake_core::CancellationGuard;

use crate::args::TranscribeArgs;
use crate::output::print_transcription_outcome;

pub fn run(args: TranscribeArgs) -> io::Result<()> {
    let outcome = transcribe_media_cancellable_with_progress(
        TranscriptionRequest {
            media_path: args.media_path,
            output_path: args.output,
            settings: args.settings,
        },
        &CancellationGuard::never(),
        std::sync::Arc::new(crate::progress::CliProgress::new()),
    )?;
    print_transcription_outcome(&outcome);
    Ok(())
}
