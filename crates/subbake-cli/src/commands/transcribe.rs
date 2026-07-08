use std::io;

use subbake_adapters::{TranscriptionRequest, transcribe_media};

use crate::args::TranscribeArgs;
use crate::output::print_transcription_outcome;

pub fn run(args: TranscribeArgs) -> io::Result<()> {
    let outcome = transcribe_media(TranscriptionRequest {
        media_path: args.media_path,
        output_path: args.output,
        settings: args.settings,
    })?;
    print_transcription_outcome(&outcome);
    Ok(())
}
