use std::io;

use subbake_adapters::is_supported_subtitle_path;

use crate::args::TranslateArgs;

use super::translate;

pub fn run(args: TranslateArgs) -> io::Result<()> {
    if is_supported_subtitle_path(&args.input_path) {
        return translate::translate_file(args).map(|_| ());
    }

    Err(io::Error::other(format!(
        "pipeline transcription is pending migration for {}; subtitle inputs are supported",
        args.input_path.display()
    )))
}
