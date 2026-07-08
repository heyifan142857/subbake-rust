use std::io;
use std::path::PathBuf;

use subbake_adapters::{
    BatchTranslationRequest, TranslationRequest, translate_subtitle, translate_subtitle_batch,
};

use crate::args::{BatchArgs, TranslateArgs};
use crate::output::{print_batch_translation_outcome, print_translation_outcome};

pub fn translate_file(args: TranslateArgs) -> io::Result<Option<PathBuf>> {
    let outcome = translate_subtitle(TranslationRequest {
        input_path: args.input_path.clone(),
        output_path: args.output.clone(),
        settings: args.settings.clone(),
    })?;
    print_translation_outcome(&outcome, args.json)
}

pub fn translate_batch(args: BatchArgs) -> io::Result<()> {
    let outcome = translate_subtitle_batch(BatchTranslationRequest {
        root: args.dir,
        recursive: args.recursive,
        overwrite: args.overwrite,
        settings: args.translate.settings,
    })?;
    print_batch_translation_outcome(&outcome);
    Ok(())
}
