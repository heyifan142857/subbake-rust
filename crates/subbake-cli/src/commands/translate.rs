use std::path::PathBuf;

use subbake_adapters::{
    BatchTranslationRequest, TranslationRequest, translate_subtitle_batch_with_progress,
    translate_subtitle_cancellable_with_progress,
};
use subbake_core::CancellationGuard;

use crate::CliResult;
use crate::args::{BatchArgs, TranslateArgs};
use crate::output::{print_batch_translation_outcome, print_translation_outcome};

pub fn translate_file(args: TranslateArgs) -> CliResult<Option<PathBuf>> {
    let outcome = translate_subtitle_cancellable_with_progress(
        TranslationRequest {
            input_path: args.input_path.clone(),
            output_path: args.output.clone(),
            settings: args.settings.clone(),
        },
        &CancellationGuard::never(),
        std::sync::Arc::new(crate::progress::CliProgress::new()),
    )?;
    Ok(print_translation_outcome(&outcome, args.json)?)
}

pub fn translate_batch(args: BatchArgs) -> CliResult<()> {
    let outcome = translate_subtitle_batch_with_progress(
        BatchTranslationRequest {
            root: args.dir,
            recursive: args.recursive,
            overwrite: args.overwrite,
            settings: args.translate.settings,
        },
        &CancellationGuard::never(),
        std::sync::Arc::new(crate::progress::CliProgress::new()),
    )?;
    print_batch_translation_outcome(&outcome);
    Ok(())
}
