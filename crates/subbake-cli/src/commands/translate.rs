use std::path::PathBuf;

use crate::CliResult;
use crate::args::{BatchArgs, TranslateArgs};
use crate::output::{print_batch_translation_outcome, print_translation_outcome};
use subbake_adapters::{
    BatchTranslationRequest, TranslationRequest, translate_subtitle_batch_with_progress,
    translate_subtitle_cancellable_with_progress,
};

pub fn translate_file(args: TranslateArgs) -> CliResult<Option<PathBuf>> {
    let cancellation = crate::cancellation::CliCancellation::new()?;
    let outcome = translate_subtitle_cancellable_with_progress(
        TranslationRequest {
            input_path: args.input_path.clone(),
            output_path: args.output.clone(),
            output_language_tag: None,
            overwrite: true,
            settings: args.settings.clone(),
        },
        cancellation.guard(),
        std::sync::Arc::new(crate::progress::CliProgress::new()),
    )?;
    Ok(print_translation_outcome(&outcome, args.json)?)
}

pub fn translate_batch(args: BatchArgs) -> CliResult<()> {
    let cancellation = crate::cancellation::CliCancellation::new()?;
    let outcome = translate_subtitle_batch_with_progress(
        BatchTranslationRequest {
            root: args.dir,
            recursive: args.recursive,
            overwrite: args.overwrite,
            fail_fast: args.fail_fast,
            retry_manifest: args.retry_failed,
            output_dir: None,
            output_language_tag: None,
            settings: args.translate.settings,
        },
        cancellation.guard(),
        std::sync::Arc::new(crate::progress::CliProgress::new()),
    )?;
    print_batch_translation_outcome(&outcome);
    Ok(())
}
