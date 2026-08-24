use std::path::PathBuf;

use crate::CliResult;
use crate::args::{BatchArgs, TranslateArgs};
use crate::output::{print_batch_translation_outcome, print_translation_outcome};
use subbake_adapters::{
    BatchTranslationRequest, RuntimeReusePolicy, TranslationRequest,
    translate_input_cancellable_with_progress_and_quality,
    translate_subtitle_batch_with_progress_and_quality,
};

pub fn translate_file(args: TranslateArgs) -> CliResult<Option<PathBuf>> {
    let cancellation = crate::cancellation::CliCancellation::new()?;
    let outcome = translate_input_cancellable_with_progress_and_quality(
        TranslationRequest {
            input_path: args.input_path.clone(),
            output_path: args.output.clone(),
            output_language_tag: None,
            overwrite: args.overwrite,
            runtime_reuse: RuntimeReusePolicy::Configured,
            settings: args.settings.clone(),
        },
        cancellation.guard(),
        std::sync::Arc::new(crate::progress::CliProgress::new()),
        args.qa_fail_on,
    )?;
    Ok(print_translation_outcome(&outcome, args.json)?)
}

pub fn translate_batch(args: BatchArgs) -> CliResult<()> {
    let cancellation = crate::cancellation::CliCancellation::new()?;
    let outcome = translate_subtitle_batch_with_progress_and_quality(
        BatchTranslationRequest {
            root: args.dir,
            recursive: args.recursive,
            overwrite: args.overwrite,
            fail_fast: args.fail_fast,
            retry_manifest: args.retry_failed,
            output_dir: None,
            output_language_tag: None,
            runtime_reuse: RuntimeReusePolicy::Configured,
            settings: args.translate.settings,
        },
        cancellation.guard(),
        std::sync::Arc::new(crate::progress::CliProgress::new()),
        args.qa_fail_on,
    )?;
    print_batch_translation_outcome(&outcome, args.json);
    Ok(())
}
