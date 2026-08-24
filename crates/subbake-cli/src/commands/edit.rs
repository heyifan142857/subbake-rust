use crate::CliResult;
use crate::args::EditArgs;
use crate::output::print_subtitle_edit_outcome;
use subbake_adapters::{SubtitleEditRequest, edit_subtitle_cancellable};

pub fn run(args: EditArgs) -> CliResult<()> {
    let cancellation = crate::cancellation::CliCancellation::new()?;
    let outcome = edit_subtitle_cancellable(
        SubtitleEditRequest {
            target_path: args.target_path,
            instruction: args.instruction,
            settings: args.settings,
            allow_non_generated: args.allow_non_generated,
            dry_run: args.dry_run,
        },
        cancellation.guard(),
    )?;
    print_subtitle_edit_outcome(&outcome, args.json)?;
    Ok(())
}
