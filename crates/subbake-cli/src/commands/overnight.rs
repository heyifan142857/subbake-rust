use crate::CliResult;
use crate::args::{OvernightAction, OvernightArgs};
use subbake_adapters::{
    OvernightCollectRequest, OvernightStatusRequest, OvernightSubmitRequest, collect_overnight,
    overnight_status, submit_overnight,
};

pub fn run(args: OvernightArgs) -> CliResult<()> {
    let cancellation = crate::cancellation::CliCancellation::new()?;
    match args.action {
        OvernightAction::Submit(args) => {
            let outcome = submit_overnight(
                OvernightSubmitRequest {
                    input_path: args.input_path,
                    output_path: args.output,
                    settings: args.settings,
                },
                cancellation.guard(),
            )?;
            println!("Submitted overnight job: {}", outcome.job_id);
            println!("Requests: {}", outcome.requests);
            println!("Manifest: {}", outcome.manifest_path.display());
        }
        OvernightAction::Status(args) => {
            let outcome = overnight_status(
                OvernightStatusRequest {
                    manifest_path: args.input_path,
                    settings: args.settings,
                },
                cancellation.guard(),
            )?;
            println!("Job: {}", outcome.job_id);
            println!("Status: {}", outcome.status);
            if let Some(total) = outcome.total {
                println!(
                    "Requests: {}/{} completed, {} failed",
                    outcome.completed.unwrap_or(0),
                    total,
                    outcome.failed.unwrap_or(0)
                );
            }
        }
        OvernightAction::Collect { args, overwrite } => {
            let outcome = collect_overnight(
                OvernightCollectRequest {
                    manifest_path: args.input_path,
                    settings: args.settings,
                    overwrite,
                },
                cancellation.guard(),
            )?;
            println!("Output: {}", outcome.output_path.display());
            println!(
                "Collected {} translated subtitle entries.",
                outcome.translated_segments
            );
        }
    }
    Ok(())
}
