use crate::CliResult;
use crate::args::{OvernightAction, OvernightArgs};
use crate::output::print_json_value;
use subbake_adapters::{
    OvernightCollectRequest, OvernightStatusRequest, OvernightSubmitRequest, collect_overnight,
    overnight_status, submit_overnight,
};

pub fn run(args: OvernightArgs) -> CliResult<()> {
    let cancellation = crate::cancellation::CliCancellation::new()?;
    match args.action {
        OvernightAction::Submit(args) => {
            let json = args.json;
            let outcome = submit_overnight(
                OvernightSubmitRequest {
                    input_path: args.input_path,
                    output_path: args.output,
                    settings: args.settings,
                },
                cancellation.guard(),
            )?;
            if json {
                print_json_value(
                    "overnight_submit_result",
                    serde_json::json!({
                        "job_id": outcome.job_id,
                        "requests": outcome.requests,
                        "manifest_path": outcome.manifest_path,
                    }),
                );
                return Ok(());
            }
            println!("Submitted overnight job: {}", outcome.job_id);
            println!("Requests: {}", outcome.requests);
            println!("Manifest: {}", outcome.manifest_path.display());
        }
        OvernightAction::Status(args) => {
            let json = args.json;
            let outcome = overnight_status(
                OvernightStatusRequest {
                    manifest_path: args.input_path,
                    settings: args.settings,
                },
                cancellation.guard(),
            )?;
            if json {
                print_json_value(
                    "overnight_status_result",
                    serde_json::json!({
                        "job_id": outcome.job_id,
                        "status": outcome.status,
                        "completed": outcome.completed,
                        "failed": outcome.failed,
                        "total": outcome.total,
                        "manifest_path": outcome.manifest_path,
                    }),
                );
                return Ok(());
            }
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
            let json = args.json;
            let outcome = collect_overnight(
                OvernightCollectRequest {
                    manifest_path: args.input_path,
                    settings: args.settings,
                    overwrite,
                },
                cancellation.guard(),
            )?;
            if json {
                print_json_value(
                    "overnight_collect_result",
                    serde_json::json!({
                        "manifest_path": outcome.manifest_path,
                        "output_path": outcome.output_path,
                        "translated_segments": outcome.translated_segments,
                    }),
                );
                return Ok(());
            }
            println!("Output: {}", outcome.output_path.display());
            println!(
                "Collected {} translated subtitle entries.",
                outcome.translated_segments
            );
        }
    }
    Ok(())
}
