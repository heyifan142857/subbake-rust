use subbake_adapters::{AdapterError, read_document};
use subbake_core::{QualityPolicy, inspect_quality};

use crate::args::{QaArgs, QaFailOn};
use crate::{CliError, CliResult};

pub fn run(args: QaArgs) -> CliResult<()> {
    let document = read_document(&args.subtitle_path)?;
    let report = inspect_quality(&document, QualityPolicy::default());
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|error| {
                AdapterError::invalid_input(format!("encode QA report: {error}"))
            })?
        );
    } else {
        println!("Segments: {}", report.segments);
        println!(
            "QA findings: {} error(s), {} warning(s)",
            report.errors, report.warnings
        );
        for issue in &report.issues {
            println!(
                "  {:?} [{}] {:?}: {}",
                issue.severity, issue.segment_id, issue.kind, issue.message
            );
        }
    }

    let failed = match args.fail_on {
        QaFailOn::Never => false,
        QaFailOn::Error => report.errors > 0,
        QaFailOn::Warning => report.errors > 0 || report.warnings > 0,
    };
    if failed {
        return Err(CliError::usage(format!(
            "subtitle QA threshold failed with {} error(s) and {} warning(s)",
            report.errors, report.warnings
        )));
    }
    Ok(())
}
