use subbake_adapters::{AdapterError, read_document};
use subbake_core::evaluate;

use crate::CliResult;
use crate::args::EvaluateArgs;

pub fn run(args: EvaluateArgs) -> CliResult<()> {
    let candidate = read_document(&args.candidate_path)?;
    let reference = read_document(&args.reference_path)?;
    let report = evaluate(&candidate, &reference).map_err(AdapterError::from)?;
    if args.json {
        let json = serde_json::to_string_pretty(&report).map_err(|error| {
            AdapterError::invalid_input(format!("encode evaluation report: {error}"))
        })?;
        println!("{json}");
    } else {
        println!("Segments: {}", report.segments);
        println!("Exact matches: {}", report.exact_matches);
        println!("chrF: {:.4}", report.chrf);
        println!(
            "MQM mechanical findings: {} critical, {} major, {} minor",
            report.mqm.critical, report.mqm.major, report.mqm.minor
        );
    }
    Ok(())
}
