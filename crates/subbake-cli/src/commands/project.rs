use subbake_adapters::{ProjectInspectionRequest, inspect_subtitle_project, write_file_atomically};
use subbake_core::{ProjectConsistencyIssue, ProjectTranslationStatus};

use crate::args::ProjectArgs;
use crate::{CliError, CliResult};

pub fn run(args: ProjectArgs) -> CliResult<()> {
    let report = inspect_subtitle_project(ProjectInspectionRequest {
        root: args.root,
        recursive: args.recursive,
    })?;
    let encoded = serde_json::to_vec_pretty(&report).map_err(|error| {
        subbake_adapters::AdapterError::invalid_input(format!("encode project report: {error}"))
    })?;
    if let Some(path) = &args.output {
        write_file_atomically(path, &encoded)?;
    }

    if args.json {
        crate::output::print_json_value(
            "project_report",
            serde_json::to_value(&report).map_err(|error| {
                subbake_adapters::AdapterError::invalid_input(format!(
                    "encode project report: {error}"
                ))
            })?,
        );
    } else {
        println!("Project: {}", report.root);
        println!(
            "Files: {} total, {} pending, {} translated, {} bilingual",
            report.summary.files,
            report.summary.pending,
            report.summary.translated,
            report.summary.bilingual
        );
        println!(
            "QA: {} error(s), {} warning(s); consistency: {} issue(s)",
            report.summary.qa_errors, report.summary.qa_warnings, report.summary.consistency_issues
        );
        for file in &report.files {
            let status = match file.status {
                ProjectTranslationStatus::Pending => "pending",
                ProjectTranslationStatus::Translated => "translated",
                ProjectTranslationStatus::Bilingual => "bilingual",
            };
            println!(
                "  [{status}] {} ({} segments)",
                file.source_path, file.segments
            );
        }
        for issue in &report.consistency_issues {
            match issue {
                ProjectConsistencyIssue::SegmentAlignment {
                    source_path,
                    output_path,
                    source_segments,
                    output_segments,
                } => println!(
                    "  alignment: {source_path} -> {output_path} ({source_segments} vs {output_segments})"
                ),
                ProjectConsistencyIssue::DivergentTranslation {
                    source_text,
                    translations,
                    ..
                } => println!(
                    "  divergent: {:?} has {} translations: {}",
                    source_text,
                    translations.len(),
                    translations.join(" | ")
                ),
            }
        }
        if let Some(path) = &args.output {
            println!("Report: {}", path.display());
        }
    }

    let qa_failed = args.fail_on.fails(&subbake_core::QualityReport {
        version: 1,
        segments: report.summary.segments,
        errors: report.summary.qa_errors,
        warnings: report.summary.qa_warnings,
        issues: Vec::new(),
    });
    if qa_failed
        || report.summary.consistency_issues > 0 && args.fail_on != subbake_core::QualityGate::Never
    {
        return Err(CliError::usage(format!(
            "project preflight failed with {} QA error(s), {} QA warning(s), and {} consistency issue(s)",
            report.summary.qa_errors, report.summary.qa_warnings, report.summary.consistency_issues
        )));
    }
    Ok(())
}
