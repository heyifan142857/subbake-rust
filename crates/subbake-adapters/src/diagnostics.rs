use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use subbake_core::diagnostics::{DiagnosticReport, diagnose_failure_value, diagnose_text};

pub fn format_diagnostic_report(report: &DiagnosticReport) -> String {
    let mut lines = vec![
        format!("Source: {}", report.source),
        format!("Diagnosis: {}", report.diagnosis),
    ];
    if !report.details.is_empty() {
        lines.push("Details:".to_owned());
        lines.extend(report.details.iter().map(|detail| format!("- {detail}")));
    }
    if !report.suggestions.is_empty() {
        lines.push("Suggestions:".to_owned());
        lines.extend(
            report
                .suggestions
                .iter()
                .map(|suggestion| format!("- {suggestion}")),
        );
    }
    lines.join("\n")
}

pub fn load_diagnostic_reports(path: &Path) -> io::Result<Vec<DiagnosticReport>> {
    if path.is_file() {
        return Ok(vec![diagnose_failure_path(path)?]);
    }

    let mut reports = Vec::new();
    for failure in discover_failure_logs(path)? {
        reports.push(diagnose_failure_path(&failure)?);
    }
    Ok(reports)
}

pub fn diagnose_failure_path(path: &Path) -> io::Result<DiagnosticReport> {
    let content = fs::read_to_string(path)?;
    let source = path.display().to_string();
    if path.extension().is_some_and(|ext| ext == "json")
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(&content)
    {
        return Ok(diagnose_failure_value(&value, source));
    }
    Ok(diagnose_text(&content, source))
}

fn discover_failure_logs(path: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    discover_failure_logs_inner(path, &mut files)?;
    files.sort();
    Ok(files)
}

fn discover_failure_logs_inner(path: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    if path.is_file() {
        if path.extension().is_some_and(|ext| ext == "json") {
            files.push(path.to_path_buf());
        }
        return Ok(());
    }

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let child = entry.path();
        if child.is_dir() {
            discover_failure_logs_inner(&child, files)?;
        } else if child
            .parent()
            .and_then(Path::file_name)
            .is_some_and(|name| name == "failures")
            && child.extension().is_some_and(|ext| ext == "json")
        {
            files.push(child);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn loads_reports_from_failure_json() {
        let root = temp_root("diagnostics");
        let failure_dir = root.join("run/failures");
        fs::create_dir_all(&failure_dir).expect("create failures");
        let path = failure_dir.join("batch-1.json");
        fs::write(
            &path,
            serde_json::json!({
                "stage": "translate",
                "batch_index": 1,
                "attempts": [{
                    "error": "HTTP 429",
                    "error_meta": {"status_code": 429, "request_id": "req-1"}
                }]
            })
            .to_string(),
        )
        .expect("write failure");

        let reports = load_diagnostic_reports(&root).expect("reports");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(reports.len(), 1);
        assert!(reports[0].diagnosis.contains("rate limit"));
    }

    #[test]
    fn formats_report_for_human_readers_at_the_adapter_edge() {
        let report = DiagnosticReport {
            source: "failure.json".to_owned(),
            diagnosis: "Provider rate limit was hit.".to_owned(),
            suggestions: vec!["Wait and retry.".to_owned()],
            details: vec!["status=429".to_owned()],
        };

        assert_eq!(
            format_diagnostic_report(&report),
            "Source: failure.json\nDiagnosis: Provider rate limit was hit.\nDetails:\n- status=429\nSuggestions:\n- Wait and retry."
        );
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-{label}-{nanos}"))
    }
}
