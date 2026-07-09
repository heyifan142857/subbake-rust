use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticReport {
    pub source: String,
    pub diagnosis: String,
    pub suggestions: Vec<String>,
    pub details: Vec<String>,
}

pub fn diagnose_text(text: &str, source: impl Into<String>) -> DiagnosticReport {
    let errors = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    build_report(source.into(), &errors, &metadata_from_text(text))
}

pub fn diagnose_failure_value(
    value: &serde_json::Value,
    source: impl Into<String>,
) -> DiagnosticReport {
    let mut errors = Vec::new();
    let mut metadata = Vec::new();
    for key in ["attempts", "agent_attempts"] {
        let Some(attempts) = value.get(key).and_then(serde_json::Value::as_array) else {
            continue;
        };
        for attempt in attempts {
            collect_attempt(attempt, &mut errors, &mut metadata);
        }
    }

    let mut report = build_report(source.into(), &errors, &metadata);
    let mut details = Vec::new();
    if let (Some(stage), Some(batch)) = (
        value.get("stage").and_then(serde_json::Value::as_str),
        value.get("batch_index").and_then(serde_json::Value::as_u64),
    ) {
        details.push(format!("{stage} batch {batch}"));
    }
    if let Some(error) = errors.last() {
        details.push(format!("Last error: {error}"));
    }
    for item in metadata.iter().rev().take(2).rev() {
        let mut parts = Vec::new();
        if let Some(status) = item.status_code {
            parts.push(format!("status={status}"));
        }
        if let Some(request_id) = &item.request_id {
            parts.push(format!("request_id={request_id}"));
        }
        if let Some(reason) = &item.reason {
            parts.push(format!("reason={reason}"));
        }
        if !parts.is_empty() {
            details.push(parts.join(", "));
        }
    }
    report.details = details;
    report
}

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

#[derive(Debug, Default)]
struct ErrorMetadata {
    status_code: Option<u64>,
    request_id: Option<String>,
    reason: Option<String>,
}

fn collect_attempt(
    attempt: &serde_json::Value,
    errors: &mut Vec<String>,
    metadata: &mut Vec<ErrorMetadata>,
) {
    if let Some(error) = attempt
        .get("error")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|error| !error.is_empty())
    {
        errors.push(error.to_owned());
    }
    if let Some(value) = attempt.get("error_meta") {
        metadata.push(metadata_from_value(value));
    }
    if let Some(split) = attempt.get("split_retry") {
        collect_attempt(split, errors, metadata);
    }
}

fn metadata_from_value(value: &serde_json::Value) -> ErrorMetadata {
    ErrorMetadata {
        status_code: value.get("status_code").and_then(serde_json::Value::as_u64),
        request_id: value
            .get("request_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        reason: value
            .get("reason")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
    }
}

fn metadata_from_text(text: &str) -> Vec<ErrorMetadata> {
    let lower = text.to_lowercase();
    let mut metadata = Vec::new();
    for status in [429, 500] {
        if lower.contains(&format!("status={status}"))
            || (lower.contains("status_code") && lower.contains(&status.to_string()))
        {
            metadata.push(ErrorMetadata {
                status_code: Some(status),
                ..ErrorMetadata::default()
            });
        }
    }
    if lower.contains("timeout") || lower.contains("timed out") {
        metadata.push(ErrorMetadata {
            reason: Some("timeout".to_owned()),
            ..ErrorMetadata::default()
        });
    }
    metadata
}

fn build_report(source: String, errors: &[String], metadata: &[ErrorMetadata]) -> DiagnosticReport {
    let (diagnosis, suggestions) = classify(errors, metadata);
    DiagnosticReport {
        source,
        diagnosis,
        suggestions,
        details: errors.iter().rev().take(4).rev().cloned().collect(),
    }
}

fn classify(errors: &[String], metadata: &[ErrorMetadata]) -> (String, Vec<String>) {
    for error in errors.iter().rev() {
        let lower = error.to_lowercase();
        if lower.contains("line count mismatch")
            || (lower.contains("expected")
                && lower.contains("translated line")
                && lower.contains("got"))
        {
            return diagnosis(
                "Model output dropped, inserted, or merged subtitle entries.",
                &[
                    "Rerun with a smaller --batch-size.",
                    "Keep --resume enabled so completed batches are reused.",
                    "If this repeats, try a stronger model profile.",
                ],
            );
        }
        if lower.contains("empty translation") {
            return diagnosis(
                "Model returned an empty translation for a non-empty subtitle.",
                &[
                    "Rerun with a smaller --batch-size or without --fast.",
                    "Inspect the saved batch for short fragments or malformed tags.",
                ],
            );
        }
        if lower.contains("id mismatch") || lower.contains("unexpected translated id") {
            return diagnosis(
                "Model changed subtitle ids instead of preserving the source structure.",
                &[
                    "Rerun the same command so the correction prompt can retry.",
                    "If it repeats, lower --batch-size or switch profiles.",
                ],
            );
        }
    }

    for item in metadata.iter().rev() {
        match item.status_code {
            Some(429) => {
                return diagnosis(
                    "Provider rate limit was hit.",
                    &[
                        "Wait and rerun with resume enabled.",
                        "Use a smaller --batch-size or a different profile.",
                    ],
                );
            }
            Some(status) if (500..=599).contains(&status) => {
                return diagnosis(
                    "Provider returned a temporary server-side error.",
                    &["Rerun later with resume enabled so finished batches are reused."],
                );
            }
            _ if item.reason.is_some() => {
                return diagnosis(
                    "Request failed before receiving a valid model response.",
                    &["Check network, API base URL, and provider credentials."],
                );
            }
            _ => {}
        }
    }

    let combined = errors.join("\n").to_lowercase();
    if combined.contains("missing api key") || combined.contains("credential") {
        return diagnosis(
            "Provider credentials are missing or invalid.",
            &["Run `sbake provider check` for the active profile."],
        );
    }
    if combined.contains("unsupported input format") || combined.contains("unsupported format") {
        return diagnosis(
            "The referenced file is not a supported subtitle format.",
            &["Use .srt, .vtt, or .txt input files."],
        );
    }
    if combined.contains("timeout") || combined.contains("timed out") {
        return diagnosis(
            "The provider request timed out.",
            &["Use a faster model profile or retry later."],
        );
    }
    if errors.is_empty() {
        diagnosis(
            "No specific failure pattern was found.",
            &["Share a SubBake failure JSON or paste the terminal error text."],
        )
    } else {
        diagnosis(
            "The failure is not one of SubBake's known structural cases.",
            &["Inspect the saved failure payload and request metadata."],
        )
    }
}

fn diagnosis(summary: &str, suggestions: &[&str]) -> (String, Vec<String>) {
    (
        summary.to_owned(),
        suggestions
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnoses_credentials() {
        let report = diagnose_text(
            "Error: Missing API key for OpenAI provider.",
            "terminal output",
        );
        assert!(report.diagnosis.contains("credentials"));
        assert_eq!(report.details.len(), 1);
    }

    #[test]
    fn diagnoses_structured_rate_limit() {
        let report = diagnose_failure_value(
            &serde_json::json!({
                "stage": "translate",
                "batch_index": 2,
                "attempts": [{
                    "error": "provider rejected request",
                    "error_meta": {"status_code": 429, "request_id": "req-1"}
                }]
            }),
            "failure.json",
        );
        assert!(report.diagnosis.contains("rate limit"));
        assert!(report.details.iter().any(|line| line.contains("req-1")));
    }
}
