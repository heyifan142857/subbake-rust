use std::io;
use std::path::{Path, PathBuf};

use subbake_adapters::{
    BatchTranslationOutcome, TranscriptionOutcome, TranslationOutcome, WhisperOutcome,
};
use subbake_core::entities::{BatchPlanEntry, PipelineResult};

pub fn print_translation_outcome(
    outcome: &TranslationOutcome,
    json: bool,
) -> io::Result<Option<PathBuf>> {
    let (output, output_path) = render_translation_outcome(outcome, json)?;
    print!("{output}");
    Ok(output_path)
}

pub fn print_batch_translation_outcome(outcome: &BatchTranslationOutcome) {
    print!("{}", batch_text(outcome));
}

pub fn print_transcription_outcome(outcome: &TranscriptionOutcome) {
    println!("Output: {}", outcome.output_path.display());
}

pub fn print_whisper_outcome(outcome: &WhisperOutcome) {
    print!("{}", whisper_text(outcome));
}

fn render_translation_outcome(
    outcome: &TranslationOutcome,
    json: bool,
) -> io::Result<(String, Option<PathBuf>)> {
    if outcome.result.dry_run {
        return Ok((dry_run_text(&outcome.result, json), None));
    }

    let output_path = outcome
        .output_path
        .clone()
        .ok_or_else(|| io::Error::other("translation completed without an output path"))?;
    let output = if json {
        format!("{}\n", result_json(&outcome.result))
    } else {
        translation_text(&outcome.result, &output_path)
    };

    Ok((output, Some(output_path)))
}

fn translation_text(result: &PipelineResult, output_path: &Path) -> String {
    format!(
        "Output: {}\nUsage: {} in / {} out / {} total\nBatches: {} translated\n",
        output_path.display(),
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.total_tokens,
        result.batches_translated
    )
}

fn dry_run_text(result: &PipelineResult, json: bool) -> String {
    if json {
        return format!("{}\n", result_json(result));
    }

    let mut output = format!(
        "Dry run: no model calls were made.\nPlanned batches: {}\n",
        result.planned_batches.len()
    );
    for batch in &result.planned_batches {
        output.push_str(&format!(
            "  batch {}: {} line(s), {} -> {}\n",
            batch.index, batch.size, batch.first_id, batch.last_id
        ));
    }
    output
}

fn batch_text(outcome: &BatchTranslationOutcome) -> String {
    if outcome.processed == 0 && outcome.skipped.is_empty() {
        return "No subtitle files found.\n".to_owned();
    }

    let mut output = String::new();
    for path in &outcome.skipped {
        output.push_str(&format!(
            "Skipped existing output for: {}\n",
            path.display()
        ));
    }
    output.push_str(&format!(
        "Batch result: {} processed, {} skipped, 0 failed\n",
        outcome.processed,
        outcome.skipped.len()
    ));
    output
}

fn whisper_text(outcome: &WhisperOutcome) -> String {
    match outcome {
        WhisperOutcome::Status(status) => format!(
            "Whisper binary: {} ({})\nModel directory: {} ({})\n",
            status.binary_path.display(),
            exists_label(status.binary_exists),
            status.models_dir.display(),
            exists_label(status.models_dir_exists)
        ),
    }
}

fn exists_label(value: bool) -> &'static str {
    if value { "found" } else { "missing" }
}

pub fn result_json(result: &PipelineResult) -> String {
    let output_path = result
        .output_path
        .as_ref()
        .map(|path| quote_json(&path.to_string_lossy()))
        .unwrap_or_else(|| "null".to_owned());
    let glossary_path = result
        .glossary_path
        .as_ref()
        .map(|path| quote_json(&path.to_string_lossy()))
        .unwrap_or_else(|| "null".to_owned());
    let planned_batches = result
        .planned_batches
        .iter()
        .map(batch_json)
        .collect::<Vec<_>>()
        .join(",");

    format!(
        "{{\"output_path\":{output_path},\"batches_translated\":{},\"review_batches\":{},\"usage\":{{\"input_tokens\":{},\"output_tokens\":{},\"total_tokens\":{}}},\"dry_run\":{},\"planned_batches\":[{}],\"glossary_path\":{glossary_path}}}",
        result.batches_translated,
        result.review_batches,
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.total_tokens,
        result.dry_run,
        planned_batches
    )
}

fn batch_json(batch: &BatchPlanEntry) -> String {
    format!(
        "{{\"index\":{},\"size\":{},\"first_id\":{},\"last_id\":{}}}",
        batch.index,
        batch.size,
        quote_json(&batch.first_id),
        quote_json(&batch.last_id)
    )
}

fn quote_json(value: &str) -> String {
    let mut output = String::new();
    output.push('"');
    for ch in value.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            ch => output.push(ch),
        }
    }
    output.push('"');
    output
}

#[cfg(test)]
mod tests {
    use subbake_core::entities::Usage;

    use super::*;

    #[test]
    fn result_json_escapes_paths() {
        let result = PipelineResult {
            output_path: Some("quote\"path.txt".into()),
            batches_translated: 0,
            review_batches: 0,
            usage: Usage::default(),
            dry_run: true,
            planned_batches: Vec::new(),
            cache_hits: 0,
            resumed_translation_batches: 0,
            resumed_review_batches: 0,
            translation_memory_hits: 0,
            state_path: None,
            glossary_path: None,
            agent_repairs: Vec::new(),
        };

        assert!(result_json(&result).contains("quote\\\"path.txt"));
    }

    #[test]
    fn dry_run_text_lists_planned_batches() {
        let result = PipelineResult {
            output_path: None,
            batches_translated: 0,
            review_batches: 0,
            usage: Usage::default(),
            dry_run: true,
            planned_batches: vec![BatchPlanEntry {
                index: 1,
                size: 3,
                first_id: "1".to_owned(),
                last_id: "3".to_owned(),
            }],
            cache_hits: 0,
            resumed_translation_batches: 0,
            resumed_review_batches: 0,
            translation_memory_hits: 0,
            state_path: None,
            glossary_path: None,
            agent_repairs: Vec::new(),
        };

        let output = dry_run_text(&result, false);

        assert!(output.contains("Planned batches: 1"));
        assert!(output.contains("batch 1: 3 line(s), 1 -> 3"));
    }

    #[test]
    fn batch_text_reports_empty_directory() {
        let outcome = BatchTranslationOutcome {
            processed: 0,
            skipped: Vec::new(),
            outputs: Vec::new(),
        };

        assert_eq!(batch_text(&outcome), "No subtitle files found.\n");
    }

    #[test]
    fn whisper_text_reports_status_paths() {
        let output = whisper_text(&WhisperOutcome::Status(subbake_adapters::WhisperStatus {
            binary_path: "whisper-cli".into(),
            binary_exists: false,
            models_dir: "models".into(),
            models_dir_exists: true,
        }));

        assert!(output.contains("whisper-cli (missing)"));
        assert!(output.contains("models (found)"));
    }
}
