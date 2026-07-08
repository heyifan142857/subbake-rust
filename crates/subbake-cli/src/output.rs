use std::io;
use std::path::{Path, PathBuf};

use subbake_adapters::{
    BatchTranslationOutcome, PipelineOutcome, ProviderCheckOutcome, RuntimeOutcome,
    TranscriptionOutcome, TranslationOutcome, WhisperOutcome,
};
use subbake_agent::AgentOutcome;
use subbake_core::entities::PipelineResult;

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

pub fn print_agent_outcome(outcome: &AgentOutcome) {
    println!("{}", outcome.message);
}

pub fn print_pipeline_outcome(
    outcome: &PipelineOutcome,
    json: bool,
) -> io::Result<Option<PathBuf>> {
    match outcome {
        PipelineOutcome::Subtitle(outcome) => print_translation_outcome(outcome, json),
    }
}

pub fn print_transcription_outcome(outcome: &TranscriptionOutcome) {
    println!("Output: {}", outcome.output_path.display());
}

pub fn print_provider_check_outcome(outcome: &ProviderCheckOutcome) {
    println!("Provider check passed.");
    println!("{}", outcome.message);
}

pub fn print_runtime_outcome(outcome: &RuntimeOutcome) {
    print!("{}", runtime_text(outcome));
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
    let mut output = format!(
        "Output: {}\nUsage: {} in / {} out / {} total\nBatches: {} translated\n",
        output_path.display(),
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.total_tokens,
        result.batches_translated
    );
    let mut reuse = Vec::new();
    if result.resumed_translation_batches > 0 {
        reuse.push(format!(
            "{} translated batch(es) resumed",
            result.resumed_translation_batches
        ));
    }
    if result.resumed_review_batches > 0 {
        reuse.push(format!(
            "{} review batch(es) resumed",
            result.resumed_review_batches
        ));
    }
    if result.cache_hits > 0 {
        reuse.push(format!("{} cached request(s)", result.cache_hits));
    }
    if result.translation_memory_hits > 0 {
        reuse.push(format!(
            "{} translation-memory hit(s)",
            result.translation_memory_hits
        ));
    }
    if !reuse.is_empty() {
        output.push_str(&format!("Reused: {}\n", reuse.join(", ")));
    }
    output
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
        WhisperOutcome::ModelList(list) => {
            let mut output = format!(
                "Model directory: {} ({})\nModels: {}\n",
                list.models_dir.display(),
                exists_label(list.models_dir_exists),
                list.models.len()
            );
            for model in &list.models {
                output.push_str(&format!("  {}: {}\n", model.name, model.path.display()));
            }
            output
        }
    }
}

fn runtime_text(outcome: &RuntimeOutcome) -> String {
    match outcome {
        RuntimeOutcome::Inspection(inspection) => {
            let paths = &inspection.paths;
            format!(
                "runtime: {}\nrun: {}\ncache: {}\nstate: {}\nglossary: {}\n",
                paths.root_dir.display(),
                paths.run_dir.display(),
                paths.cache_dir.display(),
                paths.state_path.display(),
                paths.glossary_path.display()
            )
        }
        RuntimeOutcome::Clean(clean) if clean.removed => {
            format!("Removed: {}\n", clean.root_dir.display())
        }
        RuntimeOutcome::Clean(clean) => {
            format!("Nothing removed: {}\n", clean.root_dir.display())
        }
    }
}

fn exists_label(value: bool) -> &'static str {
    if value { "found" } else { "missing" }
}

pub fn result_json(result: &PipelineResult) -> String {
    let output_path = result
        .output_path
        .as_ref()
        .map(|path| path.to_string_lossy().to_string());
    let glossary_path = result
        .glossary_path
        .as_ref()
        .map(|path| path.to_string_lossy().to_string());
    let state_path = result
        .state_path
        .as_ref()
        .map(|path| path.to_string_lossy().to_string());
    let planned_batches: Vec<serde_json::Value> = result
        .planned_batches
        .iter()
        .map(|batch| {
            serde_json::json!({
                "index": batch.index,
                "size": batch.size,
                "first_id": batch.first_id,
                "last_id": batch.last_id,
            })
        })
        .collect();

    serde_json::to_string(&serde_json::json!({
        "output_path": output_path,
        "batches_translated": result.batches_translated,
        "review_batches": result.review_batches,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
        },
        "dry_run": result.dry_run,
        "planned_batches": planned_batches,
        "cache_hits": result.cache_hits,
        "resumed_translation_batches": result.resumed_translation_batches,
        "resumed_review_batches": result.resumed_review_batches,
        "translation_memory_hits": result.translation_memory_hits,
        "state_path": state_path,
        "glossary_path": glossary_path,
    }))
    .unwrap_or_else(|_| "{}".to_owned())
}

#[cfg(test)]
mod tests {
    use subbake_core::entities::{BatchPlanEntry, Usage};

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

    #[test]
    fn whisper_text_lists_models() {
        let output = whisper_text(&WhisperOutcome::ModelList(
            subbake_adapters::WhisperModelList {
                models_dir: "models".into(),
                models_dir_exists: true,
                models: vec![subbake_adapters::WhisperModel {
                    name: "ggml-base".to_owned(),
                    path: "models/ggml-base.bin".into(),
                }],
            },
        ));

        assert!(output.contains("Models: 1"));
        assert!(output.contains("ggml-base"));
    }

    #[test]
    fn runtime_text_reports_clean_result() {
        let output = runtime_text(&RuntimeOutcome::Clean(
            subbake_adapters::RuntimeCleanOutcome {
                root_dir: ".subbake".into(),
                removed: false,
            },
        ));

        assert_eq!(output, "Nothing removed: .subbake\n");
    }
}
