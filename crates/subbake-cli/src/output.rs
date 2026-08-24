use std::io;
use std::path::{Path, PathBuf};

use subbake_adapters::{
    BatchTranslationOutcome, PipelineOutcome, ProviderCheckOutcome, RuntimeOutcome,
    SubtitleEditOutcome, TranscriptionOutcome, TranslationOutcome, WhisperOutcome,
};
use subbake_core::entities::PipelineResult;

pub fn print_translation_outcome(
    outcome: &TranslationOutcome,
    json: bool,
) -> io::Result<Option<PathBuf>> {
    let (output, output_path) = render_translation_outcome(outcome, json)?;
    print!("{output}");
    Ok(output_path)
}

pub fn print_batch_translation_outcome(outcome: &BatchTranslationOutcome, json: bool) {
    if json {
        print_json_value("batch_translation_result", batch_json(outcome));
    } else {
        print!("{}", batch_text(outcome));
    }
}

pub fn print_subtitle_edit_outcome(outcome: &SubtitleEditOutcome, json: bool) -> io::Result<()> {
    if json {
        let result = serde_json::to_value(outcome)
            .map_err(|error| io::Error::other(format!("serialize edit result: {error}")))?;
        print_json_value("subtitle_edit_result", result);
        return Ok(());
    }
    println!(
        "{}: {} proposed change(s) for {}",
        if outcome.dry_run { "Dry run" } else { "Edited" },
        outcome.modified_entries,
        outcome.target_path.display()
    );
    for change in &outcome.changes {
        println!("@@ {} @@", change.id);
        for line in change.before.lines() {
            println!("- {line}");
        }
        for line in change.after.lines() {
            println!("+ {line}");
        }
    }
    if !outcome.edit_notes.trim().is_empty() {
        println!("Notes: {}", outcome.edit_notes.trim());
    }
    Ok(())
}

pub fn print_pipeline_outcome(
    outcome: &PipelineOutcome,
    json: bool,
) -> io::Result<Option<PathBuf>> {
    match outcome {
        PipelineOutcome::Subtitle(outcome) => print_translation_outcome(outcome, json),
    }
}

pub fn print_transcription_outcome(outcome: &TranscriptionOutcome, json: bool) {
    if json {
        print_json_value(
            "transcription_result",
            serde_json::json!({
                "output_path": outcome.output_path,
                "language": outcome.language,
                "provider": outcome.provider,
                "model": outcome.model,
                "model_auto_selected": outcome.model_auto_selected,
                "output_format": outcome.output_format.extension(),
                "subtitle_entries": outcome.subtitle_entries,
                "quality": outcome.quality,
                "cleanup": {
                    "removed_empty_or_silence": outcome.cleanup.removed_empty_or_silence,
                    "removed_repeated": outcome.cleanup.removed_repeated,
                    "normalized_segments": outcome.cleanup.normalized_segments,
                    "speaker_labels_detected": outcome.cleanup.speaker_labels_detected,
                }
            }),
        );
        return;
    }
    if outcome.model_auto_selected {
        println!("Model: {} (automatically selected)", outcome.model);
    }
    println!("Output: {}", outcome.output_path.display());
    let removed = outcome.cleanup.removed_empty_or_silence + outcome.cleanup.removed_repeated;
    if removed > 0 {
        println!(
            "Cleanup: {removed} segment(s) removed ({} empty/silence, {} repeated)",
            outcome.cleanup.removed_empty_or_silence, outcome.cleanup.removed_repeated
        );
    }
    if outcome.cleanup.normalized_segments > 0 || outcome.cleanup.speaker_labels_detected > 0 {
        println!(
            "Post-processing: {} normalized segment(s), {} speaker label(s) detected",
            outcome.cleanup.normalized_segments, outcome.cleanup.speaker_labels_detected
        );
    }
    println!(
        "QA: {} error(s), {} warning(s)",
        outcome.quality.errors, outcome.quality.warnings
    );
}

pub fn print_provider_check_outcome(outcome: &ProviderCheckOutcome, json: bool) {
    if json {
        print_json_value(
            "provider_check_result",
            serde_json::json!({
                "provider": outcome.provider,
                "model": outcome.model,
                "message": outcome.message,
                "passed": true,
            }),
        );
        return;
    }
    println!("Provider check passed.");
    println!("{}", outcome.message);
}

pub fn print_runtime_outcome(outcome: &RuntimeOutcome, json: bool) {
    if json {
        print_json_value("runtime_result", runtime_json(outcome));
    } else {
        print!("{}", runtime_text(outcome));
    }
}

pub fn print_whisper_outcome(outcome: &WhisperOutcome, json: bool) {
    if json {
        print_json_value("whisper_result", whisper_json(outcome));
    } else {
        print!("{}", whisper_text(outcome));
    }
}

pub fn print_json_value(kind: &str, result: serde_json::Value) {
    println!(
        "{}",
        serde_json::to_string_pretty(&json_envelope(kind, result))
            .unwrap_or_else(|_| "{}".to_owned())
    );
}

pub fn json_envelope(kind: &str, result: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "version": 1,
        "kind": kind,
        "build": crate::version::build_identity(),
        "result": result,
    })
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
        format!("{}\n", translation_outcome_json(outcome))
    } else {
        let mut rendered = translation_text(&outcome.result, &output_path);
        if let Some(quality) = &outcome.quality {
            rendered.push_str(&format!(
                "QA: {} error(s), {} warning(s)\n",
                quality.errors, quality.warnings
            ));
        }
        if let Some(ocr) = &outcome.source_ocr {
            rendered.push_str(&format!(
                "Bitmap OCR ({}): {} cue(s), {} low-confidence cue(s)\n",
                ocr.codec, ocr.cues, ocr.low_confidence_cues
            ));
        }
        rendered
    };

    Ok((output, Some(output_path)))
}

fn translation_text(result: &PipelineResult, output_path: &Path) -> String {
    let mut output = format!(
        "Output: {}\nMode: {}\nUsage: {} in / {} out / {} total ({} billable in, {} cached)\nBatches: {} translated\n",
        output_path.display(),
        result.mode.as_str(),
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.total_tokens,
        result.usage.billable_input_tokens(),
        result.usage.cached_input_tokens,
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
    if result.deduplicated_segments > 0 {
        reuse.push(format!(
            "{} duplicate segment(s) coalesced",
            result.deduplicated_segments
        ));
    }
    if result.reviewer_fallback {
        output.push_str("Review: reviewer backend unavailable; translator was used as fallback\n");
    }
    if result.terminology.candidates > 0
        || result.terminology.entries_added > 0
        || result.terminology.degraded
        || result.terminology.usage != subbake_core::Usage::default()
    {
        output.push_str(&format!(
            "Terminology: {} candidate(s), {} added, {} conflict(s) omitted{}\n",
            result.terminology.candidates,
            result.terminology.entries_added,
            result.terminology.conflicts_omitted,
            if result.terminology.degraded {
                ", preflight degraded"
            } else {
                ""
            }
        ));
        if let Some(reason) = &result.terminology.degraded_reason {
            output.push_str(&format!("Terminology preflight fallback: {reason}\n"));
        }
    }
    if result.review.batches > 0 {
        let rate = if result.review.reviewed_lines == 0 {
            0.0
        } else {
            result.review.changed_lines as f64 * 100.0 / result.review.reviewed_lines as f64
        };
        output.push_str(&format!(
            "Review: {} candidate line(s), {} changed ({rate:.2}%), {} in / {} out tokens, {} ms\n",
            result.review.candidate_lines,
            result.review.changed_lines,
            result.review.usage.input_tokens,
            result.review.usage.output_tokens,
            result.review.duration_ms,
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
        "Dry run: no model calls were made.\nPlanned batches: {}\nEstimated translation requests: {}\n",
        result.planned_batches.len(),
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
    if outcome.processed == 0 && outcome.skipped.is_empty() && outcome.failures.is_empty() {
        return "No subtitle files found.\n".to_owned();
    }

    let mut output = String::new();
    for path in &outcome.skipped {
        output.push_str(&format!(
            "Skipped existing output for: {}\n",
            path.display()
        ));
    }
    for failure in &outcome.failures {
        output.push_str(&format!(
            "Failed: {}: {}\n",
            failure.input.display(),
            failure.error
        ));
    }
    output.push_str(&format!(
        "Batch result: {} processed, {} skipped, {} failed\nManifest: {}\n",
        outcome.processed,
        outcome.skipped.len(),
        outcome.failures.len(),
        outcome.manifest_path.display()
    ));
    output
}

fn batch_json(outcome: &BatchTranslationOutcome) -> serde_json::Value {
    serde_json::json!({
        "processed": outcome.processed,
        "inputs": outcome.inputs,
        "skipped": outcome.skipped,
        "outputs": outcome.outputs,
        "failures": outcome.failures.iter().map(|failure| serde_json::json!({
            "input": failure.input,
            "error": failure.error,
        })).collect::<Vec<_>>(),
        "manifest_path": outcome.manifest_path,
        "subtitle_entries": outcome.subtitle_entries,
        "dry_run": outcome.dry_run,
        "cache_hits": outcome.cache_hits,
        "resumed_translation_batches": outcome.resumed_translation_batches,
        "resumed_review_batches": outcome.resumed_review_batches,
        "translation_memory_hits": outcome.translation_memory_hits,
        "runtime_dir": outcome.runtime_dir,
    })
}

fn whisper_text(outcome: &WhisperOutcome) -> String {
    match outcome {
        WhisperOutcome::Status(status) => format!(
            "Whisper binary: {} ({})\nModel directory: {} ({})\nDefault VAD model: {} ({})\n{}{}",
            status.binary_path.display(),
            exists_label(status.binary_exists),
            status.models_dir.display(),
            exists_label(status.models_dir_exists),
            status.default_vad_model_path.display(),
            exists_label(status.default_vad_model_exists),
            status
                .version
                .as_ref()
                .map(|version| format!("Version: {version}\n"))
                .unwrap_or_default(),
            status
                .capability_error
                .as_ref()
                .map(|error| format!("Compatibility: {error}\n"))
                .unwrap_or_default()
        ),
        WhisperOutcome::ModelList(list) | WhisperOutcome::VadModelList(list) => {
            let kind = if matches!(outcome, WhisperOutcome::VadModelList(_)) {
                "VAD models"
            } else {
                "models"
            };
            let mut output = format!(
                "Model directory: {} ({})\nInstalled {kind}: {}\n",
                list.models_dir.display(),
                exists_label(list.models_dir_exists),
                list.models.len()
            );
            for model in &list.models {
                output.push_str(&format!("  {}: {}\n", model.name, model.path.display()));
            }
            output.push_str(&format!(
                "Available {kind}: {}\n",
                list.available_models.len()
            ));
            for model in &list.available_models {
                let installed = list.models.iter().any(|item| item.name == *model);
                output.push_str(&format!(
                    "  {model}{}\n",
                    if installed { " (installed)" } else { "" }
                ));
            }
            if let Some(warning) = &list.refresh_warning {
                output.push_str(&format!("Warning: {warning}\n"));
            }
            output
        }
        WhisperOutcome::VersionList(list) => {
            let mut output = format!(
                "Pinned install version: {}\nAvailable versions: {}\n",
                list.pinned_version,
                list.versions.len()
            );
            for version in &list.versions {
                output.push_str(&format!(
                    "  {}{}{}\n",
                    version.tag,
                    if version.prerelease {
                        " (prerelease)"
                    } else {
                        ""
                    },
                    if version.installable {
                        " (installable)"
                    } else {
                        ""
                    }
                ));
            }
            if let Some(warning) = &list.refresh_warning {
                output.push_str(&format!("Warning: {warning}\n"));
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

fn runtime_json(outcome: &RuntimeOutcome) -> serde_json::Value {
    match outcome {
        RuntimeOutcome::Inspection(inspection) => {
            let paths = &inspection.paths;
            serde_json::json!({
                "action": "inspect",
                "paths": {
                    "root_dir": paths.root_dir,
                    "run_dir": paths.run_dir,
                    "cache_dir": paths.cache_dir,
                    "state_path": paths.state_path,
                    "glossary_path": paths.glossary_path,
                    "translation_memory_path": paths.translation_memory_path,
                    "review_report_path": paths.review_report_path,
                }
            })
        }
        RuntimeOutcome::Clean(clean) => serde_json::json!({
            "action": "clean",
            "root_dir": clean.root_dir,
            "removed": clean.removed,
        }),
    }
}

fn whisper_json(outcome: &WhisperOutcome) -> serde_json::Value {
    match outcome {
        WhisperOutcome::Status(status) => serde_json::json!({
            "action": "status",
            "binary_path": status.binary_path,
            "binary_exists": status.binary_exists,
            "models_dir": status.models_dir,
            "models_dir_exists": status.models_dir_exists,
            "version": status.version,
            "capability_error": status.capability_error,
            "default_vad_model_path": status.default_vad_model_path,
            "default_vad_model_exists": status.default_vad_model_exists,
        }),
        WhisperOutcome::VersionList(list) => serde_json::json!({
            "action": "list_versions",
            "pinned_version": list.pinned_version,
            "versions": list.versions.iter().map(|version| serde_json::json!({
                "tag": version.tag,
                "prerelease": version.prerelease,
                "published_at": version.published_at,
                "installable": version.installable,
            })).collect::<Vec<_>>(),
            "refresh_warning": list.refresh_warning,
        }),
        WhisperOutcome::ModelList(list) | WhisperOutcome::VadModelList(list) => serde_json::json!({
            "action": if matches!(outcome, WhisperOutcome::VadModelList(_)) {
                "list_vad_models"
            } else {
                "list_models"
            },
            "models_dir": list.models_dir,
            "models_dir_exists": list.models_dir_exists,
            "models": list.models.iter().map(|model| serde_json::json!({
                "name": model.name,
                "path": model.path,
            })).collect::<Vec<_>>(),
            "available_models": list.available_models,
            "refresh_warning": list.refresh_warning,
        }),
    }
}

fn exists_label(value: bool) -> &'static str {
    if value { "found" } else { "missing" }
}

pub fn result_json(result: &PipelineResult) -> String {
    serde_json::to_string(&json_envelope(
        "translation_result",
        translation_result_value(result, None),
    ))
    .unwrap_or_else(|_| "{}".to_owned())
}

fn translation_outcome_json(outcome: &TranslationOutcome) -> String {
    let mut result = translation_result_value(&outcome.result, outcome.quality.as_ref());
    if let Some(ocr) = &outcome.source_ocr
        && let Some(result) = result.as_object_mut()
    {
        result.insert(
            "source_ocr".to_owned(),
            serde_json::json!({
                "codec": ocr.codec,
                "cues": ocr.cues,
                "low_confidence_cues": ocr.low_confidence_cues,
            }),
        );
    }
    serde_json::to_string(&json_envelope("translation_result", result))
        .unwrap_or_else(|_| "{}".to_owned())
}

fn translation_result_value(
    result: &PipelineResult,
    quality: Option<&subbake_core::QualityReport>,
) -> serde_json::Value {
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

    let result = serde_json::json!({
        "output_path": output_path,
        "batches_translated": result.batches_translated,
        "review_batches": result.review_batches,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
            "cached_input_tokens": result.usage.cached_input_tokens,
            "billable_input_tokens": result.usage.billable_input_tokens(),
            "requests": result.usage.requests,
            "retries": result.usage.retries,
        },
        "mode": result.mode.as_str(),
        "deduplicated_segments": result.deduplicated_segments,
        "reviewer_fallback": result.reviewer_fallback,
        "dry_run": result.dry_run,
        "planned_batches": planned_batches,
        "cache_hits": result.cache_hits,
        "resumed_translation_batches": result.resumed_translation_batches,
        "resumed_review_batches": result.resumed_review_batches,
        "translation_memory_hits": result.translation_memory_hits,
        "terminology": result.terminology,
        "review": result.review,
        "state_path": state_path,
        "glossary_path": glossary_path,
        "agent_repairs": result.agent_repairs,
        "quality": quality,
    });
    result
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
            mode: subbake_core::TranslationMode::Turbo,
            deduplicated_segments: 0,
            reviewer_fallback: false,
            dry_run: true,
            planned_batches: Vec::new(),
            cache_hits: 0,
            resumed_translation_batches: 0,
            resumed_review_batches: 0,
            translation_memory_hits: 0,
            state_path: None,
            glossary_path: None,
            agent_repairs: Vec::new(),
            terminology: Default::default(),
            review: Default::default(),
        };

        let encoded = result_json(&result);
        assert!(encoded.contains("quote\\\"path.txt"));
        let value: serde_json::Value = serde_json::from_str(&encoded).expect("result JSON");
        assert_eq!(value["version"], 1);
        assert_eq!(value["kind"], "translation_result");
        assert_eq!(value["result"]["output_path"], "quote\"path.txt");
        assert!(
            value["build"]
                .as_str()
                .is_some_and(|build| !build.is_empty())
        );
    }

    #[test]
    fn dry_run_text_lists_planned_batches() {
        let result = PipelineResult {
            output_path: None,
            batches_translated: 0,
            review_batches: 0,
            usage: Usage::default(),
            mode: subbake_core::TranslationMode::Turbo,
            deduplicated_segments: 0,
            reviewer_fallback: false,
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
            terminology: Default::default(),
            review: Default::default(),
        };

        let output = dry_run_text(&result, false);

        assert!(output.contains("Planned batches: 1"));
        assert!(output.contains("batch 1: 3 line(s), 1 -> 3"));
    }

    #[test]
    fn batch_text_reports_empty_directory() {
        let outcome = BatchTranslationOutcome {
            processed: 0,
            inputs: Vec::new(),
            skipped: Vec::new(),
            outputs: Vec::new(),
            failures: Vec::new(),
            manifest_path: ".subbake/batch/test.json".into(),
            subtitle_entries: 0,
            dry_run: false,
            cache_hits: 0,
            resumed_translation_batches: 0,
            resumed_review_batches: 0,
            translation_memory_hits: 0,
            runtime_dir: None,
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
            version: None,
            capability_error: None,
            default_vad_model_path: "models/ggml-silero-v6.2.0.bin".into(),
            default_vad_model_exists: false,
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
                    name: "base".to_owned(),
                    path: "models/ggml-base.bin".into(),
                }],
                available_models: vec!["base".to_owned(), "small".to_owned()],
                refresh_warning: None,
            },
        ));

        assert!(output.contains("Installed models: 1"));
        assert!(output.contains("Available models: 2"));
        assert!(output.contains("base"));
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
