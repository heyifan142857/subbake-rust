use subbake_core::{AgentToolOutcome, ToolExecutionStatus};

pub(crate) fn render_tool_outcome(outcome: &AgentToolOutcome) -> String {
    match outcome {
        AgentToolOutcome::Translation(facts) => {
            let mode = if facts.bilingual {
                format!("bilingual ({})", facts.bilingual_order.as_str())
            } else {
                "translated".to_owned()
            };
            let mut lines = vec![format!(
                "Translation {}: {} file(s), {} subtitle entries, {} → {}, {}, {mode}, provider {}/{}.",
                status_label(facts.status),
                facts.processed_files,
                facts.subtitle_entries,
                facts.source_language,
                facts.target_language,
                facts.output_format,
                facts.provider,
                facts.model
            )];
            if facts.outputs.is_empty() {
                if facts.dry_run {
                    lines.push(
                        "Dry run: no output file was written and no undo event was recorded."
                            .to_owned(),
                    );
                }
            } else {
                lines.push(format!(
                    "Output: {}",
                    facts
                        .outputs
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if !facts.skipped.is_empty() {
                lines.push(format!(
                    "Skipped: {}",
                    facts
                        .skipped
                        .iter()
                        .map(|item| format!("{} ({})", item.path.display(), item.reason))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if facts.cache_hits > 0
                || facts.resumed_translation_batches > 0
                || facts.resumed_review_batches > 0
                || facts.translation_memory_hits > 0
            {
                lines.push(format!(
                    "Reuse: cache={}, resumed_translation={}, resumed_review={}, translation_memory={}.",
                    facts.cache_hits,
                    facts.resumed_translation_batches,
                    facts.resumed_review_batches,
                    facts.translation_memory_hits
                ));
            }
            if facts.fresh_runtime
                && let Some(runtime_dir) = &facts.runtime_dir
            {
                lines.push(format!(
                    "Fresh runtime: {} (Resume, request cache, translation memory, and accumulated glossary reuse disabled).",
                    runtime_dir.display()
                ));
            }
            lines.join("\n")
        }
        AgentToolOutcome::Transcription(facts) => format!(
            "Transcription written: {}, language {}, format {}, provider {}/{}, {} subtitle entries.\nOutput: {}",
            facts.input.display(),
            facts.language,
            facts.output_format,
            facts.provider,
            facts.model,
            facts.subtitle_entries,
            facts.output.display()
        ),
        AgentToolOutcome::SubtitleEdit(facts) => {
            let mut text = format!(
                "Subtitle edited: {}, target language {}, {} entries modified.",
                facts.target_path.display(),
                facts.target_language,
                facts.modified_entries
            );
            if !facts.edit_notes.trim().is_empty() {
                text.push_str(&format!("\n{}", facts.edit_notes));
            }
            text
        }
        AgentToolOutcome::Whisper(facts) => {
            let mut lines = vec![format!(
                "Whisper {} {}.",
                facts.action,
                status_label(facts.status)
            )];
            if let Some(path) = &facts.binary_path {
                lines.push(format!(
                    "Binary: {} ({})",
                    path.display(),
                    existence_label(facts.binary_exists)
                ));
            }
            if let Some(path) = &facts.models_dir {
                lines.push(format!(
                    "Models directory: {} ({})",
                    path.display(),
                    existence_label(facts.models_dir_exists)
                ));
            }
            if !facts.models.is_empty() {
                lines.push(format!(
                    "Models: {}",
                    facts
                        .models
                        .iter()
                        .map(|model| format!("{} ({})", model.name, model.path.display()))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            lines.join("\n")
        }
        AgentToolOutcome::File(facts) => {
            let paths = facts
                .paths
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            if facts.destination_paths.is_empty() {
                format!("File {}: {paths}", facts.action)
            } else {
                format!(
                    "File {}: {paths} → {}",
                    facts.action,
                    facts
                        .destination_paths
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        AgentToolOutcome::Profile(facts)
            if facts.action == "switch"
                && facts.status == ToolExecutionStatus::Completed
                && facts.provider.is_some()
                && facts.model.is_some() =>
        {
            format!(
                "{} ({}/{})",
                facts.message,
                facts.provider.as_deref().unwrap_or_default(),
                facts.model.as_deref().unwrap_or_default()
            )
        }
        AgentToolOutcome::Profile(facts) => facts.message.clone(),
        AgentToolOutcome::Observation(facts) => facts.content.clone(),
        AgentToolOutcome::Command(facts) => {
            let mut lines = vec![format!(
                "Command exited with code {} in {} ms (cwd {}).",
                facts.exit_code,
                facts.duration_ms,
                facts.cwd.display()
            )];
            if !facts.stdout.is_empty() {
                lines.push(format!("stdout:\n{}", facts.stdout));
            }
            if !facts.stderr.is_empty() {
                lines.push(format!("stderr:\n{}", facts.stderr));
            }
            if !facts.outputs.is_empty() {
                lines.push(format!(
                    "Outputs: {}",
                    facts
                        .outputs
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            lines.join("\n")
        }
    }
}

fn status_label(status: ToolExecutionStatus) -> &'static str {
    match status {
        ToolExecutionStatus::Written => "written",
        ToolExecutionStatus::DryRun => "dry run",
        ToolExecutionStatus::Skipped => "skipped",
        ToolExecutionStatus::Unchanged => "unchanged",
        ToolExecutionStatus::Observed => "observed",
        ToolExecutionStatus::Completed => "completed",
    }
}

fn existence_label(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "found",
        Some(false) => "missing",
        None => "not inspected",
    }
}
