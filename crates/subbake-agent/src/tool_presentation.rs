use std::time::Duration;

use serde_json::Value as JsonValue;
use subbake_core::AgentToolOutcome;

use crate::tools::{ToolExecutor, find_tool_handler};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolActivityText {
    pub(crate) headline: String,
    pub(crate) detail: Option<String>,
}

pub(crate) fn running_activity(name: &str, arguments: &JsonValue) -> ToolActivityText {
    activity_text(name, arguments, ActivityPhase::Running, None, None)
}

pub(crate) fn completed_activity(
    name: &str,
    arguments: &JsonValue,
    outcome: &AgentToolOutcome,
    elapsed: Duration,
) -> ToolActivityText {
    activity_text(
        name,
        arguments,
        ActivityPhase::Completed,
        Some(outcome),
        Some(elapsed),
    )
}

pub(crate) fn failed_activity(name: &str, arguments: &JsonValue, error: &str) -> ToolActivityText {
    let mut text = activity_text(name, arguments, ActivityPhase::Failed, None, None);
    text.detail = Some(one_line(error, 160));
    text
}

#[derive(Debug, Clone, Copy)]
enum ActivityPhase {
    Running,
    Completed,
    Failed,
}

fn activity_text(
    name: &str,
    arguments: &JsonValue,
    phase: ActivityPhase,
    outcome: Option<&AgentToolOutcome>,
    elapsed: Option<Duration>,
) -> ToolActivityText {
    let executor = find_tool_handler(name).map(|handler| handler.executor());
    let action = action_label(executor, arguments, phase);
    let target = target_label(executor, name, arguments);
    let headline = target.map_or_else(
        || action.to_owned(),
        |target| {
            if matches!(phase, ActivityPhase::Failed) {
                format!("{action}: {target}")
            } else {
                format!("{action} {target}")
            }
        },
    );
    let detail = outcome
        .map(|outcome| outcome_detail(outcome, elapsed.unwrap_or_default()))
        .or_else(|| argument_detail(executor, arguments));
    ToolActivityText { headline, detail }
}

fn action_label(
    executor: Option<ToolExecutor>,
    arguments: &JsonValue,
    phase: ActivityPhase,
) -> &'static str {
    if matches!(phase, ActivityPhase::Failed) {
        return failure_label(executor);
    }
    let completed = matches!(phase, ActivityPhase::Completed);
    match executor {
        Some(ToolExecutor::TranslateFile | ToolExecutor::TranslateSeries) => {
            if completed {
                "Translated"
            } else {
                "Translating"
            }
        }
        Some(ToolExecutor::EditSubtitle) => {
            if completed {
                "Edited"
            } else {
                "Editing"
            }
        }
        Some(ToolExecutor::TranscribeAudio) => {
            if completed {
                "Transcribed"
            } else {
                "Transcribing"
            }
        }
        Some(ToolExecutor::ManageWhisper) => whisper_action(arguments, completed),
        Some(ToolExecutor::InspectMedia) => {
            if completed {
                "Inspected media"
            } else {
                "Inspecting media"
            }
        }
        Some(ToolExecutor::DiagnosePath | ToolExecutor::DiagnoseText) => {
            if completed {
                "Diagnosed"
            } else {
                "Diagnosing"
            }
        }
        Some(
            ToolExecutor::ListFiles
            | ToolExecutor::RecentTranslations
            | ToolExecutor::CandidateSubtitles
            | ToolExecutor::ListProfiles,
        ) => {
            if completed {
                "Listed"
            } else {
                "Listing"
            }
        }
        Some(ToolExecutor::SearchFiles) => {
            if completed {
                "Searched"
            } else {
                "Searching"
            }
        }
        Some(ToolExecutor::ReadFilePreview | ToolExecutor::ReadFile) => {
            if completed {
                "Read"
            } else {
                "Reading"
            }
        }
        Some(ToolExecutor::ApplyPatch) => {
            if completed {
                "Updated"
            } else {
                "Updating"
            }
        }
        Some(ToolExecutor::RenamePath) => {
            if completed {
                "Renamed"
            } else {
                "Renaming"
            }
        }
        Some(ToolExecutor::DeleteFile | ToolExecutor::DeleteExternalPath) => {
            if completed {
                "Deleted"
            } else {
                "Deleting"
            }
        }
        Some(ToolExecutor::SwitchProfile) => {
            if completed {
                "Switched profile"
            } else {
                "Switching profile"
            }
        }
        Some(ToolExecutor::RunCommand) => {
            if completed {
                "Ran"
            } else {
                "Running"
            }
        }
        None => {
            if completed {
                "Called tool"
            } else {
                "Calling tool"
            }
        }
    }
}

fn failure_label(executor: Option<ToolExecutor>) -> &'static str {
    match executor {
        Some(ToolExecutor::TranslateFile | ToolExecutor::TranslateSeries) => "Translation failed",
        Some(ToolExecutor::EditSubtitle) => "Edit failed",
        Some(ToolExecutor::TranscribeAudio) => "Transcription failed",
        Some(ToolExecutor::ManageWhisper) => "Whisper operation failed",
        Some(ToolExecutor::InspectMedia) => "Media inspection failed",
        Some(ToolExecutor::DiagnosePath | ToolExecutor::DiagnoseText) => "Diagnosis failed",
        Some(
            ToolExecutor::ListFiles
            | ToolExecutor::RecentTranslations
            | ToolExecutor::CandidateSubtitles
            | ToolExecutor::ListProfiles,
        ) => "Listing failed",
        Some(ToolExecutor::SearchFiles) => "Search failed",
        Some(ToolExecutor::ReadFilePreview | ToolExecutor::ReadFile) => "Read failed",
        Some(
            ToolExecutor::ApplyPatch
            | ToolExecutor::RenamePath
            | ToolExecutor::DeleteFile
            | ToolExecutor::DeleteExternalPath,
        ) => "File operation failed",
        Some(ToolExecutor::SwitchProfile) => "Profile switch failed",
        Some(ToolExecutor::RunCommand) => "Command failed",
        None => "Tool failed",
    }
}

fn whisper_action(arguments: &JsonValue, completed: bool) -> &'static str {
    match string_argument(arguments, "action").unwrap_or("status") {
        "install" => {
            if completed {
                "Installed Whisper"
            } else {
                "Installing Whisper"
            }
        }
        "update" => {
            if completed {
                "Updated Whisper"
            } else {
                "Updating Whisper"
            }
        }
        "uninstall" => {
            if completed {
                "Removed Whisper"
            } else {
                "Removing Whisper"
            }
        }
        "download" | "download-vad" | "download_vad_model" => {
            if completed {
                "Downloaded model"
            } else {
                "Downloading model"
            }
        }
        "list-models" | "models" => {
            if completed {
                "Listed Whisper models"
            } else {
                "Listing Whisper models"
            }
        }
        "list-vad-models" | "vad-models" => {
            if completed {
                "Listed Whisper VAD models"
            } else {
                "Listing Whisper VAD models"
            }
        }
        "list-versions" | "versions" => {
            if completed {
                "Listed Whisper versions"
            } else {
                "Listing Whisper versions"
            }
        }
        _ => {
            if completed {
                "Checked Whisper"
            } else {
                "Checking Whisper"
            }
        }
    }
}

fn target_label(
    executor: Option<ToolExecutor>,
    name: &str,
    arguments: &JsonValue,
) -> Option<String> {
    match executor {
        Some(ToolExecutor::RunCommand)
            if matches!(
                crate::command_policy::classify(arguments),
                crate::command_policy::CommandApproval::Deny(_)
            ) =>
        {
            Some("[blocked command]".to_owned())
        }
        Some(ToolExecutor::RunCommand) => {
            string_argument(arguments, "command").map(|command| one_line(command, 100))
        }
        Some(ToolExecutor::RenamePath) => {
            let from = string_argument(arguments, "from")?;
            let to = string_argument(arguments, "to")?;
            Some(format!("{} → {}", one_line(from, 60), one_line(to, 60)))
        }
        Some(ToolExecutor::SwitchProfile) => string_argument(arguments, "name").map(str::to_owned),
        Some(ToolExecutor::ManageWhisper)
            if matches!(string_argument(arguments, "action"), Some("download")) =>
        {
            string_argument(arguments, "model").map(str::to_owned)
        }
        Some(ToolExecutor::ApplyPatch) => string_argument(arguments, "patch").map(patch_target),
        Some(ToolExecutor::DiagnoseText) => Some("provided text".to_owned()),
        Some(ToolExecutor::RecentTranslations | ToolExecutor::ListProfiles) => None,
        Some(ToolExecutor::SearchFiles) => {
            let pattern = string_argument(arguments, "pattern").unwrap_or("*");
            let path = string_argument(arguments, "path").unwrap_or(".");
            Some(format!("{pattern} in {path}"))
        }
        Some(ToolExecutor::CandidateSubtitles) => string_argument(arguments, "path")
            .or(Some("."))
            .map(str::to_owned),
        Some(_) => string_argument(arguments, "path").map(str::to_owned),
        None => Some(name.to_owned()),
    }
}

fn argument_detail(executor: Option<ToolExecutor>, arguments: &JsonValue) -> Option<String> {
    let mut parts = Vec::new();
    match executor {
        Some(ToolExecutor::TranslateFile | ToolExecutor::TranslateSeries) => {
            if let Some(language) = string_argument(arguments, "target_language") {
                parts.push(language.to_owned());
            }
            if arguments.get("bilingual").and_then(JsonValue::as_bool) == Some(true) {
                parts.push("bilingual".to_owned());
            }
            if arguments.get("recursive").and_then(JsonValue::as_bool) == Some(true) {
                parts.push("recursive".to_owned());
            }
            if let Some(output) = string_argument(arguments, "output_path")
                .or_else(|| string_argument(arguments, "output_dir"))
            {
                parts.push(format!("output {output}"));
            }
        }
        Some(ToolExecutor::TranscribeAudio) => {
            if let Some(language) = string_argument(arguments, "language") {
                parts.push(language.to_owned());
            }
            if let Some(model) = string_argument(arguments, "model") {
                parts.push(model.to_owned());
            }
        }
        Some(ToolExecutor::RunCommand) => {
            parts.push(format!(
                "cwd {}",
                string_argument(arguments, "cwd").unwrap_or(".")
            ));
            if arguments.get("network").and_then(JsonValue::as_bool) == Some(true) {
                parts.push("network".to_owned());
            }
        }
        Some(ToolExecutor::ManageWhisper) => {
            if let Some(variant) = string_argument(arguments, "variant") {
                parts.push(variant.to_owned());
            }
        }
        Some(ToolExecutor::ApplyPatch) => {
            if let Some(patch) = string_argument(arguments, "patch") {
                let (_, additions, deletions) = patch_stats(patch);
                parts.push(format!("+{additions} −{deletions}"));
            }
        }
        Some(ToolExecutor::DeleteExternalPath) => {
            parts.push("permanent · unavailable to /undo".to_owned());
            if arguments.get("recursive").and_then(JsonValue::as_bool) == Some(true) {
                parts.push("recursive".to_owned());
            }
        }
        _ => {}
    }
    (!parts.is_empty()).then(|| parts.join(" · "))
}

fn outcome_detail(outcome: &AgentToolOutcome, elapsed: Duration) -> String {
    let elapsed = format_elapsed(elapsed);
    match outcome {
        AgentToolOutcome::Translation(facts) => {
            let output = path_summary(&facts.outputs, "outputs");
            let mut parts = vec![format!("{} cues", facts.subtitle_entries)];
            if !facts.target_language.is_empty() {
                parts.push(facts.target_language.clone());
            }
            parts.push(elapsed);
            output.map_or_else(
                || parts.join(" · "),
                |output| format!("→ {output} · {}", parts.join(" · ")),
            )
        }
        AgentToolOutcome::Transcription(facts) => format!(
            "→ {} · {} cues · {} · {elapsed}",
            facts.output.display(),
            facts.subtitle_entries,
            facts.language
        ),
        AgentToolOutcome::SubtitleEdit(facts) => {
            if facts.partial_preview {
                format!(
                    "sampled {}/{} · {} modified · {} · {elapsed}",
                    facts.processed_entries,
                    facts.total_entries,
                    facts.modified_entries,
                    facts.target_language
                )
            } else {
                format!(
                    "{} entries modified · {} · {elapsed}",
                    facts.modified_entries, facts.target_language
                )
            }
        }
        AgentToolOutcome::Whisper(facts) => {
            let count = facts
                .models
                .len()
                .max(facts.available_models.len())
                .max(facts.available_versions.len());
            if count > 0 {
                format!("{count} items · {elapsed}")
            } else {
                elapsed
            }
        }
        AgentToolOutcome::File(facts) => {
            let paths = if facts.destination_paths.is_empty() {
                &facts.paths
            } else {
                &facts.destination_paths
            };
            path_summary(paths, "files")
                .map_or(elapsed.clone(), |paths| format!("→ {paths} · {elapsed}"))
        }
        AgentToolOutcome::Profile(facts) => {
            let route = match (&facts.provider, &facts.model) {
                (Some(provider), Some(model)) => Some(format!("{provider}/{model}")),
                _ => None,
            };
            route.map_or(elapsed.clone(), |route| format!("{route} · {elapsed}"))
        }
        AgentToolOutcome::Observation(facts) => {
            let count = facts
                .content
                .lines()
                .filter(|line| !line.trim().is_empty())
                .count();
            if count == 0 {
                elapsed
            } else {
                format!("{count} lines · {elapsed}")
            }
        }
        AgentToolOutcome::Command(facts) => {
            let duration = format_elapsed(Duration::from_millis(facts.duration_ms));
            if facts.outputs.is_empty() {
                format!("exit {} · {duration}", facts.exit_code)
            } else {
                format!(
                    "exit {} · {} outputs · {duration}",
                    facts.exit_code,
                    facts.outputs.len()
                )
            }
        }
    }
}

fn path_summary(paths: &[std::path::PathBuf], plural: &str) -> Option<String> {
    match paths {
        [] => None,
        [path] => Some(path.display().to_string()),
        paths => Some(format!("{} {plural}", paths.len())),
    }
}

fn patch_target(patch: &str) -> String {
    let (files, _, _) = patch_stats(patch);
    match files {
        0 => "files".to_owned(),
        1 => "1 file".to_owned(),
        count => format!("{count} files"),
    }
}

fn patch_stats(patch: &str) -> (usize, usize, usize) {
    let files = patch
        .lines()
        .filter(|line| {
            line.starts_with("*** Add File:")
                || line.starts_with("*** Update File:")
                || line.starts_with("*** Delete File:")
        })
        .count();
    let additions = patch
        .lines()
        .filter(|line| line.starts_with('+') && !line.starts_with("+++"))
        .count();
    let deletions = patch
        .lines()
        .filter(|line| line.starts_with('-') && !line.starts_with("---"))
        .count();
    (files, additions, deletions)
}

fn string_argument<'a>(arguments: &'a JsonValue, name: &str) -> Option<&'a str> {
    arguments.get(name).and_then(JsonValue::as_str)
}

fn one_line(value: &str, max_chars: usize) -> String {
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= max_chars {
        normalized
    } else {
        normalized
            .chars()
            .take(max_chars.saturating_sub(1))
            .chain(['…'])
            .collect()
    }
}

fn format_elapsed(elapsed: Duration) -> String {
    if elapsed.as_secs() >= 60 {
        format!("{}m {:02}s", elapsed.as_secs() / 60, elapsed.as_secs() % 60)
    } else if elapsed.as_secs() >= 10 {
        format!("{}s", elapsed.as_secs())
    } else {
        format!("{:.1}s", elapsed.as_secs_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::{completed_activity, failed_activity, patch_stats, running_activity};
    use std::time::Duration;
    use subbake_core::{AgentToolOutcome, ObservationToolOutcome, ToolExecutionStatus};

    #[test]
    fn translation_arguments_are_presented_without_json() {
        let activity = running_activity(
            "translate_file",
            &serde_json::json!({"path":"episode.srt","target_language":"Chinese","bilingual":true,"overwrite":false}),
        );
        assert_eq!(activity.headline, "Translating episode.srt");
        assert_eq!(activity.detail.as_deref(), Some("Chinese · bilingual"));
        assert!(!activity.headline.contains('{'));
    }

    #[test]
    fn observations_are_summarized_instead_of_echoed() {
        let outcome = AgentToolOutcome::Observation(ObservationToolOutcome {
            status: ToolExecutionStatus::Observed,
            observation: "list_files".to_owned(),
            content: "one.srt\ntwo.srt\n".to_owned(),
        });
        let activity = completed_activity(
            "list_files",
            &serde_json::json!({"path":"."}),
            &outcome,
            Duration::from_millis(1200),
        );
        assert_eq!(activity.headline, "Listed .");
        assert_eq!(activity.detail.as_deref(), Some("2 lines · 1.2s"));
    }

    #[test]
    fn patch_summary_counts_files_and_changed_lines() {
        let patch = "*** Begin Patch\n*** Update File: a.rs\n@@\n-old\n+new\n*** Add File: b.rs\n+hello\n*** End Patch";
        assert_eq!(patch_stats(patch), (2, 2, 1));
        let activity = running_activity("apply_patch", &serde_json::json!({"patch":patch}));
        assert_eq!(activity.headline, "Updating 2 files");
        assert_eq!(activity.detail.as_deref(), Some("+2 −1"));
    }

    #[test]
    fn failures_use_a_status_headline_and_concise_error() {
        let activity = failed_activity(
            "translate_file",
            &serde_json::json!({"path":"episode.srt","bilingual":true}),
            "provider rate limit exceeded",
        );
        assert_eq!(activity.headline, "Translation failed: episode.srt");
        assert_eq!(
            activity.detail.as_deref(),
            Some("provider rate limit exceeded")
        );
    }

    #[test]
    fn blocked_commands_do_not_expose_inline_credentials() {
        let activity = running_activity(
            "run_command",
            &serde_json::json!({"command":"curl -H 'Authorization: secret-value' example.test"}),
        );
        assert_eq!(activity.headline, "Running [blocked command]");
        assert!(!activity.headline.contains("secret-value"));
    }
}
