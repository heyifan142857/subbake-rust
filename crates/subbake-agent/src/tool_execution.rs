use std::path::{Path, PathBuf};

use serde_json::Value as JsonValue;
use subbake_adapters::{
    BatchTranslationRequest, ConfigFile, SettingsOverrides, SubtitleEditRequest,
    TranscriptionRequest, TranscriptionSettings, TranslationRequest, TranslationSettings,
    WhisperAction, WhisperRequest, default_output_path, diagnose_failure_path,
    edit_subtitle_cancellable, format_diagnostic_report, is_supported_subtitle_path,
    load_diagnostic_reports, transcribe_media_cancellable, translate_subtitle_cancellable,
};
use subbake_core::diagnostics::diagnose_text;
use subbake_core::{CancellationGuard, SharedProgress};

use crate::discovery::rank_subtitle_candidates;
use crate::error::{AgentError, AgentResult};
use crate::guard::{FileGuard, FileOpResult};
use crate::session::AgentEvent;
use crate::session::EventTag;
use crate::tools::ToolExecutor;

pub(crate) struct LocalToolOutcome {
    pub text: String,
    pub file_operation: Option<FileOpResult>,
}

pub(crate) struct TranslationToolOutcome {
    pub text: String,
    pub file_operations: Vec<FileOpResult>,
    pub group_file_operations: bool,
}

pub(crate) struct ProfileSwitch {
    pub name: String,
    pub config_path: PathBuf,
}

pub(crate) struct SessionToolOutcome {
    pub text: String,
    pub profile_switch: Option<ProfileSwitch>,
}

pub(crate) fn execute_local_tool(
    executor: ToolExecutor,
    args: &JsonValue,
    guard: &FileGuard,
    project_root: &Path,
) -> AgentResult<Option<LocalToolOutcome>> {
    let outcome = match executor {
        ToolExecutor::ListFiles => {
            let dir = optional_string(args, "path", ".");
            read_only(format_file_list(&guard.list_files(Path::new(dir))?))
        }
        ToolExecutor::SearchFiles => {
            let dir = optional_string(args, "path", ".");
            let pattern = optional_string(args, "pattern", "");
            read_only(format_file_list(
                &guard.search_files(Path::new(dir), pattern)?,
            ))
        }
        ToolExecutor::ReadFile => {
            let path = optional_string(args, "path", "");
            read_only(guard.read_file(Path::new(path))?)
        }
        ToolExecutor::ReadFilePreview => {
            let path = optional_string(args, "path", "");
            let content = guard.read_file(Path::new(path))?;
            let preview = content.chars().take(2000).collect::<String>();
            read_only(if preview.len() < content.len() {
                format!("{preview}\n… (truncated)")
            } else {
                preview
            })
        }
        ToolExecutor::CandidateSubtitles => {
            let dir = optional_string(args, "path", ".");
            let query = optional_string(args, "query", "");
            let files = guard.search_files(Path::new(dir), "")?;
            read_only(format_file_list(&rank_subtitle_candidates(
                files,
                query,
                project_root,
            )))
        }
        ToolExecutor::CreateFile => {
            let operation = guard.create_file(
                &required_path(args, "path")?,
                optional_string(args, "content", ""),
            )?;
            mutation(format!("Created {}", operation.path.display()), operation)
        }
        ToolExecutor::AppendFile => {
            let operation = guard.append_file(
                &required_path(args, "path")?,
                optional_string(args, "content", ""),
            )?;
            mutation(
                format!(
                    "Appended {} (backup: {})",
                    operation.path.display(),
                    backup_label(&operation)
                ),
                operation,
            )
        }
        ToolExecutor::ReplaceInFile => {
            let operation = guard.replace_in_file(
                &required_path(args, "path")?,
                optional_string(args, "old", ""),
                optional_string(args, "new", ""),
            )?;
            mutation(
                format!(
                    "Replaced in {} (backup: {})",
                    operation.path.display(),
                    backup_label(&operation)
                ),
                operation,
            )
        }
        ToolExecutor::RenamePath => {
            let operation =
                guard.rename_path(&required_path(args, "from")?, &required_path(args, "to")?)?;
            let destination = operation
                .new_path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default();
            mutation(
                format!("Renamed {} → {destination}", operation.path.display()),
                operation,
            )
        }
        ToolExecutor::DeleteFile => {
            let operation = guard.delete_file(&required_path(args, "path")?)?;
            mutation(
                format!(
                    "Deleted {} (backup: {})",
                    operation.path.display(),
                    backup_label(&operation)
                ),
                operation,
            )
        }
        _ => return Ok(None),
    };
    Ok(Some(outcome))
}

pub(crate) fn execute_adapter_tool(
    executor: ToolExecutor,
    args: &JsonValue,
    guard: &FileGuard,
    cancellation: &CancellationGuard,
    progress: Option<SharedProgress>,
) -> AgentResult<Option<String>> {
    let text = match executor {
        ToolExecutor::TranscribeAudio => {
            let input = guard.resolve_path(&required_path(args, "path")?)?;
            let request = TranscriptionRequest {
                media_path: input,
                output_path: None,
                settings: TranscriptionSettings::default(),
            };
            let transcribed = if let Some(progress) = progress {
                subbake_adapters::transcribe_media_cancellable_with_progress(
                    request,
                    cancellation,
                    progress,
                )
            } else {
                transcribe_media_cancellable(request, cancellation)
            };
            match transcribed {
                Err(error) if error.is_cancelled() => return Err(error.into()),
                Ok(outcome) => format!("Transcribed: {}", outcome.output_path.display()),
                Err(error) => return Err(error.into()),
            }
        }
        ToolExecutor::ManageWhisper => {
            let action = whisper_action(args)?;
            let request = WhisperRequest {
                action,
                binary_path: None,
                models_dir: None,
            };
            let managed = if let Some(progress) = progress {
                subbake_adapters::run_whisper_cancellable_with_progress(
                    request,
                    cancellation,
                    progress,
                )
            } else {
                subbake_adapters::run_whisper_cancellable(request, cancellation)
            };
            match managed {
                Err(error) if error.is_cancelled() => return Err(error.into()),
                Ok(_) => "whisper: done".to_owned(),
                Err(error) => return Err(error.into()),
            }
        }
        ToolExecutor::DiagnosePath => {
            let full = guard.resolve_path(&required_path(args, "path")?)?;
            let reports = if full.is_file() {
                vec![diagnose_failure_path(&full)?]
            } else {
                load_diagnostic_reports(&full)?
            };
            if reports.is_empty() {
                "No failure logs found.".to_owned()
            } else {
                reports
                    .iter()
                    .map(format_diagnostic_report)
                    .collect::<Vec<_>>()
                    .join("\n\n---\n\n")
            }
        }
        ToolExecutor::DiagnoseText => {
            let text = required_string(args, "text")?;
            format_diagnostic_report(&diagnose_text(&text, "pasted diagnostic text"))
        }
        _ => return Ok(None),
    };
    Ok(Some(text))
}

pub(crate) fn execute_translation_tool(
    executor: ToolExecutor,
    args: &JsonValue,
    guard: &FileGuard,
    cancellation: &CancellationGuard,
    progress: Option<SharedProgress>,
    mut settings: TranslationSettings,
) -> AgentResult<Option<TranslationToolOutcome>> {
    if let Some(bilingual) = args.get("bilingual").and_then(JsonValue::as_bool) {
        settings.output.bilingual = bilingual;
    }

    let outcome = match executor {
        ToolExecutor::TranslateFile => {
            let input = guard.resolve_path(&required_path(args, "path")?)?;
            let output_path =
                default_output_path(&input, settings.output_format(), settings.output.bilingual)?;
            let undo_snapshot = guard.snapshot_write(&output_path)?;
            let request = TranslationRequest {
                input_path: input,
                output_path: None,
                settings,
            };
            let translated = if let Some(progress) = progress {
                subbake_adapters::translate_subtitle_cancellable_with_progress(
                    request,
                    cancellation,
                    progress,
                )?
            } else {
                translate_subtitle_cancellable(request, cancellation)?
            };
            let file_operations = translated
                .output_path
                .as_ref()
                .map(|_| vec![undo_snapshot])
                .unwrap_or_default();
            TranslationToolOutcome {
                text: translated
                    .output_path
                    .map(|path| format!("Translated: {}", path.display()))
                    .unwrap_or_default(),
                file_operations,
                group_file_operations: false,
            }
        }
        ToolExecutor::TranslateSeries => {
            let input = guard.resolve_path(&required_path(args, "path")?)?;
            let recursive = args
                .get("recursive")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            let overwrite = args
                .get("overwrite")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            let source_files = if recursive {
                guard.search_files(&input, "")?
            } else {
                guard.list_files(&input)?
            };
            let mut undo_snapshots = Vec::new();
            for source in source_files
                .into_iter()
                .filter(|path| path.is_file() && is_supported_subtitle_path(path))
                .filter(|path| !is_generated_subtitle(path))
            {
                let output = default_output_path(
                    &source,
                    settings.output_format(),
                    settings.output.bilingual,
                )?;
                if overwrite || !output.exists() {
                    undo_snapshots.push((output.clone(), guard.snapshot_write(&output)?));
                }
            }
            let request = BatchTranslationRequest {
                root: input,
                recursive,
                overwrite,
                settings,
            };
            let translated = if let Some(progress) = progress {
                subbake_adapters::translate_subtitle_batch_with_progress(
                    request,
                    cancellation,
                    progress,
                )?
            } else {
                subbake_adapters::translate_subtitle_batch_cancellable(request, cancellation)?
            };
            let file_operations = translated
                .outputs
                .iter()
                .filter_map(|output| {
                    undo_snapshots
                        .iter()
                        .find(|(path, _)| path == output)
                        .map(|(_, snapshot)| snapshot.clone())
                })
                .collect();
            TranslationToolOutcome {
                text: format!(
                    "batch_translation: processed={}, skipped={}",
                    translated.processed,
                    translated.skipped.len()
                ),
                file_operations,
                group_file_operations: true,
            }
        }
        ToolExecutor::EditSubtitle => {
            let target_path = guard.resolve_path(&required_path(args, "path")?)?;
            let snapshot = guard.snapshot_write(&target_path)?;
            let edited = edit_subtitle_cancellable(
                SubtitleEditRequest {
                    target_path: target_path.clone(),
                    instruction: required_string(args, "instruction")?,
                    settings,
                    allow_non_generated: args
                        .get("allow_non_generated")
                        .and_then(JsonValue::as_bool)
                        .unwrap_or(false),
                },
                cancellation,
            )?;
            let mut lines = vec![format!("Edited: {}", target_path.display())];
            if !edited.edit_notes.trim().is_empty() {
                lines.push(edited.edit_notes);
            }
            TranslationToolOutcome {
                text: lines.join("\n"),
                file_operations: vec![snapshot],
                group_file_operations: false,
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(outcome))
}

pub(crate) fn execute_session_tool(
    executor: ToolExecutor,
    args: &JsonValue,
    events: &[AgentEvent],
    config: Option<(PathBuf, ConfigFile)>,
    active_profile: Option<&str>,
) -> AgentResult<Option<SessionToolOutcome>> {
    let outcome = match executor {
        ToolExecutor::RecentTranslations => SessionToolOutcome {
            text: recent_translation_paths(events),
            profile_switch: None,
        },
        ToolExecutor::ListProfiles => {
            let Some((_, config)) = config else {
                return Ok(Some(SessionToolOutcome {
                    text: "No subbake config found in project root.".to_owned(),
                    profile_switch: None,
                }));
            };
            let mut profiles = config.profiles.keys().cloned().collect::<Vec<_>>();
            profiles.sort();
            SessionToolOutcome {
                text: if profiles.is_empty() {
                    "No profiles defined in subbake.toml. Create [profiles.<name>] sections."
                        .to_owned()
                } else {
                    format_profile_list(&profiles, active_profile)
                },
                profile_switch: None,
            }
        }
        ToolExecutor::SwitchProfile => {
            let name = required_string(args, "name")?;
            let Some((config_path, config)) = config else {
                return Ok(Some(SessionToolOutcome {
                    text: "No subbake config found. Create one with [profiles.<name>] sections."
                        .to_owned(),
                    profile_switch: None,
                }));
            };
            if !config.profiles.contains_key(&name) {
                let mut profiles = config.profiles.keys().cloned().collect::<Vec<_>>();
                profiles.sort();
                return Ok(Some(SessionToolOutcome {
                    text: format!(
                        "Profile `{name}` not found. Available: {}",
                        profiles.join(", ")
                    ),
                    profile_switch: None,
                }));
            }
            let (settings, _) = config
                .resolve(Some(&name), SettingsOverrides::default())
                .map_err(subbake_adapters::AdapterError::from)?;
            SessionToolOutcome {
                text: format!(
                    "Profile switched: {name} ({}/{})",
                    settings.backend.id, settings.backend.model
                ),
                profile_switch: Some(ProfileSwitch { name, config_path }),
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(outcome))
}

fn read_only(text: String) -> LocalToolOutcome {
    LocalToolOutcome {
        text,
        file_operation: None,
    }
}

fn mutation(text: String, file_operation: FileOpResult) -> LocalToolOutcome {
    LocalToolOutcome {
        text,
        file_operation: Some(file_operation),
    }
}

fn optional_string<'a>(args: &'a JsonValue, key: &str, default: &'a str) -> &'a str {
    args.get(key).and_then(JsonValue::as_str).unwrap_or(default)
}

fn required_path(args: &JsonValue, key: &str) -> AgentResult<PathBuf> {
    args.get(key)
        .and_then(JsonValue::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| AgentError::ToolArguments {
            message: format!("missing required argument `{key}`"),
        })
}

fn required_string(args: &JsonValue, key: &str) -> AgentResult<String> {
    args.get(key)
        .and_then(JsonValue::as_str)
        .map(str::to_owned)
        .ok_or_else(|| AgentError::ToolArguments {
            message: format!("missing required argument `{key}`"),
        })
}

fn whisper_action(args: &JsonValue) -> AgentResult<WhisperAction> {
    match optional_string(args, "action", "status") {
        "install" => Ok(WhisperAction::Install),
        "update" => Ok(WhisperAction::Update),
        "uninstall" => Ok(WhisperAction::Uninstall {
            keep_models: args
                .get("keep_models")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false),
        }),
        "status" => Ok(WhisperAction::Status),
        "list-models" | "models" => Ok(WhisperAction::ListModels),
        "download" | "download_model" => Ok(WhisperAction::DownloadModel {
            name: optional_string(args, "model", "small").to_owned(),
        }),
        other => Err(AgentError::InvalidInput {
            message: format!("unknown whisper action `{other}`"),
        }),
    }
}

fn backup_label(operation: &FileOpResult) -> String {
    operation
        .backup_path
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_default()
}

fn is_generated_subtitle(path: &Path) -> bool {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .is_some_and(|stem| stem.ends_with(".translated") || stem.ends_with(".bilingual"))
}

fn recent_translation_paths(events: &[AgentEvent]) -> String {
    events
        .iter()
        .rev()
        .take(20)
        .filter(|event| {
            event.tag() == EventTag::FileOperation
                && !event
                    .data
                    .get("undone")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false)
        })
        .filter_map(|event| event.data.get("path").and_then(JsonValue::as_str))
        .filter(|path| path.contains(".translated.") || path.contains(".bilingual."))
        .map(str::to_owned)
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_profile_list(profiles: &[String], active: Option<&str>) -> String {
    let rendered = profiles
        .iter()
        .map(|name| {
            if Some(name.as_str()) == active {
                format!("{name} (active)")
            } else {
                name.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("Profiles: {rendered}")
}

fn format_file_list(files: &[PathBuf]) -> String {
    if files.is_empty() {
        return "(no files found)".to_owned();
    }
    files
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::{Arc, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;
    use subbake_core::{ProgressEvent, ProgressSink, TaskKind};

    use super::*;

    #[derive(Default)]
    struct RecordingProgress {
        events: Mutex<Vec<ProgressEvent>>,
    }

    impl ProgressSink for RecordingProgress {
        fn emit(&self, event: ProgressEvent) {
            self.events.lock().expect("progress events").push(event);
        }
    }

    #[test]
    fn local_mutation_returns_undo_bookkeeping_data() {
        let root = temp_root();
        fs::create_dir_all(&root).expect("create root");
        let guard = FileGuard::new(root.clone());
        let outcome = execute_local_tool(
            ToolExecutor::CreateFile,
            &json!({"path": "note.txt", "content": "hello"}),
            &guard,
            &root,
        )
        .expect("execute")
        .expect("local outcome");

        assert_eq!(
            outcome.text,
            format!("Created {}", root.join("note.txt").display())
        );
        assert!(outcome.file_operation.is_some());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn non_local_tool_is_left_for_the_service_executor() {
        let root = temp_root();
        let guard = FileGuard::new(root.clone());
        let outcome = execute_local_tool(ToolExecutor::TranslateFile, &json!({}), &guard, &root)
            .expect("execute");
        assert!(outcome.is_none());
    }

    #[test]
    fn diagnostic_text_is_executed_without_engine_state() {
        let root = temp_root();
        let guard = FileGuard::new(root);
        let outcome = execute_adapter_tool(
            ToolExecutor::DiagnoseText,
            &json!({"text": "HTTP status=429"}),
            &guard,
            &CancellationGuard::never(),
            None,
        )
        .expect("execute")
        .expect("adapter outcome");
        assert!(outcome.contains("Provider rate limit was hit."));
    }

    #[test]
    fn invalid_whisper_action_is_an_error_not_success_text() {
        let root = temp_root();
        let guard = FileGuard::new(root.clone());
        let error = execute_adapter_tool(
            ToolExecutor::ManageWhisper,
            &json!({"action": "not-a-command"}),
            &guard,
            &CancellationGuard::never(),
            None,
        )
        .expect_err("invalid whisper action must fail");
        assert!(matches!(error, AgentError::InvalidInput { .. }));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn translate_series_forwards_batch_progress() {
        let root = temp_root();
        fs::create_dir_all(&root).expect("create root");
        fs::write(
            root.join("episode.srt"),
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
        )
        .expect("write subtitle");
        let guard = FileGuard::new(root.clone());
        let progress = Arc::new(RecordingProgress::default());

        let outcome = execute_translation_tool(
            ToolExecutor::TranslateSeries,
            &json!({"path": root}),
            &guard,
            &CancellationGuard::never(),
            Some(progress.clone()),
            TranslationSettings::default(),
        )
        .expect("translate series")
        .expect("translation outcome");

        assert_eq!(outcome.text, "batch_translation: processed=1, skipped=0");
        assert!(
            progress
                .events
                .lock()
                .expect("progress events")
                .iter()
                .any(|event| event.task == TaskKind::BatchTranslation)
        );
        let _ = fs::remove_dir_all(root);
    }

    fn temp_root() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-local-tools-{nanos}"))
    }
}
