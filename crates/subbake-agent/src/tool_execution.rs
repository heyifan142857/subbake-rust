use std::path::{Path, PathBuf};

use serde_json::Value as JsonValue;
use subbake_adapters::{
    BatchTranslationRequest, ConfigFile, SettingsOverrides, StorageSettings, SubtitleEditRequest,
    TranscriptionFormat, TranscriptionRequest, TranscriptionSettings, TranslationRequest,
    TranslationSettings, WhisperAction, WhisperBuildVariant, WhisperOutcome, WhisperRequest,
    apply_whisper_storage, batch_translation_output_path, default_output_path_with_language,
    default_whisper_binary_path_for, default_whisper_models_dir_for, diagnose_failure_path,
    edit_subtitle_cancellable, format_diagnostic_report, is_supported_subtitle_path,
    load_diagnostic_reports, transcribe_media_cancellable, translate_subtitle_cancellable,
};
use subbake_core::diagnostics::diagnose_text;
use subbake_core::formats::{normalize_format, supported_format_from_path};
use subbake_core::languages::normalize_language;
use subbake_core::{
    AgentToolOutcome, BilingualOrder, CancellationGuard, FileToolOutcome, ObservationToolOutcome,
    ProfileToolOutcome, SharedProgress, SkippedPath, SubtitleEditToolOutcome, ToolExecutionStatus,
    TranscriptionToolOutcome, TranslationToolOutcome, WhisperModelFact, WhisperToolOutcome,
};

use crate::discovery::rank_subtitle_candidates;
use crate::error::{AgentError, AgentResult};
use crate::guard::{FileGuard, FileOpResult};
use crate::session::AgentEvent;
use crate::session::EventTag;
use crate::tools::ToolExecutor;

#[derive(Debug)]
pub(crate) struct LocalToolOutcome {
    pub outcome: AgentToolOutcome,
    pub file_operation: Option<FileOpResult>,
}

#[derive(Debug)]
pub(crate) struct TranslationExecutionOutcome {
    pub outcome: AgentToolOutcome,
    pub file_operations: Vec<FileOpResult>,
    pub group_file_operations: bool,
}

#[derive(Debug)]
pub(crate) struct AdapterToolOutcome {
    pub outcome: AgentToolOutcome,
    pub file_operation: Option<FileOpResult>,
}

#[derive(Debug)]
pub(crate) struct ProfileSwitch {
    pub name: String,
    pub config_path: PathBuf,
}

#[derive(Debug)]
pub(crate) struct SessionToolOutcome {
    pub outcome: AgentToolOutcome,
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
            read_only(
                "list_files",
                format_file_list(&guard.list_files(Path::new(dir))?),
            )
        }
        ToolExecutor::SearchFiles => {
            let dir = optional_string(args, "path", ".");
            let pattern = optional_string(args, "pattern", "");
            read_only(
                "search_files",
                format_file_list(&guard.search_files(Path::new(dir), pattern)?),
            )
        }
        ToolExecutor::ReadFile => {
            let path = optional_string(args, "path", "");
            read_only("read_file", guard.read_file(Path::new(path))?)
        }
        ToolExecutor::ReadFilePreview => {
            let path = optional_string(args, "path", "");
            let content = guard.read_file(Path::new(path))?;
            let preview = content.chars().take(2000).collect::<String>();
            read_only(
                "read_file_preview",
                if preview.len() < content.len() {
                    format!("{preview}\n… (truncated)")
                } else {
                    preview
                },
            )
        }
        ToolExecutor::CandidateSubtitles => {
            let dir = optional_string(args, "path", ".");
            let query = optional_string(args, "query", "");
            let files = guard.search_files(Path::new(dir), "")?;
            read_only(
                "candidate_subtitles",
                format_file_list(&rank_subtitle_candidates(files, query, project_root)),
            )
        }
        ToolExecutor::CreateFile => {
            let operation = guard.create_file(
                &required_path(args, "path")?,
                optional_string(args, "content", ""),
            )?;
            mutation("create", operation)
        }
        ToolExecutor::AppendFile => {
            let operation = guard.append_file(
                &required_path(args, "path")?,
                optional_string(args, "content", ""),
            )?;
            mutation("append", operation)
        }
        ToolExecutor::ReplaceInFile => {
            let operation = guard.replace_in_file(
                &required_path(args, "path")?,
                optional_string(args, "old", ""),
                optional_string(args, "new", ""),
            )?;
            mutation("replace", operation)
        }
        ToolExecutor::RenamePath => {
            let operation =
                guard.rename_path(&required_path(args, "from")?, &required_path(args, "to")?)?;
            mutation("rename", operation)
        }
        ToolExecutor::DeleteFile => {
            let operation = guard.delete_file(&required_path(args, "path")?)?;
            mutation("delete", operation)
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
    storage: Option<&StorageSettings>,
) -> AgentResult<Option<AdapterToolOutcome>> {
    let outcome = match executor {
        ToolExecutor::TranscribeAudio => {
            let input = guard.resolve_path(&required_path(args, "path")?)?;
            let mut settings = TranscriptionSettings::default();
            if let Some(storage) = storage {
                apply_whisper_storage(&mut settings, storage);
            }
            if let Some(language) = optional_argument(args, "language") {
                let normalized = normalize_language(language, true).map_err(|error| {
                    AgentError::InvalidInput {
                        message: error.to_string(),
                    }
                })?;
                settings.language = (normalized != "Auto").then_some(normalized);
            }
            if let Some(model) = optional_argument(args, "model") {
                settings.model = Some(nonempty_value("model", model)?);
            }
            if let Some(format) = optional_argument(args, "output_format") {
                settings.output_format =
                    TranscriptionFormat::parse(&format.trim().to_ascii_lowercase()).ok_or_else(
                        || AgentError::InvalidInput {
                            message: format!(
                                "unsupported transcription output format `{format}`; expected srt, vtt, or txt"
                            ),
                        },
                    )?;
            }
            let output = if let Some(path) = optional_argument(args, "output_path") {
                guard.resolve_path(Path::new(path))?
            } else {
                input.with_extension(settings.output_format.extension())
            };
            let overwrite = optional_bool(args, "overwrite", false);
            if output.exists() && !overwrite {
                return Err(AgentError::InvalidInput {
                    message: format!(
                        "output already exists and overwrite is false: {}",
                        output.display()
                    ),
                });
            }
            let snapshot = guard.snapshot_write(&output)?;
            let request = TranscriptionRequest {
                media_path: input.clone(),
                output_path: Some(output),
                overwrite,
                settings,
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
                Ok(outcome) => AdapterToolOutcome {
                    outcome: AgentToolOutcome::Transcription(TranscriptionToolOutcome {
                        status: ToolExecutionStatus::Written,
                        input,
                        output: outcome.output_path,
                        language: outcome.language,
                        provider: outcome.provider,
                        model: outcome.model,
                        output_format: outcome.output_format.extension().to_owned(),
                        subtitle_entries: outcome.subtitle_entries,
                    }),
                    file_operation: Some(snapshot),
                },
                Err(error) => return Err(error.into()),
            }
        }
        ToolExecutor::ManageWhisper => {
            let action = whisper_action(args)?;
            let build_variant = optional_argument(args, "variant")
                .map(|value| {
                    WhisperBuildVariant::parse(value).ok_or_else(|| AgentError::InvalidInput {
                        message: "whisper variant must be cpu, cuda, metal, vulkan, or openblas"
                            .to_owned(),
                    })
                })
                .transpose()?
                .unwrap_or_default();
            let action_name = whisper_action_name(&action).to_owned();
            let requested_model = match &action {
                WhisperAction::DownloadModel { name } => Some(name.clone()),
                _ => None,
            };
            let request = WhisperRequest {
                action,
                binary_path: storage.map(|storage| {
                    storage.whisper_binary_path.clone().unwrap_or_else(|| {
                        default_whisper_binary_path_for(storage.runtime_dir.as_deref())
                    })
                }),
                models_dir: storage.map(|storage| {
                    storage.whisper_models_dir.clone().unwrap_or_else(|| {
                        default_whisper_models_dir_for(storage.runtime_dir.as_deref())
                    })
                }),
                build_variant,
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
                Ok(managed) => {
                    let (
                        binary_path,
                        binary_exists,
                        models_dir,
                        models_dir_exists,
                        models,
                        available_models,
                        available_versions,
                        refresh_warning,
                    ) = match managed {
                        WhisperOutcome::Status(status) => (
                            Some(status.binary_path),
                            Some(status.binary_exists),
                            Some(status.models_dir),
                            Some(status.models_dir_exists),
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                            None,
                        ),
                        WhisperOutcome::ModelList(list) => (
                            None,
                            None,
                            Some(list.models_dir),
                            Some(list.models_dir_exists),
                            list.models
                                .into_iter()
                                .map(|model| WhisperModelFact {
                                    name: model.name,
                                    path: model.path,
                                })
                                .collect(),
                            list.available_models,
                            Vec::new(),
                            list.refresh_warning,
                        ),
                        WhisperOutcome::VersionList(list) => (
                            None,
                            None,
                            None,
                            None,
                            Vec::new(),
                            Vec::new(),
                            list.versions
                                .into_iter()
                                .map(|version| {
                                    if version.installable {
                                        format!("{} (installable)", version.tag)
                                    } else {
                                        version.tag
                                    }
                                })
                                .collect(),
                            list.refresh_warning,
                        ),
                    };
                    AdapterToolOutcome {
                        outcome: AgentToolOutcome::Whisper(WhisperToolOutcome {
                            status: if matches!(
                                action_name.as_str(),
                                "status" | "list_models" | "list_versions"
                            ) {
                                ToolExecutionStatus::Observed
                            } else {
                                ToolExecutionStatus::Completed
                            },
                            action: action_name,
                            requested_model,
                            binary_path,
                            binary_exists,
                            models_dir,
                            models_dir_exists,
                            models,
                            available_models,
                            available_versions,
                            refresh_warning,
                        }),
                        file_operation: None,
                    }
                }
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
            let text = if reports.is_empty() {
                "No failure logs found.".to_owned()
            } else {
                reports
                    .iter()
                    .map(format_diagnostic_report)
                    .collect::<Vec<_>>()
                    .join("\n\n---\n\n")
            };
            AdapterToolOutcome {
                outcome: observation("diagnose_path", text),
                file_operation: None,
            }
        }
        ToolExecutor::DiagnoseText => {
            let text = required_string(args, "text")?;
            AdapterToolOutcome {
                outcome: observation(
                    "diagnose_text",
                    format_diagnostic_report(&diagnose_text(&text, "pasted diagnostic text")),
                ),
                file_operation: None,
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(outcome))
}

pub(crate) fn execute_translation_tool(
    executor: ToolExecutor,
    args: &JsonValue,
    guard: &FileGuard,
    cancellation: &CancellationGuard,
    progress: Option<SharedProgress>,
    mut settings: TranslationSettings,
) -> AgentResult<Option<TranslationExecutionOutcome>> {
    let explicit_target_language = optional_argument(args, "target_language").is_some();
    settings.translation.source_language = normalize_language(
        optional_argument(args, "source_language").unwrap_or(&settings.translation.source_language),
        true,
    )
    .map_err(|error| AgentError::InvalidInput {
        message: error.to_string(),
    })?;
    settings.translation.target_language = normalize_language(
        optional_argument(args, "target_language").unwrap_or(&settings.translation.target_language),
        false,
    )
    .map_err(|error| AgentError::InvalidInput {
        message: error.to_string(),
    })?;
    if let Some(bilingual) = args.get("bilingual").and_then(JsonValue::as_bool) {
        settings.output.bilingual = bilingual;
    }
    if let Some(order) = optional_argument(args, "bilingual_order") {
        settings.output.bilingual_order =
            BilingualOrder::parse(order).map_err(|error| AgentError::InvalidInput {
                message: error.to_string(),
            })?;
    }
    if let Some(format) = optional_argument(args, "output_format") {
        settings.output.format =
            Some(
                normalize_format(format).map_err(|error| AgentError::InvalidInput {
                    message: error.to_string(),
                })?,
            );
    }
    settings.validate()?;

    let outcome = match executor {
        ToolExecutor::TranslateFile => {
            let input = guard.resolve_path(&required_path(args, "path")?)?;
            let language_tag =
                explicit_target_language.then(|| settings.translation.target_language.clone());
            let output_path = if let Some(path) = optional_argument(args, "output_path") {
                guard.resolve_path(Path::new(path))?
            } else {
                default_output_path_with_language(
                    &input,
                    settings.output_format(),
                    settings.output.bilingual,
                    language_tag.as_deref(),
                )?
            };
            let overwrite = optional_bool(args, "overwrite", false);
            if output_path.exists() && !overwrite && !settings.translation.dry_run {
                return Err(AgentError::InvalidInput {
                    message: format!(
                        "output already exists and overwrite is false: {}",
                        output_path.display()
                    ),
                });
            }
            let undo_snapshot = (!settings.translation.dry_run)
                .then(|| guard.snapshot_write(&output_path))
                .transpose()?;
            let output_format = resolved_translation_format(&input, &settings)?;
            let request = TranslationRequest {
                input_path: input.clone(),
                output_path: Some(output_path),
                output_language_tag: language_tag,
                overwrite,
                settings: settings.clone(),
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
            let file_operations = undo_snapshot.into_iter().collect();
            TranslationExecutionOutcome {
                outcome: AgentToolOutcome::Translation(TranslationToolOutcome {
                    status: if translated.result.dry_run {
                        ToolExecutionStatus::DryRun
                    } else {
                        ToolExecutionStatus::Written
                    },
                    source_language: settings.translation.source_language,
                    target_language: settings.translation.target_language,
                    provider: settings.backend.id,
                    model: settings.backend.model,
                    output_format,
                    bilingual: settings.output.bilingual,
                    bilingual_order: settings.output.bilingual_order,
                    inputs: vec![input],
                    outputs: translated.output_path.into_iter().collect(),
                    processed_files: 1,
                    skipped: Vec::new(),
                    subtitle_entries: translated.subtitle_entries,
                    dry_run: translated.result.dry_run,
                    cache_hits: translated.result.cache_hits,
                    resumed_translation_batches: translated.result.resumed_translation_batches,
                    resumed_review_batches: translated.result.resumed_review_batches,
                    translation_memory_hits: translated.result.translation_memory_hits,
                }),
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
            let output_dir = optional_argument(args, "output_dir")
                .map(|path| guard.resolve_path(Path::new(path)))
                .transpose()?;
            let language_tag =
                explicit_target_language.then(|| settings.translation.target_language.clone());
            let request = BatchTranslationRequest {
                root: input.clone(),
                recursive,
                overwrite,
                output_dir,
                output_language_tag: language_tag,
                settings: settings.clone(),
            };
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
                let output = batch_translation_output_path(&request, &source)?;
                if !settings.translation.dry_run && (overwrite || !output.exists()) {
                    undo_snapshots.push((output.clone(), guard.snapshot_write(&output)?));
                }
            }
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
            TranslationExecutionOutcome {
                outcome: AgentToolOutcome::Translation(TranslationToolOutcome {
                    status: if translated.dry_run {
                        ToolExecutionStatus::DryRun
                    } else if translated.processed == 0 {
                        ToolExecutionStatus::Skipped
                    } else {
                        ToolExecutionStatus::Written
                    },
                    source_language: settings.translation.source_language,
                    target_language: settings.translation.target_language,
                    provider: settings.backend.id,
                    model: settings.backend.model,
                    output_format: settings
                        .output
                        .format
                        .unwrap_or_else(|| "source".to_owned()),
                    bilingual: settings.output.bilingual,
                    bilingual_order: settings.output.bilingual_order,
                    inputs: translated.inputs,
                    outputs: translated.outputs,
                    processed_files: translated.processed,
                    skipped: translated
                        .skipped
                        .into_iter()
                        .map(|path| SkippedPath {
                            path,
                            reason: "output exists and overwrite is false".to_owned(),
                        })
                        .collect(),
                    subtitle_entries: translated.subtitle_entries,
                    dry_run: translated.dry_run,
                    cache_hits: translated.cache_hits,
                    resumed_translation_batches: translated.resumed_translation_batches,
                    resumed_review_batches: translated.resumed_review_batches,
                    translation_memory_hits: translated.translation_memory_hits,
                }),
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
            TranslationExecutionOutcome {
                outcome: AgentToolOutcome::SubtitleEdit(SubtitleEditToolOutcome {
                    status: ToolExecutionStatus::Written,
                    target_path: edited.target_path,
                    target_language: edited.target_language,
                    modified_entries: edited.modified_entries,
                    edit_notes: edited.edit_notes,
                }),
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
            outcome: observation("recent_translations", recent_translation_paths(events)),
            profile_switch: None,
        },
        ToolExecutor::ListProfiles => {
            let Some((_, config)) = config else {
                return Ok(Some(SessionToolOutcome {
                    outcome: AgentToolOutcome::Profile(ProfileToolOutcome {
                        status: ToolExecutionStatus::Observed,
                        action: "list".to_owned(),
                        active_profile: active_profile.map(str::to_owned),
                        provider: None,
                        model: None,
                        available_profiles: Vec::new(),
                        message: "No subbake config found in project root.".to_owned(),
                    }),
                    profile_switch: None,
                }));
            };
            let mut profiles = config.profiles.keys().cloned().collect::<Vec<_>>();
            profiles.sort();
            SessionToolOutcome {
                outcome: AgentToolOutcome::Profile(ProfileToolOutcome {
                    status: ToolExecutionStatus::Observed,
                    action: "list".to_owned(),
                    active_profile: active_profile.map(str::to_owned),
                    provider: None,
                    model: None,
                    available_profiles: profiles.clone(),
                    message: if profiles.is_empty() {
                        "No profiles defined in subbake.toml. Create [profiles.<name>] sections."
                            .to_owned()
                    } else {
                        format_profile_list(&profiles, active_profile)
                    },
                }),
                profile_switch: None,
            }
        }
        ToolExecutor::SwitchProfile => {
            let name = required_string(args, "name")?;
            let Some((config_path, config)) = config else {
                return Ok(Some(SessionToolOutcome {
                    outcome: AgentToolOutcome::Profile(ProfileToolOutcome {
                        status: ToolExecutionStatus::Unchanged,
                        action: "switch".to_owned(),
                        active_profile: active_profile.map(str::to_owned),
                        provider: None,
                        model: None,
                        available_profiles: Vec::new(),
                        message:
                            "No subbake config found. Create one with [profiles.<name>] sections."
                                .to_owned(),
                    }),
                    profile_switch: None,
                }));
            };
            if !config.profiles.contains_key(&name) {
                let mut profiles = config.profiles.keys().cloned().collect::<Vec<_>>();
                profiles.sort();
                return Ok(Some(SessionToolOutcome {
                    outcome: AgentToolOutcome::Profile(ProfileToolOutcome {
                        status: ToolExecutionStatus::Unchanged,
                        action: "switch".to_owned(),
                        active_profile: active_profile.map(str::to_owned),
                        provider: None,
                        model: None,
                        available_profiles: profiles.clone(),
                        message: format!(
                            "Profile `{name}` not found. Available: {}",
                            profiles.join(", ")
                        ),
                    }),
                    profile_switch: None,
                }));
            }
            let (settings, _) = config
                .resolve(Some(&name), SettingsOverrides::default())
                .map_err(subbake_adapters::AdapterError::from)?;
            SessionToolOutcome {
                outcome: AgentToolOutcome::Profile(ProfileToolOutcome {
                    status: ToolExecutionStatus::Completed,
                    action: "switch".to_owned(),
                    active_profile: Some(name.clone()),
                    provider: Some(settings.backend.id),
                    model: Some(settings.backend.model),
                    available_profiles: Vec::new(),
                    message: format!("Profile switched: {name}"),
                }),
                profile_switch: Some(ProfileSwitch { name, config_path }),
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(outcome))
}

fn read_only(observation_name: &str, content: String) -> LocalToolOutcome {
    LocalToolOutcome {
        outcome: observation(observation_name, content),
        file_operation: None,
    }
}

fn mutation(action: &str, file_operation: FileOpResult) -> LocalToolOutcome {
    let destination_paths = file_operation.new_path.clone().into_iter().collect();
    LocalToolOutcome {
        outcome: AgentToolOutcome::File(FileToolOutcome {
            status: ToolExecutionStatus::Written,
            action: action.to_owned(),
            paths: vec![file_operation.path.clone()],
            destination_paths,
        }),
        file_operation: Some(file_operation),
    }
}

fn observation(name: &str, content: String) -> AgentToolOutcome {
    AgentToolOutcome::Observation(ObservationToolOutcome {
        status: ToolExecutionStatus::Observed,
        observation: name.to_owned(),
        content,
    })
}

fn optional_string<'a>(args: &'a JsonValue, key: &str, default: &'a str) -> &'a str {
    args.get(key).and_then(JsonValue::as_str).unwrap_or(default)
}

fn optional_argument<'a>(args: &'a JsonValue, key: &str) -> Option<&'a str> {
    args.get(key).and_then(JsonValue::as_str)
}

fn optional_bool(args: &JsonValue, key: &str, default: bool) -> bool {
    args.get(key)
        .and_then(JsonValue::as_bool)
        .unwrap_or(default)
}

fn nonempty_value(name: &str, value: &str) -> AgentResult<String> {
    if value.trim().is_empty() {
        Err(AgentError::InvalidInput {
            message: format!("`{name}` must not be empty"),
        })
    } else {
        Ok(value.trim().to_owned())
    }
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
        "list-versions" | "versions" => Ok(WhisperAction::ListVersions),
        "list-models" | "models" => Ok(WhisperAction::ListModels),
        "download" | "download_model" => Ok(WhisperAction::DownloadModel {
            name: optional_argument(args, "model")
                .map(|model| nonempty_value("model", model))
                .transpose()?
                .ok_or_else(|| AgentError::ToolArguments {
                    message: "download requires an explicitly selected `model`".to_owned(),
                })?,
        }),
        other => Err(AgentError::InvalidInput {
            message: format!("unknown whisper action `{other}`"),
        }),
    }
}

fn whisper_action_name(action: &WhisperAction) -> &'static str {
    match action {
        WhisperAction::Status => "status",
        WhisperAction::ListVersions => "list_versions",
        WhisperAction::Install => "install",
        WhisperAction::Update => "update",
        WhisperAction::Uninstall { .. } => "uninstall",
        WhisperAction::ListModels => "list_models",
        WhisperAction::DownloadModel { .. } => "download_model",
    }
}

fn resolved_translation_format(
    input: &Path,
    settings: &TranslationSettings,
) -> AgentResult<String> {
    match settings.output_format() {
        Some(format) => normalize_format(format).map_err(|error| AgentError::InvalidInput {
            message: error.to_string(),
        }),
        None => supported_format_from_path(input)
            .map(str::to_owned)
            .ok_or_else(|| AgentError::InvalidInput {
                message: format!("unsupported subtitle format: {}", input.display()),
            }),
    }
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

        assert!(matches!(outcome.outcome, AgentToolOutcome::File(_)));
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
            None,
        )
        .expect("execute")
        .expect("adapter outcome");
        assert!(render_tool_outcome(&outcome.outcome).contains("Provider rate limit was hit."));
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
            None,
        )
        .expect_err("invalid whisper action must fail");
        assert!(matches!(error, AgentError::InvalidInput { .. }));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn whisper_download_requires_an_explicit_model_choice() {
        let error = whisper_action(&json!({"action": "download"}))
            .expect_err("download without a selected model must fail");

        assert!(matches!(error, AgentError::ToolArguments { .. }));
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

        let AgentToolOutcome::Translation(facts) = outcome.outcome else {
            panic!("expected translation facts");
        };
        assert_eq!(facts.processed_files, 1);
        assert!(facts.skipped.is_empty());
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

    #[test]
    fn explicit_japanese_override_changes_only_that_call_and_output_name() {
        let root = temp_root();
        fs::create_dir_all(&root).expect("create root");
        fs::write(
            root.join("sample.srt"),
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
        )
        .expect("write subtitle");
        let guard = FileGuard::new(root.clone());
        let settings = TranslationSettings::default();

        let japanese = execute_translation_tool(
            ToolExecutor::TranslateFile,
            &json!({"path": "sample.srt", "target_language": "Japanese"}),
            &guard,
            &CancellationGuard::never(),
            None,
            settings.clone(),
        )
        .expect("translate Japanese")
        .expect("Japanese outcome");
        let AgentToolOutcome::Translation(japanese_facts) = japanese.outcome else {
            panic!("expected translation facts");
        };

        let profile_default = execute_translation_tool(
            ToolExecutor::TranslateFile,
            &json!({"path": "sample.srt"}),
            &guard,
            &CancellationGuard::never(),
            None,
            settings,
        )
        .expect("translate profile default")
        .expect("default outcome");
        let AgentToolOutcome::Translation(default_facts) = profile_default.outcome else {
            panic!("expected translation facts");
        };

        assert_eq!(japanese_facts.target_language, "ja");
        assert_eq!(
            japanese_facts.outputs,
            vec![root.join("sample.ja.translated.srt")]
        );
        assert_eq!(default_facts.target_language, "zh-Hans");
        assert_eq!(
            default_facts.outputs,
            vec![root.join("sample.translated.srt")]
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn dry_run_reports_no_output_and_creates_no_undo_operation() {
        let root = temp_root();
        fs::create_dir_all(&root).expect("create root");
        fs::write(root.join("sample.txt"), "Hello\n").expect("write subtitle");
        let guard = FileGuard::new(root.clone());
        let mut settings = TranslationSettings::default();
        settings.translation.dry_run = true;

        let outcome = execute_translation_tool(
            ToolExecutor::TranslateFile,
            &json!({"path": "sample.txt", "target_language": "Japanese"}),
            &guard,
            &CancellationGuard::never(),
            None,
            settings,
        )
        .expect("dry run")
        .expect("translation outcome");
        let AgentToolOutcome::Translation(facts) = outcome.outcome else {
            panic!("expected translation facts");
        };

        assert_eq!(facts.status, ToolExecutionStatus::DryRun);
        assert!(facts.outputs.is_empty());
        assert!(outcome.file_operations.is_empty());
        assert!(!root.join("sample.ja.translated.txt").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn invalid_language_fails_before_creating_output_or_undo_data() {
        let root = temp_root();
        fs::create_dir_all(&root).expect("create root");
        fs::write(root.join("sample.txt"), "Hello\n").expect("write subtitle");
        let guard = FileGuard::new(root.clone());

        let error = execute_translation_tool(
            ToolExecutor::TranslateFile,
            &json!({"path": "sample.txt", "target_language": "und"}),
            &guard,
            &CancellationGuard::never(),
            None,
            TranslationSettings::default(),
        )
        .expect_err("invalid language must fail");

        assert!(error.to_string().contains("BCP-47"));
        assert!(!root.join("sample.translated.txt").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn recursive_series_preserves_relative_structure_in_output_dir() {
        let root = temp_root();
        fs::create_dir_all(root.join("season")).expect("create nested root");
        fs::write(
            root.join("season/episode.srt"),
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
        )
        .expect("write subtitle");
        let guard = FileGuard::new(root.clone());

        let outcome = execute_translation_tool(
            ToolExecutor::TranslateSeries,
            &json!({
                "path": ".",
                "recursive": true,
                "target_language": "Japanese",
                "output_dir": "translated"
            }),
            &guard,
            &CancellationGuard::never(),
            None,
            TranslationSettings::default(),
        )
        .expect("translate series")
        .expect("series outcome");
        let AgentToolOutcome::Translation(facts) = outcome.outcome else {
            panic!("expected translation facts");
        };

        assert_eq!(
            facts.outputs,
            vec![root.join("translated/season/episode.ja.translated.srt")]
        );
        assert!(facts.outputs[0].exists());
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
