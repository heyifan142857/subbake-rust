use std::path::Path;

use serde_json::Value as JsonValue;
use subbake_core::{AgentToolOutcome, FileToolOutcome, ToolExecutionStatus};

use crate::engine::AgentEngine;
use crate::error::{AgentError, AgentResult};
use crate::event::{EventKind, FileOpEventData};
use crate::guard::{FileOpAction, FileOpResult};
use crate::profile_coordinator::ProfileCoordinator;
use crate::tool_execution::{
    execute_adapter_tool, execute_local_tool, execute_session_tool, execute_translation_tool,
};
use crate::tools::ToolExecutor;

pub(crate) struct ToolRunner;

impl ToolRunner {
    pub(crate) fn run(
        engine: &mut AgentEngine,
        name: &str,
        args: &JsonValue,
    ) -> AgentResult<AgentToolOutcome> {
        engine.check_cancelled()?;
        engine.record_if_active(EventKind::ToolCall {
            tool_name: name.to_owned(),
            arguments: args.clone(),
        })?;
        let executor = crate::tools::find_tool_spec(name)
            .map(|spec| spec.executor)
            .ok_or_else(|| AgentError::InvalidInput {
                message: format!("unknown agent tool `{name}`"),
            })?;

        if executor == ToolExecutor::ApplyPatch {
            let patch = args
                .get("patch")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| AgentError::ToolArguments {
                    message: "missing required argument `patch`".to_owned(),
                })?;
            let outcome = crate::patch::apply_patch(patch, &engine.guard)?;
            let group_id = (outcome.file_operations.len() > 1)
                .then(|| format!("apply-patch-{}", crate::session::iso_now()));
            for operation in &outcome.file_operations {
                Self::record_file_operation(engine, operation, group_id.clone())?;
            }
            return Ok(AgentToolOutcome::File(FileToolOutcome {
                status: ToolExecutionStatus::Written,
                action: "apply_patch".to_owned(),
                paths: outcome
                    .file_operations
                    .iter()
                    .map(|operation| operation.path.clone())
                    .collect(),
                destination_paths: outcome
                    .file_operations
                    .iter()
                    .filter_map(|operation| operation.new_path.clone())
                    .collect(),
            }));
        }

        if let Some(outcome) =
            execute_local_tool(executor, args, &engine.guard, &engine.project_root)?
        {
            if let Some(operation) = outcome.file_operation {
                Self::record_file_operation(engine, &operation, None)?;
            }
            return Ok(outcome.outcome);
        }
        let adapter_settings = if matches!(
            executor,
            ToolExecutor::TranscribeAudio | ToolExecutor::ManageWhisper
        ) {
            Some(
                ProfileCoordinator::new(&engine.project_root, engine.session.as_ref())
                    .active_settings()?,
            )
        } else {
            None
        };
        if let Some(outcome) = execute_adapter_tool(
            executor,
            args,
            &engine.guard,
            &engine.operation_guard,
            engine.progress.clone(),
            adapter_settings.as_ref().map(|settings| &settings.storage),
        )? {
            if let Some(operation) = outcome.file_operation {
                Self::record_file_operation(engine, &operation, None)?;
            }
            return Ok(outcome.outcome);
        }
        if matches!(
            executor,
            ToolExecutor::TranslateFile
                | ToolExecutor::TranslateSeries
                | ToolExecutor::EditSubtitle
        ) {
            let settings = ProfileCoordinator::new(&engine.project_root, engine.session.as_ref())
                .active_settings()?;
            let outcome = execute_translation_tool(
                executor,
                args,
                &engine.guard,
                &engine.operation_guard,
                engine.progress.clone(),
                settings,
            )?
            .ok_or_else(|| AgentError::InvalidState {
                message: "translation tool executor rejected its tool".to_owned(),
            })?;
            let group_id = outcome
                .group_file_operations
                .then(|| format!("translate-series-{}", crate::session::iso_now()));
            for operation in &outcome.file_operations {
                Self::record_file_operation(engine, operation, group_id.clone())?;
            }
            return Ok(outcome.outcome);
        }
        if matches!(
            executor,
            ToolExecutor::RecentTranslations
                | ToolExecutor::ListProfiles
                | ToolExecutor::SwitchProfile
        ) {
            let config = if matches!(executor, ToolExecutor::RecentTranslations) {
                None
            } else {
                ProfileCoordinator::new(&engine.project_root, engine.session.as_ref())
                    .load_config()?
            };
            let events = engine
                .session
                .as_ref()
                .map(|session| session.events.as_slice())
                .unwrap_or(&[]);
            let active_profile = engine
                .session
                .as_ref()
                .and_then(|session| session.profile.as_deref());
            let outcome = execute_session_tool(executor, args, events, config, active_profile)?
                .ok_or_else(|| AgentError::InvalidState {
                    message: "session tool executor rejected its tool".to_owned(),
                })?;
            if let Some(profile_switch) = outcome.profile_switch {
                let session = engine
                    .session
                    .as_mut()
                    .ok_or_else(|| AgentError::invalid_state("no active session"))?;
                session.profile = Some(profile_switch.name.clone());
                session.config_path =
                    Some(profile_switch.config_path.to_string_lossy().to_string());
                engine.save()?;
                engine.record_if_active(EventKind::Profile {
                    name: profile_switch.name,
                })?;
            }
            return Ok(outcome.outcome);
        }
        Err(AgentError::InvalidState {
            message: "tool was not handled by its registered executor".to_owned(),
        })
    }

    fn record_file_operation(
        engine: &mut AgentEngine,
        result: &FileOpResult,
        group_id: Option<String>,
    ) -> AgentResult<()> {
        engine.record_if_active(EventKind::FileOperation(FileOpEventData {
            action: action_label(result.action).to_owned(),
            path: event_path(&engine.project_root, &result.path),
            new_path: result
                .new_path
                .as_ref()
                .map(|path| event_path(&engine.project_root, path)),
            backup_path: result
                .backup_path
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
            group_id,
            undone: false,
        }))
    }
}

fn action_label(action: FileOpAction) -> &'static str {
    match action {
        FileOpAction::Create => "created",
        FileOpAction::Append => "appended",
        FileOpAction::Modified => "modified",
        FileOpAction::Renamed => "renamed",
        FileOpAction::Deleted => "deleted",
    }
}

fn event_path(root: &Path, path: &Path) -> String {
    let canonical_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    path.strip_prefix(&canonical_root)
        .or_else(|_| path.strip_prefix(root))
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}
