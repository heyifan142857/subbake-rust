use std::path::{Path, PathBuf};

use crate::error::AgentResult;
use crate::guard::{FileGuard, SemanticUndo};
use crate::session::{AgentSession, AgentSessionStore, EventTag};

pub(crate) struct UndoService;

impl UndoService {
    pub(crate) fn undo_last(
        project_root: &Path,
        store: &AgentSessionStore,
        session: &mut AgentSession,
    ) -> AgentResult<usize> {
        let events = session.events.clone();
        let target = events
            .iter()
            .rev()
            .find(|event| {
                event.tag() == EventTag::FileOperation
                    && !event
                        .data
                        .get("undone")
                        .and_then(|value| value.as_bool())
                        .unwrap_or(false)
            })
            .cloned()
            .ok_or_else(|| std::io::Error::other("nothing to undo"))?;

        let group_id = target
            .data
            .get("group_id")
            .and_then(|value| value.as_str())
            .map(String::from);
        let targets = if let Some(group_id) = group_id.as_ref() {
            events
                .iter()
                .filter(|event| {
                    event.tag() == EventTag::FileOperation
                        && event.data.get("group_id").and_then(|value| value.as_str())
                            == Some(group_id.as_str())
                        && !event
                            .data
                            .get("undone")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(false)
                })
                .cloned()
                .collect::<Vec<_>>()
        } else {
            vec![target]
        };

        for event in &targets {
            restore_event(project_root, event)?;
            if let Some(stored) = session.events.iter_mut().rev().find(|stored| {
                stored.created_at == event.created_at && stored.tag() == EventTag::FileOperation
            }) && let Some(data) = stored.data.as_object_mut()
            {
                data.insert("undone".to_owned(), serde_json::Value::Bool(true));
            }
        }
        store.save(session)?;
        Ok(targets.len())
    }
}

fn restore_event(project_root: &Path, event: &crate::session::AgentEvent) -> AgentResult<()> {
    let action = event
        .data
        .get("action")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    let path = event
        .data
        .get("path")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    let backup = event
        .data
        .get("backup_path")
        .and_then(|value| value.as_str());
    let target_path = project_root.join(path);

    if let Some(semantic) = event.data.get("semantic_undo")
        && !semantic.is_null()
    {
        let semantic: SemanticUndo = serde_json::from_value(semantic.clone()).map_err(|error| {
            std::io::Error::other(format!("invalid semantic undo data: {error}"))
        })?;
        match semantic {
            SemanticUndo::RemoveEmbeddedSubtitle { title } => {
                subbake_adapters::remove_embedded_subtitle_by_title(
                    &target_path,
                    &title,
                    &subbake_core::CancellationGuard::never(),
                )?;
                return Ok(());
            }
            SemanticUndo::RestoreEmbeddedSubtitle {
                title,
                subtitle_backup_path,
            } => {
                subbake_adapters::restore_embedded_subtitle_from_srt(
                    &target_path,
                    &title,
                    &subtitle_backup_path,
                    &subbake_core::CancellationGuard::never(),
                )?;
                return Ok(());
            }
        }
    }

    match action {
        "created" => {
            let _ = std::fs::remove_file(&target_path);
            let _ = std::fs::remove_dir_all(&target_path);
        }
        "renamed" => {
            if let Some(new_path) = event.data.get("new_path").and_then(|value| value.as_str()) {
                let moved_path = project_root.join(new_path);
                let _ = std::fs::remove_file(&moved_path);
                let _ = std::fs::remove_dir_all(&moved_path);
            }
            if let Some(backup) = backup {
                FileGuard::restore_backup(PathBuf::from(backup).as_path(), &target_path)?;
            }
        }
        "deleted" | "modified" | "appended" => {
            if let Some(backup) = backup {
                FileGuard::restore_backup(PathBuf::from(backup).as_path(), &target_path)?;
            }
        }
        _ => {}
    }
    Ok(())
}
