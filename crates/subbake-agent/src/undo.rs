use std::path::{Path, PathBuf};

use crate::error::AgentResult;
use crate::event::{EventKind, FileOpEventData};
use crate::guard::{FileGuard, SemanticUndo};
use crate::session::{AgentSession, AgentSessionStore, EventTag};
use subbake_core::CancellationGuard;

pub(crate) struct UndoService;

impl UndoService {
    pub(crate) fn undo_last(
        project_root: &Path,
        store: &AgentSessionStore,
        session: &mut AgentSession,
        cancellation: &CancellationGuard,
    ) -> AgentResult<usize> {
        let target_index = session
            .events
            .iter()
            .rposition(|event| {
                event.tag() == EventTag::FileOperation
                    && !event
                        .data
                        .get("undone")
                        .and_then(|value| value.as_bool())
                        .unwrap_or(false)
            })
            .ok_or_else(|| std::io::Error::other("nothing to undo"))?;
        let target = file_operation(&session.events[target_index])?;
        let indices = if let Some(group_id) = target.group_id.as_deref() {
            let mut indices = Vec::new();
            for (index, event) in session.events.iter().enumerate() {
                if event.tag() != EventTag::FileOperation
                    || event.data.get("group_id").and_then(|value| value.as_str()) != Some(group_id)
                {
                    continue;
                }
                let data = file_operation(event)?;
                if !data.undone {
                    indices.push(index);
                }
            }
            indices
        } else {
            vec![target_index]
        };

        let mut guard = FileGuard::new(project_root.to_path_buf())?;
        if let Some(path) = session.config_path.as_deref() {
            guard.protect_path(Path::new(path))?;
        }
        for index in indices.iter().rev().copied() {
            let data = file_operation(&session.events[index])?;
            restore_event(&guard, &data, cancellation)?;
            if let Some(data) = session.events[index].data.as_object_mut() {
                data.insert("undone".to_owned(), serde_json::Value::Bool(true));
            }
            store.save(session)?;
        }
        Ok(indices.len())
    }
}

fn file_operation(event: &crate::session::AgentEvent) -> AgentResult<FileOpEventData> {
    match event.typed() {
        Some(EventKind::FileOperation(data)) => Ok(data),
        _ => Err(std::io::Error::other("invalid persisted file operation event").into()),
    }
}

fn restore_event(
    guard: &FileGuard,
    event: &FileOpEventData,
    cancellation: &CancellationGuard,
) -> AgentResult<()> {
    cancellation.check()?;
    let target_path = guard.resolve_undo_target(Path::new(&event.path))?;

    if let Some(semantic) = event.semantic_undo.clone() {
        match semantic {
            SemanticUndo::RestoreRenamed {
                destination_backup_path,
            } => {
                let new_path = event
                    .new_path
                    .as_deref()
                    .ok_or_else(|| std::io::Error::other("renamed undo is missing new_path"))?;
                let new_path = guard.resolve_undo_target(Path::new(new_path))?;
                if new_path == target_path {
                    return Err(std::io::Error::other("rename undo targets must differ").into());
                }
                let source_backup = guard.resolve_undo_backup(&required_backup(event)?)?;
                let destination_backup = destination_backup_path
                    .as_deref()
                    .map(|path| guard.resolve_undo_backup(path))
                    .transpose()?;
                // Validate both backups and targets before restoring either side.
                guard.restore_for_undo(&source_backup, &target_path)?;
                if let Some(backup) = destination_backup {
                    guard.restore_for_undo(&backup, &new_path)?;
                } else {
                    guard.remove_for_undo(&new_path)?;
                }
                return Ok(());
            }
            SemanticUndo::RemoveEmbeddedSubtitle { title } => {
                subbake_adapters::remove_embedded_subtitle_by_title(
                    &target_path,
                    &title,
                    cancellation,
                )?;
                return Ok(());
            }
            SemanticUndo::RestoreEmbeddedSubtitle {
                title,
                subtitle_backup_path,
                subtitle_format,
            } => {
                let subtitle_format = subbake_adapters::SubtitlePayloadFormat::parse(
                    subtitle_format.as_deref().unwrap_or("srt"),
                )?;
                let subtitle_backup_path = guard.resolve_undo_backup(&subtitle_backup_path)?;
                subbake_adapters::restore_embedded_subtitle(
                    &target_path,
                    &title,
                    &subtitle_backup_path,
                    subtitle_format,
                    cancellation,
                )?;
                return Ok(());
            }
        }
    }

    match event.action.as_str() {
        "created" => guard.remove_for_undo(Path::new(&event.path))?,
        "renamed" => {
            let new_path = event
                .new_path
                .as_deref()
                .ok_or_else(|| std::io::Error::other("renamed undo is missing new_path"))?;
            if guard.resolve_undo_target(Path::new(new_path))? == target_path {
                return Err(std::io::Error::other("rename undo targets must differ").into());
            }
            let backup = required_backup(event)?;
            guard.restore_for_undo(&backup, Path::new(&event.path))?;
            // Restore first. If removing the renamed path then fails, retrying
            // is safe and the user's only surviving copy was never destroyed.
            guard.remove_for_undo(Path::new(new_path))?;
        }
        "deleted" | "modified" | "appended" => {
            let backup = required_backup(event)?;
            guard.restore_for_undo(&backup, Path::new(&event.path))?;
        }
        action => {
            return Err(std::io::Error::other(format!("unknown undo action `{action}`")).into());
        }
    }
    Ok(())
}

fn required_backup(event: &FileOpEventData) -> AgentResult<PathBuf> {
    event
        .backup_path
        .as_deref()
        .map(PathBuf::from)
        .ok_or_else(|| {
            std::io::Error::other(format!("{} undo is missing backup_path", event.action)).into()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{AgentEvent, AgentSession};

    fn setup(label: &str) -> (tempfile::TempDir, PathBuf, AgentSessionStore, AgentSession) {
        let temporary = tempfile::Builder::new()
            .prefix(&format!("subbake-undo-{label}-"))
            .tempdir()
            .expect("create temporary undo root");
        let root = temporary.path().to_path_buf();
        let store = AgentSessionStore::new(root.clone());
        let session = AgentSession::new(format!(
            "undo-{label}-{}",
            root.file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("temporary")
        ));
        (temporary, root, store, session)
    }

    fn operation(data: FileOpEventData) -> AgentEvent {
        AgentEvent::from_kind(&EventKind::FileOperation(data))
    }

    #[test]
    fn overwrite_rename_undo_restores_both_files_after_session_reload() {
        let (_temporary, root, _store, _) = setup("overwrite-rename");
        std::fs::write(root.join("a.txt"), "original A").expect("source");
        std::fs::write(root.join("b.txt"), "original B").expect("destination");
        let mut engine = crate::AgentEngine::new(root.clone()).expect("engine");
        engine.start_session().expect("session");
        engine
            .run_tool(
                "rename_path",
                &serde_json::json!({"from":"a.txt","to":"b.txt"}),
            )
            .expect("rename");
        let mut resumed = crate::AgentEngine::new(root.clone()).expect("engine");
        resumed.resume_session(None).expect("resume");
        resumed.undo_last().expect("undo");
        assert_eq!(
            std::fs::read_to_string(root.join("a.txt")).expect("source"),
            "original A"
        );
        assert_eq!(
            std::fs::read_to_string(root.join("b.txt")).expect("destination"),
            "original B"
        );
    }

    #[test]
    fn self_rename_fails_before_mutation_or_backup() {
        let (_temporary, root, _store, _) = setup("self-rename");
        std::fs::write(root.join("a.txt"), "only copy").expect("source");
        let guard = FileGuard::new(root.clone()).expect("guard");
        assert!(
            guard
                .rename_path(Path::new("a.txt"), Path::new("./a.txt"))
                .is_err()
        );
        assert_eq!(
            std::fs::read_to_string(root.join("a.txt")).expect("source"),
            "only copy"
        );
        assert!(!root.join(".subbake/agent/backups").exists());
    }

    #[test]
    fn created_file_undo_removes_the_file_and_persists_progress() {
        let (_temporary, root, store, mut session) = setup("created");
        std::fs::write(root.join("created.txt"), "content").expect("write created file");
        session.events.push(operation(FileOpEventData {
            action: "created".to_owned(),
            path: "created.txt".to_owned(),
            new_path: None,
            backup_path: None,
            semantic_undo: None,
            group_id: None,
            undone: false,
        }));

        let count =
            UndoService::undo_last(&root, &store, &mut session, &CancellationGuard::never())
                .expect("undo created file");

        assert_eq!(count, 1);
        assert!(!root.join("created.txt").exists());
        let persisted = store.load(&session.id).expect("load persisted session");
        assert!(
            persisted.events[0].data["undone"]
                .as_bool()
                .unwrap_or(false)
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn modified_file_undo_rejects_a_missing_backup() {
        let (_temporary, root, store, mut session) = setup("missing-backup");
        std::fs::write(root.join("modified.txt"), "new").expect("write modified file");
        session.events.push(operation(FileOpEventData {
            action: "modified".to_owned(),
            path: "modified.txt".to_owned(),
            new_path: None,
            backup_path: None,
            semantic_undo: None,
            group_id: None,
            undone: false,
        }));

        let error =
            UndoService::undo_last(&root, &store, &mut session, &CancellationGuard::never())
                .expect_err("missing backup must fail");

        assert!(error.to_string().contains("missing backup_path"), "{error}");
        assert_eq!(
            std::fs::read_to_string(root.join("modified.txt")).expect("read unchanged file"),
            "new"
        );
        assert!(!session.events[0].data["undone"].as_bool().unwrap_or(false));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn renamed_file_undo_validates_backup_before_removing_the_new_path() {
        let (_temporary, root, store, mut session) = setup("renamed-missing-backup");
        std::fs::write(root.join("new-name.txt"), "only copy").expect("write renamed file");
        session.events.push(operation(FileOpEventData {
            action: "renamed".to_owned(),
            path: "old-name.txt".to_owned(),
            new_path: Some("new-name.txt".to_owned()),
            backup_path: Some(
                root.join(".subbake/agent/backups/missing.txt")
                    .to_string_lossy()
                    .into_owned(),
            ),
            semantic_undo: None,
            group_id: None,
            undone: false,
        }));

        UndoService::undo_last(&root, &store, &mut session, &CancellationGuard::never())
            .expect_err("missing rename backup must fail before removal");

        assert_eq!(
            std::fs::read_to_string(root.join("new-name.txt")).expect("renamed file remains"),
            "only copy"
        );
        assert!(!root.join("old-name.txt").exists());
        assert!(!session.events[0].data["undone"].as_bool().unwrap_or(false));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn created_file_undo_never_recursively_removes_a_directory() {
        let (_temporary, root, store, mut session) = setup("created-directory");
        std::fs::create_dir_all(root.join("important")).expect("create important directory");
        std::fs::write(root.join("important/data.txt"), "keep").expect("write important file");
        session.events.push(operation(FileOpEventData {
            action: "created".to_owned(),
            path: "important".to_owned(),
            new_path: None,
            backup_path: None,
            semantic_undo: None,
            group_id: None,
            undone: false,
        }));

        let error =
            UndoService::undo_last(&root, &store, &mut session, &CancellationGuard::never())
                .expect_err("directory removal must be rejected");

        assert!(error.to_string().contains("regular file"), "{error}");
        assert_eq!(
            std::fs::read_to_string(root.join("important/data.txt")).expect("important file"),
            "keep"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn undo_rejects_a_leaf_symlink_redirect() {
        use std::os::unix::fs::symlink;

        let (_temporary, root, store, mut session) = setup("symlink");
        let outside_root = tempfile::Builder::new()
            .prefix("subbake-undo-outside-")
            .tempdir()
            .expect("create outside temporary root");
        let outside = outside_root.path().join("outside.txt");
        std::fs::write(&outside, "keep").expect("write outside file");
        symlink(&outside, root.join("created.txt")).expect("create redirecting symlink");
        session.events.push(operation(FileOpEventData {
            action: "created".to_owned(),
            path: "created.txt".to_owned(),
            new_path: None,
            backup_path: None,
            semantic_undo: None,
            group_id: None,
            undone: false,
        }));

        let error =
            UndoService::undo_last(&root, &store, &mut session, &CancellationGuard::never())
                .expect_err("redirecting symlink must fail");

        assert!(error.to_string().contains("symbolic links"), "{error}");
        assert_eq!(
            std::fs::read_to_string(&outside).expect("outside remains"),
            "keep"
        );
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_file(outside);
    }
}
