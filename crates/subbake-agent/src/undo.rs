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

        let guard = FileGuard::new(project_root.to_path_buf());
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

    fn setup(label: &str) -> (PathBuf, AgentSessionStore, AgentSession) {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("subbake-undo-{label}-{nonce}"));
        std::fs::create_dir_all(&root).expect("create root");
        let store = AgentSessionStore::new(root.clone());
        let session = AgentSession::new(format!("undo-{label}-{nonce}"));
        (root, store, session)
    }

    fn operation(data: FileOpEventData) -> AgentEvent {
        AgentEvent::from_kind(&EventKind::FileOperation(data))
    }

    #[test]
    fn created_file_undo_removes_the_file_and_persists_progress() {
        let (root, store, mut session) = setup("created");
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
        let (root, store, mut session) = setup("missing-backup");
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
        let (root, store, mut session) = setup("renamed-missing-backup");
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
        let (root, store, mut session) = setup("created-directory");
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

        let (root, store, mut session) = setup("symlink");
        let outside = std::env::temp_dir().join(format!("subbake-undo-outside-{}", session.id));
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
