use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

use crate::error::{AgentError, AgentResult};
use crate::guard::{FileGuard, FileOpAction, FileOpResult};

#[derive(Debug, Clone, PartialEq, Eq)]
enum PatchOperation {
    Add {
        path: PathBuf,
        content: String,
    },
    Update {
        path: PathBuf,
        hunks: Vec<UpdateHunk>,
    },
    Delete {
        path: PathBuf,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UpdateHunk {
    old: String,
    new: String,
}

#[derive(Debug)]
enum PreparedOperation {
    Add { path: PathBuf, content: String },
    Update { path: PathBuf, content: String },
    Delete { path: PathBuf },
}

#[derive(Debug)]
pub(crate) struct PatchOutcome {
    pub(crate) text: String,
    pub(crate) file_operations: Vec<FileOpResult>,
}

pub(crate) fn apply_patch(patch: &str, guard: &FileGuard) -> AgentResult<PatchOutcome> {
    let parsed = parse_patch(patch)?;
    let prepared = prepare_operations(parsed, guard)?;
    let mut completed = Vec::new();

    for operation in prepared {
        let result = match operation {
            PreparedOperation::Add { path, content } => guard.create_file(&path, &content),
            PreparedOperation::Update { path, content } => guard.replace_file(&path, &content),
            PreparedOperation::Delete { path } => guard.delete_file(&path),
        };
        match result {
            Ok(result) => completed.push(result),
            Err(error) => {
                let rollback_errors = rollback(&completed);
                let suffix = if rollback_errors.is_empty() {
                    String::new()
                } else {
                    format!("; rollback also failed: {}", rollback_errors.join("; "))
                };
                return Err(AgentError::InvalidInput {
                    message: format!("patch application failed: {error}{suffix}"),
                });
            }
        }
    }

    let changed = completed
        .iter()
        .map(|operation| {
            let action = match operation.action {
                FileOpAction::Create => "added",
                FileOpAction::Modified => "updated",
                FileOpAction::Deleted => "deleted",
                FileOpAction::Append | FileOpAction::Renamed => "changed",
            };
            format!("{action} {}", operation.path.display())
        })
        .collect::<Vec<_>>();
    Ok(PatchOutcome {
        text: format!("Applied patch:\n{}", changed.join("\n")),
        file_operations: completed,
    })
}

fn parse_patch(patch: &str) -> AgentResult<Vec<PatchOperation>> {
    let lines = patch.lines().collect::<Vec<_>>();
    if lines.first().copied() != Some("*** Begin Patch") {
        return patch_error("patch must start with `*** Begin Patch`");
    }
    if lines.last().copied() != Some("*** End Patch") {
        return patch_error("patch must end with `*** End Patch`");
    }

    let mut operations = Vec::new();
    let mut seen_paths = HashSet::new();
    let mut index = 1;
    while index + 1 < lines.len() {
        let header = lines[index];
        let (kind, raw_path) = if let Some(path) = header.strip_prefix("*** Add File: ") {
            ("add", path)
        } else if let Some(path) = header.strip_prefix("*** Update File: ") {
            ("update", path)
        } else if let Some(path) = header.strip_prefix("*** Delete File: ") {
            ("delete", path)
        } else {
            return patch_error(format!("expected a file header, found `{header}`"));
        };
        let path = validate_relative_path(raw_path)?;
        if !seen_paths.insert(path.clone()) {
            return patch_error(format!(
                "patch contains more than one operation for `{}`",
                path.display()
            ));
        }
        index += 1;
        let body_start = index;
        while index + 1 < lines.len() && !is_file_header(lines[index]) {
            index += 1;
        }
        let body = &lines[body_start..index];
        let operation = match kind {
            "add" => PatchOperation::Add {
                path,
                content: parse_add_body(body)?,
            },
            "update" => PatchOperation::Update {
                path,
                hunks: parse_update_body(body)?,
            },
            "delete" => {
                if !body.is_empty() {
                    return patch_error("delete file sections cannot contain body lines");
                }
                PatchOperation::Delete { path }
            }
            _ => unreachable!("matched patch operation kind"),
        };
        operations.push(operation);
    }
    if operations.is_empty() {
        return patch_error("patch must contain at least one file operation");
    }
    Ok(operations)
}

fn is_file_header(line: &str) -> bool {
    line.starts_with("*** Add File: ")
        || line.starts_with("*** Update File: ")
        || line.starts_with("*** Delete File: ")
}

fn validate_relative_path(value: &str) -> AgentResult<PathBuf> {
    let path = PathBuf::from(value.trim());
    if value.trim().is_empty() || path.is_absolute() {
        return patch_error("patch paths must be non-empty relative paths");
    }
    if path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return patch_error(format!(
            "patch path must stay inside the project: `{}`",
            path.display()
        ));
    }
    if !path
        .components()
        .any(|component| matches!(component, Component::Normal(_)))
    {
        return patch_error("patch path must name a file");
    }
    Ok(path)
}

fn parse_add_body(lines: &[&str]) -> AgentResult<String> {
    let mut content = Vec::new();
    for line in lines {
        let Some(line) = line.strip_prefix('+') else {
            return patch_error("every add-file content line must start with `+`");
        };
        content.push(line);
    }
    Ok(render_lines(&content))
}

fn parse_update_body(lines: &[&str]) -> AgentResult<Vec<UpdateHunk>> {
    let mut hunks = Vec::new();
    let mut current = Vec::new();
    for line in lines {
        if line.starts_with("@@") {
            if !current.is_empty() {
                hunks.push(parse_hunk(&current)?);
                current.clear();
            }
            continue;
        }
        current.push(*line);
    }
    if !current.is_empty() {
        hunks.push(parse_hunk(&current)?);
    }
    if hunks.is_empty() {
        return patch_error("update file sections require at least one exact change hunk");
    }
    Ok(hunks)
}

fn parse_hunk(lines: &[&str]) -> AgentResult<UpdateHunk> {
    let mut old = Vec::new();
    let mut new = Vec::new();
    let mut changed = false;
    for line in lines {
        let Some((marker, content)) = line.chars().next().map(|marker| {
            let offset = marker.len_utf8();
            (marker, &line[offset..])
        }) else {
            return patch_error("update hunk lines must start with ` `, `-`, or `+`");
        };
        match marker {
            ' ' => {
                old.push(content);
                new.push(content);
            }
            '-' => {
                old.push(content);
                changed = true;
            }
            '+' => {
                new.push(content);
                changed = true;
            }
            _ => return patch_error("update hunk lines must start with ` `, `-`, or `+`"),
        }
    }
    if !changed {
        return patch_error("update hunk does not contain a `-` or `+` change");
    }
    if old.is_empty() {
        return patch_error("update hunks must include exact existing text to match");
    }
    Ok(UpdateHunk {
        old: render_lines(&old),
        new: render_lines(&new),
    })
}

fn render_lines(lines: &[&str]) -> String {
    if lines.is_empty() {
        String::new()
    } else {
        format!("{}\n", lines.join("\n"))
    }
}

fn prepare_operations(
    operations: Vec<PatchOperation>,
    guard: &FileGuard,
) -> AgentResult<Vec<PreparedOperation>> {
    operations
        .into_iter()
        .map(|operation| match operation {
            PatchOperation::Add { path, content } => {
                let resolved = guard.resolve_path(&path)?;
                if resolved.exists() {
                    return patch_error(format!("add target already exists: {}", path.display()));
                }
                Ok(PreparedOperation::Add { path, content })
            }
            PatchOperation::Update { path, hunks } => {
                let mut content = guard.read_file(&path)?;
                for hunk in hunks {
                    let matches = content.match_indices(&hunk.old).count();
                    if matches == 0 {
                        return patch_error(format!(
                            "update text was not found in `{}`",
                            path.display()
                        ));
                    }
                    if matches > 1 {
                        return patch_error(format!(
                            "update text is ambiguous in `{}` ({matches} matches)",
                            path.display()
                        ));
                    }
                    content = content.replacen(&hunk.old, &hunk.new, 1);
                }
                Ok(PreparedOperation::Update { path, content })
            }
            PatchOperation::Delete { path } => {
                let resolved = guard.resolve_path(&path)?;
                if !resolved.is_file() {
                    return patch_error(format!(
                        "delete target is not an existing file: {}",
                        path.display()
                    ));
                }
                Ok(PreparedOperation::Delete { path })
            }
        })
        .collect()
}

fn rollback(completed: &[FileOpResult]) -> Vec<String> {
    let mut errors = Vec::new();
    for operation in completed.iter().rev() {
        let result = match operation.action {
            FileOpAction::Create => remove_created(&operation.path),
            FileOpAction::Modified | FileOpAction::Deleted | FileOpAction::Append => operation
                .backup_path
                .as_deref()
                .ok_or_else(|| "missing backup path".to_owned())
                .and_then(|backup| {
                    FileGuard::restore_backup(backup, &operation.path)
                        .map_err(|error| format!("restore {}: {error}", operation.path.display()))
                }),
            FileOpAction::Renamed => Err("patch rollback encountered a rename".to_owned()),
        };
        if let Err(error) = result {
            errors.push(error);
        }
    }
    errors
}

fn remove_created(path: &Path) -> Result<(), String> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("remove {}: {error}", path.display())),
    }
}

fn patch_error<T>(message: impl Into<String>) -> AgentResult<T> {
    Err(AgentError::ToolArguments {
        message: message.into(),
    })
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn applies_add_update_and_delete_after_full_validation() {
        let (root, guard) = setup();
        std::fs::write(root.join("old.txt"), "old\n").expect("write old");
        std::fs::write(root.join("delete.txt"), "gone\n").expect("write delete");
        let outcome = apply_patch(
            "*** Begin Patch\n*** Add File: new.txt\n+new\n*** Update File: old.txt\n-old\n+updated\n*** Delete File: delete.txt\n*** End Patch",
            &guard,
        )
        .expect("apply patch");

        assert_eq!(outcome.file_operations.len(), 3);
        assert_eq!(
            std::fs::read_to_string(root.join("new.txt")).expect("new"),
            "new\n"
        );
        assert_eq!(
            std::fs::read_to_string(root.join("old.txt")).expect("updated"),
            "updated\n"
        );
        assert!(!root.join("delete.txt").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_missing_ambiguous_and_escaping_updates_without_changes() {
        let (root, guard) = setup();
        std::fs::write(root.join("same.txt"), "same\nsame\n").expect("write");

        let ambiguous = apply_patch(
            "*** Begin Patch\n*** Update File: same.txt\n-same\n+other\n*** End Patch",
            &guard,
        )
        .expect_err("ambiguous");
        assert!(ambiguous.to_string().contains("ambiguous"));
        assert_eq!(
            std::fs::read_to_string(root.join("same.txt")).expect("unchanged"),
            "same\nsame\n"
        );

        let escape = apply_patch(
            "*** Begin Patch\n*** Add File: ../escape.txt\n+no\n*** End Patch",
            &guard,
        )
        .expect_err("escape");
        assert!(escape.to_string().contains("inside the project"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn validates_every_file_before_applying_any_file() {
        let (root, guard) = setup();
        let error = apply_patch(
            "*** Begin Patch\n*** Add File: would-exist.txt\n+no\n*** Update File: missing.txt\n-old\n+new\n*** End Patch",
            &guard,
        )
        .expect_err("missing update");
        assert!(error.to_string().contains("missing.txt"));
        assert!(!root.join("would-exist.txt").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn rolls_back_an_earlier_file_when_a_later_write_fails() {
        let (root, guard) = setup();
        let error = apply_patch(
            "*** Begin Patch\n*** Add File: conflict\n+temporary\n*** Add File: conflict/nested.txt\n+cannot be written\n*** End Patch",
            &guard,
        )
        .expect_err("second add must fail after the first add");
        assert!(error.to_string().contains("patch application failed"));
        assert!(!root.join("conflict").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_protected_path_and_symlink_escape() {
        let (root, guard) = setup();
        let protected = apply_patch(
            "*** Begin Patch\n*** Add File: .git/config\n+no\n*** End Patch",
            &guard,
        )
        .expect_err("protected");
        assert!(protected.to_string().contains("protected"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let outside = root.with_extension("outside");
            std::fs::create_dir_all(&outside).expect("outside");
            symlink(&outside, root.join("link")).expect("symlink");
            let escaped = apply_patch(
                "*** Begin Patch\n*** Add File: link/nested/file.txt\n+no\n*** End Patch",
                &guard,
            )
            .expect_err("symlink escape");
            assert!(escaped.to_string().contains("escapes project root"));
            let _ = std::fs::remove_dir_all(outside);
        }
        let _ = std::fs::remove_dir_all(root);
    }

    fn setup() -> (PathBuf, FileGuard) {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("subbake-patch-{nanos}"));
        std::fs::create_dir_all(&root).expect("root");
        (root.clone(), FileGuard::new(root))
    }
}
