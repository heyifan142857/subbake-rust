//! Safe file operations with sandbox enforcement and automatic backups.
//!
//! Python equivalent: `agent/file_ops.py` + backup logic from `agent/executor.py`.
//!
//! Key improvements over Python:
//! - Single `FileGuard` struct instead of two separate backup paths
//! - Atomic write via rename (Python uses separate write-then-verify)
//! - `PathBuf` returns actions so callers can log events without re-parsing

use std::path::{Path, PathBuf};

/// Path components that are never allowed in file operations.
pub const PROTECTED_PATH_PARTS: [&str; 7] = [
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    ".subbake",
    "__pycache__",
];

/// The result of a successful file operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileOpResult {
    pub action: FileOpAction,
    pub path: PathBuf,
    pub backup_path: Option<PathBuf>,
    pub new_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileOpAction {
    Create,
    Append,
    Modified,
    Renamed,
    Deleted,
}

/// Safe file operations within a project root.
///
/// Every mutating operation:
/// 1. Resolves paths relative to the project root
/// 2. Rejects paths containing protected components
/// 3. Creates a timestamped backup before overwriting
#[derive(Debug, Clone)]
pub struct FileGuard {
    project_root: PathBuf,
    backup_root: PathBuf,
}

impl FileGuard {
    pub fn new(project_root: PathBuf) -> Self {
        let backup_root = project_root.join(".subbake/agent/backups");
        Self {
            project_root,
            backup_root,
        }
    }

    // ------------------------------------------------------------------
    // Public operations
    // ------------------------------------------------------------------

    pub fn read_file(&self, path: &Path) -> std::io::Result<String> {
        let safe = self.resolve(path)?;
        std::fs::read_to_string(&safe)
            .map_err(|e| std::io::Error::other(format!("read {}: {e}", safe.display())))
    }

    pub fn create_file(&self, path: &Path, content: &str) -> std::io::Result<FileOpResult> {
        let safe = self.resolve(path)?;
        if safe.exists() {
            return Err(std::io::Error::other(format!("file already exists: {}", safe.display())));
        }
        self.write_atomically(&safe, content)?;
        Ok(FileOpResult {
            action: FileOpAction::Create,
            path: safe,
            backup_path: None,
            new_path: None,
        })
    }

    pub fn append_file(&self, path: &Path, content: &str) -> std::io::Result<FileOpResult> {
        let safe = self.resolve(path)?;
        let backup = self.backup(&safe)?;
        let mut existing = if safe.exists() {
            std::fs::read_to_string(&safe)?
        } else {
            String::new()
        };
        existing.push_str(content);
        self.write_atomically(&safe, &existing)?;
        Ok(FileOpResult {
            action: FileOpAction::Append,
            path: safe,
            backup_path: Some(backup),
            new_path: None,
        })
    }

    pub fn replace_in_file(&self, path: &Path, old: &str, new: &str) -> std::io::Result<FileOpResult> {
        let safe = self.resolve(path)?;
        let backup = self.backup(&safe)?;
        let content = std::fs::read_to_string(&safe)?;
        let updated = content.replace(old, new);
        self.write_atomically(&safe, &updated)?;
        Ok(FileOpResult {
            action: FileOpAction::Modified,
            path: safe,
            backup_path: Some(backup),
            new_path: None,
        })
    }

    pub fn rename_path(&self, from: &Path, to: &Path) -> std::io::Result<FileOpResult> {
        let safe_from = self.resolve(from)?;
        let safe_to = self.resolve(to)?;
        // Backup both: the source (will be gone) and the destination (will be overwritten).
        let backup = self.backup(&safe_from)?;
        if safe_to.exists() {
            let _ = self.backup(&safe_to)?;
        }
        if let Some(parent) = safe_to.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::rename(&safe_from, &safe_to)?;
        Ok(FileOpResult {
            action: FileOpAction::Renamed,
            path: safe_from,
            backup_path: Some(backup),
            new_path: Some(safe_to),
        })
    }

    pub fn delete_file(&self, path: &Path) -> std::io::Result<FileOpResult> {
        let safe = self.resolve(path)?;
        let backup = self.backup(&safe)?;
        if safe.is_dir() {
            std::fs::remove_dir_all(&safe)?;
        } else {
            std::fs::remove_file(&safe)?;
        }
        Ok(FileOpResult {
            action: FileOpAction::Deleted,
            path: safe,
            backup_path: Some(backup),
            new_path: None,
        })
    }

    pub fn list_files(&self, dir: &Path) -> std::io::Result<Vec<PathBuf>> {
        let safe = self.resolve(dir)?;
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&safe)? {
            let entry = entry?;
            files.push(entry.path());
        }
        files.sort();
        Ok(files)
    }

    /// Search for files matching a glob-like name pattern under a directory.
    pub fn search_files(&self, dir: &Path, pattern: &str) -> std::io::Result<Vec<PathBuf>> {
        let safe = self.resolve(dir)?;
        let mut results = Vec::new();
        self.search_recursive(&safe, pattern, &mut results)?;
        results.sort();
        Ok(results)
    }

    // ------------------------------------------------------------------
    // Path resolution + sandbox
    // ------------------------------------------------------------------

    /// Resolve a user-supplied path to an absolute path under the project root,
    /// rejecting paths that escape the project root or contain protected components.
    fn resolve(&self, user_path: &Path) -> std::io::Result<PathBuf> {
        // Normalise `..` components so `root/../etc/passwd` is caught below.
        let anchored = normalize_path(
            if user_path.is_absolute() {
                user_path.to_path_buf()
            } else {
                self.project_root.join(user_path)
            },
        );

        // ── Escape guard: anchor must be under project_root ──
        let root_canon = self.project_root.canonicalize().unwrap_or_else(|_| self.project_root.clone());
        if !anchored.starts_with(&root_canon) {
            return Err(std::io::Error::other(format!(
                "path escapes project root `{}`: {}",
                root_canon.display(),
                anchored.display(),
            )));
        }

        // Canonicalize existing paths; for new files canonicalize the parent.
        let safe = if anchored.exists() {
            anchored.canonicalize().map_err(|e| {
                std::io::Error::other(format!("resolve existing path: {e}"))
            })?
        } else if let Some(parent) = anchored.parent() {
            if parent.exists() {
                let canonical_parent = parent.canonicalize().map_err(|e| {
                    std::io::Error::other(format!("resolve parent: {e}"))
                })?;
                canonical_parent.join(anchored.file_name().unwrap_or_default())
            } else {
                // Parent doesn't exist yet — trust the escape check above.
                anchored
            }
        } else {
            anchored
        };

        // Check for protected components (e.g. .git, .subbake).
        for component in safe.components() {
            if let Some(name) = component.as_os_str().to_str()
                && PROTECTED_PATH_PARTS.contains(&name) {
                    return Err(std::io::Error::other(format!(
                        "path contains protected component `{name}`: {}",
                        safe.display()
                    )));
                }
        }

        Ok(safe)
    }

    fn search_recursive(&self, dir: &Path, pattern: &str, results: &mut Vec<PathBuf>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                self.search_recursive(&path, pattern, results)?;
            } else if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|name| name.contains(pattern))
            {
                results.push(path);
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Backups
    // ------------------------------------------------------------------

    /// Create a timestamped backup of a file before mutating it.
    fn backup(&self, path: &Path) -> std::io::Result<PathBuf> {
        if !path.exists() {
            return Err(std::io::Error::other(format!(
                "cannot back up non-existent file: {}",
                path.display()
            )));
        }

        let rel = path
            .strip_prefix(&self.project_root)
            .unwrap_or(path);
        let ts = nanos_since_epoch();
        let backup_path = self.backup_root.join(format!("{ts}")).join(rel);

        if let Some(parent) = backup_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(path, &backup_path)?;
        Ok(backup_path)
    }

    /// Write content to a file atomically via temp + rename.
    fn write_atomically(&self, path: &Path, content: &str) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_file_name(format!(
            ".{}.tmp",
            path.file_name().and_then(|s| s.to_str()).unwrap_or("file")
        ));
        std::fs::write(&tmp, content)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Restore a file from a backup. Used by undo.
    pub fn restore_backup(backup_path: &Path, target: &Path) -> std::io::Result<()> {
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(backup_path, target)?;
        Ok(())
    }
}

fn nanos_since_epoch() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos()
}

/// Remove `..` and `.` components from a path without touching the filesystem.
/// Mirrors `std::fs::canonicalize` but works for non-existent paths.
fn normalize_path(path: PathBuf) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                components.pop();
            }
            std::path::Component::CurDir => {
                // skip
            }
            other => components.push(other),
        }
    }
    components.iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (PathBuf, FileGuard) {
        let ts = nanos_since_epoch();
        let root = std::env::temp_dir().join(format!("subbake-guard-{ts}"));
        let guard = FileGuard::new(root.clone());
        (root, guard)
    }

    #[test]
    fn creates_file() {
        let (root, guard) = setup();
        let path = Path::new("test.txt");
        let result = guard.create_file(path, "hello").expect("create");
        assert_eq!(result.action, FileOpAction::Create);
        assert_eq!(guard.read_file(path).expect("read"), "hello");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_path_traversal_via_dotdot() {
        let (root, guard) = setup();
        let err = guard
            .create_file(Path::new("../etc/passwd"), "data")
            .expect_err("path traversal should be rejected");
        assert!(err.to_string().contains("escapes project root"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_protected_path() {
        let (root, guard) = setup();
        let err = guard.create_file(Path::new(".git/config"), "data").expect_err("should reject");
        assert!(err.to_string().contains(".git"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn append_and_backup() {
        let (root, guard) = setup();
        let path = Path::new("log.txt");
        guard.create_file(path, "line1\n").expect("create");
        let result = guard.append_file(path, "line2\n").expect("append");
        assert_eq!(result.action, FileOpAction::Append);
        assert!(result.backup_path.is_some());
        assert_eq!(guard.read_file(path).expect("read"), "line1\nline2\n");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn delete_and_restore() {
        let (root, guard) = setup();
        let path = Path::new("del.txt");
        guard.create_file(path, "data").expect("create");
        let result = guard.delete_file(path).expect("delete");
        assert!(!root.join(path).exists());
        // Restore from backup
        let backup = result.backup_path.expect("backup");
        FileGuard::restore_backup(&backup, &root.join(path)).expect("restore");
        assert_eq!(guard.read_file(path).expect("read"), "data");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rename_moves_file() {
        let (root, guard) = setup();
        guard.create_file(Path::new("a.txt"), "data").expect("create");
        let result = guard.rename_path(Path::new("a.txt"), Path::new("b.txt")).expect("rename");
        assert_eq!(result.action, FileOpAction::Renamed);
        assert!(root.join("b.txt").exists());
        assert!(!root.join("a.txt").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_absolute_path_outside_root() {
        let (root, guard) = setup();
        // Even though /tmp exists, the guard's project_root is a subdir,
        // so an absolute path pointing outside should be rejected.
        let err = guard
            .create_file(Path::new("/tmp/outside-root.txt"), "data")
            .expect_err("should reject path outside project root");
        let msg = err.to_string();
        assert!(msg.contains("escapes project root"), "{msg}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_nonexistent_fails() {
        let (root, guard) = setup();
        let err = guard.read_file(Path::new("missing.txt")).expect_err("should fail");
        assert!(err.to_string().contains("missing.txt"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
