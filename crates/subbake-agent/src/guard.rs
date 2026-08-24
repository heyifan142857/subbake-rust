//! Safe file operations with sandbox enforcement and automatic backups.
//!
//! Python equivalent: `agent/file_ops.py` + backup logic from `agent/executor.py`.
//!
//! Key improvements over Python:
//! - Single `FileGuard` struct instead of two separate backup paths
//! - Atomic write via rename (Python uses separate write-then-verify)
//! - `PathBuf` returns actions so callers can log events without re-parsing

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use subbake_adapters::PlatformPaths;
use thiserror::Error;

pub type FileGuardResult<T> = Result<T, FileGuardError>;

#[derive(Debug, Error)]
pub enum FileGuardError {
    #[error("path escapes project root `{root}`: {path}")]
    PathEscape { root: PathBuf, path: PathBuf },
    #[error("path contains protected component `{component}`: {path}")]
    ProtectedPath { component: String, path: PathBuf },
    #[error("file already exists: {path}")]
    AlreadyExists { path: PathBuf },
    #[error("cannot back up non-existent file: {path}")]
    MissingBackupSource { path: PathBuf },
    #[error("external deletion requires an absolute path: {path}")]
    ExternalPathMustBeAbsolute { path: PathBuf },
    #[error("use project-local file tools for paths inside `{root}`: {path}")]
    ExternalPathInsideProject { root: PathBuf, path: PathBuf },
    #[error("refusing to delete protected filesystem root `{path}`")]
    CriticalExternalPath { path: PathBuf },
    #[error(
        "external deletion target changed after approval: approved `{approved}`, resolved `{resolved}`"
    )]
    ExternalPathChanged {
        approved: PathBuf,
        resolved: PathBuf,
    },
    #[error("external directory is not empty; set `recursive` to true to delete it: {path}")]
    ExternalDirectoryRequiresRecursive { path: PathBuf },
    #[error("{operation}{path_suffix}: {source}", path_suffix = path.as_ref().map(|value| format!(" `{}`", value.display())).unwrap_or_default())]
    Io {
        operation: &'static str,
        path: Option<PathBuf>,
        #[source]
        source: std::io::Error,
    },
}

impl From<std::io::Error> for FileGuardError {
    fn from(source: std::io::Error) -> Self {
        Self::Io {
            operation: "file operation failed",
            path: None,
            source,
        }
    }
}

/// Path components that are never allowed in file operations.
pub const PROTECTED_PATH_PARTS: [&str; 15] = [
    ".git",
    ".hg",
    ".svn",
    ".env",
    ".ssh",
    ".venv",
    "venv",
    ".subbake",
    "__pycache__",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "credentials.json",
    "service-account.json",
];

/// The result of a successful file operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileOpResult {
    pub action: FileOpAction,
    pub path: PathBuf,
    pub backup_path: Option<PathBuf>,
    pub new_path: Option<PathBuf>,
    pub semantic_undo: Option<SemanticUndo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SemanticUndo {
    RemoveEmbeddedSubtitle {
        title: String,
    },
    RestoreEmbeddedSubtitle {
        title: String,
        subtitle_backup_path: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        subtitle_format: Option<String>,
    },
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

/// Guard for the deliberately exceptional operation that deletes paths outside
/// the active project. It does not create backups or participate in `/undo`;
/// callers must put every invocation behind explicit user approval.
#[derive(Debug, Clone)]
pub(crate) struct ExternalPathGuard {
    project_root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PreparedExternalDelete {
    pub path: PathBuf,
    pub is_directory: bool,
    pub is_symlink: bool,
}

impl ExternalPathGuard {
    pub(crate) fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }

    /// Resolve the exact path shown at the approval boundary. A leaf symlink
    /// is kept as the target so deletion removes the link rather than what it
    /// points to; existing parent components are canonicalized.
    pub(crate) fn prepare(&self, requested_path: &Path) -> FileGuardResult<PreparedExternalDelete> {
        if !requested_path.is_absolute() {
            return Err(FileGuardError::ExternalPathMustBeAbsolute {
                path: requested_path.to_path_buf(),
            });
        }
        let normalized = normalize_path(requested_path.to_path_buf());
        let metadata =
            std::fs::symlink_metadata(&normalized).map_err(|source| FileGuardError::Io {
                operation: "inspect external deletion target",
                path: Some(normalized.clone()),
                source,
            })?;
        let is_symlink = metadata.file_type().is_symlink();
        let resolved = if is_symlink {
            let parent =
                normalized
                    .parent()
                    .ok_or_else(|| FileGuardError::CriticalExternalPath {
                        path: normalized.clone(),
                    })?;
            let canonical_parent = parent.canonicalize().map_err(|source| FileGuardError::Io {
                operation: "resolve external deletion parent",
                path: Some(parent.to_path_buf()),
                source,
            })?;
            canonical_parent.join(normalized.file_name().ok_or_else(|| {
                FileGuardError::CriticalExternalPath {
                    path: normalized.clone(),
                }
            })?)
        } else {
            normalized
                .canonicalize()
                .map_err(|source| FileGuardError::Io {
                    operation: "resolve external deletion target",
                    path: Some(normalized.clone()),
                    source,
                })?
        };

        self.reject_critical_path(&resolved)?;
        Ok(PreparedExternalDelete {
            path: resolved,
            is_directory: metadata.is_dir(),
            is_symlink,
        })
    }

    pub(crate) fn delete(
        &self,
        approved_path: &Path,
        recursive: bool,
    ) -> FileGuardResult<PreparedExternalDelete> {
        let approved = normalize_path(approved_path.to_path_buf());
        let prepared = self.prepare(&approved)?;
        if prepared.path != approved {
            return Err(FileGuardError::ExternalPathChanged {
                approved,
                resolved: prepared.path,
            });
        }

        if prepared.is_symlink || !prepared.is_directory {
            std::fs::remove_file(&prepared.path).map_err(|source| FileGuardError::Io {
                operation: "delete external path",
                path: Some(prepared.path.clone()),
                source,
            })?;
        } else if recursive {
            std::fs::remove_dir_all(&prepared.path).map_err(|source| FileGuardError::Io {
                operation: "recursively delete external directory",
                path: Some(prepared.path.clone()),
                source,
            })?;
        } else if let Err(source) = std::fs::remove_dir(&prepared.path) {
            if source.kind() == std::io::ErrorKind::DirectoryNotEmpty {
                return Err(FileGuardError::ExternalDirectoryRequiresRecursive {
                    path: prepared.path,
                });
            }
            return Err(FileGuardError::Io {
                operation: "delete external directory",
                path: Some(prepared.path),
                source,
            });
        }
        Ok(prepared)
    }

    fn reject_critical_path(&self, path: &Path) -> FileGuardResult<()> {
        let project_root = self
            .project_root
            .canonicalize()
            .unwrap_or_else(|_| normalize_path(self.project_root.clone()));
        if path.starts_with(&project_root) {
            return Err(FileGuardError::ExternalPathInsideProject {
                root: project_root,
                path: path.to_path_buf(),
            });
        }
        if path.parent().is_none() || PlatformPaths::canonical_home_dir().as_deref() == Some(path) {
            return Err(FileGuardError::CriticalExternalPath {
                path: path.to_path_buf(),
            });
        }
        Ok(())
    }
}

impl FileGuard {
    pub fn new(project_root: PathBuf) -> Self {
        let backup_root = project_root.join(".subbake/agent/backups");
        Self {
            project_root,
            backup_root,
        }
    }

    pub(crate) fn project_root(&self) -> &Path {
        &self.project_root
    }

    // ------------------------------------------------------------------
    // Public operations
    // ------------------------------------------------------------------

    pub fn read_file(&self, path: &Path) -> FileGuardResult<String> {
        let safe = self.resolve(path)?;
        std::fs::read_to_string(&safe).map_err(|source| FileGuardError::Io {
            operation: "read file",
            path: Some(safe),
            source,
        })
    }

    pub fn create_file(&self, path: &Path, content: &str) -> FileGuardResult<FileOpResult> {
        let safe = self.resolve(path)?;
        if safe.exists() {
            return Err(FileGuardError::AlreadyExists { path: safe });
        }
        self.write_atomically(&safe, content)?;
        Ok(FileOpResult {
            action: FileOpAction::Create,
            path: safe,
            backup_path: None,
            new_path: None,
            semantic_undo: None,
        })
    }

    pub fn append_file(&self, path: &Path, content: &str) -> FileGuardResult<FileOpResult> {
        let safe = self.resolve(path)?;
        let (mut existing, backup) = if safe.exists() {
            let backup = self.backup(&safe)?;
            (std::fs::read_to_string(&safe)?, Some(backup))
        } else {
            (String::new(), None)
        };
        existing.push_str(content);
        self.write_atomically(&safe, &existing)?;
        Ok(FileOpResult {
            action: if backup.is_some() {
                FileOpAction::Append
            } else {
                FileOpAction::Create
            },
            path: safe,
            backup_path: backup,
            new_path: None,
            semantic_undo: None,
        })
    }

    pub fn replace_in_file(
        &self,
        path: &Path,
        old: &str,
        new: &str,
    ) -> FileGuardResult<FileOpResult> {
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
            semantic_undo: None,
        })
    }

    /// Replace the complete contents of an existing text file.
    pub fn replace_file(&self, path: &Path, content: &str) -> FileGuardResult<FileOpResult> {
        let safe = self.resolve(path)?;
        let backup = self.backup(&safe)?;
        self.write_atomically(&safe, content)?;
        Ok(FileOpResult {
            action: FileOpAction::Modified,
            path: safe,
            backup_path: Some(backup),
            new_path: None,
            semantic_undo: None,
        })
    }

    pub fn rename_path(&self, from: &Path, to: &Path) -> FileGuardResult<FileOpResult> {
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
            semantic_undo: None,
        })
    }

    pub fn delete_file(&self, path: &Path) -> FileGuardResult<FileOpResult> {
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
            semantic_undo: None,
        })
    }

    /// Snapshot a path before an adapter writes it, so the resulting external
    /// write can participate in the same undo log as direct file operations.
    pub fn snapshot_write(&self, path: &Path) -> FileGuardResult<FileOpResult> {
        let safe = self.resolve(path)?;
        if safe.exists() {
            Ok(FileOpResult {
                action: FileOpAction::Modified,
                path: safe.clone(),
                backup_path: Some(self.backup(&safe)?),
                new_path: None,
                semantic_undo: None,
            })
        } else {
            Ok(FileOpResult {
                action: FileOpAction::Create,
                path: safe,
                backup_path: None,
                new_path: None,
                semantic_undo: None,
            })
        }
    }

    /// Allocate a private, same-filesystem staging directory for a sandboxed
    /// command. The child sees this directory only through its private output mount.
    pub fn create_command_staging(&self) -> FileGuardResult<PathBuf> {
        let path = self
            .project_root
            .join(".subbake/agent/command-runs")
            .join(format!("{}", nanos_since_epoch()));
        std::fs::create_dir_all(&path)?;
        Ok(path)
    }

    /// Commit regular files produced in a private command staging directory.
    /// All targets are validated before the first mutation and a failed
    /// multi-file commit is rolled back to its pre-command state.
    pub fn commit_staged_files(
        &self,
        files: &[(PathBuf, PathBuf)],
        overwrite: bool,
    ) -> FileGuardResult<Vec<FileOpResult>> {
        let mut prepared = Vec::with_capacity(files.len());
        let mut destinations = std::collections::HashSet::new();
        for (staged, destination) in files {
            let metadata =
                std::fs::symlink_metadata(staged).map_err(|source| FileGuardError::Io {
                    operation: "inspect staged command output",
                    path: Some(staged.clone()),
                    source,
                })?;
            if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
                return Err(FileGuardError::Io {
                    operation: "staged command output is not a regular file",
                    path: Some(staged.clone()),
                    source: std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "only regular file outputs are supported",
                    ),
                });
            }
            let safe = self.resolve(destination)?;
            if !destinations.insert(safe.clone()) {
                return Err(FileGuardError::Io {
                    operation: "commit staged command outputs",
                    path: Some(safe),
                    source: std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "duplicate output destination",
                    ),
                });
            }
            if safe.exists() && !overwrite {
                return Err(FileGuardError::AlreadyExists { path: safe });
            }
            if safe.exists() && !safe.is_file() {
                return Err(FileGuardError::Io {
                    operation: "replace command output",
                    path: Some(safe),
                    source: std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "output destination is not a regular file",
                    ),
                });
            }
            prepared.push((staged.clone(), safe));
        }

        let transaction_root = self
            .backup_root
            .join(format!("command-{}", nanos_since_epoch()));
        let backups = prepared
            .iter()
            .map(|(staged, destination)| {
                if let Some(parent) = destination.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                if destination.exists() {
                    let permissions = std::fs::metadata(destination)?.permissions();
                    std::fs::set_permissions(staged, permissions)?;
                    let relative = destination
                        .strip_prefix(&self.project_root)
                        .unwrap_or(destination);
                    let backup = transaction_root.join(relative);
                    if let Some(parent) = backup.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    Ok(Some(backup))
                } else {
                    Ok(None)
                }
            })
            .collect::<FileGuardResult<Vec<_>>>()?;
        let mut committed: Vec<(PathBuf, Option<PathBuf>)> = Vec::new();
        for (index, ((staged, destination), backup)) in prepared.iter().zip(backups).enumerate() {
            if let Some(backup) = &backup
                && let Err(source) = std::fs::rename(destination, backup)
            {
                rollback_committed_files(&committed);
                return Err(FileGuardError::Io {
                    operation: "back up command output destination",
                    path: Some(destination.clone()),
                    source,
                });
            }
            if let Err(source) = std::fs::rename(staged, destination) {
                if let Some(backup) = &backup {
                    let _ = std::fs::rename(backup, destination);
                }
                rollback_committed_files(&committed);
                return Err(FileGuardError::Io {
                    operation: "commit staged command output",
                    path: Some(prepared[index].1.clone()),
                    source,
                });
            }
            committed.push((destination.clone(), backup));
        }

        Ok(committed
            .into_iter()
            .map(|(path, backup_path)| FileOpResult {
                action: if backup_path.is_some() {
                    FileOpAction::Modified
                } else {
                    FileOpAction::Create
                },
                path,
                backup_path,
                new_path: None,
                semantic_undo: None,
            })
            .collect())
    }

    /// Persist only the previous text subtitle when an in-place media remux
    /// replaces a SubBake-managed track. This keeps undo proportional to the
    /// subtitle size instead of duplicating the entire media file.
    pub fn store_embedded_subtitle_undo(
        &self,
        container_path: &Path,
        contents: &[u8],
        subtitle_format: &str,
    ) -> FileGuardResult<PathBuf> {
        let safe = self.resolve(container_path)?;
        let relative = safe.strip_prefix(&self.project_root).unwrap_or(&safe);
        let mut payload_path = self
            .backup_root
            .join(format!("{}", nanos_since_epoch()))
            .join(relative);
        let name = payload_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("container");
        payload_path.set_file_name(format!("{name}.previous.{subtitle_format}"));
        if let Some(parent) = payload_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&payload_path, contents)?;
        Ok(payload_path)
    }

    pub fn list_files(&self, dir: &Path) -> FileGuardResult<Vec<PathBuf>> {
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
    pub fn search_files(&self, dir: &Path, pattern: &str) -> FileGuardResult<Vec<PathBuf>> {
        let safe = self.resolve(dir)?;
        let mut results = Vec::new();
        self.search_recursive(&safe, pattern, &mut results)?;
        results.sort();
        Ok(results)
    }

    pub fn resolve_path(&self, path: &Path) -> FileGuardResult<PathBuf> {
        self.resolve(path)
    }

    /// Resolve a persisted undo target through the same project boundary as
    /// live tool calls. A leaf symlink is rejected so an event cannot be
    /// redirected after the original mutation was recorded.
    pub(crate) fn resolve_undo_target(&self, path: &Path) -> FileGuardResult<PathBuf> {
        let anchored = normalize_path(if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_root.join(path)
        });
        if std::fs::symlink_metadata(&anchored)
            .is_ok_and(|metadata| metadata.file_type().is_symlink())
        {
            return Err(FileGuardError::Io {
                operation: "resolve undo target",
                path: Some(anchored),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "undo targets must not be symbolic links",
                ),
            });
        }
        self.resolve(path)
    }

    /// Validate a persisted backup path without applying the protected-path
    /// rule: backups intentionally live under `.subbake`, but nowhere else is
    /// accepted as an undo source.
    pub(crate) fn resolve_undo_backup(&self, path: &Path) -> FileGuardResult<PathBuf> {
        let anchored = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_root.join(path)
        };
        let metadata =
            std::fs::symlink_metadata(&anchored).map_err(|source| FileGuardError::Io {
                operation: "inspect undo backup",
                path: Some(anchored.clone()),
                source,
            })?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Err(FileGuardError::Io {
                operation: "inspect undo backup",
                path: Some(anchored),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "undo backup must be a regular file",
                ),
            });
        }
        let safe = anchored
            .canonicalize()
            .map_err(|source| FileGuardError::Io {
                operation: "resolve undo backup",
                path: Some(anchored.clone()),
                source,
            })?;
        let backup_root = self
            .backup_root
            .canonicalize()
            .map_err(|source| FileGuardError::Io {
                operation: "resolve undo backup directory",
                path: Some(self.backup_root.clone()),
                source,
            })?;
        if !safe.starts_with(&backup_root) {
            return Err(FileGuardError::PathEscape {
                root: backup_root,
                path: safe,
            });
        }
        Ok(safe)
    }

    pub(crate) fn remove_for_undo(&self, path: &Path) -> FileGuardResult<()> {
        let safe = self.resolve_undo_target(path)?;
        let metadata = match std::fs::symlink_metadata(&safe) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(source) => {
                return Err(FileGuardError::Io {
                    operation: "inspect undo target",
                    path: Some(safe),
                    source,
                });
            }
        };
        if !metadata.file_type().is_file() {
            return Err(FileGuardError::Io {
                operation: "remove undo target",
                path: Some(safe),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "undo can remove only a regular file created by SubBake",
                ),
            });
        }
        std::fs::remove_file(&safe).map_err(|source| FileGuardError::Io {
            operation: "remove undo target",
            path: Some(safe),
            source,
        })
    }

    pub(crate) fn restore_for_undo(
        &self,
        backup_path: &Path,
        target: &Path,
    ) -> FileGuardResult<()> {
        let backup = self.resolve_undo_backup(backup_path)?;
        let target = self.resolve_undo_target(target)?;
        Self::restore_backup(&backup, &target)
    }

    // ------------------------------------------------------------------
    // Path resolution + sandbox
    // ------------------------------------------------------------------

    /// Resolve a user-supplied path to an absolute path under the project root,
    /// rejecting paths that escape the project root or contain protected components.
    fn resolve(&self, user_path: &Path) -> FileGuardResult<PathBuf> {
        // Normalise `..` components so `root/../etc/passwd` is caught below.
        let anchored = normalize_path(if user_path.is_absolute() {
            user_path.to_path_buf()
        } else {
            self.project_root.join(user_path)
        });

        // ── Escape guard: anchor must be under project_root ──
        let root_canon = self
            .project_root
            .canonicalize()
            .unwrap_or_else(|_| self.project_root.clone());
        if !anchored.starts_with(&root_canon) {
            return Err(FileGuardError::PathEscape {
                root: root_canon,
                path: anchored,
            });
        }
        self.reject_protected_components(&anchored)?;

        // Canonicalize existing paths. For new nested paths, canonicalize the
        // nearest existing ancestor so a symlink in any parent component
        // cannot redirect a later create_dir_all outside the project.
        let safe = if anchored.exists() {
            anchored
                .canonicalize()
                .map_err(|e| std::io::Error::other(format!("resolve existing path: {e}")))?
        } else {
            let mut ancestor = anchored.as_path();
            while !ancestor.exists() {
                ancestor = ancestor
                    .parent()
                    .ok_or_else(|| FileGuardError::PathEscape {
                        root: root_canon.clone(),
                        path: anchored.clone(),
                    })?;
            }
            let canonical_ancestor = ancestor
                .canonicalize()
                .map_err(|e| std::io::Error::other(format!("resolve ancestor: {e}")))?;
            let suffix = anchored
                .strip_prefix(ancestor)
                .unwrap_or_else(|_| Path::new(""));
            canonical_ancestor.join(suffix)
        };

        if !safe.starts_with(&root_canon) {
            return Err(FileGuardError::PathEscape {
                root: root_canon,
                path: safe,
            });
        }

        self.reject_protected_components(&safe)?;

        Ok(safe)
    }

    fn reject_protected_components(&self, path: &Path) -> FileGuardResult<()> {
        for component in path.components() {
            if let Some(name) = component.as_os_str().to_str()
                && is_protected_component(name)
            {
                return Err(FileGuardError::ProtectedPath {
                    component: name.to_owned(),
                    path: path.to_path_buf(),
                });
            }
        }
        Ok(())
    }

    fn search_recursive(
        &self,
        dir: &Path,
        pattern: &str,
        results: &mut Vec<PathBuf>,
    ) -> FileGuardResult<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            // Recursive discovery must not follow links after the initial
            // directory has passed `resolve`. Following a directory symlink
            // here could escape the project root or recurse through a cycle.
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                self.search_recursive(&path, pattern, results)?;
            } else if file_type.is_file()
                && path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|name| {
                        if pattern.contains(['*', '?']) {
                            wildcard_matches(pattern, name)
                        } else {
                            name.contains(pattern)
                        }
                    })
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
    fn backup(&self, path: &Path) -> FileGuardResult<PathBuf> {
        if !path.exists() {
            return Err(FileGuardError::MissingBackupSource {
                path: path.to_path_buf(),
            });
        }

        let rel = path.strip_prefix(&self.project_root).unwrap_or(path);
        let ts = nanos_since_epoch();
        let backup_path = self.backup_root.join(format!("{ts}")).join(rel);

        if let Some(parent) = backup_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(path, &backup_path)?;
        Ok(backup_path)
    }

    /// Write content to a file atomically via temp + rename.
    fn write_atomically(&self, path: &Path, content: &str) -> FileGuardResult<()> {
        subbake_adapters::write_file_atomically(path, content.as_bytes()).map_err(|source| {
            FileGuardError::Io {
                operation: "atomically write project file",
                path: Some(path.to_path_buf()),
                source: std::io::Error::other(source),
            }
        })
    }

    /// Restore a file from a backup. Used by undo.
    pub fn restore_backup(backup_path: &Path, target: &Path) -> FileGuardResult<()> {
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(backup_path, target)?;
        Ok(())
    }
}

pub(crate) fn is_protected_component(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
    PROTECTED_PATH_PARTS.contains(&normalized.as_str())
        || (normalized.starts_with(".env.") && normalized != ".env.example")
}

fn rollback_committed_files(committed: &[(PathBuf, Option<PathBuf>)]) {
    for (path, prior) in committed.iter().rev() {
        let _ = std::fs::remove_file(path);
        if let Some(prior) = prior {
            let _ = std::fs::rename(prior, path);
        }
    }
}

fn wildcard_matches(pattern: &str, value: &str) -> bool {
    let pattern = pattern.chars().collect::<Vec<_>>();
    let value = value.chars().collect::<Vec<_>>();
    let mut matches = vec![vec![false; value.len() + 1]; pattern.len() + 1];
    matches[0][0] = true;
    for pattern_index in 1..=pattern.len() {
        if pattern[pattern_index - 1] == '*' {
            matches[pattern_index][0] = matches[pattern_index - 1][0];
        }
        for value_index in 1..=value.len() {
            matches[pattern_index][value_index] = match pattern[pattern_index - 1] {
                '*' => {
                    matches[pattern_index - 1][value_index]
                        || matches[pattern_index][value_index - 1]
                }
                '?' => matches[pattern_index - 1][value_index - 1],
                literal => {
                    literal == value[value_index - 1] && matches[pattern_index - 1][value_index - 1]
                }
            };
        }
    }
    matches[pattern.len()][value.len()]
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
    fn external_delete_rejects_project_paths_and_filesystem_root() {
        let (root, _) = setup();
        std::fs::create_dir_all(&root).expect("create project root");
        std::fs::write(root.join("inside.txt"), "keep").expect("write project file");
        let guard = ExternalPathGuard::new(root.clone());

        let project_error = guard
            .prepare(&root.join("inside.txt"))
            .expect_err("project path must use FileGuard");
        let root_error = guard
            .prepare(Path::new("/"))
            .expect_err("filesystem root must be protected");

        assert!(matches!(
            project_error,
            FileGuardError::ExternalPathInsideProject { .. }
        ));
        assert!(matches!(
            root_error,
            FileGuardError::CriticalExternalPath { .. }
        ));
        assert!(root.join("inside.txt").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn external_directory_requires_an_explicit_recursive_delete() {
        let (project_root, _) = setup();
        let outside = std::env::temp_dir().join(format!(
            "subbake-external-delete-directory-{}",
            nanos_since_epoch()
        ));
        std::fs::create_dir_all(&project_root).expect("create project root");
        std::fs::create_dir_all(&outside).expect("create external directory");
        std::fs::write(outside.join("data.txt"), "data").expect("write external file");
        let guard = ExternalPathGuard::new(project_root.clone());
        let approved = guard.prepare(&outside).expect("prepare external directory");

        let error = guard
            .delete(&approved.path, false)
            .expect_err("non-empty directory should require recursive=true");
        assert!(matches!(
            error,
            FileGuardError::ExternalDirectoryRequiresRecursive { .. }
        ));
        assert!(outside.exists());

        let deleted = guard
            .delete(&approved.path, true)
            .expect("recursive external deletion");
        assert_eq!(deleted.path, outside);
        assert!(!outside.exists());
        let _ = std::fs::remove_dir_all(project_root);
    }

    #[cfg(unix)]
    #[test]
    fn external_delete_removes_a_leaf_symlink_without_following_it() {
        use std::os::unix::fs::symlink;

        let (project_root, _) = setup();
        let outside = std::env::temp_dir().join(format!(
            "subbake-external-delete-symlink-{}",
            nanos_since_epoch()
        ));
        let target = outside.join("target");
        let link = outside.join("link");
        std::fs::create_dir_all(&project_root).expect("create project root");
        std::fs::create_dir_all(&target).expect("create symlink target");
        std::fs::write(target.join("keep.txt"), "keep").expect("write target file");
        symlink(&target, &link).expect("create symlink");
        let guard = ExternalPathGuard::new(project_root.clone());
        let approved = guard.prepare(&link).expect("prepare symlink deletion");

        assert!(approved.is_symlink);
        guard
            .delete(&approved.path, true)
            .expect("delete leaf symlink");
        assert!(!link.exists());
        assert!(target.join("keep.txt").exists());

        let _ = std::fs::remove_dir_all(project_root);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[test]
    fn staged_files_commit_transactionally_and_keep_overwrite_backup() {
        let (root, guard) = setup();
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join("existing.bin"), b"old").expect("old output");
        let staging = guard.create_command_staging().expect("staging");
        std::fs::write(staging.join("first"), b"new").expect("staged replacement");
        std::fs::write(staging.join("second"), b"created").expect("staged creation");

        let operations = guard
            .commit_staged_files(
                &[
                    (staging.join("first"), PathBuf::from("existing.bin")),
                    (staging.join("second"), PathBuf::from("created.bin")),
                ],
                true,
            )
            .expect("commit outputs");

        assert_eq!(
            std::fs::read(root.join("existing.bin")).expect("new"),
            b"new"
        );
        assert_eq!(
            std::fs::read(root.join("created.bin")).expect("created"),
            b"created"
        );
        let backup = operations[0]
            .backup_path
            .as_ref()
            .expect("overwrite backup");
        assert_eq!(std::fs::read(backup).expect("backup"), b"old");
        let _ = std::fs::remove_dir_all(root);
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
        assert!(matches!(&err, FileGuardError::PathEscape { .. }));
        assert!(err.to_string().contains("escapes project root"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_protected_path() {
        let (root, guard) = setup();
        let err = guard
            .create_file(Path::new(".git/config"), "data")
            .expect_err("should reject");
        assert!(matches!(&err, FileGuardError::ProtectedPath { .. }));
        assert!(err.to_string().contains(".git"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_secret_files_case_insensitively_but_allows_env_examples() {
        let (root, guard) = setup();
        for path in [".env", ".EnV.Local", ".SSH/config", "keys/ID_ED25519"] {
            let error = guard
                .read_file(Path::new(path))
                .expect_err("secret path should be rejected before reading");
            assert!(matches!(error, FileGuardError::ProtectedPath { .. }));
        }
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join(".env.example"), "TOKEN=replace-me\n").expect("write env example");
        assert_eq!(
            guard
                .read_file(Path::new(".env.example"))
                .expect("read env example"),
            "TOKEN=replace-me\n"
        );
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
    fn append_creates_a_missing_file_without_a_fake_backup() {
        let (root, guard) = setup();
        let path = Path::new("new-log.txt");

        let result = guard.append_file(path, "first line\n").expect("append");

        assert_eq!(result.action, FileOpAction::Create);
        assert!(result.backup_path.is_none());
        assert_eq!(guard.read_file(path).expect("read"), "first line\n");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn embedded_subtitle_undo_stores_only_the_small_payload() {
        let (root, guard) = setup();
        std::fs::create_dir_all(&root).expect("create project root");
        let container = root.join("movie.mkv");
        std::fs::write(&container, b"media bytes").expect("write container placeholder");

        let payload = guard
            .store_embedded_subtitle_undo(&container, b"previous subtitle", "srt")
            .expect("store subtitle undo");

        assert!(payload.starts_with(root.join(".subbake/agent/backups")));
        assert_eq!(
            std::fs::read(&payload).expect("read undo payload"),
            b"previous subtitle"
        );
        assert_ne!(payload, container);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn legacy_embedded_subtitle_undo_defaults_to_srt_at_restore_time() {
        let value = serde_json::json!({
            "kind": "restore_embedded_subtitle",
            "title": "zh-Hans (SubBake translation)",
            "subtitle_backup_path": "backup.srt"
        });

        let undo: SemanticUndo = serde_json::from_value(value).expect("legacy semantic undo");

        assert!(matches!(
            undo,
            SemanticUndo::RestoreEmbeddedSubtitle {
                subtitle_format: None,
                ..
            }
        ));
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
        guard
            .create_file(Path::new("a.txt"), "data")
            .expect("create");
        let result = guard
            .rename_path(Path::new("a.txt"), Path::new("b.txt"))
            .expect("rename");
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
        let err = guard
            .read_file(Path::new("missing.txt"))
            .expect_err("should fail");
        assert!(err.to_string().contains("missing.txt"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn search_files_supports_wildcards_and_keeps_substring_matching() {
        let (root, guard) = setup();
        std::fs::create_dir_all(root.join("nested")).expect("create nested directory");
        std::fs::write(root.join("movie.srt"), "one").expect("write srt");
        std::fs::write(root.join("nested/notes.txt"), "two").expect("write txt");

        let srt = guard
            .search_files(Path::new("."), "*.srt")
            .expect("search wildcard");
        assert_eq!(srt, vec![root.join("movie.srt")]);

        let text = guard
            .search_files(Path::new("."), "notes")
            .expect("search substring");
        assert_eq!(text, vec![root.join("nested/notes.txt")]);

        let all = guard.search_files(Path::new("."), "").expect("search all");
        assert_eq!(all.len(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn wildcard_match_supports_single_character_patterns() {
        assert!(wildcard_matches("episode-??.srt", "episode-01.srt"));
        assert!(!wildcard_matches("episode-?.srt", "episode-01.srt"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let (root, guard) = setup();
        std::fs::create_dir_all(&root).expect("create root");
        let outside = std::env::temp_dir().join(format!("subbake-outside-{}", nanos_since_epoch()));
        std::fs::create_dir_all(&outside).expect("create outside");
        symlink(&outside, root.join("outside-link")).expect("create symlink");

        let err = guard
            .create_file(Path::new("outside-link/escape.txt"), "data")
            .expect_err("symlink escape should be rejected");

        assert!(err.to_string().contains("escapes project root"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[cfg(windows)]
    #[test]
    fn rejects_windows_junction_escape() {
        let (root, guard) = setup();
        std::fs::create_dir_all(&root).expect("create root");
        let outside = std::env::temp_dir().join(format!("subbake-outside-{}", nanos_since_epoch()));
        std::fs::create_dir_all(&outside).expect("create outside");
        let link = root.join("outside-link");
        let status = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(&link)
            .arg(&outside)
            .status()
            .expect("create junction");
        assert!(status.success(), "mklink failed with {status}");

        let error = guard
            .create_file(Path::new("outside-link/escape.txt"), "data")
            .expect_err("junction escape should be rejected");

        assert!(
            error.to_string().contains("escapes project root"),
            "{error}"
        );
        std::fs::remove_dir(&link).expect("remove junction");
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[cfg(unix)]
    #[test]
    fn recursive_search_skips_external_and_cyclic_symlink_directories() {
        use std::os::unix::fs::symlink;

        let (root, guard) = setup();
        let outside =
            std::env::temp_dir().join(format!("subbake-search-outside-{}", nanos_since_epoch()));
        std::fs::create_dir_all(root.join("nested")).expect("create project tree");
        std::fs::create_dir_all(&outside).expect("create outside tree");
        std::fs::write(root.join("nested/inside.srt"), "inside").expect("write inside file");
        std::fs::write(outside.join("outside.srt"), "outside").expect("write outside file");
        symlink(&outside, root.join("outside-link")).expect("link outside directory");
        symlink(&root, root.join("nested/cycle")).expect("link project cycle");

        let files = guard
            .search_files(Path::new("."), "*.srt")
            .expect("bounded recursive search");

        assert_eq!(files, vec![root.join("nested/inside.srt")]);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outside);
    }
}
