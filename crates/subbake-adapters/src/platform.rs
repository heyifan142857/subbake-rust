use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::error::{AdapterError, AdapterResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatingSystem {
    Linux,
    MacOs,
    Windows,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    X86_64,
    Aarch64,
    Other,
}

/// Capabilities derived from the compiled target, kept in one place so
/// feature registration and adapter selection cannot drift independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilitySet {
    operating_system: OperatingSystem,
    architecture: Architecture,
}

impl CapabilitySet {
    pub fn current() -> Self {
        Self::from_target(std::env::consts::OS, std::env::consts::ARCH)
    }

    pub fn from_target(os: &str, architecture: &str) -> Self {
        let operating_system = match os {
            "linux" => OperatingSystem::Linux,
            "macos" => OperatingSystem::MacOs,
            "windows" => OperatingSystem::Windows,
            _ => OperatingSystem::Other,
        };
        let architecture = match architecture {
            "x86_64" => Architecture::X86_64,
            "aarch64" => Architecture::Aarch64,
            _ => Architecture::Other,
        };
        Self {
            operating_system,
            architecture,
        }
    }

    pub const fn operating_system(self) -> OperatingSystem {
        self.operating_system
    }

    pub const fn architecture(self) -> Architecture {
        self.architecture
    }

    pub const fn supports_command_sandbox(self) -> bool {
        matches!(self.operating_system, OperatingSystem::Linux)
    }

    pub const fn has_prebuilt_whisper(self) -> bool {
        matches!(
            (self.operating_system, self.architecture),
            (
                OperatingSystem::Linux,
                Architecture::X86_64 | Architecture::Aarch64
            ) | (OperatingSystem::Windows, Architecture::X86_64)
        )
    }
}

/// Native home/config path resolution shared by configuration and destructive
/// path guards. Environment access remains at the adapter edge.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlatformPaths;

/// The two filesystem identities of one path.
///
/// `logical` is an absolute, lexically-normalized spelling suitable for
/// presentation and stable application state. `resolved` is the operating
/// system identity used only for containment and filesystem operations. The
/// latter may contain platform-specific spellings such as `/private/var` or a
/// Windows verbatim prefix and must not leak into user-facing paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathIdentity {
    logical: PathBuf,
    resolved: PathBuf,
}

/// Identity and type of an existing directory entry without following a
/// symbolic-link leaf. Parent aliases are still resolved for authorization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathEntryIdentity {
    identity: PathIdentity,
    is_directory: bool,
    is_symlink: bool,
}

impl PathEntryIdentity {
    pub fn identity(&self) -> &PathIdentity {
        &self.identity
    }

    pub const fn is_directory(&self) -> bool {
        self.is_directory
    }

    pub const fn is_symlink(&self) -> bool {
        self.is_symlink
    }
}

impl PathIdentity {
    pub fn logical(&self) -> &Path {
        &self.logical
    }

    pub fn resolved(&self) -> &Path {
        &self.resolved
    }

    pub fn resolved_relative_to(&self, root: &Self) -> Option<&Path> {
        self.resolved.strip_prefix(&root.resolved).ok()
    }
}

impl PlatformPaths {
    pub fn home_dir() -> Option<PathBuf> {
        home_dir_with(CapabilitySet::current().operating_system(), |key| {
            std::env::var_os(key)
        })
    }

    pub fn canonical_home_dir() -> Option<PathBuf> {
        let home = Self::home_dir()?;
        Some(home.canonicalize().unwrap_or(home))
    }

    pub fn config_dir() -> Option<PathBuf> {
        config_dir_with(CapabilitySet::current().operating_system(), |key| {
            std::env::var_os(key)
        })
    }

    /// Identify an existing path. Callers can compare `resolved` identities
    /// while retaining `logical` for stable output.
    pub fn identify_existing(path: &Path) -> AdapterResult<PathIdentity> {
        let logical = absolute_lexical(path)?;
        let resolved = logical.canonicalize().map_err(|source| {
            AdapterError::external_io("resolve existing path", Some(logical.clone()), source)
        })?;
        Ok(PathIdentity { logical, resolved })
    }

    /// Identify an existing directory entry while preserving a symbolic-link
    /// leaf as the entry being operated on. This is the safe identity for
    /// delete and rename operations that must affect the link, not its target.
    pub fn identify_entry_without_following_leaf(path: &Path) -> AdapterResult<PathEntryIdentity> {
        let logical = absolute_lexical(path)?;
        let metadata = std::fs::symlink_metadata(&logical).map_err(|source| {
            AdapterError::external_io("inspect path entry", Some(logical.clone()), source)
        })?;
        let is_symlink = metadata.file_type().is_symlink();
        let identity = if is_symlink {
            let parent = logical.parent().ok_or_else(|| {
                AdapterError::invalid_input(format!(
                    "path entry has no parent: {}",
                    logical.display()
                ))
            })?;
            let parent = Self::identify_existing(parent)?;
            let file_name = logical.file_name().map(ToOwned::to_owned).ok_or_else(|| {
                AdapterError::invalid_input(format!(
                    "path entry has no file name: {}",
                    logical.display()
                ))
            })?;
            PathIdentity {
                logical,
                resolved: parent.resolved.join(file_name),
            }
        } else {
            Self::identify_existing(&logical)?
        };
        Ok(PathEntryIdentity {
            identity,
            is_directory: metadata.is_dir(),
            is_symlink,
        })
    }

    /// Identify a path whose final components may not exist yet.
    ///
    /// The deepest filesystem entry is resolved first, including symlinks and
    /// junctions, and the missing suffix is then appended. A dangling symlink
    /// counts as an entry and therefore fails canonicalization safely.
    pub fn identify_with_missing_tail(path: &Path) -> AdapterResult<PathIdentity> {
        let logical = absolute_lexical(path)?;
        let mut ancestor = logical.as_path();
        loop {
            match std::fs::symlink_metadata(ancestor) {
                Ok(_) => break,
                Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                    ancestor = ancestor.parent().ok_or_else(|| {
                        AdapterError::external_io(
                            "locate existing path ancestor",
                            Some(logical.clone()),
                            source,
                        )
                    })?;
                }
                Err(source) => {
                    return Err(AdapterError::external_io(
                        "inspect path ancestor",
                        Some(ancestor.to_path_buf()),
                        source,
                    ));
                }
            }
        }
        let resolved_ancestor = ancestor.canonicalize().map_err(|source| {
            AdapterError::external_io(
                "resolve existing path ancestor",
                Some(ancestor.to_path_buf()),
                source,
            )
        })?;
        let suffix = logical
            .strip_prefix(ancestor)
            .map(Path::to_path_buf)
            .map_err(|source| {
                AdapterError::external_io(
                    "derive missing path suffix",
                    Some(logical.clone()),
                    std::io::Error::other(source),
                )
            })?;
        let resolved = if suffix.as_os_str().is_empty() {
            resolved_ancestor
        } else {
            resolved_ancestor.join(suffix)
        };
        Ok(PathIdentity { logical, resolved })
    }
}

fn absolute_lexical(path: &Path) -> AdapterResult<PathBuf> {
    let anchored = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|source| AdapterError::external_io("resolve current directory", None, source))?
            .join(path)
    };
    Ok(normalize_absolute(anchored))
}

fn normalize_absolute(path: PathBuf) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if normalized.file_name().is_some() {
                    normalized.pop();
                }
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

fn home_dir_with(
    operating_system: OperatingSystem,
    lookup: impl Fn(&str) -> Option<OsString>,
) -> Option<PathBuf> {
    if operating_system == OperatingSystem::Windows {
        if let Some(profile) = lookup("USERPROFILE") {
            return Some(PathBuf::from(profile));
        }
        let drive = lookup("HOMEDRIVE")?;
        let path = lookup("HOMEPATH")?;
        let mut home = PathBuf::from(drive);
        home.push(path);
        return Some(home);
    }
    lookup("HOME").map(PathBuf::from)
}

fn config_dir_with(
    operating_system: OperatingSystem,
    lookup: impl Fn(&str) -> Option<OsString> + Copy,
) -> Option<PathBuf> {
    if operating_system == OperatingSystem::Windows
        && let Some(path) = lookup("APPDATA")
    {
        return Some(PathBuf::from(path));
    }
    if let Some(path) = lookup("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(path));
    }
    home_dir_with(operating_system, lookup).map(|home| home.join(".config"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(values: &[(&str, &str)], key: &str) -> Option<OsString> {
        values
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, value)| OsString::from(value))
    }

    #[test]
    fn windows_paths_use_native_environment_fallbacks() {
        let values = [
            ("HOME", "/home/msys-user"),
            ("XDG_CONFIG_HOME", "/home/msys-user/.config"),
            ("USERPROFILE", r"C:\Users\Alice"),
            ("APPDATA", r"C:\Config"),
        ];

        assert_eq!(
            home_dir_with(OperatingSystem::Windows, |key| lookup(&values, key)),
            Some(PathBuf::from(r"C:\Users\Alice"))
        );
        assert_eq!(
            config_dir_with(OperatingSystem::Windows, |key| lookup(&values, key)),
            Some(PathBuf::from(r"C:\Config"))
        );
    }

    #[test]
    fn capabilities_are_derived_from_one_target_identity() {
        let linux = CapabilitySet::from_target("linux", "aarch64");
        let mac = CapabilitySet::from_target("macos", "aarch64");
        let windows = CapabilitySet::from_target("windows", "x86_64");

        assert!(linux.supports_command_sandbox());
        assert!(linux.has_prebuilt_whisper());
        assert!(!mac.supports_command_sandbox());
        assert!(!mac.has_prebuilt_whisper());
        assert!(windows.has_prebuilt_whisper());
    }

    #[cfg(unix)]
    #[test]
    fn path_identity_separates_logical_spelling_from_filesystem_identity() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::Builder::new()
            .prefix("subbake-path-identity-")
            .tempdir()
            .expect("create temporary directory");
        let container = temporary.path();
        let actual = container.join("actual");
        let alias = container.join("alias");
        std::fs::create_dir_all(&actual).expect("create actual directory");
        symlink(&actual, &alias).expect("create alias");

        let identity = PlatformPaths::identify_with_missing_tail(&alias.join("new/file.srt"))
            .expect("identify missing child");

        assert_eq!(identity.logical(), alias.join("new/file.srt"));
        assert_eq!(
            identity.resolved(),
            actual
                .canonicalize()
                .expect("canonical actual directory")
                .join("new/file.srt")
        );
    }

    #[cfg(unix)]
    #[test]
    fn path_identity_rejects_a_dangling_symlink_ancestor() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::Builder::new()
            .prefix("subbake-path-dangling-")
            .tempdir()
            .expect("create temporary directory");
        let container = temporary.path();
        symlink(container.join("missing"), container.join("link")).expect("create dangling link");

        PlatformPaths::identify_with_missing_tail(&container.join("link/file.srt"))
            .expect_err("dangling symlink must not be treated as a missing directory");
    }

    #[cfg(unix)]
    #[test]
    fn leaf_symlink_identity_preserves_the_link_entry() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::Builder::new()
            .prefix("subbake-path-leaf-link-")
            .tempdir()
            .expect("create temporary directory");
        let target = temporary.path().join("target.txt");
        let link = temporary.path().join("link.txt");
        std::fs::write(&target, "target").expect("write target");
        symlink(&target, &link).expect("create link");

        let entry = PlatformPaths::identify_entry_without_following_leaf(&link)
            .expect("identify leaf symlink");

        assert!(entry.is_symlink());
        assert!(!entry.is_directory());
        assert_eq!(entry.identity().logical(), link);
        assert_eq!(
            entry.identity().resolved(),
            temporary
                .path()
                .canonicalize()
                .expect("resolve temporary directory")
                .join("link.txt")
        );
        assert_ne!(
            entry.identity().resolved(),
            target.canonicalize().expect("resolve target")
        );
    }

    #[test]
    fn missing_tail_keeps_logical_spelling_and_resolves_existing_parent() {
        let temporary = tempfile::Builder::new()
            .prefix("subbake-path-missing-tail-")
            .tempdir()
            .expect("create temporary directory");
        let missing = temporary.path().join("not-created/child.srt");

        let identity = PlatformPaths::identify_with_missing_tail(&missing)
            .expect("identify path with missing tail");

        assert_eq!(identity.logical(), missing);
        assert_eq!(
            identity.resolved(),
            temporary
                .path()
                .canonicalize()
                .expect("resolve temporary directory")
                .join("not-created/child.srt")
        );
    }

    #[test]
    fn temporary_test_directories_are_unique_and_cleanup_during_unwind() {
        let parent = tempfile::Builder::new()
            .prefix("subbake-tempdir-parent-")
            .tempdir()
            .expect("create temporary parent");
        let roots = (0..16)
            .map(|_| {
                let parent = parent.path().to_path_buf();
                std::thread::spawn(move || {
                    tempfile::Builder::new()
                        .prefix("parallel-")
                        .tempdir_in(parent)
                        .expect("create parallel temporary directory")
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|worker| worker.join().expect("join temporary directory worker"))
            .collect::<Vec<_>>();
        let unique = roots
            .iter()
            .map(tempfile::TempDir::path)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), roots.len());
        assert!(roots.iter().all(|root| root.path().is_dir()));
        drop(roots);
        assert!(
            std::fs::read_dir(parent.path())
                .expect("read temporary parent")
                .next()
                .is_none()
        );

        let panic_path = parent.path().join("panic-cleanup");
        let child_path = panic_path.clone();
        let result = std::thread::spawn(move || {
            let temporary = tempfile::Builder::new()
                .prefix("panic-cleanup")
                .tempdir_in(child_path.parent().expect("panic cleanup parent"))
                .expect("create panic cleanup directory");
            let created = temporary.path().to_path_buf();
            assert!(created.is_dir());
            panic!("exercise TempDir cleanup during unwind");
        })
        .join();
        assert!(result.is_err());
        assert!(
            std::fs::read_dir(parent.path())
                .expect("read temporary parent after panic")
                .next()
                .is_none()
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_identity_does_not_leak_verbatim_spelling_to_logical_path() {
        let temporary = tempfile::Builder::new()
            .prefix("subbake-windows-path-")
            .tempdir()
            .expect("create temporary directory");
        let identity = PlatformPaths::identify_existing(temporary.path())
            .expect("identify Windows temporary directory");

        assert!(!identity.logical().to_string_lossy().starts_with(r"\\?\"));
        assert!(identity.resolved().is_absolute());
    }

    #[cfg(windows)]
    #[test]
    fn windows_home_drive_and_path_form_an_absolute_native_path() {
        let values = [("HOMEDRIVE", "C:"), ("HOMEPATH", r"\Users\Alice")];

        assert_eq!(
            home_dir_with(OperatingSystem::Windows, |key| lookup(&values, key)),
            Some(PathBuf::from(r"C:\Users\Alice"))
        );
    }
}
