use std::fs;
use std::path::{Path, PathBuf};

use subbake_core::storage::{RuntimePaths, build_runtime_paths};

use crate::error::{AdapterError, AdapterResult};
use crate::fs::stable_runtime_input_path;
use crate::platform::PlatformPaths;
use crate::runtime_store::{RUNTIME_MARKER_CONTENT, RUNTIME_MARKER_NAME};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeRequest {
    pub action: RuntimeAction,
    pub target_path: PathBuf,
    pub runtime_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeAction {
    Inspect,
    Clean {
        yes: bool,
        clean_runs: bool,
        clean_cache: bool,
        clean_glossary: bool,
        all: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeOutcome {
    Inspection(Box<RuntimeInspection>),
    Clean(RuntimeCleanOutcome),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeInspection {
    pub paths: RuntimePaths,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeCleanOutcome {
    pub root_dir: PathBuf,
    pub removed: bool,
}

pub fn run_runtime(request: RuntimeRequest) -> AdapterResult<RuntimeOutcome> {
    let (paths, stable_target_path) = runtime_paths(&request)?;
    match request.action {
        RuntimeAction::Inspect => Ok(RuntimeOutcome::Inspection(Box::new(RuntimeInspection {
            paths,
        }))),
        RuntimeAction::Clean {
            yes,
            clean_runs,
            clean_cache,
            clean_glossary,
            all,
        } => clean_runtime(
            paths,
            &stable_target_path,
            yes,
            clean_runs,
            clean_cache,
            clean_glossary,
            all,
        ),
    }
}

fn runtime_paths(request: &RuntimeRequest) -> AdapterResult<(RuntimePaths, PathBuf)> {
    let stable_input_path = stable_runtime_input_path(&request.target_path)?;
    Ok((
        build_runtime_paths(
            &request.target_path,
            &stable_input_path,
            request.runtime_dir.as_deref(),
            None,
            "Auto",
            "Chinese",
            false,
        ),
        stable_input_path,
    ))
}

fn clean_runtime(
    paths: RuntimePaths,
    stable_target_path: &Path,
    yes: bool,
    clean_runs: bool,
    clean_cache: bool,
    clean_glossary: bool,
    all: bool,
) -> AdapterResult<RuntimeOutcome> {
    if !yes {
        return Err(AdapterError::invalid_input(
            "runtime clean requires --yes in the current non-interactive implementation",
        ));
    }

    if !all && !clean_runs && !clean_cache && !clean_glossary {
        return Err(AdapterError::invalid_input(
            "runtime clean requires at least one of --runs, --cache, --glossary, or --all",
        ));
    }

    if !paths.root_dir.exists() {
        return Ok(RuntimeOutcome::Clean(RuntimeCleanOutcome {
            root_dir: paths.root_dir,
            removed: false,
        }));
    }

    let root_dir = validate_managed_runtime_root(&paths.root_dir, stable_target_path)?;

    let mut removed = false;
    if all {
        removed |= remove_dir_if_exists(&managed_target(&root_dir, &root_dir.join("runs"))?)?;
        removed |= remove_dir_if_exists(&managed_target(&root_dir, &root_dir.join("cache"))?)?;
        for entry in fs::read_dir(&root_dir)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if file_type.is_file()
                && (is_managed_json(&name, "glossary.")
                    || is_managed_json(&name, "translation_memory."))
            {
                fs::remove_file(entry.path())?;
                removed = true;
            }
        }
    } else {
        if clean_runs {
            let run_dir = rebase_runtime_path(&paths.root_dir, &root_dir, &paths.run_dir)?;
            removed |= remove_dir_if_exists(&managed_target(&root_dir, &run_dir)?)?;
        }
        if clean_cache {
            removed |= remove_dir_if_exists(&managed_target(&root_dir, &root_dir.join("cache"))?)?;
        }
        if clean_glossary {
            let glossary = rebase_runtime_path(&paths.root_dir, &root_dir, &paths.glossary_path)?;
            removed |= remove_file_if_exists(&managed_target(&root_dir, &glossary)?)?;
        }
    }
    Ok(RuntimeOutcome::Clean(RuntimeCleanOutcome {
        root_dir: paths.root_dir,
        removed,
    }))
}

fn validate_managed_runtime_root(root: &Path, stable_target_path: &Path) -> AdapterResult<PathBuf> {
    let root_identity = PlatformPaths::identify_existing(root)?;
    let target_identity = PlatformPaths::identify_with_missing_tail(stable_target_path)?;
    let canonical_root = root_identity.resolved().to_path_buf();
    let named_default = canonical_root
        .file_name()
        .is_some_and(|name| name == std::ffi::OsStr::new(".subbake"));
    let marker = canonical_root.join(RUNTIME_MARKER_NAME);
    let marked = fs::read_to_string(marker).is_ok_and(|content| content == RUNTIME_MARKER_CONTENT);
    let critical = canonical_root.parent().is_none()
        || PlatformPaths::canonical_home_dir().as_deref() == Some(canonical_root.as_path())
        || target_identity.resolved().starts_with(&canonical_root);
    if (named_default || marked) && !critical {
        return Ok(canonical_root);
    }
    Err(AdapterError::invalid_input(format!(
        "refusing to clean unsafe or unmarked runtime directory `{}`; use a dedicated SubBake-created runtime directory or the default `.subbake` directory",
        root.display()
    )))
}

fn rebase_runtime_path(
    original_root: &Path,
    canonical_root: &Path,
    path: &Path,
) -> AdapterResult<PathBuf> {
    let relative = path.strip_prefix(original_root).map_err(|_| {
        AdapterError::invalid_input(format!(
            "runtime artifact `{}` is outside runtime root `{}`",
            path.display(),
            original_root.display()
        ))
    })?;
    Ok(canonical_root.join(relative))
}

fn managed_target(root: &Path, path: &Path) -> AdapterResult<PathBuf> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(path.to_path_buf()),
        Err(source) => {
            return Err(AdapterError::external_io(
                "inspect runtime artifact",
                Some(path.to_path_buf()),
                source,
            ));
        }
    };
    if metadata.file_type().is_symlink() {
        return Ok(path.to_path_buf());
    }
    let resolved = path.canonicalize().map_err(|source| {
        AdapterError::external_io("resolve runtime artifact", Some(path.to_path_buf()), source)
    })?;
    if !resolved.starts_with(root) {
        return Err(AdapterError::invalid_input(format!(
            "refusing to clean runtime artifact that resolves outside `{}`: {}",
            root.display(),
            resolved.display()
        )));
    }
    Ok(resolved)
}

fn is_managed_json(name: &str, prefix: &str) -> bool {
    name.starts_with(prefix) && name.ends_with(".json")
}

fn remove_dir_if_exists(path: &Path) -> AdapterResult<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(source) => {
            return Err(AdapterError::external_io(
                "inspect runtime directory",
                Some(path.to_path_buf()),
                source,
            ));
        }
    }
    fs::remove_dir_all(path)?;
    Ok(true)
}

fn remove_file_if_exists(path: &Path) -> AdapterResult<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(source) => {
            return Err(AdapterError::external_io(
                "inspect runtime file",
                Some(path.to_path_buf()),
                source,
            ));
        }
    }
    fs::remove_file(path)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn inspect_returns_runtime_paths() {
        let root = temp_root("inspect");
        let outcome = run_runtime(RuntimeRequest {
            action: RuntimeAction::Inspect,
            target_path: root.join("clip.srt"),
            runtime_dir: Some(root.join(".runtime")),
        })
        .expect("inspect runtime");

        let RuntimeOutcome::Inspection(inspection) = outcome else {
            panic!("expected inspection");
        };
        assert_eq!(inspection.paths.root_dir, root.join(".runtime"));
        assert!(inspection.paths.state_path.ends_with("run_state.json"));
    }

    #[test]
    fn clean_requires_yes() {
        let error = run_runtime(RuntimeRequest {
            action: RuntimeAction::Clean {
                yes: false,
                clean_runs: false,
                clean_cache: false,
                clean_glossary: false,
                all: true,
            },
            target_path: PathBuf::from("clip.srt"),
            runtime_dir: None,
        })
        .expect_err("clean should require confirmation");

        assert!(error.to_string().contains("--yes"));
    }

    #[test]
    fn clean_all_removes_only_managed_artifacts_and_preserves_root() {
        let root = temp_root("clean");
        let runtime_dir = root.join(".runtime");
        fs::create_dir_all(runtime_dir.join("cache")).expect("create runtime");
        fs::create_dir_all(runtime_dir.join("runs/other-run")).expect("create runs");
        fs::write(
            runtime_dir.join(RUNTIME_MARKER_NAME),
            RUNTIME_MARKER_CONTENT,
        )
        .expect("write marker");
        fs::write(runtime_dir.join("glossary.en-zh.json"), "{}").expect("write glossary");
        fs::write(runtime_dir.join("keep.txt"), "keep").expect("write unrelated file");

        let outcome = run_runtime(RuntimeRequest {
            action: RuntimeAction::Clean {
                yes: true,
                clean_runs: false,
                clean_cache: false,
                clean_glossary: false,
                all: true,
            },
            target_path: root.join("clip.srt"),
            runtime_dir: Some(runtime_dir.clone()),
        })
        .expect("clean runtime");
        let exists = runtime_dir.exists();
        let unrelated_exists = runtime_dir.join("keep.txt").exists();
        let runs_exist = runtime_dir.join("runs").exists();
        let cache_exists = runtime_dir.join("cache").exists();
        let _ = fs::remove_dir_all(&root);

        assert_eq!(
            outcome,
            RuntimeOutcome::Clean(RuntimeCleanOutcome {
                root_dir: runtime_dir,
                removed: true
            })
        );
        assert!(exists);
        assert!(unrelated_exists);
        assert!(!runs_exist);
        assert!(!cache_exists);
    }

    #[test]
    fn clean_requires_an_explicit_scope() {
        let error = run_runtime(RuntimeRequest {
            action: RuntimeAction::Clean {
                yes: true,
                clean_runs: false,
                clean_cache: false,
                clean_glossary: false,
                all: false,
            },
            target_path: PathBuf::from("clip.srt"),
            runtime_dir: None,
        })
        .expect_err("clean should require a scope");

        assert!(error.to_string().contains("--runs"));
    }

    #[test]
    fn clean_rejects_an_unmarked_arbitrary_root() {
        let root = temp_root("unmarked");
        fs::create_dir_all(root.join("cache")).expect("create arbitrary root");

        let error = run_runtime(RuntimeRequest {
            action: RuntimeAction::Clean {
                yes: true,
                clean_runs: false,
                clean_cache: true,
                clean_glossary: false,
                all: false,
            },
            target_path: root.join("clip.srt"),
            runtime_dir: Some(root.clone()),
        })
        .expect_err("unmarked arbitrary root should be rejected");
        let cache_exists = root.join("cache").exists();
        let _ = fs::remove_dir_all(root);

        assert!(error.to_string().contains("unmarked runtime directory"));
        assert!(cache_exists);
    }

    #[test]
    fn clean_rejects_a_marked_directory_that_contains_the_target() {
        let root = temp_root("target-ancestor");
        fs::create_dir_all(root.join("subtitles")).expect("create target directory");
        fs::create_dir_all(root.join("cache")).expect("create cache");
        fs::write(root.join(RUNTIME_MARKER_NAME), RUNTIME_MARKER_CONTENT)
            .expect("write misleading marker");
        fs::write(root.join("cache/keep.txt"), "keep").expect("write cache file");

        let error = run_runtime(RuntimeRequest {
            action: RuntimeAction::Clean {
                yes: true,
                clean_runs: false,
                clean_cache: true,
                clean_glossary: false,
                all: false,
            },
            target_path: root.join("subtitles/clip.srt"),
            runtime_dir: Some(root.clone()),
        })
        .expect_err("a target ancestor is not a dedicated runtime directory");
        let cache_exists = root.join("cache/keep.txt").exists();
        let _ = fs::remove_dir_all(root);

        assert!(error.to_string().contains("unsafe or unmarked"));
        assert!(cache_exists);
    }

    #[cfg(unix)]
    #[test]
    fn clean_rejects_a_default_named_symlink_to_an_unmanaged_directory() {
        use std::os::unix::fs::symlink;

        let root = temp_root("runtime-link");
        let project = root.join("project");
        let outside = root.join("outside");
        fs::create_dir_all(&project).expect("create project");
        fs::create_dir_all(outside.join("cache")).expect("create outside cache");
        fs::write(outside.join("cache/keep.txt"), "keep").expect("write outside file");
        symlink(&outside, project.join(".subbake")).expect("create runtime symlink");

        let error = run_runtime(RuntimeRequest {
            action: RuntimeAction::Clean {
                yes: true,
                clean_runs: false,
                clean_cache: true,
                clean_glossary: false,
                all: false,
            },
            target_path: project.join("clip.srt"),
            runtime_dir: None,
        })
        .expect_err("symlink name must not establish runtime ownership");
        let outside_exists = outside.join("cache/keep.txt").exists();
        let _ = fs::remove_dir_all(root);

        assert!(error.to_string().contains("unsafe or unmarked"));
        assert!(outside_exists);
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-runtime-service-{label}-{nanos}"))
    }
}
