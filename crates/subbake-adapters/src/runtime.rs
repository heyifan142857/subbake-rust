use std::fs;
use std::path::PathBuf;

use subbake_core::storage::{RuntimePaths, build_runtime_paths};

use crate::error::{AdapterError, AdapterResult};
use crate::fs::stable_runtime_input_path;

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
    let paths = runtime_paths(&request)?;
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
        } => clean_runtime(paths, yes, clean_runs, clean_cache, clean_glossary, all),
    }
}

fn runtime_paths(request: &RuntimeRequest) -> AdapterResult<RuntimePaths> {
    let stable_input_path = stable_runtime_input_path(&request.target_path)?;
    Ok(build_runtime_paths(
        &request.target_path,
        &stable_input_path,
        request.runtime_dir.as_deref(),
        None,
        "Auto",
        "Chinese",
        false,
    ))
}

fn clean_runtime(
    paths: RuntimePaths,
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

    if !paths.root_dir.exists() {
        return Ok(RuntimeOutcome::Clean(RuntimeCleanOutcome {
            root_dir: paths.root_dir,
            removed: false,
        }));
    }

    let should_remove_all = all || (!clean_runs && !clean_cache && !clean_glossary);
    if should_remove_all {
        fs::remove_dir_all(&paths.root_dir)?;
        return Ok(RuntimeOutcome::Clean(RuntimeCleanOutcome {
            root_dir: paths.root_dir,
            removed: true,
        }));
    }

    if clean_runs && paths.run_dir.exists() {
        fs::remove_dir_all(&paths.run_dir)?;
    }
    if clean_cache && paths.cache_dir.exists() {
        fs::remove_dir_all(&paths.cache_dir)?;
    }
    if clean_glossary && paths.glossary_path.exists() {
        fs::remove_file(&paths.glossary_path)?;
    }
    Ok(RuntimeOutcome::Clean(RuntimeCleanOutcome {
        root_dir: paths.root_dir,
        removed: true,
    }))
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
    fn clean_removes_runtime_root() {
        let root = temp_root("clean");
        let runtime_dir = root.join(".runtime");
        fs::create_dir_all(runtime_dir.join("cache")).expect("create runtime");

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
        let _ = fs::remove_dir_all(&root);

        assert_eq!(
            outcome,
            RuntimeOutcome::Clean(RuntimeCleanOutcome {
                root_dir: runtime_dir,
                removed: true
            })
        );
        assert!(!exists);
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-runtime-service-{label}-{nanos}"))
    }
}
