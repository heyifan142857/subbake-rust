use std::fs;
use std::io;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperRequest {
    pub action: WhisperAction,
    pub binary_path: Option<PathBuf>,
    pub models_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhisperAction {
    Status,
    Install,
    ListModels,
    DownloadModel { name: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhisperOutcome {
    Status(WhisperStatus),
    ModelList(WhisperModelList),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperStatus {
    pub binary_path: PathBuf,
    pub binary_exists: bool,
    pub models_dir: PathBuf,
    pub models_dir_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperModelList {
    pub models_dir: PathBuf,
    pub models_dir_exists: bool,
    pub models: Vec<WhisperModel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperModel {
    pub name: String,
    pub path: PathBuf,
}

pub fn run_whisper(request: WhisperRequest) -> io::Result<WhisperOutcome> {
    match request.action {
        WhisperAction::Status => Ok(WhisperOutcome::Status(inspect_status(&request))),
        WhisperAction::ListModels => Ok(WhisperOutcome::ModelList(list_models(&request)?)),
        WhisperAction::Install => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "whisper install is pending adapter migration",
        )),
        WhisperAction::DownloadModel { name } => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("whisper model download is pending adapter migration for `{name}`"),
        )),
    }
}

fn list_models(request: &WhisperRequest) -> io::Result<WhisperModelList> {
    let models_dir = request
        .models_dir
        .clone()
        .unwrap_or_else(default_whisper_models_dir);
    if !models_dir.is_dir() {
        return Ok(WhisperModelList {
            models_dir,
            models_dir_exists: false,
            models: Vec::new(),
        });
    }

    let mut models = Vec::new();
    for entry in fs::read_dir(&models_dir)? {
        let path = entry?.path();
        if !path.is_file() || !is_whisper_model_file(&path) {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("model")
            .to_owned();
        models.push(WhisperModel { name, path });
    }
    models.sort_by(|left, right| left.path.cmp(&right.path));

    Ok(WhisperModelList {
        models_dir,
        models_dir_exists: true,
        models,
    })
}

fn inspect_status(request: &WhisperRequest) -> WhisperStatus {
    let binary_path = request
        .binary_path
        .clone()
        .unwrap_or_else(default_whisper_binary_path);
    let models_dir = request
        .models_dir
        .clone()
        .unwrap_or_else(default_whisper_models_dir);

    WhisperStatus {
        binary_exists: binary_path.is_file(),
        models_dir_exists: models_dir.is_dir(),
        binary_path,
        models_dir,
    }
}

fn is_whisper_model_file(path: &std::path::Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()),
        Some("bin" | "gguf")
    )
}

fn default_whisper_binary_path() -> PathBuf {
    PathBuf::from(".subbake/whisper/bin/whisper-cli")
}

fn default_whisper_models_dir() -> PathBuf {
    PathBuf::from(".subbake/whisper/models")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn status_uses_configured_paths() {
        let outcome = run_whisper(WhisperRequest {
            action: WhisperAction::Status,
            binary_path: Some(PathBuf::from("/tmp/whisper-cli")),
            models_dir: Some(PathBuf::from("/tmp/models")),
        })
        .expect("status should not require installed whisper");

        let WhisperOutcome::Status(status) = outcome else {
            panic!("expected status");
        };
        assert_eq!(status.binary_path, PathBuf::from("/tmp/whisper-cli"));
        assert_eq!(status.models_dir, PathBuf::from("/tmp/models"));
    }

    #[test]
    fn install_reports_pending_backend() {
        let error = run_whisper(WhisperRequest {
            action: WhisperAction::Install,
            binary_path: None,
            models_dir: None,
        })
        .expect_err("install should be pending");

        assert_eq!(error.kind(), io::ErrorKind::Unsupported);
    }

    #[test]
    fn list_models_returns_sorted_model_files() {
        let root = temp_root("models");
        let models_dir = root.join("models");
        fs::create_dir_all(&models_dir).expect("create models dir");
        fs::write(models_dir.join("ggml-small.bin"), b"model").expect("write model");
        fs::write(models_dir.join("ggml-base.gguf"), b"model").expect("write model");
        fs::write(models_dir.join("notes.txt"), b"ignore").expect("write note");

        let outcome = run_whisper(WhisperRequest {
            action: WhisperAction::ListModels,
            binary_path: None,
            models_dir: Some(models_dir),
        })
        .expect("list models");
        let _ = fs::remove_dir_all(&root);

        let WhisperOutcome::ModelList(list) = outcome else {
            panic!("expected model list");
        };
        assert!(list.models_dir_exists);
        assert_eq!(
            list.models
                .iter()
                .map(|model| model.name.as_str())
                .collect::<Vec<_>>(),
            vec!["ggml-base", "ggml-small"]
        );
    }

    #[test]
    fn model_download_reports_pending_backend() {
        let error = run_whisper(WhisperRequest {
            action: WhisperAction::DownloadModel {
                name: "base".to_owned(),
            },
            binary_path: None,
            models_dir: None,
        })
        .expect_err("model download should be pending");

        assert!(error.to_string().contains("base"));
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-whisper-{label}-{nanos}"))
    }
}
