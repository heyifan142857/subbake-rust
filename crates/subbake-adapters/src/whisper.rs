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
    DownloadModel { name: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhisperOutcome {
    Status(WhisperStatus),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperStatus {
    pub binary_path: PathBuf,
    pub binary_exists: bool,
    pub models_dir: PathBuf,
    pub models_dir_exists: bool,
}

pub fn run_whisper(request: WhisperRequest) -> io::Result<WhisperOutcome> {
    match request.action {
        WhisperAction::Status => Ok(WhisperOutcome::Status(inspect_status(&request))),
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

fn default_whisper_binary_path() -> PathBuf {
    PathBuf::from(".subbake/whisper/bin/whisper-cli")
}

fn default_whisper_models_dir() -> PathBuf {
    PathBuf::from(".subbake/whisper/models")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_uses_configured_paths() {
        let outcome = run_whisper(WhisperRequest {
            action: WhisperAction::Status,
            binary_path: Some(PathBuf::from("/tmp/whisper-cli")),
            models_dir: Some(PathBuf::from("/tmp/models")),
        })
        .expect("status should not require installed whisper");

        let WhisperOutcome::Status(status) = outcome;
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
}
