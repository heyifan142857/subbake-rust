use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use tokio::runtime::Runtime;

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
        WhisperAction::Install => {
            install_binary(&request)?;
            Ok(WhisperOutcome::Status(inspect_status(&request)))
        }
        WhisperAction::DownloadModel { ref name } => {
            download_model(&request, name)?;
            // Re-list models so the caller can see the new file.
            Ok(WhisperOutcome::ModelList(list_models(&request)?))
        }
    }
}

// ---------------------------------------------------------------------------
// Binary install: prebuilt download from GitHub Releases → cmake fallback
// ---------------------------------------------------------------------------

fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| Runtime::new().unwrap_or_else(|_| panic!("unable to start whisper runtime")))
}

const GITHUB_REPO_OWNER: &str = "ggerganov";
const GITHUB_REPO_NAME: &str = "whisper.cpp";
const GITHUB_RELEASES_API: &str = "https://api.github.com/repos/ggerganov/whisper.cpp/releases";
const HF_MODEL_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

struct PlatformAssets {
    archive_name: String,
    binary_name: &'static str,
}

fn detect_platform() -> Option<PlatformAssets> {
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;
    match (os, arch) {
        ("linux", "x86_64") => Some(PlatformAssets {
            archive_name: format!(
                "whisper-cli-{tag}-linux-x64.tar.gz",
                tag = "{tag}"
            ),
            binary_name: "whisper-cli",
        }),
        ("macos", "aarch64") => Some(PlatformAssets {
            archive_name: format!(
                "whisper-cli-{tag}-macos-arm64.tar.gz",
                tag = "{tag}"
            ),
            binary_name: "whisper-cli",
        }),
        ("macos", "x86_64") => Some(PlatformAssets {
            archive_name: format!(
                "whisper-cli-{tag}-macos-x64.tar.gz",
                tag = "{tag}"
            ),
            binary_name: "whisper-cli",
        }),
        ("windows", "x86_64") => Some(PlatformAssets {
            archive_name: format!(
                "whisper-cli-{tag}-win-x64.zip",
                tag = "{tag}"
            ),
            binary_name: "whisper-cli.exe",
        }),
        _ => None,
    }
}

fn install_binary(request: &WhisperRequest) -> io::Result<()> {
    let bin_dir = request
        .binary_path
        .clone()
        .map(|p| p.parent().unwrap_or_else(|| Path::new(".subbake/whisper/bin")).to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".subbake/whisper/bin"));
    let version_tag = "latest";
    let target_dir = bin_dir.clone();

    if let Some(platform) = detect_platform() {
        let archive_name = platform
            .archive_name
            .replace("{tag}", version_tag);
        let url = format!("{GITHUB_RELEASES_API}/{version_tag}/assets/{archive_name}");
        let download_dir = std::env::temp_dir().join("subbake-whisper-install");
        std::fs::create_dir_all(&download_dir)
            .map_err(|e| io::Error::other(format!("create download dir: {e}")))?;
        let archive_path = download_dir.join(&archive_name);

        // Download the archive using reqwest.
        runtime().block_on(async {
            download_file(&url, &archive_path).await
        })?;

        // Extract archive.
        if archive_name.ends_with(".tar.gz") {
            extract_tar_gz(&archive_path, &target_dir)?;
        } else if archive_name.ends_with(".zip") {
            extract_zip_system(&archive_path, &target_dir)?;
        }

        // Promote binary to final location.
        promote_binary(&download_dir, &target_dir, platform.binary_name)?;
        write_version_file(&target_dir, version_tag)?;
        let _ = std::fs::remove_dir_all(&download_dir);
        return Ok(());
    }

    // Fallback: cmake from source
    build_from_source(&target_dir, version_tag)
}

async fn download_file(url: &str, dest: &Path) -> io::Result<()> {
    let client = reqwest::Client::builder()
        .user_agent("subbake/0.1")
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| io::Error::other(format!("http client: {e}")))?;

    let response = client.get(url).send().await
        .map_err(|e| io::Error::other(format!("download request: {e}")))?;
    let status = response.status();
    let bytes = response.bytes().await
        .map_err(|e| io::Error::other(format!("download response: {e}")))?;

    if !status.is_success() {
        return Err(io::Error::other(format!("download failed ({status}) from {url}")));
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| io::Error::other(format!("create dest dir: {e}")))?;
    }
    std::fs::write(dest, &bytes)
        .map_err(|e| io::Error::other(format!("write download: {e}")))?;
    Ok(())
}

fn extract_tar_gz(archive: &Path, target: &Path) -> io::Result<()> {
    let status = Command::new("tar")
        .args(["-xzf", &archive.to_string_lossy(), "-C", &target.to_string_lossy()])
        .status()
        .map_err(|e| io::Error::other(format!("tar not found: {e}")))?;
    if !status.success() {
        return Err(io::Error::other("tar extraction failed"));
    }
    Ok(())
}

fn extract_zip_system(archive: &Path, target: &Path) -> io::Result<()> {
    let status = Command::new("unzip")
        .args(["-o", &archive.to_string_lossy(), "-d", &target.to_string_lossy()])
        .status()
        .map_err(|e| io::Error::other(format!("unzip not found: {e}")))?;
    if !status.success() {
        return Err(io::Error::other("unzip extraction failed"));
    }
    Ok(())
}

fn promote_binary(download_dir: &Path, target_dir: &Path, binary_name: &str) -> io::Result<()> {
    // Search for the binary recursively in the extraction directory.
    for entry in walk_dir(download_dir) {
        if entry
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n == binary_name || n == "main" || n == "whisper-cli")
        {
            let dest = target_dir.join("whisper-cli");
            std::fs::copy(&entry, &dest)
                .map_err(|e| io::Error::other(format!("copy binary: {e}")))?;
            // chmod +x
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755))
                    .map_err(|e| io::Error::other(format!("chmod binary: {e}")))?;
            }
            return Ok(());
        }
    }
    Err(io::Error::other("could not find extracted binary"))
}

fn walk_dir(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(walk_dir(&path));
            } else {
                files.push(path);
            }
        }
    }
    files
}

fn write_version_file(target_dir: &Path, version: &str) -> io::Result<()> {
    let path = target_dir.join("version.txt");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| io::Error::other(format!("create dir: {e}")))?;
    }
    std::fs::write(&path, version)
        .map_err(|e| io::Error::other(format!("write version: {e}")))?;
    Ok(())
}

fn build_from_source(target_dir: &Path, _version: &str) -> io::Result<()> {
    // cmake source build: clone/download source, cmake + make.
    let build_dir = std::env::temp_dir().join("subbake-whisper-build");
    let src_dir = build_dir.join("source");
    let build_out_dir = build_dir.join("build");

    std::fs::create_dir_all(&src_dir)
        .map_err(|e| io::Error::other(format!("create src dir: {e}")))?;

    // Download source tarball from GitHub.
    let tarball_url = format!(
        "https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/tarball/HEAD"
    );
    let tarball_path = build_dir.join("source.tar.gz");
    runtime().block_on(async {
        download_file(&tarball_url, &tarball_path).await
    })?;

    // Extract with tar.
    std::fs::create_dir_all(&src_dir)
        .map_err(|e| io::Error::other(format!("create src: {e}")))?;
    let status = Command::new("tar")
        .args(["-xzf", &tarball_path.to_string_lossy(), "-C", &src_dir.to_string_lossy(), "--strip-components=1"])
        .status()
        .map_err(|e| io::Error::other(format!("tar source: {e}")))?;
    if !status.success() {
        return Err(io::Error::other("source extraction failed"));
    }

    // cmake configure & build
    std::fs::create_dir_all(&build_out_dir)
        .map_err(|e| io::Error::other(format!("create build dir: {e}")))?;

    let cmake = Command::new("cmake")
        .args([
            "-S", &src_dir.to_string_lossy(),
            "-B", &build_out_dir.to_string_lossy(),
            "-DWHISPER_BUILD_TESTS=OFF",
            "-DWHISPER_BUILD_EXAMPLES=ON",
        ])
        .output()
        .map_err(|e| io::Error::other(format!("cmake configure: {e}")))?;
    if !cmake.status.success() {
        let stderr = String::from_utf8_lossy(&cmake.stderr);
        return Err(io::Error::other(format!("cmake failed: {stderr}")));
    }

    let make = Command::new("cmake")
        .args(["--build", &build_out_dir.to_string_lossy(), "--config", "Release", "--target", "whisper-cli", "-j"])
        .arg(num_cpus().to_string())
        .output()
        .map_err(|e| io::Error::other(format!("cmake build: {e}")))?;
    if !make.status.success() {
        let stderr = String::from_utf8_lossy(&make.stderr);
        // Retry with target "main" (older whisper.cpp releases).
        let make2 = Command::new("cmake")
            .args(["--build", &build_out_dir.to_string_lossy(), "--config", "Release", "--target", "main", "-j"])
            .arg(num_cpus().to_string())
            .output()
            .map_err(|e| io::Error::other(format!("cmake build main: {e}")))?;
        if !make2.status.success() {
            let stderr2 = String::from_utf8_lossy(&make2.stderr);
            return Err(io::Error::other(format!("cmake build failed: {stderr} / {stderr2}")));
        }
    }

    // Find and copy the built binary.
    std::fs::create_dir_all(target_dir)
        .map_err(|e| io::Error::other(format!("create target dir: {e}")))?;

    for entry in walk_dir(&build_out_dir) {
        if entry
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n == "whisper-cli" || n == "main" || n == "whisper-cli.exe")
        {
            let dest = target_dir.join("whisper-cli");
            std::fs::copy(&entry, &dest)
                .map_err(|e| io::Error::other(format!("copy built binary: {e}")))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755))
                    .map_err(|e| io::Error::other(format!("chmod binary: {e}")))?;
            }
            let _ = std::fs::remove_dir_all(&build_dir);
            return Ok(());
        }
    }

    let _ = std::fs::remove_dir_all(&build_dir);
    Err(io::Error::other("built binary not found after cmake build"))
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

// ---------------------------------------------------------------------------
// Model download from HuggingFace
// ---------------------------------------------------------------------------

pub const SUPPORTED_MODELS: &[&str] = &["tiny", "base", "small", "medium", "large-v3"];

fn download_model(request: &WhisperRequest, name: &str) -> io::Result<()> {
    if !SUPPORTED_MODELS.contains(&name) {
        return Err(io::Error::other(format!(
            "unknown model `{name}`; supported: {}",
            SUPPORTED_MODELS.join(", ")
        )));
    }

    let models_dir = request
        .models_dir
        .clone()
        .unwrap_or_else(default_whisper_models_dir);
    std::fs::create_dir_all(&models_dir)
        .map_err(|e| io::Error::other(format!("create models dir: {e}")))?;

    let dest = models_dir.join(format!("ggml-{name}.bin"));
    if dest.exists() {
        return Ok(()); // Already downloaded.
    }

    let url = format!("{HF_MODEL_BASE}/ggml-{name}.bin");
    runtime().block_on(async {
        download_file(&url, &dest).await
    })?;

    Ok(())
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
    fn install_attempts_download() {
        let error = run_whisper(WhisperRequest {
            action: WhisperAction::Install,
            binary_path: None,
            models_dir: None,
        })
        .expect_err("install should attempt download");

        let msg = error.to_string();
        // The old stub said "pending adapter migration".
        assert!(!msg.contains("pending"), "should no longer be a stub: {msg}");
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
    fn model_download_attempts_download() {
        // Use an unsupported model name to avoid hitting a real file on disk.
        let error = run_whisper(WhisperRequest {
            action: WhisperAction::DownloadModel {
                name: "unknown-model-test".to_owned(),
            },
            binary_path: None,
            models_dir: None,
        })
        .expect_err("model download should reject unknown names");

        let msg = error.to_string();
        // The old stub said "pending adapter migration".
        assert!(!msg.contains("pending"), "should no longer be a stub: {msg}");
        assert!(msg.contains("unknown model"));
    }

    #[test]
    fn model_download_succeeds_when_file_exists() {
        let root = temp_root("exists");
        let models_dir = root.join("models");
        std::fs::create_dir_all(&models_dir).expect("create models dir");
        std::fs::write(models_dir.join("ggml-base.bin"), b"fake").expect("write fake model");

        let outcome = run_whisper(WhisperRequest {
            action: WhisperAction::DownloadModel {
                name: "base".to_owned(),
            },
            binary_path: None,
            models_dir: Some(models_dir),
        })
        .expect("existing file should succeed");
        let _ = std::fs::remove_dir_all(&root);

        assert!(matches!(outcome, WhisperOutcome::ModelList(_)));
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-whisper-{label}-{nanos}"))
    }
}
