use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use subbake_core::{
    CancellationGuard, NoopProgress, ProgressEvent, ProgressUnit, SharedProgress, TaskKind,
    TaskState,
};
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
    Update,
    Uninstall { keep_models: bool },
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
    run_whisper_cancellable(request, &CancellationGuard::never())
}

pub fn run_whisper_cancellable(
    request: WhisperRequest,
    cancellation: &CancellationGuard,
) -> io::Result<WhisperOutcome> {
    run_whisper_cancellable_with_progress(request, cancellation, std::sync::Arc::new(NoopProgress))
}

pub fn run_whisper_cancellable_with_progress(
    request: WhisperRequest,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
) -> io::Result<WhisperOutcome> {
    cancellation
        .check()
        .map_err(|_| io::Error::new(io::ErrorKind::Interrupted, "operation cancelled"))?;
    let stage = match request.action {
        WhisperAction::Install | WhisperAction::Update => "INSTALL",
        WhisperAction::DownloadModel { .. } => "DOWNLOAD",
        WhisperAction::Uninstall { .. } => "UNINSTALL",
        _ => "INSPECT",
    };
    progress.emit(ProgressEvent::running(
        TaskKind::Installation,
        stage,
        0,
        None,
        ProgressUnit::Steps,
    ));
    let outcome: io::Result<WhisperOutcome> = match request.action {
        WhisperAction::Status => Ok(WhisperOutcome::Status(inspect_status(&request))),
        WhisperAction::ListModels => Ok(WhisperOutcome::ModelList(list_models(&request)?)),
        WhisperAction::Install => {
            install_binary(&request)?;
            Ok(WhisperOutcome::Status(inspect_status(&request)))
        }
        WhisperAction::Update => {
            install_binary(&request)?;
            Ok(WhisperOutcome::Status(inspect_status(&request)))
        }
        WhisperAction::Uninstall { keep_models } => {
            uninstall_whisper(&request, keep_models)?;
            Ok(WhisperOutcome::Status(inspect_status(&request)))
        }
        WhisperAction::DownloadModel { ref name } => {
            download_model(&request, name)?;
            // Re-list models so the caller can see the new file.
            Ok(WhisperOutcome::ModelList(list_models(&request)?))
        }
    };
    let outcome = outcome?;
    let mut done = ProgressEvent::running(
        TaskKind::Installation,
        "COMPLETE",
        1,
        Some(1),
        ProgressUnit::Steps,
    );
    done.state = TaskState::Completed;
    progress.emit(done);
    Ok(outcome)
}

// ---------------------------------------------------------------------------
// Binary install: prebuilt download from GitHub Releases → cmake fallback
// ---------------------------------------------------------------------------

fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        Runtime::new().unwrap_or_else(|_| panic!("unable to start whisper runtime"))
    })
}

const GITHUB_REPO_OWNER: &str = "ggml-org";
const GITHUB_REPO_NAME: &str = "whisper.cpp";
const GITHUB_RELEASES_API: &str = "https://api.github.com/repos/ggml-org/whisper.cpp/releases";
const HF_MODEL_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";
const INSTALL_MANIFEST_NAME: &str = "install-manifest.json";

#[derive(Debug, Serialize, Deserialize)]
struct InstallManifest {
    version: u32,
    files: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReleasePlatform {
    Linux,
    Windows,
}

struct PlatformAssets {
    release_platform: ReleasePlatform,
    arch_terms: &'static [&'static str],
    executable_names: &'static [&'static str],
}

fn detect_platform() -> Option<PlatformAssets> {
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;
    match (os, arch) {
        ("linux", "x86_64") => Some(PlatformAssets {
            release_platform: ReleasePlatform::Linux,
            arch_terms: &["x64", "x86_64", "amd64"],
            executable_names: &["whisper-whisper-cli", "whisper-cli", "main"],
        }),
        ("linux", "aarch64") => Some(PlatformAssets {
            release_platform: ReleasePlatform::Linux,
            arch_terms: &["arm64", "aarch64"],
            executable_names: &["whisper-whisper-cli", "whisper-cli", "main"],
        }),
        ("windows", "x86_64") => Some(PlatformAssets {
            release_platform: ReleasePlatform::Windows,
            arch_terms: &["x64", "x86_64", "amd64"],
            executable_names: &["whisper-whisper-cli.exe", "whisper-cli.exe", "main.exe"],
        }),
        _ => None,
    }
}

fn install_binary(request: &WhisperRequest) -> io::Result<()> {
    let binary_path = whisper_binary_path(request);
    let bin_dir = binary_path
        .parent()
        .unwrap_or_else(|| Path::new(".subbake/whisper/bin"))
        .to_path_buf();
    let version_tag = "latest";
    let target_dir = bin_dir.clone();
    std::fs::create_dir_all(&target_dir)
        .map_err(|e| io::Error::other(format!("create binary dir: {e}")))?;

    if let Some(platform) = detect_platform()
        && let Some(asset) =
            runtime().block_on(async { find_latest_release_asset(&platform).await })?
    {
        let download_dir = std::env::temp_dir().join("subbake-whisper-install");
        let extract_dir = download_dir.join("extract");
        std::fs::create_dir_all(&download_dir)
            .map_err(|e| io::Error::other(format!("create download dir: {e}")))?;
        std::fs::create_dir_all(&extract_dir)
            .map_err(|e| io::Error::other(format!("create extract dir: {e}")))?;
        let archive_path = download_dir.join(&asset.name);

        // Download the archive using reqwest.
        runtime().block_on(async { download_file(&asset.url, &archive_path).await })?;

        // Extract archive.
        if asset.name.ends_with(".tar.gz") || asset.name.ends_with(".tgz") {
            extract_tar_gz(&archive_path, &extract_dir)?;
        } else if asset.name.ends_with(".zip") {
            extract_zip_system(&archive_path, &extract_dir)?;
        } else {
            let direct_path = extract_dir.join(platform.executable_names[0]);
            std::fs::copy(&archive_path, &direct_path)
                .map_err(|e| io::Error::other(format!("copy direct binary: {e}")))?;
        }

        // Promote binary to final location.
        remove_managed_install_files(&target_dir)?;
        let runtime_libraries = promote_runtime_libraries(&extract_dir, &target_dir)?;
        promote_binary(&extract_dir, &binary_path, platform.executable_names)?;
        write_version_file(&target_dir, &asset.tag)?;
        write_install_manifest(&target_dir, &binary_path, runtime_libraries)?;
        let _ = std::fs::remove_dir_all(&download_dir);
        return Ok(());
    }

    // Fallback: cmake from source
    build_from_source(&target_dir, &binary_path, version_tag)
}

struct ReleaseAsset {
    name: String,
    url: String,
    tag: String,
}

async fn find_latest_release_asset(platform: &PlatformAssets) -> io::Result<Option<ReleaseAsset>> {
    let client = reqwest::Client::builder()
        .user_agent("subbake/0.1")
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| io::Error::other(format!("http client: {e}")))?;

    let response = client
        .get(format!("{GITHUB_RELEASES_API}/latest"))
        .send()
        .await
        .map_err(|e| io::Error::other(format!("release lookup request: {e}")))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| io::Error::other(format!("release lookup response: {e}")))?;
    if !status.is_success() {
        return Err(io::Error::other(format!(
            "release lookup failed ({status}): {body}"
        )));
    }

    let payload: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| io::Error::other(format!("release metadata parse: {e}")))?;
    let tag = payload["tag_name"].as_str().unwrap_or("latest").to_owned();
    let assets = payload["assets"]
        .as_array()
        .ok_or_else(|| io::Error::other("release metadata does not contain an assets array"))?;

    Ok(assets
        .iter()
        .filter_map(|asset| {
            let name = asset["name"].as_str()?;
            let url = asset["browser_download_url"].as_str()?;
            let score = asset_match_score(name, platform)?;
            Some((score, name.to_owned(), url.to_owned()))
        })
        .max_by_key(|(score, _, _)| *score)
        .map(|(_, name, url)| ReleaseAsset { name, url, tag }))
}

fn asset_match_score(name: &str, platform: &PlatformAssets) -> Option<usize> {
    let lower = name.to_lowercase();
    let supported_archive =
        lower.ends_with(".tar.gz") || lower.ends_with(".tgz") || lower.ends_with(".zip");
    let direct_binary = platform
        .executable_names
        .iter()
        .any(|binary_name| lower.ends_with(binary_name));
    if !supported_archive && !direct_binary {
        return None;
    }
    if !lower.contains("whisper-bin") {
        return None;
    }
    if !platform.arch_terms.iter().any(|term| lower.contains(term)) {
        return None;
    }
    if ["blas", "cublas", "cuda", "metal", "vulkan", "opencl"]
        .iter()
        .any(|term| lower.contains(term))
    {
        return None;
    }
    match platform.release_platform {
        ReleasePlatform::Linux
            if !["ubuntu", "linux", "debian", "manylinux"]
                .iter()
                .any(|term| lower.contains(term)) =>
        {
            None
        }
        ReleasePlatform::Windows
            if ["ubuntu", "linux", "debian", "macos", "darwin", "osx"]
                .iter()
                .any(|term| lower.contains(term)) =>
        {
            None
        }
        _ => Some(1),
    }
}

async fn download_file(url: &str, dest: &Path) -> io::Result<()> {
    let client = reqwest::Client::builder()
        .user_agent("subbake/0.1")
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| io::Error::other(format!("http client: {e}")))?;

    let mut response = client
        .get(url)
        .send()
        .await
        .map_err(|e| io::Error::other(format!("download request: {e}")))?;
    let status = response.status();

    if !status.is_success() {
        return Err(io::Error::other(format!(
            "download failed ({status}) from {url}"
        )));
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| io::Error::other(format!("create dest dir: {e}")))?;
    }
    let tmp = dest.with_extension("download.tmp");
    let mut file = std::fs::File::create(&tmp)
        .map_err(|e| io::Error::other(format!("create download file: {e}")))?;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| io::Error::other(format!("download response: {e}")))?
    {
        file.write_all(&chunk)
            .map_err(|e| io::Error::other(format!("write download: {e}")))?;
    }
    file.flush()
        .map_err(|e| io::Error::other(format!("flush download: {e}")))?;
    std::fs::rename(&tmp, dest).map_err(|e| io::Error::other(format!("finalize download: {e}")))?;
    Ok(())
}

fn extract_tar_gz(archive: &Path, target: &Path) -> io::Result<()> {
    std::fs::create_dir_all(target)
        .map_err(|e| io::Error::other(format!("create extract target: {e}")))?;
    let status = Command::new("tar")
        .args([
            "-xzf",
            &archive.to_string_lossy(),
            "-C",
            &target.to_string_lossy(),
        ])
        .status()
        .map_err(|e| io::Error::other(format!("tar not found: {e}")))?;
    if !status.success() {
        return Err(io::Error::other("tar extraction failed"));
    }
    Ok(())
}

fn extract_zip_system(archive: &Path, target: &Path) -> io::Result<()> {
    std::fs::create_dir_all(target)
        .map_err(|e| io::Error::other(format!("create extract target: {e}")))?;
    #[cfg(windows)]
    let mut command = {
        let mut command = Command::new("tar");
        command.args([
            "-xf",
            &archive.to_string_lossy(),
            "-C",
            &target.to_string_lossy(),
        ]);
        command
    };
    #[cfg(not(windows))]
    let mut command = {
        let mut command = Command::new("unzip");
        command.args([
            "-o",
            &archive.to_string_lossy(),
            "-d",
            &target.to_string_lossy(),
        ]);
        command
    };
    let status = command
        .status()
        .map_err(|e| io::Error::other(format!("zip extractor not found: {e}")))?;
    if !status.success() {
        return Err(io::Error::other("unzip extraction failed"));
    }
    Ok(())
}

fn promote_binary(
    download_dir: &Path,
    destination: &Path,
    executable_names: &[&str],
) -> io::Result<()> {
    // Search for the binary recursively in the extraction directory.
    let extracted_files = walk_dir(download_dir);
    for executable_name in executable_names {
        if let Some(entry) = extracted_files.iter().find(|entry| {
            entry
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == *executable_name)
        }) {
            std::fs::copy(entry, destination)
                .map_err(|e| io::Error::other(format!("copy binary: {e}")))?;
            // chmod +x
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(destination, std::fs::Permissions::from_mode(0o755))
                    .map_err(|e| io::Error::other(format!("chmod binary: {e}")))?;
            }
            return Ok(());
        }
    }
    Err(io::Error::other("could not find extracted binary"))
}

fn promote_runtime_libraries(source_dir: &Path, target_dir: &Path) -> io::Result<Vec<String>> {
    let mut installed = Vec::new();
    for entry in walk_dir(source_dir) {
        let Some(name) = entry.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !is_runtime_library(name) {
            continue;
        }
        std::fs::copy(&entry, target_dir.join(name))
            .map_err(|e| io::Error::other(format!("copy runtime library `{name}`: {e}")))?;
        installed.push(name.to_owned());
    }
    installed.sort();
    installed.dedup();
    Ok(installed)
}

fn is_runtime_library(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains(".so") || lower.ends_with(".dylib") || lower.ends_with(".dll")
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
    std::fs::write(&path, version).map_err(|e| io::Error::other(format!("write version: {e}")))?;
    Ok(())
}

fn write_install_manifest(
    target_dir: &Path,
    binary_path: &Path,
    mut runtime_libraries: Vec<String>,
) -> io::Result<()> {
    let binary_name = binary_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::other("whisper binary path has no valid file name"))?;
    runtime_libraries.push(binary_name.to_owned());
    runtime_libraries.push("version.txt".to_owned());
    runtime_libraries.sort();
    runtime_libraries.dedup();
    let manifest = InstallManifest {
        version: 1,
        files: runtime_libraries,
    };
    let content = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| io::Error::other(format!("serialize install manifest: {e}")))?;
    std::fs::write(target_dir.join(INSTALL_MANIFEST_NAME), content)
        .map_err(|e| io::Error::other(format!("write install manifest: {e}")))
}

fn build_from_source(target_dir: &Path, binary_path: &Path, version: &str) -> io::Result<()> {
    // cmake source build: clone/download source, cmake + make.
    let build_dir = std::env::temp_dir().join("subbake-whisper-build");
    let src_dir = build_dir.join("source");
    let build_out_dir = build_dir.join("build");

    std::fs::create_dir_all(&src_dir)
        .map_err(|e| io::Error::other(format!("create src dir: {e}")))?;

    // Download source tarball from GitHub.
    let tarball_url =
        format!("https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/tarball/HEAD");
    let tarball_path = build_dir.join("source.tar.gz");
    runtime().block_on(async { download_file(&tarball_url, &tarball_path).await })?;

    // Extract with tar.
    std::fs::create_dir_all(&src_dir).map_err(|e| io::Error::other(format!("create src: {e}")))?;
    let status = Command::new("tar")
        .args([
            "-xzf",
            &tarball_path.to_string_lossy(),
            "-C",
            &src_dir.to_string_lossy(),
            "--strip-components=1",
        ])
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
            "-S",
            &src_dir.to_string_lossy(),
            "-B",
            &build_out_dir.to_string_lossy(),
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
        .args([
            "--build",
            &build_out_dir.to_string_lossy(),
            "--config",
            "Release",
            "--target",
            "whisper-cli",
            "-j",
        ])
        .arg(num_cpus().to_string())
        .output()
        .map_err(|e| io::Error::other(format!("cmake build: {e}")))?;
    if !make.status.success() {
        let stderr = String::from_utf8_lossy(&make.stderr);
        // Retry with target "main" (older whisper.cpp releases).
        let make2 = Command::new("cmake")
            .args([
                "--build",
                &build_out_dir.to_string_lossy(),
                "--config",
                "Release",
                "--target",
                "main",
                "-j",
            ])
            .arg(num_cpus().to_string())
            .output()
            .map_err(|e| io::Error::other(format!("cmake build main: {e}")))?;
        if !make2.status.success() {
            let stderr2 = String::from_utf8_lossy(&make2.stderr);
            return Err(io::Error::other(format!(
                "cmake build failed: {stderr} / {stderr2}"
            )));
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
            remove_managed_install_files(target_dir)?;
            let runtime_libraries = promote_runtime_libraries(&build_out_dir, target_dir)?;
            std::fs::copy(&entry, binary_path)
                .map_err(|e| io::Error::other(format!("copy built binary: {e}")))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(binary_path, std::fs::Permissions::from_mode(0o755))
                    .map_err(|e| io::Error::other(format!("chmod binary: {e}")))?;
            }
            let _ = std::fs::remove_dir_all(&build_dir);
            write_version_file(target_dir, version)?;
            write_install_manifest(target_dir, binary_path, runtime_libraries)?;
            return Ok(());
        }
    }

    let _ = std::fs::remove_dir_all(&build_dir);
    Err(io::Error::other("built binary not found after cmake build"))
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
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
    runtime().block_on(async { download_file(&url, &dest).await })?;

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

fn uninstall_whisper(request: &WhisperRequest, keep_models: bool) -> io::Result<()> {
    let binary_path = request
        .binary_path
        .clone()
        .unwrap_or_else(default_whisper_binary_path);
    if binary_path.exists() {
        fs::remove_file(&binary_path)
            .map_err(|error| io::Error::other(format!("remove whisper binary: {error}")))?;
    }
    if let Some(parent) = binary_path.parent() {
        remove_managed_install_files(parent)?;
    }
    if let Some(parent) = binary_path.parent()
        && parent.is_dir()
        && fs::read_dir(parent)?.next().is_none()
    {
        let _ = fs::remove_dir(parent);
    }

    if !keep_models {
        let models_dir = request
            .models_dir
            .clone()
            .unwrap_or_else(default_whisper_models_dir);
        if models_dir.exists() {
            fs::remove_dir_all(&models_dir)
                .map_err(|error| io::Error::other(format!("remove models dir: {error}")))?;
        }
    }

    Ok(())
}

fn remove_managed_install_files(bin_dir: &Path) -> io::Result<()> {
    let manifest_path = bin_dir.join(INSTALL_MANIFEST_NAME);
    if manifest_path.is_file() {
        let content = fs::read(&manifest_path)
            .map_err(|error| io::Error::other(format!("read install manifest: {error}")))?;
        let manifest: InstallManifest = serde_json::from_slice(&content)
            .map_err(|error| io::Error::other(format!("parse install manifest: {error}")))?;
        for name in manifest.files {
            if Path::new(&name)
                .file_name()
                .and_then(|value| value.to_str())
                != Some(name.as_str())
            {
                return Err(io::Error::other(format!(
                    "invalid file name in install manifest: {name}"
                )));
            }
            let path = bin_dir.join(name);
            if path.is_file() {
                fs::remove_file(&path).map_err(|error| {
                    io::Error::other(format!("remove managed whisper file: {error}"))
                })?;
            }
        }
        fs::remove_file(&manifest_path)
            .map_err(|error| io::Error::other(format!("remove install manifest: {error}")))?;
    } else {
        let version_path = bin_dir.join("version.txt");
        if version_path.is_file() {
            fs::remove_file(version_path)
                .map_err(|error| io::Error::other(format!("remove version file: {error}")))?;
        }
    }
    Ok(())
}

fn is_whisper_model_file(path: &std::path::Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()),
        Some("bin" | "gguf")
    )
}

pub fn default_whisper_binary_path() -> PathBuf {
    PathBuf::from(".subbake/whisper/bin").join(default_whisper_binary_name())
}

const fn default_whisper_binary_name() -> &'static str {
    if cfg!(windows) {
        "whisper-cli.exe"
    } else {
        "whisper-cli"
    }
}

fn whisper_binary_path(request: &WhisperRequest) -> PathBuf {
    request
        .binary_path
        .clone()
        .unwrap_or_else(default_whisper_binary_path)
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
    fn current_release_asset_names_select_cpu_builds() {
        let linux_x64 = PlatformAssets {
            release_platform: ReleasePlatform::Linux,
            arch_terms: &["x64", "x86_64", "amd64"],
            executable_names: &["whisper-whisper-cli", "whisper-cli", "main"],
        };
        let linux_arm64 = PlatformAssets {
            release_platform: ReleasePlatform::Linux,
            arch_terms: &["arm64", "aarch64"],
            executable_names: &["whisper-whisper-cli", "whisper-cli", "main"],
        };
        let windows_x64 = PlatformAssets {
            release_platform: ReleasePlatform::Windows,
            arch_terms: &["x64", "x86_64", "amd64"],
            executable_names: &["whisper-whisper-cli.exe", "whisper-cli.exe", "main.exe"],
        };

        assert!(asset_match_score("whisper-bin-ubuntu-x64.tar.gz", &linux_x64).is_some());
        assert!(asset_match_score("whisper-bin-ubuntu-arm64.tar.gz", &linux_arm64).is_some());
        assert!(asset_match_score("whisper-bin-x64.zip", &windows_x64).is_some());
        assert!(asset_match_score("whisper-blas-bin-x64.zip", &windows_x64).is_none());
        assert!(asset_match_score("whisper-cublas-12.4.0-bin-x64.zip", &windows_x64).is_none());
        assert!(asset_match_score("whisper-bin-ubuntu-x64.tar.gz", &windows_x64).is_none());
    }

    #[test]
    fn promote_binary_preserves_requested_destination() {
        let root = temp_root("promote");
        let extracted = root.join("extract").join("bin");
        let destination = root.join("custom").join("whisper");
        fs::create_dir_all(&extracted).expect("create extract dir");
        fs::create_dir_all(destination.parent().expect("destination parent"))
            .expect("create destination dir");
        fs::write(extracted.join("whisper-cli"), b"deprecated").expect("write fallback binary");
        fs::write(extracted.join("whisper-whisper-cli"), b"current").expect("write binary");

        promote_binary(
            &root.join("extract"),
            &destination,
            &["whisper-whisper-cli", "whisper-cli", "main"],
        )
        .expect("promote binary");
        let content = fs::read(&destination).expect("read promoted binary");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(content, b"current");
    }

    #[test]
    fn promote_runtime_libraries_copies_shared_objects_only() {
        let root = temp_root("libraries");
        let extracted = root.join("extract").join("lib");
        let destination = root.join("destination");
        fs::create_dir_all(&extracted).expect("create extract dir");
        fs::create_dir_all(&destination).expect("create destination dir");
        fs::write(extracted.join("libwhisper.so.1"), b"linux").expect("write linux library");
        fs::write(extracted.join("whisper.dll"), b"windows").expect("write windows library");
        fs::write(extracted.join("README.txt"), b"ignore").expect("write ignored file");

        let installed = promote_runtime_libraries(&root.join("extract"), &destination)
            .expect("promote runtime libraries");
        let linux_library = fs::read(destination.join("libwhisper.so.1")).expect("read library");
        let windows_library = fs::read(destination.join("whisper.dll")).expect("read library");
        let ignored_exists = destination.join("README.txt").exists();
        let _ = fs::remove_dir_all(&root);

        assert_eq!(linux_library, b"linux");
        assert_eq!(windows_library, b"windows");
        assert!(!ignored_exists);
        assert_eq!(installed, vec!["libwhisper.so.1", "whisper.dll"]);
    }

    #[test]
    fn uninstall_removes_manifest_managed_files_only() {
        let root = temp_root("managed-uninstall");
        let bin_dir = root.join("bin");
        let binary_path = bin_dir.join("whisper-cli");
        fs::create_dir_all(&bin_dir).expect("create bin dir");
        fs::write(&binary_path, b"binary").expect("write binary");
        fs::write(bin_dir.join("libwhisper.so.1"), b"library").expect("write library");
        fs::write(bin_dir.join("keep.txt"), b"user file").expect("write user file");
        write_version_file(&bin_dir, "v1").expect("write version");
        write_install_manifest(&bin_dir, &binary_path, vec!["libwhisper.so.1".to_owned()])
            .expect("write manifest");

        run_whisper(WhisperRequest {
            action: WhisperAction::Uninstall { keep_models: true },
            binary_path: Some(binary_path),
            models_dir: Some(root.join("models")),
        })
        .expect("uninstall whisper");
        let user_file_exists = bin_dir.join("keep.txt").is_file();
        let library_exists = bin_dir.join("libwhisper.so.1").exists();
        let manifest_exists = bin_dir.join(INSTALL_MANIFEST_NAME).exists();
        let _ = fs::remove_dir_all(&root);

        assert!(user_file_exists);
        assert!(!library_exists);
        assert!(!manifest_exists);
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
        assert!(
            !msg.contains("pending"),
            "should no longer be a stub: {msg}"
        );
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

    #[test]
    fn uninstall_removes_binary_and_models() {
        let root = temp_root("uninstall");
        let bin_dir = root.join("bin");
        let models_dir = root.join("models");
        fs::create_dir_all(&bin_dir).expect("create bin dir");
        fs::create_dir_all(&models_dir).expect("create models dir");
        fs::write(bin_dir.join("whisper-cli"), b"fake").expect("write binary");
        fs::write(models_dir.join("ggml-base.bin"), b"fake").expect("write model");

        let outcome = run_whisper(WhisperRequest {
            action: WhisperAction::Uninstall { keep_models: false },
            binary_path: Some(bin_dir.join("whisper-cli")),
            models_dir: Some(models_dir),
        })
        .expect("uninstall whisper");
        let _ = fs::remove_dir_all(&root);

        let WhisperOutcome::Status(status) = outcome else {
            panic!("expected status");
        };
        assert!(!status.binary_exists);
        assert!(!status.models_dir_exists);
    }

    #[test]
    fn uninstall_can_keep_models() {
        let root = temp_root("uninstall-keep");
        let bin_dir = root.join("bin");
        let models_dir = root.join("models");
        fs::create_dir_all(&bin_dir).expect("create bin dir");
        fs::create_dir_all(&models_dir).expect("create models dir");
        fs::write(bin_dir.join("whisper-cli"), b"fake").expect("write binary");
        fs::write(models_dir.join("ggml-base.bin"), b"fake").expect("write model");

        let outcome = run_whisper(WhisperRequest {
            action: WhisperAction::Uninstall { keep_models: true },
            binary_path: Some(bin_dir.join("whisper-cli")),
            models_dir: Some(models_dir.clone()),
        })
        .expect("uninstall whisper");
        let kept = models_dir.exists();
        let _ = fs::remove_dir_all(&root);

        let WhisperOutcome::Status(status) = outcome else {
            panic!("expected status");
        };
        assert!(!status.binary_exists);
        assert!(status.models_dir_exists);
        assert!(kept);
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-whisper-{label}-{nanos}"))
    }
}
