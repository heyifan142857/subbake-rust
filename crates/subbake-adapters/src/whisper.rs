use std::fs;
use std::future::Future;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subbake_core::{
    CancellationGuard, NoopProgress, ProgressEvent, ProgressUnit, SharedProgress, TaskKind,
    TaskState,
};
use tokio::runtime::Runtime;

use crate::error::{AdapterError, AdapterResult};
use crate::process::run_command_cancellable;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperRequest {
    pub action: WhisperAction,
    pub binary_path: Option<PathBuf>,
    pub models_dir: Option<PathBuf>,
    pub build_variant: WhisperBuildVariant,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum WhisperBuildVariant {
    #[default]
    Cpu,
    Cuda,
    Metal,
    Vulkan,
    OpenBlas,
}

impl WhisperBuildVariant {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "cuda" => Some(Self::Cuda),
            "metal" => Some(Self::Metal),
            "vulkan" => Some(Self::Vulkan),
            "openblas" | "blas" => Some(Self::OpenBlas),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhisperAction {
    Status,
    ListVersions,
    Install,
    Update,
    Uninstall { keep_models: bool },
    ListModels,
    DownloadModel { name: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhisperOutcome {
    Status(WhisperStatus),
    VersionList(WhisperVersionList),
    ModelList(WhisperModelList),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperVersionList {
    pub pinned_version: String,
    pub versions: Vec<WhisperVersion>,
    pub refresh_warning: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperVersion {
    pub tag: String,
    pub prerelease: bool,
    pub published_at: Option<String>,
    pub installable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperStatus {
    pub binary_path: PathBuf,
    pub binary_exists: bool,
    pub models_dir: PathBuf,
    pub models_dir_exists: bool,
    pub version: Option<String>,
    pub capability_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperModelList {
    pub models_dir: PathBuf,
    pub models_dir_exists: bool,
    pub models: Vec<WhisperModel>,
    pub available_models: Vec<String>,
    pub refresh_warning: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperModel {
    pub name: String,
    pub path: PathBuf,
}

pub fn run_whisper(request: WhisperRequest) -> AdapterResult<WhisperOutcome> {
    run_whisper_cancellable(request, &CancellationGuard::never())
}

pub fn run_whisper_cancellable(
    request: WhisperRequest,
    cancellation: &CancellationGuard,
) -> AdapterResult<WhisperOutcome> {
    run_whisper_cancellable_with_progress(request, cancellation, std::sync::Arc::new(NoopProgress))
}

pub fn run_whisper_cancellable_with_progress(
    request: WhisperRequest,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
) -> AdapterResult<WhisperOutcome> {
    cancellation.check().map_err(AdapterError::from)?;
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
    let outcome: AdapterResult<WhisperOutcome> = match request.action {
        WhisperAction::Status => Ok(WhisperOutcome::Status(inspect_status(
            &request,
            cancellation,
        )?)),
        WhisperAction::ListVersions => {
            Ok(WhisperOutcome::VersionList(list_versions(cancellation)?))
        }
        WhisperAction::ListModels => {
            let (models, warning) = fetch_available_models(cancellation)?;
            Ok(WhisperOutcome::ModelList(list_models(
                &request,
                Some(models),
                warning,
            )?))
        }
        WhisperAction::Install => {
            install_binary(&request, cancellation, &progress)?;
            Ok(WhisperOutcome::Status(inspect_status(
                &request,
                cancellation,
            )?))
        }
        WhisperAction::Update => {
            install_binary(&request, cancellation, &progress)?;
            Ok(WhisperOutcome::Status(inspect_status(
                &request,
                cancellation,
            )?))
        }
        WhisperAction::Uninstall { keep_models } => {
            uninstall_whisper(&request, keep_models)?;
            Ok(WhisperOutcome::Status(inspect_status(
                &request,
                cancellation,
            )?))
        }
        WhisperAction::DownloadModel { ref name } => {
            download_model(&request, name, cancellation, &progress)?;
            // Re-list models so the caller can see the new file.
            Ok(WhisperOutcome::ModelList(list_models(
                &request, None, None,
            )?))
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
const HF_MODEL_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";
const INSTALL_MANIFEST_NAME: &str = "install-manifest.json";
const WHISPER_VERSION_TAG: &str = "v1.9.1";
const WHISPER_SOURCE_SHA256: &str =
    "147267177eef7b22ec3d2476dd514d1b12e160e176230b740e3d1bd600118447";
const GITHUB_RELEASES_URL: &str =
    "https://api.github.com/repos/ggml-org/whisper.cpp/releases?per_page=100";
const HF_MODEL_TREE_URL: &str = "https://huggingface.co/api/models/ggerganov/whisper.cpp/tree/main?recursive=false&expand=false&limit=1000";
const FALLBACK_MODELS: &[&str] = &[
    "tiny",
    "tiny.en",
    "tiny-q5_1",
    "tiny.en-q5_1",
    "tiny-q8_0",
    "tiny.en-q8_0",
    "base",
    "base.en",
    "base-q5_1",
    "base.en-q5_1",
    "base-q8_0",
    "base.en-q8_0",
    "small",
    "small.en",
    "small-q5_1",
    "small.en-q5_1",
    "small-q8_0",
    "small.en-q8_0",
    "medium",
    "medium.en",
    "medium-q5_0",
    "medium.en-q5_0",
    "medium-q8_0",
    "medium.en-q8_0",
    "large-v1",
    "large-v2",
    "large-v2-q5_0",
    "large-v2-q8_0",
    "large-v3",
    "large-v3-q5_0",
    "large-v3-turbo",
    "large-v3-turbo-q5_0",
    "large-v3-turbo-q8_0",
];

#[derive(Debug, Deserialize)]
struct GithubRelease {
    tag_name: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    published_at: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceEntry {
    path: String,
    #[serde(rename = "type")]
    entry_type: String,
}

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

fn install_binary(
    request: &WhisperRequest,
    cancellation: &CancellationGuard,
    progress: &SharedProgress,
) -> AdapterResult<()> {
    let binary_path = whisper_binary_path(request);
    let bin_dir = binary_path
        .parent()
        .unwrap_or_else(|| Path::new(".subbake/whisper/bin"))
        .to_path_buf();
    let version_tag = WHISPER_VERSION_TAG;
    let target_dir = bin_dir.clone();
    std::fs::create_dir_all(&target_dir).map_err(|source| {
        AdapterError::external_io(
            "create whisper binary directory",
            Some(target_dir.clone()),
            source,
        )
    })?;

    if let Some(platform) = detect_platform()
        && let Some(asset) = pinned_release_asset(&platform, request.build_variant)
    {
        let download_dir = unique_temp_dir("subbake-whisper-install")?;
        let extract_dir = download_dir.join("extract");
        std::fs::create_dir_all(&download_dir)
            .map_err(|e| io::Error::other(format!("create download dir: {e}")))?;
        std::fs::create_dir_all(&extract_dir)
            .map_err(|e| io::Error::other(format!("create extract dir: {e}")))?;
        let archive_path = download_dir.join(&asset.name);

        // Download the archive using reqwest.
        runtime().block_on(async {
            download_file(
                &asset.url,
                &archive_path,
                Some(&asset.sha256),
                cancellation,
                progress,
                "DOWNLOAD_BINARY",
            )
            .await
        })?;

        // Extract archive.
        emit_install_stage(progress, "EXTRACT");
        if asset.name.ends_with(".tar.gz") || asset.name.ends_with(".tgz") {
            extract_tar_gz(&archive_path, &extract_dir, cancellation)?;
        } else if asset.name.ends_with(".zip") {
            extract_zip_system(&archive_path, &extract_dir, cancellation)?;
        } else {
            let direct_path = extract_dir.join(platform.executable_names[0]);
            std::fs::copy(&archive_path, &direct_path)
                .map_err(|e| io::Error::other(format!("copy direct binary: {e}")))?;
        }

        // Assemble and validate a complete installation before touching the
        // currently installed files.
        emit_install_stage(progress, "VERIFY");
        let staged_dir = download_dir.join("staged");
        fs::create_dir_all(&staged_dir)?;
        let staged_binary = staged_dir.join(
            binary_path
                .file_name()
                .ok_or_else(|| io::Error::other("whisper binary has no file name"))?,
        );
        let runtime_libraries = promote_runtime_libraries(&extract_dir, &staged_dir)?;
        promote_binary(&extract_dir, &staged_binary, platform.executable_names)?;
        write_version_file(&staged_dir, &asset.tag)?;
        write_install_manifest(&staged_dir, &staged_binary, runtime_libraries)?;
        verify_whisper_cli(&staged_binary, cancellation)?;
        emit_install_stage(progress, "COMMIT");
        commit_staged_install(&staged_dir, &target_dir)?;
        return Ok(());
    }

    // Fallback: cmake from source
    build_from_source(
        &target_dir,
        &binary_path,
        version_tag,
        request.build_variant,
        cancellation,
        progress,
    )
}

struct ReleaseAsset {
    name: String,
    url: String,
    tag: String,
    sha256: String,
}

fn pinned_release_asset(
    platform: &PlatformAssets,
    variant: WhisperBuildVariant,
) -> Option<ReleaseAsset> {
    let is_x64 = platform.arch_terms.contains(&"x64");
    let (name, sha256) = match (platform.release_platform, is_x64, variant) {
        (ReleasePlatform::Linux, true, WhisperBuildVariant::Cpu) => (
            "whisper-bin-ubuntu-x64.tar.gz",
            "f3bf3b4369a99b54665b0f19b88483b30de27f25963b0414235dea03198515c5",
        ),
        (ReleasePlatform::Linux, false, WhisperBuildVariant::Cpu) => (
            "whisper-bin-ubuntu-arm64.tar.gz",
            "e0b66cd551ff6f2a28fabe3c6e89691eea037bb76833493abb9a71ca788994b3",
        ),
        (ReleasePlatform::Windows, true, WhisperBuildVariant::Cpu) => (
            "whisper-bin-x64.zip",
            "7d8be46ecd31828e1eb7a2ecdd0d6b314feafd82163038ab6092594b0a063539",
        ),
        (ReleasePlatform::Windows, true, WhisperBuildVariant::OpenBlas) => (
            "whisper-blas-bin-x64.zip",
            "3c319eab3e87f85883e1ff3d14426c0a1986c661c5eb5985e8af431ed9c4f71f",
        ),
        (ReleasePlatform::Windows, true, WhisperBuildVariant::Cuda) => (
            "whisper-cublas-12.4.0-bin-x64.zip",
            "106a2030eff8998e4ef320fe72e263a78449e9040386ee27c41ea80b001b601b",
        ),
        _ => return None,
    };
    Some(ReleaseAsset {
        name: name.to_owned(),
        url: format!(
            "https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases/download/{WHISPER_VERSION_TAG}/{name}"
        ),
        tag: WHISPER_VERSION_TAG.to_owned(),
        sha256: sha256.to_owned(),
    })
}

async fn download_file(
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
    cancellation: &CancellationGuard,
    progress: &SharedProgress,
    stage: &str,
) -> AdapterResult<String> {
    cancellation.check().map_err(AdapterError::from)?;
    let client = reqwest::Client::builder()
        .user_agent("subbake/0.1")
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|error| AdapterError::from_http("download client", error))?;

    let mut response =
        await_http_cancellable(client.get(url).send(), cancellation, "download request").await?;
    let status = response.status();

    if !status.is_success() {
        return Err(AdapterError::from_http_status(
            "download service",
            status.as_u16(),
            format!("download failed from {url}"),
            retry_after_ms(response.headers()),
        ));
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|source| {
            AdapterError::external_io(
                "create download destination",
                Some(parent.to_path_buf()),
                source,
            )
        })?;
    }
    let tmp = temporary_sibling(dest, "download");
    let mut cleanup = TemporaryFile::new(tmp.clone());
    let mut file = std::fs::File::create(&tmp).map_err(|source| {
        AdapterError::external_io("create temporary download", Some(tmp.clone()), source)
    })?;
    let header_sha256 = response
        .headers()
        .get("x-linked-etag")
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim_matches('"'))
        .filter(|value| value.len() == 64 && value.chars().all(|ch| ch.is_ascii_hexdigit()))
        .map(str::to_owned);
    let expected_sha256 = expected_sha256.map(str::to_owned).or(header_sha256);
    let total = response.content_length();
    let mut downloaded = 0_u64;
    let progress_step = total
        .map(|bytes| (bytes / 100).max(1024 * 1024))
        .unwrap_or(1024 * 1024);
    let mut last_progress = 0_u64;
    let mut hasher = Sha256::new();
    loop {
        cancellation.check().map_err(AdapterError::from)?;
        let Some(chunk) =
            await_http_cancellable(response.chunk(), cancellation, "download response").await?
        else {
            break;
        };
        hasher.update(&chunk);
        file.write_all(&chunk).map_err(|source| {
            AdapterError::external_io("write temporary download", Some(tmp.clone()), source)
        })?;
        downloaded = downloaded.saturating_add(chunk.len() as u64);
        if downloaded.saturating_sub(last_progress) >= progress_step
            || total.is_some_and(|total| downloaded >= total)
        {
            progress.emit(ProgressEvent::running(
                TaskKind::Download,
                stage,
                downloaded,
                total,
                ProgressUnit::Bytes,
            ));
            last_progress = downloaded;
        }
    }
    file.flush().map_err(|source| {
        AdapterError::external_io("flush temporary download", Some(tmp.clone()), source)
    })?;
    let actual_sha256 = format!("{:x}", hasher.finalize());
    if let Some(expected) = expected_sha256
        && !actual_sha256.eq_ignore_ascii_case(&expected)
    {
        let _ = std::fs::remove_file(&tmp);
        return Err(AdapterError::invalid_input(format!(
            "SHA-256 mismatch for {url}: expected {expected}, got {actual_sha256}"
        )));
    }
    replace_file(&tmp, dest)?;
    cleanup.disarm();
    Ok(actual_sha256)
}

async fn await_http_cancellable<T>(
    future: impl Future<Output = Result<T, reqwest::Error>>,
    cancellation: &CancellationGuard,
    context: &'static str,
) -> AdapterResult<T> {
    tokio::pin!(future);
    loop {
        tokio::select! {
            result = &mut future => {
                return result.map_err(|error| AdapterError::from_http(context, error));
            }
            () = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                if cancellation.is_cancelled() {
                    return Err(AdapterError::Cancelled);
                }
            }
        }
    }
}

fn list_versions(cancellation: &CancellationGuard) -> AdapterResult<WhisperVersionList> {
    let fetched: AdapterResult<Vec<GithubRelease>> = runtime().block_on(fetch_json(
        GITHUB_RELEASES_URL,
        cancellation,
        "fetch whisper.cpp versions",
    ));
    let (releases, refresh_warning) = match fetched {
        Ok(releases) => (releases, None),
        Err(AdapterError::Cancelled) => return Err(AdapterError::Cancelled),
        Err(error) => (
            vec![GithubRelease {
                tag_name: WHISPER_VERSION_TAG.to_owned(),
                draft: false,
                prerelease: false,
                published_at: None,
            }],
            Some(format!("could not refresh upstream versions: {error}")),
        ),
    };
    let versions = releases
        .into_iter()
        .filter(|release| !release.draft)
        .map(|release| WhisperVersion {
            installable: release.tag_name == WHISPER_VERSION_TAG,
            tag: release.tag_name,
            prerelease: release.prerelease,
            published_at: release.published_at,
        })
        .collect();
    Ok(WhisperVersionList {
        pinned_version: WHISPER_VERSION_TAG.to_owned(),
        versions,
        refresh_warning,
    })
}

fn fetch_available_models(
    cancellation: &CancellationGuard,
) -> AdapterResult<(Vec<String>, Option<String>)> {
    let fetched: AdapterResult<Vec<HuggingFaceEntry>> = runtime().block_on(fetch_json(
        HF_MODEL_TREE_URL,
        cancellation,
        "fetch whisper.cpp model catalog",
    ));
    match fetched {
        Ok(entries) => Ok((parse_available_models(entries), None)),
        Err(AdapterError::Cancelled) => Err(AdapterError::Cancelled),
        Err(error) => Ok((
            FALLBACK_MODELS
                .iter()
                .map(|model| (*model).to_owned())
                .collect(),
            Some(format!("could not refresh upstream models: {error}")),
        )),
    }
}

fn parse_available_models(entries: Vec<HuggingFaceEntry>) -> Vec<String> {
    let mut models = entries
        .into_iter()
        .filter(|entry| entry.entry_type == "file")
        .filter_map(|entry| {
            entry
                .path
                .strip_prefix("ggml-")
                .and_then(|name| name.strip_suffix(".bin"))
                .map(str::to_owned)
        })
        .filter(|name| !name.starts_with("for-tests-"))
        .collect::<Vec<_>>();
    models.sort();
    models.dedup();
    models
}

async fn fetch_json<T: serde::de::DeserializeOwned>(
    url: &str,
    cancellation: &CancellationGuard,
    context: &'static str,
) -> AdapterResult<T> {
    cancellation.check().map_err(AdapterError::from)?;
    let client = reqwest::Client::builder()
        .user_agent("subbake/0.1")
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|error| AdapterError::from_http(context, error))?;
    let response = await_http_cancellable(client.get(url).send(), cancellation, context).await?;
    let status = response.status();
    if !status.is_success() {
        let retry_after = retry_after_ms(response.headers());
        let body = await_http_cancellable(response.text(), cancellation, context).await?;
        return Err(AdapterError::from_http_status(
            context,
            status.as_u16(),
            body,
            retry_after,
        ));
    }
    await_http_cancellable(response.json(), cancellation, context).await
}

fn temporary_sibling(path: &Path, label: &str) -> PathBuf {
    static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    path.with_file_name(format!(
        ".{name}.{}.{nonce}.{label}.tmp",
        std::process::id()
    ))
}

fn retry_after_ms(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map(|seconds| seconds.saturating_mul(1_000))
}

fn extract_tar_gz(
    archive: &Path,
    target: &Path,
    cancellation: &CancellationGuard,
) -> AdapterResult<()> {
    std::fs::create_dir_all(target)
        .map_err(|e| io::Error::other(format!("create extract target: {e}")))?;
    let output = run_command_cancellable(
        Command::new("tar").args([
            "-xzf",
            &archive.to_string_lossy(),
            "-C",
            &target.to_string_lossy(),
        ]),
        cancellation,
        "extract whisper.cpp archive",
    )?;
    if !output.status.success() {
        return Err(AdapterError::ChildProcess {
            program: "tar",
            status: output.status.code(),
            message: process_diagnostics(&output, "tar extraction failed"),
        });
    }
    Ok(())
}

fn extract_zip_system(
    archive: &Path,
    target: &Path,
    cancellation: &CancellationGuard,
) -> AdapterResult<()> {
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
    let output = run_command_cancellable(&mut command, cancellation, "extract whisper.cpp zip")?;
    if !output.status.success() {
        return Err(AdapterError::ChildProcess {
            program: "archive extractor",
            status: output.status.code(),
            message: process_diagnostics(&output, "zip extraction failed"),
        });
    }
    Ok(())
}

fn emit_install_stage(progress: &SharedProgress, stage: &str) {
    progress.emit(ProgressEvent::running(
        TaskKind::Installation,
        stage,
        0,
        None,
        ProgressUnit::Steps,
    ));
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

fn commit_staged_install(staged_dir: &Path, target_dir: &Path) -> AdapterResult<()> {
    fs::create_dir_all(target_dir).map_err(|source| {
        AdapterError::external_io(
            "create whisper installation directory",
            Some(target_dir.to_path_buf()),
            source,
        )
    })?;
    let staged_manifest = read_install_manifest(staged_dir)?;
    let old_files = if target_dir.join(INSTALL_MANIFEST_NAME).is_file() {
        read_install_manifest(target_dir)?.files
    } else {
        Vec::new()
    };
    let mut names = staged_manifest.files.clone();
    names.push(INSTALL_MANIFEST_NAME.to_owned());

    let mut prepared = Vec::new();
    for name in &names {
        let temporary = temporary_sibling(&target_dir.join(name), "install");
        let cleanup = TemporaryFile::new(temporary.clone());
        fs::copy(staged_dir.join(name), &temporary).map_err(|source| {
            AdapterError::external_io(
                "stage whisper installation",
                Some(temporary.clone()),
                source,
            )
        })?;
        prepared.push((name.clone(), cleanup));
    }

    let mut committed: Vec<(PathBuf, Option<PathBuf>)> = Vec::new();
    for (name, mut temporary) in prepared {
        let destination = target_dir.join(&name);
        let backup = if destination.exists() {
            let backup = temporary_sibling(&destination, "install-backup");
            if let Err(source) = fs::rename(&destination, &backup) {
                rollback_install(&committed);
                return Err(AdapterError::external_io(
                    "back up whisper installation file",
                    Some(destination),
                    source,
                ));
            }
            Some(backup)
        } else {
            None
        };
        if let Err(source) = fs::rename(&temporary.path, &destination) {
            if let Some(backup) = &backup {
                let _ = fs::rename(backup, &destination);
            }
            rollback_install(&committed);
            return Err(AdapterError::external_io(
                "commit whisper installation file",
                Some(destination),
                source,
            ));
        }
        temporary.disarm();
        committed.push((destination, backup));
    }
    for (_, backup) in &committed {
        if let Some(backup) = backup {
            let _ = fs::remove_file(backup);
        }
    }
    for name in old_files {
        if !staged_manifest.files.contains(&name) {
            let _ = fs::remove_file(target_dir.join(name));
        }
    }
    Ok(())
}

fn rollback_install(committed: &[(PathBuf, Option<PathBuf>)]) {
    for (destination, backup) in committed.iter().rev() {
        let _ = fs::remove_file(destination);
        if let Some(backup) = backup {
            let _ = fs::rename(backup, destination);
        }
    }
}

fn read_install_manifest(directory: &Path) -> io::Result<InstallManifest> {
    let content = fs::read(directory.join(INSTALL_MANIFEST_NAME))?;
    let manifest: InstallManifest = serde_json::from_slice(&content)
        .map_err(|error| io::Error::other(format!("parse install manifest: {error}")))?;
    for name in &manifest.files {
        if Path::new(name).file_name().and_then(|value| value.to_str()) != Some(name.as_str()) {
            return Err(io::Error::other(format!(
                "invalid file name in install manifest: {name}"
            )));
        }
    }
    Ok(manifest)
}

fn build_from_source(
    target_dir: &Path,
    binary_path: &Path,
    version: &str,
    variant: WhisperBuildVariant,
    cancellation: &CancellationGuard,
    progress: &SharedProgress,
) -> AdapterResult<()> {
    // cmake source build: clone/download source, cmake + make.
    let build_dir = unique_temp_dir("subbake-whisper-build")?;
    let src_dir = build_dir.join("source");
    let build_out_dir = build_dir.join("build");

    std::fs::create_dir_all(&src_dir)
        .map_err(|e| io::Error::other(format!("create src dir: {e}")))?;

    // Download source tarball from GitHub.
    let tarball_url = format!(
        "https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/archive/refs/tags/{WHISPER_VERSION_TAG}.tar.gz"
    );
    let tarball_path = build_dir.join("source.tar.gz");
    runtime().block_on(async {
        download_file(
            &tarball_url,
            &tarball_path,
            Some(WHISPER_SOURCE_SHA256),
            cancellation,
            progress,
            "DOWNLOAD_SOURCE",
        )
        .await
    })?;

    // Extract with tar.
    emit_install_stage(progress, "EXTRACT_SOURCE");
    std::fs::create_dir_all(&src_dir).map_err(|e| io::Error::other(format!("create src: {e}")))?;
    let status = run_command_cancellable(
        Command::new("tar").args([
            "-xzf",
            &tarball_path.to_string_lossy(),
            "-C",
            &src_dir.to_string_lossy(),
            "--strip-components=1",
        ]),
        cancellation,
        "extract whisper.cpp source",
    )?;
    if !status.status.success() {
        return Err(AdapterError::ChildProcess {
            program: "tar",
            status: status.status.code(),
            message: "source extraction failed".to_owned(),
        });
    }

    // cmake configure & build
    emit_install_stage(progress, "CONFIGURE");
    std::fs::create_dir_all(&build_out_dir)
        .map_err(|e| io::Error::other(format!("create build dir: {e}")))?;

    let mut configure = Command::new("cmake");
    configure.args([
        "-S",
        &src_dir.to_string_lossy(),
        "-B",
        &build_out_dir.to_string_lossy(),
        "-DWHISPER_BUILD_TESTS=OFF",
        "-DWHISPER_BUILD_EXAMPLES=ON",
        "-DGGML_CUDA=OFF",
        "-DGGML_METAL=OFF",
        "-DGGML_VULKAN=OFF",
        "-DGGML_BLAS=OFF",
    ]);
    match variant {
        WhisperBuildVariant::Cpu => {}
        WhisperBuildVariant::Cuda => {
            configure.arg("-DGGML_CUDA=ON");
        }
        WhisperBuildVariant::Metal => {
            configure.arg("-DGGML_METAL=ON");
        }
        WhisperBuildVariant::Vulkan => {
            configure.arg("-DGGML_VULKAN=ON");
        }
        WhisperBuildVariant::OpenBlas => {
            configure.args(["-DGGML_BLAS=ON", "-DGGML_BLAS_VENDOR=OpenBLAS"]);
        }
    }
    let cmake = run_command_cancellable(&mut configure, cancellation, "configure whisper.cpp")?;
    if !cmake.status.success() {
        let stderr = String::from_utf8_lossy(&cmake.stderr);
        return Err(AdapterError::ChildProcess {
            program: "cmake",
            status: cmake.status.code(),
            message: stderr.into_owned(),
        });
    }

    emit_install_stage(progress, "BUILD");
    let make = run_command_cancellable(
        Command::new("cmake")
            .args([
                "--build",
                &build_out_dir.to_string_lossy(),
                "--config",
                "Release",
                "--target",
                "whisper-cli",
                "-j",
            ])
            .arg(num_cpus().to_string()),
        cancellation,
        "build whisper.cpp",
    )?;
    if !make.status.success() {
        let stderr = String::from_utf8_lossy(&make.stderr);
        // Retry with target "main" (older whisper.cpp releases).
        let make2 = run_command_cancellable(
            Command::new("cmake")
                .args([
                    "--build",
                    &build_out_dir.to_string_lossy(),
                    "--config",
                    "Release",
                    "--target",
                    "main",
                    "-j",
                ])
                .arg(num_cpus().to_string()),
            cancellation,
            "build legacy whisper.cpp CLI",
        )?;
        if !make2.status.success() {
            let stderr2 = String::from_utf8_lossy(&make2.stderr);
            return Err(AdapterError::ChildProcess {
                program: "cmake",
                status: make2.status.code(),
                message: format!("{stderr} / {stderr2}"),
            });
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
            let staged_dir = build_dir.join("staged");
            fs::create_dir_all(&staged_dir)?;
            let staged_binary = staged_dir.join(
                binary_path
                    .file_name()
                    .ok_or_else(|| io::Error::other("whisper binary has no file name"))?,
            );
            let runtime_libraries = promote_runtime_libraries(&build_out_dir, &staged_dir)?;
            std::fs::copy(&entry, &staged_binary)
                .map_err(|e| io::Error::other(format!("copy built binary: {e}")))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&staged_binary, std::fs::Permissions::from_mode(0o755))
                    .map_err(|e| io::Error::other(format!("chmod binary: {e}")))?;
            }
            write_version_file(&staged_dir, version)?;
            write_install_manifest(&staged_dir, &staged_binary, runtime_libraries)?;
            emit_install_stage(progress, "VERIFY");
            verify_whisper_cli(&staged_binary, cancellation)?;
            emit_install_stage(progress, "COMMIT");
            commit_staged_install(&staged_dir, target_dir)?;
            return Ok(());
        }
    }

    Err(AdapterError::Core(subbake_core::CoreError::DataInvariant(
        "built binary not found after cmake build".to_owned(),
    )))
}

fn unique_temp_dir(prefix: &str) -> io::Result<TemporaryDirectory> {
    static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    for _ in 0..100 {
        let nonce = NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(TemporaryDirectory(path)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "unable to allocate a unique temporary directory",
    ))
}

struct TemporaryDirectory(PathBuf);

impl std::ops::Deref for TemporaryDirectory {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<Path> for TemporaryDirectory {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct TemporaryFile {
    path: PathBuf,
    armed: bool,
}

impl TemporaryFile {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// ---------------------------------------------------------------------------
// Model download from HuggingFace
// ---------------------------------------------------------------------------

fn download_model(
    request: &WhisperRequest,
    name: &str,
    cancellation: &CancellationGuard,
    progress: &SharedProgress,
) -> AdapterResult<()> {
    if !is_safe_model_name(name) {
        return Err(AdapterError::invalid_input(format!(
            "invalid whisper.cpp model name `{name}`"
        )));
    }

    let models_dir = request
        .models_dir
        .clone()
        .unwrap_or_else(default_whisper_models_dir);
    let dest = models_dir.join(format!("ggml-{name}.bin"));
    let checksum_path = dest.with_extension("bin.sha256");
    if dest.is_file() && checksum_path.is_file() {
        let expected = fs::read_to_string(&checksum_path).map_err(|source| {
            AdapterError::external_io("read model checksum", Some(checksum_path.clone()), source)
        })?;
        if hash_file(&dest, cancellation)?.eq_ignore_ascii_case(expected.trim()) {
            return Ok(());
        }
    }

    let (available_models, _) = fetch_available_models(cancellation)?;
    validate_available_model(name, &available_models)?;

    std::fs::create_dir_all(&models_dir).map_err(|source| {
        AdapterError::external_io(
            "create whisper models directory",
            Some(models_dir.clone()),
            source,
        )
    })?;

    let url = format!("{HF_MODEL_BASE}/ggml-{name}.bin");
    let checksum = runtime().block_on(async {
        download_file(&url, &dest, None, cancellation, progress, "DOWNLOAD_MODEL").await
    })?;
    write_atomic(&checksum_path, format!("{checksum}\n").as_bytes())?;

    Ok(())
}

fn is_safe_model_name(name: &str) -> bool {
    !name.is_empty()
        && !name.contains("..")
        && name.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        })
}

fn validate_available_model(name: &str, available_models: &[String]) -> AdapterResult<()> {
    if available_models.iter().any(|model| model == name) {
        return Ok(());
    }
    Err(AdapterError::invalid_input(format!(
        "unknown model `{name}`; run `sbake whisper model list` to see available models"
    )))
}

fn hash_file(path: &Path, cancellation: &CancellationGuard) -> AdapterResult<String> {
    use std::io::Read;

    let mut file = fs::File::open(path).map_err(|source| {
        AdapterError::external_io("open file for checksum", Some(path.to_path_buf()), source)
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        cancellation.check().map_err(AdapterError::from)?;
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn write_atomic(path: &Path, content: &[u8]) -> AdapterResult<()> {
    let temporary = temporary_sibling(path, "write");
    fs::write(&temporary, content).map_err(|source| {
        AdapterError::external_io("write temporary file", Some(temporary.clone()), source)
    })?;
    replace_file(&temporary, path)
}

fn replace_file(source: &Path, destination: &Path) -> AdapterResult<()> {
    if !destination.exists() {
        return fs::rename(source, destination).map_err(|error| {
            AdapterError::external_io("finalize file", Some(destination.to_path_buf()), error)
        });
    }
    let backup = temporary_sibling(destination, "backup");
    fs::rename(destination, &backup).map_err(|error| {
        AdapterError::external_io(
            "back up existing file",
            Some(destination.to_path_buf()),
            error,
        )
    })?;
    if let Err(error) = fs::rename(source, destination) {
        let _ = fs::rename(&backup, destination);
        return Err(AdapterError::external_io(
            "replace file",
            Some(destination.to_path_buf()),
            error,
        ));
    }
    let _ = fs::remove_file(backup);
    Ok(())
}

fn list_models(
    request: &WhisperRequest,
    available_models: Option<Vec<String>>,
    refresh_warning: Option<String>,
) -> io::Result<WhisperModelList> {
    let models_dir = request
        .models_dir
        .clone()
        .unwrap_or_else(default_whisper_models_dir);
    if !models_dir.is_dir() {
        return Ok(WhisperModelList {
            models_dir,
            models_dir_exists: false,
            models: Vec::new(),
            available_models: available_models.unwrap_or_default(),
            refresh_warning,
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
            .strip_prefix("ggml-")
            .unwrap_or("model")
            .to_owned();
        models.push(WhisperModel { name, path });
    }
    models.sort_by(|left, right| left.path.cmp(&right.path));

    Ok(WhisperModelList {
        models_dir,
        models_dir_exists: true,
        models,
        available_models: available_models.unwrap_or_default(),
        refresh_warning,
    })
}

fn inspect_status(
    request: &WhisperRequest,
    cancellation: &CancellationGuard,
) -> AdapterResult<WhisperStatus> {
    let binary_path = request
        .binary_path
        .clone()
        .unwrap_or_else(default_whisper_binary_path);
    let models_dir = request
        .models_dir
        .clone()
        .unwrap_or_else(default_whisper_models_dir);

    let (version, capability_error) = if binary_path.is_file() {
        match verify_whisper_cli(&binary_path, cancellation) {
            Ok(version) => (Some(version), None),
            Err(AdapterError::Cancelled) => return Err(AdapterError::Cancelled),
            Err(error) => (None, Some(error.to_string())),
        }
    } else {
        (None, None)
    };
    Ok(WhisperStatus {
        binary_exists: binary_path.is_file(),
        models_dir_exists: models_dir.is_dir(),
        binary_path,
        models_dir,
        version,
        capability_error,
    })
}

pub(crate) fn verify_whisper_cli(
    binary_path: &Path,
    cancellation: &CancellationGuard,
) -> AdapterResult<String> {
    let version_output = run_command_cancellable(
        Command::new(binary_path).arg("--version"),
        cancellation,
        "inspect whisper.cpp version",
    )?;
    if !version_output.status.success() {
        return Err(AdapterError::ChildProcess {
            program: "whisper.cpp",
            status: version_output.status.code(),
            message: process_diagnostics(&version_output, "`--version` failed"),
        });
    }
    let version = process_diagnostics(&version_output, "unknown whisper.cpp version");
    let help_output = run_command_cancellable(
        Command::new(binary_path).arg("--help"),
        cancellation,
        "inspect whisper.cpp capabilities",
    )?;
    let help = format!(
        "{}\n{}",
        String::from_utf8_lossy(&help_output.stdout),
        String::from_utf8_lossy(&help_output.stderr)
    );
    let missing = [
        "--model",
        "--file",
        "--output-file",
        "--output-srt",
        "--output-vtt",
    ]
    .into_iter()
    .filter(|flag| !help.contains(flag))
    .collect::<Vec<_>>();
    if !help_output.status.success() || !missing.is_empty() {
        let detail = if missing.is_empty() {
            process_diagnostics(&help_output, "`--help` failed")
        } else {
            format!("missing required options: {}", missing.join(", "))
        };
        return Err(AdapterError::invalid_input(format!(
            "incompatible whisper.cpp CLI at {}: {detail}",
            binary_path.display()
        )));
    }
    Ok(version.lines().next().unwrap_or(&version).trim().to_owned())
}

fn process_diagnostics(output: &std::process::Output, fallback: &str) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let message = [stdout.trim(), stderr.trim()]
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if message.is_empty() {
        fallback.to_owned()
    } else {
        message
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
    default_whisper_binary_path_for(None)
}

pub fn default_whisper_binary_path_for(runtime_dir: Option<&Path>) -> PathBuf {
    whisper_runtime_root(runtime_dir)
        .join("bin")
        .join(default_whisper_binary_name())
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

pub fn default_whisper_models_dir() -> PathBuf {
    default_whisper_models_dir_for(None)
}

pub fn default_whisper_models_dir_for(runtime_dir: Option<&Path>) -> PathBuf {
    whisper_runtime_root(runtime_dir).join("models")
}

fn whisper_runtime_root(runtime_dir: Option<&Path>) -> PathBuf {
    runtime_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(".subbake"))
        .join("whisper")
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
            build_variant: WhisperBuildVariant::Cpu,
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

        assert_eq!(
            pinned_release_asset(&linux_x64, WhisperBuildVariant::Cpu)
                .expect("linux x64 CPU asset")
                .name,
            "whisper-bin-ubuntu-x64.tar.gz"
        );
        assert_eq!(
            pinned_release_asset(&linux_arm64, WhisperBuildVariant::Cpu)
                .expect("linux arm64 CPU asset")
                .name,
            "whisper-bin-ubuntu-arm64.tar.gz"
        );
        assert_eq!(
            pinned_release_asset(&windows_x64, WhisperBuildVariant::Cpu)
                .expect("windows CPU asset")
                .name,
            "whisper-bin-x64.zip"
        );
        assert_eq!(
            pinned_release_asset(&windows_x64, WhisperBuildVariant::OpenBlas)
                .expect("windows OpenBLAS asset")
                .name,
            "whisper-blas-bin-x64.zip"
        );
        assert_eq!(
            pinned_release_asset(&windows_x64, WhisperBuildVariant::Cuda)
                .expect("windows CUDA asset")
                .name,
            "whisper-cublas-12.4.0-bin-x64.zip"
        );
        assert!(pinned_release_asset(&linux_x64, WhisperBuildVariant::Cuda).is_none());
        assert!(pinned_release_asset(&windows_x64, WhisperBuildVariant::Metal).is_none());
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
            build_variant: WhisperBuildVariant::Cpu,
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
    fn staged_install_replaces_managed_files_and_preserves_user_files() {
        let root = temp_root("staged-install");
        let target = root.join("target");
        let staged = root.join("staged");
        fs::create_dir_all(&target).expect("create target");
        fs::create_dir_all(&staged).expect("create staged");
        let old_binary = target.join("whisper-cli");
        fs::write(&old_binary, b"old").expect("write old binary");
        fs::write(target.join("old-library.so"), b"old library").expect("write old library");
        fs::write(target.join("user.txt"), b"keep").expect("write user file");
        write_install_manifest(&target, &old_binary, vec!["old-library.so".to_owned()])
            .expect("write old manifest");

        let staged_binary = staged.join("whisper-cli");
        fs::write(&staged_binary, b"new").expect("write staged binary");
        fs::write(staged.join("new-library.so"), b"new library").expect("write new library");
        write_version_file(&staged, "v2").expect("write staged version");
        write_install_manifest(&staged, &staged_binary, vec!["new-library.so".to_owned()])
            .expect("write staged manifest");

        commit_staged_install(&staged, &target).expect("commit staged install");
        let binary = fs::read(target.join("whisper-cli")).expect("read installed binary");
        let user = fs::read(target.join("user.txt")).expect("read user file");
        let old_exists = target.join("old-library.so").exists();
        let new_exists = target.join("new-library.so").exists();
        let _ = fs::remove_dir_all(&root);

        assert_eq!(binary, b"new");
        assert_eq!(user, b"keep");
        assert!(!old_exists);
        assert!(new_exists);
    }

    #[test]
    fn list_models_returns_sorted_model_files() {
        let root = temp_root("models");
        let models_dir = root.join("models");
        fs::create_dir_all(&models_dir).expect("create models dir");
        fs::write(models_dir.join("ggml-small.bin"), b"model").expect("write model");
        fs::write(models_dir.join("ggml-base.gguf"), b"model").expect("write model");
        fs::write(models_dir.join("notes.txt"), b"ignore").expect("write note");

        let request = WhisperRequest {
            action: WhisperAction::ListModels,
            binary_path: None,
            models_dir: Some(models_dir),
            build_variant: WhisperBuildVariant::Cpu,
        };
        let list = list_models(&request, None, None).expect("list models");
        let _ = fs::remove_dir_all(&root);

        assert!(list.models_dir_exists);
        assert_eq!(
            list.models
                .iter()
                .map(|model| model.name.as_str())
                .collect::<Vec<_>>(),
            vec!["base", "small"]
        );
    }

    #[test]
    fn model_download_validation_uses_the_catalog() {
        let catalog = FALLBACK_MODELS
            .iter()
            .map(|model| (*model).to_owned())
            .collect::<Vec<_>>();

        validate_available_model("large-v3-turbo-q8_0", &catalog)
            .expect("quantized catalog model should be accepted");
        let error = validate_available_model("unknown-model-test", &catalog)
            .expect_err("unknown model should be rejected");
        assert!(error.to_string().contains("unknown model"));
    }

    #[test]
    fn model_names_cannot_escape_the_models_directory() {
        assert!(is_safe_model_name("large-v3-turbo-q8_0"));
        assert!(is_safe_model_name("base.en-q8_0"));
        assert!(!is_safe_model_name("../large-v3"));
        assert!(!is_safe_model_name("large/v3"));
    }

    #[test]
    fn cancelled_download_stops_before_network_or_file_creation() {
        let root = temp_root("cancelled-download");
        let destination = root.join("artifact.bin");
        let token = subbake_core::CancellationToken::default();
        let guard = token.guard();
        let progress: SharedProgress = std::sync::Arc::new(NoopProgress);
        token.cancel();

        let error = runtime()
            .block_on(download_file(
                "https://invalid.example/artifact",
                &destination,
                None,
                &guard,
                &progress,
                "TEST",
            ))
            .expect_err("cancelled download must stop");
        let exists = destination.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(error.is_cancelled());
        assert!(!exists);
    }

    #[test]
    fn model_download_succeeds_when_file_exists() {
        let root = temp_root("exists");
        let models_dir = root.join("models");
        std::fs::create_dir_all(&models_dir).expect("create models dir");
        let model_path = models_dir.join("ggml-base.bin");
        std::fs::write(&model_path, b"fake").expect("write fake model");
        let checksum = hash_file(&model_path, &CancellationGuard::never()).expect("hash model");
        std::fs::write(models_dir.join("ggml-base.bin.sha256"), checksum).expect("write checksum");

        let outcome = run_whisper(WhisperRequest {
            action: WhisperAction::DownloadModel {
                name: "base".to_owned(),
            },
            binary_path: None,
            models_dir: Some(models_dir),
            build_variant: WhisperBuildVariant::Cpu,
        })
        .expect("existing file should succeed");
        let _ = std::fs::remove_dir_all(&root);

        assert!(matches!(outcome, WhisperOutcome::ModelList(_)));
    }

    #[test]
    fn available_model_catalog_keeps_downloadable_ggml_files() {
        let models = parse_available_models(vec![
            HuggingFaceEntry {
                path: "ggml-small.bin".to_owned(),
                entry_type: "file".to_owned(),
            },
            HuggingFaceEntry {
                path: "ggml-base.en.bin".to_owned(),
                entry_type: "file".to_owned(),
            },
            HuggingFaceEntry {
                path: "ggml-for-tests-tiny.bin".to_owned(),
                entry_type: "file".to_owned(),
            },
            HuggingFaceEntry {
                path: "README.md".to_owned(),
                entry_type: "file".to_owned(),
            },
        ]);

        assert_eq!(models, vec!["base.en", "small"]);
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
            build_variant: WhisperBuildVariant::Cpu,
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
            build_variant: WhisperBuildVariant::Cpu,
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
