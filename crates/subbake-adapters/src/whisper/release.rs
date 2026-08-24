use crate::platform::{Architecture, CapabilitySet, OperatingSystem};

use super::{GITHUB_REPO_NAME, GITHUB_REPO_OWNER, WHISPER_VERSION_TAG, WhisperBuildVariant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReleasePlatform {
    Linux,
    Windows,
}

pub(super) struct PlatformAssets {
    pub(super) release_platform: ReleasePlatform,
    pub(super) arch_terms: &'static [&'static str],
    pub(super) executable_names: &'static [&'static str],
}

pub(super) struct ReleaseAsset {
    pub(super) name: String,
    pub(super) url: String,
    pub(super) tag: String,
    pub(super) sha256: String,
}

pub(super) fn detect_platform() -> Option<PlatformAssets> {
    let capabilities = CapabilitySet::current();
    if !capabilities.has_prebuilt_whisper() {
        return None;
    }
    match (capabilities.operating_system(), capabilities.architecture()) {
        (OperatingSystem::Linux, Architecture::X86_64) => Some(PlatformAssets {
            release_platform: ReleasePlatform::Linux,
            arch_terms: &["x64", "x86_64", "amd64"],
            executable_names: &["whisper-whisper-cli", "whisper-cli", "main"],
        }),
        (OperatingSystem::Linux, Architecture::Aarch64) => Some(PlatformAssets {
            release_platform: ReleasePlatform::Linux,
            arch_terms: &["arm64", "aarch64"],
            executable_names: &["whisper-whisper-cli", "whisper-cli", "main"],
        }),
        (OperatingSystem::Windows, Architecture::X86_64) => Some(PlatformAssets {
            release_platform: ReleasePlatform::Windows,
            arch_terms: &["x64", "x86_64", "amd64"],
            executable_names: &["whisper-whisper-cli.exe", "whisper-cli.exe", "main.exe"],
        }),
        _ => None,
    }
}

pub(super) fn pinned_release_asset(
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
