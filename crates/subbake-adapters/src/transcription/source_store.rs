//! Mode-independent, content-addressed recognition and segmentation artifacts.
use super::TranscriptionSettings;
use crate::{error::AdapterResult, fs::write_file_atomically, runtime_store::FileRuntimeStore};
use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};
use subbake_core::{CancellationGuard, ports::RuntimeLayoutStore};

pub(super) const RECOGNITION_VERSION: u64 = 1;

pub(super) struct SourceStore {
    root: PathBuf,
    identity: serde_json::Value,
    pub use_cache: bool,
    pub resume: bool,
}

impl SourceStore {
    pub fn new(
        media: &Path,
        model: &Path,
        binary: &Path,
        version: &str,
        settings: &TranscriptionSettings,
        cancellation: &CancellationGuard,
    ) -> AdapterResult<Self> {
        cancellation.check()?;
        let stable_media = crate::fs::stable_runtime_input_path(media)?;
        let paths = subbake_core::storage::build_runtime_paths(
            media,
            &stable_media,
            settings.runtime_dir.as_deref(),
            None,
            "Auto",
            "Chinese",
            false,
        );
        let root = paths.root_dir.join("cache/transcription");
        if settings.use_cache || settings.resume {
            FileRuntimeStore::new(paths).ensure_layout()?;
        }
        let identity = serde_json::json!({
            "version": RECOGNITION_VERSION,
            "media": crate::whisper::hash_file(media, cancellation)?,
            "model": crate::whisper::hash_file(model, cancellation)?,
            "binary": crate::whisper::hash_file(binary, cancellation)?,
            "whisper_version": version,
            "language": subbake_core::transcription::SpokenLanguage::parse(settings.language.as_deref().unwrap_or("Auto"))?.recognition_code(),
            "normalization": "pcm-s16le-16khz-mono-a0-v1",
        });
        Ok(Self {
            root,
            identity,
            use_cache: settings.use_cache,
            resume: settings.resume,
        })
    }

    pub fn recognition_path(
        &self,
        descriptor: &super::TranscriptionChunkDescriptor,
        long_audio: bool,
    ) -> PathBuf {
        self.root.join(key(&serde_json::json!({ "identity": self.identity, "chunk": descriptor, "max_context": if long_audio { Some(0) } else { None } }))).join("raw.json")
    }

    pub fn segmented_path(
        &self,
        raw: &Path,
        settings: &TranscriptionSettings,
        offset: i64,
    ) -> PathBuf {
        raw.with_file_name(format!(
            "segmented-{}.json",
            key(&serde_json::json!({
                "version": subbake_core::transcription::SEGMENTATION_VERSION,
                "segmentation": settings.effective_segmentation(),
                "timeline_offset_ms": offset,
                "normalize_text": settings.effective_normalize_text(),
                "speaker_labels": settings.effective_speaker_labels(),
                "filter_silence_markers": settings.effective_filter_silence_markers(),
                "postprocessing_version": 3,
            }))
        ))
    }
}

fn key(value: &serde_json::Value) -> String {
    Sha256::digest(value.to_string().as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(super) fn read<T: DeserializeOwned>(path: &Path) -> AdapterResult<Option<T>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes).map_err(|source| {
            crate::error::AdapterError::Serialization {
                context: "read transcription artifact",
                source,
            }
        })?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

pub(super) fn write<T: Serialize>(
    path: &Path,
    value: &T,
    cancellation: &CancellationGuard,
) -> AdapterResult<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|source| {
        crate::error::AdapterError::Serialization {
            context: "write transcription artifact",
            source,
        }
    })?;
    write_bytes(path, &bytes, cancellation)
}

pub(super) fn write_bytes(
    path: &Path,
    bytes: &[u8],
    cancellation: &CancellationGuard,
) -> AdapterResult<()> {
    cancellation.check()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    write_file_atomically(path, bytes)
}
