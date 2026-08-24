use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use subbake_core::entities::{
    AgentLog, BatchTranslationResult, FailureLog, ReviewReport, ReviewResult, SubtitleSegment,
    TerminologyPreflightResult, Usage,
};
use subbake_core::error::{CoreError, CoreResult, StorageError, StorageIoKind};
use subbake_core::ports::{
    BackendJsonResult, BackendPayload, BatchShardKind, CacheStage, RuntimeCacheStore,
    RuntimeCheckpointStore, RuntimeDiagnosticStore, RuntimeLayoutStore, RuntimeMemoryStore,
};
use subbake_core::storage::{RunState, RuntimePaths};

use crate::error::AdapterError;
use crate::fs::write_file_atomically;

pub(crate) const RUNTIME_MARKER_NAME: &str = ".subbake-runtime-v1";
pub(crate) const RUNTIME_MARKER_CONTENT: &str = "subbake-runtime-v1\n";

#[derive(Debug, Clone)]
pub struct FileRuntimeStore {
    paths: RuntimePaths,
}

impl FileRuntimeStore {
    pub fn new(paths: RuntimePaths) -> Self {
        Self { paths }
    }

    pub fn batch_shard_path(&self, kind: BatchShardKind, batch_index: usize) -> PathBuf {
        let root = match kind {
            BatchShardKind::Translated => &self.paths.translated_batches_dir,
            BatchShardKind::Reviewed => &self.paths.reviewed_batches_dir,
            BatchShardKind::Finalized => &self.paths.finalized_batches_dir,
        };
        root.join(format!("{batch_index:04}.json"))
    }

    pub fn cache_path(&self, stage: CacheStage, request_hash: &str) -> PathBuf {
        self.paths
            .cache_dir
            .join(stage.as_str())
            .join(format!("{request_hash}.json"))
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct TranslationCacheEntry {
    payload: BatchTranslationResult,
    #[serde(default)]
    usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
struct ReviewCacheEntry {
    payload: ReviewResult,
    #[serde(default)]
    usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
struct TerminologyCacheEntry {
    payload: TerminologyPreflightResult,
    #[serde(default)]
    usage: Usage,
}

impl RuntimeLayoutStore for FileRuntimeStore {
    fn paths(&self) -> &RuntimePaths {
        &self.paths
    }

    fn ensure_layout(&self) -> CoreResult<()> {
        for directory in [
            &self.paths.root_dir,
            &self.paths.run_dir,
            &self.paths.cache_dir,
            &self.paths.failures_dir,
            &self.paths.translated_batches_dir,
            &self.paths.reviewed_batches_dir,
            &self.paths.finalized_batches_dir,
            &self.paths.agent_logs_dir,
        ] {
            fs::create_dir_all(directory).map_err(|error| {
                storage_error("create runtime directory", Some(directory), error)
            })?;
        }
        let marker = self.paths.root_dir.join(RUNTIME_MARKER_NAME);
        match fs::symlink_metadata(&marker) {
            Ok(metadata) if metadata.file_type().is_file() => {}
            Ok(_) => {
                return Err(storage_error(
                    "validate runtime marker",
                    Some(&marker),
                    io::Error::other("runtime marker is not a regular file"),
                ));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&marker)
                    .map_err(|error| {
                        storage_error("create runtime marker", Some(&marker), error)
                    })?;
                file.write_all(RUNTIME_MARKER_CONTENT.as_bytes())
                    .map_err(|error| storage_error("write runtime marker", Some(&marker), error))?;
            }
            Err(error) => {
                return Err(storage_error(
                    "inspect runtime marker",
                    Some(&marker),
                    error,
                ));
            }
        }
        Ok(())
    }
}

impl RuntimeMemoryStore for FileRuntimeStore {
    fn save_glossary(&self, entries: &[(String, String)]) -> CoreResult<()> {
        write_json_verified(&self.paths.glossary_path, &string_map_json(entries))
    }

    fn save_translation_memory(&self, entries: &[(String, String)]) -> CoreResult<()> {
        write_json_verified(
            &self.paths.translation_memory_path,
            &string_map_json(entries),
        )
    }

    fn save_review_report(&self, report: &ReviewReport) -> CoreResult<()> {
        let value = serde_json::to_value(report).map_err(|error| {
            CoreError::DataInvariant(format!("serialize review report failed: {error}"))
        })?;
        write_json_verified(&self.paths.review_report_path, &value)
    }

    fn load_review_report(&self) -> CoreResult<Option<ReviewReport>> {
        if !self.paths.review_report_path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&self.paths.review_report_path).map_err(|error| {
            storage_error(
                "read review report",
                Some(&self.paths.review_report_path),
                error,
            )
        })?;
        serde_json::from_str(&text).map(Some).map_err(|error| {
            CoreError::DataInvariant(format!("review report parse failed: {error}"))
        })
    }

    fn load_glossary(&self) -> CoreResult<Vec<(String, String)>> {
        if !self.paths.glossary_path.exists() {
            return Ok(Vec::new());
        }
        let text = fs::read_to_string(&self.paths.glossary_path).map_err(|error| {
            storage_error("read glossary", Some(&self.paths.glossary_path), error)
        })?;
        parse_string_map(&text, "glossary")
    }

    fn load_translation_memory(&self) -> CoreResult<Vec<(String, String)>> {
        let path = std::iter::once(self.paths.translation_memory_path.clone())
            .chain(legacy_translation_memory_paths(
                &self.paths.translation_memory_path,
            ))
            .find(|path| path.exists());
        let Some(path) = path else {
            return Ok(Vec::new());
        };
        let text = fs::read_to_string(&path)
            .map_err(|error| storage_error("read translation memory", Some(&path), error))?;
        parse_string_map(&text, "translation memory")
    }
}

impl RuntimeCheckpointStore for FileRuntimeStore {
    fn save_batch_segments(
        &self,
        kind: BatchShardKind,
        batch_index: usize,
        segments: &[SubtitleSegment],
    ) -> CoreResult<()> {
        let path = self.batch_shard_path(kind, batch_index);
        let payload = serde_json::json!({
            "batch_index": batch_index,
            "segments": segments.iter().map(segment_json).collect::<Vec<_>>(),
        });
        write_json_verified(&path, &payload)
    }

    fn load_batch_segments(
        &self,
        kind: BatchShardKind,
        completed_batches: usize,
    ) -> CoreResult<Vec<SubtitleSegment>> {
        let mut loaded = Vec::new();
        for batch_index in 1..=completed_batches {
            let path = self.batch_shard_path(kind, batch_index);
            if !path.exists() {
                return Err(CoreError::DataInvariant(format!(
                    "missing batch shard for resume: {}",
                    path.display()
                )));
            }
            let text = fs::read_to_string(&path)
                .map_err(|error| storage_error("read batch shard", Some(&path), error))?;
            let payload: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
                CoreError::DataInvariant(format!("batch shard parse failed: {error}"))
            })?;
            let segments = payload["segments"].as_array().ok_or_else(|| {
                CoreError::DataInvariant(format!(
                    "batch shard missing segments: {}",
                    path.display()
                ))
            })?;
            for segment in segments {
                loaded.push(segment_from_json(segment)?);
            }
        }
        Ok(loaded)
    }

    fn save_run_state(&self, state: &RunState) -> CoreResult<()> {
        let payload = serde_json::to_value(state).map_err(|error| {
            CoreError::DataInvariant(format!("serialize run state failed: {error}"))
        })?;
        write_json_verified(&self.paths.state_path, &payload)
    }

    fn load_run_state(&self) -> CoreResult<Option<RunState>> {
        if !self.paths.state_path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&self.paths.state_path).map_err(|error| {
            storage_error("read run state", Some(&self.paths.state_path), error)
        })?;
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|error| CoreError::DataInvariant(format!("run state parse failed: {error}")))
    }
}

impl RuntimeCacheStore for FileRuntimeStore {
    fn save_cached_response(
        &self,
        stage: CacheStage,
        request_hash: &str,
        response: &BackendJsonResult,
    ) -> CoreResult<()> {
        let value = match (stage, &response.payload) {
            (
                CacheStage::Translate | CacheStage::AgentTranslateRepair,
                BackendPayload::Translation(payload),
            ) => serde_json::to_value(TranslationCacheEntry {
                payload: payload.clone(),
                usage: response.usage,
            }),
            (
                CacheStage::Review | CacheStage::AgentReviewRepair,
                BackendPayload::Review(payload),
            ) => serde_json::to_value(ReviewCacheEntry {
                payload: payload.clone(),
                usage: response.usage,
            }),
            (CacheStage::Terminology, BackendPayload::Terminology(payload)) => {
                serde_json::to_value(TerminologyCacheEntry {
                    payload: payload.clone(),
                    usage: response.usage,
                })
            }
            _ => {
                return Err(CoreError::DataInvariant(format!(
                    "cached response payload does not match stage {}",
                    stage.as_str()
                )));
            }
        };
        let value = value.map_err(|error| {
            CoreError::DataInvariant(format!("serialize request cache failed: {error}"))
        })?;
        write_json_verified(&self.cache_path(stage, request_hash), &value)
    }

    fn load_cached_response(
        &self,
        stage: CacheStage,
        request_hash: &str,
    ) -> CoreResult<Option<BackendJsonResult>> {
        let path = self.cache_path(stage, request_hash);
        if !path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path)
            .map_err(|error| storage_error("read cached response", Some(&path), error))?;
        match stage {
            CacheStage::Translate | CacheStage::AgentTranslateRepair => {
                let entry: TranslationCacheEntry =
                    serde_json::from_str(&text).map_err(|error| {
                        CoreError::DataInvariant(format!("request cache parse failed: {error}"))
                    })?;
                Ok(Some(BackendJsonResult {
                    payload: BackendPayload::Translation(entry.payload),
                    usage: entry.usage,
                }))
            }
            CacheStage::Review | CacheStage::AgentReviewRepair => {
                let entry: ReviewCacheEntry = serde_json::from_str(&text).map_err(|error| {
                    CoreError::DataInvariant(format!("request cache parse failed: {error}"))
                })?;
                Ok(Some(BackendJsonResult {
                    payload: BackendPayload::Review(entry.payload),
                    usage: entry.usage,
                }))
            }
            CacheStage::Terminology => {
                let entry: TerminologyCacheEntry =
                    serde_json::from_str(&text).map_err(|error| {
                        CoreError::DataInvariant(format!("request cache parse failed: {error}"))
                    })?;
                Ok(Some(BackendJsonResult {
                    payload: BackendPayload::Terminology(entry.payload),
                    usage: entry.usage,
                }))
            }
        }
    }
}

impl RuntimeDiagnosticStore for FileRuntimeStore {
    fn save_failure_log(&self, log: &FailureLog) -> CoreResult<PathBuf> {
        let path = self
            .paths
            .failures_dir
            .join(format!("{}_batch_{:04}.json", log.stage, log.batch_index));
        let value = serde_json::to_value(log).map_err(|error| {
            CoreError::DataInvariant(format!("serialize failure log failed: {error}"))
        })?;
        write_json_verified(&path, &value)?;
        Ok(path)
    }

    fn save_agent_log(&self, log: &AgentLog) -> CoreResult<PathBuf> {
        let path = self
            .paths
            .agent_logs_dir
            .join(format!("{}_batch_{:04}.json", log.stage, log.batch_index));
        let value = serde_json::to_value(log).map_err(|error| {
            CoreError::DataInvariant(format!("serialize agent log failed: {error}"))
        })?;
        write_json_verified(&path, &value)?;
        Ok(path)
    }
}

fn legacy_translation_memory_paths(current: &Path) -> Vec<PathBuf> {
    let Some(file_name) = current.file_name().and_then(|name| name.to_str()) else {
        return Vec::new();
    };
    if let Some(suffix) = file_name.strip_prefix("translation_memory.v5.") {
        return vec![
            current.with_file_name(format!("translation_memory.v4.{suffix}")),
            current.with_file_name(format!("translation_memory.v3.{suffix}")),
        ];
    }
    if let Some(suffix) = file_name.strip_prefix("translation_memory.v4.") {
        return vec![current.with_file_name(format!("translation_memory.v3.{suffix}"))];
    }
    Vec::new()
}

fn string_map_json(entries: &[(String, String)]) -> serde_json::Value {
    serde_json::Value::Object(
        entries
            .iter()
            .map(|(key, value)| (key.clone(), serde_json::Value::String(value.clone())))
            .collect(),
    )
}

fn segment_json(segment: &SubtitleSegment) -> serde_json::Value {
    let mut fields = [
        (
            "id".to_owned(),
            serde_json::Value::String(segment.id.clone()),
        ),
        (
            "text".to_owned(),
            serde_json::Value::String(segment.text.clone()),
        ),
        ("start".to_owned(), option_string_json(&segment.start)),
        ("end".to_owned(), option_string_json(&segment.end)),
        (
            "identifier".to_owned(),
            option_string_json(&segment.identifier),
        ),
        ("settings".to_owned(), option_string_json(&segment.settings)),
    ]
    .into_iter()
    .collect::<serde_json::Map<_, _>>();
    if !segment.semantic.is_empty() {
        fields.insert(
            "semantic".to_owned(),
            serde_json::to_value(&segment.semantic).unwrap_or_default(),
        );
    }
    serde_json::Value::Object(fields)
}

fn segment_from_json(value: &serde_json::Value) -> CoreResult<SubtitleSegment> {
    Ok(SubtitleSegment {
        id: value["id"].as_str().unwrap_or_default().to_owned(),
        text: value["text"].as_str().unwrap_or_default().to_owned(),
        start: optional_string_from_json(&value["start"]),
        end: optional_string_from_json(&value["end"]),
        identifier: optional_string_from_json(&value["identifier"]),
        settings: optional_string_from_json(&value["settings"]),
        semantic: value
            .get("semantic")
            .filter(|semantic| !semantic.is_null())
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| {
                CoreError::DataInvariant(format!(
                    "runtime subtitle segment has invalid semantic metadata: {error}"
                ))
            })?
            .unwrap_or_default(),
    })
}

fn optional_string_from_json(value: &serde_json::Value) -> Option<String> {
    value.as_str().map(ToOwned::to_owned)
}

fn option_string_json(value: &Option<String>) -> serde_json::Value {
    value
        .as_ref()
        .map(|value| serde_json::Value::String(value.clone()))
        .unwrap_or(serde_json::Value::Null)
}

fn write_json_verified(path: &Path, payload: &serde_json::Value) -> CoreResult<()> {
    let serialized = serde_json::to_string(payload)
        .map_err(|error| CoreError::DataInvariant(format!("serialize json failed: {error}")))?;
    write_file_atomically(path, serialized.as_bytes()).map_err(atomic_storage_error)
}

fn parse_string_map(text: &str, label: &str) -> CoreResult<Vec<(String, String)>> {
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(text)
        .map_err(|error| CoreError::DataInvariant(format!("{label} parse failed: {error}")))?;
    map.into_iter()
        .map(|(key, value)| match value {
            serde_json::Value::String(value) => Ok((key, value)),
            value => Err(CoreError::DataInvariant(format!(
                "{label} entry `{key}` must be a string, got {}",
                json_type_name(&value)
            ))),
        })
        .collect()
}

fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn atomic_storage_error(error: AdapterError) -> CoreError {
    match error {
        AdapterError::ExternalIo {
            operation,
            path,
            source,
        } => storage_error(operation, path.as_deref(), source),
        other => CoreError::DataInvariant(format!("atomic runtime write failed: {other}")),
    }
}

fn storage_error(operation: &str, path: Option<&Path>, error: io::Error) -> CoreError {
    let kind = match error.kind() {
        io::ErrorKind::NotFound => StorageIoKind::NotFound,
        io::ErrorKind::PermissionDenied => StorageIoKind::PermissionDenied,
        io::ErrorKind::AlreadyExists => StorageIoKind::AlreadyExists,
        _ => StorageIoKind::Other,
    };
    CoreError::Storage(StorageError::Io {
        operation: operation.to_owned(),
        path: path.map(|path| path.display().to_string()),
        kind,
        message: error.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use subbake_core::entities::{
        AgentLog, AttemptLog, BatchTranslationResult, DocumentGuide, FailureLog, GlossaryEntry,
        PipelineOptions, ReviewChange, ReviewReport, ReviewStats, TerminologyEntity,
        TerminologyKind, TerminologyPreflightResult, TerminologyStats, TranslationLine, Usage,
    };
    use subbake_core::memory::ContextMemory;
    use subbake_core::storage::{RunState, build_runtime_paths, input_signature_from_bytes};

    use super::*;

    #[test]
    fn runtime_io_errors_include_operation_and_path() {
        let temporary = tempfile::Builder::new()
            .prefix("subbake-runtime-error-")
            .tempdir()
            .expect("create temporary directory");
        let blocked_root = temporary.path().join("runtime-file");
        fs::write(&blocked_root, "not a directory").expect("write blocking file");
        let paths = build_runtime_paths(
            &temporary.path().join("clip.srt"),
            &temporary.path().join("clip.srt"),
            Some(&blocked_root),
            None,
            "Auto",
            "Chinese",
            false,
        );
        let store = FileRuntimeStore::new(paths);

        let error = store.ensure_layout().expect_err("layout must fail");
        let CoreError::Storage(StorageError::Io {
            operation, path, ..
        }) = error
        else {
            panic!("expected contextual storage I/O error");
        };
        assert_eq!(operation, "create runtime directory");
        let path = path.expect("failed runtime path");
        assert!(path.contains(&blocked_root.display().to_string()));
    }

    #[test]
    fn saves_glossary_as_json_object() {
        let temporary = temp_root("glossary");
        let root = temporary.path().to_path_buf();
        let paths = build_runtime_paths(
            &root.join("clip.srt"),
            &root.join("clip.srt"),
            Some(&root),
            None,
            "Auto",
            "Chinese",
            false,
        );
        let store = FileRuntimeStore::new(paths);
        store
            .save_glossary(&[("Alice".to_owned(), "爱丽丝".to_owned())])
            .expect("save glossary");

        let content = fs::read_to_string(&store.paths().glossary_path).expect("read glossary");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(content, r#"{"Alice":"爱丽丝"}"#);
    }

    #[test]
    fn round_trips_glossary_via_save_and_load() {
        let temporary = temp_root("glossary-rt");
        let root = temporary.path().to_path_buf();
        let paths = build_runtime_paths(
            &root.join("clip.srt"),
            &root.join("clip.srt"),
            Some(&root),
            None,
            "Auto",
            "Chinese",
            false,
        );
        let store = FileRuntimeStore::new(paths);
        let entries = vec![
            ("Alice".to_owned(), "爱丽丝".to_owned()),
            ("Bob".to_owned(), "鲍勃".to_owned()),
        ];
        store.save_glossary(&entries).expect("save");
        let loaded = store.load_glossary().expect("load");
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains(&("Alice".to_owned(), "爱丽丝".to_owned())));
        assert!(loaded.contains(&("Bob".to_owned(), "鲍勃".to_owned())));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn invalid_glossary_value_reports_the_key() {
        let temporary = temp_root("glossary-invalid");
        let root = temporary.path().to_path_buf();
        let paths = build_runtime_paths(
            &root.join("clip.srt"),
            &root.join("clip.srt"),
            Some(&root),
            None,
            "Auto",
            "Chinese",
            false,
        );
        fs::create_dir_all(
            paths
                .glossary_path
                .parent()
                .expect("glossary parent directory"),
        )
        .expect("create glossary directory");
        fs::write(&paths.glossary_path, r#"{"Alice":42}"#).expect("write invalid glossary");
        let store = FileRuntimeStore::new(paths);

        let error = store
            .load_glossary()
            .expect_err("non-string glossary entry must fail");
        assert!(error.to_string().contains("glossary entry `Alice`"));
        assert!(error.to_string().contains("got number"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn invalid_translation_memory_value_reports_the_key() {
        let temporary = temp_root("translation-memory-invalid");
        let root = temporary.path().to_path_buf();
        let paths = build_runtime_paths(
            &root.join("clip.srt"),
            &root.join("clip.srt"),
            Some(&root),
            None,
            "Auto",
            "Chinese",
            false,
        );
        fs::create_dir_all(
            paths
                .translation_memory_path
                .parent()
                .expect("memory parent directory"),
        )
        .expect("create memory directory");
        fs::write(&paths.translation_memory_path, r#"{"ctx":false}"#)
            .expect("write invalid memory");
        let store = FileRuntimeStore::new(paths);

        let error = store
            .load_translation_memory()
            .expect_err("non-string translation memory entry must fail");
        assert!(error.to_string().contains("translation memory entry `ctx`"));
        assert!(error.to_string().contains("got boolean"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn reads_v4_translation_memory_until_v5_is_written() {
        let temporary = temp_root("translation-memory-v4");
        let root = temporary.path().to_path_buf();
        let paths = build_runtime_paths(
            &root.join("clip.srt"),
            &root.join("clip.srt"),
            Some(&root),
            None,
            "Auto",
            "Chinese",
            false,
        );
        let legacy_path = legacy_translation_memory_paths(&paths.translation_memory_path)
            .into_iter()
            .next()
            .expect("v5 path has a v4 predecessor");
        fs::create_dir_all(&root).expect("create root");
        fs::write(&legacy_path, r#"{"hello":"你好"}"#).expect("write v4 memory");
        let store = FileRuntimeStore::new(paths.clone());

        assert_eq!(
            store.load_translation_memory().expect("load v4 memory"),
            vec![("hello".to_owned(), "你好".to_owned())]
        );

        store
            .save_translation_memory(&[("ctx-v2:new:hello".to_owned(), "您好".to_owned())])
            .expect("write v5 memory");
        assert_eq!(
            store.load_translation_memory().expect("prefer v5 memory"),
            vec![("ctx-v2:new:hello".to_owned(), "您好".to_owned())]
        );
        assert!(legacy_path.exists());
        assert!(paths.translation_memory_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_glossary_returns_empty_when_file_missing() {
        let temporary = temp_root("glossary-empty");
        let root = temporary.path().to_path_buf();
        let paths = build_runtime_paths(
            &root.join("clip.srt"),
            &root.join("clip.srt"),
            Some(&root),
            None,
            "Auto",
            "Chinese",
            false,
        );
        let store = FileRuntimeStore::new(paths);
        let loaded = store.load_glossary().expect("load");
        assert!(loaded.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn saves_batch_segments_to_padded_shard_path() {
        let temporary = temp_root("batch");
        let root = temporary.path().to_path_buf();
        let paths = build_runtime_paths(
            &root.join("clip.srt"),
            &root.join("clip.srt"),
            Some(&root),
            None,
            "Auto",
            "Chinese",
            false,
        );
        let store = FileRuntimeStore::new(paths);
        let segment = SubtitleSegment {
            id: "1".to_owned(),
            text: "hello".to_owned(),
            start: Some("00:00:00,000".to_owned()),
            end: Some("00:00:01,000".to_owned()),
            identifier: Some("1".to_owned()),
            settings: None,
            semantic: Default::default(),
        };
        let path = store.batch_shard_path(BatchShardKind::Translated, 3);
        store
            .save_batch_segments(BatchShardKind::Translated, 3, &[segment])
            .expect("save shard");
        let content = fs::read_to_string(&path).expect("read shard");
        let _ = fs::remove_dir_all(&root);

        assert!(path.ends_with("0003.json"));
        assert!(content.contains(r#""batch_index":3"#));
        assert!(content.contains(r#""text":"hello""#));
        assert!(!content.contains(r#""semantic""#));
    }

    #[test]
    fn legacy_segment_without_semantic_metadata_remains_readable() {
        let value = serde_json::json!({
            "id": "1",
            "text": "hello",
            "start": null,
            "end": null,
            "identifier": null,
            "settings": null,
        });

        let segment = segment_from_json(&value).expect("legacy segment");

        assert!(segment.semantic.is_empty());
    }

    #[test]
    fn loads_batch_segments_in_order() {
        let temporary = temp_root("batch-load");
        let root = temporary.path().to_path_buf();
        let paths = build_runtime_paths(
            &root.join("clip.srt"),
            &root.join("clip.srt"),
            Some(&root),
            None,
            "Auto",
            "Chinese",
            false,
        );
        let store = FileRuntimeStore::new(paths);
        let mut first = SubtitleSegment {
            id: "1".to_owned(),
            text: "one".to_owned(),
            start: None,
            end: None,
            identifier: None,
            settings: None,
            semantic: Default::default(),
        };
        first.semantic.speaker = Some("Alice".to_owned());
        first.semantic.style = Some("Default".to_owned());
        let second = SubtitleSegment {
            id: "2".to_owned(),
            text: "two".to_owned(),
            start: None,
            end: None,
            identifier: None,
            settings: None,
            semantic: Default::default(),
        };

        store
            .save_batch_segments(BatchShardKind::Translated, 1, std::slice::from_ref(&first))
            .expect("save first");
        store
            .save_batch_segments(BatchShardKind::Translated, 2, std::slice::from_ref(&second))
            .expect("save second");

        let loaded = store
            .load_batch_segments(BatchShardKind::Translated, 2)
            .expect("load shards");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(loaded, vec![first, second]);
    }

    #[test]
    fn round_trips_versioned_run_state_shape() {
        let temporary = temp_root("run-state");
        let root = temporary.path().to_path_buf();
        let paths = build_runtime_paths(
            &root.join("clip.txt"),
            &root.join("clip.txt"),
            Some(&root),
            None,
            "Auto",
            "Chinese",
            false,
        );
        let store = FileRuntimeStore::new(paths);
        let mut options = PipelineOptions::new(root.join("clip.txt"));
        options.rendering.output_path = Some(root.join("clip.translated.txt"));
        let state = RunState::new(
            &options,
            input_signature_from_bytes(b"hello\n", Some(123)),
            Usage {
                input_tokens: 2,
                output_tokens: 3,
                total_tokens: 5,
                ..Usage::default()
            },
            ContextMemory::new(),
            1,
            0,
            true,
        );

        store.save_run_state(&state).expect("save state");
        let loaded = store
            .load_run_state()
            .expect("load state")
            .expect("state exists");
        let raw = fs::read_to_string(&store.paths().state_path).expect("read state");
        let value: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(loaded, state);
        assert_eq!(value["version"], 5);
        assert_eq!(value["translation_batches_completed"], 1);
        assert!(value.get("translated_segments").is_none());
        assert!(value.get("pipeline_fingerprint").is_none());
    }

    #[test]
    fn round_trips_python_compatible_request_cache_shape() {
        let temporary = temp_root("request-cache");
        let root = temporary.path().to_path_buf();
        let paths = build_runtime_paths(
            &root.join("clip.txt"),
            &root.join("clip.txt"),
            Some(&root),
            None,
            "Auto",
            "Chinese",
            false,
        );
        let store = FileRuntimeStore::new(paths);
        let response = BackendJsonResult {
            payload: BackendPayload::Translation(BatchTranslationResult {
                lines: vec![TranslationLine {
                    id: "1".to_owned(),
                    translation: "你好".to_owned(),
                }],
                summary: "greeting".to_owned(),
                glossary_updates: vec![GlossaryEntry {
                    source: "hello".to_owned(),
                    target: "你好".to_owned(),
                }],
                terminology_updates: Vec::new(),
            }),
            usage: Usage {
                input_tokens: 2,
                output_tokens: 3,
                total_tokens: 5,
                ..Usage::default()
            },
        };
        let hash = "8c13d80251241884e45610d3b6003c103e0421e5";

        store
            .save_cached_response(CacheStage::Translate, hash, &response)
            .expect("save cache");
        let loaded = store
            .load_cached_response(CacheStage::Translate, hash)
            .expect("load cache")
            .expect("cache exists");
        let path = store.cache_path(CacheStage::Translate, hash);
        let raw = fs::read_to_string(&path).expect("read cache");
        let value: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(loaded, response);
        assert_eq!(value["payload"]["lines"][0]["translation"], "你好");
        assert_eq!(value["payload"]["glossary_updates"][0]["source"], "hello");
        assert_eq!(value["usage"]["total_tokens"], 5);
        assert!(path.ends_with(format!("translate/{hash}.json")));
    }

    #[test]
    fn round_trips_python_compatible_review_cache_shape() {
        let temporary = temp_root("review-cache");
        let root = temporary.path().to_path_buf();
        let paths = build_runtime_paths(
            &root.join("clip.txt"),
            &root.join("clip.txt"),
            Some(&root),
            None,
            "Auto",
            "Chinese",
            false,
        );
        let store = FileRuntimeStore::new(paths);
        let response = BackendJsonResult {
            payload: BackendPayload::Review(ReviewResult {
                lines: vec![TranslationLine {
                    id: "1".to_owned(),
                    translation: "审校后".to_owned(),
                }],
                review_notes: "terminology fixed".to_owned(),
                annotations: Vec::new(),
            }),
            usage: Usage {
                input_tokens: 3,
                output_tokens: 2,
                total_tokens: 5,
                ..Usage::default()
            },
        };
        let hash = "review-hash";

        store
            .save_cached_response(CacheStage::Review, hash, &response)
            .expect("save review cache");
        let loaded = store
            .load_cached_response(CacheStage::Review, hash)
            .expect("load review cache")
            .expect("cache exists");
        let path = store.cache_path(CacheStage::Review, hash);
        let raw = fs::read_to_string(&path).expect("read cache");
        let value: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(loaded, response);
        assert_eq!(value["payload"]["lines"][0]["translation"], "审校后");
        assert_eq!(value["payload"]["review_notes"], "terminology fixed");
        assert!(path.ends_with("review/review-hash.json"));
    }

    #[test]
    fn round_trips_terminology_cache_and_writes_review_report() {
        let temporary = temp_root("terminology-cache");
        let root = temporary.path().to_path_buf();
        let paths = build_runtime_paths(
            &root.join("clip.srt"),
            &root.join("clip.srt"),
            Some(&root),
            None,
            "en",
            "zh-Hans",
            false,
        );
        let store = FileRuntimeStore::new(paths);
        let response = BackendJsonResult {
            payload: BackendPayload::Terminology(TerminologyPreflightResult {
                guide: DocumentGuide {
                    terminology: vec![TerminologyEntity {
                        canonical_source: "Axe Gang".to_owned(),
                        kind: TerminologyKind::Organization,
                        variants: vec![GlossaryEntry {
                            source: "Axe Gang".to_owned(),
                            target: "斧头帮".to_owned(),
                        }],
                    }],
                    ..DocumentGuide::default()
                },
            }),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 4,
                total_tokens: 14,
                ..Usage::default()
            },
        };
        store
            .save_cached_response(CacheStage::Terminology, "terms", &response)
            .expect("save terminology cache");
        assert_eq!(
            store
                .load_cached_response(CacheStage::Terminology, "terms")
                .expect("load terminology cache"),
            Some(response)
        );

        let report = ReviewReport {
            version: 2,
            terminology: TerminologyStats {
                candidates: 1,
                entries_added: 1,
                ..TerminologyStats::default()
            },
            review: ReviewStats {
                candidate_lines: 1,
                reviewed_lines: 1,
                changed_lines: 1,
                ..ReviewStats::default()
            },
            changes: vec![ReviewChange {
                batch: 1,
                id: "1".to_owned(),
                reasons: vec!["glossary mismatch".to_owned()],
                before: "帮派".to_owned(),
                after: "斧头帮".to_owned(),
                category: None,
                rationale: None,
            }],
            route: None,
        };
        store.save_review_report(&report).expect("save report");
        let saved: ReviewReport = serde_json::from_str(
            &fs::read_to_string(&store.paths().review_report_path).expect("read report"),
        )
        .expect("parse report");
        let loaded = store
            .load_review_report()
            .expect("load review report")
            .expect("saved review report");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(saved, report);
        assert_eq!(loaded, report);
    }

    #[test]
    fn reads_legacy_review_report_without_audit_fields() {
        let legacy = serde_json::json!({
            "terminology": {},
            "review": {"candidate_lines": 1, "reviewed_lines": 1, "changed_lines": 1},
            "changes": [{
                "batch": 1,
                "id": "1",
                "reasons": ["full review"],
                "before": "旧译",
                "after": "新译"
            }]
        });

        let report: ReviewReport = serde_json::from_value(legacy).expect("legacy report");

        assert_eq!(report.version, 0);
        assert!(report.route.is_none());
        assert!(report.changes[0].category.is_none());
        assert!(report.changes[0].rationale.is_none());
    }

    #[test]
    fn writes_python_compatible_failure_and_agent_logs() {
        let temporary = temp_root("recovery-logs");
        let root = temporary.path().to_path_buf();
        let paths = build_runtime_paths(
            &root.join("clip.txt"),
            &root.join("clip.txt"),
            Some(&root),
            None,
            "Auto",
            "Chinese",
            false,
        );
        let store = FileRuntimeStore::new(paths);
        let segment = SubtitleSegment {
            id: "1".to_owned(),
            text: "hello".to_owned(),
            start: None,
            end: None,
            identifier: None,
            settings: None,
            semantic: Default::default(),
        };
        let attempt = AttemptLog {
            attempt: 1,
            cached: false,
            error: Some("invalid output".to_owned()),
            payload: None,
            messages: vec![subbake_core::ports::ChatMessage::user("prompt")],
            split_retry: None,
        };
        let failure_path = store
            .save_failure_log(&FailureLog {
                stage: "translate".to_owned(),
                batch_index: 1,
                request_hash: "hash".to_owned(),
                batch_segments: vec![segment],
                messages: attempt.messages.clone(),
                translated_segments: Vec::new(),
                attempts: vec![attempt.clone()],
                agent_attempts: vec![attempt.clone()],
            })
            .expect("save failure");
        let agent_path = store
            .save_agent_log(&AgentLog {
                stage: "translate".to_owned(),
                batch_index: 1,
                success: false,
                attempts: vec![attempt],
                final_error: Some("invalid output".to_owned()),
            })
            .expect("save agent log");
        let failure: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&failure_path).expect("read failure"))
                .expect("failure json");
        let agent: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&agent_path).expect("read agent"))
                .expect("agent json");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(failure["stage"], "translate");
        assert_eq!(failure["attempts"][0]["messages"][0]["role"], "user");
        assert_eq!(failure["agent_attempts"].as_array().map(Vec::len), Some(1));
        assert_eq!(agent["success"], false);
        assert_eq!(agent["final_error"], "invalid output");
    }

    fn temp_root(label: &str) -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix(&format!("subbake-runtime-store-{label}-"))
            .tempdir()
            .expect("create temporary runtime store root")
    }
}
