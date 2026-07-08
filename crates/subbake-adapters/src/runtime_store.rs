use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use subbake_core::entities::SubtitleSegment;
use subbake_core::error::{CoreError, CoreResult};
use subbake_core::ports::{BatchShardKind, RuntimeStore};
use subbake_core::storage::{RunState, RuntimePaths};

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
        };
        root.join(format!("{batch_index:04}.json"))
    }
}

impl RuntimeStore for FileRuntimeStore {
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
            &self.paths.agent_logs_dir,
        ] {
            fs::create_dir_all(directory).map_err(storage_error)?;
        }
        Ok(())
    }

    fn save_glossary(&self, entries: &[(String, String)]) -> CoreResult<()> {
        write_json_verified(&self.paths.glossary_path, &string_map_json(entries))
    }

    fn save_translation_memory(&self, entries: &[(String, String)]) -> CoreResult<()> {
        write_json_verified(
            &self.paths.translation_memory_path,
            &string_map_json(entries),
        )
    }

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

    fn load_glossary(&self) -> CoreResult<Vec<(String, String)>> {
        if !self.paths.glossary_path.exists() {
            return Ok(Vec::new());
        }
        let text = fs::read_to_string(&self.paths.glossary_path).map_err(storage_error)?;
        let map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&text)
            .map_err(|error| CoreError::Data(format!("glossary parse failed: {error}",)))?;
        let entries: Vec<(String, String)> = map
            .into_iter()
            .filter_map(|(source, value)| value.as_str().map(|target| (source, target.to_owned())))
            .collect();
        Ok(entries)
    }

    fn load_translation_memory(&self) -> CoreResult<Vec<(String, String)>> {
        if !self.paths.translation_memory_path.exists() {
            return Ok(Vec::new());
        }
        let text =
            fs::read_to_string(&self.paths.translation_memory_path).map_err(storage_error)?;
        let map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&text).map_err(|error| {
                CoreError::Data(format!("translation memory parse failed: {error}",))
            })?;
        let entries: Vec<(String, String)> = map
            .into_iter()
            .filter_map(|(key, value)| {
                value
                    .as_str()
                    .map(|translation| (key, translation.to_owned()))
            })
            .collect();
        Ok(entries)
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
                return Err(CoreError::Data(format!(
                    "missing batch shard for resume: {}",
                    path.display()
                )));
            }
            let text = fs::read_to_string(&path).map_err(storage_error)?;
            let payload: serde_json::Value = serde_json::from_str(&text)
                .map_err(|error| CoreError::Data(format!("batch shard parse failed: {error}")))?;
            let segments = payload["segments"].as_array().ok_or_else(|| {
                CoreError::Data(format!("batch shard missing segments: {}", path.display()))
            })?;
            for segment in segments {
                loaded.push(segment_from_json(segment)?);
            }
        }
        Ok(loaded)
    }

    fn save_run_state(&self, state: &RunState) -> CoreResult<()> {
        let payload = serde_json::to_value(state)
            .map_err(|error| CoreError::Data(format!("serialize run state failed: {error}")))?;
        write_json_verified(&self.paths.state_path, &payload)
    }

    fn load_run_state(&self) -> CoreResult<Option<RunState>> {
        if !self.paths.state_path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&self.paths.state_path).map_err(storage_error)?;
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|error| CoreError::Data(format!("run state parse failed: {error}")))
    }
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
    serde_json::Value::Object(
        [
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
        .collect(),
    )
}

fn segment_from_json(value: &serde_json::Value) -> CoreResult<SubtitleSegment> {
    Ok(SubtitleSegment {
        id: value["id"].as_str().unwrap_or_default().to_owned(),
        text: value["text"].as_str().unwrap_or_default().to_owned(),
        start: optional_string_from_json(&value["start"]),
        end: optional_string_from_json(&value["end"]),
        identifier: optional_string_from_json(&value["identifier"]),
        settings: optional_string_from_json(&value["settings"]),
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
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(storage_error)?;
    }
    let serialized = serde_json::to_string(payload)
        .map_err(|error| CoreError::Data(format!("serialize json failed: {error}")))?;
    let temp_path = path.with_file_name(format!(
        "{}.tmp",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("runtime")
    ));
    fs::write(&temp_path, serialized.as_bytes()).map_err(storage_error)?;
    fs::rename(&temp_path, path).map_err(storage_error)?;
    let written = fs::read_to_string(path).map_err(storage_error)?;
    if written != serialized {
        return Err(CoreError::Data(format!(
            "write verification failed for {}",
            path.display()
        )));
    }
    Ok(())
}

fn storage_error(error: io::Error) -> CoreError {
    CoreError::Data(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use subbake_core::entities::{PipelineOptions, Usage};
    use subbake_core::memory::ContextMemory;
    use subbake_core::storage::{RunState, build_runtime_paths, input_signature_from_bytes};

    use super::*;

    #[test]
    fn saves_glossary_as_json_object() {
        let root = temp_root("glossary");
        let paths = build_runtime_paths(
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
        let root = temp_root("glossary-rt");
        let paths = build_runtime_paths(
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
    fn load_glossary_returns_empty_when_file_missing() {
        let root = temp_root("glossary-empty");
        let paths = build_runtime_paths(
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
        let root = temp_root("batch");
        let paths = build_runtime_paths(
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
    }

    #[test]
    fn loads_batch_segments_in_order() {
        let root = temp_root("batch-load");
        let paths = build_runtime_paths(
            &root.join("clip.srt"),
            Some(&root),
            None,
            "Auto",
            "Chinese",
            false,
        );
        let store = FileRuntimeStore::new(paths);
        let first = SubtitleSegment {
            id: "1".to_owned(),
            text: "one".to_owned(),
            start: None,
            end: None,
            identifier: None,
            settings: None,
        };
        let second = SubtitleSegment {
            id: "2".to_owned(),
            text: "two".to_owned(),
            start: None,
            end: None,
            identifier: None,
            settings: None,
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
    fn round_trips_python_compatible_run_state_shape() {
        let root = temp_root("run-state");
        let paths = build_runtime_paths(
            &root.join("clip.txt"),
            Some(&root),
            None,
            "Auto",
            "Chinese",
            false,
        );
        let store = FileRuntimeStore::new(paths);
        let mut options = PipelineOptions::new(root.join("clip.txt"));
        options.output_path = Some(root.join("clip.translated.txt"));
        let state = RunState::new(
            &options,
            input_signature_from_bytes(b"hello\n", Some(123)),
            Usage {
                input_tokens: 2,
                output_tokens: 3,
                total_tokens: 5,
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
        assert_eq!(value["version"], 3);
        assert_eq!(value["translation_batches_completed"], 1);
        assert!(value.get("translated_segments").is_none());
        assert!(value.get("pipeline_fingerprint").is_none());
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-runtime-store-{label}-{nanos}"))
    }
}
