use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use subbake_core::entities::SubtitleSegment;
use subbake_core::error::{CoreError, CoreResult};
use subbake_core::ports::{BatchShardKind, RuntimeStore};
use subbake_core::storage::{JsonValue, RuntimePaths, canonical_json};

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
        let payload = JsonValue::Object(vec![
            ("batch_index".to_owned(), JsonValue::from(batch_index)),
            (
                "segments".to_owned(),
                JsonValue::Array(segments.iter().map(segment_json).collect()),
            ),
        ]);
        write_json_verified(&path, &payload)
    }

    fn load_glossary(&self) -> CoreResult<Vec<(String, String)>> {
        if !self.paths.glossary_path.exists() {
            return Ok(Vec::new());
        }
        let text = fs::read_to_string(&self.paths.glossary_path)
            .map_err(storage_error)?;
        let map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&text)
                .map_err(|error| CoreError::Data(format!(
                    "glossary parse failed: {error}",
                )))?;
        let entries: Vec<(String, String)> = map
            .into_iter()
            .filter_map(|(source, value)| {
                value
                    .as_str()
                    .map(|target| (source, target.to_owned()))
            })
            .collect();
        Ok(entries)
    }

    fn load_translation_memory(&self) -> CoreResult<Vec<(String, String)>> {
        if !self.paths.translation_memory_path.exists() {
            return Ok(Vec::new());
        }
        let text = fs::read_to_string(&self.paths.translation_memory_path)
            .map_err(storage_error)?;
        let map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&text)
                .map_err(|error| CoreError::Data(format!(
                    "translation memory parse failed: {error}",
                )))?;
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
}

fn string_map_json(entries: &[(String, String)]) -> JsonValue {
    JsonValue::Object(
        entries
            .iter()
            .map(|(key, value)| (key.clone(), JsonValue::String(value.clone())))
            .collect(),
    )
}

fn segment_json(segment: &SubtitleSegment) -> JsonValue {
    JsonValue::Object(vec![
        ("id".to_owned(), JsonValue::String(segment.id.clone())),
        ("text".to_owned(), JsonValue::String(segment.text.clone())),
        ("start".to_owned(), option_string_json(&segment.start)),
        ("end".to_owned(), option_string_json(&segment.end)),
        (
            "identifier".to_owned(),
            option_string_json(&segment.identifier),
        ),
        ("settings".to_owned(), option_string_json(&segment.settings)),
    ])
}

fn option_string_json(value: &Option<String>) -> JsonValue {
    value
        .as_ref()
        .map(|value| JsonValue::String(value.clone()))
        .unwrap_or(JsonValue::Null)
}

fn write_json_verified(path: &Path, payload: &JsonValue) -> CoreResult<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(storage_error)?;
    }
    let serialized = canonical_json(payload);
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

    use subbake_core::storage::build_runtime_paths;

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

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-runtime-store-{label}-{nanos}"))
    }
}
