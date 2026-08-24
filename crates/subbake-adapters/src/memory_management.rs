use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use subbake_core::ports::{RuntimeLayoutStore, RuntimeMemoryStore};
use subbake_core::storage::build_runtime_paths;

use crate::error::{AdapterError, AdapterResult};
use crate::fs::stable_runtime_input_path;
use crate::runtime_store::FileRuntimeStore;
use crate::settings::ResolvedSettings;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryAction {
    Inspect,
    Export { path: PathBuf },
    Import { path: PathBuf },
    Prune { yes: bool },
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRequest {
    pub action: MemoryAction,
    pub target_path: PathBuf,
    pub settings: ResolvedSettings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryOutcome {
    pub glossary_entries: usize,
    pub translation_memory_entries: usize,
    pub changed_entries: usize,
    pub bundle_path: Option<PathBuf>,
    pub glossary_path: PathBuf,
    pub translation_memory_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct MemoryBundle {
    version: u64,
    glossary: BTreeMap<String, String>,
    translation_memory: BTreeMap<String, String>,
}

pub fn manage_memory(request: MemoryRequest) -> AdapterResult<MemoryOutcome> {
    let stable = stable_runtime_input_path(&request.target_path)?;
    let paths = build_runtime_paths(
        &request.target_path,
        &stable,
        request.settings.storage.runtime_dir.as_deref(),
        request.settings.storage.glossary_path.as_deref(),
        &request.settings.translation.source_language,
        &request.settings.translation.target_language,
        request.settings.translation.mode == subbake_core::TranslationMode::Economy,
    );
    let store = FileRuntimeStore::new(paths.clone());
    store.ensure_layout().map_err(AdapterError::from)?;
    let mut glossary = store
        .load_glossary()
        .map_err(AdapterError::from)?
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let mut translation_memory = store
        .load_translation_memory()
        .map_err(AdapterError::from)?
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let mut changed_entries = 0usize;
    let mut bundle_path = None;

    match request.action {
        MemoryAction::Inspect => {}
        MemoryAction::Export { path } => {
            let bundle = MemoryBundle {
                version: 1,
                glossary: glossary.clone(),
                translation_memory: translation_memory.clone(),
            };
            let bytes = serde_json::to_vec_pretty(&bundle).map_err(|source| {
                AdapterError::Serialization {
                    context: "serialize memory bundle",
                    source,
                }
            })?;
            fs::write(&path, bytes).map_err(|source| {
                AdapterError::external_io("write memory bundle", Some(path.clone()), source)
            })?;
            bundle_path = Some(path);
        }
        MemoryAction::Import { path } => {
            let bytes = fs::read(&path).map_err(|source| {
                AdapterError::external_io("read memory bundle", Some(path.clone()), source)
            })?;
            let bundle: MemoryBundle =
                serde_json::from_slice(&bytes).map_err(|source| AdapterError::Serialization {
                    context: "parse memory bundle",
                    source,
                })?;
            if bundle.version != 1 {
                return Err(AdapterError::invalid_input(format!(
                    "unsupported memory bundle version {}",
                    bundle.version
                )));
            }
            for (source, target) in bundle.glossary {
                if !source.trim().is_empty()
                    && !target.trim().is_empty()
                    && !glossary.contains_key(&source)
                {
                    glossary.insert(source, target);
                    changed_entries += 1;
                }
            }
            for (source, target) in bundle.translation_memory {
                if !source.trim().is_empty()
                    && !target.trim().is_empty()
                    && !translation_memory.contains_key(&source)
                {
                    translation_memory.insert(source, target);
                    changed_entries += 1;
                }
            }
            save_maps(&store, &glossary, &translation_memory)?;
            bundle_path = Some(path);
        }
        MemoryAction::Prune { yes } => {
            if !yes {
                return Err(AdapterError::invalid_input("memory prune requires --yes"));
            }
            let glossary_before = glossary.len();
            glossary
                .retain(|source, target| !source.trim().is_empty() && !target.trim().is_empty());
            let memory_before = translation_memory.len();
            translation_memory
                .retain(|source, target| !source.trim().is_empty() && !target.trim().is_empty());
            changed_entries =
                glossary_before - glossary.len() + memory_before - translation_memory.len();
            save_maps(&store, &glossary, &translation_memory)?;
        }
    }

    Ok(MemoryOutcome {
        glossary_entries: glossary.len(),
        translation_memory_entries: translation_memory.len(),
        changed_entries,
        bundle_path,
        glossary_path: paths.glossary_path,
        translation_memory_path: paths.translation_memory_path,
    })
}

fn save_maps(
    store: &FileRuntimeStore,
    glossary: &BTreeMap<String, String>,
    translation_memory: &BTreeMap<String, String>,
) -> AdapterResult<()> {
    store
        .save_glossary(
            &glossary
                .iter()
                .map(|(source, target)| (source.clone(), target.clone()))
                .collect::<Vec<_>>(),
        )
        .map_err(AdapterError::from)?;
    store
        .save_translation_memory(
            &translation_memory
                .iter()
                .map(|(source, target)| (source.clone(), target.clone()))
                .collect::<Vec<_>>(),
        )
        .map_err(AdapterError::from)
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn versioned_bundle_import_export_and_prune_round_trip() {
        let root = std::env::temp_dir().join(format!(
            "subbake-memory-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("create root");
        let target = root.join("episode.srt");
        fs::write(&target, "1\n00:00:00,000 --> 00:00:01,000\nHello\n").expect("write target");
        let bundle_path = root.join("import.json");
        fs::write(
            &bundle_path,
            serde_json::to_vec_pretty(&MemoryBundle {
                version: 1,
                glossary: BTreeMap::from([
                    ("Alice".to_owned(), "爱丽丝".to_owned()),
                    ("Same".to_owned(), "Same".to_owned()),
                ]),
                translation_memory: BTreeMap::from([("hello".to_owned(), "你好".to_owned())]),
            })
            .expect("serialize bundle"),
        )
        .expect("write bundle");
        let mut settings = ResolvedSettings::default();
        settings.storage.runtime_dir = Some(root.join("runtime"));

        let imported = manage_memory(MemoryRequest {
            action: MemoryAction::Import {
                path: bundle_path.clone(),
            },
            target_path: target.clone(),
            settings: settings.clone(),
        })
        .expect("import memory");
        assert_eq!(imported.glossary_entries, 2);
        assert_eq!(imported.translation_memory_entries, 1);

        let mut raw_glossary =
            serde_json::from_slice::<serde_json::Map<String, serde_json::Value>>(
                &fs::read(&imported.glossary_path).expect("read glossary"),
            )
            .expect("parse glossary");
        raw_glossary.insert("blank".to_owned(), serde_json::Value::String(String::new()));
        fs::write(
            &imported.glossary_path,
            serde_json::to_vec_pretty(&raw_glossary).expect("serialize glossary"),
        )
        .expect("write blank glossary entry");

        let pruned = manage_memory(MemoryRequest {
            action: MemoryAction::Prune { yes: true },
            target_path: target.clone(),
            settings: settings.clone(),
        })
        .expect("prune memory");
        assert_eq!(pruned.changed_entries, 1);
        assert_eq!(pruned.glossary_entries, 2);

        let export_path = root.join("export.json");
        manage_memory(MemoryRequest {
            action: MemoryAction::Export {
                path: export_path.clone(),
            },
            target_path: target,
            settings,
        })
        .expect("export memory");
        let exported: MemoryBundle =
            serde_json::from_slice(&fs::read(export_path).expect("read export"))
                .expect("parse export");
        let _ = fs::remove_dir_all(root);
        assert_eq!(exported.version, 1);
        assert_eq!(exported.glossary.get("Alice"), Some(&"爱丽丝".to_owned()));
    }
}
