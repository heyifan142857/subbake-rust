//! Provider-managed, recoverable asynchronous subtitle translation.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};
use subbake_core::formats::RenderOptions;
use subbake_core::overnight::{OvernightBatch, parse_translation_output, plan_translation};
use subbake_core::storage::{InputSignature, build_runtime_paths, input_signature_from_bytes};
use subbake_core::{CancellationGuard, SubtitleSegment};

use crate::error::{AdapterError, AdapterResult};
use crate::fs::{
    default_output_path_with_language, is_supported_subtitle_path, read_document,
    render_and_write_document, stable_runtime_input_path,
};
use crate::llm_backends::{
    OpenAiBatchClient, OpenAiBatchStatus, build_openai_batch_client, default_timeout_seconds,
};
use crate::settings::TranslationSettings;

const MANIFEST_VERSION: u64 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct OvernightSubmitRequest {
    pub input_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub settings: TranslationSettings,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OvernightStatusRequest {
    pub manifest_path: PathBuf,
    pub settings: TranslationSettings,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OvernightCollectRequest {
    pub manifest_path: PathBuf,
    pub settings: TranslationSettings,
    pub overwrite: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OvernightSubmitOutcome {
    pub manifest_path: PathBuf,
    pub job_id: String,
    pub requests: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OvernightStatusOutcome {
    pub manifest_path: PathBuf,
    pub job_id: String,
    pub status: String,
    pub completed: Option<usize>,
    pub failed: Option<usize>,
    pub total: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OvernightCollectOutcome {
    pub manifest_path: PathBuf,
    pub output_path: PathBuf,
    pub translated_segments: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OvernightManifest {
    version: u64,
    job_id: String,
    input_file_id: String,
    endpoint: String,
    status: String,
    output_file_id: Option<String>,
    error_file_id: Option<String>,
    input_path: PathBuf,
    output_path: PathBuf,
    input_signature: InputSignature,
    source_language: String,
    target_language: String,
    bilingual: bool,
    output_format: Option<String>,
    batches: Vec<ManifestBatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestBatch {
    custom_id: String,
    segment_ids: Vec<String>,
}

pub fn submit_overnight(
    request: OvernightSubmitRequest,
    cancellation: &CancellationGuard,
) -> AdapterResult<OvernightSubmitOutcome> {
    cancellation.check().map_err(AdapterError::from)?;
    if request.settings.translation.mode != subbake_core::TranslationMode::Economy {
        return Err(AdapterError::invalid_input(
            "overnight is economy-only; pass --mode economy to submit a recoverable batch job",
        ));
    }
    if !is_supported_subtitle_path(&request.input_path) {
        return Err(AdapterError::invalid_input(
            "overnight submit accepts subtitle files only",
        ));
    }
    let input_bytes = fs::read(&request.input_path)?;
    let input_signature = input_signature(&request.input_path, &input_bytes)?;
    let document = read_document(&request.input_path)?;
    let output_path = request
        .output_path
        .unwrap_or(default_output_path_with_language(
            &request.input_path,
            request.settings.output.format.as_deref(),
            request.settings.output.bilingual,
            None,
        )?);
    let stable = stable_runtime_input_path(&request.input_path)?;
    let runtime = build_runtime_paths(
        &request.input_path,
        &stable,
        request.settings.runtime_dir(),
        request.settings.glossary_path(),
        &request.settings.translation.source_language,
        &request.settings.translation.target_language,
        true,
    );
    let prefix = format!(
        "subbake-{}",
        &input_signature.sha1[..12.min(input_signature.sha1.len())]
    );
    let batches = plan_translation(
        &document,
        &request.settings.translation.source_language,
        &request.settings.translation.target_language,
        request.settings.translation.batch_size,
        request.settings.translation.batch_token_budget,
        &prefix,
    )?;
    let client = build_openai_batch_client(
        &request.settings.backend_config(),
        default_timeout_seconds(),
    )?;
    let jsonl = batch_jsonl(&client, &batches)?;
    let status = client.submit_jsonl(
        &jsonl,
        serde_json::json!({"application":"subbake", "kind":"subtitle_translation", "mode":"economy"}),
        cancellation,
    )?;
    let manifest = OvernightManifest {
        version: MANIFEST_VERSION,
        job_id: status.id.clone(),
        input_file_id: status.input_file_id.clone(),
        endpoint: client.endpoint_path().to_owned(),
        status: status.status,
        output_file_id: status.output_file_id,
        error_file_id: status.error_file_id,
        input_path: request.input_path,
        output_path,
        input_signature,
        source_language: request.settings.translation.source_language,
        target_language: request.settings.translation.target_language,
        bilingual: request.settings.output.bilingual,
        output_format: request.settings.output.format,
        batches: batches
            .into_iter()
            .map(|batch| ManifestBatch {
                custom_id: batch.custom_id,
                segment_ids: batch.segment_ids,
            })
            .collect(),
    };
    let manifest_path = runtime
        .root_dir
        .join("overnight")
        .join(format!("{}.json", manifest.job_id));
    write_manifest(&manifest_path, &manifest)?;
    Ok(OvernightSubmitOutcome {
        manifest_path,
        job_id: manifest.job_id,
        requests: manifest.batches.len(),
    })
}

pub fn overnight_status(
    request: OvernightStatusRequest,
    cancellation: &CancellationGuard,
) -> AdapterResult<OvernightStatusOutcome> {
    let mut manifest = read_manifest(&request.manifest_path)?;
    let client = client_for_manifest(&request.settings, &manifest)?;
    let status = client.status(&manifest.job_id, cancellation)?;
    update_manifest_status(&mut manifest, &status);
    write_manifest(&request.manifest_path, &manifest)?;
    Ok(status_outcome(&request.manifest_path, &status))
}

pub fn collect_overnight(
    request: OvernightCollectRequest,
    cancellation: &CancellationGuard,
) -> AdapterResult<OvernightCollectOutcome> {
    let mut manifest = read_manifest(&request.manifest_path)?;
    let client = client_for_manifest(&request.settings, &manifest)?;
    let status = client.status(&manifest.job_id, cancellation)?;
    update_manifest_status(&mut manifest, &status);
    write_manifest(&request.manifest_path, &manifest)?;
    if status.status != "completed" {
        return Err(AdapterError::invalid_input(format!(
            "overnight job {} is {}; collect requires completed",
            status.id, status.status
        )));
    }
    if status.failed.unwrap_or(0) > 0 {
        return Err(AdapterError::invalid_input(format!(
            "overnight job {} completed with {} failed request(s); inspect its error file before collecting",
            status.id,
            status.failed.unwrap_or(0)
        )));
    }
    let output_file_id = status
        .output_file_id
        .as_deref()
        .ok_or_else(|| AdapterError::invalid_input("completed overnight job has no output file"))?;
    let now_bytes = fs::read(&manifest.input_path)?;
    if input_signature(&manifest.input_path, &now_bytes)? != manifest.input_signature {
        return Err(AdapterError::invalid_input(
            "subtitle input changed after overnight submission; refusing to apply results to different content",
        ));
    }
    if manifest.output_path.exists() && !request.overwrite {
        return Err(AdapterError::invalid_input(format!(
            "output already exists: {}",
            manifest.output_path.display()
        )));
    }
    let document = read_document(&manifest.input_path)?;
    let raw = client.download_output(output_file_id, cancellation)?;
    let translations = parse_output_lines(&client, &manifest, &raw)?;
    let translated = apply_lines(&document.segments, &translations)?;
    let render_options = RenderOptions::new(manifest.bilingual, manifest.output_format.clone());
    render_and_write_document(
        &document,
        &translated,
        &manifest.output_path,
        &render_options,
    )?;
    Ok(OvernightCollectOutcome {
        manifest_path: request.manifest_path,
        output_path: manifest.output_path,
        translated_segments: translated.len(),
    })
}

fn batch_jsonl(client: &OpenAiBatchClient, batches: &[OvernightBatch]) -> AdapterResult<String> {
    batches
        .iter()
        .map(|batch| {
            serde_json::to_string(&serde_json::json!({
                "custom_id": batch.custom_id, "method": "POST", "url": client.endpoint_path(),
                "body": client.request_body(&batch.messages),
            }))
            .map_err(|source| AdapterError::Serialization {
                context: "encode overnight JSONL request",
                source,
            })
        })
        .collect::<AdapterResult<Vec<_>>>()
        .map(|lines| lines.join("\n") + "\n")
}

fn parse_output_lines(
    client: &OpenAiBatchClient,
    manifest: &OvernightManifest,
    raw: &str,
) -> AdapterResult<Vec<subbake_core::TranslationLine>> {
    let expected = manifest
        .batches
        .iter()
        .map(|batch| (batch.custom_id.as_str(), batch))
        .collect::<BTreeMap<_, _>>();
    let mut result = Vec::new();
    let mut seen = BTreeMap::new();
    for line in raw.lines().filter(|line| !line.trim().is_empty()) {
        let value: serde_json::Value =
            serde_json::from_str(line).map_err(|source| AdapterError::Serialization {
                context: "decode overnight output JSONL",
                source,
            })?;
        let custom_id = value["custom_id"]
            .as_str()
            .ok_or_else(|| AdapterError::invalid_input("overnight output entry missing custom_id"))?
            .to_owned();
        let batch = expected.get(custom_id.as_str()).ok_or_else(|| {
            AdapterError::invalid_input(format!(
                "overnight output contains unknown custom_id `{custom_id}`"
            ))
        })?;
        if seen.insert(custom_id.clone(), ()).is_some() {
            return Err(AdapterError::invalid_input(format!(
                "overnight output repeats custom_id `{custom_id}`"
            )));
        }
        let code = value["response"]["status_code"].as_u64().unwrap_or(0);
        if !(200..300).contains(&code) {
            return Err(AdapterError::invalid_input(format!(
                "overnight request `{custom_id}` failed with status {code}"
            )));
        }
        let payload = client.parse_output_json(&value["response"]["body"])?;
        let planned = OvernightBatch {
            custom_id: batch.custom_id.clone(),
            segment_ids: batch.segment_ids.clone(),
            messages: Vec::new(),
        };
        result.extend(parse_translation_output(&planned, &payload)?);
    }
    if seen.len() != manifest.batches.len() {
        return Err(AdapterError::invalid_input(format!(
            "overnight output is incomplete: received {} of {} requests",
            seen.len(),
            manifest.batches.len()
        )));
    }
    Ok(result)
}

fn apply_lines(
    source: &[SubtitleSegment],
    lines: &[subbake_core::TranslationLine],
) -> AdapterResult<Vec<SubtitleSegment>> {
    let translations = lines
        .iter()
        .map(|line| (line.id.as_str(), line.translation.as_str()))
        .collect::<BTreeMap<_, _>>();
    source
        .iter()
        .map(|segment| {
            let translation = translations.get(segment.id.as_str()).ok_or_else(|| {
                AdapterError::invalid_input(format!(
                    "overnight output omitted subtitle id `{}`",
                    segment.id
                ))
            })?;
            let mut translated = segment.clone();
            translated.text = (*translation).to_owned();
            Ok(translated)
        })
        .collect()
}

fn client_for_manifest(
    settings: &TranslationSettings,
    manifest: &OvernightManifest,
) -> AdapterResult<OpenAiBatchClient> {
    let client = build_openai_batch_client(&settings.backend_config(), default_timeout_seconds())?;
    if client.endpoint_path() != manifest.endpoint {
        return Err(AdapterError::invalid_input(
            "configured OpenAI API format does not match the overnight manifest",
        ));
    }
    Ok(client)
}

fn update_manifest_status(manifest: &mut OvernightManifest, status: &OpenAiBatchStatus) {
    manifest.status = status.status.clone();
    manifest.output_file_id = status.output_file_id.clone();
    manifest.error_file_id = status.error_file_id.clone();
}

fn status_outcome(path: &Path, status: &OpenAiBatchStatus) -> OvernightStatusOutcome {
    OvernightStatusOutcome {
        manifest_path: path.to_path_buf(),
        job_id: status.id.clone(),
        status: status.status.clone(),
        completed: status.completed,
        failed: status.failed,
        total: status.total,
    }
}

fn input_signature(path: &Path, bytes: &[u8]) -> AdapterResult<InputSignature> {
    let mtime_ns = fs::metadata(path)?
        .modified()
        .ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .map(|value| value.as_nanos());
    Ok(input_signature_from_bytes(bytes, mtime_ns))
}

fn read_manifest(path: &Path) -> AdapterResult<OvernightManifest> {
    let text = fs::read_to_string(path).map_err(|source| {
        AdapterError::external_io("read overnight manifest", Some(path.to_path_buf()), source)
    })?;
    let manifest: OvernightManifest =
        serde_json::from_str(&text).map_err(|source| AdapterError::Serialization {
            context: "decode overnight manifest",
            source,
        })?;
    if manifest.version != MANIFEST_VERSION {
        return Err(AdapterError::invalid_input(format!(
            "unsupported overnight manifest version {}",
            manifest.version
        )));
    }
    Ok(manifest)
}

fn write_manifest(path: &Path, manifest: &OvernightManifest) -> AdapterResult<()> {
    let parent = path
        .parent()
        .ok_or_else(|| AdapterError::invalid_input("overnight manifest has no parent directory"))?;
    fs::create_dir_all(parent).map_err(|source| {
        AdapterError::external_io(
            "create overnight manifest directory",
            Some(parent.to_path_buf()),
            source,
        )
    })?;
    let data =
        serde_json::to_vec_pretty(manifest).map_err(|source| AdapterError::Serialization {
            context: "encode overnight manifest",
            source,
        })?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, data).map_err(|source| {
        AdapterError::external_io("write overnight manifest", Some(temporary.clone()), source)
    })?;
    fs::rename(&temporary, path).map_err(|source| {
        AdapterError::external_io(
            "commit overnight manifest",
            Some(path.to_path_buf()),
            source,
        )
    })?;
    Ok(())
}
