use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};
use subbake_core::formats::RenderOptions;
use subbake_core::storage::{
    InputSignature, JsonValue, build_translation_fingerprint, input_signature_from_bytes,
    stable_hash,
};
use subbake_core::{
    CancellationGuard, ConfirmedTranslationContext, NoopProgress, PipelineResult, QualityGate,
    QualityPolicy, SharedProgress, SubtitleDocument, SubtitleDocumentMetadata, SubtitleSegment,
    TranslationMode, Usage, inspect_quality,
};

use crate::embedded_subtitles::{
    has_translatable_text_subtitle, is_supported_subtitle_container_path,
    translate_embedded_subtitle_cancellable_with_progress_and_quality,
};
use crate::error::{AdapterError, AdapterResult};
use crate::fs::{
    default_output_path_with_language, is_supported_subtitle_path, read_document,
    render_and_write_document_atomic, stable_runtime_input_path, write_file_atomically,
};
use crate::settings::ResolvedSettings;
use crate::transcription::{
    IncrementalTranscriptionOutcome, StableTranscriptionChunk, TranscriptionChunkObserver,
    TranscriptionRequest, TranscriptionSettings, transcribe_media_cancellable_with_progress,
    transcribe_media_incremental_with_progress,
};
use crate::translation::{
    TranslationInputIdentity, TranslationOutcome, TranslationRequest,
    translate_subtitle_cancellable_with_progress_and_identity,
};

const STREAMING_PIPELINE_VERSION: u64 = 3;
const STREAMING_CHANNEL_CAPACITY: usize = 2;

#[derive(Debug, Clone, PartialEq)]
pub struct PipelineRequest {
    pub input_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub settings: ResolvedSettings,
    pub transcription_settings: TranscriptionSettings,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineOutcome {
    Subtitle(TranslationOutcome),
    Media {
        translation: TranslationOutcome,
        transcription: subbake_core::transcription::TranscriptionReport,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MediaPipelineStrategy {
    Sequential,
    TurboImmediate,
    EconomyBuffered,
}

impl MediaPipelineStrategy {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::TurboImmediate => "turbo_immediate",
            Self::EconomyBuffered => "economy_buffered",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PipelineContract {
    version: u64,
    fingerprint: String,
    strategy: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TranscriptionChunkShard {
    version: u64,
    fingerprint: String,
    index: usize,
    input_start_ms: u64,
    input_end_ms: u64,
    core_start_ms: u64,
    core_end_ms: u64,
    format: String,
    segments: Vec<SubtitleSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TranslationGroupShard {
    source_fingerprint: String,
    version: u64,
    fingerprint: String,
    index: usize,
    first_chunk: usize,
    last_chunk: usize,
    source_first_id: String,
    source_last_id: String,
    subtitle_entries: usize,
    batches_translated: usize,
    review_batches: usize,
    usage: Usage,
    cache_hits: usize,
    resumed_translation_batches: usize,
    resumed_review_batches: usize,
    translation_memory_hits: usize,
    deduplicated_segments: usize,
    translated_path: PathBuf,
}

#[derive(Debug, Clone)]
struct StreamingPipelineStore {
    root: PathBuf,
    fingerprint: String,
}

impl StreamingPipelineStore {
    fn open(request: &PipelineRequest, strategy: MediaPipelineStrategy) -> AdapterResult<Self> {
        let fingerprint = streaming_fingerprint(request, strategy)?;
        let runtime_root = request
            .settings
            .storage
            .runtime_dir
            .clone()
            .or_else(|| request.transcription_settings.runtime_dir.clone())
            .unwrap_or_else(|| {
                request
                    .input_path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(".subbake")
            });
        let stable_input = stable_runtime_input_path(&request.input_path)?;
        let source_key = stable_hash(&JsonValue::Object(vec![(
            "path".to_owned(),
            JsonValue::String(stable_input.to_string_lossy().into_owned()),
        )]));
        let stem = request
            .input_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("media");
        let safe_stem: String = stem
            .chars()
            .map(|value| {
                if value.is_ascii_alphanumeric() || matches!(value, '-' | '_') {
                    value
                } else {
                    '-'
                }
            })
            .collect();
        let root = runtime_root
            .join("pipelines")
            .join(format!("{safe_stem}-{}", &source_key[..12]))
            .join(&fingerprint[..12]);
        fs::create_dir_all(root.join("transcription_chunks")).map_err(|source| {
            AdapterError::external_io(
                "create pipeline transcription shard directory",
                Some(root.clone()),
                source,
            )
        })?;
        fs::create_dir_all(root.join("translation_groups")).map_err(|source| {
            AdapterError::external_io(
                "create pipeline translation shard directory",
                Some(root.clone()),
                source,
            )
        })?;
        let store = Self { root, fingerprint };
        store.write_json(
            &store.root.join("contract.json"),
            &PipelineContract {
                version: STREAMING_PIPELINE_VERSION,
                fingerprint: store.fingerprint.clone(),
                strategy: strategy.as_str().to_owned(),
            },
        )?;
        Ok(store)
    }

    fn chunk_path(&self, index: usize) -> PathBuf {
        self.root
            .join("transcription_chunks")
            .join(format!("{index:04}.json"))
    }

    fn group_dir(&self, index: usize) -> PathBuf {
        self.root
            .join("translation_groups")
            .join(format!("{index:04}"))
    }

    fn save_chunk(&self, chunk: &StableTranscriptionChunk) -> AdapterResult<()> {
        self.write_json(
            &self.chunk_path(chunk.descriptor.index),
            &TranscriptionChunkShard {
                version: STREAMING_PIPELINE_VERSION,
                fingerprint: self.fingerprint.clone(),
                index: chunk.descriptor.index,
                input_start_ms: chunk.descriptor.input_start_ms,
                input_end_ms: chunk.descriptor.input_end_ms,
                core_start_ms: chunk.descriptor.core_start_ms,
                core_end_ms: chunk.descriptor.core_end_ms,
                format: chunk.format.extension().to_owned(),
                segments: chunk.segments.clone(),
            },
        )
    }

    fn save_group(&self, shard: &TranslationGroupShard) -> AdapterResult<()> {
        self.write_json(&self.group_dir(shard.index).join("complete.json"), shard)
    }

    fn mark_complete(&self, output_path: &Path, chunks: usize, groups: usize) -> AdapterResult<()> {
        #[derive(Serialize)]
        struct Complete<'a> {
            version: u64,
            fingerprint: &'a str,
            output_path: &'a Path,
            chunks: usize,
            groups: usize,
        }
        self.write_json(
            &self.root.join("complete.json"),
            &Complete {
                version: STREAMING_PIPELINE_VERSION,
                fingerprint: &self.fingerprint,
                output_path,
                chunks,
                groups,
            },
        )
    }

    fn write_json<T: Serialize>(&self, path: &Path, value: &T) -> AdapterResult<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| {
                AdapterError::external_io(
                    "create pipeline Resume directory",
                    Some(parent.to_path_buf()),
                    source,
                )
            })?;
        }
        let bytes =
            serde_json::to_vec_pretty(value).map_err(|source| AdapterError::Serialization {
                context: "serialize pipeline Resume shard",
                source,
            })?;
        write_file_atomically(path, &bytes)
    }
}

struct ChannelChunkObserver {
    store: StreamingPipelineStore,
    sender: SyncSender<StableTranscriptionChunk>,
}

impl TranscriptionChunkObserver for ChannelChunkObserver {
    fn stable(&mut self, chunk: StableTranscriptionChunk) -> AdapterResult<()> {
        self.store.save_chunk(&chunk)?;
        self.sender
            .send(chunk)
            .map_err(|_| AdapterError::invalid_input("incremental translation consumer stopped"))
    }
}

pub fn run_pipeline(request: PipelineRequest) -> AdapterResult<PipelineOutcome> {
    run_pipeline_cancellable_with_progress(
        request,
        &CancellationGuard::never(),
        std::sync::Arc::new(NoopProgress),
    )
}

pub fn run_pipeline_cancellable_with_progress(
    request: PipelineRequest,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
) -> AdapterResult<PipelineOutcome> {
    run_pipeline_cancellable_with_progress_and_quality(
        request,
        cancellation,
        progress,
        QualityGate::Never,
    )
}

pub fn run_pipeline_cancellable_with_progress_and_quality(
    mut request: PipelineRequest,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
    quality_gate: QualityGate,
) -> AdapterResult<PipelineOutcome> {
    crate::transcription::apply_whisper_configuration(
        &mut request.transcription_settings,
        &request.settings,
    );
    if is_supported_subtitle_path(&request.input_path) {
        let outcome = crate::translation::translate_subtitle_cancellable_with_progress_and_quality(
            TranslationRequest {
                input_path: request.input_path,
                output_path: request.output_path,
                output_language_tag: None,
                overwrite: true,
                runtime_reuse: crate::translation::RuntimeReusePolicy::Configured,
                settings: request.settings,
            },
            cancellation,
            progress,
            quality_gate,
        )?;
        return Ok(PipelineOutcome::Subtitle(outcome));
    }

    if is_supported_subtitle_container_path(&request.input_path)
        && should_translate_embedded_subtitle(
            &request.settings,
            has_translatable_text_subtitle(&request.input_path, cancellation)?,
        )
    {
        let outcome = translate_embedded_subtitle_cancellable_with_progress_and_quality(
            TranslationRequest {
                input_path: request.input_path,
                output_path: request.output_path,
                output_language_tag: None,
                overwrite: true,
                runtime_reuse: crate::translation::RuntimeReusePolicy::Configured,
                settings: request.settings,
            },
            cancellation,
            progress,
            quality_gate,
        )?;
        return Ok(PipelineOutcome::Subtitle(outcome));
    }

    match media_pipeline_strategy(&request.settings) {
        MediaPipelineStrategy::Sequential => {
            run_sequential_media_pipeline(request, cancellation, progress, quality_gate)
        }
        strategy => {
            run_streaming_media_pipeline(request, strategy, cancellation, progress, quality_gate)
        }
    }
}

fn should_translate_embedded_subtitle(
    settings: &ResolvedSettings,
    has_translatable_stream: bool,
) -> bool {
    settings.translation.subtitle_stream_index.is_some() || has_translatable_stream
}

fn media_pipeline_strategy(settings: &ResolvedSettings) -> MediaPipelineStrategy {
    if settings.translation.mode == TranslationMode::Cinema
        || settings.translation.terminology_preflight
        || settings.translation.dry_run
    {
        return MediaPipelineStrategy::Sequential;
    }
    match settings.translation.mode {
        TranslationMode::Economy => MediaPipelineStrategy::EconomyBuffered,
        TranslationMode::Turbo => MediaPipelineStrategy::TurboImmediate,
        TranslationMode::Cinema => MediaPipelineStrategy::Sequential,
    }
}

fn run_sequential_media_pipeline(
    request: PipelineRequest,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
    quality_gate: QualityGate,
) -> AdapterResult<PipelineOutcome> {
    let transcription_out = transcribe_media_cancellable_with_progress(
        TranscriptionRequest {
            media_path: request.input_path,
            output_path: None,
            overwrite: true,
            settings: request.transcription_settings,
        },
        cancellation,
        progress.clone(),
    )?;
    let mut settings = request.settings;
    settings
        .output
        .format
        .get_or_insert_with(|| transcription_out.output_format.extension().to_owned());
    let bytes = serde_json::to_vec(&transcription_out.document.segments).map_err(|source| {
        AdapterError::Serialization {
            context: "fingerprint stable transcription",
            source,
        }
    })?;
    let output_path = request.output_path.clone().map(Ok).unwrap_or_else(|| {
        default_output_path_with_language(
            &transcription_out.output_path,
            settings.output_format(),
            settings.output.bilingual,
            None,
        )
    })?;
    let translation_out = translate_subtitle_cancellable_with_progress_and_identity(
        TranslationRequest {
            input_path: transcription_out.output_path.clone(),
            output_path: Some(output_path.clone()),
            output_language_tag: None,
            overwrite: true,
            runtime_reuse: crate::translation::RuntimeReusePolicy::Configured,
            settings,
        },
        cancellation,
        progress,
        Some(TranslationInputIdentity {
            document: Some(transcription_out.document),
            path: transcription_out.output_path.clone(),
            signature: input_signature_from_bytes(&bytes, None),
            output_path,
            execution_fingerprint: Some(format!(
                "stable-source-v{}",
                subbake_core::transcription::SEGMENTATION_VERSION
            )),
            initial_confirmed_context: Vec::new(),
            ocr_source: None,
        }),
        quality_gate,
    )?;
    Ok(PipelineOutcome::Media {
        translation: translation_out,
        transcription: transcription_out.report,
    })
}

fn run_streaming_media_pipeline(
    request: PipelineRequest,
    strategy: MediaPipelineStrategy,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
    quality_gate: QualityGate,
) -> AdapterResult<PipelineOutcome> {
    let store = StreamingPipelineStore::open(&request, strategy)?;
    let (sender, receiver) = sync_channel(STREAMING_CHANNEL_CAPACITY);
    let transcription_request = TranscriptionRequest {
        media_path: request.input_path.clone(),
        output_path: None,
        overwrite: true,
        settings: request.transcription_settings.clone(),
    };
    let producer_store = store.clone();
    let producer_progress = progress.clone();

    let (transcription, translated) = std::thread::scope(|scope| {
        let producer = scope.spawn(|| {
            let mut observer = ChannelChunkObserver {
                store: producer_store,
                sender,
            };
            transcribe_media_incremental_with_progress(
                transcription_request,
                cancellation,
                producer_progress,
                &mut observer,
            )
        });
        let translated = consume_translation_chunks(
            receiver,
            &request,
            strategy,
            &store,
            cancellation,
            progress,
        );
        let transcription = producer.join().map_err(|_| {
            AdapterError::invalid_input("incremental transcription worker panicked")
        })?;
        Ok::<_, AdapterError>((transcription?, translated?))
    })?;

    finalize_streaming_pipeline_with_quality(
        request,
        store,
        transcription,
        translated,
        quality_gate,
    )
}

#[derive(Debug)]
struct ConsumedTranslations {
    segments: Vec<SubtitleSegment>,
    result: PipelineResult,
    groups: usize,
    chunks: usize,
}

fn consume_translation_chunks(
    receiver: Receiver<StableTranscriptionChunk>,
    request: &PipelineRequest,
    strategy: MediaPipelineStrategy,
    store: &StreamingPipelineStore,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
) -> AdapterResult<ConsumedTranslations> {
    let mut pending = Vec::new();
    let mut translated_segments = Vec::new();
    let mut aggregate = empty_pipeline_result(request, store);
    let mut group_index = 0usize;
    let mut chunk_count = 0usize;
    let mut confirmed_context = Vec::new();

    for chunk in receiver {
        cancellation.check().map_err(AdapterError::from)?;
        chunk_count += 1;
        pending.push(chunk);
        if strategy == MediaPipelineStrategy::TurboImmediate
            || economy_buffer_ready(&pending, request)
        {
            let translated = translate_chunk_group(
                group_index,
                &pending,
                request,
                store,
                cancellation,
                progress.clone(),
                &confirmed_context,
            )?;
            if strategy == MediaPipelineStrategy::TurboImmediate {
                confirmed_context = confirmed_translation_tail(
                    &pending,
                    &translated.segments,
                    request.settings.translation.batch_size,
                );
            }
            translated_segments.extend(translated.segments);
            add_group_result(&mut aggregate, &translated.shard);
            group_index += 1;
            pending.clear();
        }
    }
    if !pending.is_empty() {
        let translated = translate_chunk_group(
            group_index,
            &pending,
            request,
            store,
            cancellation,
            progress,
            &confirmed_context,
        )?;
        translated_segments.extend(translated.segments);
        add_group_result(&mut aggregate, &translated.shard);
        group_index += 1;
    }
    Ok(ConsumedTranslations {
        segments: translated_segments,
        result: aggregate,
        groups: group_index,
        chunks: chunk_count,
    })
}

fn economy_buffer_ready(chunks: &[StableTranscriptionChunk], request: &PipelineRequest) -> bool {
    let segment_count: usize = chunks.iter().map(|chunk| chunk.segments.len()).sum();
    let estimated_tokens: usize = chunks
        .iter()
        .flat_map(|chunk| &chunk.segments)
        .map(|segment| segment.text.chars().count().div_ceil(4) + 8)
        .sum();
    segment_count >= request.settings.translation.batch_size
        || estimated_tokens >= request.settings.translation.batch_token_budget
}

struct TranslatedGroup {
    segments: Vec<SubtitleSegment>,
    shard: TranslationGroupShard,
}

fn translate_chunk_group(
    index: usize,
    chunks: &[StableTranscriptionChunk],
    request: &PipelineRequest,
    store: &StreamingPipelineStore,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
    initial_confirmed_context: &[ConfirmedTranslationContext],
) -> AdapterResult<TranslatedGroup> {
    let first_chunk = chunks
        .first()
        .map(|chunk| chunk.descriptor.index)
        .ok_or_else(|| AdapterError::invalid_input("empty incremental translation group"))?;
    let last_chunk = chunks
        .last()
        .map(|chunk| chunk.descriptor.index)
        .unwrap_or(first_chunk);
    let source_segments: Vec<_> = chunks
        .iter()
        .flat_map(|chunk| chunk.segments.clone())
        .collect();
    let source_first_id = source_segments
        .first()
        .map(|segment| segment.id.clone())
        .unwrap_or_default();
    let source_last_id = source_segments
        .last()
        .map(|segment| segment.id.clone())
        .unwrap_or_default();

    let source_fingerprint = subbake_core::storage::sha1_hex(
        &serde_json::to_vec(&(&source_segments, initial_confirmed_context)).map_err(|source| {
            AdapterError::Serialization {
                context: "fingerprint transcription source",
                source,
            }
        })?,
    );
    // Re-enter the core pipeline even for completed groups: it owns resume
    // fingerprints (including loaded glossary/routes) and deterministic final validation.

    let group_dir = store.group_dir(index);
    fs::create_dir_all(&group_dir).map_err(|source| {
        AdapterError::external_io(
            "create pipeline translation group",
            Some(group_dir.clone()),
            source,
        )
    })?;
    let format = chunks
        .first()
        .map(|chunk| chunk.format.extension())
        .unwrap_or("srt");
    let source_path = group_dir.join(format!("source.{format}"));
    let translated_path = group_dir.join(format!("translated.{format}"));
    let source_document = SubtitleDocument {
        path: source_path.clone(),
        format: format.to_owned(),
        segments: source_segments.clone(),
        header: None,
        passthrough_blocks: Vec::new(),
        metadata: SubtitleDocumentMetadata::None,
    };
    render_and_write_document_atomic(
        &source_document,
        &source_segments,
        &source_path,
        &RenderOptions::new(false, Some(format.to_owned())),
    )?;
    let bytes = fs::read(&source_path).map_err(|source| {
        AdapterError::external_io(
            "read pipeline translation source",
            Some(source_path.clone()),
            source,
        )
    })?;
    let mut settings = request.settings.clone();
    settings.output.bilingual = false;
    settings.output.format = Some(format.to_owned());
    settings.storage.runtime_dir = Some(store.root.join("translation_runtime"));
    let outcome = translate_subtitle_cancellable_with_progress_and_identity(
        TranslationRequest {
            input_path: source_path.clone(),
            output_path: Some(translated_path.clone()),
            output_language_tag: None,
            overwrite: true,
            runtime_reuse: crate::translation::RuntimeReusePolicy::Configured,
            settings,
        },
        cancellation,
        progress,
        Some(TranslationInputIdentity {
            document: Some(source_document),
            path: source_path,
            signature: input_signature_from_bytes(&bytes, None),
            output_path: translated_path.clone(),
            execution_fingerprint: Some(format!(
                "{}:group:{index}:chunks:{first_chunk}-{last_chunk}:source:{source_fingerprint}",
                store.fingerprint
            )),
            initial_confirmed_context: initial_confirmed_context.to_vec(),
            ocr_source: None,
        }),
        subbake_core::QualityGate::Never,
    )?;
    let mut translated = read_document(&translated_path)?;
    restore_stable_source_identity(&source_segments, &mut translated.segments)?;
    validate_group_alignment(&source_segments, &translated.segments)?;
    let shard = TranslationGroupShard {
        source_fingerprint,
        version: STREAMING_PIPELINE_VERSION,
        fingerprint: store.fingerprint.clone(),
        index,
        first_chunk,
        last_chunk,
        source_first_id,
        source_last_id,
        subtitle_entries: source_segments.len(),
        batches_translated: outcome.result.batches_translated,
        review_batches: outcome.result.review_batches,
        usage: outcome.result.usage,
        cache_hits: outcome.result.cache_hits,
        resumed_translation_batches: outcome.result.resumed_translation_batches,
        resumed_review_batches: outcome.result.resumed_review_batches,
        translation_memory_hits: outcome.result.translation_memory_hits,
        deduplicated_segments: outcome.result.deduplicated_segments,
        translated_path,
    };
    store.save_group(&shard)?;
    Ok(TranslatedGroup {
        segments: translated.segments,
        shard,
    })
}

fn confirmed_translation_tail(
    chunks: &[StableTranscriptionChunk],
    translated: &[SubtitleSegment],
    limit: usize,
) -> Vec<ConfirmedTranslationContext> {
    let source = chunks
        .iter()
        .flat_map(|chunk| &chunk.segments)
        .collect::<Vec<_>>();
    let keep = source.len().min(translated.len()).min(limit.max(1));
    source
        .iter()
        .zip(translated)
        .skip(source.len().saturating_sub(keep))
        .map(|(source, translated)| ConfirmedTranslationContext {
            id: source.id.clone(),
            source: source.text.clone(),
            translation: translated.text.clone(),
        })
        .collect()
}

/// SRT display identifiers carry the stable IDs; generic parsers deliberately use
/// positional internal IDs. Verify the serialization boundary before restoring them.
fn restore_stable_source_identity(
    source: &[SubtitleSegment],
    translated: &mut [SubtitleSegment],
) -> AdapterResult<()> {
    if source.len() != translated.len() {
        return Err(AdapterError::invalid_input(
            "translation changed stable cue count",
        ));
    }
    for (source, translated) in source.iter().zip(translated) {
        if translated.identifier.as_deref()
            != source.identifier.as_deref().or(Some(source.id.as_str()))
            || translated.start != source.start
            || translated.end != source.end
        {
            return Err(AdapterError::invalid_input(
                "translation changed stable cue identity or timing",
            ));
        }
        let text = std::mem::take(&mut translated.text);
        *translated = source.clone();
        translated.text = text;
    }
    Ok(())
}

fn validate_group_alignment(
    source: &[SubtitleSegment],
    translated: &[SubtitleSegment],
) -> AdapterResult<()> {
    if source.len() != translated.len()
        || source
            .iter()
            .zip(translated)
            .any(|(source, translated)| source.id != translated.id)
    {
        return Err(AdapterError::invalid_input(
            "persisted incremental translation shard is not aligned with its source chunk",
        ));
    }
    Ok(())
}

fn empty_pipeline_result(
    request: &PipelineRequest,
    store: &StreamingPipelineStore,
) -> PipelineResult {
    PipelineResult {
        output_path: None,
        batches_translated: 0,
        review_batches: 0,
        usage: Usage::default(),
        mode: request.settings.translation.mode,
        deduplicated_segments: 0,
        reviewer_fallback: false,
        dry_run: false,
        planned_batches: Vec::new(),
        cache_hits: 0,
        resumed_translation_batches: 0,
        resumed_review_batches: 0,
        translation_memory_hits: 0,
        state_path: Some(store.root.join("complete.json")),
        glossary_path: request.settings.storage.glossary_path.clone(),
        model_repairs: Vec::new(),
        terminology: subbake_core::TerminologyStats::default(),
        review: subbake_core::ReviewStats::default(),
    }
}

fn add_group_result(result: &mut PipelineResult, group: &TranslationGroupShard) {
    result.batches_translated += group.batches_translated;
    result.review_batches += group.review_batches;
    result.usage.add(group.usage);
    result.cache_hits += group.cache_hits;
    result.resumed_translation_batches += group.resumed_translation_batches;
    result.resumed_review_batches += group.resumed_review_batches;
    result.translation_memory_hits += group.translation_memory_hits;
    result.deduplicated_segments += group.deduplicated_segments;
}

#[cfg(test)]
fn finalize_streaming_pipeline(
    request: PipelineRequest,
    store: StreamingPipelineStore,
    transcription: IncrementalTranscriptionOutcome,
    translated: ConsumedTranslations,
) -> AdapterResult<PipelineOutcome> {
    finalize_streaming_pipeline_with_quality(
        request,
        store,
        transcription,
        translated,
        QualityGate::Never,
    )
}

fn finalize_streaming_pipeline_with_quality(
    request: PipelineRequest,
    store: StreamingPipelineStore,
    transcription: IncrementalTranscriptionOutcome,
    mut translated: ConsumedTranslations,
    quality_gate: QualityGate,
) -> AdapterResult<PipelineOutcome> {
    validate_group_alignment(&transcription.document.segments, &translated.segments)?;
    let source_path = request
        .input_path
        .with_extension(transcription.output_format.extension());
    let output_path = match request.output_path {
        Some(path) => path,
        None => default_output_path_with_language(
            &source_path,
            request.settings.output_format(),
            request.settings.output.bilingual,
            None,
        )?,
    };
    let final_format = request
        .settings
        .output
        .format
        .clone()
        .unwrap_or_else(|| transcription.output_format.extension().to_owned());
    let source_options = RenderOptions::new(
        false,
        Some(transcription.output_format.extension().to_owned()),
    );
    let final_options = RenderOptions::new(request.settings.output.bilingual, Some(final_format))
        .with_bilingual_order(request.settings.output.bilingual_order)
        .with_bilingual_font_scale(request.settings.output.bilingual_font_scale);

    let mut translated_document = transcription.document.clone();
    translated_document
        .segments
        .clone_from(&translated.segments);
    let defaults = QualityPolicy::default();
    let quality = inspect_quality(
        &translated_document,
        QualityPolicy {
            max_characters_per_second: request
                .settings
                .translation
                .max_characters_per_second
                .unwrap_or(defaults.max_characters_per_second),
            max_characters_per_line: request
                .settings
                .translation
                .max_characters_per_line
                .unwrap_or(defaults.max_characters_per_line),
            max_lines: request
                .settings
                .translation
                .max_lines
                .unwrap_or(defaults.max_lines),
        },
    );
    if quality_gate.fails(&quality) {
        return Err(AdapterError::invalid_input(format!(
            "subtitle QA threshold failed before publication with {} error(s) and {} warning(s)",
            quality.errors, quality.warnings
        )));
    }

    // User-visible files are published only after transcription coverage and
    // exact source/translation alignment have both succeeded.
    if source_path != output_path {
        render_and_write_document_atomic(
            &transcription.document,
            &transcription.document.segments,
            &source_path,
            &source_options,
        )?;
    }
    render_and_write_document_atomic(
        &transcription.document,
        &translated.segments,
        &output_path,
        &final_options,
    )?;
    store.mark_complete(&output_path, translated.chunks, translated.groups)?;
    translated.result.output_path = Some(output_path.clone());
    Ok(PipelineOutcome::Media {
        translation: TranslationOutcome {
            result: translated.result,
            output_path: Some(output_path),
            subtitle_entries: transcription.document.segments.len(),
            container_change: None,
            runtime_dir: None,
            quality: Some(quality),
            source_ocr: None,
        },
        transcription: transcription.report,
    })
}

fn streaming_fingerprint(
    request: &PipelineRequest,
    strategy: MediaPipelineStrategy,
) -> AdapterResult<String> {
    let metadata = fs::metadata(&request.input_path).map_err(|source| {
        AdapterError::external_io(
            "inspect pipeline media input",
            Some(request.input_path.clone()),
            source,
        )
    })?;
    let mtime_ns = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos().to_string())
        .unwrap_or_default();
    let stable_input = stable_runtime_input_path(&request.input_path)?;
    let mut options = request
        .settings
        .to_pipeline_options(stable_input.clone(), request.output_path.clone());
    options.identity.execution_fingerprint = Some(format!(
        "streaming-v{STREAMING_PIPELINE_VERSION}:{}",
        strategy.as_str()
    ));
    let source_identity = JsonValue::Object(vec![
        (
            "path".to_owned(),
            JsonValue::String(stable_input.to_string_lossy().into_owned()),
        ),
        (
            "size".to_owned(),
            JsonValue::Number(metadata.len().to_string()),
        ),
        ("mtime_ns".to_owned(), JsonValue::String(mtime_ns.clone())),
    ]);
    let media_signature = InputSignature {
        sha1: stable_hash(&source_identity),
        size: metadata.len(),
        mtime_ns: metadata
            .modified()
            .ok()
            .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos()),
    };
    let translation_fingerprint = build_translation_fingerprint(&options, &media_signature);
    Ok(stable_hash(&JsonValue::Object(vec![
        (
            "version".to_owned(),
            JsonValue::Number(STREAMING_PIPELINE_VERSION.to_string()),
        ),
        (
            "strategy".to_owned(),
            JsonValue::String(strategy.as_str().to_owned()),
        ),
        ("source".to_owned(), source_identity),
        (
            "chunk_contract".to_owned(),
            JsonValue::String(
                "core=600000;overlap=30000;ownership=midpoint;max_context=0".to_owned(),
            ),
        ),
        (
            "translation".to_owned(),
            JsonValue::String(translation_fingerprint),
        ),
        (
            "transcription".to_owned(),
            JsonValue::Object(vec![
                (
                    "language".to_owned(),
                    request
                        .transcription_settings
                        .language
                        .clone()
                        .unwrap_or_default()
                        .into(),
                ),
                (
                    "model".to_owned(),
                    request
                        .transcription_settings
                        .model
                        .clone()
                        .unwrap_or_default()
                        .into(),
                ),
                (
                    "whisper_binary".to_owned(),
                    request
                        .transcription_settings
                        .whisper_binary_path
                        .as_ref()
                        .map(|path| path.to_string_lossy().into_owned())
                        .unwrap_or_default()
                        .into(),
                ),
                (
                    "whisper_models_dir".to_owned(),
                    request
                        .transcription_settings
                        .whisper_models_dir
                        .as_ref()
                        .map(|path| path.to_string_lossy().into_owned())
                        .unwrap_or_default()
                        .into(),
                ),
                (
                    "sidecar".to_owned(),
                    request
                        .transcription_settings
                        .sidecar_path
                        .as_ref()
                        .map(|path| path.to_string_lossy().into_owned())
                        .unwrap_or_default()
                        .into(),
                ),
                (
                    "segmentation".to_owned(),
                    request
                        .transcription_settings
                        .effective_segmentation()
                        .as_str()
                        .into(),
                ),
                (
                    "segmentation_version".to_owned(),
                    subbake_core::transcription::SEGMENTATION_VERSION
                        .to_string()
                        .into(),
                ),
                (
                    "output_format".to_owned(),
                    request
                        .transcription_settings
                        .output_format
                        .extension()
                        .into(),
                ),
                (
                    "filter_silence_markers".to_owned(),
                    request
                        .transcription_settings
                        .effective_filter_silence_markers()
                        .into(),
                ),
            ]),
        ),
    ])))
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn subtitle_inputs_use_translation_service() {
        let root = temp_root("subtitle");
        fs::create_dir_all(&root).expect("create temp root");
        let input_path = root.join("clip.txt");
        fs::write(&input_path, "hello\n").expect("write input");
        let mut settings = ResolvedSettings::default();
        settings.translation.target_language = "en".to_owned();
        settings.translation.review_policy = subbake_core::ReviewPolicy::Off;

        let outcome = run_pipeline(PipelineRequest {
            input_path,
            output_path: None,
            settings,
            transcription_settings: TranscriptionSettings::default(),
        })
        .expect("run pipeline");
        let output_path = match outcome {
            PipelineOutcome::Subtitle(outcome) => outcome.output_path.expect("output path"),
            PipelineOutcome::Media { .. } => panic!("subtitle input should not transcribe"),
        };
        let output = fs::read_to_string(&output_path).expect("read output");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(output, "[MOCK-EN] hello\n");
    }

    #[test]
    fn modes_select_expected_media_strategy() {
        let mut settings = ResolvedSettings::default();
        settings.translation.mode = TranslationMode::Turbo;
        assert_eq!(
            media_pipeline_strategy(&settings),
            MediaPipelineStrategy::TurboImmediate
        );
        settings.translation.mode = TranslationMode::Economy;
        assert_eq!(
            media_pipeline_strategy(&settings),
            MediaPipelineStrategy::EconomyBuffered
        );
        settings.translation.mode = TranslationMode::Cinema;
        assert_eq!(
            media_pipeline_strategy(&settings),
            MediaPipelineStrategy::Sequential
        );
    }

    #[test]
    fn default_settings_select_turbo_immediate_media_strategy() {
        assert_eq!(
            media_pipeline_strategy(&ResolvedSettings::default()),
            MediaPipelineStrategy::TurboImmediate
        );
    }

    #[test]
    fn container_pipeline_prefers_text_subtitles_and_falls_back_without_them() {
        let mut settings = ResolvedSettings::default();
        assert!(should_translate_embedded_subtitle(&settings, true));
        assert!(!should_translate_embedded_subtitle(&settings, false));

        settings.translation.subtitle_stream_index = Some(3);
        assert!(should_translate_embedded_subtitle(&settings, false));
    }

    #[test]
    fn terminology_preflight_forces_sequential_pipeline() {
        let mut settings = ResolvedSettings::default();
        settings.translation.mode = TranslationMode::Turbo;
        settings.translation.terminology_preflight = true;
        assert_eq!(
            media_pipeline_strategy(&settings),
            MediaPipelineStrategy::Sequential
        );
    }

    #[test]
    fn economy_waits_for_configured_batch_threshold() {
        let mut settings = ResolvedSettings::default();
        settings.translation.batch_size = 3;
        settings.translation.batch_token_budget = 10_000;
        let request = PipelineRequest {
            input_path: PathBuf::from("movie.mkv"),
            output_path: None,
            settings,
            transcription_settings: TranscriptionSettings::default(),
        };
        let chunks = vec![test_chunk(0, 2), test_chunk(1, 1)];
        assert!(!economy_buffer_ready(&chunks[..1], &request));
        assert!(economy_buffer_ready(&chunks, &request));
    }

    #[test]
    fn confirmed_translation_tail_preserves_source_and_translation_order() {
        let chunks = vec![test_chunk(0, 2), test_chunk(1, 2)];
        let translated = chunks
            .iter()
            .flat_map(|chunk| &chunk.segments)
            .enumerate()
            .map(|(index, source)| {
                let mut translated = source.clone();
                translated.text = format!("translated {index}");
                translated
            })
            .collect::<Vec<_>>();

        let tail = confirmed_translation_tail(&chunks, &translated, 3);

        assert_eq!(tail.len(), 3);
        assert_eq!(tail[0].id, chunks[0].segments[1].id);
        assert_eq!(tail[0].translation, "translated 1");
        assert_eq!(tail[2].id, chunks[1].segments[1].id);
        assert_eq!(tail[2].translation, "translated 3");
    }

    #[test]
    fn execution_strategy_changes_resume_fingerprint() {
        let root = temp_root("fingerprint");
        fs::create_dir_all(&root).expect("create root");
        let input = root.join("movie.mkv");
        fs::write(&input, b"media").expect("write media");
        let mut settings = ResolvedSettings::default();
        settings.translation.terminology_preflight = false;
        let request = PipelineRequest {
            input_path: input,
            output_path: None,
            settings,
            transcription_settings: TranscriptionSettings::default(),
        };
        let turbo = streaming_fingerprint(&request, MediaPipelineStrategy::TurboImmediate)
            .expect("turbo fingerprint");
        let economy = streaming_fingerprint(&request, MediaPipelineStrategy::EconomyBuffered)
            .expect("economy fingerprint");
        let _ = fs::remove_dir_all(root);
        assert_ne!(turbo, economy);
    }

    #[test]
    fn pipeline_keeps_an_audit_copy_of_stable_source() {
        let root = temp_root("transcription-resume");
        fs::create_dir_all(&root).expect("create root");
        let input = root.join("movie.mkv");
        fs::write(&input, b"media").expect("write media");
        let mut settings = ResolvedSettings::default();
        settings.translation.terminology_preflight = false;
        settings.storage.runtime_dir = Some(root.join("runtime"));
        let request = PipelineRequest {
            input_path: input,
            output_path: None,
            settings,
            transcription_settings: TranscriptionSettings::default(),
        };
        let store = StreamingPipelineStore::open(&request, MediaPipelineStrategy::TurboImmediate)
            .expect("open store");
        let chunk = test_chunk(0, 2);
        store.save_chunk(&chunk).expect("save chunk");
        let resumed: TranscriptionChunkShard = serde_json::from_slice(
            &fs::read(store.chunk_path(chunk.descriptor.index)).expect("audit file"),
        )
        .expect("audit shape");
        let _ = fs::remove_dir_all(root);
        assert_eq!(resumed.segments, chunk.segments);
    }

    #[test]
    fn incomplete_alignment_never_publishes_final_subtitle() {
        let root = temp_root("no-partial-final");
        fs::create_dir_all(&root).expect("create root");
        let input = root.join("movie.mkv");
        fs::write(&input, b"media").expect("write media");
        let mut settings = ResolvedSettings::default();
        settings.translation.terminology_preflight = false;
        settings.storage.runtime_dir = Some(root.join("runtime"));
        let output = root.join("final.srt");
        let request = PipelineRequest {
            input_path: input,
            output_path: Some(output.clone()),
            settings,
            transcription_settings: TranscriptionSettings::default(),
        };
        let store = StreamingPipelineStore::open(&request, MediaPipelineStrategy::TurboImmediate)
            .expect("open store");
        let chunk = test_chunk(0, 1);
        let transcription = IncrementalTranscriptionOutcome {
            document: SubtitleDocument {
                path: request.input_path.clone(),
                format: "srt".to_owned(),
                segments: chunk.segments,
                header: None,
                passthrough_blocks: Vec::new(),
                metadata: SubtitleDocumentMetadata::None,
            },
            language: "en".to_owned(),
            provider: "fake".to_owned(),
            model: "fake".to_owned(),
            model_auto_selected: false,
            output_format: subbake_core::TranscriptionFormat::Srt,
            cleanup: Default::default(),
            report: Default::default(),
        };
        let translated = ConsumedTranslations {
            segments: Vec::new(),
            result: empty_pipeline_result(&request, &store),
            groups: 0,
            chunks: 1,
        };
        let error = finalize_streaming_pipeline(request, store, transcription, translated)
            .expect_err("misaligned finalization must fail");
        assert!(error.to_string().contains("not aligned"));
        assert!(!output.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn group_resume_uses_current_source_and_does_not_trust_published_translation() {
        let temporary = tempfile::tempdir().expect("fixture");
        let input = temporary.path().join("media.wav");
        fs::write(&input, b"fixture").expect("media");
        let mut settings = ResolvedSettings::default();
        settings.storage.runtime_dir = Some(temporary.path().join("runtime"));
        settings.translation.target_language = "en".to_owned();
        settings.translation.review_policy = subbake_core::ReviewPolicy::Off;
        let request = PipelineRequest {
            input_path: input,
            output_path: None,
            settings,
            transcription_settings: Default::default(),
        };
        let store = StreamingPipelineStore::open(&request, MediaPipelineStrategy::TurboImmediate)
            .expect("store");
        let mut chunks = vec![test_chunk(0, 1)];
        chunks[0].segments[0].identifier = Some(chunks[0].segments[0].id.clone());
        let run = |chunks: &[StableTranscriptionChunk]| {
            translate_chunk_group(
                0,
                chunks,
                &request,
                &store,
                &CancellationGuard::never(),
                std::sync::Arc::new(NoopProgress),
                &[],
            )
            .expect("translation")
        };
        let first = run(&chunks);
        assert_eq!(first.segments[0].text, "[MOCK-EN] hello");
        // A completed file is an output, not a trusted substitute for core validation.
        fs::write(&first.shard.translated_path, b"corrupted published file")
            .expect("corrupt fixture");
        let resumed = run(&chunks);
        assert_eq!(resumed.segments, first.segments);
        chunks[0].segments[0].text = "goodbye".to_owned();
        chunks[0].segments[0].start = Some("00:00:10,000".to_owned());
        chunks[0].segments[0].end = Some("00:00:11,000".to_owned());
        let changed = run(&chunks);
        assert_ne!(
            changed.shard.source_fingerprint,
            first.shard.source_fingerprint
        );
        assert_eq!(changed.segments[0].text, "[MOCK-EN] goodbye");
        assert_eq!(changed.segments[0].start, chunks[0].segments[0].start);
    }

    fn test_chunk(index: usize, segments: usize) -> StableTranscriptionChunk {
        StableTranscriptionChunk {
            descriptor: crate::transcription::TranscriptionChunkDescriptor {
                index,
                input_start_ms: index as u64 * 600_000,
                input_end_ms: (index as u64 + 1) * 600_000,
                core_start_ms: index as u64 * 600_000,
                core_end_ms: (index as u64 + 1) * 600_000,
            },
            format: subbake_core::TranscriptionFormat::Srt,
            segments: (0..segments)
                .map(|local| SubtitleSegment {
                    id: format!("c{index:04}-{local:06}"),
                    text: "hello".to_owned(),
                    start: Some("00:00:00,000".to_owned()),
                    end: Some("00:00:01,000".to_owned()),
                    identifier: None,
                    settings: None,
                    semantic: Default::default(),
                })
                .collect(),
            resumed: false,
        }
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-pipeline-service-{label}-{nanos}"))
    }
}
