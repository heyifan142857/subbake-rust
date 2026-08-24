// Media transcription through the local whisper.cpp sidecar.
//
// Orchestration: ffmpeg audio extraction (video-only) → backend transcribe → render.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{fs, io};
use subbake_core::entities::SubtitleSegment;
use subbake_core::formats::RenderOptions;
use subbake_core::languages::normalize_language;
use subbake_core::{
    CancellationGuard, NoopProgress, ProgressEvent, ProgressUnit, SharedProgress, SubtitleDocument,
    TaskKind, TaskState,
};
pub use subbake_core::{TranscriberBackend, TranscriptionFormat};

use crate::error::{AdapterError, AdapterResult};
use crate::fs::{read_document, render_and_write_document};
use crate::process::ProcessSupervisor;
use crate::settings::{
    DEFAULT_VAD_MIN_SILENCE_DURATION_MS, DEFAULT_VAD_MIN_SPEECH_DURATION_MS,
    DEFAULT_VAD_SPEECH_PAD_MS, DEFAULT_VAD_THRESHOLD, ResolvedSettings, StorageSettings,
};
use crate::whisper::{
    DEFAULT_WHISPER_VAD_MODEL, default_whisper_binary_path_for, default_whisper_models_dir_for,
    vad_model_path, verify_whisper_cli,
};

mod model_resolution;

use model_resolution::{ResolvedWhisperModel, locate_whisper_binary, resolve_whisper_model};

const LONG_AUDIO_THRESHOLD_MS: u64 = 12 * 60 * 1_000;
const TRANSCRIPTION_CHUNK_MS: u64 = 10 * 60 * 1_000;
const TRANSCRIPTION_OVERLAP_MS: u64 = 30 * 1_000;
const MAX_UNCOVERED_TRAILING_MS: u64 = 30 * 60 * 1_000;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionRequest {
    pub media_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub overwrite: bool,
    pub settings: TranscriptionSettings,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionSettings {
    pub language: Option<String>,
    pub model: Option<String>,
    pub output_format: TranscriptionFormat,
    pub sidecar_path: Option<PathBuf>,
    pub whisper_binary_path: Option<PathBuf>,
    pub whisper_models_dir: Option<PathBuf>,
    pub runtime_dir: Option<PathBuf>,
    pub multiple_model_policy: MultipleModelPolicy,
    pub filter_hallucinations: bool,
    pub vad_enabled: Option<bool>,
    pub vad_model: Option<String>,
    pub vad_threshold: Option<f32>,
    pub vad_min_speech_duration_ms: Option<u64>,
    pub vad_min_silence_duration_ms: Option<u64>,
    pub vad_speech_pad_ms: Option<u64>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MultipleModelPolicy {
    #[default]
    RequireExplicit,
    PreferRanked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionOutcome {
    pub output_path: PathBuf,
    pub language: String,
    pub provider: String,
    pub model: String,
    pub model_auto_selected: bool,
    pub output_format: TranscriptionFormat,
    pub subtitle_entries: usize,
    pub cleanup: TranscriptionCleanupStats,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptionCleanupStats {
    pub removed_empty_or_silence: usize,
    pub removed_repeated: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TranscriptionChunkDescriptor {
    pub index: usize,
    pub input_start_ms: u64,
    pub input_end_ms: u64,
    pub core_start_ms: u64,
    pub core_end_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StableTranscriptionChunk {
    pub descriptor: TranscriptionChunkDescriptor,
    pub format: TranscriptionFormat,
    pub segments: Vec<SubtitleSegment>,
    pub resumed: bool,
}

pub(crate) trait TranscriptionChunkObserver: Send {
    fn load(
        &mut self,
        descriptor: TranscriptionChunkDescriptor,
        format: TranscriptionFormat,
    ) -> AdapterResult<Option<Vec<SubtitleSegment>>>;

    fn stable(&mut self, chunk: StableTranscriptionChunk) -> AdapterResult<()>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IncrementalTranscriptionOutcome {
    pub document: SubtitleDocument,
    pub language: String,
    pub provider: String,
    pub model: String,
    pub model_auto_selected: bool,
    pub output_format: TranscriptionFormat,
    pub cleanup: TranscriptionCleanupStats,
}

impl Default for TranscriptionSettings {
    fn default() -> Self {
        Self {
            language: None,
            model: None,
            output_format: TranscriptionFormat::Srt,
            sidecar_path: None,
            whisper_binary_path: None,
            whisper_models_dir: None,
            runtime_dir: None,
            multiple_model_policy: MultipleModelPolicy::RequireExplicit,
            filter_hallucinations: true,
            vad_enabled: None,
            vad_model: None,
            vad_threshold: None,
            vad_min_speech_duration_ms: None,
            vad_min_silence_duration_ms: None,
            vad_speech_pad_ms: None,
        }
    }
}

impl TranscriptionSettings {
    pub(crate) fn effective_vad_enabled(&self) -> bool {
        self.vad_enabled.unwrap_or(true)
    }

    pub(crate) fn effective_vad_model(&self) -> &str {
        self.vad_model
            .as_deref()
            .unwrap_or(DEFAULT_WHISPER_VAD_MODEL)
    }

    pub(crate) fn effective_vad_threshold(&self) -> f32 {
        self.vad_threshold.unwrap_or(DEFAULT_VAD_THRESHOLD)
    }

    pub(crate) fn effective_vad_min_speech_duration_ms(&self) -> u64 {
        self.vad_min_speech_duration_ms
            .unwrap_or(DEFAULT_VAD_MIN_SPEECH_DURATION_MS)
    }

    pub(crate) fn effective_vad_min_silence_duration_ms(&self) -> u64 {
        self.vad_min_silence_duration_ms
            .unwrap_or(DEFAULT_VAD_MIN_SILENCE_DURATION_MS)
    }

    pub(crate) fn effective_vad_speech_pad_ms(&self) -> u64 {
        self.vad_speech_pad_ms.unwrap_or(DEFAULT_VAD_SPEECH_PAD_MS)
    }
}

// ---------------------------------------------------------------------------
// whisper.cpp backend (local subprocess)
// ---------------------------------------------------------------------------

pub struct WhisperCppTranscriber {
    binary: PathBuf,
    model_path: PathBuf,
    extra_args: Vec<String>,
    progress: SharedProgress,
    threads: usize,
    progress_start: u64,
    progress_end: u64,
}

impl WhisperCppTranscriber {
    pub fn new(binary: PathBuf, model_path: PathBuf, extra_args: Vec<String>) -> Self {
        Self {
            binary,
            model_path,
            extra_args,
            progress: std::sync::Arc::new(NoopProgress),
            threads: default_whisper_threads(),
            progress_start: 0,
            progress_end: 100,
        }
    }

    fn with_progress(mut self, progress: SharedProgress) -> Self {
        self.progress = progress;
        self
    }

    fn with_progress_window(mut self, start: u64, end: u64) -> Self {
        self.progress_start = start.min(100);
        self.progress_end = end.clamp(self.progress_start, 100);
        self
    }

    fn mapped_progress(&self, current: u64) -> u64 {
        self.progress_start + current.min(100) * (self.progress_end - self.progress_start) / 100
    }
}

impl TranscriberBackend for WhisperCppTranscriber {
    type Error = AdapterError;

    fn transcribe(
        &self,
        audio_path: &Path,
        language: Option<&str>,
        output_format: TranscriptionFormat,
    ) -> AdapterResult<SubtitleDocument> {
        self.transcribe_cancellable(
            audio_path,
            language,
            output_format,
            &CancellationGuard::never(),
        )
    }

    fn transcribe_cancellable(
        &self,
        audio_path: &Path,
        language: Option<&str>,
        output_format: TranscriptionFormat,
        cancellation: &CancellationGuard,
    ) -> AdapterResult<SubtitleDocument> {
        check_cancelled(cancellation)?;
        let output_dir = audio_path.parent().unwrap_or_else(|| Path::new("."));
        let base_name = audio_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("audio");
        let output_base = output_dir.join(base_name);
        let threads = self.threads.to_string();

        let mut cmd = Command::new(&self.binary);
        cmd.args([
            "-m",
            &self.model_path.to_string_lossy(),
            "-f",
            &audio_path.to_string_lossy(),
            "--output-file",
            &output_base.to_string_lossy(),
            "--threads",
            &threads,
        ]);
        match output_format {
            TranscriptionFormat::Srt | TranscriptionFormat::Txt => {
                cmd.arg("--output-srt");
            }
            TranscriptionFormat::Vtt => {
                cmd.arg("--output-vtt");
            }
        }
        if let Some(lang) = language {
            cmd.args(["-l", whisper_language_code(lang)]);
        }
        for arg in &self.extra_args {
            cmd.arg(arg);
        }
        cmd.args(["--print-progress", "--no-prints"]);

        let mut last_progress = 0_u64;
        let out = ProcessSupervisor::run_with_stderr_lines(
            &mut cmd,
            cancellation,
            "whisper.cpp execution",
            |line| {
                let Some(current) = parse_whisper_progress(line) else {
                    return;
                };
                if current <= last_progress {
                    return;
                }
                last_progress = current;
                self.progress.emit(ProgressEvent::running(
                    TaskKind::Transcription,
                    "TRANSCRIBE",
                    self.mapped_progress(current),
                    Some(100),
                    ProgressUnit::Percent,
                ));
            },
        )?;
        if !out.status.success() {
            return Err(AdapterError::ChildProcess {
                program: "whisper.cpp",
                status: out.status.code(),
                message: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        if last_progress < 100 {
            self.progress.emit(ProgressEvent::running(
                TaskKind::Transcription,
                "TRANSCRIBE",
                self.progress_end,
                Some(100),
                ProgressUnit::Percent,
            ));
        }

        let suffix = match output_format {
            TranscriptionFormat::Vtt => "vtt",
            _ => "srt",
        };
        let generated = output_base.with_extension(suffix);
        if !generated.is_file() {
            return Err(AdapterError::ChildProcess {
                program: "whisper.cpp",
                status: out.status.code(),
                message: child_diagnostics(&out, "whisper.cpp did not create its output file"),
            });
        }
        let mut doc = read_document(&generated)?;
        let _ = std::fs::remove_file(&generated);

        if matches!(output_format, TranscriptionFormat::Txt) {
            doc.segments = doc
                .segments
                .iter()
                .map(|s| SubtitleSegment {
                    start: None,
                    end: None,
                    identifier: None,
                    settings: None,
                    semantic: Default::default(),
                    ..s.clone()
                })
                .collect();
            doc.format = "txt".to_owned();
        }
        Ok(doc)
    }
}

fn whisper_language_code(language: &str) -> &str {
    language.split('-').next().unwrap_or(language)
}

fn default_whisper_threads() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| recommended_whisper_threads(parallelism.get()))
        .unwrap_or(4)
}

fn recommended_whisper_threads(parallelism: usize) -> usize {
    (parallelism / 2).clamp(1, 16)
}

fn parse_whisper_progress(line: &str) -> Option<u64> {
    line.split_once("progress =")
        .and_then(|(_, value)| value.trim().strip_suffix('%'))
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(|value| value.min(100))
}

fn whisper_vad_args(settings: &TranscriptionSettings) -> AdapterResult<Vec<String>> {
    if !settings.effective_vad_enabled() {
        return Ok(Vec::new());
    }
    let model = settings.effective_vad_model().trim();
    if model.is_empty() {
        return Err(AdapterError::invalid_input(
            "VAD model must not be empty when VAD is enabled",
        ));
    }
    let configured_path = PathBuf::from(model);
    let model_path = if configured_path.is_file()
        || configured_path.is_absolute()
        || configured_path.components().count() > 1
    {
        configured_path
    } else {
        let models_dir = settings
            .whisper_models_dir
            .clone()
            .unwrap_or_else(|| default_whisper_models_dir_for(None));
        vad_model_path(&models_dir, model)
    };
    if !model_path.is_file() {
        return Err(AdapterError::invalid_input(format!(
            "whisper.cpp VAD model `{model}` was not found at `{}`; run `sbake whisper vad-model {model}` or disable VAD explicitly with `--no-vad`",
            model_path.display()
        )));
    }

    let threshold = settings.effective_vad_threshold();
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(AdapterError::invalid_input(
            "VAD threshold must be from 0 through 1",
        ));
    }
    let min_speech = settings.effective_vad_min_speech_duration_ms();
    let min_silence = settings.effective_vad_min_silence_duration_ms();
    let speech_pad = settings.effective_vad_speech_pad_ms();
    if min_speech == 0 || min_silence == 0 {
        return Err(AdapterError::invalid_input(
            "VAD minimum speech and silence durations must be greater than zero",
        ));
    }
    Ok(vec![
        "--vad".to_owned(),
        "--vad-model".to_owned(),
        model_path.to_string_lossy().into_owned(),
        "--vad-threshold".to_owned(),
        threshold.to_string(),
        "--vad-min-speech-duration-ms".to_owned(),
        min_speech.to_string(),
        "--vad-min-silence-duration-ms".to_owned(),
        min_silence.to_string(),
        "--vad-speech-pad-ms".to_owned(),
        speech_pad.to_string(),
    ])
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

pub fn transcribe_media(request: TranscriptionRequest) -> AdapterResult<TranscriptionOutcome> {
    transcribe_media_cancellable(request, &CancellationGuard::never())
}

pub fn transcribe_media_cancellable(
    request: TranscriptionRequest,
    cancellation: &CancellationGuard,
) -> AdapterResult<TranscriptionOutcome> {
    transcribe_media_cancellable_with_progress(
        request,
        cancellation,
        std::sync::Arc::new(NoopProgress),
    )
}

pub fn transcribe_media_cancellable_with_progress(
    mut request: TranscriptionRequest,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
) -> AdapterResult<TranscriptionOutcome> {
    check_cancelled(cancellation)?;
    let language = match request.settings.language.as_deref() {
        Some(value) => normalize_language(value, true)
            .map_err(|error| AdapterError::invalid_input(error.to_string()))?,
        None => "Auto".to_owned(),
    };
    request.settings.language = (language != "Auto").then(|| language.clone());
    if request
        .settings
        .model
        .as_deref()
        .is_some_and(|model| model.trim().is_empty())
    {
        return Err(AdapterError::invalid_input(
            "transcription model must not be empty",
        ));
    }
    let output_path = request.output_path.unwrap_or_else(|| {
        default_output_path(&request.media_path, request.settings.output_format)
    });
    if output_path.exists() && !request.overwrite {
        return Err(AdapterError::invalid_input(format!(
            "output already exists and overwrite is false: {}",
            output_path.display()
        )));
    }

    if let Some(ref sidecar_path) = request.settings.sidecar_path {
        check_cancelled(cancellation)?;
        render_sidecar(sidecar_path, &output_path, request.settings.output_format)?;
        let mut done = ProgressEvent::running(
            TaskKind::Transcription,
            "COMPLETE",
            1,
            Some(1),
            ProgressUnit::Steps,
        );
        done.state = TaskState::Completed;
        progress.emit(done);
        let document = read_document(sidecar_path)?;
        return Ok(TranscriptionOutcome {
            output_path,
            language,
            provider: "sidecar".to_owned(),
            model: "none".to_owned(),
            model_auto_selected: false,
            output_format: request.settings.output_format,
            subtitle_entries: document.segments.len(),
            cleanup: TranscriptionCleanupStats::default(),
        });
    }

    let prepared_audio = prepare_audio(
        &request.media_path,
        &request.settings,
        cancellation,
        &progress,
    )?;
    progress.emit(ProgressEvent::running(
        TaskKind::Transcription,
        "TRANSCRIBE",
        0,
        Some(100),
        ProgressUnit::Percent,
    ));
    let fmt = request.settings.output_format;

    let ResolvedWhisperModel {
        name: effective_model,
        path: model_path,
        auto_selected: model_auto_selected,
    } = resolve_whisper_model(&request.settings)?;
    let binary = locate_whisper_binary(&request.settings)?;
    verify_whisper_cli(&binary, cancellation)?;
    let mut doc = transcribe_prepared_audio(
        &binary,
        &model_path,
        &prepared_audio,
        request.settings.language.as_deref(),
        fmt,
        &request.settings,
        cancellation,
        &progress,
    )?;
    let raw_last_timestamp = last_timed_end_ms(&doc);
    let cleanup = if request.settings.filter_hallucinations {
        clean_transcription_document(&mut doc)
    } else {
        TranscriptionCleanupStats::default()
    };
    validate_transcription_coverage(
        &doc,
        raw_last_timestamp,
        prepared_audio.duration_ms(),
        cleanup,
        0,
    )?;
    shift_document_timestamps(&mut doc, prepared_audio.timeline_offset_ms());

    check_cancelled(cancellation)?;
    let opts = RenderOptions::new(false, Some(fmt.extension().to_owned()));
    render_and_write_document(&doc, &doc.segments, &output_path, &opts)?;
    let mut done = ProgressEvent::running(
        TaskKind::Transcription,
        "COMPLETE",
        1,
        Some(1),
        ProgressUnit::Steps,
    );
    done.state = TaskState::Completed;
    progress.emit(done);
    Ok(TranscriptionOutcome {
        output_path,
        language,
        provider: "whisper_cpp".to_owned(),
        model: effective_model,
        model_auto_selected,
        output_format: fmt,
        subtitle_entries: doc.segments.len(),
        cleanup,
    })
}

/// Transcribe media without publishing a user-visible subtitle file. Stable
/// core chunks are delivered as soon as their overlap ownership is final.
/// The observer may return persisted chunks to resume whisper independently
/// from downstream translation.
pub(crate) fn transcribe_media_incremental_with_progress(
    mut request: TranscriptionRequest,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
    observer: &mut dyn TranscriptionChunkObserver,
) -> AdapterResult<IncrementalTranscriptionOutcome> {
    check_cancelled(cancellation)?;
    let language = match request.settings.language.as_deref() {
        Some(value) => normalize_language(value, true)
            .map_err(|error| AdapterError::invalid_input(error.to_string()))?,
        None => "Auto".to_owned(),
    };
    request.settings.language = (language != "Auto").then(|| language.clone());
    if request
        .settings
        .model
        .as_deref()
        .is_some_and(|model| model.trim().is_empty())
    {
        return Err(AdapterError::invalid_input(
            "transcription model must not be empty",
        ));
    }

    if let Some(ref sidecar_path) = request.settings.sidecar_path {
        let document = read_document(sidecar_path)?;
        let descriptor = TranscriptionChunkDescriptor {
            index: 0,
            input_start_ms: 0,
            input_end_ms: last_timed_end_ms(&document).unwrap_or(0),
            core_start_ms: 0,
            core_end_ms: last_timed_end_ms(&document).unwrap_or(0),
        };
        observer.stable(StableTranscriptionChunk {
            descriptor,
            format: request.settings.output_format,
            segments: document.segments.clone(),
            resumed: false,
        })?;
        return Ok(IncrementalTranscriptionOutcome {
            document,
            language,
            provider: "sidecar".to_owned(),
            model: "none".to_owned(),
            model_auto_selected: false,
            output_format: request.settings.output_format,
            cleanup: TranscriptionCleanupStats::default(),
        });
    }

    let prepared_audio = prepare_audio(
        &request.media_path,
        &request.settings,
        cancellation,
        &progress,
    )?;
    progress.emit(ProgressEvent::running(
        TaskKind::Transcription,
        "TRANSCRIBE",
        0,
        Some(100),
        ProgressUnit::Percent,
    ));
    let ResolvedWhisperModel {
        name: effective_model,
        path: model_path,
        auto_selected: model_auto_selected,
    } = resolve_whisper_model(&request.settings)?;
    let binary = locate_whisper_binary(&request.settings)?;
    verify_whisper_cli(&binary, cancellation)?;
    let (document, cleanup) = transcribe_prepared_audio_incremental(
        &binary,
        &model_path,
        &prepared_audio,
        request.settings.language.as_deref(),
        request.settings.output_format,
        &request.settings,
        cancellation,
        &progress,
        observer,
        prepared_audio.timeline_offset_ms(),
    )?;

    Ok(IncrementalTranscriptionOutcome {
        document,
        language,
        provider: "whisper_cpp".to_owned(),
        model: effective_model,
        model_auto_selected,
        output_format: request.settings.output_format,
        cleanup,
    })
}

fn clean_transcription_document(document: &mut SubtitleDocument) -> TranscriptionCleanupStats {
    let mut stats = TranscriptionCleanupStats::default();
    let mut previous = String::new();
    let mut repeated = 0usize;
    document.segments.retain(|segment| {
        let normalized = segment
            .text
            .split_whitespace()
            .collect::<String>()
            .to_ascii_lowercase();
        if normalized.is_empty() || matches!(normalized.as_str(), "[blank_audio]" | "[silence]") {
            stats.removed_empty_or_silence += 1;
            return false;
        }
        if normalized == previous {
            repeated += 1;
            if repeated >= 2 {
                stats.removed_repeated += 1;
                return false;
            }
        } else {
            previous = normalized;
            repeated = 0;
        }
        true
    });
    stats
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TranscriptionChunk {
    input_start_ms: u64,
    input_end_ms: u64,
    core_start_ms: u64,
    core_end_ms: u64,
}

fn transcription_chunks(duration_ms: u64) -> Vec<TranscriptionChunk> {
    if duration_ms <= LONG_AUDIO_THRESHOLD_MS {
        return vec![TranscriptionChunk {
            input_start_ms: 0,
            input_end_ms: duration_ms,
            core_start_ms: 0,
            core_end_ms: duration_ms,
        }];
    }
    let mut chunks = Vec::new();
    let mut core_start_ms = 0;
    while core_start_ms < duration_ms {
        let core_end_ms = (core_start_ms + TRANSCRIPTION_CHUNK_MS).min(duration_ms);
        chunks.push(TranscriptionChunk {
            input_start_ms: core_start_ms.saturating_sub(TRANSCRIPTION_OVERLAP_MS),
            input_end_ms: (core_end_ms + TRANSCRIPTION_OVERLAP_MS).min(duration_ms),
            core_start_ms,
            core_end_ms,
        });
        core_start_ms = core_end_ms;
    }
    chunks
}

#[allow(clippy::too_many_arguments)]
fn transcribe_prepared_audio(
    binary: &Path,
    model_path: &Path,
    prepared_audio: &PreparedAudio,
    language: Option<&str>,
    output_format: TranscriptionFormat,
    settings: &TranscriptionSettings,
    cancellation: &CancellationGuard,
    progress: &SharedProgress,
) -> AdapterResult<SubtitleDocument> {
    let vad_args = whisper_vad_args(settings)?;
    let Some(duration_ms) = prepared_audio.duration_ms() else {
        return WhisperCppTranscriber::new(
            binary.to_path_buf(),
            model_path.to_path_buf(),
            vad_args,
        )
        .with_progress(progress.clone())
        .transcribe_cancellable(
            prepared_audio.path(),
            language,
            output_format,
            cancellation,
        );
    };
    let chunks = transcription_chunks(duration_ms);
    if chunks.len() == 1 {
        return WhisperCppTranscriber::new(
            binary.to_path_buf(),
            model_path.to_path_buf(),
            vad_args,
        )
        .with_progress(progress.clone())
        .transcribe_cancellable(
            prepared_audio.path(),
            language,
            output_format,
            cancellation,
        );
    }

    transcribe_long_audio(
        binary,
        model_path,
        prepared_audio.path(),
        language,
        output_format,
        settings,
        cancellation,
        progress,
        &chunks,
        &vad_args,
    )
}

#[allow(clippy::too_many_arguments)]
fn transcribe_prepared_audio_incremental(
    binary: &Path,
    model_path: &Path,
    prepared_audio: &PreparedAudio,
    language: Option<&str>,
    output_format: TranscriptionFormat,
    settings: &TranscriptionSettings,
    cancellation: &CancellationGuard,
    progress: &SharedProgress,
    observer: &mut dyn TranscriptionChunkObserver,
    timeline_offset_ms: i64,
) -> AdapterResult<(SubtitleDocument, TranscriptionCleanupStats)> {
    let vad_args = whisper_vad_args(settings)?;
    let duration_ms = prepared_audio.duration_ms();
    let chunks = duration_ms.map(transcription_chunks).unwrap_or_else(|| {
        vec![TranscriptionChunk {
            input_start_ms: 0,
            input_end_ms: 0,
            core_start_ms: 0,
            core_end_ms: 0,
        }]
    });
    let whisper_format = if output_format == TranscriptionFormat::Txt {
        TranscriptionFormat::Srt
    } else {
        output_format
    };

    if chunks.len() == 1 {
        let descriptor = chunk_descriptor(0, chunks[0]);
        let cached = observer.load(descriptor, whisper_format)?;
        let resumed = cached.is_some();
        let mut document = match cached {
            Some(segments) => SubtitleDocument {
                path: prepared_audio.path().to_path_buf(),
                format: whisper_format.extension().to_owned(),
                segments,
                header: None,
                passthrough_blocks: Vec::new(),
                metadata: subbake_core::SubtitleDocumentMetadata::None,
            },
            None => {
                let mut document = WhisperCppTranscriber::new(
                    binary.to_path_buf(),
                    model_path.to_path_buf(),
                    vad_args.clone(),
                )
                .with_progress(progress.clone())
                .transcribe_cancellable(
                    prepared_audio.path(),
                    language,
                    whisper_format,
                    cancellation,
                )?;
                shift_document_timestamps(&mut document, timeline_offset_ms);
                document
            }
        };
        let raw_last_timestamp = last_timed_end_ms_on_audio(&document, timeline_offset_ms);
        let cleanup = if settings.filter_hallucinations && !resumed {
            clean_transcription_document(&mut document)
        } else {
            TranscriptionCleanupStats::default()
        };
        assign_stable_chunk_ids(&mut document.segments, 0, whisper_format);
        observer.stable(StableTranscriptionChunk {
            descriptor,
            format: whisper_format,
            segments: document.segments.clone(),
            resumed,
        })?;
        validate_transcription_coverage(
            &document,
            raw_last_timestamp,
            duration_ms,
            cleanup,
            timeline_offset_ms,
        )?;
        if output_format == TranscriptionFormat::Txt {
            clear_timing_for_text(&mut document);
        }
        return Ok((document, cleanup));
    }

    transcribe_long_audio_incremental(
        binary,
        model_path,
        prepared_audio.path(),
        language,
        output_format,
        settings,
        cancellation,
        progress,
        &chunks,
        duration_ms,
        observer,
        &vad_args,
        timeline_offset_ms,
    )
}

#[allow(clippy::too_many_arguments)]
fn transcribe_long_audio_incremental(
    binary: &Path,
    model_path: &Path,
    audio_path: &Path,
    language: Option<&str>,
    output_format: TranscriptionFormat,
    settings: &TranscriptionSettings,
    cancellation: &CancellationGuard,
    progress: &SharedProgress,
    chunks: &[TranscriptionChunk],
    duration_ms: Option<u64>,
    observer: &mut dyn TranscriptionChunkObserver,
    vad_args: &[String],
    timeline_offset_ms: i64,
) -> AdapterResult<(SubtitleDocument, TranscriptionCleanupStats)> {
    let runtime_dir = settings
        .runtime_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(".subbake"));
    let chunk_root = runtime_dir.join("tmp").join("transcription");
    fs::create_dir_all(&chunk_root).map_err(|source| {
        AdapterError::external_io(
            "create transcription chunk root",
            Some(chunk_root.clone()),
            source,
        )
    })?;
    let chunk_dir = unique_audio_temp_dir(&chunk_root)?;
    let whisper_format = if output_format == TranscriptionFormat::Txt {
        TranscriptionFormat::Srt
    } else {
        output_format
    };
    let mut merged: Option<SubtitleDocument> = None;
    let mut cleanup = TranscriptionCleanupStats::default();
    let mut raw_last_timestamp = None;

    for (index, chunk) in chunks.iter().copied().enumerate() {
        check_cancelled(cancellation)?;
        let descriptor = chunk_descriptor(index, chunk);
        let cached = observer.load(descriptor, whisper_format)?;
        let resumed = cached.is_some();
        let mut document = match cached {
            Some(segments) => SubtitleDocument {
                path: audio_path.to_path_buf(),
                format: whisper_format.extension().to_owned(),
                segments,
                header: None,
                passthrough_blocks: Vec::new(),
                metadata: subbake_core::SubtitleDocumentMetadata::None,
            },
            None => {
                let chunk_path = chunk_dir.path().join(format!("chunk-{index:04}.wav"));
                extract_audio_chunk(audio_path, &chunk_path, chunk, cancellation)?;
                let progress_start = (index as u64 * 100) / chunks.len() as u64;
                let progress_end = ((index as u64 + 1) * 100) / chunks.len() as u64;
                let mut extra_args = vad_args.to_vec();
                extra_args.extend(["--max-context".to_owned(), "0".to_owned()]);
                let transcriber = WhisperCppTranscriber::new(
                    binary.to_path_buf(),
                    model_path.to_path_buf(),
                    extra_args,
                )
                .with_progress(progress.clone())
                .with_progress_window(progress_start, progress_end);
                let mut document = transcriber.transcribe_cancellable(
                    &chunk_path,
                    language,
                    whisper_format,
                    cancellation,
                )?;
                let _ = fs::remove_file(&chunk_path);
                select_and_shift_chunk_segments(
                    &mut document.segments,
                    chunk,
                    index + 1 == chunks.len(),
                    whisper_format,
                );
                shift_document_timestamps(&mut document, timeline_offset_ms);
                document
            }
        };
        raw_last_timestamp =
            last_timed_end_ms_on_audio(&document, timeline_offset_ms).or(raw_last_timestamp);
        if settings.filter_hallucinations && !resumed {
            let chunk_cleanup = clean_transcription_document(&mut document);
            cleanup.removed_empty_or_silence += chunk_cleanup.removed_empty_or_silence;
            cleanup.removed_repeated += chunk_cleanup.removed_repeated;
        }
        assign_stable_chunk_ids(&mut document.segments, index, whisper_format);
        observer.stable(StableTranscriptionChunk {
            descriptor,
            format: whisper_format,
            segments: document.segments.clone(),
            resumed,
        })?;
        if let Some(output) = &mut merged {
            output.segments.extend(document.segments);
        } else {
            merged = Some(document);
        }
    }

    let mut document = merged.ok_or_else(|| {
        AdapterError::invalid_input("long-audio transcription produced no chunks")
    })?;
    validate_transcription_coverage(
        &document,
        raw_last_timestamp,
        duration_ms,
        cleanup,
        timeline_offset_ms,
    )?;
    if output_format == TranscriptionFormat::Txt {
        clear_timing_for_text(&mut document);
    }
    Ok((document, cleanup))
}

fn chunk_descriptor(index: usize, chunk: TranscriptionChunk) -> TranscriptionChunkDescriptor {
    TranscriptionChunkDescriptor {
        index,
        input_start_ms: chunk.input_start_ms,
        input_end_ms: chunk.input_end_ms,
        core_start_ms: chunk.core_start_ms,
        core_end_ms: chunk.core_end_ms,
    }
}

fn assign_stable_chunk_ids(
    segments: &mut [SubtitleSegment],
    chunk_index: usize,
    format: TranscriptionFormat,
) {
    for (local_index, segment) in segments.iter_mut().enumerate() {
        let id = format!("c{chunk_index:04}-{local_index:06}");
        segment.id.clone_from(&id);
        segment.identifier = (format == TranscriptionFormat::Srt).then_some(id);
    }
}

fn clear_timing_for_text(document: &mut SubtitleDocument) {
    for segment in &mut document.segments {
        segment.start = None;
        segment.end = None;
        segment.identifier = None;
        segment.settings = None;
    }
    document.format = "txt".to_owned();
}

#[allow(clippy::too_many_arguments)]
fn transcribe_long_audio(
    binary: &Path,
    model_path: &Path,
    audio_path: &Path,
    language: Option<&str>,
    output_format: TranscriptionFormat,
    settings: &TranscriptionSettings,
    cancellation: &CancellationGuard,
    progress: &SharedProgress,
    chunks: &[TranscriptionChunk],
    vad_args: &[String],
) -> AdapterResult<SubtitleDocument> {
    let runtime_dir = settings
        .runtime_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(".subbake"));
    let chunk_root = runtime_dir.join("tmp").join("transcription");
    fs::create_dir_all(&chunk_root).map_err(|source| {
        AdapterError::external_io(
            "create transcription chunk root",
            Some(chunk_root.clone()),
            source,
        )
    })?;
    let chunk_dir = unique_audio_temp_dir(&chunk_root)?;
    let whisper_format = if output_format == TranscriptionFormat::Txt {
        TranscriptionFormat::Srt
    } else {
        output_format
    };
    let mut merged: Option<SubtitleDocument> = None;

    for (index, chunk) in chunks.iter().copied().enumerate() {
        check_cancelled(cancellation)?;
        let chunk_path = chunk_dir.path().join(format!("chunk-{index:04}.wav"));
        extract_audio_chunk(audio_path, &chunk_path, chunk, cancellation)?;
        let progress_start = (index as u64 * 100) / chunks.len() as u64;
        let progress_end = ((index as u64 + 1) * 100) / chunks.len() as u64;
        let mut extra_args = vad_args.to_vec();
        extra_args.extend(["--max-context".to_owned(), "0".to_owned()]);
        let transcriber =
            WhisperCppTranscriber::new(binary.to_path_buf(), model_path.to_path_buf(), extra_args)
                .with_progress(progress.clone())
                .with_progress_window(progress_start, progress_end);
        let mut document = transcriber.transcribe_cancellable(
            &chunk_path,
            language,
            whisper_format,
            cancellation,
        )?;
        let _ = fs::remove_file(&chunk_path);
        select_and_shift_chunk_segments(
            &mut document.segments,
            chunk,
            index + 1 == chunks.len(),
            whisper_format,
        );
        if let Some(output) = &mut merged {
            output.segments.extend(document.segments);
        } else {
            merged = Some(document);
        }
    }

    let mut document = merged.ok_or_else(|| {
        AdapterError::invalid_input("long-audio transcription produced no chunks")
    })?;
    for (index, segment) in document.segments.iter_mut().enumerate() {
        let id = (index + 1).to_string();
        segment.id.clone_from(&id);
        segment.identifier = (whisper_format == TranscriptionFormat::Srt).then_some(id);
    }
    if output_format == TranscriptionFormat::Txt {
        document.format = "txt".to_owned();
    }
    Ok(document)
}

fn extract_audio_chunk(
    source: &Path,
    destination: &Path,
    chunk: TranscriptionChunk,
    cancellation: &CancellationGuard,
) -> AdapterResult<()> {
    let start = format_seconds(chunk.input_start_ms);
    let duration = format_seconds(chunk.input_end_ms - chunk.input_start_ms);
    let output = ProcessSupervisor::run(
        Command::new("ffmpeg").args([
            "-nostdin",
            "-hide_banner",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            &start,
            "-i",
            &source.to_string_lossy(),
            "-t",
            &duration,
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            &destination.to_string_lossy(),
        ]),
        cancellation,
        "extract transcription chunk",
    )?;
    if !output.status.success() || !destination.is_file() {
        return Err(AdapterError::ChildProcess {
            program: "ffmpeg",
            status: output.status.code(),
            message: child_diagnostics(&output, "failed to extract transcription chunk"),
        });
    }
    Ok(())
}

fn format_seconds(milliseconds: u64) -> String {
    format!("{}.{:03}", milliseconds / 1_000, milliseconds % 1_000)
}

fn select_and_shift_chunk_segments(
    segments: &mut Vec<SubtitleSegment>,
    chunk: TranscriptionChunk,
    is_last: bool,
    format: TranscriptionFormat,
) {
    let separator = if format == TranscriptionFormat::Vtt {
        '.'
    } else {
        ','
    };
    let mut selected = Vec::new();
    for mut segment in segments.drain(..) {
        let Some(start) = segment.start.as_deref().and_then(parse_timestamp_ms) else {
            continue;
        };
        let Some(end) = segment.end.as_deref().and_then(parse_timestamp_ms) else {
            continue;
        };
        let global_start = start.saturating_add(chunk.input_start_ms);
        let global_end = end.saturating_add(chunk.input_start_ms);
        let midpoint = global_start.saturating_add(global_end.saturating_sub(global_start) / 2);
        let in_core = midpoint >= chunk.core_start_ms
            && (midpoint < chunk.core_end_ms || (is_last && midpoint <= chunk.core_end_ms));
        if in_core {
            segment.start = Some(format_timestamp_ms(global_start, separator));
            segment.end = Some(format_timestamp_ms(global_end, separator));
            selected.push(segment);
        }
    }
    *segments = selected;
}

fn parse_timestamp_ms(value: &str) -> Option<u64> {
    let normalized = value.trim().replace(',', ".");
    let mut parts = normalized.split(':');
    let hours = parts.next()?.parse::<u64>().ok()?;
    let minutes = parts.next()?.parse::<u64>().ok()?;
    let seconds_part = parts.next()?;
    if parts.next().is_some() || minutes >= 60 {
        return None;
    }
    let (seconds, fraction) = seconds_part.split_once('.').unwrap_or((seconds_part, "0"));
    let seconds = seconds.parse::<u64>().ok()?;
    if seconds >= 60 {
        return None;
    }
    let mut milliseconds = fraction.chars().take(3).collect::<String>();
    while milliseconds.len() < 3 {
        milliseconds.push('0');
    }
    Some((((hours * 60 + minutes) * 60 + seconds) * 1_000) + milliseconds.parse::<u64>().ok()?)
}

fn format_timestamp_ms(milliseconds: u64, separator: char) -> String {
    let hours = milliseconds / 3_600_000;
    let minutes = (milliseconds / 60_000) % 60;
    let seconds = (milliseconds / 1_000) % 60;
    let fraction = milliseconds % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}{separator}{fraction:03}")
}

fn last_timed_end_ms(document: &SubtitleDocument) -> Option<u64> {
    document
        .segments
        .iter()
        .filter_map(|segment| segment.end.as_deref().and_then(parse_timestamp_ms))
        .max()
}

fn last_timed_end_ms_on_audio(document: &SubtitleDocument, timeline_offset_ms: i64) -> Option<u64> {
    last_timed_end_ms(document)
        .map(|timestamp| unapply_timeline_offset(timestamp, timeline_offset_ms))
}

fn unapply_timeline_offset(timestamp_ms: u64, timeline_offset_ms: i64) -> u64 {
    if timeline_offset_ms >= 0 {
        timestamp_ms.saturating_sub(timeline_offset_ms as u64)
    } else {
        timestamp_ms.saturating_add(timeline_offset_ms.unsigned_abs())
    }
}

fn shift_document_timestamps(document: &mut SubtitleDocument, timeline_offset_ms: i64) {
    if timeline_offset_ms == 0 {
        return;
    }
    document.segments.retain_mut(|segment| {
        let (Some(start), Some(end)) = (
            segment.start.as_deref().and_then(parse_timestamp_ms),
            segment.end.as_deref().and_then(parse_timestamp_ms),
        ) else {
            return true;
        };
        let shifted_end = i128::from(end) + i128::from(timeline_offset_ms);
        if shifted_end <= 0 {
            return false;
        }
        let shifted_start = (i128::from(start) + i128::from(timeline_offset_ms)).max(0) as u64;
        let shifted_end = shifted_end as u64;
        let separator = if segment
            .start
            .as_deref()
            .is_some_and(|value| value.contains('.'))
        {
            '.'
        } else {
            ','
        };
        segment.start = Some(format_timestamp_ms(shifted_start, separator));
        segment.end = Some(format_timestamp_ms(shifted_end, separator));
        true
    });
}

fn validate_transcription_coverage(
    document: &SubtitleDocument,
    raw_last_timestamp: Option<u64>,
    duration_ms: Option<u64>,
    cleanup: TranscriptionCleanupStats,
    timeline_offset_ms: i64,
) -> AdapterResult<()> {
    let Some(duration_ms) = duration_ms.filter(|duration| *duration > LONG_AUDIO_THRESHOLD_MS)
    else {
        return Ok(());
    };
    let last_timestamp = last_timed_end_ms_on_audio(document, timeline_offset_ms).unwrap_or(0);
    let trailing_gap = duration_ms.saturating_sub(last_timestamp);
    if trailing_gap <= MAX_UNCOVERED_TRAILING_MS {
        return Ok(());
    }
    Err(AdapterError::invalid_input(format!(
        "transcription is incomplete: the media is {}, but the last retained cue ends at {} ({} missing at the end); raw output reached {} and cleanup removed {} repeated cues. No output was written. This usually indicates a Whisper hallucination loop",
        display_duration(duration_ms),
        display_duration(last_timestamp),
        display_duration(trailing_gap),
        raw_last_timestamp.map_or_else(|| "no timed cue".to_owned(), display_duration),
        cleanup.removed_repeated,
    )))
}

fn display_duration(milliseconds: u64) -> String {
    let total_seconds = milliseconds / 1_000;
    let hours = total_seconds / 3_600;
    let minutes = (total_seconds / 60) % 60;
    let seconds = total_seconds % 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct PreparedAudio {
    path: PathBuf,
    duration_ms: Option<u64>,
    timeline_offset_ms: i64,
    _temporary_dir: Option<AudioTempDirectory>,
}

impl PreparedAudio {
    fn borrowed(path: &Path, duration_ms: Option<u64>) -> Self {
        Self {
            path: path.to_path_buf(),
            duration_ms,
            timeline_offset_ms: 0,
            _temporary_dir: None,
        }
    }

    fn temporary(
        path: PathBuf,
        duration_ms: Option<u64>,
        timeline_offset_ms: i64,
        directory: AudioTempDirectory,
    ) -> Self {
        Self {
            path,
            duration_ms,
            timeline_offset_ms,
            _temporary_dir: Some(directory),
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn duration_ms(&self) -> Option<u64> {
        self.duration_ms
    }

    fn timeline_offset_ms(&self) -> i64 {
        self.timeline_offset_ms
    }
}

#[derive(Debug)]
struct AudioTempDirectory(PathBuf);

impl AudioTempDirectory {
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for AudioTempDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn prepare_audio(
    media_path: &Path,
    settings: &TranscriptionSettings,
    cancellation: &CancellationGuard,
    progress: &SharedProgress,
) -> AdapterResult<PreparedAudio> {
    prepare_audio_with_programs(
        media_path,
        settings,
        cancellation,
        progress,
        Path::new("ffmpeg"),
        Path::new("ffprobe"),
    )
}

fn prepare_audio_with_programs(
    media_path: &Path,
    settings: &TranscriptionSettings,
    cancellation: &CancellationGuard,
    progress: &SharedProgress,
    ffmpeg: &Path,
    ffprobe: &Path,
) -> AdapterResult<PreparedAudio> {
    check_cancelled(cancellation)?;
    if is_wav_ext(media_path) {
        let duration_ms = wav_duration_ms(media_path);
        let mut done = ProgressEvent::running(
            TaskKind::Transcription,
            "PREPARE_AUDIO",
            1,
            Some(1),
            ProgressUnit::Steps,
        );
        done.state = TaskState::Completed;
        progress.emit(done);
        return Ok(PreparedAudio::borrowed(media_path, duration_ms));
    }

    let audio_info = probe_audio_info(ffprobe, media_path, cancellation)?;
    validate_audio_decodable(ffmpeg, media_path, &audio_info.codec, cancellation)?;
    let runtime_dir = settings
        .runtime_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(".subbake"));
    let temp_root = runtime_dir.join("tmp").join("transcription");
    fs::create_dir_all(&temp_root).map_err(|source| {
        AdapterError::external_io(
            "create transcription temp root",
            Some(temp_root.clone()),
            source,
        )
    })?;
    let temp_dir = unique_audio_temp_dir(&temp_root)?;
    let output = temp_dir.path().join("audio.wav");
    let progress_duration_ms = audio_info.duration_ms;
    progress.emit(ProgressEvent::running(
        TaskKind::Transcription,
        "PREPARE_AUDIO",
        0,
        progress_duration_ms,
        ProgressUnit::Duration,
    ));

    let mut command = Command::new(ffmpeg);
    command.args([
        "-nostdin",
        "-hide_banner",
        "-y",
        "-nostats",
        "-loglevel",
        "error",
        "-i",
        &media_path.to_string_lossy(),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-progress",
        "pipe:1",
        &output.to_string_lossy(),
    ]);
    let mut processed_ms = 0_u64;
    let out = ProcessSupervisor::run_with_stdout_lines(
        &mut command,
        cancellation,
        "ffmpeg audio preparation",
        |line| {
            if let Some(current) = parse_ffmpeg_progress_ms(line) {
                processed_ms = current;
                progress.emit(ProgressEvent::running(
                    TaskKind::Transcription,
                    "PREPARE_AUDIO",
                    progress_duration_ms.map_or(current, |total| current.min(total)),
                    progress_duration_ms,
                    ProgressUnit::Duration,
                ));
            }
        },
    )?;
    if !out.status.success() {
        return Err(AdapterError::ChildProcess {
            program: "ffmpeg",
            status: out.status.code(),
            message: child_diagnostics(&out, "ffmpeg audio preparation failed"),
        });
    }
    check_cancelled(cancellation)?;
    if !output.is_file() {
        return Err(AdapterError::ChildProcess {
            program: "ffmpeg",
            status: out.status.code(),
            message: "ffmpeg did not create the prepared WAV file".to_owned(),
        });
    }
    let mut done = ProgressEvent::running(
        TaskKind::Transcription,
        "PREPARE_AUDIO",
        progress_duration_ms.unwrap_or(processed_ms),
        progress_duration_ms,
        ProgressUnit::Duration,
    );
    done.state = TaskState::Completed;
    progress.emit(done);
    let duration_ms = wav_duration_ms(&output).or(progress_duration_ms);
    Ok(PreparedAudio::temporary(
        output,
        duration_ms,
        audio_info.start_time_ms,
        temp_dir,
    ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MediaAudioInfo {
    codec: String,
    duration_ms: Option<u64>,
    start_time_ms: i64,
}

fn probe_audio_info(
    ffprobe: &Path,
    media_path: &Path,
    cancellation: &CancellationGuard,
) -> AdapterResult<MediaAudioInfo> {
    let output = ProcessSupervisor::run(
        Command::new(ffprobe).args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,start_time:format=duration",
            "-of",
            "json",
            &media_path.to_string_lossy(),
        ]),
        cancellation,
        "ffprobe media audio streams",
    )?;
    if !output.status.success() {
        return Err(AdapterError::ChildProcess {
            program: "ffprobe",
            status: output.status.code(),
            message: child_diagnostics(&output, "failed to inspect media audio streams"),
        });
    }
    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(|source| AdapterError::Serialization {
            context: "parse ffprobe audio stream response",
            source,
        })?;
    let codec = value
        .get("streams")
        .and_then(serde_json::Value::as_array)
        .and_then(|streams| streams.first())
        .and_then(|stream| stream.get("codec_name"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            AdapterError::invalid_input(format!(
                "media contains no audio stream: {}",
                media_path.display()
            ))
        })?
        .to_owned();
    let duration_ms = value
        .pointer("/format/duration")
        .and_then(serde_json::Value::as_str)
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
        .map(|seconds| (seconds * 1_000.0).round() as u64);
    let start_time_ms = value
        .pointer("/streams/0/start_time")
        .and_then(serde_json::Value::as_str)
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|seconds| seconds.is_finite())
        .map(|seconds| (seconds * 1_000.0).round() as i64)
        .unwrap_or(0);
    Ok(MediaAudioInfo {
        codec,
        duration_ms,
        start_time_ms,
    })
}

fn validate_audio_decodable(
    ffmpeg: &Path,
    media_path: &Path,
    codec: &str,
    cancellation: &CancellationGuard,
) -> AdapterResult<()> {
    let output = ProcessSupervisor::run(
        Command::new(ffmpeg).args([
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-i",
            &media_path.to_string_lossy(),
            "-map",
            "0:a:0",
            "-t",
            "0.1",
            "-vn",
            "-sn",
            "-dn",
            "-f",
            "null",
            "-",
        ]),
        cancellation,
        "ffmpeg audio decoder check",
    )?;
    if output.status.success() {
        return Ok(());
    }
    let diagnostics = child_diagnostics(&output, "audio decoder check failed");
    let message = if diagnostics.contains("no decoder found")
        || diagnostics.contains("Unknown decoder")
        || diagnostics.contains("Decoder not found")
    {
        format!(
            "audio stream uses `{codec}`, but this FFmpeg build cannot decode it; install an FFmpeg build with `{codec}` decoder support"
        )
    } else {
        format!("cannot decode the first audio stream (`{codec}`): {diagnostics}")
    };
    Err(AdapterError::ChildProcess {
        program: "ffmpeg",
        status: output.status.code(),
        message,
    })
}

fn parse_ffmpeg_progress_ms(line: &str) -> Option<u64> {
    line.strip_prefix("out_time_us=")
        .and_then(|value| value.parse::<u64>().ok())
        .map(|microseconds| microseconds / 1_000)
}

fn unique_audio_temp_dir(root: &Path) -> AdapterResult<AudioTempDirectory> {
    static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    for _ in 0..100 {
        let nonce = NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = root.join(format!("{}-{nonce}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(AudioTempDirectory(path)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "unable to allocate transcription temp directory",
    )
    .into())
}

fn check_cancelled(cancellation: &CancellationGuard) -> AdapterResult<()> {
    cancellation.check().map_err(AdapterError::from)
}

fn child_diagnostics(output: &std::process::Output, fallback: &str) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let message = [stderr.trim(), stdout.trim()]
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

fn is_wav_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
}

fn wav_duration_ms(path: &Path) -> Option<u64> {
    let mut file = fs::File::open(path).ok()?;
    let mut header = [0_u8; 12];
    file.read_exact(&mut header).ok()?;
    if &header[..4] != b"RIFF" || &header[8..] != b"WAVE" {
        return None;
    }
    let mut byte_rate = None;
    loop {
        let mut chunk_header = [0_u8; 8];
        file.read_exact(&mut chunk_header).ok()?;
        let chunk_size = u32::from_le_bytes(chunk_header[4..].try_into().ok()?) as u64;
        match &chunk_header[..4] {
            b"fmt " if chunk_size >= 12 => {
                let mut format = [0_u8; 12];
                file.read_exact(&mut format).ok()?;
                byte_rate = Some(u32::from_le_bytes(format[8..12].try_into().ok()?) as u64);
                file.seek(SeekFrom::Current((chunk_size - 12) as i64))
                    .ok()?;
            }
            b"data" => {
                let rate = byte_rate.filter(|rate| *rate > 0)?;
                return Some(chunk_size.saturating_mul(1_000) / rate);
            }
            _ => {
                file.seek(SeekFrom::Current(chunk_size as i64)).ok()?;
            }
        }
        if chunk_size % 2 == 1 {
            file.seek(SeekFrom::Current(1)).ok()?;
        }
    }
}

pub fn apply_whisper_storage(transcription: &mut TranscriptionSettings, storage: &StorageSettings) {
    transcription.runtime_dir = Some(
        storage
            .runtime_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(".subbake")),
    );
    transcription.whisper_binary_path = Some(
        storage
            .whisper_binary_path
            .clone()
            .unwrap_or_else(|| default_whisper_binary_path_for(storage.runtime_dir.as_deref())),
    );
    transcription.whisper_models_dir = Some(
        storage
            .whisper_models_dir
            .clone()
            .unwrap_or_else(|| default_whisper_models_dir_for(storage.runtime_dir.as_deref())),
    );
}

pub fn apply_whisper_configuration(
    transcription: &mut TranscriptionSettings,
    settings: &ResolvedSettings,
) {
    apply_whisper_storage(transcription, &settings.storage);
    if transcription.model.is_none() {
        transcription.model = settings.transcription.model.clone();
    }
    if transcription.vad_enabled.is_none() {
        transcription.vad_enabled = Some(settings.transcription.vad_enabled);
    }
    if transcription.vad_model.is_none() {
        transcription.vad_model = Some(settings.transcription.vad_model.clone());
    }
    if transcription.vad_threshold.is_none() {
        transcription.vad_threshold = Some(settings.transcription.vad_threshold);
    }
    if transcription.vad_min_speech_duration_ms.is_none() {
        transcription.vad_min_speech_duration_ms =
            Some(settings.transcription.vad_min_speech_duration_ms);
    }
    if transcription.vad_min_silence_duration_ms.is_none() {
        transcription.vad_min_silence_duration_ms =
            Some(settings.transcription.vad_min_silence_duration_ms);
    }
    if transcription.vad_speech_pad_ms.is_none() {
        transcription.vad_speech_pad_ms = Some(settings.transcription.vad_speech_pad_ms);
    }
}

fn default_output_path(media_path: &Path, fmt: TranscriptionFormat) -> PathBuf {
    media_path.with_extension(fmt.extension())
}

fn render_sidecar(path: &Path, output: &Path, fmt: TranscriptionFormat) -> AdapterResult<()> {
    let doc = read_document(path)?;
    if fmt != TranscriptionFormat::Txt
        && doc
            .segments
            .iter()
            .any(|s| s.start.is_none() || s.end.is_none())
    {
        return Err(AdapterError::invalid_input(
            "sidecar lacks timing data; use --format txt or a timed subtitle file",
        ));
    }
    let opts = RenderOptions::new(false, Some(fmt.extension().to_owned()));
    render_and_write_document(&doc, &doc.segments, output, &opts)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn default_output_uses_format() {
        assert_eq!(
            default_output_path(Path::new("/m.mp4"), TranscriptionFormat::Vtt),
            PathBuf::from("/m.vtt"),
        );
    }

    #[test]
    fn transcribe_from_timed_sidecar() {
        let root = t("sidecar");
        fs::create_dir_all(&root).expect("mkdtemp");
        let src = root.join("in.srt");
        fs::write(&src, "1\n00:00:0,0-->00:00:1,0\nhello\n\n").expect("write src");
        let out = root.join("out.srt");
        let r = transcribe_media(TranscriptionRequest {
            media_path: root.join("x.mp4"),
            output_path: Some(out.clone()),
            overwrite: true,
            settings: TranscriptionSettings {
                sidecar_path: Some(src),
                ..Default::default()
            },
        })
        .expect("transcribe");
        assert_eq!(r.output_path, out);
        assert_eq!(r.provider, "sidecar");
        assert_eq!(r.model, "none");
        assert_eq!(r.language, "Auto");
        assert_eq!(r.output_format, TranscriptionFormat::Srt);
        assert_eq!(r.subtitle_entries, 1);
        assert!(
            fs::read_to_string(&out)
                .expect("read out")
                .contains("hello")
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn conservative_cleanup_removes_silence_markers_and_third_repetition() {
        let mut document = SubtitleDocument {
            path: PathBuf::from("whisper.srt"),
            format: "srt".to_owned(),
            segments: ["[BLANK_AUDIO]", "Hello", "Hello", "Hello"]
                .into_iter()
                .enumerate()
                .map(|(index, text)| SubtitleSegment {
                    id: (index + 1).to_string(),
                    text: text.to_owned(),
                    start: None,
                    end: None,
                    identifier: None,
                    settings: None,
                    semantic: Default::default(),
                })
                .collect(),
            header: None,
            passthrough_blocks: Vec::new(),
            metadata: Default::default(),
        };
        let stats = clean_transcription_document(&mut document);
        assert_eq!(stats.removed_empty_or_silence, 1);
        assert_eq!(stats.removed_repeated, 1);
        assert_eq!(document.segments.len(), 2);
    }

    #[test]
    fn long_audio_chunks_overlap_without_hard_cutting_the_core_boundary() {
        let chunks = transcription_chunks(25 * 60 * 1_000);
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks[0],
            TranscriptionChunk {
                input_start_ms: 0,
                input_end_ms: 630_000,
                core_start_ms: 0,
                core_end_ms: 600_000,
            }
        );
        assert_eq!(
            chunks[1],
            TranscriptionChunk {
                input_start_ms: 570_000,
                input_end_ms: 1_230_000,
                core_start_ms: 600_000,
                core_end_ms: 1_200_000,
            }
        );
        assert_eq!(chunks[2].input_start_ms, 1_170_000);
        assert_eq!(chunks[2].core_end_ms, 1_500_000);
    }

    #[test]
    fn overlap_merge_keeps_cross_boundary_dialogue_once() {
        let mut left = vec![timed_segment(
            "left",
            "crossing dialogue",
            "00:09:58,000",
            "00:10:03,000",
        )];
        select_and_shift_chunk_segments(
            &mut left,
            TranscriptionChunk {
                input_start_ms: 0,
                input_end_ms: 630_000,
                core_start_ms: 0,
                core_end_ms: 600_000,
            },
            false,
            TranscriptionFormat::Srt,
        );
        assert!(left.is_empty());

        let mut right = vec![timed_segment(
            "right",
            "crossing dialogue",
            "00:00:28,000",
            "00:00:33,000",
        )];
        select_and_shift_chunk_segments(
            &mut right,
            TranscriptionChunk {
                input_start_ms: 570_000,
                input_end_ms: 1_230_000,
                core_start_ms: 600_000,
                core_end_ms: 1_200_000,
            },
            false,
            TranscriptionFormat::Srt,
        );
        assert_eq!(right.len(), 1);
        assert_eq!(right[0].start.as_deref(), Some("00:09:58,000"));
        assert_eq!(right[0].end.as_deref(), Some("00:10:03,000"));
    }

    #[test]
    fn incomplete_long_transcription_is_rejected_before_rendering() {
        let document = SubtitleDocument {
            path: PathBuf::from("whisper.srt"),
            format: "srt".to_owned(),
            segments: vec![timed_segment(
                "1",
                "last usable line",
                "00:16:50,000",
                "00:17:00,000",
            )],
            header: None,
            passthrough_blocks: Vec::new(),
            metadata: Default::default(),
        };
        let error = validate_transcription_coverage(
            &document,
            Some(8_500_000),
            Some(8_582_000),
            TranscriptionCleanupStats {
                removed_empty_or_silence: 0,
                removed_repeated: 250,
            },
            0,
        )
        .expect_err("truncated long transcription must fail");
        let message = error.to_string();
        assert!(message.contains("transcription is incomplete"));
        assert!(message.contains("cleanup removed 250 repeated cues"));
    }

    #[test]
    fn wav_duration_reads_pcm_header_without_external_tools() {
        let root = t("wav-duration");
        fs::create_dir_all(&root).expect("create root");
        let path = root.join("audio.wav");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&32_036_u32.to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&16_000_u32.to_le_bytes());
        bytes.extend_from_slice(&32_000_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&32_000_u32.to_le_bytes());
        bytes.resize(bytes.len() + 32_000, 0);
        fs::write(&path, bytes).expect("write wav");
        assert_eq!(wav_duration_ms(&path), Some(1_000));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn untimed_sidecar_requires_txt() {
        let root = t("untimed");
        fs::create_dir_all(&root).expect("mkdtemp");
        let src = root.join("in.txt");
        fs::write(&src, "hello\n").expect("write src");
        let e = transcribe_media(TranscriptionRequest {
            media_path: root.join("x.mp4"),
            output_path: None,
            overwrite: true,
            settings: TranscriptionSettings {
                sidecar_path: Some(src),
                ..Default::default()
            },
        })
        .expect_err("untimed should error");
        let _ = fs::remove_dir_all(&root);
        assert!(e.to_string().contains("lacks timing"));
    }

    #[test]
    fn existing_output_is_rejected_before_sidecar_render_when_overwrite_is_false() {
        let root = t("overwrite");
        fs::create_dir_all(&root).expect("create root");
        let sidecar = root.join("in.srt");
        let output = root.join("out.srt");
        fs::write(&sidecar, "1\n00:00:00,000 --> 00:00:01,000\nnew\n").expect("write sidecar");
        fs::write(&output, "existing\n").expect("write existing output");

        let error = transcribe_media(TranscriptionRequest {
            media_path: root.join("media.wav"),
            output_path: Some(output.clone()),
            overwrite: false,
            settings: TranscriptionSettings {
                sidecar_path: Some(sidecar),
                ..Default::default()
            },
        })
        .expect_err("existing output must fail");
        let content = fs::read_to_string(&output).expect("read output");
        let _ = fs::remove_dir_all(&root);

        assert!(error.to_string().contains("overwrite is false"));
        assert_eq!(content, "existing\n");
    }

    #[test]
    fn wav_extension_check() {
        assert!(is_wav_ext(Path::new("x.wav")));
        assert!(is_wav_ext(Path::new("x.WAV")));
        assert!(!is_wav_ext(Path::new("x.mp3")));
        assert!(!is_wav_ext(Path::new("x.mp4")));
    }

    #[test]
    fn whisper_progress_parser_accepts_cli_callback_lines() {
        assert_eq!(
            parse_whisper_progress("whisper_print_progress_callback: progress =  42%"),
            Some(42)
        );
        assert_eq!(parse_whisper_progress("progress = 100%"), Some(100));
        assert_eq!(parse_whisper_progress("system_info: threads = 16"), None);
    }

    #[test]
    fn whisper_thread_default_uses_half_the_parallelism_with_a_safe_cap() {
        assert_eq!(recommended_whisper_threads(1), 1);
        assert_eq!(recommended_whisper_threads(8), 4);
        assert_eq!(recommended_whisper_threads(32), 16);
        assert_eq!(recommended_whisper_threads(128), 16);
    }

    #[test]
    fn explicit_model_wins_over_automatic_selection() {
        let root = model_dir("explicit", &["small", "medium-q8_0"]);
        let settings = TranscriptionSettings {
            model: Some("medium-q8_0".to_owned()),
            whisper_models_dir: Some(root.clone()),
            ..TranscriptionSettings::default()
        };

        let selected = resolve_whisper_model(&settings).expect("resolve explicit model");
        let _ = fs::remove_dir_all(root);

        assert_eq!(selected.name, "medium-q8_0");
        assert!(!selected.auto_selected);
    }

    #[test]
    fn one_installed_model_is_selected_automatically() {
        let root = model_dir("single", &["large-v3-turbo-q8_0"]);
        let settings = TranscriptionSettings {
            whisper_models_dir: Some(root.clone()),
            ..TranscriptionSettings::default()
        };

        let selected = resolve_whisper_model(&settings).expect("resolve only model");
        let _ = fs::remove_dir_all(root);

        assert_eq!(selected.name, "large-v3-turbo-q8_0");
        assert!(selected.auto_selected);
    }

    #[test]
    fn exact_small_is_preferred_when_multiple_models_are_installed() {
        let root = model_dir("small-default", &["medium", "small", "base"]);
        let settings = TranscriptionSettings {
            whisper_models_dir: Some(root.clone()),
            ..TranscriptionSettings::default()
        };

        let selected = resolve_whisper_model(&settings).expect("resolve small default");
        let _ = fs::remove_dir_all(root);

        assert_eq!(selected.name, "small");
        assert!(selected.auto_selected);
    }

    #[test]
    fn agent_policy_ranks_families_and_variants_deterministically() {
        let root = model_dir(
            "ranked",
            &["medium", "base", "small-q5_1", "small-q8_0", "small.en"],
        );
        let settings = TranscriptionSettings {
            whisper_models_dir: Some(root.clone()),
            multiple_model_policy: MultipleModelPolicy::PreferRanked,
            ..TranscriptionSettings::default()
        };

        let selected = resolve_whisper_model(&settings).expect("resolve ranked model");
        let _ = fs::remove_dir_all(root);

        assert_eq!(selected.name, "small-q8_0");
        assert!(selected.auto_selected);
    }

    #[test]
    fn cli_policy_lists_installed_models_when_multiple_need_a_choice() {
        let root = model_dir("multiple", &["medium", "base-q8_0"]);
        let settings = TranscriptionSettings {
            whisper_models_dir: Some(root.clone()),
            ..TranscriptionSettings::default()
        };

        let error = resolve_whisper_model(&settings).expect_err("CLI should require a choice");
        let _ = fs::remove_dir_all(root);
        let message = error.to_string();

        assert!(message.contains("multiple whisper.cpp models"));
        assert!(message.contains("base-q8_0"));
        assert!(message.contains("medium"));
    }

    #[test]
    fn no_installed_models_explains_how_to_download_one() {
        let root = t("no-models");
        let settings = TranscriptionSettings {
            whisper_models_dir: Some(root),
            ..TranscriptionSettings::default()
        };

        let error = resolve_whisper_model(&settings).expect_err("missing models should fail");

        assert!(error.to_string().contains("whisper model list"));
    }

    #[cfg(unix)]
    #[test]
    fn compressed_audio_is_normalized_with_progress_and_cleaned_on_drop() {
        use std::os::unix::fs::PermissionsExt;
        use std::sync::{Arc, Mutex};

        #[derive(Default)]
        struct Recorder(Mutex<Vec<ProgressEvent>>);
        impl subbake_core::ProgressSink for Recorder {
            fn emit(&self, event: ProgressEvent) {
                self.0.lock().expect("progress lock").push(event);
            }
        }

        let root = t("prepare-audio");
        fs::create_dir_all(&root).expect("create root");
        let input = root.join("input.mp3");
        fs::write(&input, b"compressed audio").expect("write input");
        let ffprobe = root.join("ffprobe");
        let ffmpeg = root.join("ffmpeg");
        fs::write(
            &ffprobe,
            "#!/bin/sh\nprintf '{\"streams\":[{\"codec_name\":\"mp3\",\"start_time\":\"10.0\"}],\"format\":{\"duration\":\"10.0\"}}\\n'\n",
        )
        .expect("write ffprobe");
        fs::write(
            &ffmpeg,
            "#!/bin/sh\ncase \" $* \" in *\" -f null - \"*) exit 0;; esac\nfor output in \"$@\"; do :; done\necho out_time_us=2500000\necho out_time_us=7500000\nprintf RIFF > \"$output\"\n",
        )
        .expect("write ffmpeg");
        fs::set_permissions(&ffprobe, fs::Permissions::from_mode(0o755)).expect("chmod ffprobe");
        fs::set_permissions(&ffmpeg, fs::Permissions::from_mode(0o755)).expect("chmod ffmpeg");
        let runtime_dir = root.join("runtime");
        let settings = TranscriptionSettings {
            runtime_dir: Some(runtime_dir.clone()),
            ..TranscriptionSettings::default()
        };
        let recorder = Arc::new(Recorder::default());
        let progress: SharedProgress = recorder.clone();

        let prepared = prepare_audio_with_programs(
            &input,
            &settings,
            &CancellationGuard::never(),
            &progress,
            &ffmpeg,
            &ffprobe,
        )
        .expect("prepare audio");
        let temporary_dir = prepared
            .path()
            .parent()
            .expect("temporary parent")
            .to_path_buf();
        assert!(prepared.path().is_file());
        assert_eq!(prepared.timeline_offset_ms(), 10_000);
        assert!(temporary_dir.starts_with(runtime_dir.join("tmp/transcription")));
        let events = recorder.0.lock().expect("progress lock");
        assert!(events.iter().any(|event| {
            event.stage == "PREPARE_AUDIO"
                && event.unit == ProgressUnit::Duration
                && event.current == 7_500
                && event.total == Some(10_000)
        }));
        drop(events);
        drop(prepared);
        assert!(!temporary_dir.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn media_timeline_offset_shifts_cues_without_changing_duration() {
        let mut document = SubtitleDocument {
            path: PathBuf::from("audio.srt"),
            format: "srt".to_owned(),
            segments: vec![timed_segment("1", "hello", "00:00:01,000", "00:00:02,500")],
            header: None,
            passthrough_blocks: Vec::new(),
            metadata: Default::default(),
        };

        shift_document_timestamps(&mut document, 10_000);

        assert_eq!(document.segments[0].start.as_deref(), Some("00:00:11,000"));
        assert_eq!(document.segments[0].end.as_deref(), Some("00:00:12,500"));
        assert_eq!(last_timed_end_ms_on_audio(&document, 10_000), Some(2_500));
    }

    #[cfg(unix)]
    #[test]
    fn failed_audio_preparation_removes_its_unique_temp_directory() {
        use std::os::unix::fs::PermissionsExt;

        let root = t("prepare-audio-failure");
        fs::create_dir_all(&root).expect("create root");
        let input = root.join("input.flac");
        fs::write(&input, b"compressed audio").expect("write input");
        let ffprobe = root.join("ffprobe");
        let ffmpeg = root.join("ffmpeg");
        fs::write(
            &ffprobe,
            "#!/bin/sh\nprintf '{\"streams\":[{\"codec_name\":\"flac\"}],\"format\":{\"duration\":\"10.0\"}}\\n'\n",
        )
        .expect("write ffprobe");
        fs::write(
            &ffmpeg,
            "#!/bin/sh\ncase \" $* \" in *\" -f null - \"*) exit 0;; esac\necho conversion-failed >&2\nexit 2\n",
        )
        .expect("write ffmpeg");
        fs::set_permissions(&ffprobe, fs::Permissions::from_mode(0o755)).expect("chmod ffprobe");
        fs::set_permissions(&ffmpeg, fs::Permissions::from_mode(0o755)).expect("chmod ffmpeg");
        let runtime_dir = root.join("runtime");
        let settings = TranscriptionSettings {
            runtime_dir: Some(runtime_dir.clone()),
            ..TranscriptionSettings::default()
        };
        let progress: SharedProgress = std::sync::Arc::new(NoopProgress);

        let error = prepare_audio_with_programs(
            &input,
            &settings,
            &CancellationGuard::never(),
            &progress,
            &ffmpeg,
            &ffprobe,
        )
        .expect_err("preparation should fail");
        let temp_root = runtime_dir.join("tmp/transcription");
        let remaining = fs::read_dir(&temp_root).expect("read temp root").count();
        let _ = fs::remove_dir_all(root);

        assert!(error.to_string().contains("conversion-failed"));
        assert_eq!(remaining, 0);
    }

    #[cfg(unix)]
    #[test]
    fn media_without_an_audio_stream_fails_before_ffmpeg_or_temp_creation() {
        use std::os::unix::fs::PermissionsExt;

        let root = t("prepare-audio-no-stream");
        fs::create_dir_all(&root).expect("create root");
        let input = root.join("input.mp4");
        fs::write(&input, b"video only").expect("write input");
        let ffprobe = root.join("ffprobe");
        let ffmpeg = root.join("ffmpeg");
        let marker = root.join("ffmpeg-was-called");
        fs::write(
            &ffprobe,
            "#!/bin/sh\nprintf '{\"streams\":[],\"format\":{\"duration\":\"10.0\"}}\\n'\n",
        )
        .expect("write ffprobe");
        fs::write(
            &ffmpeg,
            format!("#!/bin/sh\nprintf called > '{}'\n", marker.display()),
        )
        .expect("write ffmpeg");
        fs::set_permissions(&ffprobe, fs::Permissions::from_mode(0o755)).expect("chmod ffprobe");
        fs::set_permissions(&ffmpeg, fs::Permissions::from_mode(0o755)).expect("chmod ffmpeg");
        let runtime_dir = root.join("runtime");
        let settings = TranscriptionSettings {
            runtime_dir: Some(runtime_dir.clone()),
            ..TranscriptionSettings::default()
        };
        let progress: SharedProgress = std::sync::Arc::new(NoopProgress);

        let error = prepare_audio_with_programs(
            &input,
            &settings,
            &CancellationGuard::never(),
            &progress,
            &ffmpeg,
            &ffprobe,
        )
        .expect_err("media without audio must fail");
        let message = error.to_string();

        assert!(message.contains("contains no audio stream"), "{message}");
        assert!(!marker.exists());
        assert!(!runtime_dir.join("tmp/transcription").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn unsupported_audio_decoder_reports_codec_before_temp_creation() {
        use std::os::unix::fs::PermissionsExt;

        let root = t("prepare-audio-no-decoder");
        fs::create_dir_all(&root).expect("create root");
        let input = root.join("input.mp4");
        fs::write(&input, b"eac3 audio").expect("write input");
        let ffprobe = root.join("ffprobe");
        let ffmpeg = root.join("ffmpeg");
        fs::write(
            &ffprobe,
            "#!/bin/sh\nprintf '{\"streams\":[{\"codec_name\":\"eac3\"}],\"format\":{\"duration\":\"6533.216\"}}\\n'\n",
        )
        .expect("write ffprobe");
        fs::write(
            &ffmpeg,
            "#!/bin/sh\necho 'Decoding requested, but no decoder found for: eac3' >&2\nexit 234\n",
        )
        .expect("write ffmpeg");
        fs::set_permissions(&ffprobe, fs::Permissions::from_mode(0o755)).expect("chmod ffprobe");
        fs::set_permissions(&ffmpeg, fs::Permissions::from_mode(0o755)).expect("chmod ffmpeg");
        let runtime_dir = root.join("runtime");
        let settings = TranscriptionSettings {
            runtime_dir: Some(runtime_dir.clone()),
            ..TranscriptionSettings::default()
        };
        let progress: SharedProgress = std::sync::Arc::new(NoopProgress);

        let error = prepare_audio_with_programs(
            &input,
            &settings,
            &CancellationGuard::never(),
            &progress,
            &ffmpeg,
            &ffprobe,
        )
        .expect_err("unsupported decoder must fail");
        let message = error.to_string();

        assert!(message.contains("audio stream uses `eac3`"));
        assert!(message.contains("cannot decode it"));
        assert!(!runtime_dir.join("tmp/transcription").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn whisper_cpp_fake_cli_uses_supported_arguments_and_drains_output() {
        use std::os::unix::fs::PermissionsExt;
        use std::sync::{Arc, Mutex};

        #[derive(Default)]
        struct Recorder(Mutex<Vec<ProgressEvent>>);
        impl subbake_core::ProgressSink for Recorder {
            fn emit(&self, event: ProgressEvent) {
                self.0.lock().expect("progress lock").push(event);
            }
        }

        let root = t("fake-whisper-cli");
        fs::create_dir_all(&root).expect("create root");
        let binary = root.join("whisper-cli");
        let script = r#"#!/bin/sh
if [ "$1" = "--version" ]; then echo "whisper.cpp version: fake-1"; exit 0; fi
if [ "$1" = "--help" ]; then
  echo "--model --file --output-file --output-srt --output-vtt --threads --print-progress --no-prints --max-context --vad --vad-model" >&2
  exit 0
fi
output=""
format=""
threads=""
print_progress=0
vad=0
vad_model=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -os) echo "unexpected legacy -os" >&2; exit 17 ;;
    --output-file) shift; output="$1" ;;
    --output-srt) format="srt" ;;
    --output-vtt) format="vtt" ;;
    --threads) shift; threads="$1" ;;
    --print-progress) print_progress=1 ;;
    --vad) vad=1 ;;
    --vad-model) shift; vad_model="$1" ;;
  esac
  shift
done
if [ -z "$threads" ]; then echo "missing --threads" >&2; exit 18; fi
if [ "$print_progress" -ne 1 ]; then echo "missing --print-progress" >&2; exit 19; fi
if [ "$vad" -ne 1 ]; then echo "missing --vad" >&2; exit 20; fi
if [ ! -f "$vad_model" ]; then echo "missing --vad-model" >&2; exit 21; fi
i=0
while [ "$i" -lt 20000 ]; do echo "diagnostic-$i" >&2; i=$((i + 1)); done
echo "whisper_print_progress_callback: progress =  25%" >&2
echo "whisper_print_progress_callback: progress = 100%" >&2
if [ "$format" = "srt" ]; then
  printf '1\n00:00:00,000 --> 00:00:01,000\nhello\n' > "${output}.srt"
else
  printf 'WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n' > "${output}.vtt"
fi
"#;
        fs::write(&binary, script).expect("write fake CLI");
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).expect("chmod fake CLI");
        let audio = root.join("audio.wav");
        let model = root.join("ggml-fake.bin");
        let vad_model = root.join("ggml-silero-v6.2.0.bin");
        fs::write(&audio, b"fake audio").expect("write audio");
        fs::write(&model, b"fake model").expect("write model");
        fs::write(&vad_model, b"fake VAD model").expect("write VAD model");

        let output = root.join("result.srt");
        let recorder = Arc::new(Recorder::default());
        let progress: SharedProgress = recorder.clone();
        let outcome = transcribe_media_cancellable_with_progress(
            TranscriptionRequest {
                media_path: audio,
                output_path: Some(output.clone()),
                overwrite: true,
                settings: TranscriptionSettings {
                    language: Some("en".to_owned()),
                    model: Some("fake".to_owned()),
                    whisper_binary_path: Some(binary),
                    whisper_models_dir: Some(root.clone()),
                    ..TranscriptionSettings::default()
                },
            },
            &CancellationGuard::never(),
            progress,
        )
        .expect("fake CLI transcription");
        let content = fs::read_to_string(&output).expect("read rendered output");

        assert_eq!(outcome.subtitle_entries, 1);
        assert!(content.contains("hello"));
        assert!(
            recorder
                .0
                .lock()
                .expect("progress lock")
                .iter()
                .any(|event| event.stage == "TRANSCRIBE"
                    && event.current == 25
                    && event.total == Some(100)
                    && event.unit == ProgressUnit::Percent)
        );
        let _ = fs::remove_dir_all(&root);
    }

    fn t(l: &str) -> PathBuf {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-transcription-{l}-{n}"))
    }

    fn timed_segment(id: &str, text: &str, start: &str, end: &str) -> SubtitleSegment {
        SubtitleSegment {
            id: id.to_owned(),
            text: text.to_owned(),
            start: Some(start.to_owned()),
            end: Some(end.to_owned()),
            identifier: Some(id.to_owned()),
            settings: None,
            semantic: Default::default(),
        }
    }

    fn model_dir(label: &str, models: &[&str]) -> PathBuf {
        let root = t(label);
        fs::create_dir_all(&root).expect("create model directory");
        for model in models {
            fs::write(root.join(format!("ggml-{model}.bin")), b"model").expect("write model");
        }
        root
    }
}
