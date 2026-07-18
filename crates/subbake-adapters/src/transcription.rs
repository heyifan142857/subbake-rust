// Media transcription through the local whisper.cpp sidecar.
//
// Orchestration: ffmpeg audio extraction (video-only) → backend transcribe → render.

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

use crate::error::{AdapterError, AdapterResult};
use crate::fs::{read_document, render_and_write_document};
use crate::process::{
    run_command_cancellable, run_command_cancellable_with_stderr_lines,
    run_command_cancellable_with_stdout_lines,
};
use crate::settings::{ResolvedSettings, StorageSettings};
use crate::whisper::{
    default_whisper_binary_path_for, default_whisper_models_dir_for, installed_models_in,
    verify_whisper_cli,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionRequest {
    pub media_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub overwrite: bool,
    pub settings: TranscriptionSettings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionSettings {
    pub language: Option<String>,
    pub model: Option<String>,
    pub output_format: TranscriptionFormat,
    pub sidecar_path: Option<PathBuf>,
    pub whisper_binary_path: Option<PathBuf>,
    pub whisper_models_dir: Option<PathBuf>,
    pub runtime_dir: Option<PathBuf>,
    pub multiple_model_policy: MultipleModelPolicy,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptionFormat {
    Srt,
    Vtt,
    Txt,
}

impl TranscriptionFormat {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "srt" => Some(Self::Srt),
            "vtt" => Some(Self::Vtt),
            "txt" => Some(Self::Txt),
            _ => None,
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Txt => "txt",
        }
    }
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
        }
    }
}

// ---------------------------------------------------------------------------
// Transcriber backend trait
// ---------------------------------------------------------------------------

pub trait TranscriberBackend {
    fn transcribe(
        &self,
        audio_path: &Path,
        language: Option<&str>,
        output_format: TranscriptionFormat,
    ) -> AdapterResult<SubtitleDocument>;
    fn transcribe_cancellable(
        &self,
        audio_path: &Path,
        language: Option<&str>,
        output_format: TranscriptionFormat,
        cancellation: &CancellationGuard,
    ) -> AdapterResult<SubtitleDocument> {
        check_cancelled(cancellation)?;
        self.transcribe(audio_path, language, output_format)
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
}

impl WhisperCppTranscriber {
    pub fn new(binary: PathBuf, model_path: PathBuf, extra_args: Vec<String>) -> Self {
        Self {
            binary,
            model_path,
            extra_args,
            progress: std::sync::Arc::new(NoopProgress),
            threads: default_whisper_threads(),
        }
    }

    fn with_progress(mut self, progress: SharedProgress) -> Self {
        self.progress = progress;
        self
    }
}

impl TranscriberBackend for WhisperCppTranscriber {
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
        let out = run_command_cancellable_with_stderr_lines(
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
                    current,
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
                100,
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
    let transcriber =
        WhisperCppTranscriber::new(binary, model_path, Vec::new()).with_progress(progress.clone());
    let doc = transcriber.transcribe_cancellable(
        prepared_audio.path(),
        request.settings.language.as_deref(),
        fmt,
        cancellation,
    )?;

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
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct PreparedAudio {
    path: PathBuf,
    _temporary_dir: Option<AudioTempDirectory>,
}

impl PreparedAudio {
    fn borrowed(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            _temporary_dir: None,
        }
    }

    fn temporary(path: PathBuf, directory: AudioTempDirectory) -> Self {
        Self {
            path,
            _temporary_dir: Some(directory),
        }
    }

    fn path(&self) -> &Path {
        &self.path
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
        let mut done = ProgressEvent::running(
            TaskKind::Transcription,
            "PREPARE_AUDIO",
            1,
            Some(1),
            ProgressUnit::Steps,
        );
        done.state = TaskState::Completed;
        progress.emit(done);
        return Ok(PreparedAudio::borrowed(media_path));
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
    let duration_ms = audio_info.duration_ms;
    progress.emit(ProgressEvent::running(
        TaskKind::Transcription,
        "PREPARE_AUDIO",
        0,
        duration_ms,
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
    let out = run_command_cancellable_with_stdout_lines(
        &mut command,
        cancellation,
        "ffmpeg audio preparation",
        |line| {
            if let Some(current) = parse_ffmpeg_progress_ms(line) {
                processed_ms = current;
                progress.emit(ProgressEvent::running(
                    TaskKind::Transcription,
                    "PREPARE_AUDIO",
                    duration_ms.map_or(current, |total| current.min(total)),
                    duration_ms,
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
        duration_ms.unwrap_or(processed_ms),
        duration_ms,
        ProgressUnit::Duration,
    );
    done.state = TaskState::Completed;
    progress.emit(done);
    Ok(PreparedAudio::temporary(output, temp_dir))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MediaAudioInfo {
    codec: String,
    duration_ms: Option<u64>,
}

fn probe_audio_info(
    ffprobe: &Path,
    media_path: &Path,
    cancellation: &CancellationGuard,
) -> AdapterResult<MediaAudioInfo> {
    let output = run_command_cancellable(
        Command::new(ffprobe).args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name:format=duration",
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
    Ok(MediaAudioInfo { codec, duration_ms })
}

fn validate_audio_decodable(
    ffmpeg: &Path,
    media_path: &Path,
    codec: &str,
    cancellation: &CancellationGuard,
) -> AdapterResult<()> {
    let output = run_command_cancellable(
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

fn locate_whisper_binary(settings: &TranscriptionSettings) -> AdapterResult<PathBuf> {
    let p = settings
        .whisper_binary_path
        .clone()
        .unwrap_or_else(|| default_whisper_binary_path_for(None));
    if p.exists() {
        Ok(p)
    } else {
        Err(AdapterError::invalid_input(
            "whisper.cpp binary not found. Run `sbake whisper install` first.",
        ))
    }
}

#[derive(Debug)]
struct ResolvedWhisperModel {
    name: String,
    path: PathBuf,
    auto_selected: bool,
}

fn resolve_whisper_model(settings: &TranscriptionSettings) -> AdapterResult<ResolvedWhisperModel> {
    let models_dir = settings
        .whisper_models_dir
        .clone()
        .unwrap_or_else(|| default_whisper_models_dir_for(None));
    let mut installed = if models_dir.is_dir() {
        installed_models_in(&models_dir).map_err(|source| {
            AdapterError::external_io("list installed whisper models", Some(models_dir), source)
        })?
    } else {
        Vec::new()
    };
    installed.sort_by(|left, right| model_rank(&left.name).cmp(&model_rank(&right.name)));
    let installed_names = installed
        .iter()
        .map(|model| model.name.clone())
        .collect::<Vec<_>>();

    if let Some(requested) = settings.model.as_deref() {
        return installed
            .into_iter()
            .find(|model| model.name == requested)
            .map(|model| ResolvedWhisperModel {
                name: model.name,
                path: model.path,
                auto_selected: false,
            })
            .ok_or_else(|| {
                AdapterError::invalid_input(format!(
                    "model `{requested}` is not installed; installed models: {}. Run `sbake whisper model {requested}`.",
                    display_model_names(&installed_names)
                ))
            });
    }

    let selected = match installed.as_slice() {
        [] => {
            return Err(AdapterError::invalid_input(
                "no whisper.cpp models are installed. Run `sbake whisper model list`, then `sbake whisper model <NAME>`.",
            ));
        }
        [only] => only,
        many => {
            if let Some(small) = many.iter().find(|model| model.name == "small") {
                small
            } else if settings.multiple_model_policy == MultipleModelPolicy::PreferRanked {
                &many[0]
            } else {
                return Err(AdapterError::invalid_input(format!(
                    "multiple whisper.cpp models are installed; specify `--model <NAME>`. Available: {}",
                    display_model_names(&installed_names)
                )));
            }
        }
    };
    Ok(ResolvedWhisperModel {
        name: selected.name.clone(),
        path: selected.path.clone(),
        auto_selected: true,
    })
}

fn display_model_names(names: &[String]) -> String {
    if names.is_empty() {
        "none".to_owned()
    } else {
        names.join(", ")
    }
}

fn model_rank(name: &str) -> (usize, usize, usize, &str) {
    const FAMILIES: &[&str] = &[
        "small",
        "base",
        "medium",
        "large-v3-turbo",
        "large-v3",
        "large-v2",
        "large-v1",
        "tiny",
    ];
    let family = FAMILIES
        .iter()
        .position(|family| {
            name == *family
                || name
                    .strip_prefix(*family)
                    .is_some_and(|suffix| suffix.starts_with('-') || suffix.starts_with('.'))
        })
        .unwrap_or(FAMILIES.len());
    let english_only = usize::from(name.contains(".en"));
    let quantization = if name.contains("q8_") {
        1
    } else if name.contains("q5_") {
        2
    } else {
        0
    };
    (family, english_only, quantization, name)
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
            "#!/bin/sh\nprintf '{\"streams\":[{\"codec_name\":\"mp3\"}],\"format\":{\"duration\":\"10.0\"}}\\n'\n",
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

        assert!(message.contains("contains no audio stream"));
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
  echo "--model --file --output-file --output-srt --output-vtt --threads --print-progress --no-prints" >&2
  exit 0
fi
output=""
format=""
threads=""
print_progress=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    -os) echo "unexpected legacy -os" >&2; exit 17 ;;
    --output-file) shift; output="$1" ;;
    --output-srt) format="srt" ;;
    --output-vtt) format="vtt" ;;
    --threads) shift; threads="$1" ;;
    --print-progress) print_progress=1 ;;
  esac
  shift
done
if [ -z "$threads" ]; then echo "missing --threads" >&2; exit 18; fi
if [ "$print_progress" -ne 1 ]; then echo "missing --print-progress" >&2; exit 19; fi
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
        fs::write(&audio, b"fake audio").expect("write audio");
        fs::write(&model, b"fake model").expect("write model");

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

    fn model_dir(label: &str, models: &[&str]) -> PathBuf {
        let root = t(label);
        fs::create_dir_all(&root).expect("create model directory");
        for model in models {
            fs::write(root.join(format!("ggml-{model}.bin")), b"model").expect("write model");
        }
        root
    }
}
