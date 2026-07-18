// Media transcription through the local whisper.cpp sidecar.
//
// Orchestration: ffmpeg audio extraction (video-only) → backend transcribe → render.

use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use subbake_core::entities::SubtitleSegment;
use subbake_core::formats::RenderOptions;
use subbake_core::languages::normalize_language;
use subbake_core::{
    CancellationGuard, NoopProgress, ProgressEvent, ProgressUnit, SharedProgress, SubtitleDocument,
    TaskKind, TaskState,
};

use crate::error::{AdapterError, AdapterResult};
use crate::fs::{read_document, render_and_write_document};
use crate::process::run_command_cancellable;
use crate::settings::StorageSettings;
use crate::whisper::{
    default_whisper_binary_path_for, default_whisper_models_dir_for, verify_whisper_cli,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionOutcome {
    pub output_path: PathBuf,
    pub language: String,
    pub provider: String,
    pub model: String,
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
}

impl WhisperCppTranscriber {
    pub fn new(binary: PathBuf, model_path: PathBuf, extra_args: Vec<String>) -> Self {
        Self {
            binary,
            model_path,
            extra_args,
        }
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

        let mut cmd = Command::new(&self.binary);
        cmd.args([
            "-m",
            &self.model_path.to_string_lossy(),
            "-f",
            &audio_path.to_string_lossy(),
            "--output-file",
            &output_base.to_string_lossy(),
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
        cmd.arg("--no-prints");

        let out = run_command_cancellable(&mut cmd, cancellation, "whisper.cpp execution")?;
        if !out.status.success() {
            return Err(AdapterError::ChildProcess {
                program: "whisper.cpp",
                status: out.status.code(),
                message: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
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
    progress.emit(ProgressEvent::running(
        TaskKind::Transcription,
        "PREPARE_AUDIO",
        0,
        None,
        ProgressUnit::Steps,
    ));
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
            output_format: request.settings.output_format,
            subtitle_entries: document.segments.len(),
        });
    }

    let audio_path = ensure_audio(&request.media_path, cancellation)?;
    progress.emit(ProgressEvent::running(
        TaskKind::Transcription,
        "TRANSCRIBE",
        0,
        None,
        ProgressUnit::Steps,
    ));
    let fmt = request.settings.output_format;

    let effective_model = request
        .settings
        .model
        .clone()
        .unwrap_or_else(|| "small".to_owned());
    let binary = locate_whisper_binary(&request.settings)?;
    verify_whisper_cli(&binary, cancellation)?;
    let model_path = locate_whisper_model(&request.settings, &effective_model)?;
    let transcriber = WhisperCppTranscriber::new(binary, model_path, Vec::new());
    let doc = transcriber.transcribe_cancellable(
        &audio_path,
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
        output_format: fmt,
        subtitle_entries: doc.segments.len(),
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ensure_audio(media_path: &Path, cancellation: &CancellationGuard) -> AdapterResult<PathBuf> {
    check_cancelled(cancellation)?;
    if is_audio_ext(media_path) {
        return Ok(media_path.to_path_buf());
    }

    let parent = media_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = media_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("audio");
    let tmp_dir = parent.join(".subbake").join("tmp");
    std::fs::create_dir_all(&tmp_dir)
        .map_err(|e| io::Error::other(format!("create tmp dir: {e}")))?;
    let output = tmp_dir.join(format!("{stem}_audio.wav"));

    let mut command = Command::new("ffmpeg");
    command.args([
        "-y",
        "-i",
        &media_path.to_string_lossy(),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        &output.to_string_lossy(),
    ]);
    let result = run_command_cancellable(&mut command, cancellation, "ffmpeg execution");
    if result.is_err() {
        let _ = std::fs::remove_file(&output);
    }
    let out = result?;
    if !out.status.success() {
        let _ = std::fs::remove_file(&output);
        return Err(AdapterError::ChildProcess {
            program: "ffmpeg",
            status: out.status.code(),
            message: String::from_utf8_lossy(&out.stderr).into_owned(),
        });
    }
    check_cancelled(cancellation).inspect_err(|_| {
        let _ = std::fs::remove_file(&output);
    })?;
    Ok(output)
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

fn is_audio_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| matches!(ext, "wav" | "mp3" | "ogg" | "m4a" | "flac"))
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

fn locate_whisper_model(settings: &TranscriptionSettings, name: &str) -> AdapterResult<PathBuf> {
    let p = settings
        .whisper_models_dir
        .clone()
        .unwrap_or_else(|| default_whisper_models_dir_for(None))
        .join(format!("ggml-{name}.bin"));
    if p.exists() {
        Ok(p)
    } else {
        Err(AdapterError::invalid_input(format!(
            "model `{name}` not found at {}. Run `sbake whisper model {name}`.",
            p.display()
        )))
    }
}

pub fn apply_whisper_storage(transcription: &mut TranscriptionSettings, storage: &StorageSettings) {
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
    fn audio_extension_check() {
        assert!(is_audio_ext(Path::new("x.wav")));
        assert!(!is_audio_ext(Path::new("x.mp4")));
    }

    #[cfg(unix)]
    #[test]
    fn whisper_cpp_fake_cli_uses_supported_arguments_and_drains_output() {
        use std::os::unix::fs::PermissionsExt;

        let root = t("fake-whisper-cli");
        fs::create_dir_all(&root).expect("create root");
        let binary = root.join("whisper-cli");
        let script = r#"#!/bin/sh
if [ "$1" = "--version" ]; then echo "whisper.cpp version: fake-1"; exit 0; fi
if [ "$1" = "--help" ]; then
  echo "--model --file --output-file --output-srt --output-vtt" >&2
  exit 0
fi
output=""
format=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -os) echo "unexpected legacy -os" >&2; exit 17 ;;
    --output-file) shift; output="$1" ;;
    --output-srt) format="srt" ;;
    --output-vtt) format="vtt" ;;
  esac
  shift
done
i=0
while [ "$i" -lt 20000 ]; do echo "diagnostic-$i" >&2; i=$((i + 1)); done
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
        let outcome = transcribe_media(TranscriptionRequest {
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
        })
        .expect("fake CLI transcription");
        let content = fs::read_to_string(&output).expect("read rendered output");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(outcome.subtitle_entries, 1);
        assert!(content.contains("hello"));
    }

    fn t(l: &str) -> PathBuf {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-transcription-{l}-{n}"))
    }
}
