// Media transcription backends: Whisper API (HTTP multipart) and whisper.cpp (subprocess).
//
// Orchestration: ffmpeg audio extraction (video-only) → backend transcribe → render.

use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use reqwest::multipart;
use subbake_core::SubtitleDocument;
use subbake_core::entities::SubtitleSegment;
use subbake_core::formats::RenderOptions;
use tokio::runtime::Runtime;

use crate::fs::{read_document, render_and_write_document};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionRequest {
    pub media_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub settings: TranscriptionSettings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionSettings {
    pub provider: String,
    pub language: Option<String>,
    pub model: Option<String>,
    pub output_format: TranscriptionFormat,
    pub sidecar_path: Option<PathBuf>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionOutcome {
    pub output_path: PathBuf,
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

    fn response_format_arg(self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Txt => "json",
        }
    }
}

impl Default for TranscriptionSettings {
    fn default() -> Self {
        Self {
            provider: "whisper_api".to_owned(),
            language: None,
            model: None,
            output_format: TranscriptionFormat::Srt,
            sidecar_path: None,
            api_key: None,
            base_url: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared tokio runtime for async reqwest calls
// ---------------------------------------------------------------------------

fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME
        .get_or_init(|| Runtime::new().unwrap_or_else(|_| panic!("unable to start subbake transcription runtime")))
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
    ) -> io::Result<SubtitleDocument>;
}

// ---------------------------------------------------------------------------
// Whisper API backend (HTTP POST multipart)
// ---------------------------------------------------------------------------

pub struct WhisperApiTranscriber {
    api_key: String,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl WhisperApiTranscriber {
    pub fn new(api_key: String, base_url: String, model: String, timeout_seconds: f64) -> io::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs_f64(timeout_seconds.max(1.0)))
            .build()
            .map_err(|e| io::Error::other(format!("http client build: {e}")))?;
        Ok(Self { api_key, base_url, model, client })
    }
}

impl TranscriberBackend for WhisperApiTranscriber {
    fn transcribe(
        &self,
        audio_path: &Path,
        language: Option<&str>,
        output_format: TranscriptionFormat,
    ) -> io::Result<SubtitleDocument> {
        runtime().block_on(async {
            let url = format!("{}/audio/transcriptions", self.base_url.trim_end_matches('/'));
            let audio_bytes = std::fs::read(audio_path)
                .map_err(|e| io::Error::other(format!("read audio file: {e}")))?;
            let file_name = audio_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("audio.wav")
                .to_owned();

            let mut form = multipart::Form::new()
                .part("file", multipart::Part::bytes(audio_bytes).file_name(file_name))
                .text("model", self.model.clone())
                .text("response_format", output_format.response_format_arg().to_owned());
            if let Some(lang) = language {
                form = form.text("language", lang.to_owned());
            }

            let resp = self
                .client
                .post(&url)
                .bearer_auth(&self.api_key)
                .multipart(form)
                .send()
                .await
                .map_err(|e| io::Error::other(format!("whisper api request: {e}")))?;

            let status = resp.status();
            let body = resp.text().await
                .map_err(|e| io::Error::other(format!("whisper api response: {e}")))?;
            if !status.is_success() {
                return Err(io::Error::other(format!("whisper api rejected ({status}): {body}")));
            }
            parse_whisper_response(&body, output_format)
        })
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
        Self { binary, model_path, extra_args }
    }
}

impl TranscriberBackend for WhisperCppTranscriber {
    fn transcribe(
        &self,
        audio_path: &Path,
        language: Option<&str>,
        output_format: TranscriptionFormat,
    ) -> io::Result<SubtitleDocument> {
        let output_dir = audio_path.parent().unwrap_or_else(|| Path::new("."));
        let base_name = audio_path.file_stem().and_then(|s| s.to_str()).unwrap_or("audio");
        let output_base = output_dir.join(base_name);

        let mut cmd = Command::new(&self.binary);
        cmd.args([
            "-m", &self.model_path.to_string_lossy(),
            "-f", &audio_path.to_string_lossy(),
            "-os", "--output-file", &output_base.to_string_lossy(),
        ]);
        match output_format {
            TranscriptionFormat::Srt | TranscriptionFormat::Txt => { cmd.arg("--output-srt"); }
            TranscriptionFormat::Vtt => { cmd.arg("--output-vtt"); }
        }
        if let Some(lang) = language { cmd.args(["-l", lang]); }
        for arg in &self.extra_args { cmd.arg(arg); }

        let out = cmd.output()
            .map_err(|e| io::Error::other(format!("whisper.cpp execution: {e}")))?;
        if !out.status.success() {
            return Err(io::Error::other(format!(
                "whisper.cpp exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr),
            )));
        }

        let suffix = match output_format { TranscriptionFormat::Vtt => "vtt", _ => "srt" };
        let generated = output_base.with_extension(suffix);
        let mut doc = read_document(&generated)?;
        let _ = std::fs::remove_file(&generated);

        if matches!(output_format, TranscriptionFormat::Txt) {
            doc.segments = doc.segments.iter().map(|s| SubtitleSegment {
                start: None, end: None, identifier: None, settings: None, ..s.clone()
            }).collect();
            doc.format = "txt".to_owned();
        }
        Ok(doc)
    }
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

pub fn transcribe_media(request: TranscriptionRequest) -> io::Result<TranscriptionOutcome> {
    let output_path = request.output_path.unwrap_or_else(|| {
        default_output_path(&request.media_path, request.settings.output_format)
    });

    if let Some(ref sidecar_path) = request.settings.sidecar_path {
        render_sidecar(sidecar_path, &output_path, request.settings.output_format)?;
        return Ok(TranscriptionOutcome { output_path });
    }

    let audio_path = ensure_audio(&request.media_path)?;
    let fmt = request.settings.output_format;

    let doc = match request.settings.provider.as_str() {
        "whisper_api" => {
            let api_key = request.settings.api_key.clone()
                .or_else(default_whisper_api_key)
                .ok_or_else(|| io::Error::other(
                    "Missing API key for Whisper API. Set OPENAI_API_KEY or use --api-key.",
                ))?;
            let base_url = request.settings.base_url.clone()
                .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
                .unwrap_or_else(|| "https://api.openai.com/v1".to_owned());
            let model = request.settings.model.clone().unwrap_or_else(|| "whisper-1".to_owned());
            let t = WhisperApiTranscriber::new(api_key, base_url, model, 120.0)?;
            t.transcribe(&audio_path, request.settings.language.as_deref(), fmt)?
        }
        "whisper_cpp" => {
            let model = request.settings.model.clone().unwrap_or_else(|| "small".to_owned());
            let binary = locate_whisper_binary()?;
            let model_path = locate_whisper_model(&model)?;
            let t = WhisperCppTranscriber::new(binary, model_path, Vec::new());
            t.transcribe(&audio_path, request.settings.language.as_deref(), fmt)?
        }
        other => return Err(io::Error::other(format!("unsupported transcriber `{other}"))),
    };

    let opts = RenderOptions::new(false, Some(fmt.extension().to_owned()));
    render_and_write_document(&doc, &doc.segments, &output_path, &opts)?;
    Ok(TranscriptionOutcome { output_path })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ensure_audio(media_path: &Path) -> io::Result<PathBuf> {
    if is_audio_ext(media_path) { return Ok(media_path.to_path_buf()); }

    let parent = media_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = media_path.file_stem().and_then(|s| s.to_str()).unwrap_or("audio");
    let tmp_dir = parent.join(".subbake").join("tmp");
    std::fs::create_dir_all(&tmp_dir)
        .map_err(|e| io::Error::other(format!("create tmp dir: {e}")))?;
    let output = tmp_dir.join(format!("{stem}_audio.wav"));

    let status = Command::new("ffmpeg")
        .args(["-y", "-i", &media_path.to_string_lossy(), "-vn",
               "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
               &output.to_string_lossy()])
        .status()
        .map_err(|e| io::Error::other(format!("ffmpeg not found: {e}")))?;
    if !status.success() {
        return Err(io::Error::other("ffmpeg audio extraction failed"));
    }
    Ok(output)
}

fn is_audio_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| matches!(ext, "wav" | "mp3" | "ogg" | "m4a" | "flac"))
}

fn default_whisper_api_key() -> Option<String> {
    std::env::var("OPENAI_API_KEY").ok().filter(|v| !v.trim().is_empty())
}

fn locate_whisper_binary() -> io::Result<PathBuf> {
    let p = PathBuf::from(".subbake/whisper/bin/whisper-cli");
    if p.exists() { Ok(p) } else {
        Err(io::Error::other("whisper.cpp binary not found. Run `sbake whisper install` first."))
    }
}

fn locate_whisper_model(name: &str) -> io::Result<PathBuf> {
    let p = PathBuf::from(".subbake/whisper/models").join(format!("ggml-{name}.bin"));
    if p.exists() { Ok(p) } else {
        Err(io::Error::other(format!(
            "model `{name}` not found at {}. Run `sbake whisper models --download {name}`.",
            p.display()
        )))
    }
}

fn default_output_path(media_path: &Path, fmt: TranscriptionFormat) -> PathBuf {
    media_path.with_extension(fmt.extension())
}

fn render_sidecar(path: &Path, output: &Path, fmt: TranscriptionFormat) -> io::Result<()> {
    let doc = read_document(path)?;
    if fmt != TranscriptionFormat::Txt
        && doc.segments.iter().any(|s| s.start.is_none() || s.end.is_none())
    {
        return Err(io::Error::other(
            "sidecar lacks timing data; use --format txt or a timed subtitle file",
        ));
    }
    let opts = RenderOptions::new(false, Some(fmt.extension().to_owned()));
    render_and_write_document(&doc, &doc.segments, output, &opts)?;
    Ok(())
}

fn parse_whisper_response(body: &str, fmt: TranscriptionFormat) -> io::Result<SubtitleDocument> {
    match fmt {
        TranscriptionFormat::Srt | TranscriptionFormat::Vtt => {
            let ext = fmt.extension();
            let dir = std::env::temp_dir().join("subbake-whisper");
            std::fs::create_dir_all(&dir)
                .map_err(|e| io::Error::other(format!("create tmp dir: {e}")))?;
            let tmp = dir.join(format!("response.{ext}"));
            std::fs::write(&tmp, body)
                .map_err(|e| io::Error::other(format!("write tmp: {e}")))?;
            let doc = read_document(&tmp)?;
            let _ = std::fs::remove_file(&tmp);
            Ok(doc)
        }
        TranscriptionFormat::Txt => {
            let parsed: serde_json::Value = serde_json::from_str(body)
                .map_err(|e| io::Error::other(format!("whisper json parse: {e}")))?;
            let text = parsed["segments"]
                .as_array()
                .map(|segs| segs.iter()
                    .filter_map(|s| s["text"].as_str())
                    .collect::<Vec<_>>().join("\n"))
                .or_else(|| parsed["text"].as_str().map(String::from))
                .unwrap_or_default();
            Ok(SubtitleDocument {
                path: PathBuf::new(), format: "txt".to_owned(),
                segments: vec![SubtitleSegment {
                    id: "1".to_owned(), text,
                    start: None, end: None, identifier: None, settings: None,
                }],
                header: None, passthrough_blocks: Vec::new(),
            })
        }
    }
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
            settings: TranscriptionSettings {
                sidecar_path: Some(src), ..Default::default()
            },
        }).expect("transcribe");
        assert_eq!(r.output_path, out);
        assert!(fs::read_to_string(&out).expect("read out").contains("hello"));
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
            settings: TranscriptionSettings {
                sidecar_path: Some(src), ..Default::default()
            },
        }).expect_err("untimed should error");
        let _ = fs::remove_dir_all(&root);
        assert!(e.to_string().contains("lacks timing"));
    }

    #[test]
    fn whisper_api_without_key_fails() {
        let e = transcribe_media(TranscriptionRequest {
            media_path: PathBuf::from("m.wav"),
            output_path: None,
            settings: TranscriptionSettings {
                provider: "whisper_api".to_owned(), api_key: None, ..Default::default()
            },
        }).expect_err("no key should error");
        assert!(e.to_string().contains("API key"));
    }

    #[test]
    fn unknown_provider_rejected() {
        let e = transcribe_media(TranscriptionRequest {
            media_path: PathBuf::from("m.wav"),
            output_path: None,
            settings: TranscriptionSettings {
                provider: "nope".to_owned(), ..Default::default()
            },
        }).expect_err("unknown provider should error");
        assert!(e.to_string().contains("unsupported transcriber"));
    }

    #[test]
    fn audio_extension_check() {
        assert!(is_audio_ext(Path::new("x.wav")));
        assert!(!is_audio_ext(Path::new("x.mp4")));
    }

    fn t(l: &str) -> PathBuf {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-transcription-{l}-{n}"))
    }
}
