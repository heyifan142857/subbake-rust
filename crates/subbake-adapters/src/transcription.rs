use std::io;
use std::path::{Path, PathBuf};

use subbake_core::formats::RenderOptions;

use crate::fs::{read_document, render_and_write_document};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionRequest {
    pub media_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub settings: TranscriptionSettings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionSettings {
    pub language: Option<String>,
    pub model: Option<String>,
    pub output_format: TranscriptionFormat,
    pub sidecar_path: Option<PathBuf>,
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
}

impl Default for TranscriptionSettings {
    fn default() -> Self {
        Self {
            language: None,
            model: None,
            output_format: TranscriptionFormat::Srt,
            sidecar_path: None,
        }
    }
}

pub fn transcribe_media(request: TranscriptionRequest) -> io::Result<TranscriptionOutcome> {
    let output_path = request.output_path.unwrap_or_else(|| {
        default_transcription_output_path(&request.media_path, request.settings.output_format)
    });

    if let Some(sidecar_path) = request.settings.sidecar_path {
        render_sidecar_transcript(&sidecar_path, &output_path, request.settings.output_format)?;
        return Ok(TranscriptionOutcome { output_path });
    }

    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        format!(
            "transcription backend is pending migration for {}; planned output: {}",
            request.media_path.display(),
            output_path.display()
        ),
    ))
}

fn render_sidecar_transcript(
    sidecar_path: &Path,
    output_path: &Path,
    output_format: TranscriptionFormat,
) -> io::Result<()> {
    let document = read_document(sidecar_path)?;
    if output_format != TranscriptionFormat::Txt
        && document
            .segments
            .iter()
            .any(|segment| segment.start.is_none() || segment.end.is_none())
    {
        return Err(io::Error::other(
            "sidecar transcript lacks timing data; use --format txt or a timed subtitle sidecar",
        ));
    }

    let options = RenderOptions::new(false, Some(output_format.extension().to_owned()));
    render_and_write_document(&document, &document.segments, output_path, &options)?;
    Ok(())
}

fn default_transcription_output_path(
    media_path: &Path,
    output_format: TranscriptionFormat,
) -> PathBuf {
    media_path.with_extension(output_format.extension())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn default_output_uses_requested_format() {
        let output = default_transcription_output_path(
            Path::new("/tmp/movie.mp4"),
            TranscriptionFormat::Vtt,
        );

        assert_eq!(output, PathBuf::from("/tmp/movie.vtt"));
    }

    #[test]
    fn transcribe_reports_pending_backend_with_planned_output() {
        let error = transcribe_media(TranscriptionRequest {
            media_path: PathBuf::from("movie.mp4"),
            output_path: None,
            settings: TranscriptionSettings::default(),
        })
        .expect_err("transcription backend should be pending");

        assert_eq!(error.kind(), io::ErrorKind::Unsupported);
        assert!(error.to_string().contains("movie.srt"));
    }

    #[test]
    fn transcribes_from_timed_sidecar() {
        let root = temp_root("sidecar");
        fs::create_dir_all(&root).expect("create temp root");
        let sidecar_path = root.join("movie.srt");
        fs::write(&sidecar_path, "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n")
            .expect("write sidecar");
        let output_path = root.join("movie.out.srt");

        let outcome = transcribe_media(TranscriptionRequest {
            media_path: root.join("movie.mp4"),
            output_path: Some(output_path.clone()),
            settings: TranscriptionSettings {
                sidecar_path: Some(sidecar_path),
                ..TranscriptionSettings::default()
            },
        })
        .expect("transcribe sidecar");
        let output = fs::read_to_string(&output_path).expect("read output");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(outcome.output_path, output_path);
        assert!(output.contains("hello"));
        assert!(output.contains("00:00:00,000 --> 00:00:01,000"));
    }

    #[test]
    fn untimed_sidecar_requires_txt_output() {
        let root = temp_root("untimed");
        fs::create_dir_all(&root).expect("create temp root");
        let sidecar_path = root.join("movie.txt");
        fs::write(&sidecar_path, "hello\n").expect("write sidecar");

        let error = transcribe_media(TranscriptionRequest {
            media_path: root.join("movie.mp4"),
            output_path: None,
            settings: TranscriptionSettings {
                sidecar_path: Some(sidecar_path),
                ..TranscriptionSettings::default()
            },
        })
        .expect_err("untimed sidecar should require txt output");
        let _ = fs::remove_dir_all(&root);

        assert!(error.to_string().contains("lacks timing"));
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-transcription-{label}-{nanos}"))
    }
}
