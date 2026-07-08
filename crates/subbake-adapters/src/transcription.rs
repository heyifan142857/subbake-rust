use std::io;
use std::path::{Path, PathBuf};

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
        }
    }
}

pub fn transcribe_media(request: TranscriptionRequest) -> io::Result<TranscriptionOutcome> {
    let output_path = request.output_path.unwrap_or_else(|| {
        default_transcription_output_path(&request.media_path, request.settings.output_format)
    });

    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        format!(
            "transcription backend is pending migration for {}; planned output: {}",
            request.media_path.display(),
            output_path.display()
        ),
    ))
}

fn default_transcription_output_path(
    media_path: &Path,
    output_format: TranscriptionFormat,
) -> PathBuf {
    media_path.with_extension(output_format.extension())
}

#[cfg(test)]
mod tests {
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
}
