use std::path::Path;

use crate::entities::{SubtitleDocument, SubtitleSegment};
use crate::error::{CoreError, CoreResult};

mod srt;
mod txt;
mod vtt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderOptions {
    pub bilingual: bool,
    pub output_format: Option<String>,
}

impl RenderOptions {
    pub fn new(bilingual: bool, output_format: Option<String>) -> Self {
        Self {
            bilingual,
            output_format,
        }
    }
}

pub fn supported_format_from_path(path: &Path) -> Option<&'static str> {
    match path.extension().and_then(|value| value.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("srt") => Some("srt"),
        Some(ext) if ext.eq_ignore_ascii_case("vtt") => Some("vtt"),
        Some(ext) if ext.eq_ignore_ascii_case("txt") => Some("txt"),
        _ => None,
    }
}

pub fn parse_document_text(
    path: &Path,
    text: &str,
    format: Option<&str>,
) -> CoreResult<SubtitleDocument> {
    let format = match format {
        Some(value) => normalize_format(value)?,
        None => supported_format_from_path(path)
            .ok_or_else(|| CoreError::UnsupportedFormat(path.display().to_string()))?
            .to_owned(),
    };

    match format.as_str() {
        "srt" => srt::parse(path, text),
        "vtt" => vtt::parse(path, text),
        "txt" => Ok(txt::parse(path, text)),
        _ => Err(CoreError::UnsupportedFormat(format)),
    }
}

pub fn render_document(
    document: &SubtitleDocument,
    translations: &[SubtitleSegment],
    options: &RenderOptions,
) -> CoreResult<String> {
    let target_format = match options.output_format.as_deref() {
        Some(value) => normalize_format(value)?,
        None => document.format.clone(),
    };

    match target_format.as_str() {
        "srt" => srt::render(&document.segments, translations, options.bilingual),
        "vtt" => vtt::render(document, translations, options.bilingual),
        "txt" => txt::render(&document.segments, translations, options.bilingual),
        _ => Err(CoreError::UnsupportedFormat(target_format)),
    }
}

pub fn normalize_format(value: &str) -> CoreResult<String> {
    let normalized = value.trim().trim_start_matches('.').to_lowercase();
    match normalized.as_str() {
        "srt" | "vtt" | "txt" => Ok(normalized),
        _ => Err(CoreError::UnsupportedFormat(value.to_owned())),
    }
}

pub(crate) fn split_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current = Vec::new();

    for line in text.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                blocks.push(current.join("\n"));
                current.clear();
            }
        } else {
            current.push(line.to_owned());
        }
    }

    if !current.is_empty() {
        blocks.push(current.join("\n"));
    }

    blocks
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn parses_and_renders_srt() {
        let path = PathBuf::from("clip.srt");
        let doc = parse_document_text(
            &path,
            "1\n00:00:00,000 --> 00:00:01,000 position:10%\nhello\n\n",
            None,
        )
        .expect("parse srt");

        assert_eq!(doc.segments[0].id, "1");
        assert_eq!(doc.segments[0].settings.as_deref(), Some("position:10%"));

        let rendered = render_document(&doc, &doc.segments, &RenderOptions::new(false, None))
            .expect("render srt");
        assert!(rendered.contains("00:00:00,000 --> 00:00:01,000 position:10%"));
    }

    #[test]
    fn preserves_vtt_passthrough_blocks() {
        let path = PathBuf::from("clip.vtt");
        let doc = parse_document_text(
            &path,
            "WEBVTT\n\nNOTE hello\n\n00:00.000 --> 00:01.000\nhello\n",
            None,
        )
        .expect("parse vtt");

        assert_eq!(doc.passthrough_blocks.len(), 1);
        let rendered = render_document(&doc, &doc.segments, &RenderOptions::new(false, None))
            .expect("render vtt");
        assert!(rendered.contains("NOTE hello"));
    }

    #[test]
    fn renders_bilingual_txt() {
        let path = PathBuf::from("clip.txt");
        let doc = parse_document_text(&path, "hello\nworld\n", None).expect("parse txt");
        let translated = vec![
            SubtitleSegment {
                id: "1".to_owned(),
                text: "你好".to_owned(),
                start: None,
                end: None,
                identifier: None,
                settings: None,
            },
            SubtitleSegment {
                id: "2".to_owned(),
                text: "世界".to_owned(),
                start: None,
                end: None,
                identifier: None,
                settings: None,
            },
        ];

        let rendered = render_document(&doc, &translated, &RenderOptions::new(true, None))
            .expect("render txt");
        assert_eq!(rendered, "hello\n你好\nworld\n世界\n");
    }

    #[test]
    fn renders_bilingual_srt() {
        let path = PathBuf::from("clip.srt");
        let doc = parse_document_text(&path, "1\n00:00:00,000 --> 00:00:01,000\nHello\n", None)
            .expect("parse srt");
        let mut translated = doc.segments.clone();
        translated[0].text = "你好".to_owned();

        let rendered = render_document(&doc, &translated, &RenderOptions::new(true, None))
            .expect("render srt");
        assert!(rendered.contains("Hello\n你好"));
        assert!(rendered.contains("00:00:00,000 --> 00:00:01,000"));
    }

    #[test]
    fn renders_bilingual_vtt_without_losing_metadata() {
        let path = PathBuf::from("clip.vtt");
        let doc = parse_document_text(
            &path,
            "WEBVTT\n\nNOTE hello\n\nc1\n00:00.000 --> 00:01.000 align:start\nHello\n",
            None,
        )
        .expect("parse vtt");
        let mut translated = doc.segments.clone();
        translated[0].text = "你好".to_owned();

        let rendered = render_document(&doc, &translated, &RenderOptions::new(true, None))
            .expect("render vtt");
        assert!(rendered.contains("NOTE hello"));
        assert!(rendered.contains("c1\n00:00.000 --> 00:01.000 align:start\nHello\n你好"));
    }
}
