use std::borrow::Cow;
use std::path::Path;

use crate::entities::{BilingualOrder, SubtitleDocument, SubtitleSegment};
use crate::error::{CoreError, CoreResult};

mod ass;
mod srt;
mod ttml;
mod txt;
mod vtt;

#[derive(Debug, Clone, PartialEq)]
pub struct RenderOptions {
    pub bilingual: bool,
    pub bilingual_order: BilingualOrder,
    pub output_format: Option<String>,
    pub bilingual_font_scale: f64,
}

impl RenderOptions {
    pub fn new(bilingual: bool, output_format: Option<String>) -> Self {
        Self {
            bilingual,
            bilingual_order: BilingualOrder::default(),
            output_format,
            bilingual_font_scale: 1.0,
        }
    }

    pub fn with_bilingual_order(mut self, bilingual_order: BilingualOrder) -> Self {
        self.bilingual_order = bilingual_order;
        self
    }

    pub fn with_bilingual_font_scale(mut self, scale: f64) -> Self {
        self.bilingual_font_scale = scale;
        self
    }
}

pub use srt::sanitize_ass_derived_font_tags;

pub fn supported_format_from_path(path: &Path) -> Option<&'static str> {
    match path.extension().and_then(|value| value.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("srt") => Some("srt"),
        Some(ext) if ext.eq_ignore_ascii_case("vtt") => Some("vtt"),
        Some(ext) if ext.eq_ignore_ascii_case("txt") => Some("txt"),
        Some(ext) if ext.eq_ignore_ascii_case("ass") => Some("ass"),
        Some(ext) if ext.eq_ignore_ascii_case("ssa") => Some("ssa"),
        Some(ext) if ext.eq_ignore_ascii_case("ttml") => Some("ttml"),
        Some(ext) if ext.eq_ignore_ascii_case("dfxp") => Some("dfxp"),
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
        "ass" => ass::parse(path, text),
        "ssa" => ass::parse(path, text).map(|mut document| {
            document.format = "ssa".to_owned();
            document
        }),
        "ttml" => ttml::parse(path, text),
        "dfxp" => ttml::parse(path, text).map(|mut document| {
            document.format = "dfxp".to_owned();
            document
        }),
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

    let portable_document;
    let portable_segments;
    let (document, translations) = if matches!(document.format.as_str(), "ass" | "ssa")
        && !matches!(target_format.as_str(), "ass" | "ssa")
    {
        portable_document = SubtitleDocument {
            segments: ass::portable_segments(&document.segments, &target_format)?,
            metadata: Default::default(),
            ..document.clone()
        };
        portable_segments = ass::portable_segments(translations, &target_format)?;
        (&portable_document, portable_segments.as_slice())
    } else if matches!(document.format.as_str(), "ttml" | "dfxp")
        && !matches!(target_format.as_str(), "ttml" | "dfxp")
    {
        portable_document = SubtitleDocument {
            segments: ttml::portable_segments(&document.segments, &target_format)?,
            metadata: Default::default(),
            ..document.clone()
        };
        portable_segments = ttml::portable_segments(translations, &target_format)?;
        (&portable_document, portable_segments.as_slice())
    } else {
        (document, translations)
    };

    match target_format.as_str() {
        "srt" => srt::render(
            &document.segments,
            translations,
            options.bilingual,
            options.bilingual_order,
        ),
        "vtt" => vtt::render(
            document,
            translations,
            options.bilingual,
            options.bilingual_order,
        ),
        "txt" => txt::render(
            &document.segments,
            translations,
            options.bilingual,
            options.bilingual_order,
        ),
        "ass" => ass::render(
            document,
            translations,
            options.bilingual,
            options.bilingual_order,
            options.bilingual_font_scale,
        ),
        "ssa" => ass::render(
            document,
            translations,
            options.bilingual,
            options.bilingual_order,
            options.bilingual_font_scale,
        ),
        "ttml" => ttml::render(
            document,
            translations,
            options.bilingual,
            options.bilingual_order,
        ),
        "dfxp" => ttml::render(
            document,
            translations,
            options.bilingual,
            options.bilingual_order,
        ),
        _ => Err(CoreError::UnsupportedFormat(target_format)),
    }
}

pub(crate) fn bilingual_text(source: &str, target: &str, order: BilingualOrder) -> String {
    match order {
        BilingualOrder::SourceFirst => format!("{source}\n{target}"),
        BilingualOrder::TargetFirst => format!("{target}\n{source}"),
    }
}

/// Normalizes text-file line endings at the format boundary.
///
/// Parsers accept Unix, Windows, and legacy carriage-return separators, while
/// renderers emit the workspace's canonical `\n` representation.
pub(crate) fn normalize_line_endings(text: &str) -> Cow<'_, str> {
    if text.contains('\r') {
        Cow::Owned(text.replace("\r\n", "\n").replace('\r', "\n"))
    } else {
        Cow::Borrowed(text)
    }
}

pub fn normalize_format(value: &str) -> CoreResult<String> {
    let normalized = value.trim().trim_start_matches('.').to_lowercase();
    match normalized.as_str() {
        "srt" | "vtt" | "txt" | "ass" | "ssa" | "ttml" | "dfxp" => Ok(normalized),
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
    fn srt_accepts_platform_line_endings_and_renders_canonical_lf() {
        let canonical = "1\n00:00:00,000 --> 00:00:01,000\nhello\n";

        for text in [
            canonical.to_owned(),
            canonical.replace('\n', "\r\n"),
            canonical.replace('\n', "\r"),
        ] {
            let doc = parse_document_text(&PathBuf::from("clip.srt"), &text, None)
                .expect("parse platform line endings");
            let rendered = render_document(&doc, &doc.segments, &RenderOptions::new(false, None))
                .expect("render canonical SRT");

            assert_eq!(rendered, canonical);
        }
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
                semantic: Default::default(),
            },
            SubtitleSegment {
                id: "2".to_owned(),
                text: "世界".to_owned(),
                start: None,
                end: None,
                identifier: None,
                settings: None,
                semantic: Default::default(),
            },
        ];

        let rendered = render_document(&doc, &translated, &RenderOptions::new(true, None))
            .expect("render txt");
        assert_eq!(rendered, "你好\nhello\n世界\nworld\n");
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
        assert!(rendered.contains("你好\nHello"));
        assert!(rendered.contains("00:00:00,000 --> 00:00:01,000"));
    }

    #[test]
    fn can_render_source_language_first() {
        let path = PathBuf::from("clip.srt");
        let doc = parse_document_text(&path, "1\n00:00:00,000 --> 00:00:01,000\nHello\n", None)
            .expect("parse srt");
        let mut translated = doc.segments.clone();
        translated[0].text = "你好".to_owned();
        let options =
            RenderOptions::new(true, None).with_bilingual_order(BilingualOrder::SourceFirst);

        let rendered = render_document(&doc, &translated, &options).expect("render srt");

        assert!(rendered.contains("Hello\n你好"));
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
        assert!(rendered.contains("c1\n00:00.000 --> 00:01.000 align:start\n你好\nHello"));
    }

    #[test]
    fn ass_to_srt_drops_layout_overrides_but_keeps_semantic_emphasis() {
        let ass = "[Script Info]\nPlayResY: 1080\n\n[V4+ Styles]\nFormat: Name, Fontsize\nStyle: Default,54\n\n[Events]\nFormat: Start, End, Text\nDialogue: 0:00:00.00,0:00:01.00,{\\an8\\i1}Hello{\\i0}\n";
        let document = parse_document_text(Path::new("clip.ass"), ass, None).expect("parse ASS");
        let rendered = render_document(
            &document,
            &document.segments,
            &RenderOptions::new(false, Some("srt".to_owned())),
        )
        .expect("render SRT fallback");

        assert!(rendered.contains("<i>Hello</i>"));
        assert!(rendered.contains("00:00:00,000 --> 00:00:01,000"));
        assert!(!rendered.contains("\\an8"));
    }

    #[test]
    fn non_ass_source_cannot_invent_ass_style_metadata() {
        let document = parse_document_text(
            Path::new("clip.srt"),
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
            None,
        )
        .expect("parse SRT");

        let error = render_document(
            &document,
            &document.segments,
            &RenderOptions::new(false, Some("ass".to_owned())),
        )
        .expect_err("ASS output needs source metadata");

        assert!(error.to_string().contains("ASS source style metadata"));
    }
}
