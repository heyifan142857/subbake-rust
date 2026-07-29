use std::path::Path;

use crate::entities::{
    BilingualOrder, SubtitleDocument, SubtitleDocumentMetadata, SubtitleSegment,
};
use crate::error::{CoreError, CoreResult};
use crate::formats::{bilingual_text, split_blocks};

const TIMESTAMP_SEPARATOR: &str = "-->";

pub fn sanitize_ass_derived_font_tags(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    let mut font_stack = Vec::new();
    while let Some(relative_start) = text[cursor..].find('<') {
        let start = cursor + relative_start;
        output.push_str(&text[cursor..start]);
        let Some(relative_end) = text[start + 1..].find('>') else {
            output.push_str(&text[start..]);
            return output;
        };
        let end = start + 1 + relative_end;
        let raw = &text[start..=end];
        match parse_font_tag(&text[start + 1..end]) {
            Some(FontTag::Open(attributes)) => {
                let keep = !attributes.is_empty();
                font_stack.push(keep);
                if keep {
                    output.push_str("<font ");
                    output.push_str(&attributes.join(" "));
                    output.push('>');
                }
            }
            Some(FontTag::Close) => match font_stack.pop() {
                Some(true) => output.push_str("</font>"),
                Some(false) => {}
                None => output.push_str(raw),
            },
            None => output.push_str(raw),
        }
        cursor = end + 1;
    }
    output.push_str(&text[cursor..]);
    output
}

enum FontTag {
    Open(Vec<String>),
    Close,
}

fn parse_font_tag(inner: &str) -> Option<FontTag> {
    let trimmed = inner.trim();
    if trimmed.eq_ignore_ascii_case("/font") {
        return Some(FontTag::Close);
    }
    let prefix = trimmed.get(..4)?;
    if !prefix.eq_ignore_ascii_case("font") {
        return None;
    }
    let rest = trimmed.get(4..)?;
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let attributes = parse_attributes(rest.trim())?
        .into_iter()
        .filter(|(name, _)| {
            !name.eq_ignore_ascii_case("face") && !name.eq_ignore_ascii_case("size")
        })
        .map(|(_, raw)| raw)
        .collect();
    Some(FontTag::Open(attributes))
}

fn parse_attributes(text: &str) -> Option<Vec<(String, String)>> {
    let mut attributes = Vec::new();
    let bytes = text.as_bytes();
    let mut cursor = 0;
    while cursor < bytes.len() {
        while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        if cursor == bytes.len() {
            break;
        }
        let start = cursor;
        while bytes
            .get(cursor)
            .is_some_and(|byte| !byte.is_ascii_whitespace() && *byte != b'=' && *byte != b'/')
        {
            cursor += 1;
        }
        if cursor == start {
            return None;
        }
        let name = text[start..cursor].to_owned();
        while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        if bytes.get(cursor) == Some(&b'=') {
            cursor += 1;
            while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                cursor += 1;
            }
            match bytes.get(cursor) {
                Some(b'\'' | b'"') => {
                    let quote = bytes[cursor];
                    cursor += 1;
                    while bytes.get(cursor).is_some_and(|byte| *byte != quote) {
                        cursor += 1;
                    }
                    if bytes.get(cursor) != Some(&quote) {
                        return None;
                    }
                    cursor += 1;
                }
                Some(_) => {
                    while bytes
                        .get(cursor)
                        .is_some_and(|byte| !byte.is_ascii_whitespace() && *byte != b'/')
                    {
                        cursor += 1;
                    }
                }
                None => return None,
            }
        }
        attributes.push((name, text[start..cursor].trim().to_owned()));
    }
    Some(attributes)
}

pub fn parse(path: &Path, text: &str) -> CoreResult<SubtitleDocument> {
    let normalized = text
        .trim_start_matches('\u{feff}')
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let normalized = normalized.trim();

    if normalized.is_empty() {
        return Ok(SubtitleDocument {
            path: path.to_path_buf(),
            format: "srt".to_owned(),
            segments: Vec::new(),
            header: None,
            passthrough_blocks: Vec::new(),
            metadata: SubtitleDocumentMetadata::None,
        });
    }

    let segments = split_blocks(normalized)
        .iter()
        .enumerate()
        .map(|(index, block)| parse_block(block, index + 1))
        .collect::<CoreResult<Vec<_>>>()?;

    Ok(SubtitleDocument {
        path: path.to_path_buf(),
        format: "srt".to_owned(),
        segments,
        header: None,
        passthrough_blocks: Vec::new(),
        metadata: SubtitleDocumentMetadata::None,
    })
}

pub fn render(
    source_segments: &[SubtitleSegment],
    segments: &[SubtitleSegment],
    bilingual: bool,
    bilingual_order: BilingualOrder,
) -> CoreResult<String> {
    let mut blocks = Vec::new();

    for segment in segments {
        let start = segment.start.as_deref().unwrap_or_default();
        let end = segment.end.as_deref().unwrap_or_default();
        let mut timing_line = format!("{start} {TIMESTAMP_SEPARATOR} {end}");
        if let Some(settings) = segment
            .settings
            .as_deref()
            .filter(|value| !value.is_empty())
        {
            timing_line.push(' ');
            timing_line.push_str(settings);
        }

        let text = if bilingual {
            source_segments
                .iter()
                .find(|source| source.id == segment.id)
                .map(|source| bilingual_text(&source.text, &segment.text, bilingual_order))
                .unwrap_or_else(|| segment.text.clone())
        } else {
            segment.text.clone()
        };
        let mut block = format!(
            "{}\n{}\n{}",
            segment.identifier.as_deref().unwrap_or(&segment.id),
            timing_line,
            text
        );
        while block.ends_with('\n') {
            block.pop();
        }
        blocks.push(block);
    }

    if blocks.is_empty() {
        Ok(String::new())
    } else {
        Ok(format!("{}\n", blocks.join("\n\n")))
    }
}

fn parse_block(block: &str, cue_index: usize) -> CoreResult<SubtitleSegment> {
    let lines = block.lines().map(str::trim_end).collect::<Vec<_>>();
    let (timing_index, timing) = lines
        .iter()
        .enumerate()
        .find_map(|(index, line)| parse_timing_line(line).map(|timing| (index, timing)))
        .ok_or_else(|| CoreError::MalformedSubtitle(format!("Malformed SRT block:\n{block}")))?;

    let identifier = lines[..timing_index]
        .iter()
        .map(|line| line.trim())
        .find(|line| !line.is_empty())
        .map(ToOwned::to_owned);
    let id = identifier.clone().unwrap_or_else(|| cue_index.to_string());
    let text = lines[timing_index + 1..].join("\n");

    Ok(SubtitleSegment {
        id,
        text,
        start: Some(timing.start),
        end: Some(timing.end),
        identifier,
        settings: timing.settings,
    })
}

struct Timing {
    start: String,
    end: String,
    settings: Option<String>,
}

fn parse_timing_line(line: &str) -> Option<Timing> {
    let (start, rest) = line.trim().split_once(TIMESTAMP_SEPARATOR)?;
    let start = start.trim();
    let rest = rest.trim();
    if start.is_empty() || rest.is_empty() {
        return None;
    }

    let mut parts = rest.split_whitespace();
    let end = parts.next()?;
    let settings = parts.collect::<Vec<_>>().join(" ");

    Some(Timing {
        start: start.to_owned(),
        end: end.to_owned(),
        settings: if settings.is_empty() {
            None
        } else {
            Some(settings)
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_nested_ass_font_size_and_face_tags() {
        let source = "<font face=\"sans-serif\"><font size=\"71\"><i>Hello</i></font></font>";
        assert_eq!(sanitize_ass_derived_font_tags(source), "<i>Hello</i>");
    }

    #[test]
    fn retains_other_font_attributes_and_balanced_closer() {
        let source = "<FONT face=\"Arial\" color=\"#ff0000\" size=71>Hello</FONT>";
        assert_eq!(
            sanitize_ass_derived_font_tags(source),
            "<font color=\"#ff0000\">Hello</font>"
        );
    }
}
