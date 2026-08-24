use std::path::Path;

use crate::entities::{BilingualOrder, SubtitleDocument, SubtitleSegment};
use crate::error::{CoreError, CoreResult};
use crate::formats::bilingual_text;

pub fn parse(path: &Path, text: &str) -> CoreResult<SubtitleDocument> {
    let mut segments = Vec::new();
    let mut cursor = 0usize;
    while let Some((start, open_end, tag_name)) = find_open_paragraph(text, cursor) {
        let closing = format!("</{tag_name}>");
        let remainder = &text[open_end + 1..];
        let close_offset = remainder
            .to_ascii_lowercase()
            .find(&closing.to_ascii_lowercase())
            .ok_or_else(|| {
                CoreError::MalformedSubtitle("TTML paragraph is not closed".to_owned())
            })?;
        let close_start = open_end + 1 + close_offset;
        let attributes = &text[start + 1 + tag_name.len()..open_end];
        let begin = attribute(attributes, "begin").ok_or_else(|| {
            CoreError::MalformedSubtitle("TTML paragraph is missing begin".to_owned())
        })?;
        let end = match attribute(attributes, "end") {
            Some(end) => end,
            None => {
                let duration = attribute(attributes, "dur").ok_or_else(|| {
                    CoreError::MalformedSubtitle("TTML paragraph requires end or dur".to_owned())
                })?;
                format_ttml_time(parse_time_ms(&begin)? + parse_time_ms(&duration)?)
            }
        };
        let id = attribute(attributes, "xml:id")
            .or_else(|| attribute(attributes, "id"))
            .unwrap_or_else(|| (segments.len() + 1).to_string());
        let content = decode_content(&text[open_end + 1..close_start]);
        segments.push(SubtitleSegment {
            id,
            text: content,
            start: Some(normalize_clock_time(&begin)?),
            end: Some(normalize_clock_time(&end)?),
            identifier: None,
            settings: None,
            semantic: Default::default(),
        });
        cursor = close_start + closing.len();
    }
    if segments.is_empty() {
        return Err(CoreError::MalformedSubtitle(
            "Malformed TTML file: no timed paragraphs".to_owned(),
        ));
    }
    Ok(SubtitleDocument {
        path: path.to_path_buf(),
        format: "ttml".to_owned(),
        segments,
        header: None,
        passthrough_blocks: Vec::new(),
        metadata: Default::default(),
    })
}

pub fn render(
    document: &SubtitleDocument,
    translations: &[SubtitleSegment],
    bilingual: bool,
    order: BilingualOrder,
) -> CoreResult<String> {
    let mut output = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<tt xmlns=\"http://www.w3.org/ns/ttml\" xmlns:tts=\"http://www.w3.org/ns/ttml#styling\" xml:lang=\"und\">\n  <body>\n    <div>\n",
    );
    for source in &document.segments {
        let translated = translations
            .iter()
            .find(|segment| segment.id == source.id)
            .ok_or_else(|| {
                CoreError::DataInvariant(format!("missing translated TTML segment {}", source.id))
            })?;
        let text = if bilingual {
            bilingual_text(&source.text, &translated.text, order)
        } else {
            translated.text.clone()
        };
        let begin = source.start.as_deref().ok_or_else(|| {
            CoreError::MalformedSubtitle(format!("TTML segment {} has no begin", source.id))
        })?;
        let end = source.end.as_deref().ok_or_else(|| {
            CoreError::MalformedSubtitle(format!("TTML segment {} has no end", source.id))
        })?;
        output.push_str(&format!(
            "      <p xml:id=\"{}\" begin=\"{}\" end=\"{}\">{}</p>\n",
            escape_xml(&source.id),
            normalize_clock_time(begin)?,
            normalize_clock_time(end)?,
            escape_text(&text)
        ));
    }
    output.push_str("    </div>\n  </body>\n</tt>\n");
    Ok(output)
}

pub fn portable_segments(
    segments: &[SubtitleSegment],
    target_format: &str,
) -> CoreResult<Vec<SubtitleSegment>> {
    segments
        .iter()
        .cloned()
        .map(|mut segment| {
            if target_format == "txt" {
                segment.start = None;
                segment.end = None;
                return Ok(segment);
            }
            let separator = if target_format == "srt" { ',' } else { '.' };
            segment.start = segment
                .start
                .as_deref()
                .map(|value| parse_time_ms(value).map(|time| format_portable_time(time, separator)))
                .transpose()?;
            segment.end = segment
                .end
                .as_deref()
                .map(|value| parse_time_ms(value).map(|time| format_portable_time(time, separator)))
                .transpose()?;
            Ok(segment)
        })
        .collect()
}

fn find_open_paragraph(text: &str, mut cursor: usize) -> Option<(usize, usize, String)> {
    while let Some(relative) = text[cursor..].find('<') {
        let start = cursor + relative;
        let end = start + text[start..].find('>')?;
        let raw = text[start + 1..end].trim();
        if !raw.starts_with('/') && !raw.starts_with('!') && !raw.starts_with('?') {
            let name = raw.split_whitespace().next()?.trim_end_matches('/');
            if name
                .rsplit(':')
                .next()
                .is_some_and(|local| local.eq_ignore_ascii_case("p"))
            {
                return Some((start, end, name.to_owned()));
            }
        }
        cursor = end + 1;
    }
    None
}

fn attribute(attributes: &str, wanted: &str) -> Option<String> {
    let bytes = attributes.as_bytes();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        let name_start = cursor;
        while cursor < bytes.len() && !bytes[cursor].is_ascii_whitespace() && bytes[cursor] != b'='
        {
            cursor += 1;
        }
        let name = attributes.get(name_start..cursor)?;
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if bytes.get(cursor) != Some(&b'=') {
            cursor += 1;
            continue;
        }
        cursor += 1;
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        let quote = *bytes.get(cursor)?;
        if quote != b'\'' && quote != b'"' {
            continue;
        }
        cursor += 1;
        let value_start = cursor;
        while cursor < bytes.len() && bytes[cursor] != quote {
            cursor += 1;
        }
        let value = attributes.get(value_start..cursor)?.to_owned();
        cursor += 1;
        if name.eq_ignore_ascii_case(wanted) {
            return Some(decode_entities(&value));
        }
    }
    None
}

fn decode_content(content: &str) -> String {
    let mut output = String::new();
    let mut inline_stack = Vec::new();
    let mut cursor = 0usize;
    while let Some(relative) = content[cursor..].find('<') {
        let start = cursor + relative;
        output.push_str(&decode_entities(&content[cursor..start]));
        let Some(end_relative) = content[start..].find('>') else {
            break;
        };
        let end = start + end_relative;
        let tag = content[start + 1..end].trim().to_ascii_lowercase();
        if tag.starts_with("br") || tag.starts_with("tt:br") {
            output.push('\n');
        } else if tag.starts_with("span") || tag.starts_with("tt:span") {
            let marker =
                if tag.contains("fontstyle=\"italic\"") || tag.contains("fontstyle='italic'") {
                    Some("i")
                } else if tag.contains("fontweight=\"bold\"") || tag.contains("fontweight='bold'") {
                    Some("b")
                } else if tag.contains("textdecoration=\"underline\"")
                    || tag.contains("textdecoration='underline'")
                {
                    Some("u")
                } else {
                    None
                };
            if let Some(marker) = marker {
                output.push_str(&format!("<{marker}>"));
                inline_stack.push(marker);
            }
        } else if (tag == "/span" || tag == "/tt:span")
            && let Some(marker) = inline_stack.pop()
        {
            output.push_str(&format!("</{marker}>"));
        }
        cursor = end + 1;
    }
    output.push_str(&decode_entities(&content[cursor..]));
    output.trim().to_owned()
}

fn decode_entities(value: &str) -> String {
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_text(value: &str) -> String {
    let mut output = String::new();
    let mut cursor = 0usize;
    while cursor < value.len() {
        let tail = &value[cursor..];
        let marker = [
            ("<i>", "<span tts:fontStyle=\"italic\">"),
            ("</i>", "</span>"),
            ("<b>", "<span tts:fontWeight=\"bold\">"),
            ("</b>", "</span>"),
            ("<u>", "<span tts:textDecoration=\"underline\">"),
            ("</u>", "</span>"),
        ]
        .into_iter()
        .find(|(source, _)| tail.starts_with(source));
        if let Some((source, rendered)) = marker {
            output.push_str(rendered);
            cursor += source.len();
            continue;
        }
        let character = tail.chars().next().unwrap_or_default();
        match character {
            '&' => output.push_str("&amp;"),
            '<' => output.push_str("&lt;"),
            '>' => output.push_str("&gt;"),
            '"' => output.push_str("&quot;"),
            '\n' => output.push_str("<br/>"),
            _ => output.push(character),
        }
        cursor += character.len_utf8();
    }
    output
}

fn normalize_clock_time(value: &str) -> CoreResult<String> {
    Ok(format_ttml_time(parse_time_ms(value)?))
}

fn parse_time_ms(value: &str) -> CoreResult<u64> {
    let value = value.trim();
    if let Some(milliseconds) = value.strip_suffix("ms") {
        return milliseconds
            .trim()
            .parse::<u64>()
            .map_err(|_| malformed_time(value));
    }
    if let Some(seconds) = value.strip_suffix('s') {
        return seconds
            .trim()
            .parse::<f64>()
            .map(|seconds| (seconds * 1_000.0).round() as u64)
            .map_err(|_| malformed_time(value));
    }
    let normalized = value.replace(',', ".");
    let mut parts = normalized.split(':');
    let hours = parts.next().and_then(|part| part.parse::<u64>().ok());
    let minutes = parts.next().and_then(|part| part.parse::<u64>().ok());
    let seconds = parts.next().and_then(|part| part.parse::<f64>().ok());
    if parts.next().is_some() {
        return Err(malformed_time(value));
    }
    match (hours, minutes, seconds) {
        (Some(hours), Some(minutes), Some(seconds)) if minutes < 60 && seconds < 60.0 => {
            Ok(((hours * 3_600 + minutes * 60) * 1_000) + (seconds * 1_000.0).round() as u64)
        }
        _ => Err(malformed_time(value)),
    }
}

fn format_ttml_time(milliseconds: u64) -> String {
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        milliseconds / 3_600_000,
        (milliseconds / 60_000) % 60,
        (milliseconds / 1_000) % 60,
        milliseconds % 1_000
    )
}

fn format_portable_time(milliseconds: u64, separator: char) -> String {
    format!(
        "{:02}:{:02}:{:02}{}{:03}",
        milliseconds / 3_600_000,
        (milliseconds / 60_000) % 60,
        (milliseconds / 1_000) % 60,
        separator,
        milliseconds % 1_000
    )
}

fn malformed_time(value: &str) -> CoreError {
    CoreError::MalformedSubtitle(format!("unsupported TTML time expression `{value}`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_namespaced_paragraphs_duration_and_line_breaks() {
        let document = parse(
            Path::new("sample.dfxp"),
            "<tt:tt><tt:body><tt:div><tt:p xml:id='cue1' begin='1.5s' dur='2s'>Hello<br/>world &amp; all</tt:p></tt:div></tt:body></tt:tt>",
        )
        .expect("parse DFXP");

        assert_eq!(document.segments[0].id, "cue1");
        assert_eq!(document.segments[0].start.as_deref(), Some("00:00:01.500"));
        assert_eq!(document.segments[0].end.as_deref(), Some("00:00:03.500"));
        assert_eq!(document.segments[0].text, "Hello\nworld & all");
    }

    #[test]
    fn semantic_inline_styles_round_trip_as_protected_markers() {
        let document = parse(
            Path::new("styled.ttml"),
            "<tt><body><div><p begin='0s' end='1s'><span tts:fontStyle='italic'>Hello</span></p></div></body></tt>",
        )
        .expect("parse styled TTML");
        assert_eq!(document.segments[0].text, "<i>Hello</i>");

        let rendered = render(
            &document,
            &document.segments,
            false,
            BilingualOrder::TargetFirst,
        )
        .expect("render styled TTML");
        assert!(rendered.contains("<span tts:fontStyle=\"italic\">Hello</span>"));
    }
}
