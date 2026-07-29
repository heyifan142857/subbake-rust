use std::path::Path;

use crate::entities::{
    AssDialogueRecord, AssDocumentMetadata, AssRecord, BilingualOrder, SubtitleDocument,
    SubtitleDocumentMetadata, SubtitleSegment,
};
use crate::error::{CoreError, CoreResult};
use crate::formats::bilingual_text;

const EVENTS_SECTION: &str = "[events]";

pub fn parse(path: &Path, text: &str) -> CoreResult<SubtitleDocument> {
    let had_bom = text.starts_with('\u{feff}');
    let normalized = text
        .trim_start_matches('\u{feff}')
        .replace('\0', "")
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    if normalized.trim().is_empty() {
        return Err(CoreError::MalformedSubtitle(
            "Malformed ASS file: empty input.".to_owned(),
        ));
    }

    let mut in_events = false;
    let mut event_format: Option<Vec<String>> = None;
    let mut records = Vec::new();
    let mut segments = Vec::new();

    for (line_index, line) in normalized.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_events = trimmed.eq_ignore_ascii_case(EVENTS_SECTION);
            event_format = None;
            records.push(AssRecord::Raw(line.to_owned()));
            continue;
        }

        if in_events && starts_with_label(trimmed, "Format") {
            let (_, value) =
                split_label(trimmed).ok_or_else(|| malformed_line(line_index, line))?;
            let fields = value
                .split(',')
                .map(|field| field.trim().to_ascii_lowercase())
                .collect::<Vec<_>>();
            validate_event_format(&fields, line_index)?;
            event_format = Some(fields);
            records.push(AssRecord::Raw(line.to_owned()));
            continue;
        }

        if in_events && starts_with_label(trimmed, "Dialogue") {
            let Some(format) = event_format.as_ref() else {
                return Err(CoreError::MalformedSubtitle(format!(
                    "Malformed ASS line {}: Dialogue appears before an Events Format line.",
                    line_index + 1
                )));
            };
            let (event_kind, value) =
                split_label(trimmed).ok_or_else(|| malformed_line(line_index, line))?;
            let fields = value
                .splitn(format.len(), ',')
                .map(str::to_owned)
                .collect::<Vec<_>>();
            if fields.len() != format.len() {
                return Err(malformed_line(line_index, line));
            }
            let start_index = field_index(format, "start", line_index)?;
            let end_index = field_index(format, "end", line_index)?;
            let text_index = field_index(format, "text", line_index)?;
            let source_text = ass_text_to_model(&fields[text_index]);
            if source_text.trim().is_empty() || is_drawing_dialogue(&fields[text_index]) {
                records.push(AssRecord::Raw(line.to_owned()));
                continue;
            }
            let segment_id = (segments.len() + 1).to_string();
            segments.push(SubtitleSegment {
                id: segment_id.clone(),
                text: source_text,
                start: Some(fields[start_index].trim().to_owned()),
                end: Some(fields[end_index].trim().to_owned()),
                identifier: None,
                settings: None,
            });
            records.push(AssRecord::Dialogue(AssDialogueRecord {
                segment_id,
                event_kind: event_kind.to_owned(),
                fields,
                start_index,
                end_index,
                text_index,
            }));
            continue;
        }

        records.push(AssRecord::Raw(line.to_owned()));
    }

    if !records.iter().any(|record| {
        matches!(record, AssRecord::Raw(line) if line.trim().eq_ignore_ascii_case(EVENTS_SECTION))
    }) {
        return Err(CoreError::MalformedSubtitle(
            "Malformed ASS file: missing [Events] section.".to_owned(),
        ));
    }

    Ok(SubtitleDocument {
        path: path.to_path_buf(),
        format: "ass".to_owned(),
        segments,
        header: None,
        passthrough_blocks: Vec::new(),
        metadata: SubtitleDocumentMetadata::Ass(AssDocumentMetadata { had_bom, records }),
    })
}

pub fn render(
    document: &SubtitleDocument,
    segments: &[SubtitleSegment],
    bilingual: bool,
    bilingual_order: BilingualOrder,
    bilingual_font_scale: f64,
) -> CoreResult<String> {
    let SubtitleDocumentMetadata::Ass(metadata) = &document.metadata else {
        return Err(CoreError::UnsupportedFormat(
            "ASS output requires ASS source style metadata".to_owned(),
        ));
    };
    let scale = if bilingual { bilingual_font_scale } else { 1.0 };
    validate_scale(scale)?;

    let mut output = String::new();
    if metadata.had_bom {
        output.push('\u{feff}');
    }
    let mut in_styles = false;
    let mut style_format: Option<Vec<String>> = None;

    for record in &metadata.records {
        let line = match record {
            AssRecord::Raw(line) => {
                let trimmed = line.trim();
                if trimmed.starts_with('[') && trimmed.ends_with(']') {
                    in_styles = trimmed.eq_ignore_ascii_case("[V4+ Styles]")
                        || trimmed.eq_ignore_ascii_case("[V4 Styles]");
                    style_format = None;
                    line.clone()
                } else if in_styles && starts_with_label(trimmed, "Format") {
                    let (_, value) = split_label(trimmed)
                        .ok_or_else(|| CoreError::MalformedSubtitle(line.clone()))?;
                    style_format = Some(
                        value
                            .split(',')
                            .map(|field| field.trim().to_ascii_lowercase())
                            .collect(),
                    );
                    line.clone()
                } else if in_styles && starts_with_label(trimmed, "Style") && scale != 1.0 {
                    scale_style_line(line, style_format.as_deref(), scale)?
                } else {
                    line.clone()
                }
            }
            AssRecord::Dialogue(template) => {
                let translated = segments
                    .iter()
                    .find(|segment| segment.id == template.segment_id)
                    .ok_or_else(|| {
                        CoreError::DataInvariant(format!(
                            "missing translated ASS segment {}",
                            template.segment_id
                        ))
                    })?;
                let source = document
                    .segments
                    .iter()
                    .find(|segment| segment.id == template.segment_id);
                let text = if bilingual {
                    source
                        .map(|source| {
                            bilingual_text(&source.text, &translated.text, bilingual_order)
                        })
                        .unwrap_or_else(|| translated.text.clone())
                } else {
                    translated.text.clone()
                };
                let mut fields = template.fields.clone();
                fields[template.start_index] = translated
                    .start
                    .clone()
                    .unwrap_or_else(|| template.fields[template.start_index].clone());
                fields[template.end_index] = translated
                    .end
                    .clone()
                    .unwrap_or_else(|| template.fields[template.end_index].clone());
                let rendered_text = model_text_to_ass(&text);
                fields[template.text_index] = if scale == 1.0 {
                    rendered_text
                } else {
                    scale_inline_font_sizes(&rendered_text, scale)?
                };
                format!("{}: {}", template.event_kind, fields.join(","))
            }
        };
        output.push_str(&line);
        output.push('\n');
    }
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
            segment.text = ass_text_to_portable(&segment.text);
            if matches!(target_format, "srt" | "vtt") {
                segment.start = segment
                    .start
                    .as_deref()
                    .map(|value| ass_timestamp_to_portable(value, target_format))
                    .transpose()?;
                segment.end = segment
                    .end
                    .as_deref()
                    .map(|value| ass_timestamp_to_portable(value, target_format))
                    .transpose()?;
            }
            Ok(segment)
        })
        .collect()
}

fn ass_timestamp_to_portable(value: &str, target_format: &str) -> CoreResult<String> {
    let parts = value.trim().split(':').collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(CoreError::MalformedSubtitle(format!(
            "invalid ASS timestamp `{value}`"
        )));
    }
    let hours = parts[0]
        .parse::<u64>()
        .map_err(|_| CoreError::MalformedSubtitle(format!("invalid ASS timestamp `{value}`")))?;
    let minutes = parts[1]
        .parse::<u64>()
        .map_err(|_| CoreError::MalformedSubtitle(format!("invalid ASS timestamp `{value}`")))?;
    let (seconds, fraction) = parts[2].split_once('.').unwrap_or((parts[2], ""));
    let seconds = seconds
        .parse::<u64>()
        .map_err(|_| CoreError::MalformedSubtitle(format!("invalid ASS timestamp `{value}`")))?;
    if minutes > 59 || seconds > 59 || !fraction.chars().all(|ch| ch.is_ascii_digit()) {
        return Err(CoreError::MalformedSubtitle(format!(
            "invalid ASS timestamp `{value}`"
        )));
    }
    let mut milliseconds = fraction.chars().take(3).collect::<String>();
    while milliseconds.len() < 3 {
        milliseconds.push('0');
    }
    let milliseconds = if milliseconds.is_empty() {
        0
    } else {
        milliseconds
            .parse::<u64>()
            .map_err(|_| CoreError::MalformedSubtitle(format!("invalid ASS timestamp `{value}`")))?
    };
    let separator = if target_format == "srt" { ',' } else { '.' };
    Ok(format!(
        "{hours:02}:{minutes:02}:{seconds:02}{separator}{milliseconds:03}"
    ))
}

fn validate_event_format(fields: &[String], line_index: usize) -> CoreResult<()> {
    for required in ["start", "end", "text"] {
        if !fields.iter().any(|field| field == required) {
            return Err(CoreError::MalformedSubtitle(format!(
                "Malformed ASS line {}: Events Format is missing {required}.",
                line_index + 1
            )));
        }
    }
    if fields.last().map(String::as_str) != Some("text") {
        return Err(CoreError::MalformedSubtitle(format!(
            "Malformed ASS line {}: Events Text must be the final field.",
            line_index + 1
        )));
    }
    Ok(())
}

fn field_index(fields: &[String], name: &str, line_index: usize) -> CoreResult<usize> {
    fields
        .iter()
        .position(|field| field == name)
        .ok_or_else(|| {
            CoreError::MalformedSubtitle(format!(
                "Malformed ASS line {}: missing {name} field.",
                line_index + 1
            ))
        })
}

fn malformed_line(line_index: usize, line: &str) -> CoreError {
    CoreError::MalformedSubtitle(format!("Malformed ASS line {}: {line}", line_index + 1))
}

fn starts_with_label(line: &str, label: &str) -> bool {
    line.get(..label.len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(label))
        && line[label.len()..].trim_start().starts_with(':')
}

fn split_label(line: &str) -> Option<(&str, &str)> {
    let (label, value) = line.split_once(':')?;
    Some((label.trim(), value.trim_start()))
}

fn ass_text_to_model(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut in_override = false;
    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                in_override = true;
                output.push(ch);
            }
            '}' => {
                in_override = false;
                output.push(ch);
            }
            '\\' if !in_override && matches!(chars.peek(), Some('N' | 'n')) => {
                let _ = chars.next();
                output.push('\n');
            }
            _ => output.push(ch),
        }
    }
    output
}

fn model_text_to_ass(text: &str) -> String {
    text.replace('\n', "\\N")
}

fn is_drawing_dialogue(text: &str) -> bool {
    text.split('{').skip(1).any(|part| {
        part.split_once('}').is_some_and(|(override_block, _)| {
            override_block.split('\\').skip(1).any(|command| {
                let command = command.trim_start();
                command
                    .strip_prefix('p')
                    .and_then(|value| value.chars().next())
                    .is_some_and(|value| matches!(value, '1'..='9'))
            })
        })
    })
}

fn ass_text_to_portable(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    let mut state = SemanticState::default();
    while let Some(relative_start) = text[cursor..].find('{') {
        let start = cursor + relative_start;
        output.push_str(&text[cursor..start]);
        let Some(relative_end) = text[start + 1..].find('}') else {
            output.push_str(&text[start..]);
            cursor = text.len();
            break;
        };
        let end = start + 1 + relative_end;
        let next = semantic_state(&text[start + 1..end], state);
        write_state_transition(&mut output, state, next);
        state = next;
        cursor = end + 1;
    }
    output.push_str(&text[cursor..]);
    write_state_transition(&mut output, state, SemanticState::default());
    output
}

#[derive(Debug, Clone, Copy, Default)]
struct SemanticState {
    bold: bool,
    italic: bool,
    underline: bool,
}

fn semantic_state(block: &str, mut state: SemanticState) -> SemanticState {
    for command in block.split('\\').skip(1) {
        let command = command.trim_start();
        let Some(kind) = command.chars().next() else {
            continue;
        };
        if !matches!(kind, 'b' | 'i' | 'u') {
            continue;
        }
        let value = command[kind.len_utf8()..]
            .chars()
            .take_while(|ch| ch.is_ascii_digit() || *ch == '-')
            .collect::<String>();
        let Ok(value) = value.parse::<i32>() else {
            continue;
        };
        match kind {
            'b' => state.bold = value != 0,
            'i' => state.italic = value != 0,
            'u' => state.underline = value != 0,
            _ => {}
        }
    }
    state
}

fn write_state_transition(output: &mut String, from: SemanticState, to: SemanticState) {
    if from.underline {
        output.push_str("</u>");
    }
    if from.italic {
        output.push_str("</i>");
    }
    if from.bold {
        output.push_str("</b>");
    }
    if to.bold {
        output.push_str("<b>");
    }
    if to.italic {
        output.push_str("<i>");
    }
    if to.underline {
        output.push_str("<u>");
    }
}

fn validate_scale(scale: f64) -> CoreResult<()> {
    if !scale.is_finite() || !(0.1..=2.0).contains(&scale) {
        return Err(CoreError::DataInvariant(format!(
            "bilingual font scale must be between 0.1 and 2.0, got {scale}"
        )));
    }
    Ok(())
}

fn scale_style_line(line: &str, format: Option<&[String]>, scale: f64) -> CoreResult<String> {
    let format = format.ok_or_else(|| {
        CoreError::MalformedSubtitle(
            "Malformed ASS styles: Style appears before a Format line.".to_owned(),
        )
    })?;
    let font_size_index = format
        .iter()
        .position(|field| field == "fontsize")
        .ok_or_else(|| {
            CoreError::MalformedSubtitle(
                "Malformed ASS styles: Format is missing Fontsize.".to_owned(),
            )
        })?;
    let (label, value) = split_label(line.trim())
        .ok_or_else(|| CoreError::MalformedSubtitle(format!("Malformed ASS style: {line}")))?;
    let mut fields = value.split(',').map(str::to_owned).collect::<Vec<_>>();
    if fields.len() != format.len() {
        return Err(CoreError::MalformedSubtitle(format!(
            "Malformed ASS style: {line}"
        )));
    }
    fields[font_size_index] = scale_number(&fields[font_size_index], scale)?;
    Ok(format!("{label}: {}", fields.join(",")))
}

fn scale_inline_font_sizes(text: &str, scale: f64) -> CoreResult<String> {
    let mut output = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut cursor = 0;
    while cursor < bytes.len() {
        if bytes[cursor] == b'\\'
            && bytes.get(cursor + 1) == Some(&b'f')
            && bytes.get(cursor + 2) == Some(&b's')
            && bytes
                .get(cursor + 3)
                .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'.')
        {
            output.push_str("\\fs");
            cursor += 3;
            let start = cursor;
            while bytes
                .get(cursor)
                .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'.')
            {
                cursor += 1;
            }
            output.push_str(&scale_number(&text[start..cursor], scale)?);
        } else {
            let ch = text[cursor..]
                .chars()
                .next()
                .ok_or_else(|| CoreError::DataInvariant("invalid ASS text boundary".to_owned()))?;
            output.push(ch);
            cursor += ch.len_utf8();
        }
    }
    Ok(output)
}

fn scale_number(value: &str, scale: f64) -> CoreResult<String> {
    let number = value
        .trim()
        .parse::<f64>()
        .map_err(|_| CoreError::MalformedSubtitle(format!("invalid ASS font size `{value}`")))?;
    let scaled = number * scale;
    let mut rendered = format!("{scaled:.2}");
    while rendered.contains('.') && rendered.ends_with('0') {
        rendered.pop();
    }
    if rendered.ends_with('.') {
        rendered.pop();
    }
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "[Script Info]\nPlayResX: 1920\nPlayResY: 1040\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize\nStyle: Default,sans-serif,71\n\n[Events]\nFormat: Layer, Start, End, Style, Text\nComment: 0,0:00:00.00,0:00:01.00,Default,note\nDialogue: 0,0:00:01.00,0:00:02.00,Default,{\\i1}Hello, world{\\i0}\\Nagain\n";

    #[test]
    fn parses_and_renders_ass_without_losing_structure() {
        let document = parse(Path::new("sample.ass"), SAMPLE).expect("parse ASS");
        assert_eq!(document.segments.len(), 1);
        assert_eq!(document.segments[0].text, "{\\i1}Hello, world{\\i0}\nagain");

        let rendered = render(
            &document,
            &document.segments,
            false,
            BilingualOrder::TargetFirst,
            1.0,
        )
        .expect("render ASS");

        assert_eq!(rendered, SAMPLE);
    }

    #[test]
    fn bilingual_scale_changes_style_and_inline_font_size() {
        let source = SAMPLE.replace("{\\i1}", "{\\fs80\\i1}");
        let document = parse(Path::new("sample.ass"), &source).expect("parse ASS");
        let mut translated = document.segments.clone();
        translated[0].text = "{\\fs80\\i1}你好{\\i0}".to_owned();

        let rendered = render(
            &document,
            &translated,
            true,
            BilingualOrder::TargetFirst,
            0.9,
        )
        .expect("render scaled ASS");

        assert!(rendered.contains("Style: Default,sans-serif,63.9"));
        assert!(rendered.contains("{\\fs72\\i1}你好{\\i0}\\N{\\fs72\\i1}Hello"));
    }

    #[test]
    fn portable_text_keeps_semantic_styles_only() {
        assert_eq!(
            ass_text_to_portable("{\\an8\\i1}Hello{\\i0} {\\fs71}world"),
            "<i>Hello</i> world"
        );
    }

    #[test]
    fn strips_ffmpeg_extradata_nul_separator() {
        let source = SAMPLE.replace("\n[Events]", "\n\0\n[Events]");
        let document = parse(Path::new("sample.ass"), &source).expect("parse FFmpeg ASS");
        let rendered = render(
            &document,
            &document.segments,
            false,
            BilingualOrder::TargetFirst,
            1.0,
        )
        .expect("render ASS");

        assert!(!rendered.contains('\0'));
    }

    #[test]
    fn converts_ass_centiseconds_to_srt_and_vtt_milliseconds() {
        assert_eq!(
            ass_timestamp_to_portable("0:01:02.34", "srt").expect("SRT timestamp"),
            "00:01:02,340"
        );
        assert_eq!(
            ass_timestamp_to_portable("1:02:03.4", "vtt").expect("VTT timestamp"),
            "01:02:03.400"
        );
    }
}
