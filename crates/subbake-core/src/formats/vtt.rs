use std::path::Path;

use crate::entities::{PassthroughBlock, SubtitleDocument, SubtitleSegment};
use crate::error::{CoreError, CoreResult};
use crate::formats::split_blocks;

const TIMESTAMP_SEPARATOR: &str = "-->";

pub fn parse(path: &Path, text: &str) -> CoreResult<SubtitleDocument> {
    let normalized = text
        .trim_start_matches('\u{feff}')
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    if normalized.trim().is_empty() {
        return Err(CoreError::MalformedSubtitle(
            "Malformed VTT file: missing WEBVTT header.".to_owned(),
        ));
    }

    let mut lines = normalized.lines();
    let header = lines.next().unwrap_or_default().trim().to_owned();
    if !header.starts_with("WEBVTT") {
        return Err(CoreError::MalformedSubtitle(
            "Malformed VTT file: first line must start with WEBVTT.".to_owned(),
        ));
    }

    let body = lines.collect::<Vec<_>>().join("\n");
    let body = body.trim();
    if body.is_empty() {
        return Ok(SubtitleDocument {
            path: path.to_path_buf(),
            format: "vtt".to_owned(),
            segments: Vec::new(),
            header: Some(header),
            passthrough_blocks: Vec::new(),
        });
    }

    let mut segments = Vec::new();
    let mut passthrough_blocks = Vec::new();
    for block in split_blocks(body) {
        match parse_cue(&block, segments.len() + 1) {
            Some(segment) => segments.push(segment),
            None => passthrough_blocks.push(PassthroughBlock {
                insert_before: segments.len(),
                content: block.trim().to_owned(),
            }),
        }
    }

    Ok(SubtitleDocument {
        path: path.to_path_buf(),
        format: "vtt".to_owned(),
        segments,
        header: Some(header),
        passthrough_blocks,
    })
}

pub fn render(
    document: &SubtitleDocument,
    segments: &[SubtitleSegment],
    bilingual: bool,
) -> CoreResult<String> {
    let mut blocks = Vec::new();

    for index in 0..=segments.len() {
        for block in document
            .passthrough_blocks
            .iter()
            .filter(|block| block.insert_before == index)
        {
            blocks.push(block.content.trim_end().to_owned());
        }

        if index == segments.len() {
            continue;
        }

        let segment = &segments[index];
        let mut cue_lines = Vec::new();
        if let Some(identifier) = segment
            .identifier
            .as_deref()
            .filter(|value| !value.is_empty())
        {
            cue_lines.push(identifier.to_owned());
        }

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
        cue_lines.push(timing_line);

        let text = if bilingual {
            document
                .segments
                .iter()
                .find(|source| source.id == segment.id)
                .map(|source| format!("{}\n{}", source.text, segment.text))
                .unwrap_or_else(|| segment.text.clone())
        } else {
            segment.text.clone()
        };
        if text.is_empty() {
            cue_lines.push(String::new());
        } else {
            cue_lines.extend(text.lines().map(ToOwned::to_owned));
        }

        blocks.push(cue_lines.join("\n").trim_end().to_owned());
    }

    let header = document.header.as_deref().unwrap_or("WEBVTT");
    if blocks.is_empty() {
        Ok(format!("{header}\n"))
    } else {
        Ok(format!("{header}\n\n{}\n", blocks.join("\n\n")))
    }
}

fn parse_cue(block: &str, cue_index: usize) -> Option<SubtitleSegment> {
    let lines = block.lines().map(str::trim_end).collect::<Vec<_>>();
    if lines.is_empty() {
        return None;
    }

    let mut identifier = None;
    let mut timing_index = 0;
    if !lines[0].contains(TIMESTAMP_SEPARATOR) {
        if lines.len() < 2 {
            return None;
        }
        identifier = Some(lines[0].trim().to_owned()).filter(|value| !value.is_empty());
        timing_index = 1;
    }

    let timing = parse_timing_line(lines[timing_index])?;
    let text = lines[timing_index + 1..].join("\n");

    Some(SubtitleSegment {
        id: cue_index.to_string(),
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
