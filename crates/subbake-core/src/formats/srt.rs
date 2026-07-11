use std::path::Path;

use crate::entities::{SubtitleDocument, SubtitleSegment};
use crate::error::{CoreError, CoreResult};
use crate::formats::split_blocks;

const TIMESTAMP_SEPARATOR: &str = "-->";

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
    })
}

pub fn render(
    source_segments: &[SubtitleSegment],
    segments: &[SubtitleSegment],
    bilingual: bool,
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
                .map(|source| format!("{}\n{}", source.text, segment.text))
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
