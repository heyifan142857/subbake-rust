use std::path::Path;

use crate::entities::{SubtitleDocument, SubtitleSegment};
use crate::error::CoreResult;

pub fn parse(path: &Path, text: &str) -> SubtitleDocument {
    let text = text.trim_start_matches('\u{feff}');
    let segments = text
        .lines()
        .enumerate()
        .map(|(index, line)| SubtitleSegment {
            id: (index + 1).to_string(),
            text: line.to_owned(),
            start: None,
            end: None,
            identifier: None,
            settings: None,
        })
        .collect();

    SubtitleDocument {
        path: path.to_path_buf(),
        format: "txt".to_owned(),
        segments,
        header: None,
        passthrough_blocks: Vec::new(),
    }
}

pub fn render(
    source_segments: &[SubtitleSegment],
    translated_segments: &[SubtitleSegment],
    bilingual: bool,
) -> CoreResult<String> {
    let mut rendered_lines = Vec::new();

    for translated in translated_segments {
        if bilingual
            && let Some(source) = source_segments
                .iter()
                .find(|segment| segment.id == translated.id)
        {
            rendered_lines.push(source.text.clone());
        }
        rendered_lines.push(translated.text.clone());
    }

    if rendered_lines.is_empty() {
        Ok(String::new())
    } else {
        Ok(format!("{}\n", rendered_lines.join("\n")))
    }
}
