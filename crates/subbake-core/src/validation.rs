use std::collections::HashSet;

use crate::entities::{SubtitleSegment, TranslationLine};
use crate::error::{CoreError, CoreResult};

pub fn validate_translation_batch(
    source: &[SubtitleSegment],
    lines: &[TranslationLine],
) -> CoreResult<()> {
    if source.len() != lines.len() {
        return Err(CoreError::InvalidTranslation(format!(
            "expected {} translated line(s), got {}",
            source.len(),
            lines.len()
        )));
    }

    let source_ids = source
        .iter()
        .map(|segment| segment.id.as_str())
        .collect::<HashSet<_>>();
    for line in lines {
        if !source_ids.contains(line.id.as_str()) {
            return Err(CoreError::InvalidTranslation(format!(
                "unexpected translated id `{}`",
                line.id
            )));
        }
    }

    for segment in source {
        if segment.text.trim().is_empty() {
            continue;
        }
        let translation = lines
            .iter()
            .find(|line| line.id == segment.id)
            .ok_or_else(|| CoreError::InvalidTranslation(format!("missing id `{}`", segment.id)))?;
        if translation.translation.trim().is_empty() {
            return Err(CoreError::InvalidTranslation(format!(
                "empty translation for id `{}`",
                segment.id
            )));
        }
    }

    Ok(())
}

pub fn validate_full_alignment(
    source: &[SubtitleSegment],
    translated: &[SubtitleSegment],
) -> CoreResult<()> {
    if source.len() != translated.len() {
        return Err(CoreError::InvalidTranslation(format!(
            "source has {} segment(s), translated has {}",
            source.len(),
            translated.len()
        )));
    }

    for (source, translated) in source.iter().zip(translated) {
        if source.id != translated.id {
            return Err(CoreError::InvalidTranslation(format!(
                "id mismatch: expected `{}`, got `{}`",
                source.id, translated.id
            )));
        }
    }

    Ok(())
}
