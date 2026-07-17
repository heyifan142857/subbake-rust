//! Pure planning and validation for provider-managed asynchronous batches.
//!
//! The domain deliberately only describes the economy translation contract.
//! Uploading JSONL, polling a provider, and writing manifests are adapter work.

use serde::{Deserialize, Serialize};

use crate::entities::{SubtitleDocument, SubtitleSegment, TranslationLine};
use crate::error::{CoreError, CoreResult};
use crate::ports::ChatMessage;
use crate::validation::validate_translation_batch;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OvernightBatch {
    pub custom_id: String,
    pub segment_ids: Vec<String>,
    pub messages: Vec<ChatMessage>,
}

/// Create compact, self-contained economy requests suitable for an external
/// batch queue. Context-dependent modes are intentionally excluded: a remote
/// job can complete out of order and must be safely recoverable from a manifest.
pub fn plan_translation(
    document: &SubtitleDocument,
    source_language: &str,
    target_language: &str,
    batch_size: usize,
    batch_token_budget: usize,
    id_prefix: &str,
) -> CoreResult<Vec<OvernightBatch>> {
    if batch_size == 0 {
        return Err(CoreError::InvalidTranslation(
            "overnight batch size must be greater than zero".to_owned(),
        ));
    }
    let mut batches = Vec::new();
    let mut current = Vec::new();
    let mut token_count = 0usize;
    for segment in &document.segments {
        let estimate = estimated_text_tokens(&segment.text).saturating_add(8);
        if !current.is_empty()
            && (current.len() >= batch_size
                || (batch_token_budget > 0
                    && token_count.saturating_add(estimate) > batch_token_budget))
        {
            batches.push(build_batch(
                batches.len() + 1,
                &current,
                source_language,
                target_language,
                id_prefix,
            ));
            current.clear();
            token_count = 0;
        }
        current.push(segment.clone());
        token_count = token_count.saturating_add(estimate);
    }
    if !current.is_empty() {
        batches.push(build_batch(
            batches.len() + 1,
            &current,
            source_language,
            target_language,
            id_prefix,
        ));
    }
    Ok(batches)
}

pub fn parse_translation_output(
    batch: &OvernightBatch,
    response: &serde_json::Value,
) -> CoreResult<Vec<TranslationLine>> {
    let lines = response["lines"]
        .as_array()
        .ok_or_else(|| CoreError::InvalidTranslation("response missing lines array".to_owned()))?
        .iter()
        .enumerate()
        .map(|(index, line)| {
            let id = line["id"].as_str().ok_or_else(|| {
                CoreError::InvalidTranslation(format!(
                    "line {} is missing string field `id`",
                    index + 1
                ))
            })?;
            let translation = ["translation", "translated_text", "text"]
                .iter()
                .find_map(|field| line[*field].as_str())
                .ok_or_else(|| {
                    CoreError::InvalidTranslation(format!(
                        "translation for id `{id}` is missing string field `translation`"
                    ))
                })?;
            Ok(TranslationLine {
                id: id.to_owned(),
                translation: translation.to_owned(),
            })
        })
        .collect::<CoreResult<Vec<_>>>()?;
    let source = batch
        .segment_ids
        .iter()
        .map(|id| SubtitleSegment {
            id: id.clone(),
            text: String::new(),
            start: None,
            end: None,
            identifier: None,
            settings: None,
        })
        .collect::<Vec<_>>();
    validate_translation_batch(&source, &lines)?;
    Ok(lines)
}

fn build_batch(
    index: usize,
    segments: &[SubtitleSegment],
    source_language: &str,
    target_language: &str,
    id_prefix: &str,
) -> OvernightBatch {
    let system = format!(
        "You translate subtitles from {source_language} to {target_language}. Preserve meaning, speaker labels, line breaks, and subtitle-safe brevity. Return JSON only: {{\"lines\":[{{\"id\":\"...\",\"translation\":\"...\"}}]}}. Return every id exactly once; do not add commentary."
    );
    let lines = segments
        .iter()
        .map(|segment| serde_json::json!({"id": segment.id, "text": segment.text}))
        .collect::<Vec<_>>();
    OvernightBatch {
        custom_id: format!("{id_prefix}-{index:05}"),
        segment_ids: segments.iter().map(|segment| segment.id.clone()).collect(),
        messages: vec![
            ChatMessage::cacheable_system(system),
            ChatMessage::user(serde_json::json!({"lines": lines}).to_string()),
        ],
    }
}

fn estimated_text_tokens(text: &str) -> usize {
    let (ascii, non_ascii) = text.chars().fold((0usize, 0usize), |(ascii, other), ch| {
        if ch.is_ascii() {
            (ascii + 1, other)
        } else {
            (ascii, other + 1)
        }
    });
    ascii.div_ceil(4).saturating_add(non_ascii)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn document() -> SubtitleDocument {
        SubtitleDocument {
            path: PathBuf::from("clip.srt"),
            format: "srt".to_owned(),
            header: None,
            passthrough_blocks: Vec::new(),
            segments: vec![
                SubtitleSegment {
                    id: "1".to_owned(),
                    text: "Hello".to_owned(),
                    start: None,
                    end: None,
                    identifier: None,
                    settings: None,
                },
                SubtitleSegment {
                    id: "2".to_owned(),
                    text: "World".to_owned(),
                    start: None,
                    end: None,
                    identifier: None,
                    settings: None,
                },
            ],
        }
    }

    #[test]
    fn plans_self_contained_economy_batches_and_validates_outputs() {
        let batches =
            plan_translation(&document(), "English", "Chinese", 1, 100, "job").expect("plan");
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].custom_id, "job-00001");
        let output = parse_translation_output(
            &batches[0],
            &serde_json::json!({"lines":[{"id":"1","translation":"你好"}]}),
        )
        .expect("output");
        assert_eq!(output[0].translation, "你好");
    }
}
