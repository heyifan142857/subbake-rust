use serde::{Deserialize, Serialize};

use crate::entities::{SubtitleSegment, TranslationLine};
use crate::error::{CoreError, CoreResult};
use crate::ports::ChatMessage;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubtitleEditPayload {
    pub lines: Vec<TranslationLine>,
    #[serde(default)]
    pub edit_notes: String,
}

pub fn build_subtitle_edit_messages(
    target_segments: &[SubtitleSegment],
    source_segments: Option<&[SubtitleSegment]>,
    instruction: &str,
    target_language: &str,
) -> CoreResult<Vec<ChatMessage>> {
    let mut payload = serde_json::json!({
        "target_language": target_language,
        "instruction": instruction,
        "expected_count": target_segments.len(),
        "expected_ids": target_segments.iter().map(|segment| &segment.id).collect::<Vec<_>>(),
        "lines": target_segments.iter().map(|segment| serde_json::json!({
            "id": segment.id,
            "translation": segment.text,
        })).collect::<Vec<_>>(),
    });
    if let Some(source) = source_segments {
        payload["source_lines"] = serde_json::Value::Array(
            source
                .iter()
                .map(|segment| {
                    serde_json::json!({
                        "id": segment.id,
                        "text": segment.text,
                    })
                })
                .collect(),
        );
    }
    let compact = serde_json::to_string(&payload).map_err(|error| {
        CoreError::InvalidBackendResponse(format!("serialize edit payload: {error}"))
    })?;
    Ok(vec![
        ChatMessage::system(format!(
            "You are SubBake's subtitle editing agent.\n\
             Return valid JSON only.\n\
             Edit {target_language} subtitles according to the user's instruction.\n\
             Do not change subtitle ids, order, count, timings, or file format."
        )),
        ChatMessage::user(format!(
            "TASK_START\nagent_edit_subtitle\nTASK_END\n\
             Apply the requested edit only where needed.\n\
             Use expected_ids as the complete authoritative list and preserve that exact order.\n\
             Keep good lines unchanged. Keep blank lines blank.\n\
             Return JSON only with keys \"lines\" and \"edit_notes\".\n\
             EDIT_JSON_START\n{compact}\nEDIT_JSON_END"
        )),
    ])
}

pub fn parse_subtitle_edit_payload(
    value: serde_json::Value,
    target_segments: &[SubtitleSegment],
) -> CoreResult<SubtitleEditPayload> {
    let payload: SubtitleEditPayload = serde_json::from_value(value).map_err(|error| {
        CoreError::InvalidTranslation(format!("invalid edit response: {error}"))
    })?;
    if payload.lines.len() != target_segments.len() {
        return Err(CoreError::InvalidTranslation(format!(
            "edit expected {} line(s), got {}",
            target_segments.len(),
            payload.lines.len()
        )));
    }
    for (target, line) in target_segments.iter().zip(&payload.lines) {
        if target.id != line.id {
            return Err(CoreError::InvalidTranslation(format!(
                "edit id mismatch: expected `{}`, got `{}`",
                target.id, line.id
            )));
        }
        if !target.text.trim().is_empty() && line.translation.trim().is_empty() {
            return Err(CoreError::InvalidTranslation(format!(
                "empty edited subtitle for id `{}`",
                target.id
            )));
        }
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(id: &str, text: &str) -> SubtitleSegment {
        SubtitleSegment {
            id: id.to_owned(),
            text: text.to_owned(),
            start: None,
            end: None,
            identifier: None,
            settings: None,
        }
    }

    #[test]
    fn edit_prompt_contains_source_context() {
        let messages = build_subtitle_edit_messages(
            &[segment("1", "你好")],
            Some(&[segment("1", "hello")]),
            "make it formal",
            "Chinese",
        )
        .expect("prompt");
        assert!(messages[1].content.contains("\"source_lines\""));
        assert!(messages[1].content.contains("make it formal"));
    }

    #[test]
    fn edit_response_requires_exact_id_order() {
        let error = parse_subtitle_edit_payload(
            serde_json::json!({
                "lines": [
                    {"id": "2", "translation": "B"},
                    {"id": "1", "translation": "A"}
                ]
            }),
            &[segment("1", "a"), segment("2", "b")],
        )
        .expect_err("reordered ids must fail");
        assert!(error.to_string().contains("id mismatch"));
    }
}
