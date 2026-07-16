use crate::entities::{
    AttemptLog, BatchTranslationResult, GlossaryEntry, SubtitleSegment, TranslationLine,
};
use crate::error::{CoreError, CoreResult};
use crate::ports::{BackendPayload, ChatMessage};

pub(crate) fn retry_correction_message(error: &CoreError) -> ChatMessage {
    ChatMessage::user(format!(
        "The previous response failed validation.\nValidation error: {error}\nRe-send corrected JSON only."
    ))
}

pub(crate) fn split_index(batch: &[SubtitleSegment]) -> usize {
    let midpoint = batch.len() / 2;
    (1..batch.len())
        .filter(|index| semantic_boundary(&batch[*index - 1], &batch[*index]))
        .min_by_key(|index| index.abs_diff(midpoint))
        .unwrap_or(midpoint)
}

fn semantic_boundary(current: &SubtitleSegment, next: &SubtitleSegment) -> bool {
    let current = current.text.trim();
    let next = next.text.trim();
    current.is_empty()
        || next.is_empty()
        || has_speaker_marker(current) != has_speaker_marker(next)
        || ends_sentence(current)
}

fn has_speaker_marker(text: &str) -> bool {
    let text = text.trim_start();
    text.starts_with('-')
        || text.starts_with('–')
        || text.starts_with('—')
        || text.starts_with(">>")
        || text.find(':').is_some_and(|colon| {
            let label = &text[..colon];
            let mut chars = label.chars();
            chars.next().is_some_and(|first| first.is_ascii_uppercase())
                && label.chars().count() <= 21
                && chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, ' ' | '_' | '-'))
                && text[colon + 1..]
                    .chars()
                    .next()
                    .is_some_and(char::is_whitespace)
        })
}

fn ends_sentence(text: &str) -> bool {
    text.trim()
        .trim_end_matches(['"', '\'', ')', ']'])
        .ends_with(['.', '!', '?', '。', '！', '？', '…'])
}

pub(crate) fn combine_summaries(left: &str, right: &str, limit: usize) -> String {
    let mut summaries = Vec::new();
    for summary in [left.trim(), right.trim()] {
        if !summary.is_empty() && !summaries.contains(&summary) {
            summaries.push(summary);
        }
    }
    summaries
        .into_iter()
        .take(limit)
        .collect::<Vec<_>>()
        .join(" | ")
}

pub(crate) fn combine_glossary(
    left: Vec<GlossaryEntry>,
    right: Vec<GlossaryEntry>,
) -> Vec<GlossaryEntry> {
    let mut entries = std::collections::BTreeMap::new();
    for entry in left.into_iter().chain(right) {
        entries.insert(entry.source, entry.target);
    }
    entries
        .into_iter()
        .map(|(source, target)| GlossaryEntry { source, target })
        .collect()
}

pub(crate) fn parse_translation_payload(
    payload: &serde_json::Value,
) -> CoreResult<BatchTranslationResult> {
    let lines = payload["lines"]
        .as_array()
        .ok_or_else(|| CoreError::InvalidTranslation("response missing lines array".to_owned()))?
        .iter()
        .enumerate()
        .map(|(index, line)| parse_translation_line(line, index))
        .collect::<CoreResult<Vec<_>>>()?;
    let glossary_updates = match &payload["glossary_updates"] {
        serde_json::Value::Array(entries) => entries
            .iter()
            .map(|entry| GlossaryEntry {
                source: entry["source"].as_str().unwrap_or_default().to_owned(),
                target: entry["target"].as_str().unwrap_or_default().to_owned(),
            })
            .collect(),
        serde_json::Value::Object(entries) => entries
            .iter()
            .map(|(source, target)| GlossaryEntry {
                source: source.clone(),
                target: target.as_str().unwrap_or_default().to_owned(),
            })
            .collect(),
        _ => Vec::new(),
    };
    Ok(BatchTranslationResult {
        lines,
        summary: payload["summary"].as_str().unwrap_or_default().to_owned(),
        glossary_updates,
    })
}

fn parse_translation_line(line: &serde_json::Value, index: usize) -> CoreResult<TranslationLine> {
    let id = line["id"].as_str().ok_or_else(|| {
        CoreError::InvalidTranslation(format!("line {} is missing string field `id`", index + 1))
    })?;
    let translation = ["translation", "translated_text", "text"]
        .into_iter()
        .find_map(|field| line[field].as_str())
        .ok_or_else(|| {
            CoreError::InvalidTranslation(format!(
                "translation for id `{id}` is missing string field `translation`"
            ))
        })?;
    Ok(TranslationLine {
        id: id.to_owned(),
        translation: translation.to_owned(),
    })
}

pub(crate) fn build_agent_repair_messages(
    stage: &str,
    source: &[SubtitleSegment],
    translated: Option<&[SubtitleSegment]>,
    target_language: &str,
    last_error: &CoreError,
    failed_attempts: &[AttemptLog],
    agent_attempts: &[AttemptLog],
) -> Vec<ChatMessage> {
    let task = if stage == "translate" {
        "agent_repair_translation"
    } else {
        "agent_repair_review"
    };
    let return_keys = if stage == "translate" {
        "\"lines\", \"summary\", and \"glossary_updates\""
    } else {
        "\"lines\" and \"review_notes\""
    };
    let mut payload = serde_json::json!({
        "stage": stage,
        "target_language": target_language,
        "last_error": last_error.to_string(),
        "expected_count": source.len(),
        "expected_ids": source.iter().map(|segment| segment.id.as_str()).collect::<Vec<_>>(),
        "source_lines": source.iter().map(|segment| {
            serde_json::json!({"id": segment.id, "text": segment.text})
        }).collect::<Vec<_>>(),
        "failed_attempts": failed_attempts,
        "agent_attempts": agent_attempts,
    });
    if let Some(translated) = translated {
        payload["current_translations"] = serde_json::json!(
            translated
                .iter()
                .map(|segment| {
                    serde_json::json!({"id": segment.id, "translation": segment.text})
                })
                .collect::<Vec<_>>()
        );
    }
    let payload_json = serde_json::to_string(&payload).unwrap_or_default();
    vec![
        ChatMessage::system(format!(
            "You are SubBake's runtime repair agent.\nReturn valid JSON only.\nRepair the failed model output without changing source text, subtitle ids, order, count, runtime config, or files.\nEvery non-empty source entry must produce one non-empty {target_language} translation with the same id."
        )),
        ChatMessage::user(format!(
            "TASK_START\n{task}\nTASK_END\nRead this failure log and return a corrected response for the same batch.\nUse expected_ids as the complete authoritative list and preserve that exact order.\nDo not explain the fix. Do not include markdown.\nReturn JSON only with keys {return_keys}.\nAGENT_REPAIR_JSON_START{payload_json}AGENT_REPAIR_JSON_END"
        )),
    ]
}

pub(crate) fn backend_payload_json(payload: &BackendPayload) -> CoreResult<serde_json::Value> {
    match payload {
        BackendPayload::Translation(result) => serde_json::to_value(result),
        BackendPayload::Review(result) => serde_json::to_value(result),
        BackendPayload::Terminology(result) => serde_json::to_value(result),
    }
    .map_err(|error| CoreError::DataInvariant(format!("serialize backend payload failed: {error}")))
}
