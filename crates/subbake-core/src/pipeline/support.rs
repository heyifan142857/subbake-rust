use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use crate::entities::{PipelineOptions, SubtitleSegment, TranslationLine};
use crate::error::{CoreError, CoreResult};
use crate::memory::ContextMemory;
use crate::ports::{CacheStage, ChatMessage};
use crate::review::ReviewBatchPlan;
use crate::storage::{JsonValue, build_request_hash, build_request_hash_v2};

use super::BatchWithUsage;
use super::translation_stage::PreparedBatch;

pub(super) fn duration_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

pub(super) fn build_translation_messages(
    options: &PipelineOptions,
    batch_index: usize,
    batch: &[SubtitleSegment],
    memory: &ContextMemory,
    required_glossary: &BTreeMap<String, String>,
    compact_wire: bool,
) -> Vec<ChatMessage> {
    let mut context = serde_json::json!({
        "src": options.source_language,
        "tgt": options.target_language,
        "batch_index": batch_index,
        "mode": options.mode.as_str(),
    });
    let batch_texts = batch
        .iter()
        .map(|segment| segment.text.as_str())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>();
    if options.policy().include_context {
        context["rules"] = serde_json::Value::Array(
            memory
                .style_rules
                .iter()
                .cloned()
                .map(serde_json::Value::String)
                .collect(),
        );
        let recent = memory.recent_summaries_for_prompt();
        if !recent.is_empty() {
            context["recent"] = serde_json::Value::Array(
                recent
                    .iter()
                    .cloned()
                    .map(serde_json::Value::String)
                    .collect(),
            );
        }

        let glossary = memory.select_relevant_glossary(&batch_texts);
        if !glossary.is_empty() {
            let mut required = serde_json::Map::new();
            let mut advisory = serde_json::Map::new();
            for (source, target) in glossary {
                let entry = (source.clone(), serde_json::Value::String(target));
                if required_glossary.contains_key(&source) {
                    required.insert(entry.0, entry.1);
                } else {
                    advisory.insert(entry.0, entry.1);
                }
            }
            if !required.is_empty() {
                context["glossary"] = serde_json::Value::Object(required);
            }
            if !advisory.is_empty() {
                context["terminology_hints"] = serde_json::Value::Object(advisory);
            }
        }
    } else if !required_glossary.is_empty() {
        let glossary = memory.select_relevant_glossary(&batch_texts);
        let required = glossary
            .into_iter()
            .filter(|(source, _)| required_glossary.contains_key(source))
            .map(|(source, target)| (source, serde_json::Value::String(target)))
            .collect::<serde_json::Map<_, _>>();
        if !required.is_empty() {
            context["glossary"] = serde_json::Value::Object(required);
        }
    }
    let lines = batch
        .iter()
        .map(|segment| {
            if compact_wire {
                serde_json::json!([segment.id, segment.text])
            } else {
                serde_json::json!({"id": segment.id, "text": segment.text})
            }
        })
        .collect::<Vec<_>>();
    let context_json = serde_json::to_string(&context).unwrap_or_default();
    let batch_json =
        serde_json::to_string(&serde_json::json!({"lines": lines})).unwrap_or_default();
    let response_shape = if compact_wire {
        "{\"lines\":[[\"<source id>\",\"<non-empty target-language text>\"]]}"
    } else {
        "{\"lines\":[{\"id\":\"<source id>\",\"translation\":\"<non-empty target-language text>\"}]}"
    };
    let system = format!(
        "TASK_START\ntranslate_subtitles\nTASK_END\n\
Return JSON only with this shape:\n\
{response_shape}\n\
Return exactly one line for every input line, in the same order. Copy each id exactly.\n\
Every non-empty source line must have a non-empty translation. Do not include markdown or explanations.\n\
Entries in CONTEXT_JSON.glossary are user-required translations. Entries in \
CONTEXT_JSON.terminology_hints are automatically learned suggestions: use them \
only when they fit the meaning in the current context.\n{}",
        if options.preserve_names {
            "Preserve personal names exactly in their source spelling unless CONTEXT_JSON.glossary explicitly requires another form."
        } else {
            "Translate or transliterate every clearly identified personal name into the target language's conventional script and keep it consistent. Do not leave a personal name unchanged merely because it is absent from the glossary."
        }
    );
    vec![
        if options.mode == crate::entities::TranslationMode::Cinema {
            ChatMessage::cacheable_system(system)
        } else {
            ChatMessage::system(system)
        },
        ChatMessage::user(format!(
            "CONTEXT_JSON_START{context_json}CONTEXT_JSON_END\nBATCH_JSON_START{batch_json}BATCH_JSON_END"
        )),
    ]
}

pub(super) fn request_hash(
    options: &PipelineOptions,
    stage: CacheStage,
    messages: &[ChatMessage],
) -> String {
    let messages = JsonValue::Array(
        messages
            .iter()
            .map(|message| {
                JsonValue::Object(vec![
                    ("role".to_owned(), JsonValue::String(message.role.clone())),
                    (
                        "content".to_owned(),
                        JsonValue::String(message.content.clone()),
                    ),
                    ("cacheable".to_owned(), JsonValue::Bool(message.cacheable)),
                ])
            })
            .collect(),
    );
    let reviewer_stage = matches!(
        stage,
        CacheStage::Review | CacheStage::Terminology | CacheStage::AgentReviewRepair
    );
    let fingerprint = if reviewer_stage {
        options
            .reviewer_fingerprint
            .as_ref()
            .or(options.provider_fingerprint.as_ref())
    } else {
        options.provider_fingerprint.as_ref()
    };
    if let Some(fingerprint) = fingerprint {
        build_request_hash_v2(fingerprint, stage.as_str(), messages)
    } else {
        build_request_hash(&options.provider, &options.model, stage.as_str(), messages)
    }
}

pub(super) fn is_agent_repairable(error: &CoreError) -> bool {
    match error {
        CoreError::InvalidTranslation(_) => true,
        CoreError::Llm(crate::error::LlmCallError::InvalidResponse(_)) => true,
        CoreError::InvalidBackendResponse(message) => {
            message.contains("invalid JSON in response")
                || message.contains("response JSON object")
                || message.contains("response missing lines array")
        }
        _ => false,
    }
}

pub(super) fn is_operational_llm_failure(error: &CoreError) -> bool {
    matches!(
        error,
        CoreError::Llm(llm_error)
            if !matches!(llm_error, crate::error::LlmCallError::InvalidResponse(_))
    )
}

pub(super) fn merge_review_patch(
    translated: &[SubtitleSegment],
    changes: &[TranslationLine],
) -> CoreResult<Vec<TranslationLine>> {
    let mut replacements = HashMap::new();
    for change in changes {
        if change.translation.trim().is_empty()
            || replacements
                .insert(&change.id, &change.translation)
                .is_some()
        {
            return Err(CoreError::InvalidTranslation(format!(
                "review patch contains an empty or duplicate change for `{}`",
                change.id
            )));
        }
    }
    if replacements
        .keys()
        .any(|id| !translated.iter().any(|segment| segment.id == ***id))
    {
        return Err(CoreError::InvalidTranslation(
            "review patch contains an unknown id".to_owned(),
        ));
    }
    Ok(translated
        .iter()
        .map(|segment| TranslationLine {
            id: segment.id.clone(),
            translation: replacements.get(&segment.id).map_or_else(
                || segment.text.clone(),
                |translation| (*translation).clone(),
            ),
        })
        .collect())
}

pub(super) fn validate_review_candidate_ids(
    batch: &ReviewBatchPlan,
    changes: &[TranslationLine],
) -> CoreResult<()> {
    if let Some(line) = changes
        .iter()
        .find(|line| !batch.candidate_reasons.contains_key(&line.id))
    {
        return Err(CoreError::InvalidTranslation(format!(
            "review attempted to modify non-candidate id `{}`",
            line.id
        )));
    }
    Ok(())
}

pub(super) fn validate_window_terminology(
    prepared: &[PreparedBatch],
    generated: &HashMap<usize, BatchWithUsage>,
    required_glossary: &BTreeMap<String, String>,
    defer_missing_to_review: bool,
) -> CoreResult<()> {
    for batch in prepared {
        let Some(result) = generated.get(&(batch.index + 1)) else {
            continue;
        };
        for (segment, line) in batch.pending.iter().zip(&result.lines) {
            let source_lower = segment.text.to_lowercase();
            let translation_lower = line.translation.to_lowercase();
            for (term, target) in required_glossary {
                if source_lower.contains(&term.to_lowercase())
                    && !translation_lower.contains(&target.to_lowercase())
                    && !defer_missing_to_review
                {
                    return Err(CoreError::InvalidTranslation(format!(
                        "line {} does not use required glossary translation `{term}` -> `{target}`",
                        segment.id
                    )));
                }
            }
        }
    }
    Ok(())
}

pub(super) fn apply_lines(
    source: &[SubtitleSegment],
    lines: &[TranslationLine],
) -> Vec<SubtitleSegment> {
    source
        .iter()
        .map(|segment| {
            let translation = lines
                .iter()
                .find(|line| line.id == segment.id)
                .map(|line| line.translation.clone())
                .unwrap_or_default();
            let mut translated = segment.clone();
            translated.text = translation;
            translated
        })
        .collect()
}

pub fn translation_memory_key(text: &str) -> String {
    let lower = text.trim().to_lowercase();
    let mut collapsed = String::with_capacity(lower.len());
    let mut prev_was_space = false;
    for ch in lower.chars() {
        if ch.is_whitespace() {
            if !prev_was_space {
                collapsed.push(' ');
                prev_was_space = true;
            }
        } else {
            collapsed.push(ch);
            prev_was_space = false;
        }
    }
    let mut attached = String::with_capacity(collapsed.len());
    let mut chars = collapsed.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == ' '
            && chars
                .peek()
                .is_some_and(|&next| matches!(next, ',' | '.' | '!' | '?' | ';' | ':'))
        {
            continue;
        }
        attached.push(ch);
    }
    attached
}

pub(super) fn merge_translation_lines(
    batch: &[SubtitleSegment],
    tm_hits: &HashMap<String, String>,
    new_lines: &[TranslationLine],
) -> Vec<TranslationLine> {
    batch
        .iter()
        .map(|segment| {
            if let Some(translation) = tm_hits.get(&segment.id) {
                TranslationLine {
                    id: segment.id.clone(),
                    translation: translation.clone(),
                }
            } else {
                new_lines
                    .iter()
                    .find(|line| line.id == segment.id)
                    .cloned()
                    .unwrap_or_else(|| TranslationLine {
                        id: segment.id.clone(),
                        translation: String::new(),
                    })
            }
        })
        .collect()
}

pub(super) fn update_translation_memory(
    memory: &mut HashMap<String, String>,
    source: &[SubtitleSegment],
    translated: &[SubtitleSegment],
) {
    for (source, translated) in source.iter().zip(translated) {
        let key = translation_memory_key(&source.text);
        if !key.is_empty() && !translated.text.trim().is_empty() {
            memory.insert(key, translated.text.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translation_prompt_makes_the_name_policy_explicit() {
        let mut options = PipelineOptions::new("episode.srt".into());
        let memory = ContextMemory::default();

        let transliterated =
            build_translation_messages(&options, 0, &[], &memory, &BTreeMap::new(), false);
        assert!(
            transliterated[0]
                .content
                .contains("Do not leave a personal name unchanged")
        );

        options.preserve_names = true;
        let preserved =
            build_translation_messages(&options, 0, &[], &memory, &BTreeMap::new(), false);
        assert!(preserved[0].content.contains("source spelling"));
    }
}
