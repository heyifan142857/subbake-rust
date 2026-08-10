use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use crate::entities::{PipelineOptions, SubtitleSegment, TranslationLine};
use crate::error::{CoreError, CoreResult};
use crate::formatting::protect_formatting;
use crate::memory::ContextMemory;
use crate::ports::{CacheStage, ChatMessage};
use crate::review::ReviewBatchPlan;
use crate::storage::{JsonValue, build_request_hash, build_request_hash_v2, stable_hash};
use crate::term_matcher::TermMatcher;

use super::BatchWithUsage;
use super::name_alignment::{protect_names, select_markers as select_name_markers};
use super::online_terminology::{protect_terms, select_markers as select_term_markers};
#[cfg(test)]
use super::translation_stage::{ConfirmedTranslationContext, SourceBatchContext};
use super::translation_stage::{PreparedBatch, TranslationPromptContext};

const REVIEW_ROUTING_CACHE_VERSION: u64 = 2;

pub(super) fn duration_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

pub(super) fn build_translation_messages(
    options: &PipelineOptions,
    batch_index: usize,
    batch: &[SubtitleSegment],
    prompt_context: &TranslationPromptContext,
    memory: &ContextMemory,
    required_glossary: &BTreeMap<String, String>,
    compact_wire: bool,
) -> Vec<ChatMessage> {
    let mut context = serde_json::json!({
        "src": options.source_language,
        "tgt": options.target_language,
        "batch_index": batch_index,
        "mode": options.mode.as_str(),
        "editable_ids": batch.iter().map(|segment| segment.id.as_str()).collect::<Vec<_>>(),
    });
    let batch_texts = batch
        .iter()
        .map(|segment| segment.text.as_str())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>();
    let term_markers = if options.online_terminology {
        select_term_markers(batch, &memory.terminology_candidates)
    } else {
        Vec::new()
    };
    let lightweight_names = options.mode == crate::entities::TranslationMode::Turbo
        && !options.online_terminology
        && !options.preserve_names;
    let name_markers = if lightweight_names {
        select_name_markers(batch, &memory.name_candidates)
    } else {
        Vec::new()
    };
    if options.policy().include_context {
        context["rules"] = serde_json::Value::Array(
            memory
                .style_rules
                .iter()
                .cloned()
                .map(serde_json::Value::String)
                .collect(),
        );
        if !prompt_context.source.before.is_empty() || !prompt_context.source.after.is_empty() {
            context["readonly_source"] = serde_json::json!({
                "before": readonly_source_lines(&prompt_context.source.before),
                "after": readonly_source_lines(&prompt_context.source.after),
            });
        }
        if !prompt_context.previous_confirmed.is_empty() {
            context["confirmed_previous"] = serde_json::Value::Array(
                prompt_context
                    .previous_confirmed
                    .iter()
                    .map(|line| {
                        serde_json::json!({
                            "id": line.id,
                            "source": line.source,
                            "translation": line.translation,
                        })
                    })
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
            let text = protect_formatting(&segment.text);
            let text = protect_terms(&text, &term_markers);
            let text = protect_names(&text, &name_markers);
            if compact_wire {
                serde_json::json!([segment.id, text])
            } else {
                serde_json::json!({"id": segment.id, "text": text})
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
    let terminology_rule = if !name_markers.is_empty() {
        "Tokens shaped like ⟦N<number>⟧ and ⟦/N<number>⟧ mark a possible personal-name span. Translate the text inside normally while copying both markers exactly once and in order. Do not add name markers or return separate terms, glossary_updates, or terminology_updates."
    } else if term_markers.is_empty() {
        "Do not return terms, glossary_updates, or terminology_updates."
    } else {
        "Tokens shaped like ⟦T<number>⟧ and ⟦/T<number>⟧ mark a terminology span. Translate the text inside normally while copying both markers exactly once and in order. Do not return separate terms, glossary_updates, or terminology_updates."
    };
    let system = format!(
        "TASK_START\ntranslate_subtitles\nTASK_END\n\
Return JSON only with this shape:\n\
{response_shape}\n\
Return exactly one line for every input line, in the same order. Copy each id exactly.\n\
Every non-empty source line must have a non-empty translation. Do not include markdown or explanations.\n\
CONTEXT_JSON.editable_ids is the complete set of ids you may return. CONTEXT_JSON.readonly_source and CONTEXT_JSON.confirmed_previous are read-only context; never return or modify their ids.\n\
Tokens shaped like ⟦SBK_FMT_<number>⟧ are protected subtitle formatting markers. Copy each marker exactly once and in the original order.\n\
Entries in CONTEXT_JSON.glossary are user-required translations. Entries in \
CONTEXT_JSON.terminology_hints are automatically learned suggestions: use them \
only when they fit the meaning in the current context.\n{}\n{terminology_rule}",
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

fn readonly_source_lines(segments: &[SubtitleSegment]) -> serde_json::Value {
    serde_json::Value::Array(
        segments
            .iter()
            .map(|segment| serde_json::json!({"id": segment.id, "source": segment.text}))
            .collect(),
    )
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
    let fingerprint = request_backend_fingerprint(options, stage);
    if let Some(fingerprint) = fingerprint {
        build_request_hash_v2(&fingerprint, stage.as_str(), messages)
    } else {
        build_request_hash(&options.provider, &options.model, stage.as_str(), messages)
    }
}

fn request_backend_fingerprint(options: &PipelineOptions, stage: CacheStage) -> Option<String> {
    if stage == CacheStage::Review
        && let Some(reviewer) = &options.reviewer_fingerprint
    {
        return Some(format!(
            "review-routing-v{REVIEW_ROUTING_CACHE_VERSION}:{reviewer}"
        ));
    }
    if matches!(
        stage,
        CacheStage::Review | CacheStage::Terminology | CacheStage::AgentReviewRepair
    ) {
        return options
            .reviewer_fingerprint
            .clone()
            .or_else(|| options.provider_fingerprint.clone());
    }
    options.provider_fingerprint.clone()
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
            if !defer_missing_to_review
                && let Some((term, target)) = TermMatcher::case_insensitive()
                    .missing_required(&segment.text, &line.translation, required_glossary)
                    .into_iter()
                    .next()
            {
                return Err(CoreError::InvalidTranslation(format!(
                    "line {} does not use required glossary translation `{term}` -> `{target}`",
                    segment.id
                )));
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

pub(super) fn contextual_translation_memory_keys(
    scope: &str,
    segments: &[SubtitleSegment],
) -> HashMap<String, String> {
    segments
        .iter()
        .enumerate()
        .map(|(index, segment)| {
            let previous = index
                .checked_sub(1)
                .and_then(|previous| segments.get(previous))
                .map(|segment| segment.text.as_str());
            let next = segments
                .get(index.saturating_add(1))
                .map(|segment| segment.text.as_str());
            (
                segment.id.clone(),
                contextual_translation_memory_key(scope, previous, &segment.text, next),
            )
        })
        .collect()
}

fn contextual_translation_memory_key(
    scope: &str,
    previous: Option<&str>,
    text: &str,
    next: Option<&str>,
) -> String {
    let source = translation_memory_key(text);
    if source.is_empty() {
        return String::new();
    }
    let payload = JsonValue::Array(vec![
        JsonValue::String(scope.to_owned()),
        previous
            .map(translation_memory_key)
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
        JsonValue::String(source.clone()),
        next.map(translation_memory_key)
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
    ]);
    format!("ctx-v1:{}:{source}", stable_hash(&payload))
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
    memory_keys: &HashMap<String, String>,
    source: &[SubtitleSegment],
    translated: &[SubtitleSegment],
) {
    for (source, translated) in source.iter().zip(translated) {
        let key = memory_keys.get(&source.id).cloned().unwrap_or_default();
        if !key.is_empty() && !translated.text.trim().is_empty() {
            memory.insert(key, translated.text.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn review_routing_cache_version_only_salts_an_explicit_reviewer() {
        let mut options = PipelineOptions::new("episode.srt".into());
        options.provider_fingerprint = Some("translator-route".to_owned());

        assert_eq!(
            request_backend_fingerprint(&options, CacheStage::Review).as_deref(),
            Some("translator-route")
        );

        options.reviewer_fingerprint = Some("reviewer-route".to_owned());
        assert_eq!(
            request_backend_fingerprint(&options, CacheStage::Review).as_deref(),
            Some("review-routing-v2:reviewer-route")
        );
        assert_eq!(
            request_backend_fingerprint(&options, CacheStage::Terminology).as_deref(),
            Some("reviewer-route")
        );
        assert_eq!(
            request_backend_fingerprint(&options, CacheStage::AgentReviewRepair).as_deref(),
            Some("reviewer-route")
        );
        assert_eq!(
            request_backend_fingerprint(&options, CacheStage::Translate).as_deref(),
            Some("translator-route")
        );
    }

    #[test]
    fn contextual_tm_keys_distinguish_neighbors_and_project_scope() {
        let first = vec![
            segment("1", "He paid the fee."),
            segment("2", "Fine."),
            segment("3", "We can leave."),
        ];
        let second = vec![
            segment("1", "The weather is clear."),
            segment("2", "Fine."),
            segment("3", "We should stay."),
        ];

        let first_keys = contextual_translation_memory_keys("/shows/one", &first);
        let second_keys = contextual_translation_memory_keys("/shows/one", &second);
        let other_project = contextual_translation_memory_keys("/shows/two", &first);

        assert!(first_keys["2"].starts_with("ctx-v1:"));
        assert_ne!(first_keys["2"], second_keys["2"]);
        assert_ne!(first_keys["2"], other_project["2"]);
    }

    #[test]
    fn translation_prompt_makes_the_name_policy_explicit() {
        let mut options = PipelineOptions::new("episode.srt".into());
        let mut memory = ContextMemory::default();
        memory.name_candidates.push("Mary".to_owned());
        let batch = [SubtitleSegment {
            id: "1".to_owned(),
            text: "Hi Mary.".to_owned(),
            start: None,
            end: None,
            identifier: None,
            settings: None,
        }];

        let transliterated = build_translation_messages(
            &options,
            0,
            &batch,
            &TranslationPromptContext::default(),
            &memory,
            &BTreeMap::new(),
            false,
        );
        assert!(
            transliterated[0]
                .content
                .contains("Do not leave a personal name unchanged")
        );
        assert!(transliterated[0].content.contains("terminology_updates"));
        assert!(transliterated[0].content.contains("⟦N<number>⟧"));
        assert!(transliterated[1].content.contains("⟦N0⟧Mary⟦/N0⟧"));

        options.preserve_names = true;
        let preserved = build_translation_messages(
            &options,
            0,
            &batch,
            &TranslationPromptContext::default(),
            &memory,
            &BTreeMap::new(),
            false,
        );
        assert!(preserved[0].content.contains("source spelling"));
        assert!(!preserved[0].content.contains("⟦N<number>⟧"));

        options.online_terminology = false;
        let minimal = build_translation_messages(
            &options,
            0,
            &batch,
            &TranslationPromptContext::default(),
            &memory,
            &BTreeMap::new(),
            false,
        );
        assert!(
            minimal[0]
                .content
                .contains("Do not return terms, glossary_updates, or terminology_updates")
        );

        options.preserve_names = false;
        options.online_terminology = true;
        let comprehensive = build_translation_messages(
            &options,
            0,
            &batch,
            &TranslationPromptContext::default(),
            &memory,
            &BTreeMap::new(),
            false,
        );
        assert!(!comprehensive[0].content.contains("⟦N<number>⟧"));
    }

    #[test]
    fn translation_prompt_separates_editable_and_read_only_context() {
        let options = PipelineOptions::new("episode.srt".into());
        let mut memory = ContextMemory::default();
        memory.update("legacy model summary", &[]);
        let batch = [segment("2", "Who is she?")];
        let prompt_context = TranslationPromptContext {
            source: SourceBatchContext {
                before: vec![segment("1", "A woman enters.")],
                after: vec![segment("3", "Her name is Mary.")],
            },
            previous_confirmed: vec![ConfirmedTranslationContext {
                id: "1".to_owned(),
                source: "A woman enters.".to_owned(),
                translation: "一位女士走了进来。".to_owned(),
            }],
        };

        let messages = build_translation_messages(
            &options,
            2,
            &batch,
            &prompt_context,
            &memory,
            &BTreeMap::new(),
            false,
        );
        let prompt = &messages[1].content;
        let context = prompt
            .split("CONTEXT_JSON_START")
            .nth(1)
            .and_then(|value| value.split("CONTEXT_JSON_END").next())
            .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
            .expect("context json");
        let payload = prompt
            .split("BATCH_JSON_START")
            .nth(1)
            .and_then(|value| value.split("BATCH_JSON_END").next())
            .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
            .expect("batch json");

        assert_eq!(context["editable_ids"], serde_json::json!(["2"]));
        assert_eq!(context["readonly_source"]["before"][0]["id"], "1");
        assert_eq!(context["readonly_source"]["after"][0]["id"], "3");
        assert_eq!(
            context["confirmed_previous"][0]["translation"],
            "一位女士走了进来。"
        );
        assert!(context.get("recent").is_none());
        assert_eq!(payload["lines"].as_array().map(Vec::len), Some(1));
        assert_eq!(payload["lines"][0]["id"], "2");
        assert!(messages[0].content.contains("read-only context"));
    }

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
}
