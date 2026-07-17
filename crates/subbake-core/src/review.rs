use std::collections::{BTreeMap, BTreeSet};

use crate::entities::{PipelineOptions, ReviewResult, SubtitleSegment, TranslationLine};
use crate::error::{CoreError, CoreResult};
use crate::memory::ContextMemory;
use crate::ports::ChatMessage;
use crate::validation::validate_full_alignment;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReviewBatchPlan {
    pub(crate) batch_index: usize,
    pub(crate) start_offset: usize,
    pub(crate) source: Vec<SubtitleSegment>,
    pub(crate) translated: Vec<SubtitleSegment>,
    pub(crate) reasons: Vec<String>,
    pub(crate) candidate_reasons: BTreeMap<String, Vec<String>>,
}

pub(crate) fn build_review_plan(
    batches: &[Vec<SubtitleSegment>],
    translated_segments: &[SubtitleSegment],
    memory: &ContextMemory,
    source_language: &str,
    target_language: &str,
) -> Vec<ReviewBatchPlan> {
    let mut translations_by_source: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (source, translated) in batches.iter().flatten().zip(translated_segments) {
        translations_by_source
            .entry(normalize_text(&source.text))
            .or_default()
            .insert(normalize_text(&translated.text));
    }

    let mut plan = Vec::new();
    let mut offset = 0usize;
    for (batch_index, source) in batches.iter().enumerate() {
        let end = offset + source.len();
        let translated = &translated_segments[offset..end];
        let candidate_reasons = source
            .iter()
            .zip(translated)
            .filter_map(|(source, translated)| {
                let reasons = line_review_reasons(
                    source,
                    translated,
                    memory,
                    &translations_by_source,
                    !source_language.eq_ignore_ascii_case(target_language),
                );
                (!reasons.is_empty()).then(|| (source.id.clone(), reasons))
            })
            .collect::<BTreeMap<_, _>>();
        if !candidate_reasons.is_empty() {
            let reasons = candidate_reasons
                .values()
                .flatten()
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            plan.push(ReviewBatchPlan {
                batch_index: batch_index + 1,
                start_offset: offset,
                source: source.clone(),
                translated: translated.to_vec(),
                reasons,
                candidate_reasons,
            });
        }
        offset = end;
    }
    plan
}

pub(crate) fn build_full_review_plan(
    batches: &[Vec<SubtitleSegment>],
    translated_segments: &[SubtitleSegment],
) -> Vec<ReviewBatchPlan> {
    let mut offset = 0;
    batches
        .iter()
        .enumerate()
        .map(|(batch_index, source)| {
            let end = offset + source.len();
            let plan = ReviewBatchPlan {
                batch_index: batch_index + 1,
                start_offset: offset,
                source: source.clone(),
                translated: translated_segments[offset..end].to_vec(),
                reasons: vec!["full review".to_owned()],
                candidate_reasons: source
                    .iter()
                    .map(|segment| (segment.id.clone(), vec!["full review".to_owned()]))
                    .collect(),
            };
            offset = end;
            plan
        })
        .collect()
}

pub(crate) fn build_review_messages(
    options: &PipelineOptions,
    source: &[SubtitleSegment],
    translated: &[SubtitleSegment],
    reasons: &[String],
    candidate_reasons: &BTreeMap<String, Vec<String>>,
    memory: &ContextMemory,
) -> Vec<ChatMessage> {
    let texts = source
        .iter()
        .chain(translated)
        .map(|segment| segment.text.as_str())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>();
    let glossary = memory.select_relevant_glossary(&texts);
    let mut payload = serde_json::json!({
        "tgt": options.target_language,
        "reasons": reasons,
        "expected_count": candidate_reasons.len(),
        "expected_ids": candidate_reasons.keys().collect::<Vec<_>>(),
        "lines": source.iter().zip(translated)
            .filter(|(source, _)| candidate_reasons.contains_key(&source.id))
            .map(|(source, translated)| serde_json::json!({
                "id": source.id,
                "source": source.text,
                "translation": translated.text,
                "reasons": candidate_reasons.get(&source.id),
            })).collect::<Vec<_>>(),
        "context": source.iter().zip(translated).map(|(source, translated)| serde_json::json!({
            "id": source.id,
            "source": source.text,
            "translation": translated.text,
            "editable": candidate_reasons.contains_key(&source.id),
        })).collect::<Vec<_>>(),
    });
    let recent = memory.recent_summaries_for_prompt();
    if !recent.is_empty() {
        payload["recent"] = serde_json::json!(recent);
    }
    if !glossary.is_empty() {
        payload["glossary"] = serde_json::Value::Object(
            glossary
                .into_iter()
                .map(|(source, target)| (source, serde_json::Value::String(target)))
                .collect(),
        );
    }
    let payload_json = serde_json::to_string(&payload).unwrap_or_default();
    let system = format!(
        "You are performing a targeted subtitle QA review.\n\
         Return valid JSON only.\n\
         Review {} subtitles.\n\
         Only fix the stated deterministic issues without changing entry structure.",
        options.target_language
    );
    let user = format!(
        "TASK_START\nreview_translations\nTASK_END\n\
         Only ids in expected_ids are editable; context is read-only.\n\
         Prefer minimal edits and omit unchanged lines.\n\
         Return JSON only as {{\"changes\":[{{\"id\":\"<id>\",\"translation\":\"<replacement>\"}}]}}.\n\
         Return an empty changes array when every candidate is already good.\n\
         REVIEW_JSON_START{payload_json}REVIEW_JSON_END"
    );
    vec![
        if options.mode == crate::entities::TranslationMode::Cinema {
            ChatMessage::cacheable_system(system)
        } else {
            ChatMessage::system(system)
        },
        ChatMessage::user(user),
    ]
}

pub(crate) fn parse_review_payload(payload: &serde_json::Value) -> CoreResult<ReviewResult> {
    let lines = payload
        .get("changes")
        .or_else(|| payload.get("lines"))
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            CoreError::InvalidTranslation("review response missing lines array".to_owned())
        })?
        .iter()
        .map(|line| TranslationLine {
            id: line["id"].as_str().unwrap_or_default().to_owned(),
            translation: line["translation"].as_str().unwrap_or_default().to_owned(),
        })
        .collect();
    Ok(ReviewResult {
        lines,
        review_notes: payload["review_notes"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
    })
}

pub(crate) fn restore_review_progress(
    plan: &[ReviewBatchPlan],
    completed_batches: usize,
    restored_segments: &[SubtitleSegment],
    output_segments: &mut [SubtitleSegment],
) -> CoreResult<()> {
    let expected_count = plan
        .iter()
        .take(completed_batches)
        .map(|batch| batch.source.len())
        .sum::<usize>();
    if restored_segments.len() != expected_count {
        return Err(CoreError::DataInvariant(format!(
            "resume state expected {expected_count} reviewed segments across {completed_batches} batches, but loaded {}",
            restored_segments.len()
        )));
    }

    let mut restored_offset = 0usize;
    for batch in plan.iter().take(completed_batches) {
        let end = restored_offset + batch.source.len();
        let restored = &restored_segments[restored_offset..end];
        validate_full_alignment(&batch.source, restored)?;
        output_segments[batch.start_offset..batch.start_offset + restored.len()]
            .clone_from_slice(restored);
        restored_offset = end;
    }
    Ok(())
}

fn line_review_reasons(
    source: &SubtitleSegment,
    translated: &SubtitleSegment,
    memory: &ContextMemory,
    translations_by_source: &BTreeMap<String, BTreeSet<String>>,
    cross_language: bool,
) -> Vec<String> {
    let mut reasons = Vec::new();
    let source_lower = source.text.to_lowercase();
    let translated_lower = translated.text.to_lowercase();
    if memory.glossary.iter().any(|(term, target)| {
        source_lower.contains(&term.to_lowercase())
            && !translated_lower.contains(&target.to_lowercase())
    }) {
        reasons.push("glossary mismatch".to_owned());
    }
    if formatting_tokens(&source.text) != formatting_tokens(&translated.text) {
        reasons.push("formatting mismatch".to_owned());
    }
    if number_tokens(&source.text) != number_tokens(&translated.text) {
        reasons.push("number mismatch".to_owned());
    }
    if has_readability_risk(translated) {
        reasons.push("subtitle readability risk".to_owned());
    }
    if translations_by_source
        .get(&normalize_text(&source.text))
        .is_some_and(|translations| translations.len() > 1)
    {
        reasons.push("inconsistent repeated translation".to_owned());
    }
    if cross_language
        && normalize_text(&source.text) == normalize_text(&translated.text)
        && source.text.trim().chars().count() >= 4
    {
        reasons.push("possibly untranslated".to_owned());
    }
    reasons
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn formatting_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for (open, close) in [('<', '>'), ('{', '}')] {
        let mut rest = text;
        while let Some(start) = rest.find(open) {
            let after = &rest[start..];
            let Some(end) = after.find(close) else {
                break;
            };
            tokens.push(after[..=end].to_owned());
            rest = &after[end + close.len_utf8()..];
        }
    }
    tokens.sort();
    tokens
}

fn number_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn has_readability_risk(segment: &SubtitleSegment) -> bool {
    let characters = segment
        .text
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .count();
    if characters > 84 {
        return true;
    }
    let (Some(start), Some(end)) = (segment.start.as_deref(), segment.end.as_deref()) else {
        return false;
    };
    let (Some(start), Some(end)) = (subtitle_timestamp_ms(start), subtitle_timestamp_ms(end))
    else {
        return false;
    };
    let duration_ms = end.saturating_sub(start);
    duration_ms > 0 && characters.saturating_mul(1_000) > duration_ms.saturating_mul(20)
}

fn subtitle_timestamp_ms(value: &str) -> Option<usize> {
    let value = value.trim().replace(',', ".");
    let (clock, milliseconds) = value.rsplit_once('.')?;
    let mut parts = clock.split(':').map(str::parse::<usize>);
    let hours = parts.next()?.ok()?;
    let minutes = parts.next()?.ok()?;
    let seconds = parts.next()?.ok()?;
    if parts.next().is_some() {
        return None;
    }
    let milliseconds = milliseconds.parse::<usize>().ok()?;
    Some((((hours * 60 + minutes) * 60 + seconds) * 1_000) + milliseconds)
}
