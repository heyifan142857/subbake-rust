use std::collections::{BTreeMap, BTreeSet};

use crate::entities::{PipelineOptions, ReviewResult, SubtitleSegment, TranslationLine};
use crate::error::{CoreError, CoreResult};
use crate::memory::ContextMemory;
use crate::ports::ChatMessage;
use crate::term_matcher::TermMatcher;
use crate::validation::validate_full_alignment;

const REVIEW_CONTEXT_LINES: usize = 3;
const MAX_EXTERNAL_CONSISTENCY_OCCURRENCES: usize = 6;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReviewContextLine {
    pub(crate) source: SubtitleSegment,
    pub(crate) translated: SubtitleSegment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReviewConsistencyOccurrence {
    pub(crate) source: SubtitleSegment,
    pub(crate) translated: SubtitleSegment,
    pub(crate) previous: Option<SubtitleSegment>,
    pub(crate) next: Option<SubtitleSegment>,
    pub(crate) editable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReviewConsistencyGroup {
    pub(crate) source_key: String,
    pub(crate) occurrences: Vec<ReviewConsistencyOccurrence>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReviewBatchPlan {
    pub(crate) batch_index: usize,
    pub(crate) start_offset: usize,
    pub(crate) source: Vec<SubtitleSegment>,
    pub(crate) translated: Vec<SubtitleSegment>,
    pub(crate) reasons: Vec<String>,
    pub(crate) candidate_reasons: BTreeMap<String, Vec<String>>,
    pub(crate) before: Vec<ReviewContextLine>,
    pub(crate) after: Vec<ReviewContextLine>,
    pub(crate) consistency_groups: Vec<ReviewConsistencyGroup>,
}

pub(crate) fn build_review_plan(
    batches: &[Vec<SubtitleSegment>],
    translated_segments: &[SubtitleSegment],
    memory: &ContextMemory,
    source_language: &str,
    target_language: &str,
) -> Vec<ReviewBatchPlan> {
    let all_source = batches.iter().flatten().cloned().collect::<Vec<_>>();
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
        let consistency_groups = build_consistency_groups(&all_source, translated_segments, source);
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
                before: review_context(
                    &all_source,
                    translated_segments,
                    offset.saturating_sub(REVIEW_CONTEXT_LINES),
                    offset,
                ),
                after: review_context(
                    &all_source,
                    translated_segments,
                    end,
                    end.saturating_add(REVIEW_CONTEXT_LINES)
                        .min(all_source.len()),
                ),
                consistency_groups,
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
    let all_source = batches.iter().flatten().cloned().collect::<Vec<_>>();
    let mut offset = 0;
    batches
        .iter()
        .enumerate()
        .map(|(batch_index, source)| {
            let end = offset + source.len();
            let consistency_groups =
                build_consistency_groups(&all_source, translated_segments, source);
            let consistency_ids = consistency_groups
                .iter()
                .flat_map(|group| &group.occurrences)
                .filter(|occurrence| occurrence.editable)
                .map(|occurrence| occurrence.source.id.as_str())
                .collect::<BTreeSet<_>>();
            let plan = ReviewBatchPlan {
                batch_index: batch_index + 1,
                start_offset: offset,
                source: source.clone(),
                translated: translated_segments[offset..end].to_vec(),
                reasons: vec!["full review".to_owned()],
                candidate_reasons: source
                    .iter()
                    .map(|segment| {
                        let mut reasons = vec!["full review".to_owned()];
                        if consistency_ids.contains(segment.id.as_str()) {
                            reasons.push("repeated source consistency".to_owned());
                        }
                        (segment.id.clone(), reasons)
                    })
                    .collect(),
                before: review_context(
                    &all_source,
                    translated_segments,
                    offset.saturating_sub(REVIEW_CONTEXT_LINES),
                    offset,
                ),
                after: review_context(
                    &all_source,
                    translated_segments,
                    end,
                    end.saturating_add(REVIEW_CONTEXT_LINES)
                        .min(all_source.len()),
                ),
                consistency_groups,
            };
            offset = end;
            plan
        })
        .collect()
}

fn build_consistency_groups(
    source: &[SubtitleSegment],
    translated: &[SubtitleSegment],
    editable: &[SubtitleSegment],
) -> Vec<ReviewConsistencyGroup> {
    let editable_ids = editable
        .iter()
        .map(|segment| segment.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut indices_by_source = BTreeMap::<String, Vec<usize>>::new();
    for (index, segment) in source.iter().enumerate() {
        let key = normalize_text(&segment.text);
        if !key.is_empty() {
            indices_by_source.entry(key).or_default().push(index);
        }
    }

    indices_by_source
        .into_iter()
        .filter(|(_, indices)| {
            indices.len() > 1
                && indices
                    .iter()
                    .any(|index| editable_ids.contains(source[*index].id.as_str()))
        })
        .map(|(source_key, indices)| {
            let mut selected = indices
                .iter()
                .copied()
                .filter(|index| editable_ids.contains(source[*index].id.as_str()))
                .collect::<Vec<_>>();
            let external = indices
                .into_iter()
                .filter(|index| !editable_ids.contains(source[*index].id.as_str()))
                .collect::<Vec<_>>();
            let mut selected_external = BTreeSet::new();
            let mut seen_translations = BTreeSet::new();
            for &index in &external {
                if seen_translations.insert(normalize_text(&translated[index].text))
                    && selected_external.len() < MAX_EXTERNAL_CONSISTENCY_OCCURRENCES
                {
                    selected_external.insert(index);
                }
            }
            let mut external_signatures = BTreeSet::new();
            for index in external {
                let signature = consistency_occurrence_signature(source, translated, index);
                if external_signatures.insert(signature)
                    && selected_external.len() < MAX_EXTERNAL_CONSISTENCY_OCCURRENCES
                {
                    selected_external.insert(index);
                }
            }
            selected.extend(selected_external);
            selected.sort_unstable();
            ReviewConsistencyGroup {
                source_key,
                occurrences: selected
                    .into_iter()
                    .map(|index| ReviewConsistencyOccurrence {
                        source: source[index].clone(),
                        translated: translated[index].clone(),
                        previous: index
                            .checked_sub(1)
                            .and_then(|previous| source.get(previous))
                            .cloned(),
                        next: source.get(index.saturating_add(1)).cloned(),
                        editable: editable_ids.contains(source[index].id.as_str()),
                    })
                    .collect(),
            }
        })
        .collect()
}

fn consistency_occurrence_signature(
    source: &[SubtitleSegment],
    translated: &[SubtitleSegment],
    index: usize,
) -> Vec<String> {
    let segment = &source[index];
    vec![
        normalize_text(&translated[index].text),
        normalize_text(segment.settings.as_deref().unwrap_or_default()),
        normalize_text(segment.semantic.speaker.as_deref().unwrap_or_default()),
        normalize_text(segment.semantic.style.as_deref().unwrap_or_default()),
        normalize_text(segment.semantic.layer.as_deref().unwrap_or_default()),
        normalize_text(segment.semantic.kind.as_deref().unwrap_or_default()),
        index
            .checked_sub(1)
            .and_then(|previous| source.get(previous))
            .map(|segment| normalize_text(&segment.text))
            .unwrap_or_default(),
        source
            .get(index.saturating_add(1))
            .map(|segment| normalize_text(&segment.text))
            .unwrap_or_default(),
    ]
}

fn review_context(
    source: &[SubtitleSegment],
    translated: &[SubtitleSegment],
    start: usize,
    end: usize,
) -> Vec<ReviewContextLine> {
    source[start..end]
        .iter()
        .cloned()
        .zip(translated[start..end].iter().cloned())
        .map(|(source, translated)| ReviewContextLine { source, translated })
        .collect()
}

pub(crate) fn build_review_messages(
    options: &PipelineOptions,
    batch: &ReviewBatchPlan,
    memory: &ContextMemory,
) -> Vec<ChatMessage> {
    let source = &batch.source;
    let translated = &batch.translated;
    let reasons = &batch.reasons;
    let candidate_reasons = &batch.candidate_reasons;
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
                "start": source.start,
                "end": source.end,
                "identifier": source.identifier,
                "settings": source.settings,
                "semantic": source.semantic,
                "reasons": candidate_reasons.get(&source.id),
            })).collect::<Vec<_>>(),
        "context": source.iter().zip(translated).map(|(source, translated)| serde_json::json!({
            "id": source.id,
            "source": source.text,
            "translation": translated.text,
            "start": source.start,
            "end": source.end,
            "identifier": source.identifier,
            "settings": source.settings,
            "semantic": source.semantic,
            "editable": candidate_reasons.contains_key(&source.id),
        })).collect::<Vec<_>>(),
        "readonly_before": review_context_json(&batch.before),
        "readonly_after": review_context_json(&batch.after),
        "repeated_source_groups": review_consistency_json(&batch.consistency_groups),
    });
    if !glossary.is_empty() {
        payload["glossary"] = serde_json::Value::Object(
            glossary
                .into_iter()
                .map(|(source, target)| (source, serde_json::Value::String(target)))
                .collect(),
        );
    }
    if !memory.terminology_entities.is_empty() {
        payload["terminology_entities"] =
            serde_json::to_value(&memory.terminology_entities).unwrap_or_default();
    }
    let payload_json = serde_json::to_string(&payload).unwrap_or_default();
    let system = format!(
        "You are performing a targeted subtitle QA review.{}\n\
         Return valid JSON only.\n\
         Review {} subtitles.\n\
         Only fix the stated deterministic issues without changing entry structure.",
        if options.mode == crate::entities::TranslationMode::Cinema {
            " First form an independent candidate translation for each editable line, then adjudicate it against the supplied candidate; return only the better final replacement when it materially improves fidelity, naturalness, consistency, or subtitle readability. Repeated source text is a consistency signal, not proof of identical meaning: keep translations identical when semantic metadata and local context are equivalent, but preserve justified differences in speaker, register, subtitle purpose, or scene meaning."
        } else {
            ""
        },
        options.target_language
    );
    let user = format!(
        "TASK_START\nreview_translations\nTASK_END\n\
         Only ids in expected_ids are editable; context is read-only.\n\
         readonly_before and readonly_after are adjacent translated context; never return their ids.\n\
         repeated_source_groups contains same-source occurrences across the document. Occurrences marked editable may be changed only when their id is in expected_ids; every other occurrence is read-only.\n\
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

fn review_context_json(lines: &[ReviewContextLine]) -> serde_json::Value {
    serde_json::Value::Array(
        lines
            .iter()
            .map(|line| {
                serde_json::json!({
                    "id": line.source.id,
                    "source": line.source.text,
                    "translation": line.translated.text,
                    "start": line.source.start,
                    "end": line.source.end,
                    "identifier": line.source.identifier,
                    "settings": line.source.settings,
                    "semantic": line.source.semantic,
                })
            })
            .collect(),
    )
}

fn review_consistency_json(groups: &[ReviewConsistencyGroup]) -> serde_json::Value {
    serde_json::Value::Array(
        groups
            .iter()
            .map(|group| {
                serde_json::json!({
                    "normalized_source": group.source_key,
                    "occurrences": group.occurrences.iter().map(|occurrence| serde_json::json!({
                        "id": occurrence.source.id,
                        "source": occurrence.source.text,
                        "translation": occurrence.translated.text,
                        "start": occurrence.source.start,
                        "end": occurrence.source.end,
                        "identifier": occurrence.source.identifier,
                        "settings": occurrence.source.settings,
                        "semantic": occurrence.source.semantic,
                        "previous_source": occurrence.previous.as_ref().map(|line| &line.text),
                        "next_source": occurrence.next.as_ref().map(|line| &line.text),
                        "editable": occurrence.editable,
                    })).collect::<Vec<_>>(),
                })
            })
            .collect(),
    )
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
    if !TermMatcher::case_insensitive()
        .missing_required(&source.text, &translated.text, &memory.glossary)
        .is_empty()
    {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(id: &str, text: &str, start: &str, end: &str) -> SubtitleSegment {
        SubtitleSegment {
            id: id.to_owned(),
            text: text.to_owned(),
            start: Some(start.to_owned()),
            end: Some(end.to_owned()),
            identifier: Some(format!("speaker-{id}")),
            settings: Some("align:start".to_owned()),
            semantic: Default::default(),
        }
    }

    #[test]
    fn cinema_review_receives_timing_metadata_and_adjacent_readonly_context() {
        let source = [
            segment("1", "one", "00:00:00,000", "00:00:01,000"),
            segment("2", "two", "00:00:02,000", "00:00:03,000"),
            segment("3", "three", "00:00:04,000", "00:00:05,000"),
        ];
        let translated = source
            .iter()
            .map(|line| {
                let mut line = line.clone();
                line.text = format!("translated {}", line.text);
                line
            })
            .collect::<Vec<_>>();
        let batches = source
            .iter()
            .cloned()
            .map(|line| vec![line])
            .collect::<Vec<_>>();
        let plan = build_full_review_plan(&batches, &translated);
        let mut options = PipelineOptions::new("review.srt".into());
        options.mode = crate::entities::TranslationMode::Cinema;

        let messages = build_review_messages(&options, &plan[1], &ContextMemory::default());
        let payload = messages[1]
            .content
            .split("REVIEW_JSON_START")
            .nth(1)
            .and_then(|value| value.split("REVIEW_JSON_END").next())
            .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
            .expect("review payload");

        assert_eq!(payload["lines"][0]["start"], "00:00:02,000");
        assert_eq!(payload["lines"][0]["identifier"], "speaker-2");
        assert_eq!(payload["readonly_before"][0]["id"], "1");
        assert_eq!(payload["readonly_after"][0]["id"], "3");
        assert_eq!(
            payload["readonly_before"][0]["translation"],
            "translated one"
        );
    }

    #[test]
    fn cinema_review_receives_cross_batch_repeated_source_context() {
        let mut spoken = segment("1", "Open", "00:00:00,000", "00:00:01,000");
        spoken.semantic.speaker = Some("Alice".to_owned());
        spoken.semantic.style = Some("Default".to_owned());
        let bridge = segment("2", "Later", "00:00:02,000", "00:00:03,000");
        let mut sign = segment("3", "Open", "00:01:00,000", "00:01:01,000");
        sign.semantic.style = Some("Sign".to_owned());
        let source = vec![spoken, bridge, sign];
        let mut translated = source.clone();
        translated[0].text = "打开".to_owned();
        translated[1].text = "稍后".to_owned();
        translated[2].text = "营业中".to_owned();
        let batches = vec![source[..2].to_vec(), source[2..].to_vec()];
        let plan = build_full_review_plan(&batches, &translated);
        let mut options = PipelineOptions::new("review.ass".into());
        options.mode = crate::entities::TranslationMode::Cinema;

        let messages = build_review_messages(&options, &plan[0], &ContextMemory::default());
        let payload = messages[1]
            .content
            .split("REVIEW_JSON_START")
            .nth(1)
            .and_then(|value| value.split("REVIEW_JSON_END").next())
            .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
            .expect("review payload");
        let group = &payload["repeated_source_groups"][0];

        assert_eq!(group["normalized_source"], "open");
        assert_eq!(group["occurrences"][0]["semantic"]["speaker"], "Alice");
        assert_eq!(group["occurrences"][0]["editable"], true);
        assert_eq!(group["occurrences"][1]["semantic"]["style"], "Sign");
        assert_eq!(group["occurrences"][1]["translation"], "营业中");
        assert_eq!(group["occurrences"][1]["editable"], false);
        assert!(
            messages[0]
                .content
                .contains("not proof of identical meaning")
        );
    }
}
