use std::collections::{BTreeMap, BTreeSet};

use crate::entities::{
    PipelineOptions, PromptCacheStrategy, ReviewAnnotation, ReviewIssueKind, ReviewPolicy,
    ReviewResult, ReviewStrategy, SubtitleSegment, TranslationLine,
};
use crate::error::{CoreError, CoreResult};
#[cfg(test)]
use crate::language_rules::LanguageRuleRegistry;
use crate::language_rules::{EnglishRules, ResolvedLanguageRules};
use crate::memory::ContextMemory;
use crate::number_facts::{NumberFactComparison, compare_number_facts};
use crate::ports::ChatMessage;
use crate::term_matcher::TermMatcher;
use crate::validation::{FinalValidationPolicy, final_validation_issues, validate_full_alignment};

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
            .entry(consistency_key(&source.text))
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

#[cfg(test)]
pub(crate) fn build_full_review_plan(
    batches: &[Vec<SubtitleSegment>],
    translated_segments: &[SubtitleSegment],
    memory: &ContextMemory,
    source_language: &str,
    target_language: &str,
) -> Vec<ReviewBatchPlan> {
    let language_rules = LanguageRuleRegistry::resolve(source_language, target_language);
    build_full_review_plan_with_rules(batches, translated_segments, memory, &language_rules)
}

pub(crate) fn build_full_review_plan_with_rules(
    batches: &[Vec<SubtitleSegment>],
    translated_segments: &[SubtitleSegment],
    memory: &ContextMemory,
    language_rules: &ResolvedLanguageRules,
) -> Vec<ReviewBatchPlan> {
    let all_source = batches.iter().flatten().cloned().collect::<Vec<_>>();
    let mut translations_by_source: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (source, translated) in all_source.iter().zip(translated_segments) {
        translations_by_source
            .entry(consistency_key(&source.text))
            .or_default()
            .insert(normalize_text(&translated.text));
    }
    let mut plan = Vec::new();
    let mut offset = 0usize;
    for (batch_index, source) in batches.iter().enumerate() {
        let end = offset + source.len();
        let translated = &translated_segments[offset..end];
        let consistency_groups = build_consistency_groups(&all_source, translated_segments, source);
        let mut candidate_reasons = BTreeMap::new();
        for (local_index, (source, translated)) in source.iter().zip(translated).enumerate() {
            let mut reasons = line_review_reasons(
                source,
                translated,
                memory,
                &translations_by_source,
                language_rules.is_cross_language(),
            );
            reasons.extend(document_guide_review_reasons(
                source,
                translated,
                memory,
                language_rules,
            ));
            let document_index = offset + local_index;
            if has_contextual_pronoun_risk(&all_source, document_index, memory, language_rules) {
                reasons.push("pronoun/coreference continuity".to_owned());
            }
            reasons.sort();
            reasons.dedup();
            if !reasons.is_empty() {
                candidate_reasons.insert(source.id.clone(), reasons);
            }
        }
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
        let key = consistency_key(&segment.text);
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

#[cfg(test)]
pub(crate) fn build_review_messages(
    options: &PipelineOptions,
    batch: &ReviewBatchPlan,
    memory: &ContextMemory,
    required_glossary: &BTreeMap<String, String>,
) -> Vec<ChatMessage> {
    let language_rules =
        LanguageRuleRegistry::resolve(&options.source_language, &options.target_language);
    build_review_messages_with_rules(options, &language_rules, batch, memory, required_glossary)
}

pub(crate) fn build_review_messages_with_rules(
    options: &PipelineOptions,
    language_rules: &ResolvedLanguageRules,
    batch: &ReviewBatchPlan,
    memory: &ContextMemory,
    required_glossary: &BTreeMap<String, String>,
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
    let required_entries = required_glossary.iter().collect::<Vec<_>>();
    let required_terms = required_entries
        .iter()
        .map(|(source, _)| source.as_str())
        .collect::<Vec<_>>();
    let glossary = memory
        .select_relevant_glossary(&texts)
        .into_iter()
        .filter(|(source, _)| {
            TermMatcher::case_insensitive()
                .matching_indices(source, &required_terms)
                .is_empty()
                && required_terms
                    .iter()
                    .all(|required| !TermMatcher::case_insensitive().contains(required, source))
        })
        .collect::<BTreeMap<_, _>>();
    let mut required_indices = BTreeSet::new();
    for segment in source {
        required_indices.extend(
            TermMatcher::case_insensitive().matching_indices(&segment.text, &required_terms),
        );
    }
    let selected_required = required_indices
        .into_iter()
        .filter_map(|index| required_entries.get(index).copied())
        .map(|(source, target)| (source.clone(), target.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut payload = serde_json::json!({
        "tgt": options.target_language,
        "policy": options.review_policy.as_str(),
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
                "deterministic_issues": final_validation_issues(
                    std::slice::from_ref(source),
                    std::slice::from_ref(translated),
                    required_glossary,
                    &options.source_language,
                    &options.target_language,
                    FinalValidationPolicy {
                        max_characters_per_second: options.max_characters_per_second,
                        max_characters_per_line: options.max_characters_per_line,
                        max_lines: options.max_lines,
                    },
                ).unwrap_or_default().into_iter().map(|issue| issue.message).collect::<Vec<_>>(),
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
        payload["terminology_hints"] = serde_json::Value::Object(
            glossary
                .into_iter()
                .map(|(source, target)| (source, serde_json::Value::String(target)))
                .collect(),
        );
    }
    if !selected_required.is_empty() {
        payload["required_glossary"] = serde_json::Value::Object(
            selected_required
                .into_iter()
                .map(|(source, target)| (source, serde_json::Value::String(target)))
                .collect(),
        );
    }
    let guide = memory.select_relevant_document_guide(&texts);
    if !guide.is_empty() {
        payload["document_guide"] = serde_json::to_value(guide).unwrap_or_default();
    }
    let payload_json = serde_json::to_string(&payload).unwrap_or_default();
    let language_guidance = language_rules.review_guidance().unwrap_or_default();
    let (system, response_shape, instructions) = match options.review_policy {
        ReviewPolicy::Full => (
            format!(
                "You are SubBake's document-level targeted subtitle revision adjudicator.\n\
                 Return valid JSON only.\n\
                 Review only the listed risk candidates in {} subtitles. The complete translation was scanned before this request; non-candidate context is read-only. Check meaning accuracy, omissions, terminology, cross-document consistency, pronouns, forms of address, speaker register, natural fluency, and subtitle readability.\n\
                 First resolve every deterministic_issues entry. Treat required_glossary values as mandatory exact targets; terminology_hints are advisory and never override them.\n\
                 Treat document_guide as frozen evidence, but do not force advisory character or terminology guidance when local meaning clearly differs. Preserve facts, numbers, formatting markers, required terminology, tone, humor, emotion, profanity, and intentionally incomplete phrasing across cue boundaries.\n\
                 Do not replace a sound translation with a merely different synonym.{}\n{language_guidance}",
                options.target_language,
                if options.policy().review_strategy == ReviewStrategy::Adjudicated {
                    " Independently determine the intended translation for each editable line before comparing it with the supplied candidate, then keep whichever is materially better."
                } else {
                    ""
                }
            ),
            "{\"changes\":[{\"id\":\"<id>\",\"translation\":\"<replacement>\",\"category\":\"<category>\",\"rationale\":\"<short reason>\"}]}",
            "Return a change for every listed deterministic issue. For other candidate reasons, change a line only when the supplied document evidence confirms a material problem. category must be one of accuracy, omission, terminology, consistency, register, fluency, or readability. rationale must be concise and specific. Omit candidates that do not need a change.",
        ),
        ReviewPolicy::Targeted | ReviewPolicy::Off => (
            format!(
                "You are SubBake's targeted deterministic subtitle repair reviewer.\n\
                 Return valid JSON only.\n\
                 Repair only the stated issues in {} subtitles without changing entry structure or unrelated wording. Ambiguous number-expression candidates require verification, not automatic rewriting; preserve idioms and lexicalized expressions when they do not state a numeric fact.",
                options.target_language
            ),
            "{\"changes\":[{\"id\":\"<id>\",\"translation\":\"<replacement>\"}]}",
            "Make the smallest change that fixes each stated candidate reason. Omit candidates that already satisfy the stated requirement.",
        ),
    };
    let user = format!(
        "TASK_START\nreview_translations\nTASK_END\n\
         Only ids in expected_ids are editable; all other content is read-only.\n\
         readonly_before and readonly_after are adjacent translated context; never return their ids.\n\
         repeated_source_groups contains same-source occurrences across the document. Repeated text is a consistency signal, not proof of identical meaning: preserve justified differences in speaker, register, subtitle purpose, or scene meaning.\n\
         {instructions}\n\
         Return JSON only as {response_shape}.\n\
         Return an empty changes array when no candidate needs replacement.\n\
         REVIEW_JSON_START{payload_json}REVIEW_JSON_END"
    );
    vec![
        if options.policy().prompt_cache_strategy == PromptCacheStrategy::CacheableSystem {
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

pub(crate) fn parse_review_payload(
    payload: &serde_json::Value,
    require_annotations: bool,
) -> CoreResult<ReviewResult> {
    let raw_lines = payload
        .get("changes")
        .or_else(|| payload.get("lines"))
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            CoreError::InvalidTranslation("review response missing lines array".to_owned())
        })?;
    let mut lines = Vec::with_capacity(raw_lines.len());
    let mut annotations = Vec::with_capacity(raw_lines.len());
    for line in raw_lines {
        let id = line["id"].as_str().unwrap_or_default().to_owned();
        lines.push(TranslationLine {
            id: id.clone(),
            translation: line["translation"].as_str().unwrap_or_default().to_owned(),
        });
        let category = line.get("category");
        let rationale = line.get("rationale").and_then(serde_json::Value::as_str);
        if category.is_some() || rationale.is_some() || require_annotations {
            let category = category
                .cloned()
                .ok_or_else(|| {
                    CoreError::InvalidTranslation(format!(
                        "full review change for `{id}` is missing category"
                    ))
                })
                .and_then(|value| {
                    serde_json::from_value::<ReviewIssueKind>(value).map_err(|_| {
                        CoreError::InvalidTranslation(format!(
                            "full review change for `{id}` has an invalid category"
                        ))
                    })
                })?;
            let rationale = rationale.unwrap_or_default().trim();
            if rationale.is_empty() || rationale.chars().count() > 200 {
                return Err(CoreError::InvalidTranslation(format!(
                    "full review change for `{id}` must have a 1-200 character rationale"
                )));
            }
            annotations.push(ReviewAnnotation {
                id,
                category,
                rationale: rationale.to_owned(),
            });
        }
    }
    Ok(ReviewResult {
        lines,
        review_notes: payload["review_notes"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        annotations,
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
    match compare_number_facts(&source.text, &translated.text) {
        NumberFactComparison::HardMismatch { .. } => reasons.push("number mismatch".to_owned()),
        NumberFactComparison::Uncertain => reasons.push(
            "ambiguous number expression; verify the fact without rewriting idioms".to_owned(),
        ),
        NumberFactComparison::Match => {}
    }
    if has_readability_risk(translated) {
        reasons.push("subtitle readability risk".to_owned());
    }
    if translations_by_source
        .get(&consistency_key(&source.text))
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

fn document_guide_review_reasons(
    source: &SubtitleSegment,
    translated: &SubtitleSegment,
    memory: &ContextMemory,
    language_rules: &ResolvedLanguageRules,
) -> Vec<String> {
    let matcher = TermMatcher::case_insensitive();
    let mut reasons = Vec::new();
    for character in &memory.document_guide.characters {
        let source_forms = std::iter::once(character.canonical_source.as_str())
            .chain(character.aliases.iter().map(|alias| alias.source.as_str()))
            .collect::<Vec<_>>();
        if source_forms
            .iter()
            .any(|form| !form.is_empty() && matcher.contains(&source.text, form))
        {
            let target_forms = std::iter::once(character.canonical_target.as_str())
                .chain(character.aliases.iter().map(|alias| alias.target.as_str()));
            if !target_forms
                .filter(|form| !form.is_empty())
                .any(|form| matcher.contains(&translated.text, form))
            {
                reasons.push("character-name continuity".to_owned());
            }
        }
        for form in &character.forms_of_address {
            if matcher.contains(&source.text, &form.source)
                && !matcher.contains(&translated.text, &form.target)
            {
                reasons.push(language_rules.form_of_address_review_reason().to_owned());
            }
        }
    }
    for entity in &memory.document_guide.terminology {
        for variant in &entity.variants {
            if matcher.contains(&source.text, &variant.source)
                && !matcher.contains(&translated.text, &variant.target)
            {
                reasons.push("document terminology consistency".to_owned());
            }
        }
    }
    reasons
}

fn has_contextual_pronoun_risk(
    source: &[SubtitleSegment],
    index: usize,
    memory: &ContextMemory,
    language_rules: &ResolvedLanguageRules,
) -> bool {
    if !language_rules.supports_english_coreference() || memory.document_guide.characters.is_empty()
    {
        return false;
    }
    let Some(current) = source.get(index) else {
        return false;
    };
    let words = current
        .text
        .split(|character: char| !character.is_alphanumeric())
        .map(str::to_ascii_lowercase)
        .collect::<BTreeSet<_>>();
    if !words
        .iter()
        .any(|word| EnglishRules::is_coreference_pronoun(word))
    {
        return false;
    }
    let start = index.saturating_sub(REVIEW_CONTEXT_LINES);
    let end = index
        .saturating_add(REVIEW_CONTEXT_LINES + 1)
        .min(source.len());
    let context = source[start..end]
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let matcher = TermMatcher::case_insensitive();
    memory.document_guide.characters.iter().any(|character| {
        std::iter::once(character.canonical_source.as_str())
            .chain(character.aliases.iter().map(|alias| alias.source.as_str()))
            .any(|name| !name.is_empty() && matcher.contains(&context, name))
    })
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn consistency_key(text: &str) -> String {
    text.chars()
        .map(|character| {
            if character.is_alphanumeric() {
                character
            } else {
                ' '
            }
        })
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
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
    use crate::entities::{DocumentCharacter, GlossaryEntry, TranslationMode};

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
        let mut memory = ContextMemory::default();
        memory.load_glossary(&[("two".to_owned(), "二".to_owned())]);
        let plan = build_full_review_plan(&batches, &translated, &memory, "English", "Chinese");
        let mut options = PipelineOptions::new("review.srt".into());
        options.mode = crate::entities::TranslationMode::Cinema;

        let messages = build_review_messages(&options, &plan[0], &memory, &BTreeMap::new());
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
        let plan = build_full_review_plan(
            &batches,
            &translated,
            &ContextMemory::default(),
            "English",
            "Chinese",
        );
        let mut options = PipelineOptions::new("review.ass".into());
        options.mode = crate::entities::TranslationMode::Cinema;

        let messages = build_review_messages(
            &options,
            &plan[0],
            &ContextMemory::default(),
            &BTreeMap::new(),
        );
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
            messages[1]
                .content
                .contains("not proof of identical meaning")
        );
    }

    #[test]
    fn full_and_targeted_review_use_distinct_instructions() {
        let source = vec![segment(
            "1",
            "Leave school again.",
            "00:00:00,000",
            "00:00:01,000",
        )];
        let mut translated = source.clone();
        translated[0].text = "再次离开学校。".to_owned();
        let mut full = PipelineOptions::new("review.srt".into());
        full.mode = TranslationMode::Cinema;
        full.review_policy = ReviewPolicy::Full;
        let required_glossary = BTreeMap::from([("school".to_owned(), "学校".to_owned())]);
        let mut memory = ContextMemory::default();
        memory.load_glossary(&[("leave school".to_owned(), "退学".to_owned())]);
        let plan = build_full_review_plan(&[source], &translated, &memory, "English", "Chinese");
        let full_messages = build_review_messages(&full, &plan[0], &memory, &required_glossary);
        assert!(
            full_messages[0]
                .content
                .contains("document-level targeted subtitle revision adjudicator")
        );
        assert!(
            full_messages[0]
                .content
                .contains("intentionally incomplete phrasing")
        );
        assert!(full_messages[1].content.contains("\"category\""));
        assert!(full_messages[1].content.contains("required_glossary"));
        assert!(full_messages[1].content.contains("deterministic_issues"));
        assert!(!full_messages[1].content.contains("退学"));
        assert!(!full_messages[0].content.contains("targeted subtitle QA"));

        let mut targeted = full;
        targeted.review_policy = ReviewPolicy::Targeted;
        let targeted_messages =
            build_review_messages(&targeted, &plan[0], &memory, &required_glossary);
        assert!(
            targeted_messages[0]
                .content
                .contains("targeted deterministic")
        );
        assert!(targeted_messages[1].content.contains("smallest change"));
        assert!(!targeted_messages[1].content.contains("\"category\""));
        assert!(
            targeted_messages[0]
                .content
                .contains("preserve idioms and lexicalized expressions")
        );
    }

    #[test]
    fn targeted_review_distinguishes_hard_and_ambiguous_number_candidates() {
        let source = vec![
            segment(
                "1",
                "Look at all this mess.",
                "00:00:00,000",
                "00:00:01,000",
            ),
            segment("2", "She is 12 years old.", "00:00:01,000", "00:00:02,000"),
        ];
        let translated = vec![
            segment(
                "1",
                "看看这些乱七八糟的东西。",
                "00:00:00,000",
                "00:00:01,000",
            ),
            segment("2", "她十三岁。", "00:00:01,000", "00:00:02,000"),
        ];

        let plan = build_review_plan(
            &[source],
            &translated,
            &ContextMemory::default(),
            "English",
            "Chinese",
        );

        assert_eq!(plan.len(), 1);
        assert!(plan[0].candidate_reasons["1"][0].contains("ambiguous number expression"));
        assert_eq!(plan[0].candidate_reasons["2"], vec!["number mismatch"]);
    }

    #[test]
    fn full_review_skips_a_clean_document() {
        let source = vec![segment(
            "1",
            "Open the door.",
            "00:00:00,000",
            "00:00:01,000",
        )];
        let translated = vec![segment("1", "把门打开。", "00:00:00,000", "00:00:01,000")];

        let plan = build_full_review_plan(
            &[source],
            &translated,
            &ContextMemory::default(),
            "English",
            "Chinese",
        );

        assert!(plan.is_empty());
    }

    #[test]
    fn full_review_targets_document_guide_and_pronoun_risks_only() {
        let source = vec![
            segment("1", "Alice arrived.", "00:00:00,000", "00:00:01,000"),
            segment("2", "She sat down.", "00:00:01,000", "00:00:02,000"),
            segment("3", "Hello there.", "00:00:02,000", "00:00:03,000"),
        ];
        let translated = vec![
            segment("1", "爱丽丝到了。", "00:00:00,000", "00:00:01,000"),
            segment("2", "他坐下了。", "00:00:01,000", "00:00:02,000"),
            segment("3", "你好。", "00:00:02,000", "00:00:03,000"),
        ];
        let mut memory = ContextMemory::default();
        memory.document_guide.characters.push(DocumentCharacter {
            canonical_source: "Alice".to_owned(),
            canonical_target: "爱丽丝".to_owned(),
            aliases: vec![GlossaryEntry {
                source: "Alice".to_owned(),
                target: "爱丽丝".to_owned(),
            }],
            gender: Some("female".to_owned()),
            ..DocumentCharacter::default()
        });

        let plan = build_full_review_plan(&[source], &translated, &memory, "English", "Chinese");

        assert_eq!(plan.len(), 1);
        assert_eq!(
            plan[0].candidate_reasons["2"],
            vec!["pronoun/coreference continuity"]
        );
        assert!(!plan[0].candidate_reasons.contains_key("1"));
        assert!(!plan[0].candidate_reasons.contains_key("3"));
    }

    #[test]
    fn japanese_chinese_honorific_drift_is_advisory_full_review_only() {
        let source = vec![segment(
            "1",
            "田中さんが来た。",
            "00:00:00,000",
            "00:00:01,000",
        )];
        let translated = vec![segment("1", "田中来了。", "00:00:00,000", "00:00:01,000")];
        let mut memory = ContextMemory::default();
        memory.document_guide.characters.push(DocumentCharacter {
            canonical_source: "田中".to_owned(),
            canonical_target: "田中".to_owned(),
            aliases: vec![GlossaryEntry {
                source: "田中".to_owned(),
                target: "田中".to_owned(),
            }],
            forms_of_address: vec![GlossaryEntry {
                source: "田中さん".to_owned(),
                target: "田中先生".to_owned(),
            }],
            ..DocumentCharacter::default()
        });

        assert!(
            final_validation_issues(
                &source,
                &translated,
                &BTreeMap::new(),
                "ja",
                "zh-Hans",
                FinalValidationPolicy::default(),
            )
            .expect("final validation")
            .is_empty()
        );

        let full = build_full_review_plan(
            std::slice::from_ref(&source),
            &translated,
            &memory,
            "ja",
            "zh-Hans",
        );
        assert_eq!(
            full[0].candidate_reasons["1"],
            vec!["Japanese honorific/form-of-address consistency"]
        );
        let targeted = build_review_plan(
            std::slice::from_ref(&source),
            &full[0].translated,
            &memory,
            "ja",
            "zh-Hans",
        );
        assert!(targeted.is_empty());

        let corrected = vec![segment(
            "1",
            "田中先生来了。",
            "00:00:00,000",
            "00:00:01,000",
        )];
        assert!(build_full_review_plan(&[source], &corrected, &memory, "ja", "zh-Hans").is_empty());
    }

    #[test]
    fn full_review_parses_typed_audit_metadata() {
        let payload = serde_json::json!({
            "changes": [{
                "id": "7",
                "translation": "别再逃课了。",
                "category": "accuracy",
                "rationale": "leave school 在此处指逃课"
            }]
        });

        let parsed = parse_review_payload(&payload, true).expect("full review payload");

        assert_eq!(parsed.lines[0].id, "7");
        assert_eq!(parsed.annotations[0].category, ReviewIssueKind::Accuracy);
        assert_eq!(parsed.annotations[0].rationale, "leave school 在此处指逃课");
    }

    #[test]
    fn full_review_rejects_changes_without_audit_metadata() {
        let payload = serde_json::json!({
            "changes": [{"id": "7", "translation": "别再逃课了。"}]
        });

        let error = parse_review_payload(&payload, true).expect_err("metadata is required");

        assert!(error.to_string().contains("missing category"));
    }
}
