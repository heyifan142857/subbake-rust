use crate::entities::{PipelineOptions, ReviewResult, SubtitleSegment, TranslationLine};
use crate::error::{CoreError, CoreResult};
use crate::memory::ContextMemory;
use crate::ports::ChatMessage;
use crate::validation::validate_full_alignment;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReviewBatchPlan {
    pub(crate) start_offset: usize,
    pub(crate) source: Vec<SubtitleSegment>,
    pub(crate) translated: Vec<SubtitleSegment>,
    pub(crate) reasons: Vec<String>,
}

pub(crate) fn build_review_plan(
    batches: &[Vec<SubtitleSegment>],
    translated_segments: &[SubtitleSegment],
    memory: &ContextMemory,
) -> Vec<ReviewBatchPlan> {
    let mut plan = Vec::new();
    let mut offset = 0usize;
    for source in batches {
        let end = offset + source.len();
        let translated = &translated_segments[offset..end];
        let reasons = review_reasons(source, translated, memory);
        if !reasons.is_empty() {
            plan.push(ReviewBatchPlan {
                start_offset: offset,
                source: source.clone(),
                translated: translated.to_vec(),
                reasons,
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
        .map(|source| {
            let end = offset + source.len();
            let plan = ReviewBatchPlan {
                start_offset: offset,
                source: source.clone(),
                translated: translated_segments[offset..end].to_vec(),
                reasons: vec!["full review".to_owned()],
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
        "expected_count": source.len(),
        "expected_ids": source.iter().map(|segment| segment.id.as_str()).collect::<Vec<_>>(),
        "lines": source.iter().zip(translated).map(|(source, translated)| {
            serde_json::json!({
                "id": source.id,
                "source": source.text,
                "translation": translated.text,
            })
        }).collect::<Vec<_>>(),
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
         Only fix terminology, consistency, and readability issues without changing the number of entries.",
        options.target_language
    );
    let user = format!(
        "TASK_START\nreview_translations\nTASK_END\n\
         Review only this high-risk batch.\n\
         Use the input lines array as the complete authoritative list of subtitle entries.\n\
         Do not remove, reorder, merge, or renumber entries.\n\
         Return exactly one output object for each input line id, in the same order as expected_ids.\n\
         Prefer minimal edits; leave good lines untouched.\n\
         Return JSON only as {{\"changes\":[{{\"id\":\"<id>\",\"translation\":\"<replacement>\"}}]}}.\n\
         Return an empty changes array when every translation is already good.\n\
         REVIEW_JSON_START{payload_json}REVIEW_JSON_END"
    );
    vec![ChatMessage::system(system), ChatMessage::user(user)]
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
        return Err(CoreError::Data(format!(
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

fn review_reasons(
    source: &[SubtitleSegment],
    translated: &[SubtitleSegment],
    memory: &ContextMemory,
) -> Vec<String> {
    let texts = source
        .iter()
        .chain(translated)
        .map(|segment| segment.text.as_str())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>();
    let glossary_risk = memory
        .select_relevant_glossary(&texts)
        .iter()
        .any(|(term, _)| is_glossary_term_risky(term, source));

    let mut reasons = Vec::new();
    let mut score = 0usize;
    if glossary_risk {
        reasons.push("glossary consistency".to_owned());
        score += 2;
    } else if source.iter().any(|segment| has_named_terms(&segment.text)) {
        reasons.push("names and terms".to_owned());
        score += 2;
    }
    if source
        .iter()
        .any(|segment| has_speaker_marker(&segment.text))
    {
        reasons.push("speaker changes".to_owned());
        score += 2;
    }
    if source
        .iter()
        .any(|segment| contains_formatting(&segment.text))
    {
        reasons.push("formatting and tags".to_owned());
        score += 2;
    }
    if source.iter().zip(translated).any(|(source, translated)| {
        is_dense_segment(&source.text) || is_dense_segment(&translated.text)
    }) {
        reasons.push("readability".to_owned());
        score += 1;
    }
    if score >= 2 { reasons } else { Vec::new() }
}

fn has_speaker_marker(text: &str) -> bool {
    let stripped = text.trim_start();
    if stripped.starts_with('-')
        || stripped.starts_with('–')
        || stripped.starts_with('—')
        || stripped.starts_with(">>")
    {
        return true;
    }
    let Some(colon) = stripped.find(':') else {
        return false;
    };
    let label = &stripped[..colon];
    let mut chars = label.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_uppercase()
        && label.chars().count() <= 21
        && chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, ' ' | '_' | '-'))
        && stripped[colon + 1..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
}

fn contains_formatting(text: &str) -> bool {
    has_delimited_text(text, '<', '>') || has_delimited_text(text, '{', '}')
}

fn has_delimited_text(text: &str, open: char, close: char) -> bool {
    text.find(open)
        .and_then(|start| text[start + open.len_utf8()..].find(close))
        .is_some()
}

fn has_named_terms(text: &str) -> bool {
    let mut matches = Vec::new();
    let mut word_start = None;
    for (index, ch) in text
        .char_indices()
        .chain(std::iter::once((text.len(), ' ')))
    {
        if ch.is_ascii_alphabetic() {
            word_start.get_or_insert(index);
            continue;
        }
        let Some(start) = word_start.take() else {
            continue;
        };
        let word = &text[start..index];
        let mut chars = word.chars();
        if chars.next().is_some_and(|first| first.is_ascii_uppercase())
            && chars.clone().count() >= 2
            && chars.all(|rest| rest.is_ascii_lowercase())
        {
            matches.push(start);
        }
    }
    if matches.is_empty() {
        return false;
    }
    if matches.len() == 1 {
        return !text[..matches[0]]
            .chars()
            .all(|ch| !ch.is_ascii_alphanumeric() && ch != '_');
    }
    true
}

fn is_dense_segment(text: &str) -> bool {
    text.trim().chars().count() >= 84 || text.trim().contains('\n')
}

fn is_glossary_term_risky(term: &str, source: &[SubtitleSegment]) -> bool {
    let normalized = term.trim();
    if normalized.is_empty() {
        return false;
    }
    if normalized.contains(' ')
        || normalized.contains('-')
        || normalized.chars().any(|ch| ch.is_ascii_digit())
        || normalized
            .chars()
            .skip(1)
            .filter(|ch| ch.is_ascii_uppercase())
            .count()
            >= 2
    {
        return true;
    }
    source.iter().any(|segment| {
        segment.text.match_indices(normalized).any(|(start, _)| {
            start > 0 && has_word_boundaries(&segment.text, start, normalized.len())
        })
    })
}

fn has_word_boundaries(text: &str, start: usize, length: usize) -> bool {
    let before = text[..start].chars().next_back();
    let after = text[start + length..].chars().next();
    before.is_none_or(|ch| !is_word_char(ch)) && after.is_none_or(|ch| !is_word_char(ch))
}

fn is_word_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}
