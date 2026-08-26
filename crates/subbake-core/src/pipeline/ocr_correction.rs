use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::entities::{
    BitmapOcrSource, OcrCorrectionChange, OcrCorrectionMode, OcrCorrectionOrigin,
    OcrCorrectionReport, OcrCorrectionResult, OcrCorrectionSummary, OcrWordConfidence,
    SubtitleDocument, SubtitleSegment, TranslationMode, Usage,
};
use crate::error::{CoreError, CoreResult};
use crate::formatting::formatting_tokens;
use crate::ports::{BackendJsonResult, BackendPayload, CacheStage, ChatMessage};
use crate::progress::{ProgressEvent, ProgressUnit, TaskKind, TaskState};
use crate::storage::OCR_CORRECTION_REPORT_VERSION;

use super::SubtitlePipeline;
use super::support::{estimated_request_tokens, request_hash};

const LOW_CONFIDENCE_THRESHOLD: u8 = 70;

#[derive(Debug)]
pub(super) struct OcrCorrectionRun {
    pub source_segments: Vec<SubtitleSegment>,
    pub report: Option<OcrCorrectionReport>,
    pub usage: Usage,
}

#[derive(Debug, Clone)]
struct Candidate {
    index: usize,
    reasons: Vec<String>,
}

#[derive(Serialize)]
struct PromptCue<'a> {
    id: &'a str,
    source: &'a str,
    reasons: &'a [String],
    word_confidences: &'a [OcrWordConfidence],
    context_before: Vec<ContextCue<'a>>,
    context_after: Vec<ContextCue<'a>>,
}

#[derive(Serialize)]
struct ContextCue<'a> {
    id: &'a str,
    source: &'a str,
}

pub(super) fn run<B>(
    pipeline: &mut SubtitlePipeline<B>,
    document: &SubtitleDocument,
) -> CoreResult<OcrCorrectionRun>
where
    B: crate::ports::LlmBackend,
{
    let Some(source) = pipeline.options.ocr_source.clone() else {
        return Ok(OcrCorrectionRun {
            source_segments: document.segments.clone(),
            report: None,
            usage: Usage::default(),
        });
    };
    let requested_mode = pipeline.options.execution.ocr_correction;
    let mode = requested_mode.resolve(pipeline.options.execution.mode);
    if mode == OcrCorrectionMode::Off {
        return Ok(OcrCorrectionRun {
            source_segments: document.segments.clone(),
            report: Some(build_report(
                pipeline,
                &source,
                mode,
                &document.segments,
                &document.segments,
                &[],
                None,
            )),
            usage: Usage::default(),
        });
    }

    pipeline.cancellation.check()?;
    let english = is_english(&source.source_language)
        || is_english(&pipeline.options.validation.source_language);
    let mut corrected = document.segments.clone();
    let mut deterministic_ids = BTreeSet::new();
    if english {
        for segment in &mut corrected {
            let replacement = deterministic_correction(&segment.text);
            if replacement != segment.text {
                deterministic_ids.insert(segment.id.clone());
                segment.text = replacement;
            }
        }
    }
    let metadata = metadata_by_id(&source);
    let candidates = detect_candidates(
        &document.segments,
        &corrected,
        &metadata,
        &deterministic_ids,
    );
    emit_progress(pipeline, TaskState::Running, 0, candidates.len(), None);

    let planned_model_requests = if mode == OcrCorrectionMode::Model && !candidates.is_empty() {
        candidate_batches(pipeline, &corrected, &metadata, &candidates).len()
    } else {
        0
    };
    if pipeline.options.execution.dry_run {
        let mut report = build_report(
            pipeline,
            &source,
            mode,
            &document.segments,
            &corrected,
            &candidates,
            None,
        );
        report.summary.planned_model_requests = planned_model_requests;
        emit_progress(
            pipeline,
            TaskState::Completed,
            candidates.len(),
            candidates.len(),
            Some(format!(
                "{} suspicious OCR cue(s); {planned_model_requests} model request(s) planned",
                candidates.len()
            )),
        );
        return Ok(OcrCorrectionRun {
            source_segments: corrected,
            report: Some(report),
            usage: Usage::default(),
        });
    }

    let deterministic = corrected.clone();
    let mut usage = Usage::default();
    let mut fallback_error = None;
    if mode == OcrCorrectionMode::Model
        && !candidates.is_empty()
        && let Err(error) =
            correct_with_model(pipeline, &mut corrected, &metadata, &candidates, &mut usage)
    {
        if matches!(error, CoreError::Cancelled)
            || pipeline.options.execution.mode == TranslationMode::Cinema
        {
            emit_progress(
                pipeline,
                TaskState::Failed,
                0,
                candidates.len(),
                Some(error.to_string()),
            );
            return Err(error);
        }
        corrected = deterministic;
        fallback_error = Some(error.to_string());
    }

    let mut report = build_report(
        pipeline,
        &source,
        mode,
        &document.segments,
        &corrected,
        &candidates,
        fallback_error.clone(),
    );
    report.summary.planned_model_requests = planned_model_requests;
    if let Some(store) = pipeline.store.as_ref() {
        pipeline.cancellation.check()?;
        report.summary.report_path = Some(store.paths().ocr_correction_report_path.clone());
        store.save_ocr_correction_report(&report)?;
    }
    emit_progress(
        pipeline,
        TaskState::Completed,
        candidates.len(),
        candidates.len(),
        fallback_error.map(|error| format!("model correction fell back: {error}")),
    );
    Ok(OcrCorrectionRun {
        source_segments: corrected,
        report: Some(report),
        usage,
    })
}

fn correct_with_model<B>(
    pipeline: &mut SubtitlePipeline<B>,
    corrected: &mut [SubtitleSegment],
    metadata: &BTreeMap<String, Vec<OcrWordConfidence>>,
    candidates: &[Candidate],
    usage: &mut Usage,
) -> CoreResult<()>
where
    B: crate::ports::LlmBackend,
{
    for batch in candidate_batches(pipeline, corrected, metadata, candidates) {
        pipeline.cancellation.check()?;
        let messages = build_messages(corrected, metadata, &batch)?;
        let batch_segments = batch
            .iter()
            .map(|candidate| corrected[candidate.index].clone())
            .collect::<Vec<_>>();
        let estimated_tokens = estimated_request_tokens(&messages, &batch_segments);
        if estimated_tokens > pipeline.options.execution.request_token_budget {
            return Err(CoreError::ResourceBudgetExceeded(format!(
                "OCR correction request needs an estimated {estimated_tokens} tokens, above the {} token request budget",
                pipeline.options.execution.request_token_budget
            )));
        }
        let hash = request_hash(&pipeline.options, CacheStage::OcrCorrection, &messages);
        let mut cached = false;
        let mut response = if pipeline.options.execution.use_cache {
            pipeline
                .store
                .as_ref()
                .map(|store| store.load_cached_response(CacheStage::OcrCorrection, &hash))
                .transpose()?
                .flatten()
        } else {
            None
        };
        if response.is_some() {
            cached = true;
            pipeline.accounting.record_cache_hit();
        }
        let mut last_error = None;
        for _attempt in 0..=pipeline.options.execution.retries {
            let generated = if let Some(response) = response.take() {
                Ok(response)
            } else {
                pipeline
                    .execute_review_json(&messages)
                    .and_then(|(json, call_usage)| {
                        usage.add(call_usage);
                        parse_payload(json).map(|payload| BackendJsonResult {
                            payload: BackendPayload::OcrCorrection(payload),
                            usage: call_usage,
                        })
                    })
            };
            match generated.and_then(|response| {
                let BackendPayload::OcrCorrection(result) = response.payload else {
                    return Err(CoreError::DataInvariant(
                        "OCR correction cache returned an incompatible payload".to_owned(),
                    ));
                };
                validate_and_apply(corrected, &batch, &result)?;
                if pipeline.options.execution.use_cache
                    && !cached
                    && let Some(store) = pipeline.store.as_ref()
                {
                    store.save_cached_response(
                        CacheStage::OcrCorrection,
                        &hash,
                        &BackendJsonResult {
                            payload: BackendPayload::OcrCorrection(result),
                            usage: response.usage,
                        },
                    )?;
                }
                Ok(())
            }) {
                Ok(()) => {
                    last_error = None;
                    break;
                }
                Err(error) => {
                    if matches!(error, CoreError::Cancelled) {
                        return Err(error);
                    }
                    last_error = Some(error);
                    cached = false;
                }
            }
        }
        if let Some(error) = last_error {
            return Err(CoreError::InvalidTranslation(format!(
                "OCR correction failed after retries: {error}"
            )));
        }
    }
    Ok(())
}

fn candidate_batches<B>(
    pipeline: &SubtitlePipeline<B>,
    segments: &[SubtitleSegment],
    metadata: &BTreeMap<String, Vec<OcrWordConfidence>>,
    candidates: &[Candidate],
) -> Vec<Vec<Candidate>>
where
    B: crate::ports::LlmBackend,
{
    let mut batches = Vec::new();
    let mut current = Vec::new();
    for candidate in candidates {
        current.push(candidate.clone());
        let over_budget = build_messages(segments, metadata, &current)
            .map(|messages| {
                let batch_segments = current
                    .iter()
                    .map(|candidate| segments[candidate.index].clone())
                    .collect::<Vec<_>>();
                estimated_request_tokens(&messages, &batch_segments)
                    > pipeline.options.execution.request_token_budget
            })
            .unwrap_or(true);
        if over_budget && current.len() > 1 {
            let last = current.pop().expect("candidate batch is non-empty");
            batches.push(std::mem::take(&mut current));
            current.push(last);
        }
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

fn build_messages(
    segments: &[SubtitleSegment],
    metadata: &BTreeMap<String, Vec<OcrWordConfidence>>,
    candidates: &[Candidate],
) -> CoreResult<Vec<ChatMessage>> {
    let payload = candidates
        .iter()
        .map(|candidate| {
            let segment = &segments[candidate.index];
            PromptCue {
                id: &segment.id,
                source: &segment.text,
                reasons: &candidate.reasons,
                word_confidences: metadata
                    .get(&segment.id)
                    .map(Vec::as_slice)
                    .unwrap_or_default(),
                context_before: context_before(segments, candidate.index),
                context_after: context_after(segments, candidate.index),
            }
        })
        .collect::<Vec<_>>();
    let json = serde_json::to_string(&payload).map_err(|error| {
        CoreError::DataInvariant(format!("serialize OCR correction prompt failed: {error}"))
    })?;
    Ok(vec![
        ChatMessage::cacheable_system(
            "You correct only obvious OCR recognition errors in subtitle source text. Do not translate, polish, paraphrase, change tone, add content, or use context as output. Return a JSON object with a `lines` array in exactly the supplied candidate ID order; each item has only `id` and `corrected_source`. Return the source unchanged when uncertain. Preserve line breaks, word count, formatting tags, standalone numbers, and all meaning.",
        ),
        ChatMessage::user(format!(
            "Correct the candidate cues below. Reasons, word confidence, and two neighboring cues on each side are evidence only.\n{json}"
        )),
    ])
}

fn context_before(segments: &[SubtitleSegment], index: usize) -> Vec<ContextCue<'_>> {
    segments[index.saturating_sub(2)..index]
        .iter()
        .map(|segment| ContextCue {
            id: &segment.id,
            source: &segment.text,
        })
        .collect()
}

fn context_after(segments: &[SubtitleSegment], index: usize) -> Vec<ContextCue<'_>> {
    segments[index + 1..segments.len().min(index + 3)]
        .iter()
        .map(|segment| ContextCue {
            id: &segment.id,
            source: &segment.text,
        })
        .collect()
}

fn parse_payload(value: serde_json::Value) -> CoreResult<OcrCorrectionResult> {
    serde_json::from_value(value).map_err(|error| {
        CoreError::InvalidTranslation(format!("invalid OCR correction response: {error}"))
    })
}

fn validate_and_apply(
    segments: &mut [SubtitleSegment],
    candidates: &[Candidate],
    result: &OcrCorrectionResult,
) -> CoreResult<()> {
    if result.lines.len() != candidates.len() {
        return Err(CoreError::InvalidTranslation(format!(
            "OCR correction returned {} line(s), expected {}",
            result.lines.len(),
            candidates.len()
        )));
    }
    for (candidate, line) in candidates.iter().zip(&result.lines) {
        let source = &segments[candidate.index];
        if line.id != source.id {
            return Err(CoreError::InvalidTranslation(format!(
                "OCR correction returned ID `{}` where `{}` was expected",
                line.id, source.id
            )));
        }
        validate_correction(&source.text, &line.corrected_source, &candidate.reasons)?;
    }
    for (candidate, line) in candidates.iter().zip(&result.lines) {
        segments[candidate.index]
            .text
            .clone_from(&line.corrected_source);
    }
    Ok(())
}

fn validate_correction(source: &str, corrected: &str, reasons: &[String]) -> CoreResult<()> {
    if corrected.trim().is_empty() {
        return Err(CoreError::InvalidTranslation(
            "OCR correction returned empty source text".to_owned(),
        ));
    }
    if source.matches('\n').count() != corrected.matches('\n').count() {
        return Err(CoreError::InvalidTranslation(
            "OCR correction changed the line count".to_owned(),
        ));
    }
    let source_words = source.split_whitespace().collect::<Vec<_>>();
    let corrected_words = corrected.split_whitespace().collect::<Vec<_>>();
    if source_words.len() != corrected_words.len() {
        return Err(CoreError::InvalidTranslation(
            "OCR correction changed the word count".to_owned(),
        ));
    }
    if formatting_tokens(source) != formatting_tokens(corrected) {
        return Err(CoreError::InvalidTranslation(
            "OCR correction changed formatting markers".to_owned(),
        ));
    }
    let limit = source.chars().count().div_ceil(5).clamp(2, 12);
    if edit_distance(source, corrected) > limit {
        return Err(CoreError::InvalidTranslation(format!(
            "OCR correction changed more than {limit} character(s)"
        )));
    }
    let mixed_flagged = reasons
        .iter()
        .any(|reason| reason == "abnormal_mixed_token");
    for (before, after) in source_words.iter().zip(corrected_words) {
        if is_standalone_numeric_token(before) && *before != after {
            return Err(CoreError::InvalidTranslation(
                "OCR correction changed a standalone numeric token".to_owned(),
            ));
        }
        if contains_ascii_letter_and_digit(before) && *before != after && !mixed_flagged {
            return Err(CoreError::InvalidTranslation(
                "OCR correction changed an unmarked mixed alphanumeric token".to_owned(),
            ));
        }
    }
    Ok(())
}

fn deterministic_correction(text: &str) -> String {
    text.split('\n')
        .map(|line| {
            let line = correct_leading_bang(line);
            line.split_inclusive(char::is_whitespace)
                .map(|piece| {
                    let token_end = piece.find(char::is_whitespace).unwrap_or(piece.len());
                    if &piece[..token_end] == "|" {
                        format!("I{}", &piece[token_end..])
                    } else {
                        piece.to_owned()
                    }
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn correct_leading_bang(line: &str) -> String {
    let whitespace = line.len() - line.trim_start_matches([' ', '\t']).len();
    let rest = &line[whitespace..];
    let bang = if rest.starts_with("! ") || rest.starts_with("!\t") {
        Some(whitespace)
    } else if rest.starts_with("-! ") || rest.starts_with("-!\t") {
        Some(whitespace + 1)
    } else if rest.starts_with("—! ") || rest.starts_with("—!\t") {
        Some(whitespace + '—'.len_utf8())
    } else {
        None
    };
    let Some(index) = bang else {
        return line.to_owned();
    };
    let mut output = line.to_owned();
    output.replace_range(index..index + 1, "I");
    output
}

fn detect_candidates(
    original: &[SubtitleSegment],
    corrected: &[SubtitleSegment],
    metadata: &BTreeMap<String, Vec<OcrWordConfidence>>,
    deterministic_ids: &BTreeSet<String>,
) -> Vec<Candidate> {
    original
        .iter()
        .zip(corrected)
        .enumerate()
        .filter_map(|(index, (original, corrected))| {
            let mut reasons = Vec::new();
            if deterministic_ids.contains(&original.id) {
                reasons.push("deterministic_correction".to_owned());
            }
            if metadata.get(&original.id).is_some_and(|words| {
                words.iter().any(|word| {
                    word.confidence
                        .is_some_and(|confidence| confidence < LOW_CONFIDENCE_THRESHOLD)
                })
            }) {
                reasons.push("low_word_confidence".to_owned());
            }
            if corrected.text.contains('|') {
                reasons.push("remaining_vertical_bar".to_owned());
            }
            if has_pronoun_bang(&corrected.text) {
                reasons.push("pronoun_position_bang".to_owned());
            }
            if corrected
                .text
                .split_whitespace()
                .any(is_abnormal_mixed_token)
            {
                reasons.push("abnormal_mixed_token".to_owned());
            }
            (!reasons.is_empty()).then_some(Candidate { index, reasons })
        })
        .collect()
}

fn has_pronoun_bang(text: &str) -> bool {
    text.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with("! ")
            || line.starts_with("!\t")
            || line.starts_with("-! ")
            || line.starts_with("—! ")
            || line.split_whitespace().any(|token| token == "!")
    })
}

fn is_abnormal_mixed_token(token: &str) -> bool {
    let trimmed = token
        .trim_matches(|ch: char| matches!(ch, ',' | '.' | ':' | ';' | '?' | '"'))
        .trim_end_matches('!');
    contains_ascii_letter_and_digit(trimmed)
        || (trimmed.chars().any(|ch| ch.is_ascii_alphabetic())
            && trimmed.chars().any(|ch| {
                matches!(
                    ch,
                    '|' | '!' | '@' | '#' | '$' | '%' | '^' | '&' | '*' | '_' | '+' | '=' | '~'
                )
            }))
}

fn contains_ascii_letter_and_digit(token: &str) -> bool {
    token.chars().any(|ch| ch.is_ascii_alphabetic()) && token.chars().any(|ch| ch.is_ascii_digit())
}

fn is_standalone_numeric_token(token: &str) -> bool {
    let trimmed = token.trim_matches(|ch: char| ch.is_ascii_punctuation());
    !trimmed.is_empty() && trimmed.chars().all(|ch| ch.is_ascii_digit())
}

fn metadata_by_id(source: &BitmapOcrSource) -> BTreeMap<String, Vec<OcrWordConfidence>> {
    source
        .cues
        .iter()
        .map(|cue| (cue.id.clone(), cue.words.clone()))
        .collect()
}

fn build_report<B>(
    pipeline: &SubtitlePipeline<B>,
    source: &BitmapOcrSource,
    mode: OcrCorrectionMode,
    original: &[SubtitleSegment],
    corrected: &[SubtitleSegment],
    candidates: &[Candidate],
    fallback_error: Option<String>,
) -> OcrCorrectionReport
where
    B: crate::ports::LlmBackend,
{
    let metadata = metadata_by_id(source);
    let reasons = candidates
        .iter()
        .map(|candidate| (candidate.index, candidate.reasons.clone()))
        .collect::<BTreeMap<_, _>>();
    let changes = original
        .iter()
        .zip(corrected)
        .enumerate()
        .map(|(index, (before, after))| {
            let cue_reasons = reasons.get(&index).cloned().unwrap_or_default();
            let origin = if before.text == after.text {
                OcrCorrectionOrigin::Unchanged
            } else if cue_reasons
                .iter()
                .any(|reason| reason == "deterministic_correction")
                && deterministic_correction(&before.text) == after.text
            {
                OcrCorrectionOrigin::Deterministic
            } else {
                OcrCorrectionOrigin::Model
            };
            OcrCorrectionChange {
                id: before.id.clone(),
                original_source: before.text.clone(),
                corrected_source: after.text.clone(),
                word_confidences: metadata.get(&before.id).cloned().unwrap_or_default(),
                reasons: cue_reasons,
                origin,
            }
        })
        .collect::<Vec<_>>();
    let deterministic_corrections = changes
        .iter()
        .filter(|change| change.origin == OcrCorrectionOrigin::Deterministic)
        .count();
    let model_corrections = changes
        .iter()
        .filter(|change| change.origin == OcrCorrectionOrigin::Model)
        .count();
    OcrCorrectionReport {
        version: OCR_CORRECTION_REPORT_VERSION,
        mode,
        codec: source.codec.clone(),
        summary: OcrCorrectionSummary {
            candidates: candidates.len(),
            deterministic_corrections,
            model_corrections,
            unchanged: candidates
                .len()
                .saturating_sub(deterministic_corrections + model_corrections),
            planned_model_requests: 0,
            fallback: fallback_error.is_some(),
            fallback_error,
            report_path: None,
        },
        changes,
        backend_fingerprint: pipeline
            .options
            .identity
            .reviewer_fingerprint
            .clone()
            .or_else(|| pipeline.options.identity.provider_fingerprint.clone()),
    }
}

fn emit_progress<B>(
    pipeline: &SubtitlePipeline<B>,
    state: TaskState,
    current: usize,
    total: usize,
    message: Option<String>,
) where
    B: crate::ports::LlmBackend,
{
    let Some(progress) = pipeline.progress.as_ref() else {
        return;
    };
    let mut event = ProgressEvent::running(
        TaskKind::Translation,
        "CORRECT_BITMAP_OCR",
        current as u64,
        Some(total as u64),
        ProgressUnit::Lines,
    );
    event.state = state;
    event.message = message;
    progress.emit(event);
}

fn is_english(language: &str) -> bool {
    matches!(
        language.trim().to_ascii_lowercase().as_str(),
        "english" | "en" | "eng" | "en-us" | "en-gb"
    )
}

fn edit_distance(left: &str, right: &str) -> usize {
    let right = right.chars().collect::<Vec<_>>();
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    for (left_index, left_char) in left.chars().enumerate() {
        let mut current = vec![left_index + 1];
        for (right_index, right_char) in right.iter().enumerate() {
            current.push(
                (previous[right_index + 1] + 1)
                    .min(current[right_index] + 1)
                    .min(previous[right_index] + usize::from(left_char != *right_char)),
            );
        }
        previous = current;
    }
    previous[right.len()]
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::CancellationToken;
    use crate::OcrCorrectionLine;
    use crate::error::LlmCallError;
    use crate::ports::{GenerationRequest, GenerationResponse, LlmBackend};

    use super::*;

    struct ScriptedBackend {
        calls: Arc<AtomicUsize>,
        fail: bool,
        invalid_first: bool,
        corrected_source: &'static str,
    }

    impl LlmBackend for ScriptedBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "ocr-corrector"
        }

        fn execute(
            &mut self,
            _request: GenerationRequest,
            cancellation: &crate::CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            cancellation.check().map_err(LlmCallError::from)?;
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err(LlmCallError::Transport("scripted failure".to_owned()));
            }
            let json = if self.invalid_first && call == 0 {
                serde_json::json!({"lines": []})
            } else {
                serde_json::json!({
                    "lines": [{"id": "1", "corrected_source": self.corrected_source}]
                })
            };
            Ok(GenerationResponse::json(
                json,
                Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    total_tokens: 15,
                    requests: 1,
                    ..Usage::default()
                },
            ))
        }
    }

    fn segment(id: &str, text: &str) -> SubtitleSegment {
        SubtitleSegment {
            id: id.to_owned(),
            text: text.to_owned(),
            start: Some("00:00:10,000".to_owned()),
            end: Some("00:00:11,000".to_owned()),
            identifier: None,
            settings: None,
            semantic: Default::default(),
        }
    }

    fn document(text: &str) -> SubtitleDocument {
        SubtitleDocument {
            path: "bitmap.srt".into(),
            format: "srt".to_owned(),
            segments: vec![segment("1", text)],
            header: None,
            passthrough_blocks: Vec::new(),
            metadata: Default::default(),
        }
    }

    fn pipeline(
        mode: TranslationMode,
        correction: OcrCorrectionMode,
        backend: ScriptedBackend,
    ) -> SubtitlePipeline<ScriptedBackend> {
        let mut options = crate::entities::PipelineOptions::new("bitmap.srt".into());
        options.execution.mode = mode;
        options.execution.ocr_correction = correction;
        options.execution.retries = 0;
        options.validation.source_language = "English".to_owned();
        options.ocr_source = Some(BitmapOcrSource {
            codec: "hdmv_pgs_subtitle".to_owned(),
            source_language: "eng".to_owned(),
            cues: vec![crate::entities::OcrCueMetadata {
                id: "1".to_owned(),
                words: vec![OcrWordConfidence {
                    text: "|".to_owned(),
                    confidence: Some(40),
                }],
            }],
        });
        SubtitlePipeline::new(backend, options)
    }

    #[test]
    fn deterministic_repairs_pronoun_shapes_only() {
        assert_eq!(deterministic_correction("| want this"), "I want this");
        assert_eq!(deterministic_correction("-! don't"), "-I don't");
        assert_eq!(deterministic_correction("Stop!"), "Stop!");
        assert_eq!(deterministic_correction("What! Really?"), "What! Really?");
    }

    #[test]
    fn candidate_detection_covers_confidence_and_symbols() {
        let original = vec![
            segment("1", "normal words"),
            segment("2", "code 4S"),
            segment("3", "still | here"),
            segment("4", "low word"),
            segment("5", "Stop!"),
        ];
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "4".to_owned(),
            vec![OcrWordConfidence {
                text: "word".to_owned(),
                confidence: Some(69),
            }],
        );
        let candidates = detect_candidates(&original, &original, &metadata, &BTreeSet::new());
        assert_eq!(
            candidates.iter().map(|item| item.index).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn validation_rejects_overwriting_numbers_and_formatting() {
        assert!(validate_correction("Flight 123", "Flight 124", &[]).is_err());
        assert!(validate_correction("<i>Hello</i>", "Hello", &[]).is_err());
    }

    #[test]
    fn contract_rejects_missing_duplicate_reordered_and_empty_lines() {
        let mut segments = vec![segment("1", "| want"), segment("2", "Stop!")];
        let candidates = vec![
            Candidate {
                index: 0,
                reasons: vec!["deterministic_correction".to_owned()],
            },
            Candidate {
                index: 1,
                reasons: vec!["low_word_confidence".to_owned()],
            },
        ];
        let result = |lines: Vec<(&str, &str)>| OcrCorrectionResult {
            lines: lines
                .into_iter()
                .map(|(id, corrected_source)| OcrCorrectionLine {
                    id: id.to_owned(),
                    corrected_source: corrected_source.to_owned(),
                })
                .collect(),
        };
        assert!(validate_and_apply(&mut segments, &candidates, &result(vec![])).is_err());
        assert!(
            validate_and_apply(
                &mut segments,
                &candidates,
                &result(vec![("1", "I want"), ("1", "Stop!")])
            )
            .is_err()
        );
        assert!(
            validate_and_apply(
                &mut segments,
                &candidates,
                &result(vec![("2", "Stop!"), ("1", "I want")])
            )
            .is_err()
        );
        assert!(
            validate_and_apply(
                &mut segments,
                &candidates,
                &result(vec![("1", ""), ("2", "Stop!")])
            )
            .is_err()
        );
    }

    #[test]
    fn mode_defaults_and_explicit_overrides_control_model_failure() {
        let cases = [
            (TranslationMode::Economy, OcrCorrectionMode::Auto, false, 0),
            (TranslationMode::Turbo, OcrCorrectionMode::Auto, false, 1),
            (
                TranslationMode::Turbo,
                OcrCorrectionMode::Deterministic,
                false,
                0,
            ),
            (TranslationMode::Economy, OcrCorrectionMode::Model, false, 1),
            (TranslationMode::Cinema, OcrCorrectionMode::Off, false, 0),
            (TranslationMode::Cinema, OcrCorrectionMode::Model, true, 1),
        ];
        for (mode, correction, should_fail, expected_calls) in cases {
            let calls = Arc::new(AtomicUsize::new(0));
            let backend = ScriptedBackend {
                calls: calls.clone(),
                fail: true,
                invalid_first: false,
                corrected_source: "I want this",
            };
            let result = run(
                &mut pipeline(mode, correction, backend),
                &document("| want this"),
            );
            assert_eq!(result.is_err(), should_fail, "{mode:?} {correction:?}");
            assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
            if let Ok(result) = result
                && correction != OcrCorrectionMode::Off
            {
                assert_eq!(result.source_segments[0].text, "I want this");
                assert_eq!(
                    result.report.expect("OCR report").summary.fallback,
                    expected_calls > 0
                );
            }
        }
    }

    #[test]
    fn invalid_model_response_retries_and_cancellation_stops_before_call() {
        let calls = Arc::new(AtomicUsize::new(0));
        let backend = ScriptedBackend {
            calls: calls.clone(),
            fail: false,
            invalid_first: true,
            corrected_source: "I want this",
        };
        let mut retry_pipeline =
            pipeline(TranslationMode::Cinema, OcrCorrectionMode::Model, backend);
        retry_pipeline.options.execution.retries = 1;
        let corrected = run(&mut retry_pipeline, &document("| want this")).expect("retry succeeds");
        assert_eq!(corrected.source_segments[0].text, "I want this");
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        let calls = Arc::new(AtomicUsize::new(0));
        let backend = ScriptedBackend {
            calls: calls.clone(),
            fail: false,
            invalid_first: false,
            corrected_source: "I want this",
        };
        let token = CancellationToken::default();
        let guard = token.guard();
        token.cancel();
        let mut pipeline = pipeline(TranslationMode::Cinema, OcrCorrectionMode::Model, backend)
            .with_cancellation(guard);
        assert!(matches!(
            run(&mut pipeline, &document("| want this")),
            Err(CoreError::Cancelled)
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn non_english_text_skips_deterministic_repairs() {
        let calls = Arc::new(AtomicUsize::new(0));
        let backend = ScriptedBackend {
            calls,
            fail: false,
            invalid_first: false,
            corrected_source: "I want this",
        };
        let mut pipeline = pipeline(
            TranslationMode::Economy,
            OcrCorrectionMode::Deterministic,
            backend,
        );
        pipeline.options.validation.source_language = "French".to_owned();
        pipeline
            .options
            .ocr_source
            .as_mut()
            .expect("OCR source")
            .source_language = "fra".to_owned();
        let result = run(&mut pipeline, &document("| veux ceci")).expect("correction run");
        assert_eq!(result.source_segments[0].text, "| veux ceci");
    }

    #[test]
    fn model_can_locally_correct_another_ocr_error_in_a_candidate() {
        let calls = Arc::new(AtomicUsize::new(0));
        let backend = ScriptedBackend {
            calls,
            fail: false,
            invalid_first: false,
            corrected_source: "If I can",
        };
        let mut pipeline = pipeline(TranslationMode::Turbo, OcrCorrectionMode::Model, backend);
        let result = run(&mut pipeline, &document("lf | can")).expect("model correction");
        assert_eq!(result.source_segments[0].text, "If I can");
        let report = result.report.expect("OCR correction report");
        assert_eq!(report.summary.deterministic_corrections, 0);
        assert_eq!(report.summary.model_corrections, 1);
        assert_eq!(report.changes[0].original_source, "lf | can");
        assert_eq!(report.changes[0].corrected_source, "If I can");
    }
}
