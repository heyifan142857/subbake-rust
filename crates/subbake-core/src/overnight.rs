//! Pure planning and validation for provider-managed asynchronous batches.
//!
//! The domain deliberately only describes the economy translation contract.
//! Uploading JSONL, polling a provider, and writing manifests are adapter work.

use serde::{Deserialize, Serialize};

use std::collections::BTreeMap;

use crate::entities::{SubtitleDocument, SubtitleSegment, TranslationLine};
use crate::error::{CoreError, CoreResult};
use crate::formatting::{protect_formatting, restore_batch_formatting};
use crate::ports::ChatMessage;
use crate::term_matcher::TermMatcher;
use crate::validation::{FinalValidationPolicy, validate_final_output, validate_translation_batch};

pub const OVERNIGHT_TRANSLATION_CONTRACT_VERSION: u64 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OvernightBatch {
    pub custom_id: String,
    pub segment_ids: Vec<String>,
    pub messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone, Copy)]
pub struct OvernightPlanOptions<'a> {
    pub source_language: &'a str,
    pub target_language: &'a str,
    pub batch_size: usize,
    pub batch_token_budget: usize,
    pub id_prefix: &'a str,
    pub required_glossary: &'a BTreeMap<String, String>,
    pub preserve_names: bool,
}

/// Create compact, self-contained economy requests suitable for an external
/// batch queue. Context-dependent modes are intentionally excluded: a remote
/// job can complete out of order and must be safely recoverable from a manifest.
pub fn plan_translation(
    document: &SubtitleDocument,
    options: &OvernightPlanOptions<'_>,
) -> CoreResult<Vec<OvernightBatch>> {
    if options.batch_size == 0 {
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
            && (current.len() >= options.batch_size
                || (options.batch_token_budget > 0
                    && token_count.saturating_add(estimate) > options.batch_token_budget))
        {
            batches.push(build_batch(batches.len() + 1, &current, options));
            current.clear();
            token_count = 0;
        }
        current.push(segment.clone());
        token_count = token_count.saturating_add(estimate);
    }
    if !current.is_empty() {
        batches.push(build_batch(batches.len() + 1, &current, options));
    }
    Ok(batches)
}

pub fn finalize_translation_output(
    document: &SubtitleDocument,
    mut lines: Vec<TranslationLine>,
    required_glossary: &BTreeMap<String, String>,
    source_language: &str,
    target_language: &str,
    validation_policy: FinalValidationPolicy,
) -> CoreResult<Vec<SubtitleSegment>> {
    restore_batch_formatting(&document.segments, &mut lines);
    validate_translation_batch(&document.segments, &lines)?;
    let translations = lines
        .iter()
        .map(|line| (line.id.as_str(), line.translation.as_str()))
        .collect::<BTreeMap<_, _>>();
    let translated = document
        .segments
        .iter()
        .map(|segment| {
            let text = translations.get(segment.id.as_str()).ok_or_else(|| {
                CoreError::InvalidTranslation(format!(
                    "overnight output omitted subtitle id `{}`",
                    segment.id
                ))
            })?;
            let mut translated = segment.clone();
            translated.text = (*text).to_owned();
            Ok(translated)
        })
        .collect::<CoreResult<Vec<_>>>()?;
    validate_final_output(
        &document.segments,
        &translated,
        required_glossary,
        source_language,
        target_language,
        validation_policy,
    )?;
    Ok(translated)
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
            semantic: Default::default(),
        })
        .collect::<Vec<_>>();
    validate_translation_batch(&source, &lines)?;
    Ok(lines)
}

fn build_batch(
    index: usize,
    segments: &[SubtitleSegment],
    options: &OvernightPlanOptions<'_>,
) -> OvernightBatch {
    let batch_text = segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let relevant_glossary = options
        .required_glossary
        .iter()
        .filter(|(source, _)| TermMatcher::case_insensitive().contains(&batch_text, source))
        .map(|(source, target)| (source.clone(), target.clone()))
        .collect::<BTreeMap<_, _>>();
    let name_policy = if options.preserve_names {
        "Preserve personal names exactly in their source spelling unless the glossary requires another form."
    } else {
        "Translate or transliterate personal names into the target language's conventional script and keep them consistent."
    };
    let system = format!(
        "You translate subtitles from {} to {}. Preserve meaning, speaker labels, line breaks, and subtitle-safe brevity. {name_policy} Tokens shaped like ⟦SBK_FMT_<number>⟧ are protected subtitle formatting markers; copy each marker exactly once and in order. Glossary entries are user-required translations. Return JSON only: {{\"lines\":[{{\"id\":\"...\",\"translation\":\"...\"}}]}}. Return every id exactly once; do not add commentary.",
        options.source_language, options.target_language
    );
    let lines = segments
        .iter()
        .map(|segment| {
            serde_json::json!({
                "id": segment.id,
                "text": protect_formatting(&segment.text),
            })
        })
        .collect::<Vec<_>>();
    OvernightBatch {
        custom_id: format!("{}-{index:05}", options.id_prefix),
        segment_ids: segments.iter().map(|segment| segment.id.clone()).collect(),
        messages: vec![
            ChatMessage::cacheable_system(system),
            ChatMessage::user(
                serde_json::json!({"lines": lines, "glossary": relevant_glossary}).to_string(),
            ),
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
            metadata: Default::default(),
            segments: vec![
                SubtitleSegment {
                    id: "1".to_owned(),
                    text: "Hello".to_owned(),
                    start: None,
                    end: None,
                    identifier: None,
                    settings: None,
                    semantic: Default::default(),
                },
                SubtitleSegment {
                    id: "2".to_owned(),
                    text: "World".to_owned(),
                    start: None,
                    end: None,
                    identifier: None,
                    settings: None,
                    semantic: Default::default(),
                },
            ],
        }
    }

    #[test]
    fn plans_self_contained_economy_batches_and_validates_outputs() {
        let batches = plan_translation(
            &document(),
            &OvernightPlanOptions {
                source_language: "English",
                target_language: "Chinese",
                batch_size: 1,
                batch_token_budget: 100,
                id_prefix: "job",
                required_glossary: &BTreeMap::new(),
                preserve_names: false,
            },
        )
        .expect("plan");
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].custom_id, "job-00001");
        let output = parse_translation_output(
            &batches[0],
            &serde_json::json!({"lines":[{"id":"1","translation":"你好"}]}),
        )
        .expect("output");
        assert_eq!(output[0].translation, "你好");
    }

    #[test]
    fn overnight_contract_protects_formatting_and_selects_required_glossary() {
        let mut document = document();
        document.segments[0].text = "<i>Hello Alice</i>".to_owned();
        let batches = plan_translation(
            &document,
            &OvernightPlanOptions {
                source_language: "English",
                target_language: "Chinese",
                batch_size: 10,
                batch_token_budget: 1_000,
                id_prefix: "job",
                required_glossary: &BTreeMap::from([
                    ("Alice".to_owned(), "爱丽丝".to_owned()),
                    ("Missing".to_owned(), "缺失".to_owned()),
                ]),
                preserve_names: false,
            },
        )
        .expect("plan");
        let payload: serde_json::Value =
            serde_json::from_str(&batches[0].messages[1].content).expect("payload");

        assert_eq!(
            payload["lines"][0]["text"],
            "⟦SBK_FMT_0⟧Hello Alice⟦SBK_FMT_1⟧"
        );
        assert_eq!(payload["glossary"]["Alice"], "爱丽丝");
        assert!(payload["glossary"].get("Missing").is_none());
    }

    #[test]
    fn overnight_finalization_restores_formatting_and_enforces_glossary() {
        let mut document = document();
        document.segments[0].text = "<i>Hello Alice</i>".to_owned();
        let glossary = BTreeMap::from([("Alice".to_owned(), "爱丽丝".to_owned())]);
        let translated = finalize_translation_output(
            &document,
            vec![
                TranslationLine {
                    id: "1".to_owned(),
                    translation: "⟦SBK_FMT_0⟧你好，爱丽丝⟦SBK_FMT_1⟧".to_owned(),
                },
                TranslationLine {
                    id: "2".to_owned(),
                    translation: "世界".to_owned(),
                },
            ],
            &glossary,
            "English",
            "Chinese",
            FinalValidationPolicy::default(),
        )
        .expect("finalized");

        assert_eq!(translated[0].text, "<i>你好，爱丽丝</i>");

        let error = finalize_translation_output(
            &document,
            vec![
                TranslationLine {
                    id: "1".to_owned(),
                    translation: "你好".to_owned(),
                },
                TranslationLine {
                    id: "2".to_owned(),
                    translation: "世界".to_owned(),
                },
            ],
            &glossary,
            "English",
            "Chinese",
            FinalValidationPolicy::default(),
        )
        .expect_err("missing glossary must fail");
        assert!(error.to_string().contains("Alice"));
    }
}
