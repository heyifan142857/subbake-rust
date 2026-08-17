use std::collections::HashMap;

use crate::entities::{ConfirmedTranslationContext, SubtitleSegment, TranslationLine};
use crate::error::{CoreError, CoreResult};

use super::BatchWithUsage;
use super::support::{apply_lines, merge_translation_lines};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct SourceBatchContext {
    pub before: Vec<SubtitleSegment>,
    pub after: Vec<SubtitleSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct TranslationPromptContext {
    pub source: SourceBatchContext,
    pub previous_confirmed: Vec<ConfirmedTranslationContext>,
}

impl TranslationPromptContext {
    pub fn for_left_split(&self, right: &[SubtitleSegment]) -> Self {
        let mut context = self.clone();
        context.source.after = right.iter().cloned().chain(context.source.after).collect();
        context
    }

    pub fn for_right_split(&self, left: &[SubtitleSegment]) -> Self {
        let mut context = self.clone();
        context.source.before.extend_from_slice(left);
        context
    }
}

pub(super) struct PreparedBatch {
    pub index: usize,
    pub memory_hits: HashMap<String, String>,
    pub pending: Vec<SubtitleSegment>,
    pub prompt_context: TranslationPromptContext,
}

pub(super) struct AppliedBatch {
    pub index: usize,
    pub translated: Vec<SubtitleSegment>,
    pub result: Option<BatchWithUsage>,
}

/// Owns translation-stage progress and deterministic result assembly.
///
/// Backend generation and persistence remain with the pipeline orchestrator;
/// window selection, translation-memory lookup, and ordered output assembly are
/// kept here so partially resumed runs have one source of truth.
pub(super) struct TranslationStage {
    batches: Vec<Vec<SubtitleSegment>>,
    output: Vec<SubtitleSegment>,
    next_batch: usize,
    memory_hits: usize,
    memory_keys: HashMap<String, String>,
}

impl TranslationStage {
    pub fn new(
        batches: Vec<Vec<SubtitleSegment>>,
        resumed: usize,
        output: Vec<SubtitleSegment>,
        memory_keys: HashMap<String, String>,
    ) -> CoreResult<Self> {
        if resumed > batches.len() {
            return Err(CoreError::DataInvariant(format!(
                "resume state has {resumed} translated batches, but the current input has only {}",
                batches.len()
            )));
        }
        let expected = batches.iter().take(resumed).map(Vec::len).sum::<usize>();
        if output.len() != expected {
            return Err(CoreError::DataInvariant(format!(
                "translation stage expected {expected} resumed segments, but received {}",
                output.len()
            )));
        }
        Ok(Self {
            batches,
            output,
            next_batch: resumed,
            memory_hits: 0,
            memory_keys,
        })
    }

    pub fn batches(&self) -> &[Vec<SubtitleSegment>] {
        &self.batches
    }

    pub fn len(&self) -> usize {
        self.batches.len()
    }

    pub fn next_batch(&self) -> usize {
        self.next_batch
    }

    pub fn is_complete(&self) -> bool {
        self.next_batch == self.batches.len()
    }

    pub fn prepare_window(
        &self,
        concurrency: usize,
        use_cache: bool,
        memory: &HashMap<String, String>,
    ) -> Vec<PreparedBatch> {
        self.batches
            .iter()
            .enumerate()
            .skip(self.next_batch)
            .take(concurrency.max(1))
            .map(|(index, batch)| {
                let mut memory_hits = HashMap::new();
                let mut pending = Vec::new();
                for segment in batch {
                    let key = self
                        .memory_keys
                        .get(&segment.id)
                        .map(String::as_str)
                        .unwrap_or_default();
                    if use_cache
                        && !key.is_empty()
                        && let Some(text) = memory.get(key)
                    {
                        memory_hits.insert(segment.id.clone(), text.clone());
                    } else {
                        pending.push(segment.clone());
                    }
                }
                PreparedBatch {
                    index,
                    memory_hits,
                    pending,
                    prompt_context: TranslationPromptContext::default(),
                }
            })
            .collect()
    }

    pub fn apply(
        &mut self,
        prepared: PreparedBatch,
        result: Option<BatchWithUsage>,
    ) -> CoreResult<AppliedBatch> {
        if prepared.index != self.next_batch {
            return Err(CoreError::DataInvariant(format!(
                "translation stage expected batch {}, but received batch {}",
                self.next_batch + 1,
                prepared.index + 1
            )));
        }
        if prepared.pending.is_empty() != result.is_none() {
            return Err(CoreError::DataInvariant(format!(
                "translation result availability does not match pending lines for batch {}",
                prepared.index + 1
            )));
        }
        let source = self.batches[prepared.index].clone();
        let new_lines: &[TranslationLine] = result
            .as_ref()
            .map(|value| value.lines.as_slice())
            .unwrap_or_default();
        let merged = merge_translation_lines(&source, &prepared.memory_hits, new_lines);
        let translated = apply_lines(&source, &merged);
        self.memory_hits += prepared.memory_hits.len();
        self.output.extend(translated.iter().cloned());
        self.next_batch += 1;
        Ok(AppliedBatch {
            index: prepared.index,
            translated,
            result,
        })
    }

    pub fn memory_hits(&self) -> usize {
        self.memory_hits
    }

    pub fn output(&self) -> &[SubtitleSegment] {
        &self.output
    }

    pub fn previous_confirmed_context(
        &self,
        max_lines: usize,
        token_budget: usize,
    ) -> Vec<ConfirmedTranslationContext> {
        let Some(previous_index) = self.next_batch.checked_sub(1) else {
            return Vec::new();
        };
        let Some(source) = self.batches.get(previous_index) else {
            return Vec::new();
        };
        let translated_start = self.output.len().saturating_sub(source.len());
        let context = source
            .iter()
            .zip(&self.output[translated_start..])
            .map(|(source, translated)| ConfirmedTranslationContext {
                id: source.id.clone(),
                source: source.text.clone(),
                translation: translated.text.clone(),
            })
            .collect::<Vec<_>>();
        bounded_confirmed_context(&context, max_lines, token_budget)
    }

    pub fn finish(self) -> Vec<SubtitleSegment> {
        self.output
    }
}

pub(super) fn bounded_confirmed_context(
    context: &[ConfirmedTranslationContext],
    max_lines: usize,
    token_budget: usize,
) -> Vec<ConfirmedTranslationContext> {
    if max_lines == 0 || token_budget == 0 {
        return Vec::new();
    }
    let mut selected = Vec::new();
    let mut tokens = 0usize;
    for line in context.iter().rev().take(max_lines) {
        let estimate = super::planning::estimated_text_tokens(&line.source)
            .saturating_add(super::planning::estimated_text_tokens(&line.translation))
            .saturating_add(8);
        if !selected.is_empty() && tokens.saturating_add(estimate) > token_budget {
            break;
        }
        selected.push(line.clone());
        tokens = tokens.saturating_add(estimate);
    }
    selected.reverse();
    selected
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
            semantic: Default::default(),
        }
    }

    #[test]
    fn prepares_and_applies_translation_memory_hits_in_order() {
        let batches = vec![vec![segment("1", "Hello"), segment("2", "world")]];
        let memory_key = "ctx-v2:test:hello".to_owned();
        let memory_keys = HashMap::from([
            ("1".to_owned(), memory_key.clone()),
            ("2".to_owned(), "ctx-v2:test:world".to_owned()),
        ]);
        let mut stage = TranslationStage::new(batches, 0, Vec::new(), memory_keys).expect("stage");
        let memory = HashMap::from([(memory_key, "Bonjour".to_owned())]);
        let mut prepared = stage.prepare_window(1, true, &memory);
        assert_eq!(prepared[0].pending, vec![segment("2", "world")]);
        let applied = stage
            .apply(
                prepared.remove(0),
                Some(BatchWithUsage {
                    lines: vec![TranslationLine {
                        id: "2".to_owned(),
                        translation: "monde".to_owned(),
                    }],
                    summary: String::new(),
                    glossary_updates: Vec::new(),
                    terminology_updates: Vec::new(),
                    usage: Default::default(),
                    cache_key: None,
                }),
            )
            .expect("apply");
        assert_eq!(
            applied
                .translated
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<Vec<_>>(),
            vec!["Bonjour", "monde"]
        );
        assert_eq!(stage.memory_hits(), 1);
        assert!(stage.is_complete());
    }

    #[test]
    fn confirmed_context_keeps_only_the_recent_bounded_tail() {
        let context = (1..=20)
            .map(|id| ConfirmedTranslationContext {
                id: id.to_string(),
                source: format!("source {id}"),
                translation: format!("translation {id}"),
            })
            .collect::<Vec<_>>();

        let bounded = bounded_confirmed_context(&context, 12, 10_000);

        assert_eq!(bounded.len(), 12);
        assert_eq!(bounded.first().map(|line| line.id.as_str()), Some("9"));
        assert_eq!(bounded.last().map(|line| line.id.as_str()), Some("20"));

        let token_bounded = bounded_confirmed_context(&context, 12, 30);
        assert!(!token_bounded.is_empty());
        assert!(token_bounded.len() < 12);
        assert_eq!(
            token_bounded.last().map(|line| line.id.as_str()),
            Some("20")
        );
    }
}
