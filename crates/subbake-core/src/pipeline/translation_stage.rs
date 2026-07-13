use std::collections::HashMap;

use crate::entities::{SubtitleSegment, TranslationLine};
use crate::error::{CoreError, CoreResult};

use super::{BatchWithUsage, apply_lines, merge_translation_lines, translation_memory_key};

pub(super) struct PreparedBatch {
    pub index: usize,
    pub memory_hits: HashMap<String, String>,
    pub pending: Vec<SubtitleSegment>,
}

pub(super) struct AppliedBatch {
    pub index: usize,
    pub source: Vec<SubtitleSegment>,
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
}

impl TranslationStage {
    pub fn new(
        batches: Vec<Vec<SubtitleSegment>>,
        resumed: usize,
        output: Vec<SubtitleSegment>,
    ) -> CoreResult<Self> {
        if resumed > batches.len() {
            return Err(CoreError::Data(format!(
                "resume state has {resumed} translated batches, but the current input has only {}",
                batches.len()
            )));
        }
        let expected = batches.iter().take(resumed).map(Vec::len).sum::<usize>();
        if output.len() != expected {
            return Err(CoreError::Data(format!(
                "translation stage expected {expected} resumed segments, but received {}",
                output.len()
            )));
        }
        Ok(Self {
            batches,
            output,
            next_batch: resumed,
            memory_hits: 0,
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
                    let key = translation_memory_key(&segment.text);
                    if use_cache
                        && !key.is_empty()
                        && let Some(text) = memory.get(&key)
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
            return Err(CoreError::Data(format!(
                "translation stage expected batch {}, but received batch {}",
                self.next_batch + 1,
                prepared.index + 1
            )));
        }
        if prepared.pending.is_empty() != result.is_none() {
            return Err(CoreError::Data(format!(
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
            source,
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

    pub fn finish(self) -> Vec<SubtitleSegment> {
        self.output
    }
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
        }
    }

    #[test]
    fn prepares_and_applies_translation_memory_hits_in_order() {
        let batches = vec![vec![segment("1", "Hello"), segment("2", "world")]];
        let mut stage = TranslationStage::new(batches, 0, Vec::new()).expect("stage");
        let memory = HashMap::from([(translation_memory_key("Hello"), "Bonjour".to_owned())]);
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
                    usage: Default::default(),
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
}
