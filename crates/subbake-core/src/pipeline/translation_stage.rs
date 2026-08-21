use std::collections::{BTreeSet, HashMap, HashSet};

use crate::entities::{ConfirmedTranslationContext, SubtitleSegment, TranslationLine};
use crate::error::{CoreError, CoreResult};
use crate::language_rules::CjkRules;

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
    pub relevant_previous: Vec<ConfirmedTranslationContext>,
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

    pub fn relevant_previous_context(
        &self,
        query: &[SubtitleSegment],
        excluded_ids: &HashSet<&str>,
        max_lines: usize,
        token_budget: usize,
    ) -> Vec<ConfirmedTranslationContext> {
        if query.is_empty() || max_lines == 0 || token_budget == 0 {
            return Vec::new();
        }
        let query_terms = query
            .iter()
            .map(|segment| retrieval_terms(&segment.text))
            .collect::<Vec<_>>();
        let confirmed_source = self
            .batches
            .iter()
            .take(self.next_batch)
            .flatten()
            .collect::<Vec<_>>();
        let mut ranked = confirmed_source
            .into_iter()
            .zip(&self.output)
            .enumerate()
            .filter(|(_, (source, _))| !excluded_ids.contains(source.id.as_str()))
            .filter_map(|(index, (source, translated))| {
                let candidate_terms = retrieval_terms(&source.text);
                let score = query
                    .iter()
                    .zip(&query_terms)
                    .map(|(query, terms)| {
                        retrieval_score(&query.text, terms, &source.text, &candidate_terms)
                    })
                    .max()
                    .unwrap_or_default();
                (score > 0).then(|| {
                    (
                        score,
                        index,
                        ConfirmedTranslationContext {
                            id: source.id.clone(),
                            source: source.text.clone(),
                            translation: translated.text.clone(),
                        },
                    )
                })
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| right.1.cmp(&left.1)));
        let mut selected = Vec::new();
        let mut tokens = 0usize;
        for (_, index, line) in ranked.into_iter().take(max_lines.saturating_mul(3)) {
            let estimate = super::planning::estimated_text_tokens(&line.source)
                .saturating_add(super::planning::estimated_text_tokens(&line.translation))
                .saturating_add(8);
            if !selected.is_empty() && tokens.saturating_add(estimate) > token_budget {
                continue;
            }
            selected.push((index, line));
            tokens = tokens.saturating_add(estimate);
            if selected.len() == max_lines {
                break;
            }
        }
        selected.sort_by_key(|(index, _)| *index);
        selected.into_iter().map(|(_, line)| line).collect()
    }

    pub fn finish(self) -> Vec<SubtitleSegment> {
        self.output
    }
}

fn retrieval_score(
    query_text: &str,
    query_terms: &BTreeSet<String>,
    candidate_text: &str,
    candidate_terms: &BTreeSet<String>,
) -> usize {
    let query_normalized = normalized_retrieval_text(query_text);
    let candidate_normalized = normalized_retrieval_text(candidate_text);
    if !query_normalized.is_empty() && query_normalized == candidate_normalized {
        return 10_000;
    }
    let shared = query_terms
        .intersection(candidate_terms)
        .collect::<Vec<_>>();
    if shared.len() < 2 && !shared.iter().any(|term| term.chars().count() >= 6) {
        return 0;
    }
    let shared_weight = shared
        .iter()
        .map(|term| term.chars().count().min(12))
        .sum::<usize>();
    let denominator = candidate_terms.len().max(query_terms.len()).max(1);
    shared_weight
        .saturating_mul(100)
        .saturating_add(shared.len().saturating_mul(100) / denominator)
}

fn retrieval_terms(text: &str) -> BTreeSet<String> {
    const STOPWORDS: &[&str] = &[
        "and", "are", "but", "for", "from", "have", "just", "that", "the", "this", "was", "what",
        "when", "with", "you", "your",
    ];
    let mut terms = BTreeSet::new();
    let mut current = String::new();
    let flush = |current: &mut String, terms: &mut BTreeSet<String>| {
        if current.is_empty() {
            return;
        }
        let value = current.to_lowercase();
        let length = value.chars().count();
        if length >= 2 && !STOPWORDS.contains(&value.as_str()) {
            terms.insert(value.clone());
        }
        let characters = value.chars().collect::<Vec<_>>();
        if characters
            .iter()
            .all(|character| CjkRules::is_han_character(*character))
            && characters.len() > 2
        {
            for pair in characters.windows(2) {
                terms.insert(pair.iter().collect());
            }
        }
        current.clear();
    };
    for character in text.chars() {
        if character.is_alphanumeric() || CjkRules::is_han_character(character) {
            current.push(character);
        } else {
            flush(&mut current, &mut terms);
        }
    }
    flush(&mut current, &mut terms);
    terms
}

fn normalized_retrieval_text(text: &str) -> String {
    text.chars()
        .filter(|character| character.is_alphanumeric() || CjkRules::is_han_character(*character))
        .flat_map(char::to_lowercase)
        .collect()
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

    #[test]
    fn relevant_previous_retrieves_a_distant_match_and_excludes_recent_context() {
        let batches = vec![
            vec![segment("1", "The quantum beacon accepted the override.")],
            vec![segment("2", "Nothing else matters here.")],
            vec![segment("3", "Check the quantum beacon again.")],
        ];
        let mut stage =
            TranslationStage::new(batches, 0, Vec::new(), HashMap::new()).expect("stage");
        for (id, translation) in [("1", "量子信标接受了覆盖指令。"), ("2", "别的都不重要。")]
        {
            let mut prepared = stage.prepare_window(1, false, &HashMap::new());
            stage
                .apply(
                    prepared.remove(0),
                    Some(BatchWithUsage {
                        lines: vec![TranslationLine {
                            id: id.to_owned(),
                            translation: translation.to_owned(),
                        }],
                        summary: String::new(),
                        glossary_updates: Vec::new(),
                        terminology_updates: Vec::new(),
                        usage: Default::default(),
                        cache_key: None,
                    }),
                )
                .expect("apply confirmed batch");
        }

        let excluded = HashSet::from(["2"]);
        let relevant = stage.relevant_previous_context(
            &[segment("3", "Check the quantum beacon again.")],
            &excluded,
            4,
            600,
        );

        assert_eq!(relevant.len(), 1);
        assert_eq!(relevant[0].id, "1");
        assert_eq!(relevant[0].translation, "量子信标接受了覆盖指令。");
    }
}
