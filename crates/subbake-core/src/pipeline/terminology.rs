use std::time::Instant;

use crate::CancellationGuard;
use crate::entities::{
    GlossaryEntry, PipelineOptions, SubtitleDocument, SubtitleSegment, TerminologyEntity,
    TerminologyPreflightResult, TerminologyStats, Usage,
};
use crate::error::{CoreError, CoreResult};
use crate::memory::{ContextMemory, english_possessive_base};
use crate::ports::{
    BackendJsonResult, BackendPayload, CacheStage, DashboardSink, GenerationRequest, LlmBackend,
    RuntimeStore,
};
use crate::progress::{ProgressEvent, ProgressSink, ProgressUnit, TaskKind, TaskState};

pub(super) struct TerminologyStage<'a, B, D> {
    pub backend: &'a mut B,
    pub dashboard: &'a mut D,
    pub options: &'a PipelineOptions,
    pub memory: &'a mut ContextMemory,
    pub store: Option<&'a dyn RuntimeStore>,
    pub cancellation: &'a CancellationGuard,
    pub progress: Option<&'a dyn ProgressSink>,
    pub cache_hits: &'a mut usize,
}

impl<B, D> TerminologyStage<'_, B, D>
where
    B: LlmBackend,
    D: DashboardSink,
{
    pub(super) fn run(&mut self, document: &SubtitleDocument) -> CoreResult<TerminologyStats> {
        let started = Instant::now();
        let candidates = extract_candidates(&document.segments);
        self.memory.terminology_candidates = candidates
            .iter()
            .map(|candidate| candidate.source.clone())
            .collect();
        self.memory.name_candidates = candidates
            .iter()
            .filter(|candidate| candidate.align_as_name)
            .map(|candidate| candidate.source.clone())
            .collect();
        let mut stats = TerminologyStats {
            candidates: candidates.len(),
            ..TerminologyStats::default()
        };
        if !self.options.terminology_preflight
            || !self.options.policy().document_preflight
            || candidates.is_empty()
            || !self.backend.supports_terminology_preflight()
        {
            self.report(TaskState::Skipped, 0, candidates.len(), Usage::default());
            return Ok(stats);
        }

        self.report(TaskState::Running, 0, candidates.len(), Usage::default());
        let existing = self.memory.glossary.clone();
        let messages = build_messages(self.options, &candidates);
        let hash = super::support::request_hash(self.options, CacheStage::Terminology, &messages);
        let cached = if self.options.use_cache {
            self.store
                .map(|store| store.load_cached_response(CacheStage::Terminology, &hash))
                .transpose()?
                .flatten()
        } else {
            None
        };
        let result = if let Some(response) = cached {
            stats.cache_hits = 1;
            *self.cache_hits += 1;
            Ok(response)
        } else {
            self.generate(&messages, &candidates)
        };

        match result {
            Ok(response) => {
                let BackendPayload::Terminology(payload) = response.payload else {
                    return Err(CoreError::DataInvariant(
                        "terminology cache returned a different payload".to_owned(),
                    ));
                };
                stats.usage = if stats.cache_hits > 0 {
                    Usage::default()
                } else {
                    response.usage
                };
                let document_brief = payload.document_brief;
                let accepted = accept_entries(self.memory, payload.entries, &mut stats);
                let accepted_entities = accept_entities(self.memory, payload.entities, &mut stats);
                if self.options.mode == crate::entities::TranslationMode::Cinema
                    && !document_brief.trim().is_empty()
                {
                    let brief = document_brief.trim().chars().take(800).collect::<String>();
                    self.memory
                        .style_rules
                        .push(format!("Document brief: {brief}"));
                }
                stats.entries_added = self.memory.glossary.len().saturating_sub(existing.len());
                if self.options.use_cache
                    && stats.cache_hits == 0
                    && let Some(store) = self.store
                {
                    store.save_cached_response(
                        CacheStage::Terminology,
                        &hash,
                        &BackendJsonResult {
                            payload: BackendPayload::Terminology(TerminologyPreflightResult {
                                entries: accepted,
                                entities: accepted_entities,
                                document_brief,
                            }),
                            usage: response.usage,
                        },
                    )?;
                }
                if let Some(store) = self.store {
                    store.save_glossary(
                        &self
                            .memory
                            .glossary
                            .iter()
                            .map(|(source, target)| (source.clone(), target.clone()))
                            .collect::<Vec<_>>(),
                    )?;
                }
            }
            Err(error) => {
                stats.degraded = true;
                stats.degraded_reason = Some(error.to_string());
            }
        }
        stats.duration_ms = super::support::duration_ms(started);
        self.dashboard.add_usage(stats.usage);
        self.report(
            TaskState::Completed,
            candidates.len(),
            candidates.len(),
            stats.usage,
        );
        Ok(stats)
    }

    fn generate(
        &mut self,
        messages: &[crate::ports::ChatMessage],
        candidates: &[TerminologyCandidate],
    ) -> CoreResult<BackendJsonResult> {
        let mut last_error = None;
        for _ in 0..=self.options.retries {
            self.cancellation.check()?;
            let response = self
                .backend
                .execute(
                    GenerationRequest::json(messages.to_vec()).without_reasoning(),
                    self.cancellation,
                )
                .map_err(CoreError::from)
                .and_then(|response| response.into_json().map_err(CoreError::from))
                .and_then(|(json, usage)| {
                    Ok(BackendJsonResult {
                        payload: BackendPayload::Terminology(parse_payload(&json, candidates)?),
                        usage,
                    })
                });
            match response {
                Ok(value) => return Ok(value),
                Err(CoreError::Cancelled) => return Err(CoreError::Cancelled),
                Err(error) if super::support::is_operational_llm_failure(&error) => {
                    return Err(error);
                }
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error.unwrap_or_else(|| {
            CoreError::InvalidBackendResponse("terminology preflight failed".to_owned())
        }))
    }

    fn report(&self, state: TaskState, current: usize, total: usize, usage: Usage) {
        if let Some(progress) = self.progress {
            progress.emit(ProgressEvent {
                task: TaskKind::Translation,
                stage: "TERMINOLOGY_PREFLIGHT".to_owned(),
                state,
                current: current as u64,
                total: Some(total as u64),
                unit: ProgressUnit::Batches,
                resumed: 0,
                usage,
                message: None,
                translation: None,
            });
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct TerminologyCandidate {
    pub(super) source: String,
    pub(super) context: String,
    pub(super) align_as_name: bool,
}

pub(super) fn extract_candidates(segments: &[SubtitleSegment]) -> Vec<TerminologyCandidate> {
    #[derive(Clone)]
    struct RankedCandidate {
        candidate: TerminologyCandidate,
        count: usize,
        words: usize,
        acronym: bool,
        honorific: bool,
    }

    let mut candidates = std::collections::BTreeMap::<String, RankedCandidate>::new();
    for segment in segments {
        let words = segment
            .text
            .split_whitespace()
            .map(|word| {
                let word = word.trim_matches(|ch: char| {
                    !ch.is_alphanumeric() && ch != '-' && ch != '\'' && ch != '’'
                });
                english_possessive_base(word).unwrap_or(word)
            })
            .filter(|word| word.len() >= 2)
            .collect::<Vec<_>>();
        let mut index = 0;
        while index < words.len() {
            let word = words[index];
            let is_title = word
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_uppercase())
                && word.chars().skip(1).any(|ch| ch.is_ascii_lowercase());
            let is_acronym = word.chars().filter(|ch| ch.is_ascii_alphabetic()).count() >= 2
                && word
                    .chars()
                    .filter(|ch| ch.is_ascii_alphabetic())
                    .all(|ch| ch.is_ascii_uppercase());
            if !is_title && !is_acronym {
                index += 1;
                continue;
            }
            let mut end = index + 1;
            while end < words.len() && end - index < 4 {
                if !words[end]
                    .chars()
                    .next()
                    .is_some_and(|ch| ch.is_ascii_uppercase())
                {
                    break;
                }
                end += 1;
            }
            let source = words[index..end].join(" ");
            let word_count = end - index;
            candidates
                .entry(source.to_ascii_lowercase())
                .and_modify(|candidate| candidate.count += 1)
                .or_insert_with(|| RankedCandidate {
                    candidate: TerminologyCandidate {
                        source,
                        context: segment.text.chars().take(240).collect(),
                        align_as_name: false,
                    },
                    count: 1,
                    words: word_count,
                    acronym: is_acronym,
                    honorific: false,
                });
            index = end;
        }
        for source in japanese_honorific_names(&segment.text) {
            candidates
                .entry(source.to_lowercase())
                .and_modify(|candidate| candidate.count += 1)
                .or_insert_with(|| RankedCandidate {
                    candidate: TerminologyCandidate {
                        source,
                        context: segment.text.chars().take(240).collect(),
                        align_as_name: true,
                    },
                    count: 1,
                    words: 1,
                    acronym: false,
                    honorific: true,
                });
        }
    }
    let mut ranked = candidates
        .into_values()
        .filter(|candidate| {
            candidate.honorific
                || candidate.acronym
                || (candidate.words > 1
                    && candidate
                        .candidate
                        .source
                        .split_whitespace()
                        .next()
                        .is_some_and(|word| !is_common_sentence_initial(word)))
                || (candidate.count > 1 && !is_common_sentence_initial(&candidate.candidate.source))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| {
                right
                    .candidate
                    .source
                    .split_whitespace()
                    .count()
                    .cmp(&left.candidate.source.split_whitespace().count())
            })
            .then_with(|| {
                left.candidate
                    .source
                    .to_lowercase()
                    .cmp(&right.candidate.source.to_lowercase())
            })
    });
    ranked
        .into_iter()
        .map(|mut candidate| {
            candidate.candidate.align_as_name =
                candidate.honorific || (!candidate.acronym && candidate.count > 1);
            candidate.candidate
        })
        .take(256)
        .collect()
}

fn japanese_honorific_names(text: &str) -> Vec<String> {
    const HONORIFICS: [&str; 8] = ["ちゃん", "さん", "さま", "先生", "博士", "君", "様", "氏"];
    let mut names = Vec::new();
    for honorific in HONORIFICS {
        for (honorific_at, _) in text.match_indices(honorific) {
            let mut start = honorific_at;
            let mut length = 0;
            for (index, character) in text[..honorific_at].char_indices().rev() {
                if length == 8 || !is_japanese_name_character(character) {
                    break;
                }
                start = index;
                length += 1;
            }
            if length < 2 {
                continue;
            }
            let candidate = &text[start..honorific_at];
            if !names.iter().any(|name| name == candidate) {
                names.push(candidate.to_owned());
            }
        }
    }
    names
}

fn is_japanese_name_character(character: char) -> bool {
    matches!(
        character,
        '\u{3400}'..='\u{9fff}'
            | '\u{30a0}'..='\u{30ff}'
            | '\u{ff66}'..='\u{ff9f}'
            | '·'
    )
}

fn is_common_sentence_initial(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "but"
            | "can"
            | "do"
            | "for"
            | "from"
            | "he"
            | "her"
            | "here"
            | "his"
            | "how"
            | "i"
            | "if"
            | "in"
            | "is"
            | "it"
            | "its"
            | "my"
            | "no"
            | "not"
            | "of"
            | "official"
            | "oh"
            | "on"
            | "or"
            | "our"
            | "she"
            | "so"
            | "that"
            | "the"
            | "their"
            | "then"
            | "there"
            | "these"
            | "they"
            | "this"
            | "those"
            | "to"
            | "we"
            | "what"
            | "when"
            | "where"
            | "who"
            | "why"
            | "will"
            | "with"
            | "yes"
            | "you"
            | "your"
    )
}

fn build_messages(
    options: &PipelineOptions,
    candidates: &[TerminologyCandidate],
) -> Vec<crate::ports::ChatMessage> {
    let payload = serde_json::json!({
        "source_language": options.source_language,
        "target_language": options.target_language,
        "candidates": candidates.iter().map(|candidate| serde_json::json!({
            "source": candidate.source,
            "context": candidate.context,
        })).collect::<Vec<_>>(),
    });
    let payload = serde_json::to_string(&payload).unwrap_or_default();
    let name_policy = if options.preserve_names {
        "For personal names, use the exact source spelling as target."
    } else {
        "Include every clearly identified personal name and translate or transliterate it into the target language's conventional script; do not omit a clear personal name merely because its canonical spelling is uncertain."
    };
    vec![
        crate::ports::ChatMessage::system(format!(
            "TASK_START\nextract_terminology\nTASK_END\nReturn JSON only as {{\"entries\":[{{\"source\":\"exact candidate\",\"target\":\"canonical translation\"}}],\"entities\":[{{\"canonical_source\":\"canonical entity name\",\"kind\":\"person|organization|place|proper_name|domain_term\",\"variants\":[{{\"source\":\"exact candidate\",\"target\":\"natural translation of that source form\"}}]}}],\"document_brief\":\"short genre, tone, relationship, and register guidance\"}}. Include only names, titles, organizations, places, recurring objects, and domain terms whose translation should stay consistent. Group aliases of one entity while preserving the natural granularity of each source form. {name_policy} Copy every source form exactly from a candidate. Omit ordinary sentence-initial words and uncertain non-name entries. The brief must be short and advisory; never invent plot facts."
        )),
        crate::ports::ChatMessage::user(format!(
            "TERMINOLOGY_JSON_START{payload}TERMINOLOGY_JSON_END"
        )),
    ]
}

pub(super) fn parse_payload(
    payload: &serde_json::Value,
    candidates: &[TerminologyCandidate],
) -> CoreResult<TerminologyPreflightResult> {
    let entries = payload
        .get("entries")
        .or_else(|| payload.get("glossary"))
        .and_then(serde_json::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let mut parsed = Vec::new();
    for entry in entries {
        let source = entry["source"].as_str().unwrap_or_default().trim();
        let target = entry["target"].as_str().unwrap_or_default().trim();
        if source.is_empty() || target.is_empty() {
            return Err(CoreError::InvalidTranslation(
                "terminology entry contains an empty source or target".to_owned(),
            ));
        }
        let Some(candidate) = candidates
            .iter()
            .find(|candidate| candidate.source.eq_ignore_ascii_case(source))
        else {
            return Err(CoreError::InvalidTranslation(format!(
                "terminology response contains unknown source `{source}`"
            )));
        };
        parsed.push(GlossaryEntry {
            source: candidate.source.clone(),
            target: target.to_owned(),
        });
    }
    let entities = serde_json::from_value::<Vec<TerminologyEntity>>(payload["entities"].clone())
        .unwrap_or_default();
    for entity in &entities {
        if entity.canonical_source.trim().is_empty() || entity.variants.is_empty() {
            return Err(CoreError::InvalidTranslation(
                "terminology entity is missing a canonical source or variants".to_owned(),
            ));
        }
        for variant in &entity.variants {
            if variant.target.trim().is_empty()
                || !candidates
                    .iter()
                    .any(|candidate| candidate.source.eq_ignore_ascii_case(variant.source.trim()))
            {
                return Err(CoreError::InvalidTranslation(format!(
                    "terminology entity contains unknown or empty variant `{}`",
                    variant.source
                )));
            }
        }
    }
    Ok(TerminologyPreflightResult {
        entries: parsed,
        entities,
        document_brief: payload["document_brief"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
    })
}

fn accept_entities(
    memory: &mut ContextMemory,
    entities: Vec<TerminologyEntity>,
    stats: &mut TerminologyStats,
) -> Vec<TerminologyEntity> {
    let mut accepted = Vec::new();
    for mut entity in entities {
        let mut variants = Vec::new();
        for variant in std::mem::take(&mut entity.variants) {
            match memory.glossary.get(&variant.source) {
                Some(current) if current.eq_ignore_ascii_case(&variant.target) => {
                    variants.push(variant);
                }
                Some(_) => stats.conflicts_omitted += 1,
                None => {
                    memory.update("", std::slice::from_ref(&variant));
                    variants.push(variant);
                }
            }
        }
        if variants.is_empty() {
            continue;
        }
        entity.variants = variants;
        memory.add_terminology_entity(entity.clone());
        accepted.push(entity);
    }
    accepted
}

fn accept_entries(
    memory: &mut ContextMemory,
    entries: Vec<GlossaryEntry>,
    stats: &mut TerminologyStats,
) -> Vec<GlossaryEntry> {
    let mut accepted = Vec::new();
    for entry in entries {
        if let Some(current) = memory.glossary.get(&entry.source) {
            if !current.eq_ignore_ascii_case(&entry.target) {
                stats.conflicts_omitted += 1;
            }
            continue;
        }
        if accepted.iter().any(|value: &GlossaryEntry| {
            value.source.eq_ignore_ascii_case(&entry.source)
                && !value.target.eq_ignore_ascii_case(&entry.target)
        }) {
            stats.conflicts_omitted += 1;
            continue;
        }
        accepted.push(entry);
    }
    memory.update("", &accepted);
    accepted
}
