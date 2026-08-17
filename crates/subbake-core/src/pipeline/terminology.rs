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
use crate::term_matcher::TermMatcher;

pub(super) struct TerminologyStage<'a, B, D> {
    pub backend: &'a mut B,
    pub dashboard: &'a mut D,
    pub options: &'a PipelineOptions,
    pub memory: &'a mut ContextMemory,
    pub store: Option<&'a dyn RuntimeStore>,
    pub cancellation: &'a CancellationGuard,
    pub progress: Option<&'a dyn ProgressSink>,
    pub cache_hits: &'a mut usize,
    pub provider_requests: &'a mut usize,
    pub provider_tokens: &'a mut usize,
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
        if !self.options.terminology_preflight {
            self.report(TaskState::Skipped, 0, candidates.len(), Usage::default());
            return Ok(stats);
        }
        if !self.backend.supports_terminology_preflight() {
            if !self.options.allow_degraded_preflight {
                return Err(CoreError::UnsupportedCapability(
                    "configured backend does not support strict terminology preflight".to_owned(),
                ));
            }
            stats.degraded = true;
            stats.degraded_reason =
                Some("configured backend does not support terminology preflight".to_owned());
            self.report(TaskState::Skipped, 0, candidates.len(), Usage::default());
            return Ok(stats);
        }

        self.report(TaskState::Running, 0, candidates.len(), Usage::default());
        let existing = self.memory.glossary.clone();
        let messages = build_messages(self.options, &candidates, &document.segments);
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
            self.generate(&messages, &candidates, &document.segments)
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
            Err(error @ CoreError::ResourceBudgetExceeded(_)) => return Err(error),
            Err(error) if !self.options.allow_degraded_preflight => return Err(error),
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
        segments: &[SubtitleSegment],
    ) -> CoreResult<BackendJsonResult> {
        let mut last_error = None;
        for _ in 0..=self.options.retries {
            self.cancellation.check()?;
            reserve_provider_request(self.options, self.provider_requests, *self.provider_tokens)?;
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
                        payload: BackendPayload::Terminology(parse_payload(
                            &json, candidates, segments,
                        )?),
                        usage,
                    })
                });
            match response {
                Ok(value) => {
                    *self.provider_tokens = self
                        .provider_tokens
                        .saturating_add(value.usage.total_tokens);
                    return Ok(value);
                }
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
            let is_title = word.chars().next().is_some_and(char::is_uppercase)
                && word.chars().skip(1).any(char::is_lowercase);
            let is_acronym = word.chars().filter(|ch| ch.is_alphabetic()).count() >= 2
                && word
                    .chars()
                    .filter(|ch| ch.is_alphabetic())
                    .all(char::is_uppercase);
            if !is_title && !is_acronym {
                index += 1;
                continue;
            }
            // Subtitle lines capitalize ordinary dialogue starters. Do not
            // let them absorb a following proper noun ("Meet Alice") or
            // become a recurring pseudo-name ("I'm", "Please", "Wait").
            if index == 0 && is_common_sentence_initial(word) {
                index += 1;
                continue;
            }
            let mut end = index + 1;
            while end < words.len() && end - index < 4 {
                if !words[end].chars().next().is_some_and(char::is_uppercase)
                    || is_common_sentence_initial(words[end])
                {
                    break;
                }
                end += 1;
            }
            let source = words[index..end].join(" ");
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
                    acronym: false,
                    honorific: true,
                });
        }
    }
    let mut ranked = candidates
        .into_values()
        // A capitalized phrase seen once is weak evidence in subtitles: it is
        // usually just the beginning of a sentence. Require recurrence for
        // Latin-script terms and names; Japanese honorific syntax is strong
        // enough evidence on its own.
        .filter(|candidate| {
            candidate.honorific
                || (candidate.count > 1
                    && !candidate
                        .candidate
                        .source
                        .split_whitespace()
                        .any(is_common_sentence_initial))
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
    let normalized = value
        .trim_matches(|character: char| {
            !character.is_alphanumeric() && character != '\'' && character != '’'
        })
        .replace('’', "'")
        .to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "but"
            | "can"
            | "can't"
            | "cannot"
            | "come"
            | "could"
            | "couldn't"
            | "did"
            | "didn't"
            | "do"
            | "does"
            | "doesn't"
            | "don't"
            | "for"
            | "from"
            | "get"
            | "give"
            | "go"
            | "good"
            | "got"
            | "he"
            | "he'd"
            | "he'll"
            | "he's"
            | "her"
            | "here"
            | "here's"
            | "hey"
            | "his"
            | "hold"
            | "how"
            | "how's"
            | "i"
            | "i'd"
            | "i'll"
            | "i'm"
            | "i've"
            | "if"
            | "in"
            | "is"
            | "isn't"
            | "it"
            | "it'll"
            | "it's"
            | "its"
            | "just"
            | "keep"
            | "let"
            | "let's"
            | "listen"
            | "look"
            | "make"
            | "maybe"
            | "meet"
            | "my"
            | "no"
            | "not"
            | "now"
            | "of"
            | "official"
            | "oh"
            | "okay"
            | "on"
            | "or"
            | "our"
            | "please"
            | "right"
            | "she"
            | "she'd"
            | "she'll"
            | "she's"
            | "so"
            | "sorry"
            | "stop"
            | "sure"
            | "take"
            | "tell"
            | "that"
            | "that's"
            | "the"
            | "their"
            | "then"
            | "there"
            | "there's"
            | "these"
            | "they"
            | "they'd"
            | "they'll"
            | "they're"
            | "they've"
            | "this"
            | "those"
            | "to"
            | "wait"
            | "we"
            | "we'd"
            | "we'll"
            | "we're"
            | "we've"
            | "well"
            | "what"
            | "what's"
            | "when"
            | "when's"
            | "where"
            | "where's"
            | "who"
            | "who's"
            | "why"
            | "why's"
            | "will"
            | "won't"
            | "with"
            | "would"
            | "wouldn't"
            | "yes"
            | "you"
            | "you'd"
            | "you'll"
            | "you're"
            | "you've"
            | "your"
    )
}

fn build_messages(
    options: &PipelineOptions,
    candidates: &[TerminologyCandidate],
    segments: &[SubtitleSegment],
) -> Vec<crate::ports::ChatMessage> {
    let payload = serde_json::json!({
        "source_language": options.source_language,
        "target_language": options.target_language,
        "candidates": candidates.iter().map(|candidate| serde_json::json!({
            "source": candidate.source,
            "context": candidate.context,
        })).collect::<Vec<_>>(),
        "document_samples": document_samples(segments),
    });
    let payload = serde_json::to_string(&payload).unwrap_or_default();
    let name_policy = if options.preserve_names {
        "For personal names, use the exact source spelling as target."
    } else {
        "Include every clearly identified personal name and translate or transliterate it into the target language's conventional script; do not omit a clear personal name merely because its canonical spelling is uncertain."
    };
    vec![
        crate::ports::ChatMessage::system(format!(
            "TASK_START\nextract_terminology\nTASK_END\nReturn JSON only as {{\"entries\":[{{\"source\":\"exact source span\",\"target\":\"canonical translation\"}}],\"entities\":[{{\"canonical_source\":\"canonical entity name\",\"kind\":\"person|organization|place|proper_name|domain_term\",\"variants\":[{{\"source\":\"exact source span\",\"target\":\"natural translation of that source form\"}}]}}],\"document_brief\":\"short genre, tone, relationship, and register guidance\"}}. Include only names, titles, organizations, places, recurring objects, and domain terms whose translation should stay consistent. Group aliases of one entity while preserving the natural granularity of each source form. {name_policy} Copy every source form exactly from a supplied candidate or a document sample. For languages without case or whitespace cues, use recurring exact spans from the samples. Omit ordinary words and uncertain entries. The brief must be short and advisory; never invent plot facts."
        )),
        crate::ports::ChatMessage::user(format!(
            "TERMINOLOGY_JSON_START{payload}TERMINOLOGY_JSON_END"
        )),
    ]
}

pub(super) fn parse_payload(
    payload: &serde_json::Value,
    candidates: &[TerminologyCandidate],
    segments: &[SubtitleSegment],
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
        let canonical_source = candidates
            .iter()
            .find(|candidate| candidate.source.eq_ignore_ascii_case(source))
            .map(|candidate| candidate.source.clone())
            .or_else(|| source_span_in_document(source, segments));
        let Some(canonical_source) = canonical_source else {
            return Err(CoreError::InvalidTranslation(format!(
                "terminology response contains unknown source `{source}`"
            )));
        };
        parsed.push(GlossaryEntry {
            source: canonical_source,
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
                || (!candidates
                    .iter()
                    .any(|candidate| candidate.source.eq_ignore_ascii_case(variant.source.trim()))
                    && source_span_in_document(variant.source.trim(), segments).is_none())
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

fn document_samples(segments: &[SubtitleSegment]) -> Vec<serde_json::Value> {
    const MAX_SAMPLES: usize = 48;
    const MAX_SAMPLE_CHARS: usize = 240;
    if segments.is_empty() {
        return Vec::new();
    }
    let count = segments.len().min(MAX_SAMPLES);
    (0..count)
        .map(|sample| {
            let index = if count == 1 {
                0
            } else {
                sample.saturating_mul(segments.len().saturating_sub(1)) / (count - 1)
            };
            let segment = &segments[index];
            serde_json::json!({
                "id": segment.id,
                "source": segment.text.chars().take(MAX_SAMPLE_CHARS).collect::<String>(),
            })
        })
        .collect()
}

fn source_span_in_document(source: &str, segments: &[SubtitleSegment]) -> Option<String> {
    (!source.is_empty()).then_some(())?;
    segments.iter().find_map(|segment| {
        TermMatcher::case_insensitive()
            .contains(&segment.text, source)
            .then(|| source.to_owned())
    })
}

fn reserve_provider_request(
    options: &PipelineOptions,
    provider_requests: &mut usize,
    provider_tokens: usize,
) -> CoreResult<()> {
    if let Some(limit) = options.max_requests
        && provider_requests.saturating_add(1) > limit
    {
        return Err(CoreError::ResourceBudgetExceeded(format!(
            "request limit is {limit}; {provider_requests} request(s) already used and 1 more required"
        )));
    }
    if let Some(limit) = options.max_tokens
        && provider_tokens >= limit
    {
        return Err(CoreError::ResourceBudgetExceeded(format!(
            "token limit is {limit}; {provider_tokens} token(s) already used"
        )));
    }
    *provider_requests = provider_requests.saturating_add(1);
    Ok(())
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
