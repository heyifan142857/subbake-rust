use std::time::Instant;

use crate::CancellationGuard;
use crate::entities::{
    DocumentCharacter, DocumentGuide, GlossaryEntry, PipelineOptions, PreflightFailurePolicy,
    SubtitleDocument, SubtitleSegment, TerminologyEntity, TerminologyKind,
    TerminologyPreflightResult, TerminologyStats, Usage,
};
use crate::error::{CoreError, CoreResult};
use crate::language_rules::{EnglishRules, JapaneseRules, ResolvedLanguageRules};
use crate::memory::ContextMemory;
use crate::ports::{
    BackendJsonResult, BackendPayload, CacheStage, GenerationRequest, LlmBackend, RuntimeStore,
};
use crate::progress::{ProgressEvent, ProgressSink, ProgressUnit, TaskKind, TaskState};
use crate::term_matcher::TermMatcher;

use super::accounting::PipelineAccounting;

pub(super) struct TerminologyStage<'a, B> {
    pub backend: &'a mut B,
    pub options: &'a PipelineOptions,
    pub language_rules: &'a ResolvedLanguageRules,
    pub memory: &'a mut ContextMemory,
    pub store: Option<&'a dyn RuntimeStore>,
    pub cancellation: &'a CancellationGuard,
    pub progress: Option<&'a dyn ProgressSink>,
    pub accounting: &'a mut PipelineAccounting,
}

impl<B> TerminologyStage<'_, B>
where
    B: LlmBackend,
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
        if !self.options.execution.terminology_preflight {
            self.report(TaskState::Skipped, 0, candidates.len(), Usage::default());
            return Ok(stats);
        }
        if !self.backend.supports_terminology_preflight() {
            if self.options.preflight_failure_policy() == PreflightFailurePolicy::Fail {
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
        let messages = build_messages(
            self.options,
            self.language_rules,
            &candidates,
            &document.segments,
        );
        let hash = super::support::request_hash(self.options, CacheStage::Terminology, &messages);
        let cached = if self.options.execution.use_cache {
            self.store
                .map(|store| store.load_cached_response(CacheStage::Terminology, &hash))
                .transpose()?
                .flatten()
        } else {
            None
        };
        let result = if let Some(response) = cached {
            stats.cache_hits = 1;
            self.accounting.record_cache_hit();
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
                let guide = accept_document_guide(self.memory, payload.guide, &mut stats);
                stats.entries_added = self.memory.glossary.len().saturating_sub(existing.len());
                if self.options.execution.use_cache
                    && stats.cache_hits == 0
                    && let Some(store) = self.store
                {
                    store.save_cached_response(
                        CacheStage::Terminology,
                        &hash,
                        &BackendJsonResult {
                            payload: BackendPayload::Terminology(TerminologyPreflightResult {
                                guide,
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
            Err(error)
                if self.options.preflight_failure_policy() == PreflightFailurePolicy::Fail =>
            {
                return Err(error);
            }
            Err(error) => {
                stats.degraded = true;
                stats.degraded_reason = Some(error.to_string());
            }
        }
        stats.duration_ms = super::support::duration_ms(started);
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
        for _ in 0..=self.options.execution.retries {
            self.cancellation.check()?;
            self.accounting.reserve_requests(
                1,
                self.options.execution.max_requests,
                self.options.execution.max_tokens,
            )?;
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
                    self.accounting.record_tokens(value.usage.total_tokens);
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
                let word = EnglishRules::possessive_base(word).unwrap_or(word);
                english_stutter_base(word).unwrap_or(word)
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
        for source in JapaneseRules::honorific_names(&segment.text) {
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

fn is_common_sentence_initial(value: &str) -> bool {
    let normalized = value
        .trim_matches(|character: char| {
            !character.is_alphanumeric() && character != '\'' && character != '’'
        })
        .replace('’', "'")
        .to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "a" | "aah"
            | "all"
            | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "aw"
            | "big"
            | "but"
            | "can"
            | "can't"
            | "cannot"
            | "come"
            | "could"
            | "couldn't"
            | "dad"
            | "did"
            | "didn't"
            | "do"
            | "does"
            | "doesn't"
            | "don't"
            | "eight"
            | "eleven"
            | "fifteen"
            | "five"
            | "for"
            | "four"
            | "fourteen"
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
            | "holy"
            | "how"
            | "how's"
            | "i"
            | "i'd"
            | "i'll"
            | "i'm"
            | "i've"
            | "if"
            | "in"
            | "have"
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
            | "mm"
            | "my"
            | "nine"
            | "nineteen"
            | "no"
            | "not"
            | "now"
            | "of"
            | "official"
            | "oh"
            | "okay"
            | "on"
            | "one"
            | "or"
            | "our"
            | "ow"
            | "please"
            | "right"
            | "seven"
            | "seventeen"
            | "she"
            | "she'd"
            | "she'll"
            | "she's"
            | "so"
            | "some"
            | "sorry"
            | "stop"
            | "sure"
            | "take"
            | "tell"
            | "ten"
            | "thank"
            | "that"
            | "that's"
            | "the"
            | "their"
            | "then"
            | "thirteen"
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
            | "three"
            | "to"
            | "twelve"
            | "twenty"
            | "two"
            | "uh"
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
            | "whoa"
            | "why"
            | "why's"
            | "will"
            | "won't"
            | "with"
            | "would"
            | "wouldn't"
            | "wow"
            | "yes"
            | "yeah"
            | "you"
            | "you'd"
            | "you'll"
            | "you're"
            | "you've"
            | "your"
    )
}

fn english_stutter_base(value: &str) -> Option<&str> {
    let (prefix, base) = value.split_once('-')?;
    if prefix.is_empty() || base.is_empty() || prefix.chars().count() > 3 {
        return None;
    }
    let prefix = prefix.to_lowercase();
    let base_lower = base.to_lowercase();
    if base_lower == prefix {
        return Some("");
    }
    base_lower.starts_with(&prefix).then_some(base)
}

fn contains_english_stutter(value: &str) -> bool {
    value.split_whitespace().any(|word| {
        let word =
            word.trim_matches(|character: char| !character.is_alphanumeric() && character != '-');
        english_stutter_base(word).is_some()
    })
}

fn is_ordinary_latin_document_span(value: &str) -> bool {
    let Some(first) = value.split_whitespace().next() else {
        return true;
    };
    value
        .chars()
        .any(|character| character.is_ascii_alphabetic())
        && (is_common_sentence_initial(first) || contains_english_stutter(value))
}

fn is_redundant_possessive(value: &str, canonical: &str) -> bool {
    EnglishRules::possessive_base(value)
        .is_some_and(|base| base.trim().eq_ignore_ascii_case(canonical.trim()))
}

fn build_messages(
    options: &PipelineOptions,
    language_rules: &ResolvedLanguageRules,
    candidates: &[TerminologyCandidate],
    segments: &[SubtitleSegment],
) -> Vec<crate::ports::ChatMessage> {
    let payload = serde_json::json!({
        "source_language": options.validation.source_language,
        "target_language": options.validation.target_language,
        "candidates": candidates.iter().map(|candidate| serde_json::json!({
            "source": candidate.source,
            "context": candidate.context,
        })).collect::<Vec<_>>(),
        "document_samples": document_samples(segments),
    });
    let payload = serde_json::to_string(&payload).unwrap_or_default();
    let name_policy = if options.validation.preserve_names {
        "For personal names, use the exact source spelling as target."
    } else {
        "Include every clearly identified personal name and translate or transliterate it into the target language's conventional script; do not omit a clear personal name merely because its canonical spelling is uncertain."
    };
    let language_guidance = language_rules.document_guide_guidance().unwrap_or_default();
    vec![
        crate::ports::ChatMessage::system(format!(
            "TASK_START\nextract_document_guide\nTASK_END\nReturn JSON only as {{\"guide\":{{\"synopsis\":\"short factual synopsis\",\"genre\":\"genre\",\"tone\":\"tone\",\"target_audience\":\"audience\",\"characters\":[{{\"canonical_source\":\"exact source name\",\"canonical_target\":\"canonical target name\",\"aliases\":[{{\"source\":\"exact source alias\",\"target\":\"target alias\"}}],\"gender\":\"optional only when explicit\",\"relationships\":[\"short factual relationship\"],\"speaking_style\":\"short observed register guidance\",\"forms_of_address\":[{{\"source\":\"exact source form\",\"target\":\"contextual target form\"}}]}}],\"terminology\":[{{\"canonical_source\":\"canonical term\",\"kind\":\"organization|place|proper_name|domain_term\",\"variants\":[{{\"source\":\"exact source span\",\"target\":\"canonical translation\"}}]}}]}}}}. Build one frozen guide for the whole document. Include only recurring or meaning-critical information supported by supplied candidates or samples. Group aliases of one character or term while preserving the natural granularity of each source form. {name_policy} Copy every source name, alias, address form, and terminology variant exactly from a supplied candidate or document sample. Omit ordinary words, uncertain identities, inferred gender, speculative relationships, and invented plot facts. Keep synopsis, genre, tone, audience, relationships, and speaking-style fields concise. {language_guidance}"
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
    let mut guide =
        serde_json::from_value::<DocumentGuide>(payload["guide"].clone()).map_err(|error| {
            CoreError::InvalidTranslation(format!(
                "document guide has an invalid structure: {error}"
            ))
        })?;
    guide.synopsis = bounded_text(&guide.synopsis, 800);
    guide.genre = bounded_text(&guide.genre, 120);
    guide.tone = bounded_text(&guide.tone, 240);
    guide.target_audience = bounded_text(&guide.target_audience, 160);
    guide.characters.truncate(64);
    for character in &mut guide.characters {
        validate_character(character, candidates, segments)?;
    }
    guide.characters.retain(|character| {
        !character.canonical_source.is_empty() && !character.canonical_target.is_empty()
    });
    guide.terminology = validate_entities(guide.terminology, candidates, segments)?;
    let character_sources = guide
        .characters
        .iter()
        .map(|character| character.canonical_source.to_lowercase())
        .collect::<std::collections::BTreeSet<_>>();
    guide
        .terminology
        .retain(|entity| !character_sources.contains(&entity.canonical_source.to_lowercase()));
    Ok(TerminologyPreflightResult { guide })
}

fn validate_entities(
    raw_entities: Vec<TerminologyEntity>,
    candidates: &[TerminologyCandidate],
    segments: &[SubtitleSegment],
) -> CoreResult<Vec<TerminologyEntity>> {
    let mut entities = Vec::new();
    for mut entity in raw_entities.into_iter().take(128) {
        if entity.canonical_source.trim().is_empty() {
            return Err(CoreError::InvalidTranslation(
                "terminology entity is missing a canonical source or variants".to_owned(),
            ));
        }
        let canonical_source = entity.canonical_source.clone();
        entity.variants.retain(|variant| {
            let source = variant.source.trim();
            !is_ordinary_latin_document_span(source)
                && !is_redundant_possessive(source, &canonical_source)
        });
        if entity.variants.is_empty() {
            continue;
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
        entities.push(entity);
    }
    Ok(entities)
}

fn validate_character(
    character: &mut DocumentCharacter,
    candidates: &[TerminologyCandidate],
    segments: &[SubtitleSegment],
) -> CoreResult<()> {
    character.canonical_source = character.canonical_source.trim().to_owned();
    character.canonical_target = character.canonical_target.trim().to_owned();
    if character.canonical_source.is_empty() || character.canonical_target.is_empty() {
        return Err(CoreError::InvalidTranslation(
            "document character is missing a canonical source or target".to_owned(),
        ));
    }
    if !known_source_span(&character.canonical_source, candidates, segments) {
        return Err(CoreError::InvalidTranslation(format!(
            "document character contains unknown source `{}`",
            character.canonical_source
        )));
    }
    if is_ordinary_latin_document_span(&character.canonical_source)
        && !candidates.iter().any(|candidate| {
            candidate
                .source
                .eq_ignore_ascii_case(&character.canonical_source)
        })
    {
        character.canonical_source.clear();
        character.canonical_target.clear();
        character.aliases.clear();
        return Ok(());
    }
    validate_guide_entries(&mut character.aliases, candidates, segments)?;
    character.aliases.retain(|alias| {
        !is_ordinary_latin_document_span(&alias.source)
            && !is_redundant_possessive(&alias.source, &character.canonical_source)
    });
    validate_guide_entries(&mut character.forms_of_address, candidates, segments)?;
    if !character.aliases.iter().any(|alias| {
        alias
            .source
            .eq_ignore_ascii_case(&character.canonical_source)
    }) {
        character.aliases.insert(
            0,
            GlossaryEntry {
                source: character.canonical_source.clone(),
                target: character.canonical_target.clone(),
            },
        );
    }
    character.aliases.truncate(16);
    character.forms_of_address.truncate(16);
    character.gender = character
        .gender
        .as_deref()
        .map(|value| bounded_text(value, 40))
        .filter(|value| !value.is_empty());
    character.relationships = character
        .relationships
        .iter()
        .map(|value| bounded_text(value, 160))
        .filter(|value| !value.is_empty())
        .take(16)
        .collect();
    character.speaking_style = bounded_text(&character.speaking_style, 240);
    Ok(())
}

fn validate_guide_entries(
    entries: &mut Vec<GlossaryEntry>,
    candidates: &[TerminologyCandidate],
    segments: &[SubtitleSegment],
) -> CoreResult<()> {
    for entry in entries.iter_mut() {
        entry.source = entry.source.trim().to_owned();
        entry.target = entry.target.trim().to_owned();
        if entry.source.is_empty()
            || entry.target.is_empty()
            || !known_source_span(&entry.source, candidates, segments)
        {
            return Err(CoreError::InvalidTranslation(format!(
                "document guide contains unknown or empty source `{}`",
                entry.source
            )));
        }
    }
    entries.dedup_by(|left, right| left.source.eq_ignore_ascii_case(&right.source));
    Ok(())
}

fn known_source_span(
    source: &str,
    candidates: &[TerminologyCandidate],
    segments: &[SubtitleSegment],
) -> bool {
    candidates
        .iter()
        .any(|candidate| candidate.source.eq_ignore_ascii_case(source))
        || source_span_in_document(source, segments).is_some()
}

fn bounded_text(value: &str, max_chars: usize) -> String {
    value.trim().chars().take(max_chars).collect()
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

fn accept_document_guide(
    memory: &mut ContextMemory,
    mut guide: DocumentGuide,
    stats: &mut TerminologyStats,
) -> DocumentGuide {
    let character_entities = guide
        .characters
        .iter()
        .map(|character| TerminologyEntity {
            canonical_source: character.canonical_source.clone(),
            kind: TerminologyKind::Person,
            variants: character.aliases.clone(),
        })
        .collect();
    let accepted_characters = accept_entities(memory, character_entities, stats);
    guide.characters.retain_mut(|character| {
        let Some(accepted) = accepted_characters.iter().find(|entity| {
            entity
                .canonical_source
                .eq_ignore_ascii_case(&character.canonical_source)
        }) else {
            return false;
        };
        character.aliases = accepted.variants.clone();
        if let Some(canonical) = character.aliases.iter().find(|alias| {
            alias
                .source
                .eq_ignore_ascii_case(&character.canonical_source)
        }) {
            character.canonical_target = canonical.target.clone();
        }
        true
    });
    guide.terminology = accept_entities(memory, guide.terminology, stats);
    memory.document_guide = guide.clone();
    guide
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
        accepted.push(entity);
    }
    accepted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language_rules::LanguageRuleRegistry;

    fn segment(text: &str) -> SubtitleSegment {
        SubtitleSegment {
            id: "1".to_owned(),
            text: text.to_owned(),
            start: None,
            end: None,
            identifier: None,
            settings: None,
            semantic: Default::default(),
        }
    }

    #[test]
    fn japanese_chinese_preflight_requests_exact_honorific_surfaces() {
        let mut options = PipelineOptions::new("episode.ass".into());
        options.validation.source_language = "ja".to_owned();
        options.validation.target_language = "zh-Hans".to_owned();
        let rules = LanguageRuleRegistry::resolve("ja", "zh-Hans");
        let messages = build_messages(
            &options,
            &rules,
            &[TerminologyCandidate {
                source: "田中".to_owned(),
                context: "田中さん、行きましょう。".to_owned(),
                align_as_name: true,
            }],
            &[segment("田中さん、行きましょう。")],
        );

        assert!(
            messages[0].content.contains(
                "record each supported full Japanese honorific or address surface exactly"
            )
        );
        assert!(messages[1].content.contains("田中さん"));
    }
}
