// In-run translation context. The persisted `recent_summaries` field remains
// readable for storage compatibility, but translation prompts use deterministic
// neighboring source and confirmed-translation context instead.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::entities::{DocumentGuide, GlossaryEntry, TerminologyEntity};
use crate::language_rules::EnglishRules;
use crate::term_matcher::TermMatcher;

pub const DEFAULT_MAX_SUMMARIES: usize = 2;
pub const GLOSSARY_RELEVANCE_LIMIT: usize = 24;

pub const DEFAULT_STYLE_RULES: &[&str] = &[
    "Use natural, idiomatic target-language phrasing.",
    "Preserve tone, humor, emotion, and profanity where present.",
    "Keep subtitles concise and easy to read on screen.",
    "Do not merge or drop subtitle entries.",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextMemory {
    #[serde(default = "default_style_rules")]
    pub style_rules: Vec<String>,
    #[serde(default)]
    pub recent_summaries: Vec<String>,
    #[serde(default)]
    pub glossary: BTreeMap<String, String>,
    /// Canonical personal-name translations discovered during the current run.
    /// Kept separate from the persisted advisory glossary so a fresh run can
    /// choose its own first translation while resume remains deterministic.
    #[serde(default)]
    pub name_translations: BTreeMap<String, String>,
    /// Frozen structured guidance produced before translation.
    #[serde(default)]
    pub document_guide: DocumentGuide,
    #[serde(default)]
    pub terminology_candidates: Vec<String>,
    /// Locally high-confidence personal-name candidates eligible for Turbo's
    /// lightweight indexed markers.
    #[serde(default)]
    pub name_candidates: Vec<String>,
    #[serde(default = "default_max_summaries")]
    pub max_summaries: usize,
}

fn default_style_rules() -> Vec<String> {
    DEFAULT_STYLE_RULES
        .iter()
        .map(|rule| (*rule).to_owned())
        .collect()
}

fn default_max_summaries() -> usize {
    DEFAULT_MAX_SUMMARIES
}

impl Default for ContextMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextMemory {
    pub fn new() -> Self {
        Self {
            style_rules: default_style_rules(),
            recent_summaries: Vec::new(),
            glossary: BTreeMap::new(),
            name_translations: BTreeMap::new(),
            document_guide: DocumentGuide::default(),
            terminology_candidates: Vec::new(),
            name_candidates: Vec::new(),
            max_summaries: DEFAULT_MAX_SUMMARIES,
        }
    }

    /// Replace the glossary with persisted entries (loaded from the runtime store
    /// at startup). Mirrors `ContextMemory.load_glossary`.
    pub fn load_glossary(&mut self, entries: &[(String, String)]) {
        self.glossary = entries.iter().cloned().collect();
    }

    /// Record a batch summary and any new glossary entries the model returned.
    /// Keeps only the most recent `max_summaries` summaries.
    pub fn update(&mut self, summary: &str, glossary_updates: &[GlossaryEntry]) {
        let clean = summary.trim();
        if !clean.is_empty() {
            self.recent_summaries.push(clean.to_owned());
            let excess = self
                .recent_summaries
                .len()
                .saturating_sub(self.max_summaries);
            if excess > 0 {
                self.recent_summaries.drain(..excess);
            }
        }
        for entry in glossary_updates {
            if (entry.source.is_empty() && entry.target.is_empty())
                || EnglishRules::possessive_base(&entry.source).is_some()
            {
                continue;
            }
            self.glossary
                .insert(entry.source.clone(), entry.target.clone());
        }
    }

    pub fn add_terminology_entity(&mut self, entity: TerminologyEntity) {
        if let Some(current) = self.document_guide.terminology.iter_mut().find(|current| {
            current
                .canonical_source
                .eq_ignore_ascii_case(&entity.canonical_source)
                && current.kind == entity.kind
        }) {
            for variant in entity.variants {
                if !current
                    .variants
                    .iter()
                    .any(|item| item.source.eq_ignore_ascii_case(&variant.source))
                {
                    current.variants.push(variant);
                }
            }
        } else {
            self.document_guide.terminology.push(entity);
        }
    }

    /// Return up to `GLOSSARY_RELEVANCE_LIMIT` glossary entries whose source or
    /// target matches the batch texts using shared script-aware term rules.
    pub fn select_relevant_glossary(&self, texts: &[&str]) -> Vec<(String, String)> {
        if self.glossary.is_empty() || texts.is_empty() {
            return Vec::new();
        }
        let haystack = texts.join("\n");
        let entries = self
            .glossary
            .iter()
            .filter(|(source, _)| EnglishRules::possessive_base(source).is_none())
            .collect::<Vec<_>>();
        let terms = entries
            .iter()
            .flat_map(|(source, target)| [source.as_str(), target.as_str()])
            .collect::<Vec<_>>();
        let matcher = TermMatcher::case_insensitive();
        let mut selected = vec![false; entries.len()];
        matcher
            .matching_indices(&haystack, &terms)
            .into_iter()
            .filter_map(|term_index| {
                let index = term_index / 2;
                let entry = entries.get(index)?;
                if std::mem::replace(selected.get_mut(index)?, true) {
                    return None;
                }
                Some(((*entry.0).clone(), (*entry.1).clone()))
            })
            .take(GLOSSARY_RELEVANCE_LIMIT)
            .collect()
    }

    /// Select stable document-level guidance for the current scene. Global
    /// prose stays available, while character and terminology records are
    /// included only when one of their exact source/target forms is relevant.
    pub fn select_relevant_document_guide(&self, texts: &[&str]) -> DocumentGuide {
        let mut selected = DocumentGuide {
            synopsis: self.document_guide.synopsis.clone(),
            genre: self.document_guide.genre.clone(),
            tone: self.document_guide.tone.clone(),
            target_audience: self.document_guide.target_audience.clone(),
            ..DocumentGuide::default()
        };
        if texts.is_empty() {
            return selected;
        }
        let haystack = texts.join("\n");
        let matcher = TermMatcher::case_insensitive();
        selected.characters = self
            .document_guide
            .characters
            .iter()
            .filter(|character| {
                std::iter::once(character.canonical_source.as_str())
                    .chain(std::iter::once(character.canonical_target.as_str()))
                    .chain(
                        character
                            .aliases
                            .iter()
                            .flat_map(|alias| [alias.source.as_str(), alias.target.as_str()]),
                    )
                    .chain(
                        character
                            .forms_of_address
                            .iter()
                            .flat_map(|form| [form.source.as_str(), form.target.as_str()]),
                    )
                    .any(|term| !term.is_empty() && matcher.contains(&haystack, term))
            })
            .take(8)
            .cloned()
            .collect();
        selected.terminology =
            self.document_guide
                .terminology
                .iter()
                .filter(|entity| {
                    std::iter::once(entity.canonical_source.as_str())
                        .chain(
                            entity.variants.iter().flat_map(|variant| {
                                [variant.source.as_str(), variant.target.as_str()]
                            }),
                        )
                        .any(|term| !term.is_empty() && matcher.contains(&haystack, term))
                })
                .take(16)
                .cloned()
                .collect();
        selected
    }

    /// Legacy persisted summaries, newest last, capped at `max_summaries`.
    /// Translation and review prompts no longer consume this field.
    pub fn recent_summaries_for_prompt(&self) -> &[String] {
        let start = self
            .recent_summaries
            .len()
            .saturating_sub(self.max_summaries);
        &self.recent_summaries[start..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::{DocumentCharacter, TerminologyKind};

    #[test]
    fn recognizes_simple_english_possessive_forms() {
        assert_eq!(
            EnglishRules::possessive_base("MacAndrews'"),
            Some("MacAndrews")
        );
        assert_eq!(
            EnglishRules::possessive_base("MacClannough's"),
            Some("MacClannough")
        );
        assert_eq!(EnglishRules::possessive_base("James’"), Some("James"));
        assert_eq!(EnglishRules::possessive_base("Mornay’s"), Some("Mornay"));
        assert_eq!(EnglishRules::possessive_base("MacAndrews"), None);
    }

    #[test]
    fn update_keeps_only_recent_summaries() {
        let mut memory = ContextMemory::new();
        memory.update("first", &[]);
        memory.update("second", &[]);
        memory.update("third", &[]);

        assert_eq!(memory.recent_summaries_for_prompt(), &["second", "third"]);
    }

    #[test]
    fn update_merges_glossary_entries() {
        let mut memory = ContextMemory::new();
        memory.update(
            "ok",
            &[
                GlossaryEntry {
                    source: "alice".to_owned(),
                    target: "爱丽丝".to_owned(),
                },
                GlossaryEntry {
                    source: "alice".to_owned(),
                    target: "爱丽".to_owned(),
                },
            ],
        );
        assert_eq!(
            memory.glossary.get("alice").map(String::as_str),
            Some("爱丽")
        );
    }

    #[test]
    fn possessive_forms_are_not_retained_or_selected_as_terms() {
        let mut memory = ContextMemory::new();
        memory.update(
            "",
            &[
                GlossaryEntry {
                    source: "MacAndrews".to_owned(),
                    target: "麦克安德鲁斯".to_owned(),
                },
                GlossaryEntry {
                    source: "MacAndrews'".to_owned(),
                    target: "麦克安德鲁斯的".to_owned(),
                },
            ],
        );
        memory
            .glossary
            .insert("Mornay's".to_owned(), "莫奈的".to_owned());

        assert!(!memory.glossary.contains_key("MacAndrews'"));
        assert_eq!(
            memory.select_relevant_glossary(&["MacAndrews' and Mornay's"]),
            vec![("MacAndrews".to_owned(), "麦克安德鲁斯".to_owned())]
        );
    }

    #[test]
    fn select_relevant_glossary_filters_by_hit() {
        let mut memory = ContextMemory::new();
        memory
            .glossary
            .insert("alice".to_owned(), "爱丽丝".to_owned());
        memory.glossary.insert("bob".to_owned(), "鲍勃".to_owned());

        let matched = memory.select_relevant_glossary(&["alice runs away"]);
        assert_eq!(matched, vec![("alice".to_owned(), "爱丽丝".to_owned())]);
    }

    #[test]
    fn glossary_selection_uses_boundaries_inflections_and_cjk_longest_match() {
        let mut memory = ContextMemory::new();
        memory.glossary.extend([
            ("he".to_owned(), "他".to_owned()),
            ("actor".to_owned(), "演员".to_owned()),
            ("纽约".to_owned(), "纽约".to_owned()),
            ("纽约时报".to_owned(), "纽约时报中文版".to_owned()),
        ]);

        let matched = memory.select_relevant_glossary(&["The actors read the 纽约时报."]);

        assert!(matched.contains(&("actor".to_owned(), "演员".to_owned())));
        assert!(matched.contains(&("纽约时报".to_owned(), "纽约时报中文版".to_owned())));
        assert!(!matched.iter().any(|(source, _)| source == "he"));
        assert!(!matched.iter().any(|(source, _)| source == "纽约"));
    }

    #[test]
    fn select_relevant_glossary_respects_limit() {
        let mut memory = ContextMemory::new();
        for index in 0..(GLOSSARY_RELEVANCE_LIMIT + 5) {
            memory
                .glossary
                .insert(format!("word{index}"), format!("译{index}"));
        }
        let haystack: String = (0..(GLOSSARY_RELEVANCE_LIMIT + 5))
            .map(|index| format!("word{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        let matched = memory.select_relevant_glossary(&[haystack.as_str()]);
        assert_eq!(matched.len(), GLOSSARY_RELEVANCE_LIMIT);
    }

    #[test]
    fn serializes_and_restores_via_serde() {
        let mut memory = ContextMemory::new();
        memory.update(
            "summary",
            &[GlossaryEntry {
                source: "x".to_owned(),
                target: "y".to_owned(),
            }],
        );
        let json = serde_json::to_string(&memory).expect("serialize");
        let restored: ContextMemory = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, memory);
    }

    #[test]
    fn document_guide_selection_keeps_global_style_and_only_relevant_records() {
        let mut memory = ContextMemory::new();
        memory.document_guide = DocumentGuide {
            synopsis: "A portal experiment goes wrong.".to_owned(),
            genre: "science fiction comedy".to_owned(),
            tone: "dry and irreverent".to_owned(),
            target_audience: "adult".to_owned(),
            characters: vec![
                DocumentCharacter {
                    canonical_source: "Rick".to_owned(),
                    canonical_target: "瑞克".to_owned(),
                    ..DocumentCharacter::default()
                },
                DocumentCharacter {
                    canonical_source: "Morty".to_owned(),
                    canonical_target: "莫蒂".to_owned(),
                    ..DocumentCharacter::default()
                },
            ],
            terminology: vec![
                TerminologyEntity {
                    canonical_source: "portal gun".to_owned(),
                    kind: TerminologyKind::DomainTerm,
                    variants: vec![GlossaryEntry {
                        source: "portal gun".to_owned(),
                        target: "传送枪".to_owned(),
                    }],
                },
                TerminologyEntity {
                    canonical_source: "Council".to_owned(),
                    kind: TerminologyKind::Organization,
                    variants: vec![GlossaryEntry {
                        source: "Council".to_owned(),
                        target: "委员会".to_owned(),
                    }],
                },
            ],
        };

        let selected = memory.select_relevant_document_guide(&["Rick fired the portal gun."]);

        assert_eq!(selected.synopsis, memory.document_guide.synopsis);
        assert_eq!(selected.characters.len(), 1);
        assert_eq!(selected.characters[0].canonical_source, "Rick");
        assert_eq!(selected.terminology.len(), 1);
        assert_eq!(selected.terminology[0].canonical_source, "portal gun");
    }

    #[test]
    fn memory_without_document_guide_uses_an_empty_guide() {
        let restored: ContextMemory = serde_json::from_str(
            r#"{"style_rules":[],"recent_summaries":[],"glossary":{"Mary":"玛丽"},"terminology_candidates":[],"max_summaries":2}"#,
        )
        .expect("deserialize memory");

        assert!(restored.name_translations.is_empty());
        assert!(restored.name_candidates.is_empty());
        assert!(restored.document_guide.is_empty());
        assert_eq!(restored.glossary["Mary"], "玛丽");
    }
}
