// In-run translation context: accumulates glossary entries and recent
// summaries across batches so each prompt can carry relevant context. Mirrors
// Python `subbake/memory.py::ContextMemory`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::entities::GlossaryEntry;

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
    #[serde(default = "default_max_summaries")]
    pub max_summaries: usize,
}

fn default_style_rules() -> Vec<String> {
    DEFAULT_STYLE_RULES.iter().map(|rule| (*rule).to_owned()).collect()
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
            if entry.source.is_empty() && entry.target.is_empty() {
                continue;
            }
            self.glossary
                .insert(entry.source.clone(), entry.target.clone());
        }
    }

    /// Return up to `GLOSSARY_RELEVANCE_LIMIT` glossary entries whose source or
    /// target (case-insensitive) appears in the batch texts. Mirrors
    /// `prompts.select_relevant_glossary`.
    pub fn select_relevant_glossary(&self, texts: &[&str]) -> Vec<(String, String)> {
        if self.glossary.is_empty() || texts.is_empty() {
            return Vec::new();
        }
        let haystack = texts.join("\n").to_lowercase();
        let mut matched = Vec::new();
        for (source, target) in &self.glossary {
            if haystack.contains(&source.to_lowercase()) || haystack.contains(&target.to_lowercase())
            {
                matched.push((source.clone(), target.clone()));
                if matched.len() >= GLOSSARY_RELEVANCE_LIMIT {
                    break;
                }
            }
        }
        matched
    }

    /// Recent summaries, newest last, capped at `max_summaries` — the slice
    /// injected into prompts as `recent`.
    pub fn recent_summaries_for_prompt(&self) -> &[String] {
        let start = self.recent_summaries.len().saturating_sub(self.max_summaries);
        &self.recent_summaries[start..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(memory.glossary.get("alice").map(String::as_str), Some("爱丽"));
    }

    #[test]
    fn select_relevant_glossary_filters_by_hit() {
        let mut memory = ContextMemory::new();
        memory.glossary.insert("alice".to_owned(), "爱丽丝".to_owned());
        memory.glossary.insert("bob".to_owned(), "鲍勃".to_owned());

        let matched = memory.select_relevant_glossary(&["alice runs away"]);
        assert_eq!(matched, vec![("alice".to_owned(), "爱丽丝".to_owned())]);
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
        memory.update("summary", &[GlossaryEntry {
            source: "x".to_owned(),
            target: "y".to_owned(),
        }]);
        let json = serde_json::to_string(&memory).expect("serialize");
        let restored: ContextMemory = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, memory);
    }
}
