use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermMatcher {
    case_sensitive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermMatch {
    pub term_index: usize,
    pub start: usize,
    pub end: usize,
}

impl TermMatcher {
    pub const fn new(case_sensitive: bool) -> Self {
        Self { case_sensitive }
    }

    pub const fn case_insensitive() -> Self {
        Self::new(false)
    }

    pub fn contains(&self, text: &str, term: &str) -> bool {
        !self.matching_indices(text, &[term]).is_empty()
    }

    pub fn missing_required<'a>(
        &self,
        source_text: &str,
        translated_text: &str,
        glossary: &'a BTreeMap<String, String>,
    ) -> Vec<(&'a str, &'a str)> {
        let entries = glossary.iter().collect::<Vec<_>>();
        let terms = entries
            .iter()
            .map(|(source, _)| source.as_str())
            .collect::<Vec<_>>();
        self.matching_indices(source_text, &terms)
            .into_iter()
            .filter_map(|index| entries.get(index).copied())
            .filter(|(_, target)| !self.contains(translated_text, target))
            .map(|(source, target)| (source.as_str(), target.as_str()))
            .collect()
    }

    /// Return unique indices from `terms` after resolving overlapping matches.
    /// Longer terms win; remaining matches are returned in text order.
    pub fn matching_indices(&self, text: &str, terms: &[&str]) -> Vec<usize> {
        let mut seen = HashSet::new();
        self.find_matches(text, terms)
            .into_iter()
            .filter_map(|candidate| {
                seen.insert(candidate.term_index)
                    .then_some(candidate.term_index)
            })
            .collect()
    }

    /// Return non-overlapping matches in text order. Terms that overlap are
    /// resolved longest-first before positions are returned.
    pub fn find_matches(&self, text: &str, terms: &[&str]) -> Vec<TermMatch> {
        let mut candidates = terms
            .iter()
            .enumerate()
            .flat_map(|(term_index, term)| self.term_candidates(text, term_index, term))
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .length
                .cmp(&left.length)
                .then_with(|| left.start.cmp(&right.start))
                .then_with(|| left.term_index.cmp(&right.term_index))
        });

        let mut accepted = Vec::<MatchCandidate>::new();
        for candidate in candidates {
            if accepted
                .iter()
                .any(|current| ranges_overlap(&candidate, current))
            {
                continue;
            }
            accepted.push(candidate);
        }
        accepted.sort_by(|left, right| {
            left.start
                .cmp(&right.start)
                .then_with(|| right.length.cmp(&left.length))
                .then_with(|| left.term_index.cmp(&right.term_index))
        });

        accepted
            .into_iter()
            .map(|candidate| TermMatch {
                term_index: candidate.term_index,
                start: candidate.start,
                end: candidate.end,
            })
            .collect()
    }

    pub fn replace_matches(
        &self,
        text: &str,
        terms: &[&str],
        mut replacement: impl FnMut(usize, &str) -> String,
    ) -> String {
        let matches = self.find_matches(text, terms);
        if matches.is_empty() {
            return text.to_owned();
        }
        let mut output = String::with_capacity(text.len());
        let mut cursor = 0;
        for matched in matches {
            output.push_str(&text[cursor..matched.start]);
            output.push_str(&replacement(
                matched.term_index,
                &text[matched.start..matched.end],
            ));
            cursor = matched.end;
        }
        output.push_str(&text[cursor..]);
        output
    }

    fn term_candidates(&self, text: &str, term_index: usize, term: &str) -> Vec<MatchCandidate> {
        let term = term.trim();
        if term.is_empty() {
            return Vec::new();
        }
        if term.chars().any(is_cjk) {
            return self.literal_candidates(text, term_index, term, false);
        }
        if term.chars().all(is_latin_term_character) {
            return self.latin_token_candidates(text, term_index, term);
        }
        self.literal_candidates(text, term_index, term, true)
    }

    fn latin_token_candidates(
        &self,
        text: &str,
        term_index: usize,
        term: &str,
    ) -> Vec<MatchCandidate> {
        let text_tokens = latin_tokens(text, self.case_sensitive);
        let term_tokens = latin_tokens(term, self.case_sensitive);
        if term_tokens.is_empty() || term_tokens.len() > text_tokens.len() {
            return Vec::new();
        }
        text_tokens
            .windows(term_tokens.len())
            .filter(|window| {
                window
                    .iter()
                    .zip(&term_tokens)
                    .enumerate()
                    .all(|(index, (actual, expected))| {
                        if index + 1 == term_tokens.len() {
                            latin_word_matches(&actual.text, &expected.text)
                        } else {
                            actual.text == expected.text
                        }
                    })
            })
            .filter_map(|window| {
                let first = window.first()?;
                let last = window.last()?;
                Some(MatchCandidate {
                    term_index,
                    start: first.start,
                    end: last.end,
                    length: term.chars().count(),
                })
            })
            .collect()
    }

    fn literal_candidates(
        &self,
        text: &str,
        term_index: usize,
        term: &str,
        require_boundaries: bool,
    ) -> Vec<MatchCandidate> {
        let text = normalize_case(text, self.case_sensitive);
        let term = normalize_case(term, self.case_sensitive);
        text.match_indices(&term)
            .filter_map(|(start, matched)| {
                let end = start + matched.len();
                if require_boundaries && !has_literal_boundaries(&text, start, end) {
                    return None;
                }
                Some(MatchCandidate {
                    term_index,
                    start,
                    end,
                    length: term.chars().count(),
                })
            })
            .collect()
    }
}

impl Default for TermMatcher {
    fn default() -> Self {
        Self::case_insensitive()
    }
}

#[derive(Debug, Clone)]
struct LatinToken {
    start: usize,
    end: usize,
    text: String,
}

#[derive(Debug, Clone)]
struct MatchCandidate {
    term_index: usize,
    start: usize,
    end: usize,
    length: usize,
}

fn latin_tokens(text: &str, case_sensitive: bool) -> Vec<LatinToken> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (index, character) in text.char_indices() {
        if is_latin_word_character(character) {
            start.get_or_insert(index);
        } else if let Some(start) = start.take() {
            tokens.push(LatinToken {
                start,
                end: index,
                text: normalize_case(&text[start..index], case_sensitive),
            });
        }
    }
    if let Some(start) = start {
        tokens.push(LatinToken {
            start,
            end: text.len(),
            text: normalize_case(&text[start..], case_sensitive),
        });
    }
    tokens
}

fn latin_word_matches(actual: &str, expected: &str) -> bool {
    if actual == expected {
        return true;
    }
    let without_possessive = actual
        .strip_suffix("'s")
        .or_else(|| actual.strip_suffix("’s"))
        .or_else(|| actual.strip_suffix('\''))
        .or_else(|| actual.strip_suffix('’'))
        .unwrap_or(actual);
    if without_possessive == expected {
        return true;
    }
    if expected.chars().count() < 3 {
        return false;
    }
    inflected_forms(expected)
        .iter()
        .any(|form| form == without_possessive)
}

fn inflected_forms(term: &str) -> Vec<String> {
    let mut forms = Vec::new();
    if let Some(stem) = term.strip_suffix('y')
        && stem
            .chars()
            .next_back()
            .is_some_and(|character| !is_ascii_vowel(character))
    {
        forms.push(format!("{stem}ies"));
    } else if term.ends_with(['s', 'x', 'z']) || term.ends_with("ch") || term.ends_with("sh") {
        forms.push(format!("{term}es"));
    } else {
        forms.push(format!("{term}s"));
    }

    if let Some(stem) = term.strip_suffix('e') {
        forms.push(format!("{term}d"));
        forms.push(format!("{stem}ing"));
    } else {
        forms.push(format!("{term}ed"));
        forms.push(format!("{term}ing"));
        if let Some(last) = doubled_final_consonant(term) {
            forms.push(format!("{term}{last}ed"));
            forms.push(format!("{term}{last}ing"));
        }
    }
    forms
}

fn doubled_final_consonant(term: &str) -> Option<char> {
    let mut characters = term.chars().rev();
    let last = characters.next()?;
    let previous = characters.next()?;
    let before = characters.next()?;
    (last.is_ascii_alphabetic()
        && !is_ascii_vowel(last)
        && is_ascii_vowel(previous)
        && !is_ascii_vowel(before))
    .then_some(last)
}

fn is_ascii_vowel(character: char) -> bool {
    matches!(character.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u')
}

fn normalize_case(value: &str, case_sensitive: bool) -> String {
    if case_sensitive {
        value.to_owned()
    } else {
        value.to_lowercase()
    }
}

fn is_latin_word_character(character: char) -> bool {
    (character.is_alphanumeric() && !is_cjk(character)) || matches!(character, '_' | '\'' | '’')
}

fn is_latin_term_character(character: char) -> bool {
    is_latin_word_character(character) || character.is_whitespace() || character == '-'
}

fn has_literal_boundaries(text: &str, start: usize, end: usize) -> bool {
    text[..start]
        .chars()
        .next_back()
        .is_none_or(|character| !is_latin_word_character(character))
        && text[end..]
            .chars()
            .next()
            .is_none_or(|character| !is_latin_word_character(character))
}

fn ranges_overlap(left: &MatchCandidate, right: &MatchCandidate) -> bool {
    left.start < right.end && right.start < left.end
}

fn is_cjk(character: char) -> bool {
    matches!(
        character as u32,
        0x1100..=0x11ff
            | 0x2e80..=0x2fdf
            | 0x3040..=0x30ff
            | 0x31f0..=0x31ff
            | 0x3400..=0x4dbf
            | 0x4e00..=0x9fff
            | 0xac00..=0xd7af
            | 0xf900..=0xfaff
            | 0xff66..=0xff9d
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latin_matching_uses_boundaries_and_common_inflections() {
        let matcher = TermMatcher::case_insensitive();

        assert!(!matcher.contains("the theater", "he"));
        assert!(matcher.contains("AI助手", "AI"));
        assert!(matcher.contains("Two actors' performances", "actor"));
        assert!(matcher.contains("Several cities changed", "city"));
        assert!(matcher.contains("She was translating it", "translate"));
        assert!(matcher.contains("He stopped running", "run"));
    }

    #[test]
    fn case_sensitivity_is_explicit() {
        assert!(TermMatcher::case_insensitive().contains("Apple", "apple"));
        assert!(!TermMatcher::new(true).contains("Apple", "apple"));
        assert!(TermMatcher::new(true).contains("Apple", "Apple"));
    }

    #[test]
    fn cjk_overlap_prefers_the_longest_term_but_keeps_separate_hits() {
        let matcher = TermMatcher::case_insensitive();
        let terms = ["纽约", "纽约时报"];

        assert_eq!(matcher.matching_indices("我读纽约时报。", &terms), [1]);
        assert_eq!(
            matcher.matching_indices("我读纽约时报，也去了纽约。", &terms),
            [1, 0]
        );
    }

    #[test]
    fn required_glossary_enforces_only_resolved_source_matches() {
        let glossary = BTreeMap::from([
            ("纽约".to_owned(), "纽约".to_owned()),
            ("纽约时报".to_owned(), "纽约时报中文版".to_owned()),
        ]);

        assert!(
            TermMatcher::case_insensitive()
                .missing_required("Read the 纽约时报", "阅读纽约时报中文版", &glossary)
                .is_empty()
        );
    }

    #[test]
    fn symbolic_latin_terms_still_require_boundaries() {
        let matcher = TermMatcher::case_insensitive();

        assert!(matcher.contains("Use C++ here", "C++"));
        assert!(!matcher.contains("Use C++Builder here", "C++"));
    }

    #[test]
    fn replacement_preserves_the_matched_surface_form() {
        let matcher = TermMatcher::case_insensitive();
        let replaced = matcher.replace_matches("ACTORS arrived", &["actor"], |_, matched| {
            format!("<{matched}>")
        });

        assert_eq!(replaced, "<ACTORS> arrived");
    }
}
