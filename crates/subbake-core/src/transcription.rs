//! Provider-neutral recognition data and deterministic subtitle segmentation.
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

use crate::{CoreError, CoreResult};

pub const SEGMENTATION_VERSION: u64 = 1;
const PAUSE_MS: u64 = 700;
const MAX_DURATION_MS: u64 = 6_000;
const MAX_GRAPHEMES: usize = 84;

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpokenLanguage {
    #[default]
    Auto,
    Explicit(String),
}

impl SpokenLanguage {
    pub fn parse(value: &str) -> CoreResult<Self> {
        let normalized = crate::languages::normalize_language(value, true)
            .map_err(|error| CoreError::DataInvariant(error.to_string()))?;
        Ok(if normalized == "Auto" {
            Self::Auto
        } else {
            Self::Explicit(normalized)
        })
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Auto => "Auto",
            Self::Explicit(language) => language,
        }
    }

    pub fn recognition_code(&self) -> &str {
        match self {
            Self::Auto => "auto",
            Self::Explicit(language) => language.split('-').next().unwrap_or(language),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Segmentation {
    #[default]
    Smart,
    Segment,
}

impl Segmentation {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "smart" => Some(Self::Smart),
            "segment" => Some(Self::Segment),
            _ => None,
        }
    }
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Smart => "smart",
            Self::Segment => "segment",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeSpan {
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecognitionToken {
    pub text: String,
    pub timing: Option<TimeSpan>,
    pub probability: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecognitionSegment {
    pub text: String,
    pub timing: TimeSpan,
    pub tokens: Vec<RecognitionToken>,
    pub speaker: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecognitionResult {
    pub language: String,
    pub segments: Vec<RecognitionSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CueSource {
    pub segment_index: usize,
    /// UTF-8 byte range into the original segment, including boundary whitespace.
    pub text_start: usize,
    pub text_end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecognizedCue {
    pub text: String,
    pub timing: TimeSpan,
    pub speaker: Option<String>,
    pub source: CueSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingFallback {
    TextMismatch,
    MissingTiming,
    InvalidTiming,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentDiagnostic {
    pub segment_index: usize,
    pub fallback: Option<TimingFallback>,
    pub oversized_cues: usize,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentedRecognition {
    pub cues: Vec<RecognizedCue>,
    pub diagnostics: Vec<SegmentDiagnostic>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkTranscriptionReport {
    pub index: usize,
    pub language: String,
    pub input_start_ms: u64,
    pub input_end_ms: u64,
    pub diagnostics: Vec<SegmentDiagnostic>,
    pub sources: Vec<CueSource>,
    pub recognized_cues: usize,
    pub suspected_repetitions: usize,
    /// Overlap with the preceding chunk's last retained cue; diagnostic only.
    #[serde(default)]
    pub boundary_overlap_ms: u64,
    pub raw_json_path: Option<std::path::PathBuf>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptionReport {
    pub chunks: Vec<ChunkTranscriptionReport>,
}

impl TranscriptionReport {
    pub fn fallback_count(&self) -> usize {
        self.chunks
            .iter()
            .flat_map(|chunk| &chunk.diagnostics)
            .filter(|diagnostic| diagnostic.fallback.is_some())
            .count()
    }
    pub fn languages(&self) -> Vec<&str> {
        let mut languages = self
            .chunks
            .iter()
            .map(|chunk| chunk.language.as_str())
            .collect::<Vec<_>>();
        languages.sort_unstable();
        languages.dedup();
        languages
    }
}

/// Segment text is authoritative; uncertain token alignment falls back without dropping text.
pub fn segment_recognition(
    result: &RecognitionResult,
    policy: Segmentation,
) -> CoreResult<SegmentedRecognition> {
    let mut output = SegmentedRecognition::default();
    for (index, segment) in result.segments.iter().enumerate() {
        if segment.timing.end_ms <= segment.timing.start_ms {
            return Err(CoreError::DataInvariant(format!(
                "recognition segment {index} has invalid timing"
            )));
        }
        if segment.text.trim().is_empty() {
            continue;
        }
        let rebuilt = if policy == Segmentation::Smart {
            smart_cues(segment, index)
        } else {
            Ok(vec![whole_cue(segment, index)])
        };
        let (cues, fallback) = match rebuilt {
            Ok(cues) => (cues, None),
            Err(reason) => (vec![whole_cue(segment, index)], Some(reason)),
        };
        let mut next_byte = 0;
        for cue in &cues {
            if cue.source.text_start != next_byte
                || cue.text != segment.text[cue.source.text_start..cue.source.text_end].trim()
            {
                return Err(CoreError::DataInvariant(
                    "segmentation changed source text".to_owned(),
                ));
            }
            next_byte = cue.source.text_end;
        }
        if next_byte != segment.text.len() {
            return Err(CoreError::DataInvariant(
                "segmentation lost source text".to_owned(),
            ));
        }
        let oversized_cues = cues
            .iter()
            .filter(|cue| {
                cue.timing.end_ms - cue.timing.start_ms > MAX_DURATION_MS
                    || cue.text.graphemes(true).count() > MAX_GRAPHEMES
            })
            .count();
        output.diagnostics.push(SegmentDiagnostic {
            segment_index: index,
            fallback,
            oversized_cues,
        });
        output.cues.extend(cues);
    }
    Ok(output)
}

fn whole_cue(segment: &RecognitionSegment, index: usize) -> RecognizedCue {
    RecognizedCue {
        text: segment.text.trim().to_owned(),
        timing: segment.timing,
        speaker: segment.speaker.clone(),
        source: CueSource {
            segment_index: index,
            text_start: 0,
            text_end: segment.text.len(),
        },
    }
}

fn smart_cues(
    segment: &RecognitionSegment,
    index: usize,
) -> Result<Vec<RecognizedCue>, TimingFallback> {
    let combined = segment
        .tokens
        .iter()
        .map(|token| token.text.as_str())
        .collect::<String>();
    if combined != segment.text {
        return Err(TimingFallback::TextMismatch);
    }
    let mut offset = 0;
    let mut timed = Vec::new();
    let mut previous_start = segment.timing.start_ms;
    let mut previous_end = segment.timing.start_ms;
    for token in &segment.tokens {
        let start = offset;
        offset += token.text.len();
        if token.text.trim().is_empty() {
            continue;
        }
        let timing = token.timing.ok_or(TimingFallback::MissingTiming)?;
        if timing.start_ms < previous_start
            || timing.end_ms < previous_end
            || timing.end_ms < timing.start_ms
            || timing.start_ms < segment.timing.start_ms
            || timing.end_ms > segment.timing.end_ms
        {
            return Err(TimingFallback::InvalidTiming);
        }
        previous_start = timing.start_ms;
        previous_end = timing.end_ms;
        timed.push((start, offset, timing));
    }
    if timed.is_empty() {
        return Err(TimingFallback::MissingTiming);
    }

    // Unicode word boundaries permit CJK splits, but never cut a grapheme or a timed token.
    let boundaries = segment
        .text
        .split_word_bound_indices()
        .map(|(start, part)| start + part.len())
        .collect::<Vec<_>>();
    let mut units: Vec<(usize, usize, TimeSpan)> = Vec::new();
    for (start, end, timing) in timed {
        let can_split = units
            .last()
            .is_some_and(|(_, previous_end, previous_timing)| {
                let prefix = &segment.text[*previous_end..start];
                let current = &segment.text[start..end];
                let first = current.trim_start().chars().next();
                let word_start = prefix.chars().any(char::is_whitespace)
                    || current.starts_with(char::is_whitespace)
                    || (boundaries.contains(&start) && first.is_some_and(char::is_alphanumeric));
                word_start && timing.start_ms >= previous_timing.end_ms
            });
        if !can_split && let Some(last) = units.last_mut() {
            last.1 = end;
            last.2.end_ms = timing.end_ms;
        } else {
            units.push((start, end, timing));
        }
    }
    let mut cues = Vec::new();
    let mut begin = 0;
    for position in 1..=units.len() {
        let last = units[position - 1];
        let current = units[begin];
        let should_break = if let Some(next) = units.get(position) {
            let text =
                segment.text[current.0..last.1].trim_end_matches(['"', '\'', '”', '’', '」', '』']);
            let sentence = text.ends_with(['.', '!', '?', '。', '！', '？']);
            next.2.start_ms.saturating_sub(last.2.end_ms) >= PAUSE_MS
                || (sentence && last.2.end_ms.saturating_sub(current.2.start_ms) >= 1_000)
                || next.2.end_ms.saturating_sub(current.2.start_ms) > MAX_DURATION_MS
                || segment.text[current.0..next.1].graphemes(true).count() > MAX_GRAPHEMES
        } else {
            true
        };
        if should_break {
            let text_start = if begin == 0 { 0 } else { units[begin - 1].1 };
            let text_end = if position == units.len() {
                segment.text.len()
            } else {
                last.1
            };
            if last.2.end_ms <= current.2.start_ms {
                return Err(TimingFallback::InvalidTiming);
            }
            cues.push(RecognizedCue {
                text: segment.text[text_start..text_end].trim().to_owned(),
                timing: TimeSpan {
                    start_ms: current.2.start_ms,
                    end_ms: last.2.end_ms,
                },
                speaker: segment.speaker.clone(),
                source: CueSource {
                    segment_index: index,
                    text_start,
                    text_end,
                },
            });
            begin = position;
        }
    }
    Ok(cues)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recognition(parts: &[(&str, u64, u64)]) -> RecognitionResult {
        RecognitionResult {
            language: "en".to_owned(),
            segments: vec![RecognitionSegment {
                text: parts.iter().map(|(text, _, _)| *text).collect(),
                timing: TimeSpan {
                    start_ms: 0,
                    end_ms: 20_000,
                },
                speaker: None,
                tokens: parts
                    .iter()
                    .map(|(text, start, end)| RecognitionToken {
                        text: (*text).to_owned(),
                        timing: Some(TimeSpan {
                            start_ms: *start,
                            end_ms: *end,
                        }),
                        probability: Some(0.001),
                    })
                    .collect(),
            }],
        }
    }

    #[test]
    fn pauses_split_identical_dialogue_without_losing_low_probability_words() {
        let raw = recognition(&[
            (" On", 100, 200),
            (" your", 200, 400),
            (" left.", 400, 900),
            (" On", 12_000, 12_100),
            (" your", 12_100, 12_400),
            (" left.", 12_400, 12_900),
        ]);
        let result = segment_recognition(&raw, Segmentation::Smart).expect("segment");
        assert_eq!(result.cues.len(), 2);
        assert_eq!(result.cues[0].text, "On your left.");
        assert_eq!(result.cues[1].timing.start_ms, 12_000);
        assert_eq!(result.cues[0].timing.end_ms, 900);
        assert!(result.diagnostics.iter().all(|d| d.fallback.is_none()));
    }

    #[test]
    fn subwords_and_attached_punctuation_stay_together() {
        let result = segment_recognition(
            &recognition(&[
                (" Super", 0, 100),
                ("hero", 100, 200),
                (",", 200, 200),
                (" go", 2_000, 2_200),
                ("!", 2_200, 2_200),
            ]),
            Segmentation::Smart,
        )
        .expect("segment");
        assert_eq!(
            result
                .cues
                .iter()
                .map(|cue| cue.text.as_str())
                .collect::<Vec<_>>(),
            ["Superhero,", "go!"]
        );
    }

    #[test]
    fn unicode_cjk_and_combining_marks_preserve_exact_text() {
        for parts in [
            vec![("你", 0, 100), ("好。", 100, 300), ("再见", 2_000, 2_500)],
            vec![
                (" Cafe", 0, 100),
                ("\u{301}", 100, 200),
                (" again", 2_000, 2_500),
            ],
        ] {
            let raw = recognition(&parts);
            let result = segment_recognition(&raw, Segmentation::Smart).expect("unicode");
            assert_eq!(result.cues.len(), 2);
            assert_eq!(
                result
                    .cues
                    .iter()
                    .map(|c| &raw.segments[0].text[c.source.text_start..c.source.text_end])
                    .collect::<String>(),
                raw.segments[0].text
            );
        }
    }

    #[test]
    fn invalid_token_alignment_falls_back_to_whole_segment() {
        for reason in [
            TimingFallback::TextMismatch,
            TimingFallback::MissingTiming,
            TimingFallback::InvalidTiming,
        ] {
            let mut raw = recognition(&[("Need a medic?", 100, 2_000)]);
            match reason {
                TimingFallback::TextMismatch => raw.segments[0].tokens[0].text = "Need".to_owned(),
                TimingFallback::MissingTiming => raw.segments[0].tokens[0].timing = None,
                TimingFallback::InvalidTiming => {
                    raw.segments[0].tokens[0].timing = Some(TimeSpan {
                        start_ms: 3_000,
                        end_ms: 2_000,
                    })
                }
            }
            let result = segment_recognition(&raw, Segmentation::Smart).expect("fallback");
            assert_eq!(result.cues[0].text, "Need a medic?");
            assert_eq!(result.cues[0].timing, raw.segments[0].timing);
            assert_eq!(result.diagnostics[0].fallback, Some(reason));
        }
    }

    #[test]
    fn segment_mode_keeps_boundaries_and_invalid_segment_fails() {
        let mut raw = recognition(&[("Hi", 100, 200), (" again", 10_000, 10_500)]);
        let result = segment_recognition(&raw, Segmentation::Segment).expect("segment");
        assert_eq!(result.cues.len(), 1);
        assert_eq!(result.cues[0].timing, raw.segments[0].timing);
        raw.segments[0].timing.end_ms = 0;
        assert!(segment_recognition(&raw, Segmentation::Smart).is_err());
    }

    #[test]
    fn duration_and_length_limits_break_at_safe_boundaries() {
        let result = segment_recognition(
            &recognition(&[("one", 0, 4_000), (" two", 4_000, 8_000)]),
            Segmentation::Smart,
        )
        .expect("duration");
        assert_eq!(result.cues.len(), 2);
        let long = "x".repeat(80);
        let result = segment_recognition(
            &recognition(&[(&long, 0, 100), (" nextword", 100, 200)]),
            Segmentation::Smart,
        )
        .expect("length");
        assert_eq!(result.cues.len(), 2);
    }
}
