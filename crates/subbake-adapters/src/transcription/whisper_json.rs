//! The whisper.cpp v1.9.1 full-JSON wire contract stays at the adapter boundary.
use crate::error::{AdapterError, AdapterResult};
use serde::Deserialize;
use subbake_core::transcription::{
    RecognitionResult, RecognitionSegment, RecognitionToken, SpokenLanguage, TimeSpan,
};

#[derive(Deserialize)]
struct Output {
    model: Model,
    result: DetectedLanguage,
    transcription: Vec<Segment>,
}
#[derive(Deserialize)]
struct Model {
    multilingual: bool,
}
#[derive(Deserialize)]
struct DetectedLanguage {
    language: String,
}
#[derive(Deserialize)]
struct Offsets {
    from: i64,
    to: i64,
}
#[derive(Deserialize)]
struct Segment {
    text: String,
    offsets: Offsets,
    #[serde(default)]
    tokens: Vec<Token>,
    speaker: Option<String>,
}
#[derive(Deserialize)]
struct Token {
    id: i64,
    text: String,
    offsets: Option<Offsets>,
    p: Option<f32>,
}

impl Offsets {
    fn timing(&self) -> Option<TimeSpan> {
        Some(TimeSpan {
            start_ms: u64::try_from(self.from).ok()?,
            end_ms: u64::try_from(self.to).ok()?,
        })
    }
}

pub(super) fn parse(bytes: &[u8], requested: &SpokenLanguage) -> AdapterResult<RecognitionResult> {
    let output: Output =
        serde_json::from_slice(bytes).map_err(|source| AdapterError::Serialization {
            context: "parse whisper.cpp full JSON",
            source,
        })?;
    if !output.model.multilingual
        && matches!(requested, SpokenLanguage::Explicit(_))
        && requested.recognition_code() != "en"
    {
        return Err(AdapterError::invalid_input(
            "the selected Whisper model supports English only",
        ));
    }
    if output.result.language.is_empty() || output.result.language == "auto" {
        return Err(AdapterError::invalid_input(
            "whisper.cpp JSON did not report an actual language",
        ));
    }
    if matches!(requested, SpokenLanguage::Explicit(_))
        && output.result.language != requested.recognition_code()
    {
        return Err(AdapterError::invalid_input(format!(
            "whisper.cpp used {} instead of requested {}",
            output.result.language,
            requested.recognition_code()
        )));
    }
    let end_of_text = if output.model.multilingual {
        50257
    } else {
        50256
    };
    let segments = output
        .transcription
        .into_iter()
        .enumerate()
        .map(|(index, segment)| {
            let timing = segment
                .offsets
                .timing()
                .filter(|time| time.end_ms > time.start_ms)
                .ok_or_else(|| {
                    AdapterError::invalid_input(format!(
                        "whisper.cpp segment {index} has invalid offsets"
                    ))
                })?;
            Ok(RecognitionSegment {
                text: segment.text,
                timing,
                speaker: segment.speaker,
                tokens: segment
                    .tokens
                    .into_iter()
                    .filter(|token| token.id >= 0 && token.id < end_of_text)
                    .map(|token| RecognitionToken {
                        text: token.text,
                        timing: token.offsets.and_then(|offsets| offsets.timing()),
                        probability: token.p,
                    })
                    .collect(),
            })
        })
        .collect::<AdapterResult<Vec<_>>>()?;
    Ok(RecognitionResult {
        language: output.result.language,
        segments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn output() -> serde_json::Value {
        serde_json::json!({"model":{"multilingual":true},"result":{"language":"zh"},"transcription":[{
            "text":"你好。", "offsets":{"from":1000,"to":2000}, "tokens":[
                {"id":10,"text":"你","offsets":{"from":1000,"to":1500},"p":0.001},
                {"id":11,"text":"好。","p":0.9}, {"id":50257,"text":"[_EOT_]"}
            ]
        }]})
    }
    #[test]
    fn utf8_missing_timing_and_special_tokens_are_decoded_without_deleting_words() {
        let result = parse(output().to_string().as_bytes(), &SpokenLanguage::Auto).expect("JSON");
        assert_eq!(result.language, "zh");
        assert_eq!(result.segments[0].text, "你好。");
        assert_eq!(result.segments[0].tokens.len(), 2);
        assert_eq!(result.segments[0].tokens[0].probability, Some(0.001));
        assert!(result.segments[0].tokens[1].timing.is_none());
        let rebuilt = subbake_core::transcription::segment_recognition(&result, Default::default())
            .expect("fallback");
        assert_eq!(rebuilt.cues[0].text, "你好。");
        assert!(rebuilt.diagnostics[0].fallback.is_some());
    }
    #[test]
    fn malformed_segment_offsets_and_language_conflicts_fail_explicitly() {
        let mut value = output();
        assert!(
            parse(
                value.to_string().as_bytes(),
                &SpokenLanguage::parse("en").expect("language")
            )
            .is_err()
        );
        value["model"]["multilingual"] = false.into();
        assert!(
            parse(
                value.to_string().as_bytes(),
                &SpokenLanguage::parse("zh").expect("language")
            )
            .is_err()
        );
        value = output();
        value["transcription"][0]["offsets"]["from"] = (-1).into();
        assert!(parse(value.to_string().as_bytes(), &SpokenLanguage::Auto).is_err());
        assert!(parse(b"invalid JSON", &SpokenLanguage::Auto).is_err());
    }
}
