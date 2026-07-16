use subbake_core::CancellationGuard;
use subbake_core::editing::SubtitleEditPayload;
use subbake_core::entities::{
    BatchTranslationResult, GlossaryEntry, ReviewResult, TerminologyPreflightResult,
    TranslationLine, Usage,
};
use subbake_core::error::{CoreError, CoreResult, LlmCallError};
use subbake_core::languages::{language_short_code, normalize_language_name};
use subbake_core::ports::{GenerationInput, GenerationRequest, GenerationResponse, LlmBackend};

use serde_json::Value as JsonValue;

#[derive(Debug, Clone)]
pub struct MockBackend {
    model: String,
}

impl MockBackend {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new("mock-zh")
    }
}

impl LlmBackend for MockBackend {
    fn supports_terminology_preflight(&self) -> bool {
        true
    }

    fn provider_name(&self) -> &str {
        "mock"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn execute(
        &mut self,
        request: GenerationRequest,
        cancellation: &CancellationGuard,
    ) -> Result<GenerationResponse, LlmCallError> {
        cancellation.check().map_err(LlmCallError::from)?;
        if request.tools.is_some() {
            return Err(LlmCallError::UnsupportedCapability(
                "native tools".to_owned(),
            ));
        }
        let messages = match request.input {
            GenerationInput::Messages(messages) => messages,
            GenerationInput::Continue { .. } => {
                return Err(LlmCallError::ContinuationMismatch(
                    "mock backend cannot continue native tool calls".to_owned(),
                ));
            }
        };
        let prompt = messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let task = extract_between(&prompt, "TASK_START", "TASK_END")?.trim();
        let payload = match task {
            "translate_subtitles" => serde_json::to_value(translate_subtitles(&prompt)?),
            "review_translations" => serde_json::to_value(review_translations(&prompt)?),
            "extract_terminology" => serde_json::to_value(TerminologyPreflightResult::default()),
            "agent_edit_subtitle" => serde_json::to_value(edit_subtitles(&prompt)?),
            other => {
                return Err(LlmCallError::UnsupportedCapability(format!(
                    "unsupported mock task `{other}`"
                )));
            }
        }
        .map_err(|error| {
            LlmCallError::InvalidResponse(format!("mock response encode failed: {error}"))
        })?;
        let input_tokens = estimate_tokens(&prompt);
        let output_tokens = estimate_tokens(&payload.to_string());
        Ok(GenerationResponse::json(
            payload,
            Usage {
                input_tokens,
                output_tokens,
                total_tokens: input_tokens + output_tokens,
            },
        ))
    }

    fn check_credentials(&self) -> CoreResult<(bool, String)> {
        Ok((
            true,
            "Mock provider does not require credentials.".to_owned(),
        ))
    }
}

fn edit_subtitles(prompt: &str) -> CoreResult<SubtitleEditPayload> {
    let edit_json = extract_between(prompt, "EDIT_JSON_START", "EDIT_JSON_END")?;
    let payload: JsonValue = serde_json::from_str(edit_json)
        .map_err(|err| CoreError::Backend(format!("invalid edit json: {err}")))?;
    let instruction = payload["instruction"]
        .as_str()
        .unwrap_or_default()
        .to_lowercase();
    let entries = payload["lines"]
        .as_array()
        .ok_or_else(|| CoreError::Backend("mock edit is missing lines array".to_owned()))?;

    let lines = entries
        .iter()
        .map(|entry| {
            let id = entry["id"].as_str().unwrap_or_default().to_owned();
            let current = entry["translation"].as_str().unwrap_or_default();
            let translation = if current.trim().is_empty() {
                String::new()
            } else if instruction.contains("uppercase") || instruction.contains("大写") {
                current.to_uppercase()
            } else {
                format!("{current} [edited]")
            };
            TranslationLine { id, translation }
        })
        .collect();

    Ok(SubtitleEditPayload {
        lines,
        edit_notes: "Mock subtitle edit completed.".to_owned(),
    })
}

fn translate_subtitles(prompt: &str) -> CoreResult<BatchTranslationResult> {
    let context_json = extract_between(prompt, "CONTEXT_JSON_START", "CONTEXT_JSON_END")?;
    let context: JsonValue = serde_json::from_str(context_json)
        .map_err(|err| CoreError::Backend(format!("invalid context json: {err}")))?;
    let target_language = context["tgt"]
        .as_str()
        .map(|value| normalize_language_name(value, false))
        .unwrap_or_else(|| "zh-Hans".to_owned());
    let tag = language_short_code(&target_language);

    let batch_json = extract_between(prompt, "BATCH_JSON_START", "BATCH_JSON_END")?;
    let batch: JsonValue = serde_json::from_str(batch_json)
        .map_err(|err| CoreError::Backend(format!("invalid batch json: {err}")))?;
    let entries = batch["lines"]
        .as_array()
        .ok_or_else(|| CoreError::Backend("mock batch is missing lines array".to_owned()))?;

    let mut lines = Vec::new();
    let mut glossary_updates = Vec::new();
    for entry in entries {
        let id = entry["id"].as_str().unwrap_or_default().to_owned();
        let text = entry["text"].as_str().unwrap_or_default().to_owned();
        let translation = if text.trim().is_empty() {
            String::new()
        } else {
            format!("[MOCK-{tag}] {text}")
        };
        lines.push(TranslationLine { id, translation });

        if glossary_updates.is_empty() && !text.trim().is_empty() {
            let source_word = text.split_whitespace().next().unwrap_or(&text).to_owned();
            glossary_updates.push(GlossaryEntry {
                source: source_word,
                target: format!("[MOCK-{tag}]"),
            });
        }
    }

    Ok(BatchTranslationResult {
        lines,
        summary: "Mock summary of the latest subtitle batch.".to_owned(),
        glossary_updates,
    })
}

fn review_translations(prompt: &str) -> CoreResult<ReviewResult> {
    let review_json = extract_between(prompt, "REVIEW_JSON_START", "REVIEW_JSON_END")?;
    let review: JsonValue = serde_json::from_str(review_json)
        .map_err(|err| CoreError::Backend(format!("invalid review json: {err}")))?;
    let entries = review["lines"]
        .as_array()
        .ok_or_else(|| CoreError::Backend("mock review is missing lines array".to_owned()))?;
    let lines = entries
        .iter()
        .map(|entry| TranslationLine {
            id: entry["id"].as_str().unwrap_or_default().to_owned(),
            translation: entry["translation"].as_str().unwrap_or_default().to_owned(),
        })
        .collect();
    Ok(ReviewResult {
        lines,
        review_notes: "Mock targeted review completed.".to_owned(),
    })
}

fn extract_between<'a>(text: &'a str, start_marker: &str, end_marker: &str) -> CoreResult<&'a str> {
    let start = text
        .find(start_marker)
        .ok_or_else(|| CoreError::Backend(format!("missing marker {start_marker}")))?
        + start_marker.len();
    let tail = &text[start..];
    let end = tail
        .find(end_marker)
        .ok_or_else(|| CoreError::Backend(format!("missing marker {end_marker}")))?;
    Ok(&tail[..end])
}

fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use subbake_core::ports::ChatMessage;

    fn translate_prompt(text: &str, target_language: &str) -> String {
        let context = serde_json::json!({
            "src": "English",
            "tgt": target_language,
            "batch_index": 1,
            "fast": false,
        });
        let batch = serde_json::json!({"lines":[{"id":"1","text":text}]});
        let context_json = serde_json::to_string(&context).unwrap_or_default();
        let batch_json = serde_json::to_string(&batch).unwrap_or_default();
        format!(
            "TASK_START\ntranslate_subtitles\nTASK_END\n\
             CONTEXT_JSON_START{context_json}CONTEXT_JSON_END\n\
             BATCH_JSON_START{batch_json}BATCH_JSON_END"
        )
    }

    fn review_prompt(translation: &str) -> String {
        let review = serde_json::json!({
            "lines": [{
                "id": "1",
                "source": "Meet Alice.",
                "translation": translation,
            }],
        });
        let review_json = serde_json::to_string(&review).unwrap_or_default();
        format!(
            "TASK_START\nreview_translations\nTASK_END\n\
             REVIEW_JSON_START{review_json}REVIEW_JSON_END"
        )
    }

    #[test]
    fn mock_translates_batch_and_produces_glossary_update() {
        let mut backend = MockBackend::new("mock-zh");
        let messages = vec![
            ChatMessage::system(""),
            ChatMessage::user(translate_prompt("hello world", "Chinese")),
        ];
        let response = backend
            .execute(
                GenerationRequest::json(messages),
                &CancellationGuard::never(),
            )
            .expect("mock generate");
        let (json, _) = response.into_json().expect("JSON response");
        let batch: BatchTranslationResult =
            serde_json::from_value(json).expect("translation payload");

        assert_eq!(batch.lines.len(), 1);
        assert!(batch.lines[0].translation.contains("[MOCK-"));
        assert!(!batch.glossary_updates.is_empty());
        assert_eq!(batch.glossary_updates[0].source, "hello");
    }

    #[test]
    fn mock_leaves_empty_text_empty() {
        let mut backend = MockBackend::default();
        let messages = vec![
            ChatMessage::system(""),
            ChatMessage::user(translate_prompt("  ", "Chinese")),
        ];
        let response = backend
            .execute(
                GenerationRequest::json(messages),
                &CancellationGuard::never(),
            )
            .expect("mock generate");
        let (json, _) = response.into_json().expect("JSON response");
        let batch: BatchTranslationResult =
            serde_json::from_value(json).expect("translation payload");

        assert_eq!(batch.lines[0].translation, "");
        assert!(batch.glossary_updates.is_empty());
    }

    #[test]
    fn mock_reviews_without_changing_valid_lines() {
        let mut backend = MockBackend::default();
        let messages = vec![
            ChatMessage::system(""),
            ChatMessage::user(review_prompt("[MOCK-ZH] Meet Alice.")),
        ];
        let response = backend
            .execute(
                GenerationRequest::json(messages),
                &CancellationGuard::never(),
            )
            .expect("mock review");
        let (json, _) = response.into_json().expect("JSON response");
        let review: ReviewResult = serde_json::from_value(json).expect("review payload");

        assert_eq!(review.lines[0].translation, "[MOCK-ZH] Meet Alice.");
        assert!(!review.review_notes.is_empty());
    }
}
