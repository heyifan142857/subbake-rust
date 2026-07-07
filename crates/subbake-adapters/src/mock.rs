use subbake_core::entities::{BatchTranslationResult, TranslationLine, Usage};
use subbake_core::error::{CoreError, CoreResult};
use subbake_core::languages::{language_short_code, normalize_language_name};
use subbake_core::pipeline::unescape_field;
use subbake_core::ports::{BackendJsonResult, BackendPayload, ChatMessage, LlmBackend};

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
    fn provider_name(&self) -> &str {
        "mock"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
        let prompt = messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let task = extract_between(&prompt, "TASK_START", "TASK_END")?.trim();
        let result = match task {
            "translate_subtitles" => translate_subtitles(&prompt)?,
            other => {
                return Err(CoreError::Backend(format!(
                    "unsupported mock task `{other}`"
                )));
            }
        };
        let input_tokens = estimate_tokens(&prompt);
        let output_tokens = estimate_tokens(&format!("{result:?}"));
        Ok(BackendJsonResult {
            payload: BackendPayload::Translation(result),
            usage: Usage {
                input_tokens,
                output_tokens,
                total_tokens: input_tokens + output_tokens,
            },
        })
    }

    fn check_credentials(&self) -> CoreResult<(bool, String)> {
        Ok((true, "Mock provider does not require credentials.".to_owned()))
    }
}

fn translate_subtitles(prompt: &str) -> CoreResult<BatchTranslationResult> {
    let target_language = extract_context_value(prompt, "target_language")?
        .map(|value| normalize_language_name(&value, false))
        .unwrap_or_else(|| "Chinese".to_owned());
    let tag = language_short_code(&target_language);
    let body = extract_between(prompt, "BATCH_LINES_START", "BATCH_LINES_END")?;

    let mut lines = Vec::new();
    for raw_line in body.lines().filter(|line| !line.trim().is_empty()) {
        let (id, text) = raw_line
            .split_once('\t')
            .ok_or_else(|| CoreError::Backend("mock batch line is missing tab separator".to_owned()))?;
        let text = unescape_field(text)?;
        let translation = if text.trim().is_empty() {
            String::new()
        } else {
            format!("[MOCK-{tag}] {text}")
        };
        lines.push(TranslationLine {
            id: unescape_field(id)?,
            translation,
        });
    }

    Ok(BatchTranslationResult {
        lines,
        summary: "Mock summary of the latest subtitle batch.".to_owned(),
        glossary_updates: Vec::new(),
    })
}

fn extract_context_value(prompt: &str, key: &str) -> CoreResult<Option<String>> {
    let context = extract_between(prompt, "CONTEXT_START", "CONTEXT_END")?;
    for raw_line in context.lines() {
        let Some((left, right)) = raw_line.split_once('=') else {
            continue;
        };
        if left == key {
            return Ok(Some(unescape_field(right)?));
        }
    }
    Ok(None)
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
