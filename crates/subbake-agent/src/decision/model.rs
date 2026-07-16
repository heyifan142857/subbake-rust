use serde::Serialize;
use serde_json::Value as JsonValue;
use subbake_core::AgentToolOutcome;
use subbake_core::ports::{ModelToolCall, ModelToolResult, ToolContinuation};

use crate::error::{AgentError, AgentResult};

#[derive(Debug, Clone, Serialize)]
pub(super) struct ToolFeedback {
    pub(super) success: bool,
    pub(super) tool: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) outcome: Option<AgentToolOutcome>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) error_category: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) available_tools: Vec<String>,
}

impl ToolFeedback {
    pub(super) fn success(tool: &str, outcome: AgentToolOutcome) -> Self {
        Self {
            success: true,
            tool: tool.to_owned(),
            outcome: Some(outcome),
            error: None,
            error_category: None,
            available_tools: Vec::new(),
        }
    }

    pub(super) fn failure(
        tool: &str,
        error: String,
        category: &str,
        available_tools: Vec<String>,
    ) -> Self {
        Self {
            success: false,
            tool: tool.to_owned(),
            outcome: None,
            error: Some(error),
            error_category: Some(category.to_owned()),
            available_tools,
        }
    }

    pub(super) fn json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            r#"{"success":false,"error":"failed to serialize tool result"}"#.to_owned()
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct ToolExchange {
    pub(super) name: String,
    pub(super) arguments: JsonValue,
    pub(super) feedback: ToolFeedback,
}

#[derive(Debug, Default)]
pub(super) struct AgentTaskLoop {
    pub(super) exchanges: Vec<ToolExchange>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DecisionAction {
    Respond,
    AskUser,
    ToolCalls,
}

pub(super) struct Decision {
    pub(super) action: DecisionAction,
    pub(super) text: String,
    pub(super) calls: Vec<ModelToolCall>,
    pub(super) continuation: Option<ToolContinuation>,
}

impl Decision {
    pub(super) fn response(text: String) -> Self {
        Self {
            action: DecisionAction::Respond,
            text,
            calls: Vec::new(),
            continuation: None,
        }
    }

    pub(super) fn ask_user(text: String) -> Self {
        Self {
            action: DecisionAction::AskUser,
            text,
            calls: Vec::new(),
            continuation: None,
        }
    }

    pub(super) fn native_calls(
        text: String,
        calls: Vec<ModelToolCall>,
        continuation: Option<ToolContinuation>,
    ) -> Self {
        Self {
            action: DecisionAction::ToolCalls,
            text,
            calls,
            continuation,
        }
    }
}

pub(super) struct NativeTurn {
    pub(super) continuation: ToolContinuation,
    pub(super) results: Vec<ModelToolResult>,
}

pub(super) fn parse_json_decision(value: &JsonValue) -> AgentResult<Decision> {
    let object = value
        .as_object()
        .ok_or_else(|| AgentError::InvalidDecision {
            message: "agent decision must be a JSON object".to_owned(),
        })?;
    let action = object
        .get("action")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| AgentError::InvalidDecision {
            message: "agent decision is missing `action`".to_owned(),
        })?;
    let text = object
        .get("text")
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
        .to_owned();
    match action {
        "respond" => Ok(Decision::response(text)),
        "ask_user" => Ok(Decision::ask_user(text)),
        "tool_call" | "final_tool_call" => {
            let name = object
                .get("tool_name")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| AgentError::InvalidDecision {
                    message: "tool call is missing `tool_name`".to_owned(),
                })?;
            Ok(Decision {
                action: DecisionAction::ToolCalls,
                text,
                calls: vec![ModelToolCall {
                    id: "json-tool-call".to_owned(),
                    name: name.to_owned(),
                    arguments: object
                        .get("arguments")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({})),
                }],
                continuation: None,
            })
        }
        "plan" => Err(AgentError::InvalidDecision {
            message: "models must call mutating tools directly; plan approval is runtime-owned"
                .to_owned(),
        }),
        other => Err(AgentError::InvalidDecision {
            message: format!("unsupported agent action `{other}`"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use subbake_core::{
        AgentToolOutcome, BilingualOrder, ToolExecutionStatus, TranslationToolOutcome,
    };

    use super::*;

    #[test]
    fn successful_feedback_serializes_structured_facts_instead_of_free_text() {
        let feedback = ToolFeedback::success(
            "translate_file",
            AgentToolOutcome::Translation(TranslationToolOutcome {
                status: ToolExecutionStatus::Written,
                source_language: "Auto".to_owned(),
                target_language: "ja".to_owned(),
                provider: "mock".to_owned(),
                model: "mock-zh".to_owned(),
                output_format: "srt".to_owned(),
                bilingual: false,
                bilingual_order: BilingualOrder::TargetFirst,
                inputs: vec!["sample.srt".into()],
                outputs: vec!["sample.ja.translated.srt".into()],
                processed_files: 1,
                skipped: Vec::new(),
                subtitle_entries: 1,
                dry_run: false,
                cache_hits: 0,
                resumed_translation_batches: 0,
                resumed_review_batches: 0,
                translation_memory_hits: 0,
            }),
        );
        let json = feedback.json();

        assert!(json.contains(r#""operation":"translation""#));
        assert!(json.contains(r#""target_language":"ja""#));
        assert!(json.contains("sample.ja.translated.srt"));
        assert!(!json.contains(r#""output":"Translated:"#));
    }
}
