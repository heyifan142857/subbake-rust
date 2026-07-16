use serde::Serialize;
use serde_json::Value as JsonValue;
use subbake_core::ports::{ModelToolCall, ModelToolResult, ToolContinuation};

use crate::error::{AgentError, AgentResult};

#[derive(Debug, Clone, Serialize)]
pub(super) struct ToolFeedback {
    pub(super) success: bool,
    pub(super) tool: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) error_category: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) available_tools: Vec<String>,
}

impl ToolFeedback {
    pub(super) fn success(tool: &str, output: String) -> Self {
        Self {
            success: true,
            tool: tool.to_owned(),
            output: Some(output),
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
            output: None,
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
