use serde_json::Value as JsonValue;
use subbake_core::ports::{ModelToolCall, ModelToolResult, ToolContinuation};

use crate::error::{AgentError, AgentResult};
use crate::event::ToolCallDraft;
use crate::tools::validate_tool_call;

#[derive(Debug, Clone)]
pub(super) struct Observation {
    pub(super) tool_name: String,
    pub(super) arguments: JsonValue,
    pub(super) text: String,
    pub(super) summary: String,
}

#[derive(Debug, Clone)]
pub(super) struct LoopState {
    pub(super) step: usize,
    pub(super) max_steps: usize,
    pub(super) observations: Vec<Observation>,
    pub(super) discovery_calls: usize,
    pub(super) force_no_tools: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DecisionAction {
    Respond,
    ToolCall,
    Plan,
    AskUser,
    NativeToolCalls,
}

pub(super) struct Decision {
    pub(super) action: DecisionAction,
    pub(super) text: String,
    pub(super) tool_name: Option<String>,
    pub(super) arguments: Option<JsonValue>,
    pub(super) tool_calls: Vec<ToolCallDraft>,
    pub(super) native_calls: Vec<ModelToolCall>,
    pub(super) native_continuation: Option<ToolContinuation>,
}

pub(super) struct NativeTurn {
    pub(super) continuation: ToolContinuation,
    pub(super) results: Vec<ModelToolResult>,
}

impl Decision {
    pub(super) fn response(text: String) -> Self {
        Self {
            action: DecisionAction::Respond,
            text,
            tool_name: None,
            arguments: None,
            tool_calls: Vec::new(),
            native_calls: Vec::new(),
            native_continuation: None,
        }
    }

    pub(super) fn ask_user(text: String) -> Self {
        Self {
            action: DecisionAction::AskUser,
            text,
            tool_name: None,
            arguments: None,
            tool_calls: Vec::new(),
            native_calls: Vec::new(),
            native_continuation: None,
        }
    }

    pub(super) fn native(
        text: String,
        native_calls: Vec<ModelToolCall>,
        native_continuation: Option<ToolContinuation>,
    ) -> Self {
        Self {
            action: DecisionAction::NativeToolCalls,
            text,
            tool_name: None,
            arguments: None,
            tool_calls: Vec::new(),
            native_calls,
            native_continuation,
        }
    }
}

pub(super) fn invalid_decision_response(error: &AgentError) -> Decision {
    Decision::ask_user(format!(
        "I couldn't execute the proposed action because its arguments were invalid: {error}"
    ))
}

pub(super) fn parse_decision_value(
    parsed: &JsonValue,
    is_discovery_tool: impl Fn(&str) -> bool,
) -> AgentResult<Decision> {
    let object = parsed
        .as_object()
        .ok_or_else(|| AgentError::InvalidDecision {
            message: "agent decision must be a JSON object".to_owned(),
        })?;
    let raw_action = object
        .get("action")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| AgentError::InvalidDecision {
            message: "agent decision is missing `action`".to_owned(),
        })?;
    let action = match raw_action {
        "final_tool_call" | "tool_call" => DecisionAction::ToolCall,
        "respond" => DecisionAction::Respond,
        "plan" => DecisionAction::Plan,
        "ask_user" => DecisionAction::AskUser,
        other => {
            return Err(AgentError::InvalidDecision {
                message: format!("unsupported agent action `{other}`"),
            });
        }
    };
    let text = object
        .get("text")
        .and_then(JsonValue::as_str)
        .unwrap_or("")
        .to_owned();
    let mut tool_name = None;
    let mut arguments = None;
    let mut tool_calls = Vec::new();
    if action == DecisionAction::ToolCall {
        let name = object
            .get("tool_name")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| AgentError::InvalidDecision {
                message: "tool call is missing `tool_name`".to_owned(),
            })?;
        let args = object
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        validate_tool_call(name, &args).map_err(|error| AgentError::ToolArguments {
            message: error.to_string(),
        })?;
        tool_name = Some(name.to_owned());
        arguments = Some(args);
    } else if action == DecisionAction::Plan {
        let calls = object
            .get("tool_calls")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| AgentError::InvalidDecision {
                message: "plan is missing `tool_calls`".to_owned(),
            })?;
        if calls.is_empty() {
            return Err(AgentError::InvalidDecision {
                message: "plan must contain at least one tool call".to_owned(),
            });
        }
        for call in calls {
            let name = call
                .get("tool_name")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| AgentError::InvalidDecision {
                    message: "planned call is missing `tool_name`".to_owned(),
                })?;
            let args = call
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            validate_tool_call(name, &args).map_err(|error| AgentError::ToolArguments {
                message: error.to_string(),
            })?;
            if is_discovery_tool(name) {
                return Err(AgentError::InvalidDecision {
                    message: "discovery tools must run before creating a plan".to_owned(),
                });
            }
            tool_calls.push(ToolCallDraft {
                tool_name: name.to_owned(),
                arguments: args,
            });
        }
    }
    Ok(Decision {
        action,
        text,
        tool_name,
        arguments,
        tool_calls,
        native_calls: Vec::new(),
        native_continuation: None,
    })
}
