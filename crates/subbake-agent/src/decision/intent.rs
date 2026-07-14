use std::io;

use serde_json::Value as JsonValue;

use crate::tools::{ToolIntent, find_tool_spec};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Route {
    Respond(String),
    AskUser(String),
    Act {
        intent: ToolIntent,
        request: String,
        inspect_project: bool,
    },
}

pub(super) fn parse_route(value: &JsonValue, original: &str) -> io::Result<Route> {
    let action = value
        .get("route")
        .or_else(|| value.get("action"))
        .and_then(JsonValue::as_str)
        .ok_or_else(|| io::Error::other("semantic route is missing `route`"))?;
    let text = value
        .get("text")
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
        .to_owned();
    match action {
        "respond" => Ok(Route::Respond(text)),
        "ask_user" => Ok(Route::AskUser(text)),
        "act" => {
            let intent = parse_intent(
                value
                    .get("intent")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| io::Error::other("action route is missing `intent`"))?,
            )?;
            let request = value
                .get("restated_request")
                .and_then(JsonValue::as_str)
                .filter(|request| !request.trim().is_empty())
                .unwrap_or(original)
                .to_owned();
            Ok(Route::Act {
                intent,
                request,
                inspect_project: value
                    .get("inspect_project")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false),
            })
        }
        // Compatibility for structured decision backends: normalize their
        // proposed tool into an intent, then enforce the same allowlist.
        "tool_call" | "final_tool_call" => {
            let tool = value
                .get("tool_name")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| io::Error::other("tool route is missing `tool_name`"))?;
            Ok(Route::Act {
                intent: intent_for_tool(tool)?,
                request: original.to_owned(),
                inspect_project: false,
            })
        }
        "plan" => {
            let tool = value
                .get("tool_calls")
                .and_then(JsonValue::as_array)
                .and_then(|calls| calls.first())
                .and_then(|call| call.get("tool_name"))
                .and_then(JsonValue::as_str)
                .ok_or_else(|| io::Error::other("plan route has no tool calls"))?;
            Ok(Route::Act {
                intent: intent_for_tool(tool)?,
                request: original.to_owned(),
                inspect_project: false,
            })
        }
        other => Err(io::Error::other(format!(
            "unsupported semantic route `{other}`"
        ))),
    }
}

pub(super) fn intent_for_tool(tool: &str) -> io::Result<ToolIntent> {
    find_tool_spec(tool)
        .map(|spec| spec.default_intent)
        .ok_or_else(|| io::Error::other(format!("unknown routed tool `{tool}`")))
}

fn parse_intent(value: &str) -> io::Result<ToolIntent> {
    ToolIntent::parse(value)
        .ok_or_else(|| io::Error::other(format!("unsupported intent `{value}`")))
}
