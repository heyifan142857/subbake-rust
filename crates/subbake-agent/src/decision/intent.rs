use std::io;

use serde_json::Value as JsonValue;

use crate::tools::ToolScope;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Route {
    Respond(String),
    AskUser(String),
    Act {
        scope: ToolScope,
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
            let scope = parse_scope(
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
                scope,
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
                scope: scope_for_tool(tool)?,
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
                scope: scope_for_tool(tool)?,
                request: original.to_owned(),
                inspect_project: false,
            })
        }
        other => Err(io::Error::other(format!(
            "unsupported semantic route `{other}`"
        ))),
    }
}

pub(super) fn scope_for_tool(tool: &str) -> io::Result<ToolScope> {
    let scope = match tool {
        "list_files" | "search_files" | "read_file_preview" | "read_file" => ToolScope::Browse,
        "candidate_subtitles" | "translate_file" | "translate_series" => ToolScope::Translate,
        "transcribe_audio" => ToolScope::Transcribe,
        "recent_translations" | "edit_subtitle" => ToolScope::Edit,
        "diagnose_path" | "diagnose_text" => ToolScope::Diagnose,
        "create_file" => ToolScope::FileCreate,
        "append_file" => ToolScope::FileAppend,
        "replace_in_file" => ToolScope::FileReplace,
        "rename_path" => ToolScope::FileRename,
        "delete_file" => ToolScope::FileDelete,
        "list_profiles" | "switch_profile" => ToolScope::Profile,
        "manage_whisper" => ToolScope::Whisper,
        other => return Err(io::Error::other(format!("unknown routed tool `{other}`"))),
    };
    Ok(scope)
}

fn parse_scope(value: &str) -> io::Result<ToolScope> {
    match value {
        "browse" => Ok(ToolScope::Browse),
        "translate" => Ok(ToolScope::Translate),
        "transcribe" => Ok(ToolScope::Transcribe),
        "edit" => Ok(ToolScope::Edit),
        "diagnose" => Ok(ToolScope::Diagnose),
        "file_create" => Ok(ToolScope::FileCreate),
        "file_append" => Ok(ToolScope::FileAppend),
        "file_replace" => Ok(ToolScope::FileReplace),
        "file_rename" => Ok(ToolScope::FileRename),
        "file_delete" => Ok(ToolScope::FileDelete),
        "profile" => Ok(ToolScope::Profile),
        "whisper" => Ok(ToolScope::Whisper),
        other => Err(io::Error::other(format!("unsupported intent `{other}`"))),
    }
}
