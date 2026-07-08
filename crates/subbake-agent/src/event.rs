//! Typed event kinds for the agent session log.
//!
//! Replaces the stringly-typed `kind` field from Python
//! (`agent/session.py` events list). Every recorded event has a
//! well-known variant; unknown or ad-hoc kinds are rejected at compile time.


use serde::{Deserialize, Serialize};

/// A file-operation payload attached to a session event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileOpEventData {
    pub action: String,
    pub path: String,
    pub new_path: Option<String>,
    pub backup_path: Option<String>,
    pub group_id: Option<String>,
    #[serde(default)]
    pub undone: bool,
}

/// Every kind of event that can appear in a session trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventKind {
    User { text: String },
    Assistant { text: String },
    AskUser { text: String },
    ToolCall { tool_name: String, arguments: serde_json::Value },
    FinalToolCall { tool_name: String, arguments: serde_json::Value },
    FileOperation(FileOpEventData),
    Plan { message: String, tool_calls: Vec<ToolCallDraft> },
    Approve,
    Reject,
    Undo,
    Profile { name: String },
    Error { text: String },
}

/// Stub for a tool call within a pending plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallDraft {
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

/// A pending plan stored in the session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingPlan {
    pub message: String,
    pub tool_calls: Vec<ToolCallDraft>,
    pub created_at: String,
}
