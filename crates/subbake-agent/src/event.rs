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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_undo: Option<crate::guard::SemanticUndo>,
    pub group_id: Option<String>,
    #[serde(default)]
    pub undone: bool,
}

/// Every kind of event that can appear in a session trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventKind {
    User {
        text: String,
    },
    Assistant {
        text: String,
    },
    AskUser {
        text: String,
    },
    ToolCall {
        tool_name: String,
        arguments: serde_json::Value,
    },
    ToolFailure {
        tool_name: String,
        category: String,
        error: String,
    },
    FinalToolCall {
        tool_name: String,
        arguments: serde_json::Value,
    },
    FileOperation(FileOpEventData),
    Plan {
        message: String,
        tool_calls: Vec<ToolCallDraft>,
    },
    CommandApprovalRequested {
        tool_call: ToolCallDraft,
        reason: String,
    },
    CommandApproved,
    CommandRejected,
    Approve,
    Reject,
    Undo,
    Profile {
        name: String,
    },
    Error {
        text: String,
    },
    Cancelled,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingCommandApproval {
    pub tool_call: ToolCallDraft,
    #[serde(default)]
    pub call_id: String,
    #[serde(default)]
    pub remaining_tool_calls: Vec<PendingToolCall>,
    pub reason: String,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingToolCall {
    pub call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

/// Provider-neutral state for an agent turn paused at an approval boundary.
///
/// Native provider continuations stay in memory. This persisted transcript is
/// sufficient to rebuild a fresh model request after the process is restarted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingAgentTurn {
    pub input: String,
    #[serde(default)]
    pub dialogue: Option<String>,
    #[serde(default)]
    pub legacy_pending: Option<String>,
    pub effective_defaults: String,
    #[serde(default)]
    pub exchanges: Vec<PendingToolExchange>,
    #[serde(default)]
    pub completed_mutations: Vec<ToolCallDraft>,
    #[serde(default)]
    pub failure_counts: Vec<PendingFailureCount>,
    #[serde(default)]
    pub aggregate_failure_counts: Vec<PendingAggregateFailureCount>,
    #[serde(default)]
    pub steps_used: usize,
    #[serde(default)]
    pub legacy_mode: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingToolExchange {
    pub call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub feedback: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingFailureCount {
    pub tool_call: ToolCallDraft,
    pub category: String,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingAggregateFailureCount {
    pub tool_name: String,
    pub category: String,
    pub count: usize,
}
