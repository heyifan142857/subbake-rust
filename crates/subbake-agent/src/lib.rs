mod command_policy;
mod config_editor;
pub mod decision;
mod discovery;
pub mod engine;
pub mod error;
pub mod evaluation;
pub mod event;
pub mod guard;
mod input_editor;
mod outcome_render;
mod patch;
mod plan_coordinator;
mod presentation;
mod profile_coordinator;
mod services;
pub mod session;
mod session_controller;
mod steering;
mod tool_execution;
mod tool_presentation;
mod tool_runner;
pub mod tools;
pub mod tui;
mod tui_state;
mod undo;

pub use config_editor::{
    ConfigChange, ConfigEditorSnapshot, ConfigFieldId, ConfigFieldKind, ConfigFieldView,
    ConfigSection,
};
pub use decision::EchoDecisionBackend;
pub use engine::{
    AgentEngine, AgentRuntimePolicy, ApprovalKind, ApprovalPrompt, CommandDecision, EngineObserver,
    PlanDecision, StreamingObserver, is_known_slash_command,
};
pub use error::{AgentError, AgentResult};
pub use guard::FileGuard;
pub use presentation::{ProfileChoice, SessionChoice};
pub use session::*;
pub use steering::TurnSteering;
pub use subbake_core::{CancellationGuard, CancellationToken};
pub use tools::{ALL_TOOL_SPECS, ToolKind};
pub use tui::{
    ConfigApplyAfter, Msg, MsgStyle, MsgView, StartupInfo, SubBakeTui, ToolActivity,
    ToolActivityStatus, ToolGroup, TranscriptItem, TuiAction, TuiInteraction, TuiObserver,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentActionKind {
    Start,
    Resume,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentAction {
    pub kind: AgentActionKind,
    pub session_id: Option<String>,
}
