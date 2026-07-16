pub mod decision;
mod discovery;
pub mod engine;
pub mod error;
pub mod event;
pub mod guard;
mod input_editor;
mod plan_coordinator;
mod presentation;
mod profile_coordinator;
pub mod session;
mod session_controller;
mod tool_execution;
mod tool_runner;
pub mod tools;
pub mod tui;
mod tui_state;
mod undo;

pub use decision::EchoDecisionBackend;
pub use engine::{
    AgentEngine, EngineObserver, PlanDecision, StreamingObserver, is_known_slash_command,
};
pub use error::{AgentError, AgentResult};
pub use guard::FileGuard;
pub use presentation::{ProfileChoice, SessionChoice};
pub use session::*;
pub use subbake_core::{CancellationGuard, CancellationToken};
pub use tools::{ALL_TOOL_SPECS, ToolKind};
pub use tui::{
    Msg, MsgStyle, MsgView, StartupInfo, SubBakeTui, TuiAction, TuiInteraction, TuiObserver,
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
