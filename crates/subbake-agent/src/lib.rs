pub mod decision;
mod discovery;
pub mod engine;
pub mod event;
pub mod guard;
mod input_editor;
pub mod session;
mod tool_execution;
pub mod tools;
pub mod tui;
mod tui_state;

pub use decision::EchoDecisionBackend;
pub use engine::{
    AgentEngine, EngineObserver, PlanDecision, ProfileChoice, SessionChoice, StreamingObserver,
    is_known_slash_command,
};
pub use guard::FileGuard;
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
