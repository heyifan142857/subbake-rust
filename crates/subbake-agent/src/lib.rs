pub mod decision;
mod discovery;
pub mod engine;
pub mod event;
pub mod guard;
mod input_editor;
pub mod session;
pub mod tools;
pub mod tui;

pub use decision::EchoDecisionBackend;
pub use engine::{
    AgentEngine, EngineObserver, PlanDecision, ProfileChoice, SessionChoice, StreamingObserver,
};
pub use guard::FileGuard;
pub use session::*;
pub use subbake_core::{CancellationGuard, CancellationToken};
pub use tools::{ALL_TOOL_SPECS, APPROVAL_REQUIRED_TOOL_NAMES, DISCOVERY_TOOL_NAMES, ToolKind};
pub use tui::{
    Msg, MsgStyle, MsgView, RenderPolicy, StartupInfo, SubBakeTui, TuiAction, TuiInteraction,
    TuiObserver,
};

// ---------------------------------------------------------------------------
// Compatibility API — used by the CLI while the interactive engine is built.
// These will be replaced when the full agent loop lands in stage 5.
// ---------------------------------------------------------------------------

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentAction {
    pub kind: String,
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentRequest {
    pub action: AgentAction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentOutcome {
    pub message: String,
}

impl fmt::Display for AgentOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

pub fn run_agent(request: AgentRequest) -> AgentOutcome {
    let message = match request.action.kind.as_str() {
        "start" => format!(
            "SubBake agent session {} started.",
            request.action.session_id.as_deref().unwrap_or("(new)")
        ),
        "resume" => format!(
            "SubBake agent session resume requested for `{}`.",
            request.action.session_id.as_deref().unwrap_or("latest")
        ),
        _ => "SubBake agent command received.".to_owned(),
    };
    AgentOutcome { message }
}

pub fn start_agent() -> String {
    run_agent(AgentRequest {
        action: AgentAction {
            kind: "start".to_owned(),
            session_id: None,
        },
    })
    .message
}

pub fn resume_agent(session_id: Option<&str>) -> String {
    run_agent(AgentRequest {
        action: AgentAction {
            kind: "resume".to_owned(),
            session_id: session_id.map(str::to_owned),
        },
    })
    .message
}
