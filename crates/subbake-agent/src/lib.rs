// SubBake agent — headless engine types.
// The full interactive agent (session loop, intent gating, plan/approval, undo)
// is built on top of these core abstractions.

pub mod session;
pub mod tools;

pub use session::*;

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
