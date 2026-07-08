#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    pub name: &'static str,
    pub category: &'static str,
    pub mutating: bool,
}

pub const TOOL_SPECS: &[ToolSpec] = &[
    ToolSpec {
        name: "translate_file",
        category: "translate",
        mutating: true,
    },
    ToolSpec {
        name: "translate_batch",
        category: "translate",
        mutating: true,
    },
    ToolSpec {
        name: "transcribe_media",
        category: "transcribe",
        mutating: true,
    },
    ToolSpec {
        name: "diagnose_path",
        category: "diagnose",
        mutating: false,
    },
    ToolSpec {
        name: "list_files",
        category: "browse",
        mutating: false,
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentRequest {
    pub action: AgentAction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentAction {
    Start,
    Resume { session_id: Option<String> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentOutcome {
    pub message: String,
}

pub fn run_agent(request: AgentRequest) -> AgentOutcome {
    let message = match request.action {
        AgentAction::Start => {
            "SubBake agent is scaffolded in Rust. Full interactive behavior is pending migration."
                .to_owned()
        }
        AgentAction::Resume {
            session_id: Some(session_id),
        } => format!("SubBake agent resume requested for session `{session_id}`."),
        AgentAction::Resume { session_id: None } => {
            "SubBake agent resume requested for latest session.".to_owned()
        }
    };

    AgentOutcome { message }
}

pub fn start_agent() -> String {
    run_agent(AgentRequest {
        action: AgentAction::Start,
    })
    .message
}

pub fn resume_agent(session_id: Option<&str>) -> String {
    run_agent(AgentRequest {
        action: AgentAction::Resume {
            session_id: session_id.map(str::to_owned),
        },
    })
    .message
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_agent_starts_scaffold() {
        let outcome = run_agent(AgentRequest {
            action: AgentAction::Start,
        });

        assert!(outcome.message.contains("scaffolded"));
    }

    #[test]
    fn run_agent_resumes_specific_session() {
        let outcome = run_agent(AgentRequest {
            action: AgentAction::Resume {
                session_id: Some("abc".to_owned()),
            },
        });

        assert!(outcome.message.contains("abc"));
    }
}
