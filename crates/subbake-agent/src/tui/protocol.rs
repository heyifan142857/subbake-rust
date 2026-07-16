use crate::engine::{ProfileChoice, SessionChoice};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartupInfo {
    pub provider: String,
    pub model: String,
    pub config: String,
    pub cache_enabled: bool,
    pub cwd: String,
}

impl Default for StartupInfo {
    fn default() -> Self {
        Self {
            provider: "mock".to_owned(),
            model: "mock-zh".to_owned(),
            config: "Not configured".to_owned(),
            cache_enabled: true,
            cwd: String::new(),
        }
    }
}

pub enum TuiInteraction {
    Message {
        message: String,
    },
    PlanApproval {
        message: String,
    },
    ProfilePicker {
        message: String,
        options: Vec<ProfileChoice>,
    },
    SessionChanged {
        input_history: Vec<String>,
        events: Vec<crate::session::AgentEvent>,
        plan_mode: bool,
        model: String,
    },
    SessionPicker {
        message: String,
        options: Vec<SessionChoice>,
    },
    PlanModeChanged {
        enabled: bool,
    },
    ModelChanged {
        model: String,
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuiAction {
    SubmitText(String),
    ApprovePlan,
    RejectPlan,
    SelectProfile(String),
    CreateProfile(String),
    SelectSession(String),
    TogglePlan,
}
