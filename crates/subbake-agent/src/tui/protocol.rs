use crate::engine::{ApprovalPrompt, ProfileChoice, SessionChoice};
use crate::{ConfigChange, ConfigEditorSnapshot};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartupInfo {
    pub version: String,
    pub provider: String,
    pub model: String,
    pub config: String,
    pub cache_enabled: bool,
    pub cwd: String,
}

impl Default for StartupInfo {
    fn default() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_owned(),
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
    Approval {
        prompt: ApprovalPrompt,
    },
    ProfilePicker {
        message: String,
        options: Vec<ProfileChoice>,
    },
    ConfigEditor {
        message: String,
        snapshot: ConfigEditorSnapshot,
        provider: String,
        model: String,
        cache_enabled: bool,
    },
    ConfigClosed {
        message: String,
        provider: String,
        model: String,
        cache_enabled: bool,
    },
    SessionChanged {
        input_history: Vec<String>,
        events: Vec<crate::session::AgentEvent>,
        plan_mode: bool,
        model: String,
        approval: Option<ApprovalPrompt>,
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
    ApproveApproval,
    RejectApproval,
    ReviseApproval(String),
    SelectProfile(String),
    CreateProfile(String),
    SelectConfigProfile(String),
    CreateConfigProfile(String),
    ApplyConfig {
        changes: Vec<ConfigChange>,
        after: ConfigApplyAfter,
    },
    SelectSession(String),
    TogglePlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigApplyAfter {
    Stay,
    Close,
    SwitchProfile(String),
}
