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

pub fn start_agent() -> String {
    "SubBake agent is scaffolded in Rust. Full interactive behavior is pending migration.".to_owned()
}

pub fn resume_agent(session_id: Option<&str>) -> String {
    match session_id {
        Some(session_id) => format!("SubBake agent resume requested for session `{session_id}`."),
        None => "SubBake agent resume requested for latest session.".to_owned(),
    }
}
