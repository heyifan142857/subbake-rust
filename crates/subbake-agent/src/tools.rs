// Tool trait + ToolKind registry — core abstraction for agent-callable
// operations. Mirrors Python `agent/tool_registry.py` (19 tools organised into
// categories, with mutating/discovery/approval distinctions).

/// High-level category used for filtering which tools the LLM can see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    Translate,
    Transcribe,
    Edit,
    Diagnose,
    Browse,
    FileOp,
    Profile,
    ManageWhisper,
}

/// Metadata about a registered tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    pub name: &'static str,
    pub category: ToolKind,
    pub mutating: bool,
    pub requires_approval: bool,
    pub description: &'static str,
}

/// The 19 tools from the Python agent, grouped by category.
pub const ALL_TOOL_SPECS: &[ToolSpec] = &[
    // -- translate --
    ToolSpec {
        name: "translate_file",
        category: ToolKind::Translate,
        mutating: true,
        requires_approval: false,
        description: "Translate a subtitle file.",
    },
    ToolSpec {
        name: "translate_series",
        category: ToolKind::Translate,
        mutating: true,
        requires_approval: false,
        description: "Translate a series of subtitle files in a directory.",
    },
    // -- edit --
    ToolSpec {
        name: "edit_subtitle",
        category: ToolKind::Edit,
        mutating: true,
        requires_approval: false,
        description: "Edit an already translated subtitle file.",
    },
    // -- transcribe --
    ToolSpec {
        name: "transcribe_audio",
        category: ToolKind::Transcribe,
        mutating: true,
        requires_approval: false,
        description: "Transcribe a media file to subtitles.",
    },
    // -- manage_whisper --
    ToolSpec {
        name: "manage_whisper",
        category: ToolKind::ManageWhisper,
        mutating: true,
        requires_approval: true,
        description: "Install, update, or uninstall whisper.cpp.",
    },
    // -- diagnose --
    ToolSpec {
        name: "diagnose_path",
        category: ToolKind::Diagnose,
        mutating: false,
        requires_approval: false,
        description: "Diagnose a translation failure from a run directory.",
    },
    ToolSpec {
        name: "diagnose_text",
        category: ToolKind::Diagnose,
        mutating: false,
        requires_approval: false,
        description: "Diagnose a translation failure from text input.",
    },
    // -- browse (non-mutating, always available) --
    ToolSpec {
        name: "list_files",
        category: ToolKind::Browse,
        mutating: false,
        requires_approval: false,
        description: "List files and directories.",
    },
    ToolSpec {
        name: "search_files",
        category: ToolKind::Browse,
        mutating: false,
        requires_approval: false,
        description: "Search files by name glob.",
    },
    ToolSpec {
        name: "recent_translations",
        category: ToolKind::Browse,
        mutating: false,
        requires_approval: false,
        description: "List recent translation outputs from the session.",
    },
    ToolSpec {
        name: "candidate_subtitles",
        category: ToolKind::Browse,
        mutating: false,
        requires_approval: false,
        description: "Find subtitle files that look relevant.",
    },
    ToolSpec {
        name: "read_file_preview",
        category: ToolKind::Browse,
        mutating: false,
        requires_approval: false,
        description: "Read a short preview of a file.",
    },
    // -- file_operation --
    ToolSpec {
        name: "read_file",
        category: ToolKind::FileOp,
        mutating: false,
        requires_approval: false,
        description: "Read the full content of a file.",
    },
    ToolSpec {
        name: "create_file",
        category: ToolKind::FileOp,
        mutating: true,
        requires_approval: false,
        description: "Create a new file.",
    },
    ToolSpec {
        name: "append_file",
        category: ToolKind::FileOp,
        mutating: true,
        requires_approval: false,
        description: "Append content to a file.",
    },
    ToolSpec {
        name: "replace_in_file",
        category: ToolKind::FileOp,
        mutating: true,
        requires_approval: false,
        description: "Replace text in a file.",
    },
    ToolSpec {
        name: "rename_path",
        category: ToolKind::FileOp,
        mutating: true,
        requires_approval: false,
        description: "Rename or move a file or directory.",
    },
    ToolSpec {
        name: "delete_file",
        category: ToolKind::FileOp,
        mutating: true,
        requires_approval: false,
        description: "Delete a file or directory.",
    },
    // -- profile --
    ToolSpec {
        name: "switch_profile",
        category: ToolKind::Profile,
        mutating: false,
        requires_approval: false,
        description: "Switch the active profile.",
    },
    ToolSpec {
        name: "list_profiles",
        category: ToolKind::Profile,
        mutating: false,
        requires_approval: false,
        description: "List all available profiles.",
    },
];

/// Non-mutating discovery tools that can run without approval even in plan mode.
pub const DISCOVERY_TOOL_NAMES: &[&str] = &[
    "list_files",
    "search_files",
    "recent_translations",
    "candidate_subtitles",
    "read_file_preview",
    "read_file",
    "diagnose_path",
    "diagnose_text",
];

/// Tools that always require confirmation (equivalent to APPROVAL_REQUIRED mode).
pub const APPROVAL_REQUIRED_TOOL_NAMES: &[&str] = &["manage_whisper"];

/// Filter tool specs by a list of category names.
pub fn tool_specs_for_categories<'a>(
    specs: &'a [ToolSpec],
    categories: &[ToolKind],
) -> Vec<&'a ToolSpec> {
    let mut result: Vec<&ToolSpec> = specs
        .iter()
        .filter(|spec| categories.contains(&spec.category))
        .collect();
    result.sort_by_key(|spec| spec.name);
    result
}
