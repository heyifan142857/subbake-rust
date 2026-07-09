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

/// Rank and filter tools for an agent decision prompt.
///
/// Discovery tools stay visible because they are the safe way to resolve
/// ambiguous paths. Domain tools are selected from keywords in the request.
pub fn ranked_tool_specs(input: &str) -> Vec<&'static ToolSpec> {
    let input = input.to_lowercase();
    let mut categories = Vec::new();
    for (kind, keywords) in [
        (
            ToolKind::Translate,
            &["translate", "subtitle", "bilingual", "翻译", "字幕"][..],
        ),
        (
            ToolKind::Transcribe,
            &[
                "transcribe",
                "audio",
                "video",
                "media",
                "转录",
                "音频",
                "视频",
            ][..],
        ),
        (ToolKind::Edit, &["edit", "rewrite", "修改", "编辑"][..]),
        (
            ToolKind::Diagnose,
            &[
                "diagnose", "failure", "error", "debug", "诊断", "失败", "错误",
            ][..],
        ),
        (
            ToolKind::FileOp,
            &[
                "create",
                "append",
                "replace",
                "rename",
                "delete",
                "创建",
                "追加",
                "替换",
                "重命名",
                "删除",
            ][..],
        ),
        (ToolKind::Profile, &["profile", "配置", "预设"][..]),
        (
            ToolKind::ManageWhisper,
            &["whisper", "model", "install", "模型", "安装"][..],
        ),
    ] {
        if keywords.iter().any(|keyword| input.contains(keyword)) {
            categories.push(kind);
        }
    }

    if categories.is_empty() {
        return ALL_TOOL_SPECS.iter().collect();
    }
    categories.push(ToolKind::Browse);
    let mut specs = ALL_TOOL_SPECS
        .iter()
        .filter(|spec| categories.contains(&spec.category))
        .collect::<Vec<_>>();
    specs.sort_by_key(|spec| {
        (
            if spec.category == ToolKind::Browse {
                1
            } else {
                0
            },
            spec.name,
        )
    });
    specs
}

pub fn validate_tool_call(name: &str, arguments: &serde_json::Value) -> Result<(), String> {
    if !ALL_TOOL_SPECS.iter().any(|spec| spec.name == name) {
        return Err(format!("unknown tool `{name}`"));
    }
    let object = arguments
        .as_object()
        .ok_or_else(|| format!("arguments for `{name}` must be a JSON object"))?;
    let required: &[&str] = match name {
        "translate_file" | "translate_series" | "edit_subtitle" | "transcribe_audio"
        | "read_file" | "read_file_preview" | "create_file" | "append_file" | "replace_in_file"
        | "delete_file" | "diagnose_path" => &["path"],
        "rename_path" => &["from", "to"],
        "diagnose_text" => &["text"],
        "switch_profile" => &["name"],
        _ => &[],
    };
    for key in required {
        if object.get(*key).and_then(|value| value.as_str()).is_none() {
            return Err(format!("tool `{name}` requires string argument `{key}`"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranking_keeps_translation_and_discovery_tools() {
        let names = ranked_tool_specs("翻译 @episode.srt")
            .into_iter()
            .map(|spec| spec.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"translate_file"));
        assert!(names.contains(&"candidate_subtitles"));
        assert!(!names.contains(&"manage_whisper"));
    }

    #[test]
    fn validation_rejects_unknown_and_incomplete_calls() {
        assert!(validate_tool_call("unknown", &serde_json::json!({})).is_err());
        assert!(validate_tool_call("translate_file", &serde_json::json!({})).is_err());
        assert!(
            validate_tool_call("translate_file", &serde_json::json!({"path": "clip.srt"})).is_ok()
        );
    }
}
