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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolArgKind {
    String,
    Boolean,
}

impl ToolArgKind {
    fn name(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Boolean => "boolean",
        }
    }

    fn matches(self, value: &serde_json::Value) -> bool {
        match self {
            Self::String => value.is_string(),
            Self::Boolean => value.is_boolean(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolArgSpec {
    pub name: &'static str,
    pub kind: ToolArgKind,
    pub required: bool,
    pub description: &'static str,
}

impl ToolSpec {
    pub fn arguments(&self) -> Vec<ToolArgSpec> {
        tool_arguments(self.name)
    }

    pub fn prompt_line(&self) -> String {
        let arguments = self
            .arguments()
            .iter()
            .map(|argument| {
                let requirement = if argument.required {
                    "required"
                } else {
                    "optional"
                };
                format!(
                    "{}: {} {requirement} ({})",
                    argument.name,
                    argument.kind.name(),
                    argument.description
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        if arguments.is_empty() {
            format!("- {}: {} Arguments: {{}}", self.name, self.description)
        } else {
            format!(
                "- {}: {} Arguments: {{{arguments}}}",
                self.name, self.description
            )
        }
    }
}

const fn arg(
    name: &'static str,
    kind: ToolArgKind,
    required: bool,
    description: &'static str,
) -> ToolArgSpec {
    ToolArgSpec {
        name,
        kind,
        required,
        description,
    }
}

pub fn tool_arguments(name: &str) -> Vec<ToolArgSpec> {
    use ToolArgKind::{Boolean, String};
    match name {
        "translate_file" => vec![
            arg("path", String, true, "subtitle file path"),
            arg(
                "bilingual",
                Boolean,
                false,
                "override bilingual output for this call",
            ),
        ],
        "translate_series" => vec![
            arg(
                "path",
                String,
                true,
                "directory path; use . for the current directory",
            ),
            arg("recursive", Boolean, false, "include nested directories"),
            arg("overwrite", Boolean, false, "replace existing outputs"),
            arg(
                "bilingual",
                Boolean,
                false,
                "override bilingual output for this call",
            ),
        ],
        "edit_subtitle" => vec![
            arg("path", String, true, "generated subtitle path"),
            arg("instruction", String, true, "requested edit"),
            arg(
                "allow_non_generated",
                Boolean,
                false,
                "allow editing a source file",
            ),
        ],
        "transcribe_audio" | "read_file" | "read_file_preview" | "delete_file"
        | "diagnose_path" => vec![arg("path", String, true, "project-local path")],
        "list_files" => vec![arg("path", String, false, "directory path; defaults to .")],
        "search_files" => vec![
            arg("path", String, false, "directory path; defaults to ."),
            arg("pattern", String, false, "filename search pattern"),
        ],
        "candidate_subtitles" => vec![
            arg("path", String, false, "directory path; defaults to ."),
            arg("query", String, false, "text used to rank candidates"),
        ],
        "create_file" | "append_file" => vec![
            arg("path", String, true, "project-local path"),
            arg("content", String, false, "file content"),
        ],
        "replace_in_file" => vec![
            arg("path", String, true, "project-local path"),
            arg("old", String, false, "text to replace"),
            arg("new", String, false, "replacement text"),
        ],
        "rename_path" => vec![
            arg("from", String, true, "existing path"),
            arg("to", String, true, "new path"),
        ],
        "diagnose_text" => vec![arg("text", String, true, "diagnostic text")],
        "switch_profile" => vec![arg("name", String, true, "profile name")],
        "manage_whisper" => vec![
            arg(
                "action",
                String,
                false,
                "status, install, update, uninstall, list-models, or download",
            ),
            arg(
                "keep_models",
                Boolean,
                false,
                "keep models when uninstalling",
            ),
            arg("model", String, false, "model name to download"),
        ],
        "recent_translations" | "list_profiles" => vec![],
        _ => vec![],
    }
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
    let Some(spec) = ALL_TOOL_SPECS.iter().find(|spec| spec.name == name) else {
        return Err(format!("unknown tool `{name}`"));
    };
    let object = arguments
        .as_object()
        .ok_or_else(|| format!("arguments for `{name}` must be a JSON object"))?;
    for key in object.keys() {
        if !spec.arguments().iter().any(|argument| argument.name == key) {
            return Err(format!("tool `{name}` does not accept argument `{key}`"));
        }
    }
    for argument in spec.arguments() {
        match object.get(argument.name) {
            None if argument.required => {
                return Err(format!(
                    "tool `{name}` requires {} argument `{}`",
                    argument.kind.name(),
                    argument.name
                ));
            }
            Some(value) if !argument.kind.matches(value) => {
                return Err(format!(
                    "argument `{}` for tool `{name}` must be {}",
                    argument.name,
                    argument.kind.name()
                ));
            }
            _ => {}
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
        assert!(
            validate_tool_call(
                "translate_series",
                &serde_json::json!({"path": ".", "bilingual": true})
            )
            .is_ok()
        );
        assert!(
            validate_tool_call(
                "translate_series",
                &serde_json::json!({"path": ".", "bilingual": "yes"})
            )
            .is_err()
        );
        assert!(
            validate_tool_call(
                "translate_file",
                &serde_json::json!({"path": "clip.srt", "unexpected": true})
            )
            .is_err()
        );
    }

    #[test]
    fn prompt_contract_describes_translation_arguments() {
        let spec = ALL_TOOL_SPECS
            .iter()
            .find(|spec| spec.name == "translate_series")
            .expect("translate series spec");
        let line = spec.prompt_line();
        assert!(line.contains("path: string required"));
        assert!(line.contains("bilingual: boolean optional"));
    }
}
