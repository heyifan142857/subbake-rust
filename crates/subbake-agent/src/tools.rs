// Central registry for agent-callable operations. Each entry owns the tool's
// schema, policy, category, and executor identity.

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
    pub discovery: bool,
    pub description: &'static str,
    pub arguments: &'static [ToolArgSpec],
    pub(crate) executor: ToolExecutor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolExecutor {
    TranslateFile,
    TranslateSeries,
    EditSubtitle,
    TranscribeAudio,
    ManageWhisper,
    DiagnosePath,
    DiagnoseText,
    ListFiles,
    SearchFiles,
    RecentTranslations,
    CandidateSubtitles,
    ReadFilePreview,
    ReadFile,
    CreateFile,
    AppendFile,
    ReplaceInFile,
    RenamePath,
    DeleteFile,
    SwitchProfile,
    ListProfiles,
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
    pub fn arguments(&self) -> &'static [ToolArgSpec] {
        self.arguments
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

    pub fn native_definition(&self) -> subbake_core::ports::ToolDefinition {
        let arguments = self.arguments();
        let properties = arguments
            .iter()
            .map(|argument| {
                let kind = match argument.kind {
                    ToolArgKind::String => "string",
                    ToolArgKind::Boolean => "boolean",
                };
                (
                    argument.name.to_owned(),
                    serde_json::json!({
                        "type": kind,
                        "description": argument.description,
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let required = arguments
            .iter()
            .filter(|argument| argument.required)
            .map(|argument| argument.name)
            .collect::<Vec<_>>();
        subbake_core::ports::ToolDefinition {
            name: self.name.to_owned(),
            description: self.description.to_owned(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": false,
            }),
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

const TRANSLATE_FILE_ARGS: &[ToolArgSpec] = &[
    arg("path", StringArg, true, "subtitle file path"),
    arg(
        "bilingual",
        BooleanArg,
        false,
        "override bilingual output for this call",
    ),
];
const TRANSLATE_SERIES_ARGS: &[ToolArgSpec] = &[
    arg(
        "path",
        StringArg,
        true,
        "directory path; use . for the current directory",
    ),
    arg("recursive", BooleanArg, false, "include nested directories"),
    arg("overwrite", BooleanArg, false, "replace existing outputs"),
    arg(
        "bilingual",
        BooleanArg,
        false,
        "override bilingual output for this call",
    ),
];
const EDIT_SUBTITLE_ARGS: &[ToolArgSpec] = &[
    arg("path", StringArg, true, "generated subtitle path"),
    arg("instruction", StringArg, true, "requested edit"),
    arg(
        "allow_non_generated",
        BooleanArg,
        false,
        "allow editing a source file",
    ),
];
const PATH_ARGS: &[ToolArgSpec] = &[arg("path", StringArg, true, "project-local path")];
const LIST_FILES_ARGS: &[ToolArgSpec] = &[arg(
    "path",
    StringArg,
    false,
    "directory path; defaults to .",
)];
const SEARCH_FILES_ARGS: &[ToolArgSpec] = &[
    arg("path", StringArg, false, "directory path; defaults to ."),
    arg("pattern", StringArg, false, "filename search pattern"),
];
const CANDIDATE_SUBTITLES_ARGS: &[ToolArgSpec] = &[
    arg("path", StringArg, false, "directory path; defaults to ."),
    arg("query", StringArg, false, "text used to rank candidates"),
];
const WRITE_FILE_ARGS: &[ToolArgSpec] = &[
    arg("path", StringArg, true, "project-local path"),
    arg("content", StringArg, false, "file content"),
];
const REPLACE_IN_FILE_ARGS: &[ToolArgSpec] = &[
    arg("path", StringArg, true, "project-local path"),
    arg("old", StringArg, false, "text to replace"),
    arg("new", StringArg, false, "replacement text"),
];
const RENAME_PATH_ARGS: &[ToolArgSpec] = &[
    arg("from", StringArg, true, "existing path"),
    arg("to", StringArg, true, "new path"),
];
const DIAGNOSE_TEXT_ARGS: &[ToolArgSpec] = &[arg("text", StringArg, true, "diagnostic text")];
const SWITCH_PROFILE_ARGS: &[ToolArgSpec] = &[arg("name", StringArg, true, "profile name")];
const MANAGE_WHISPER_ARGS: &[ToolArgSpec] = &[
    arg(
        "action",
        StringArg,
        false,
        "status, install, update, uninstall, list-models, or download",
    ),
    arg(
        "keep_models",
        BooleanArg,
        false,
        "keep models when uninstalling",
    ),
    arg("model", StringArg, false, "model name to download"),
];

use ToolArgKind::{Boolean as BooleanArg, String as StringArg};

macro_rules! tool {
    ($name:literal, $category:ident, $mutating:literal, $approval:literal, $discovery:literal, $description:literal, $arguments:expr, $executor:ident) => {
        ToolSpec {
            name: $name,
            category: ToolKind::$category,
            mutating: $mutating,
            requires_approval: $approval,
            discovery: $discovery,
            description: $description,
            arguments: $arguments,
            executor: ToolExecutor::$executor,
        }
    };
}

/// All tools exposed by the Rust agent, grouped by category.
pub const ALL_TOOL_SPECS: &[ToolSpec] = &[
    tool!(
        "translate_file",
        Translate,
        true,
        false,
        false,
        "Translate a subtitle file.",
        TRANSLATE_FILE_ARGS,
        TranslateFile
    ),
    tool!(
        "translate_series",
        Translate,
        true,
        false,
        false,
        "Translate a series of subtitle files in a directory.",
        TRANSLATE_SERIES_ARGS,
        TranslateSeries
    ),
    tool!(
        "edit_subtitle",
        Edit,
        true,
        false,
        false,
        "Edit an already translated subtitle file.",
        EDIT_SUBTITLE_ARGS,
        EditSubtitle
    ),
    tool!(
        "transcribe_audio",
        Transcribe,
        true,
        false,
        false,
        "Transcribe a media file to subtitles.",
        PATH_ARGS,
        TranscribeAudio
    ),
    tool!(
        "manage_whisper",
        ManageWhisper,
        true,
        true,
        false,
        "Install, update, or uninstall whisper.cpp.",
        MANAGE_WHISPER_ARGS,
        ManageWhisper
    ),
    tool!(
        "diagnose_path",
        Diagnose,
        false,
        false,
        true,
        "Diagnose a translation failure from a run directory.",
        PATH_ARGS,
        DiagnosePath
    ),
    tool!(
        "diagnose_text",
        Diagnose,
        false,
        false,
        true,
        "Diagnose a translation failure from text input.",
        DIAGNOSE_TEXT_ARGS,
        DiagnoseText
    ),
    tool!(
        "list_files",
        Browse,
        false,
        false,
        true,
        "List files and directories.",
        LIST_FILES_ARGS,
        ListFiles
    ),
    tool!(
        "search_files",
        Browse,
        false,
        false,
        true,
        "Search files by name glob.",
        SEARCH_FILES_ARGS,
        SearchFiles
    ),
    tool!(
        "recent_translations",
        Browse,
        false,
        false,
        true,
        "List recent translation outputs from the session.",
        &[],
        RecentTranslations
    ),
    tool!(
        "candidate_subtitles",
        Browse,
        false,
        false,
        true,
        "Find subtitle files that look relevant.",
        CANDIDATE_SUBTITLES_ARGS,
        CandidateSubtitles
    ),
    tool!(
        "read_file_preview",
        Browse,
        false,
        false,
        true,
        "Read a short preview of a file.",
        PATH_ARGS,
        ReadFilePreview
    ),
    tool!(
        "read_file",
        FileOp,
        false,
        false,
        true,
        "Read the full content of a file.",
        PATH_ARGS,
        ReadFile
    ),
    tool!(
        "create_file",
        FileOp,
        true,
        false,
        false,
        "Create a new file.",
        WRITE_FILE_ARGS,
        CreateFile
    ),
    tool!(
        "append_file",
        FileOp,
        true,
        false,
        false,
        "Append content to a file.",
        WRITE_FILE_ARGS,
        AppendFile
    ),
    tool!(
        "replace_in_file",
        FileOp,
        true,
        false,
        false,
        "Replace text in a file.",
        REPLACE_IN_FILE_ARGS,
        ReplaceInFile
    ),
    tool!(
        "rename_path",
        FileOp,
        true,
        false,
        false,
        "Rename or move a file or directory.",
        RENAME_PATH_ARGS,
        RenamePath
    ),
    tool!(
        "delete_file",
        FileOp,
        true,
        false,
        false,
        "Delete a file or directory.",
        PATH_ARGS,
        DeleteFile
    ),
    tool!(
        "switch_profile",
        Profile,
        false,
        false,
        false,
        "Switch the active profile.",
        SWITCH_PROFILE_ARGS,
        SwitchProfile
    ),
    tool!(
        "list_profiles",
        Profile,
        false,
        false,
        false,
        "List all available profiles.",
        &[],
        ListProfiles
    ),
];

pub fn find_tool_spec(name: &str) -> Option<&'static ToolSpec> {
    ALL_TOOL_SPECS.iter().find(|spec| spec.name == name)
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolScope {
    Browse,
    Translate,
    Transcribe,
    Edit,
    Diagnose,
    FileCreate,
    FileAppend,
    FileReplace,
    FileRename,
    FileDelete,
    Profile,
    Whisper,
}

pub(crate) fn tool_specs_for_scope(scope: ToolScope) -> Vec<&'static ToolSpec> {
    let names: &[&str] = match scope {
        ToolScope::Browse => &[
            "list_files",
            "search_files",
            "read_file_preview",
            "read_file",
        ],
        ToolScope::Translate => &["candidate_subtitles", "translate_file", "translate_series"],
        ToolScope::Transcribe => &["search_files", "transcribe_audio"],
        ToolScope::Edit => &["recent_translations", "read_file_preview", "edit_subtitle"],
        ToolScope::Diagnose => &[
            "search_files",
            "read_file_preview",
            "diagnose_path",
            "diagnose_text",
        ],
        ToolScope::FileCreate => &["list_files", "create_file"],
        ToolScope::FileAppend => &["search_files", "read_file", "append_file"],
        ToolScope::FileReplace => &["search_files", "read_file", "replace_in_file"],
        ToolScope::FileRename => &["search_files", "rename_path"],
        ToolScope::FileDelete => &["search_files", "delete_file"],
        ToolScope::Profile => &["list_profiles", "switch_profile"],
        ToolScope::Whisper => &["manage_whisper"],
    };
    names
        .iter()
        .filter_map(|name| find_tool_spec(name))
        .collect()
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
    fn translation_scope_keeps_only_translation_and_discovery_tools() {
        let names = tool_specs_for_scope(ToolScope::Translate)
            .into_iter()
            .map(|spec| spec.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"translate_file"));
        assert!(names.contains(&"candidate_subtitles"));
        assert!(!names.contains(&"manage_whisper"));
    }

    #[test]
    fn file_mutation_scopes_expose_only_the_requested_mutation() {
        let names = tool_specs_for_scope(ToolScope::FileDelete)
            .into_iter()
            .map(|spec| spec.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"delete_file"));
        assert!(!names.contains(&"rename_path"));
        assert!(!names.contains(&"create_file"));
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

    #[test]
    fn native_schema_uses_the_same_arguments_as_local_validation() {
        let spec = ALL_TOOL_SPECS
            .iter()
            .find(|spec| spec.name == "translate_file")
            .expect("translation tool");
        let definition = spec.native_definition();
        assert_eq!(
            definition.input_schema["properties"]["path"]["type"],
            "string"
        );
        assert_eq!(
            definition.input_schema["properties"]["bilingual"]["type"],
            "boolean"
        );
        assert_eq!(
            definition.input_schema["required"],
            serde_json::json!(["path"])
        );
        assert_eq!(definition.input_schema["additionalProperties"], false);
    }

    #[test]
    fn registry_has_unique_names_and_executors() {
        for (index, spec) in ALL_TOOL_SPECS.iter().enumerate() {
            assert_eq!(find_tool_spec(spec.name), Some(spec));
            for other in &ALL_TOOL_SPECS[index + 1..] {
                assert_ne!(spec.name, other.name, "duplicate tool name");
                assert_ne!(spec.executor, other.executor, "duplicate tool executor");
            }
        }
    }

    #[test]
    fn registry_owns_discovery_and_approval_policy() {
        assert!(find_tool_spec("read_file").is_some_and(|spec| spec.discovery));
        assert!(find_tool_spec("manage_whisper").is_some_and(|spec| spec.requires_approval));
        assert!(find_tool_spec("translate_file").is_some_and(|spec| spec.mutating));
        assert!(find_tool_spec("missing").is_none());
    }
}
