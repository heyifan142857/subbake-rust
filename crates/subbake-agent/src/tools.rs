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

/// The user-facing action currently being routed. This is intentionally
/// separate from [`ToolKind`]: an intent authorizes one business action plus
/// the safe discovery tools needed to ground that action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolIntent {
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

/// Security domain for intent refinement. Only domains explicitly marked as
/// refinable may change intent without asking the user again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolDomain {
    Browse,
    SubtitleTransform,
    Transcribe,
    Diagnose,
    FileCreate,
    FileAppend,
    FileReplace,
    FileRename,
    FileDelete,
    Profile,
    Whisper,
}

impl ToolIntent {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "browse" => Self::Browse,
            "translate" => Self::Translate,
            "transcribe" => Self::Transcribe,
            "edit" => Self::Edit,
            "diagnose" => Self::Diagnose,
            "file_create" => Self::FileCreate,
            "file_append" => Self::FileAppend,
            "file_replace" => Self::FileReplace,
            "file_rename" => Self::FileRename,
            "file_delete" => Self::FileDelete,
            "profile" => Self::Profile,
            "whisper" => Self::Whisper,
            _ => return None,
        })
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Browse => "browse",
            Self::Translate => "translate",
            Self::Transcribe => "transcribe",
            Self::Edit => "edit",
            Self::Diagnose => "diagnose",
            Self::FileCreate => "file_create",
            Self::FileAppend => "file_append",
            Self::FileReplace => "file_replace",
            Self::FileRename => "file_rename",
            Self::FileDelete => "file_delete",
            Self::Profile => "profile",
            Self::Whisper => "whisper",
        }
    }

    const fn domain(self) -> ToolDomain {
        match self {
            Self::Browse => ToolDomain::Browse,
            Self::Translate | Self::Edit => ToolDomain::SubtitleTransform,
            Self::Transcribe => ToolDomain::Transcribe,
            Self::Diagnose => ToolDomain::Diagnose,
            Self::FileCreate => ToolDomain::FileCreate,
            Self::FileAppend => ToolDomain::FileAppend,
            Self::FileReplace => ToolDomain::FileReplace,
            Self::FileRename => ToolDomain::FileRename,
            Self::FileDelete => ToolDomain::FileDelete,
            Self::Profile => ToolDomain::Profile,
            Self::Whisper => ToolDomain::Whisper,
        }
    }

    pub(crate) fn can_transition_to(self, target: Self) -> bool {
        self != target
            && matches!(
                (self.domain(), target.domain()),
                (ToolDomain::SubtitleTransform, ToolDomain::SubtitleTransform)
            )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolAuthorization {
    Allowed,
    Transition(ToolIntent),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolPolicyError {
    tool: String,
    current: ToolIntent,
    required: Option<ToolIntent>,
}

impl std::fmt::Display for ToolPolicyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.required {
            Some(required) => write!(
                formatter,
                "tool `{}` requires intent `{}`, but this turn is using intent `{}`",
                self.tool,
                required.as_str(),
                self.current.as_str()
            ),
            None => write!(
                formatter,
                "unknown tool `{}` is not available for intent `{}`",
                self.tool,
                self.current.as_str()
            ),
        }
    }
}

/// Visibility policy declared by each registered tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolAccess {
    /// A non-mutating, bounded observation that can ground any action.
    SharedDiscovery,
    /// A tool available only to the listed routed actions.
    Intents(&'static [ToolIntent]),
}

/// Metadata about a registered tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    pub name: &'static str,
    pub category: ToolKind,
    pub mutating: bool,
    pub requires_approval: bool,
    pub discovery: bool,
    pub(crate) default_intent: ToolIntent,
    pub(crate) access: ToolAccess,
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
    pub(crate) fn allows(&self, intent: ToolIntent) -> bool {
        match self.access {
            ToolAccess::SharedDiscovery => true,
            ToolAccess::Intents(intents) => intents.contains(&intent),
        }
    }
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
    ($name:literal, $category:ident, $mutating:literal, $approval:literal, $discovery:literal, $intent:ident, $access:expr, $description:literal, $arguments:expr, $executor:ident) => {
        ToolSpec {
            name: $name,
            category: ToolKind::$category,
            mutating: $mutating,
            requires_approval: $approval,
            discovery: $discovery,
            default_intent: ToolIntent::$intent,
            access: $access,
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
        Translate,
        ToolAccess::Intents(&[ToolIntent::Translate]),
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
        Translate,
        ToolAccess::Intents(&[ToolIntent::Translate]),
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
        Edit,
        ToolAccess::Intents(&[ToolIntent::Edit]),
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
        Transcribe,
        ToolAccess::Intents(&[ToolIntent::Transcribe]),
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
        Whisper,
        ToolAccess::Intents(&[ToolIntent::Whisper]),
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
        Diagnose,
        ToolAccess::Intents(&[ToolIntent::Diagnose]),
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
        Diagnose,
        ToolAccess::Intents(&[ToolIntent::Diagnose]),
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
        Browse,
        ToolAccess::SharedDiscovery,
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
        Browse,
        ToolAccess::SharedDiscovery,
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
        Browse,
        ToolAccess::SharedDiscovery,
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
        Browse,
        ToolAccess::SharedDiscovery,
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
        Browse,
        ToolAccess::SharedDiscovery,
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
        FileAppend,
        ToolAccess::Intents(&[ToolIntent::FileAppend, ToolIntent::FileReplace]),
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
        FileCreate,
        ToolAccess::Intents(&[ToolIntent::FileCreate]),
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
        FileAppend,
        ToolAccess::Intents(&[ToolIntent::FileAppend]),
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
        FileReplace,
        ToolAccess::Intents(&[ToolIntent::FileReplace]),
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
        FileRename,
        ToolAccess::Intents(&[ToolIntent::FileRename]),
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
        FileDelete,
        ToolAccess::Intents(&[ToolIntent::FileDelete]),
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
        Profile,
        ToolAccess::Intents(&[ToolIntent::Profile]),
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
        Profile,
        ToolAccess::Intents(&[ToolIntent::Profile]),
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

pub(crate) fn tool_specs_for_intent(intent: ToolIntent) -> Vec<&'static ToolSpec> {
    ALL_TOOL_SPECS
        .iter()
        .filter(|spec| {
            spec.allows(intent)
                || (intent.can_transition_to(spec.default_intent)
                    && spec.allows(spec.default_intent))
        })
        .collect()
}

pub(crate) fn authorize_tool(
    intent: ToolIntent,
    name: &str,
) -> Result<ToolAuthorization, ToolPolicyError> {
    let Some(spec) = find_tool_spec(name) else {
        return Err(ToolPolicyError {
            tool: name.to_owned(),
            current: intent,
            required: None,
        });
    };
    if spec.allows(intent) {
        return Ok(ToolAuthorization::Allowed);
    }
    if intent.can_transition_to(spec.default_intent) && spec.allows(spec.default_intent) {
        return Ok(ToolAuthorization::Transition(spec.default_intent));
    }
    Err(ToolPolicyError {
        tool: name.to_owned(),
        current: intent,
        required: Some(spec.default_intent),
    })
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
    fn translation_intent_includes_shared_discovery_and_subtitle_refinements() {
        let names = tool_specs_for_intent(ToolIntent::Translate)
            .into_iter()
            .map(|spec| spec.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"translate_file"));
        assert!(names.contains(&"candidate_subtitles"));
        assert!(names.contains(&"read_file_preview"));
        assert!(names.contains(&"edit_subtitle"));
        assert!(!names.contains(&"read_file"));
        assert!(!names.contains(&"manage_whisper"));
    }

    #[test]
    fn subtitle_intents_can_transition_without_crossing_domains() {
        assert_eq!(
            authorize_tool(ToolIntent::Translate, "edit_subtitle"),
            Ok(ToolAuthorization::Transition(ToolIntent::Edit))
        );
        assert_eq!(
            authorize_tool(ToolIntent::Edit, "translate_file"),
            Ok(ToolAuthorization::Transition(ToolIntent::Translate))
        );
        let error = authorize_tool(ToolIntent::Translate, "delete_file")
            .expect_err("delete must remain outside subtitle transforms");
        assert_eq!(
            error.to_string(),
            "tool `delete_file` requires intent `file_delete`, but this turn is using intent `translate`"
        );
    }

    #[test]
    fn file_mutation_intents_expose_only_the_requested_mutation() {
        let names = tool_specs_for_intent(ToolIntent::FileDelete)
            .into_iter()
            .map(|spec| spec.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"delete_file"));
        assert!(!names.contains(&"rename_path"));
        assert!(!names.contains(&"create_file"));
    }

    #[test]
    fn shared_discovery_tools_are_bounded_and_non_mutating() {
        for spec in ALL_TOOL_SPECS {
            if spec.access == ToolAccess::SharedDiscovery {
                assert!(spec.discovery, "{} must be a discovery tool", spec.name);
                assert!(!spec.mutating, "{} must not mutate", spec.name);
            }
        }
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
