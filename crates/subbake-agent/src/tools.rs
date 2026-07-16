//! Central registry for every agent-callable operation.
//!
//! The registry is the single source of truth for schemas, execution,
//! mutation/approval policy, and whether a compatibility tool is shown to new
//! model turns.

use thiserror::Error;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    pub name: &'static str,
    pub category: ToolKind,
    pub mutating: bool,
    pub requires_approval: bool,
    pub discovery: bool,
    pub model_visible: bool,
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
    ApplyPatch,
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
        let properties = self
            .arguments()
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
        let required = self
            .arguments()
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

use ToolArgKind::{Boolean as BooleanArg, String as StringArg};

const TRANSLATE_FILE_ARGS: &[ToolArgSpec] = &[
    arg("path", StringArg, true, "subtitle file path"),
    arg(
        "source_language",
        StringArg,
        false,
        "source language name or BCP-47 tag for this call",
    ),
    arg(
        "target_language",
        StringArg,
        false,
        "target language name or BCP-47 tag for this call",
    ),
    arg(
        "bilingual",
        BooleanArg,
        false,
        "override bilingual output for this call",
    ),
    arg(
        "bilingual_order",
        StringArg,
        false,
        "source_first or target_first for this call",
    ),
    arg(
        "output_format",
        StringArg,
        false,
        "srt, vtt, or txt output format for this call",
    ),
    arg(
        "output_path",
        StringArg,
        false,
        "explicit project-local output path",
    ),
    arg(
        "overwrite",
        BooleanArg,
        false,
        "replace an existing output; defaults to false",
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
        "source_language",
        StringArg,
        false,
        "source language name or BCP-47 tag for this call",
    ),
    arg(
        "target_language",
        StringArg,
        false,
        "target language name or BCP-47 tag for this call",
    ),
    arg(
        "bilingual",
        BooleanArg,
        false,
        "override bilingual output for this call",
    ),
    arg(
        "bilingual_order",
        StringArg,
        false,
        "source_first or target_first for this call",
    ),
    arg(
        "output_format",
        StringArg,
        false,
        "srt, vtt, or txt output format for this call",
    ),
    arg(
        "output_dir",
        StringArg,
        false,
        "project-local output directory; recursive calls preserve relative directories",
    ),
];
const EDIT_SUBTITLE_ARGS: &[ToolArgSpec] = &[
    arg("path", StringArg, true, "generated subtitle path"),
    arg("instruction", StringArg, true, "requested edit"),
    arg(
        "target_language",
        StringArg,
        false,
        "target language name or BCP-47 tag for this edit",
    ),
    arg(
        "allow_non_generated",
        BooleanArg,
        false,
        "allow editing a source file",
    ),
];
const TRANSCRIBE_AUDIO_ARGS: &[ToolArgSpec] = &[
    arg("path", StringArg, true, "project-local media file path"),
    arg(
        "language",
        StringArg,
        false,
        "spoken language name or BCP-47 tag; Auto detects it",
    ),
    arg(
        "provider",
        StringArg,
        false,
        "whisper_api or whisper_cpp for this call",
    ),
    arg(
        "model",
        StringArg,
        false,
        "transcription model for this call",
    ),
    arg(
        "output_format",
        StringArg,
        false,
        "srt, vtt, or txt output format for this call",
    ),
    arg(
        "output_path",
        StringArg,
        false,
        "explicit project-local output path",
    ),
    arg(
        "overwrite",
        BooleanArg,
        false,
        "replace an existing output; defaults to false",
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
const APPLY_PATCH_ARGS: &[ToolArgSpec] = &[arg(
    "patch",
    StringArg,
    true,
    "Codex-style patch bounded by Begin Patch and End Patch markers",
)];
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

macro_rules! tool {
    ($name:literal, $category:ident, $mutating:literal, $approval:literal, $discovery:literal, $visible:literal, $description:literal, $arguments:expr, $executor:ident) => {
        ToolSpec {
            name: $name,
            category: ToolKind::$category,
            mutating: $mutating,
            requires_approval: $approval,
            discovery: $discovery,
            model_visible: $visible,
            description: $description,
            arguments: $arguments,
            executor: ToolExecutor::$executor,
        }
    };
}

pub const ALL_TOOL_SPECS: &[ToolSpec] = &[
    tool!(
        "translate_file",
        Translate,
        true,
        false,
        false,
        true,
        "Translate one subtitle file.",
        TRANSLATE_FILE_ARGS,
        TranslateFile
    ),
    tool!(
        "translate_series",
        Translate,
        true,
        false,
        false,
        true,
        "Translate all source subtitle files in a directory.",
        TRANSLATE_SERIES_ARGS,
        TranslateSeries
    ),
    tool!(
        "edit_subtitle",
        Edit,
        true,
        false,
        false,
        true,
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
        true,
        "Transcribe a media file to subtitles.",
        TRANSCRIBE_AUDIO_ARGS,
        TranscribeAudio
    ),
    tool!(
        "manage_whisper",
        ManageWhisper,
        true,
        true,
        false,
        true,
        "Install, update, uninstall, inspect, or download whisper.cpp assets.",
        MANAGE_WHISPER_ARGS,
        ManageWhisper
    ),
    tool!(
        "diagnose_path",
        Diagnose,
        false,
        false,
        true,
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
        true,
        "Read the full content of a project-local file.",
        PATH_ARGS,
        ReadFile
    ),
    tool!(
        "apply_patch",
        FileOp,
        true,
        false,
        false,
        true,
        "Atomically add, update, or delete project-local text files with one patch.",
        APPLY_PATCH_ARGS,
        ApplyPatch
    ),
    // Hidden compatibility tools remain executable for v1 pending plans and
    // resumed sessions, but new model turns use apply_patch.
    tool!(
        "create_file",
        FileOp,
        true,
        false,
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
        true,
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
        true,
        "Delete a project-local file or directory.",
        PATH_ARGS,
        DeleteFile
    ),
    tool!(
        "switch_profile",
        Profile,
        false,
        false,
        false,
        true,
        "Switch the active provider profile after validating it.",
        SWITCH_PROFILE_ARGS,
        SwitchProfile
    ),
    tool!(
        "list_profiles",
        Profile,
        false,
        false,
        false,
        true,
        "List all available profiles.",
        &[],
        ListProfiles
    ),
];

pub fn find_tool_spec(name: &str) -> Option<&'static ToolSpec> {
    ALL_TOOL_SPECS.iter().find(|spec| spec.name == name)
}

pub(crate) fn model_visible_tool_specs() -> Vec<&'static ToolSpec> {
    ALL_TOOL_SPECS
        .iter()
        .filter(|spec| spec.model_visible)
        .collect()
}

pub(crate) fn model_visible_tool_names() -> Vec<&'static str> {
    model_visible_tool_specs()
        .into_iter()
        .map(|spec| spec.name)
        .collect()
}

pub fn tool_specs_for_categories<'a>(
    specs: &'a [ToolSpec],
    categories: &[ToolKind],
) -> Vec<&'a ToolSpec> {
    let mut result = specs
        .iter()
        .filter(|spec| categories.contains(&spec.category))
        .collect::<Vec<_>>();
    result.sort_by_key(|spec| spec.name);
    result
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ToolValidationError {
    #[error("unknown tool `{name}`")]
    UnknownTool { name: String },
    #[error("arguments for `{name}` must be a JSON object")]
    ArgumentsNotObject { name: String },
    #[error("tool `{name}` does not accept argument `{argument}`")]
    UnexpectedArgument { name: String, argument: String },
    #[error("tool `{name}` requires {expected} argument `{argument}`")]
    MissingArgument {
        name: String,
        argument: String,
        expected: &'static str,
    },
    #[error("argument `{argument}` for tool `{name}` must be {expected}")]
    WrongArgumentType {
        name: String,
        argument: String,
        expected: &'static str,
    },
}

pub fn validate_tool_call(
    name: &str,
    arguments: &serde_json::Value,
) -> Result<(), ToolValidationError> {
    let Some(spec) = find_tool_spec(name) else {
        return Err(ToolValidationError::UnknownTool {
            name: name.to_owned(),
        });
    };
    let object = arguments
        .as_object()
        .ok_or_else(|| ToolValidationError::ArgumentsNotObject {
            name: name.to_owned(),
        })?;
    for key in object.keys() {
        if !spec.arguments().iter().any(|argument| argument.name == key) {
            return Err(ToolValidationError::UnexpectedArgument {
                name: name.to_owned(),
                argument: key.clone(),
            });
        }
    }
    for argument in spec.arguments() {
        match object.get(argument.name) {
            None if argument.required => {
                return Err(ToolValidationError::MissingArgument {
                    name: name.to_owned(),
                    argument: argument.name.to_owned(),
                    expected: argument.kind.name(),
                });
            }
            Some(value) if !argument.kind.matches(value) => {
                return Err(ToolValidationError::WrongArgumentType {
                    name: name.to_owned(),
                    argument: argument.name.to_owned(),
                    expected: argument.kind.name(),
                });
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
    fn model_sees_one_complete_stable_registry_without_legacy_writers() {
        let names = model_visible_tool_names();
        assert!(names.contains(&"translate_series"));
        assert!(names.contains(&"candidate_subtitles"));
        assert!(names.contains(&"apply_patch"));
        assert!(!names.contains(&"create_file"));
        assert!(!names.contains(&"append_file"));
        assert!(!names.contains(&"replace_in_file"));
    }

    #[test]
    fn compatibility_writers_remain_registered_and_validatable() {
        assert!(find_tool_spec("create_file").is_some());
        assert!(
            validate_tool_call(
                "create_file",
                &serde_json::json!({"path": "note.txt", "content": "hello"})
            )
            .is_ok()
        );
    }

    #[test]
    fn validation_rejects_unknown_incomplete_and_extra_arguments() {
        assert!(validate_tool_call("unknown", &serde_json::json!({})).is_err());
        assert!(validate_tool_call("translate_file", &serde_json::json!({})).is_err());
        assert!(
            validate_tool_call("translate_file", &serde_json::json!({"path": "clip.srt"})).is_ok()
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
    fn native_schema_uses_the_same_arguments_as_local_validation() {
        let definition = find_tool_spec("apply_patch")
            .expect("patch tool")
            .native_definition();
        assert_eq!(
            definition.input_schema["properties"]["patch"]["type"],
            "string"
        );
        assert_eq!(
            definition.input_schema["required"],
            serde_json::json!(["patch"])
        );
        assert_eq!(definition.input_schema["additionalProperties"], false);
    }

    #[test]
    fn semantic_execution_arguments_are_exposed_in_native_and_fallback_schemas() {
        let translate = find_tool_spec("translate_file").expect("translate_file");
        let translate_names = translate
            .arguments()
            .iter()
            .map(|argument| argument.name)
            .collect::<Vec<_>>();
        for expected in [
            "source_language",
            "target_language",
            "bilingual",
            "bilingual_order",
            "output_format",
            "output_path",
            "overwrite",
        ] {
            assert!(translate_names.contains(&expected));
        }

        let transcribe = find_tool_spec("transcribe_audio").expect("transcribe_audio");
        let transcribe_names = transcribe
            .arguments()
            .iter()
            .map(|argument| argument.name)
            .collect::<Vec<_>>();
        for expected in [
            "language",
            "provider",
            "model",
            "output_format",
            "output_path",
            "overwrite",
        ] {
            assert!(transcribe_names.contains(&expected));
        }
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
}
