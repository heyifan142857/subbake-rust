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
    Command,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCapability {
    CommandSandbox,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    pub name: &'static str,
    pub category: ToolKind,
    pub mutating: bool,
    pub requires_approval: bool,
    pub discovery: bool,
    pub model_visible: bool,
    pub required_capability: Option<ToolCapability>,
    pub description: &'static str,
    pub arguments: &'static [ToolArgSpec],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolHandler {
    pub spec: ToolSpec,
    executor: ToolExecutor,
}

impl std::ops::Deref for ToolHandler {
    type Target = ToolSpec;

    fn deref(&self) -> &Self::Target {
        &self.spec
    }
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
    RenamePath,
    DeleteFile,
    DeleteExternalPath,
    SwitchProfile,
    ListProfiles,
    RunCommand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolArgKind {
    String,
    Boolean,
    Integer,
    StringMap,
}

impl ToolArgKind {
    fn name(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Boolean => "boolean",
            Self::Integer => "integer",
            Self::StringMap => "object<string,string>",
        }
    }

    fn matches(self, value: &serde_json::Value) -> bool {
        match self {
            Self::String => value.is_string(),
            Self::Boolean => value.is_boolean(),
            Self::Integer => value.is_u64(),
            Self::StringMap => value
                .as_object()
                .is_some_and(|map| map.values().all(serde_json::Value::is_string)),
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
}

impl ToolHandler {
    pub(crate) const fn executor(&self) -> ToolExecutor {
        self.executor
    }

    pub(crate) fn mutates_with(&self, arguments: &serde_json::Value) -> bool {
        if self.executor == ToolExecutor::RunCommand {
            return arguments
                .get("outputs")
                .and_then(serde_json::Value::as_object)
                .is_some_and(|outputs| !outputs.is_empty());
        }
        if self.executor != ToolExecutor::ManageWhisper {
            return self.spec.mutating;
        }
        !matches!(
            arguments
                .get("action")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("status"),
            "status"
                | "list-models"
                | "models"
                | "list-vad-models"
                | "vad-models"
                | "list-versions"
                | "versions"
        )
    }

    pub(crate) fn requires_approval_with(&self, arguments: &serde_json::Value) -> bool {
        if self.executor == ToolExecutor::RunCommand {
            return matches!(
                crate::command_policy::classify(arguments),
                crate::command_policy::CommandApproval::AskUser(_)
            );
        }
        self.spec.requires_approval && self.mutates_with(arguments)
    }
}

impl ToolSpec {
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
                let mut schema = match argument.kind {
                    ToolArgKind::String => serde_json::json!({"type":"string"}),
                    ToolArgKind::Boolean => serde_json::json!({"type":"boolean"}),
                    ToolArgKind::Integer => serde_json::json!({"type":"integer"}),
                    ToolArgKind::StringMap => serde_json::json!({
                        "type":"object",
                        "additionalProperties":{"type":"string"}
                    }),
                };
                schema["description"] = serde_json::Value::String(argument.description.to_owned());
                (argument.name.to_owned(), schema)
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

use ToolArgKind::{
    Boolean as BooleanArg, Integer as IntegerArg, String as StringArg, StringMap as StringMapArg,
};

const TRANSLATE_FILE_ARGS: &[ToolArgSpec] = &[
    arg(
        "path",
        StringArg,
        true,
        "subtitle file or MKV, MP4/M4V/MOV, or WebM container with embedded text subtitles",
    ),
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
        "subtitle_stream",
        IntegerArg,
        false,
        "explicit embedded subtitle stream index",
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
        "preserve_names",
        BooleanArg,
        false,
        "keep personal names in source spelling; false transliterates them",
    ),
    arg(
        "online_terminology",
        BooleanArg,
        false,
        "merge entity-aware terminology while translating",
    ),
    arg(
        "preserve_source_container",
        BooleanArg,
        false,
        "write a separate translated media container instead of replacing the source",
    ),
    arg(
        "output_format",
        StringArg,
        false,
        "srt, vtt, txt, ass, ssa, ttml, or dfxp output format for this call",
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
    arg(
        "fresh_runtime",
        BooleanArg,
        false,
        "use a unique empty runtime for this call, disabling Resume, request cache, translation memory, and accumulated glossary reuse",
    ),
    arg(
        "max_requests",
        IntegerArg,
        false,
        "maximum provider requests for this call",
    ),
    arg(
        "max_tokens",
        IntegerArg,
        false,
        "stop before the next provider request after this many used tokens",
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
        "fresh_runtime",
        BooleanArg,
        false,
        "use a unique empty runtime for this call, disabling Resume, request cache, translation memory, and accumulated glossary reuse",
    ),
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
        "preserve_names",
        BooleanArg,
        false,
        "keep personal names in source spelling; false transliterates them",
    ),
    arg(
        "online_terminology",
        BooleanArg,
        false,
        "merge entity-aware terminology while translating",
    ),
    arg(
        "output_format",
        StringArg,
        false,
        "srt, vtt, txt, ass, ssa, ttml, or dfxp output format for this call",
    ),
    arg(
        "output_dir",
        StringArg,
        false,
        "project-local output directory; recursive calls preserve relative directories",
    ),
    arg(
        "max_requests",
        IntegerArg,
        false,
        "maximum provider requests for this call",
    ),
    arg(
        "max_tokens",
        IntegerArg,
        false,
        "stop before the next provider request after this many used tokens",
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
    arg(
        "dry_run",
        BooleanArg,
        false,
        "validate and return the proposed changes without writing the subtitle",
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
const RENAME_PATH_ARGS: &[ToolArgSpec] = &[
    arg("from", StringArg, true, "existing path"),
    arg("to", StringArg, true, "new path"),
];
const DELETE_EXTERNAL_PATH_ARGS: &[ToolArgSpec] = &[
    arg(
        "path",
        StringArg,
        true,
        "absolute path outside the active project; the runtime resolves it before approval",
    ),
    arg(
        "recursive",
        BooleanArg,
        true,
        "explicitly declare whether a non-empty directory may be deleted recursively",
    ),
];
const DIAGNOSE_TEXT_ARGS: &[ToolArgSpec] = &[arg("text", StringArg, true, "diagnostic text")];
const SWITCH_PROFILE_ARGS: &[ToolArgSpec] = &[arg("name", StringArg, true, "profile name")];
const MANAGE_WHISPER_ARGS: &[ToolArgSpec] = &[
    arg(
        "action",
        StringArg,
        false,
        "status, list-versions, install, update, uninstall, list-models, list-vad-models, download, or download-vad",
    ),
    arg(
        "keep_models",
        BooleanArg,
        false,
        "keep models when uninstalling",
    ),
    arg("model", StringArg, false, "model name to download"),
    arg(
        "variant",
        StringArg,
        false,
        "install variant: cpu, cuda, metal, vulkan, or openblas",
    ),
];
const RUN_COMMAND_ARGS: &[ToolArgSpec] = &[
    arg(
        "command",
        StringArg,
        true,
        "bash command to execute in the Linux sandbox",
    ),
    arg(
        "cwd",
        StringArg,
        false,
        "project-local working directory; defaults to .",
    ),
    arg(
        "outputs",
        StringMapArg,
        false,
        "output alias to final project-local file path; write each artifact to $SUBBAKE_OUTPUT_<ALIAS>",
    ),
    arg(
        "overwrite",
        BooleanArg,
        false,
        "replace existing declared outputs",
    ),
    arg(
        "network",
        BooleanArg,
        false,
        "request network access; defaults to false",
    ),
    arg(
        "timeout_seconds",
        IntegerArg,
        false,
        "command timeout from 1 to 1800 seconds; defaults to 120",
    ),
];

macro_rules! tool {
    ($name:literal, $category:ident, $mutating:literal, $approval:literal, $discovery:literal, $visible:literal, $description:literal, $arguments:expr, $executor:ident) => {
        ToolHandler {
            spec: ToolSpec {
                name: $name,
                category: ToolKind::$category,
                mutating: $mutating,
                requires_approval: $approval,
                discovery: $discovery,
                model_visible: $visible,
                required_capability: None,
                description: $description,
                arguments: $arguments,
            },
            executor: ToolExecutor::$executor,
        }
    };
    ($name:literal, $category:ident, $mutating:literal, $approval:literal, $discovery:literal, $visible:literal, $description:literal, $arguments:expr, $executor:ident, requires $capability:ident) => {
        ToolHandler {
            spec: ToolSpec {
                name: $name,
                category: ToolKind::$category,
                mutating: $mutating,
                requires_approval: $approval,
                discovery: $discovery,
                model_visible: $visible,
                required_capability: Some(ToolCapability::$capability),
                description: $description,
                arguments: $arguments,
            },
            executor: ToolExecutor::$executor,
        }
    };
}

pub(crate) const ALL_TOOL_HANDLERS: &[ToolHandler] = &[
    tool!(
        "run_command",
        Command,
        false,
        true,
        false,
        true,
        "Run a sandboxed Linux command. The project is read-only; persistent regular-file artifacts must be written to declared $SUBBAKE_OUTPUT_<ALIAS> paths. Use apply_patch for source edits.",
        RUN_COMMAND_ARGS,
        RunCommand,
        requires CommandSandbox
    ),
    tool!(
        "translate_file",
        Translate,
        true,
        false,
        false,
        true,
        "Translate one subtitle file, or translate and append a matching text subtitle stream in an MKV, MP4/M4V/MOV, or WebM container without transcribing it.",
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
        "Transcribe a media file to subtitles. When the model is omitted, use the configured model or deterministically select an installed model; explicitly tell the user when the result says model_auto_selected=true.",
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
        "Manage local whisper.cpp. status, list-models, list-vad-models, and list-versions are read-only checks and should be followed immediately by the next task action. VAD defaults to Silero; use download-vad without a model to install the default. For an install request: install the CLI first, install the default VAD model, then call list-models and present the transcription models to the user; do not choose or download a transcription model until the user selects one. Use list-versions to fetch upstream releases.",
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
        "delete_external_path",
        FileOp,
        true,
        true,
        false,
        true,
        "Permanently delete one path outside the active project. Every call requires explicit user approval, never follows a leaf symlink, rejects filesystem/HOME/project roots, and cannot be undone by /undo.",
        DELETE_EXTERNAL_PATH_ARGS,
        DeleteExternalPath
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

/// Public schema view derived from the executable handler registry.
pub static ALL_TOOL_SPECS: std::sync::LazyLock<Vec<ToolSpec>> = std::sync::LazyLock::new(|| {
    ALL_TOOL_HANDLERS
        .iter()
        .map(|handler| handler.spec.clone())
        .collect()
});

pub fn find_tool_spec(name: &str) -> Option<&'static ToolSpec> {
    find_tool_handler(name).map(|handler| &handler.spec)
}

pub(crate) fn find_tool_handler(name: &str) -> Option<&'static ToolHandler> {
    ALL_TOOL_HANDLERS
        .iter()
        .find(|handler| handler.spec.name == name)
}

fn tool_is_available_for(
    spec: &ToolSpec,
    capabilities: subbake_adapters::platform::CapabilitySet,
) -> bool {
    match spec.required_capability {
        None => true,
        Some(ToolCapability::CommandSandbox) => capabilities.supports_command_sandbox(),
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ToolRegistry {
    capabilities: subbake_adapters::platform::CapabilitySet,
}

impl ToolRegistry {
    pub(crate) const fn new(capabilities: subbake_adapters::platform::CapabilitySet) -> Self {
        Self { capabilities }
    }

    pub(crate) fn is_available(self, spec: &ToolSpec) -> bool {
        tool_is_available_for(spec, self.capabilities)
    }

    pub(crate) fn model_visible_specs(self) -> Vec<&'static ToolSpec> {
        ALL_TOOL_HANDLERS
            .iter()
            .map(|handler| &handler.spec)
            .filter(|spec| spec.model_visible && self.is_available(spec))
            .collect()
    }

    pub(crate) fn model_visible_names(self) -> Vec<&'static str> {
        self.model_visible_specs()
            .into_iter()
            .map(|spec| spec.name)
            .collect()
    }

    pub(crate) fn specs_for_categories(self, categories: &[ToolKind]) -> Vec<&'static ToolSpec> {
        let mut result = ALL_TOOL_HANDLERS
            .iter()
            .map(|handler| &handler.spec)
            .filter(|spec| categories.contains(&spec.category) && self.is_available(spec))
            .collect::<Vec<_>>();
        result.sort_by_key(|spec| spec.name);
        result
    }
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
    #[error("invalid argument `{argument}` for tool `{name}`: {message}")]
    InvalidArgument {
        name: String,
        argument: String,
        message: String,
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
    if name == "run_command" {
        if object
            .get("command")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|command| command.trim().is_empty())
        {
            return Err(ToolValidationError::InvalidArgument {
                name: name.to_owned(),
                argument: "command".to_owned(),
                message: "command must not be empty".to_owned(),
            });
        }
        if let Some(value) = object.get("timeout_seconds") {
            let timeout = value.as_u64().unwrap_or_default();
            if !(1..=1800).contains(&timeout) {
                return Err(ToolValidationError::InvalidArgument {
                    name: name.to_owned(),
                    argument: "timeout_seconds".to_owned(),
                    message: "expected an integer from 1 through 1800".to_owned(),
                });
            }
        }
        if let Some(outputs) = object.get("outputs").and_then(|value| value.as_object()) {
            let mut environment_names = std::collections::HashSet::new();
            for alias in outputs.keys() {
                let valid = !alias.is_empty()
                    && alias.len() <= 32
                    && alias
                        .chars()
                        .next()
                        .is_some_and(|value| value.is_ascii_alphabetic())
                    && alias
                        .chars()
                        .all(|value| value.is_ascii_alphanumeric() || value == '_');
                if !valid {
                    return Err(ToolValidationError::InvalidArgument {
                        name: name.to_owned(),
                        argument: "outputs".to_owned(),
                        message: format!(
                            "output alias `{alias}` must be an ASCII identifier up to 32 characters"
                        ),
                    });
                }
                if !environment_names.insert(alias.to_ascii_uppercase()) {
                    return Err(ToolValidationError::InvalidArgument {
                        name: name.to_owned(),
                        argument: "outputs".to_owned(),
                        message: format!(
                            "output alias `{alias}` collides case-insensitively with another alias"
                        ),
                    });
                }
            }
        }
    }
    Ok(())
}

/// A tool call that has crossed the registry boundary exactly once. Executors
/// still use JSON internally, but they can no longer receive an unknown tool,
/// extra field, missing required field, or a value of the wrong primitive type.
pub(crate) struct ValidatedToolCall<'a> {
    handler: &'static ToolHandler,
    arguments: &'a serde_json::Value,
}

impl<'a> ValidatedToolCall<'a> {
    pub(crate) fn parse(
        name: &str,
        arguments: &'a serde_json::Value,
    ) -> Result<Self, ToolValidationError> {
        validate_tool_call(name, arguments)?;
        let handler = find_tool_handler(name).ok_or_else(|| ToolValidationError::UnknownTool {
            name: name.to_owned(),
        })?;
        Ok(Self { handler, arguments })
    }

    pub(crate) fn executor(&self) -> ToolExecutor {
        self.handler.executor()
    }

    pub(crate) fn arguments(&self) -> &'a serde_json::Value {
        self.arguments
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn model_sees_one_complete_stable_registry() {
        let capabilities = subbake_adapters::platform::CapabilitySet::current();
        let names = ToolRegistry::new(capabilities).model_visible_names();
        assert!(names.contains(&"translate_series"));
        assert!(names.contains(&"candidate_subtitles"));
        assert!(names.contains(&"apply_patch"));
        assert_eq!(
            names.contains(&"run_command"),
            capabilities.supports_command_sandbox()
        );
        assert!(names.contains(&"delete_external_path"));
    }

    #[test]
    fn command_tool_is_advertised_only_on_linux() {
        let command = find_tool_spec("run_command").expect("command tool");
        let translate = find_tool_spec("translate_file").expect("translation tool");
        let linux = subbake_adapters::platform::CapabilitySet::from_target("linux", "x86_64");
        let windows = subbake_adapters::platform::CapabilitySet::from_target("windows", "x86_64");
        let mac = subbake_adapters::platform::CapabilitySet::from_target("macos", "aarch64");

        assert!(tool_is_available_for(command, linux));
        assert!(!tool_is_available_for(command, windows));
        assert!(!tool_is_available_for(command, mac));
        assert!(tool_is_available_for(translate, windows));
        assert!(tool_is_available_for(translate, mac));
    }

    proptest! {
        #[test]
        fn arbitrary_unknown_arguments_are_rejected(argument in "zz_[a-z0-9_]{1,24}") {
            let error = validate_tool_call(
                "list_files",
                &serde_json::json!({argument: "value"}),
            )
            .expect_err("generated argument is not in the list_files schema");
            let is_unexpected_argument = matches!(
                error,
                ToolValidationError::UnexpectedArgument { .. }
            );
            prop_assert!(is_unexpected_argument);
        }

        #[test]
        fn valid_output_aliases_survive_schema_validation(
            first in "[A-Za-z]",
            suffix in "[A-Za-z0-9_]{0,20}",
        ) {
            let alias = format!("{first}{suffix}");
            let arguments = serde_json::json!({
                "command": "printf artifact",
                "outputs": {alias: "artifact.txt"}
            });
            prop_assert!(validate_tool_call("run_command", &arguments).is_ok());
        }

        #[test]
        fn path_traversal_arguments_do_not_bypass_the_file_guard(name in "[A-Za-z0-9_-]{1,32}") {
            let root = std::env::temp_dir().join(format!("subbake-proptest-{}", std::process::id()));
            std::fs::create_dir_all(&root).expect("create property-test root");
            let guard = crate::guard::FileGuard::new(root).expect("create file guard");
            let traversal = format!("../{name}");
            prop_assert!(guard.resolve_path(std::path::Path::new(&traversal)).is_err());
        }
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
    fn command_schema_validates_timeout_and_output_aliases() {
        assert!(
            validate_tool_call(
                "run_command",
                &serde_json::json!({
                    "command":"printf ok > \"$SUBBAKE_OUTPUT_RESULT\"",
                    "outputs":{"result":"artifact.bin"},
                    "timeout_seconds":30
                })
            )
            .is_ok()
        );
        assert!(
            validate_tool_call(
                "run_command",
                &serde_json::json!({"command":"true","outputs":{"bad-name":"x"}})
            )
            .is_err()
        );
        assert!(
            validate_tool_call(
                "run_command",
                &serde_json::json!({"command":"true","timeout_seconds":1801})
            )
            .is_err()
        );
    }

    #[test]
    fn external_delete_requires_a_path_and_explicit_recursive_choice() {
        assert!(
            validate_tool_call(
                "delete_external_path",
                &serde_json::json!({"path":"/tmp/file"})
            )
            .is_err()
        );
        assert!(
            validate_tool_call(
                "delete_external_path",
                &serde_json::json!({"path":"/tmp/file","recursive":false})
            )
            .is_ok()
        );
        assert!(
            find_tool_handler("delete_external_path")
                .expect("external delete spec")
                .requires_approval_with(&serde_json::json!({
                    "path":"/tmp/file",
                    "recursive":false
                }))
        );
    }

    #[test]
    fn whisper_observations_are_read_only_but_asset_changes_require_approval() {
        let spec = find_tool_handler("manage_whisper").expect("manage_whisper");
        for action in ["status", "list-models", "list-vad-models", "list-versions"] {
            let arguments = serde_json::json!({"action": action});
            assert!(!spec.mutates_with(&arguments));
            assert!(!spec.requires_approval_with(&arguments));
        }
        for action in ["install", "update", "uninstall", "download", "download-vad"] {
            let arguments = serde_json::json!({"action": action});
            assert!(spec.mutates_with(&arguments));
            assert!(spec.requires_approval_with(&arguments));
        }
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
            "online_terminology",
            "output_format",
            "output_path",
            "overwrite",
            "fresh_runtime",
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
        for (index, handler) in ALL_TOOL_HANDLERS.iter().enumerate() {
            assert_eq!(find_tool_spec(handler.name), Some(&handler.spec));
            for other in &ALL_TOOL_HANDLERS[index + 1..] {
                assert_ne!(handler.name, other.name, "duplicate tool name");
                assert_ne!(
                    handler.executor(),
                    other.executor(),
                    "duplicate tool executor"
                );
            }
        }
    }
}
