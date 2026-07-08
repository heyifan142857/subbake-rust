//! Decision pipeline — quick-path matching + bounded LLM tool-call loop.
//!
//! Flow per `run_line`:
//!   1. Quick-path keywords match → immediate tool execution.
//!   2. Otherwise → bounded LLM loop (max 5 steps).
//!      - LLM returns a single structured decision (respond / tool_call / ask_user).
//!      - Discovery tools append observation + loop continues.
//!      - Mutating tools exit the loop after execution (or enter plan mode).
//!      - `respond` / `ask_user` / step-limit exit.

use std::io;
use std::path::PathBuf;

use serde_json::{Value as JsonValue, json};
use subbake_adapters::{
    TranscriptionRequest, TranscriptionSettings, TranslationRequest, TranslationSettings,
    WhisperAction, WhisperRequest, transcribe_media, translate_subtitle,
};
use subbake_core::entities::{BatchTranslationResult, TranslationLine, Usage};
use subbake_core::error::CoreResult;
use subbake_core::ports::{BackendJsonResult, BackendPayload, ChatMessage, LlmBackend};

use crate::engine::AgentEngine;
use crate::event::{EventKind, FileOpEventData};
use crate::guard::{FileOpAction, FileOpResult};
use crate::tools::ALL_TOOL_SPECS;

// ---------------------------------------------------------------------------
// Echo backend for agent decision loop (no TASK_START markers needed)
// ---------------------------------------------------------------------------

/// A lightweight `LlmBackend` that echoes the user message as a JSON
/// decision.  Used when no real LLM provider is configured — the pipeline
/// always chooses "respond" so the TUI/CLI flow can be exercised end-to-end.
pub struct EchoDecisionBackend {
    model: String,
}

impl EchoDecisionBackend {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }
}

impl LlmBackend for EchoDecisionBackend {
    fn provider_name(&self) -> &str {
        "echo-decision"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
        let (decision, usage) = self.generate_raw_json(messages)?;
        let text = serde_json::to_string(&decision).unwrap_or_default();

        Ok(BackendJsonResult {
            payload: BackendPayload::Translation(BatchTranslationResult {
                lines: vec![TranslationLine {
                    id: "1".to_owned(),
                    translation: text,
                }],
                summary: "echo decision".to_owned(),
                glossary_updates: Vec::new(),
            }),
            usage,
        })
    }

    fn generate_raw_json(
        &mut self,
        messages: &[ChatMessage],
    ) -> CoreResult<(serde_json::Value, Usage)> {
        // Extract the user message (last user message).
        let user_text = messages
            .iter()
            .rev()
            .find(|msg| msg.role == "user")
            .map(|msg| msg.content.as_str())
            .unwrap_or("");

        let decision = json!({
            "action": "respond",
            "text": user_text,
            "confidence": 1.0
        });
        let input_tokens = user_text.chars().count().div_ceil(4).max(1);
        Ok((
            decision,
            Usage {
                input_tokens,
                output_tokens: 1,
                total_tokens: input_tokens + 1,
            },
        ))
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const AGENT_LOOP_MAX_STEPS: usize = 5;
pub const CONFIDENCE_LOW: f64 = 0.4;
pub const CONFIDENCE_MEDIUM: f64 = 0.7;
pub const MIN_OBSERVATIONS: usize = 2;

// ---------------------------------------------------------------------------
// Loop-state types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Observation {
    #[allow(dead_code)]
    tool_name: String,
    text: String,
}

#[derive(Debug, Clone)]
struct LoopState {
    step: usize,
    max_steps: usize,
    observations: Vec<Observation>,
}

/// The LLM's structured decision.
struct Decision {
    action: String, // "respond" | "tool_call" | "ask_user"
    text: String,
    tool_name: Option<String>,
    arguments: Option<JsonValue>,
    confidence: f64,
}

// ---------------------------------------------------------------------------
// Engine entry point
// ---------------------------------------------------------------------------

impl AgentEngine {
    /// Process a single user input line.
    ///
    /// Returns the response text to show to the user.
    pub fn run_line(&mut self, input: &str, backend: &mut dyn LlmBackend) -> io::Result<String> {
        self.record_if_active(EventKind::User {
            text: input.to_owned(),
        })?;

        // 1. Quick-path: keyword matching without LLM.
        if let Some(result) = self.try_quick_path(input)? {
            if let Some(ref text) = result.response_text
                && let Some(ref mut obs) = self.observer
            {
                obs.on_response(text);
            }
            return Ok(result.output);
        }

        // 2. Bounded LLM loop.
        let mut state = LoopState {
            step: 0,
            max_steps: AGENT_LOOP_MAX_STEPS,
            observations: Vec::new(),
        };

        loop {
            if state.step >= state.max_steps {
                let msg = format!(
                    "I've tried {} steps without reaching a final action. Could you clarify?",
                    state.max_steps,
                );
                if let Some(ref mut obs) = self.observer {
                    obs.on_response(&msg);
                    obs.on_step_limit();
                }
                return Ok(msg);
            }
            state.step += 1;

            // Build context + call LLM.
            let decision = self.call_llm_for_decision(backend, input, &state)?;

            // Action gate: low-confidence respond → re-route.
            let decision = self.apply_confidence_gate(decision, &state);

            match decision.action.as_str() {
                "respond" => {
                    if let Some(ref mut obs) = self.observer {
                        obs.on_response(&decision.text);
                    }
                    return Ok(decision.text);
                }

                "ask_user" => {
                    let msg = decision.text;
                    if let Some(ref mut obs) = self.observer {
                        obs.on_response(&msg);
                    }
                    return Ok(msg);
                }

                "tool_call" => {
                    let tool_name = decision.tool_name.as_deref().unwrap_or("unknown");
                    let args = decision.arguments.unwrap_or(json!({}));

                    if self.is_discovery_tool(tool_name) {
                        // Discovery → run, append observation, continue.
                        let obs_text = self.run_tool(tool_name, &args)?;
                        state.observations.push(Observation {
                            tool_name: tool_name.to_owned(),
                            text: obs_text.clone(),
                        });
                        if let Some(ref mut obs) = self.observer {
                            obs.on_tool_call(tool_name, &args);
                            obs.on_observation(&obs_text);
                        }
                        continue;
                    }

                    // Mutating tool (execute, then exit loop).
                    // Check plan mode / approval.
                    let result_text = self.execute_or_plan_tool(tool_name, &args)?;
                    if let Some(ref mut obs) = self.observer {
                        obs.on_response(&result_text);
                    }
                    return Ok(result_text);
                }

                other => {
                    let msg = format!("I'm not sure how to proceed (action={other}).");
                    if let Some(ref mut obs) = self.observer {
                        obs.on_response(&msg);
                    }
                    return Ok(msg);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Quick-path deterministic matching
    // ------------------------------------------------------------------

    fn try_quick_path(&mut self, input: &str) -> io::Result<Option<QuickResult>> {
        let trimmed = input.trim();

        // Pattern: "translate @<path>" or "translate <path>"
        if let Some(path) = trimmed.strip_prefix("translate ") {
            let args = json!({"path": self.tool_path_arg(path)});
            let output = self.execute_or_plan_tool("translate_file", &args)?;
            return Ok(Some(QuickResult {
                response_text: Some(output.clone()),
                output,
            }));
        }

        // Pattern: "transcribe @<path>"
        if let Some(path) = trimmed.strip_prefix("transcribe ") {
            let args = json!({"path": self.tool_path_arg(path)});
            let output = self.execute_or_plan_tool("transcribe_audio", &args)?;
            return Ok(Some(QuickResult {
                response_text: Some(output.clone()),
                output,
            }));
        }

        // Pattern: "list files" or "ls"
        if matches!(trimmed, "list files" | "ls" | "list") {
            return Ok(Some(QuickResult {
                output: self
                    .guard
                    .list_files(std::path::Path::new("."))
                    .map(|files| {
                        files
                            .iter()
                            .map(|p| p.display().to_string())
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default(),
                response_text: None,
            }));
        }

        Ok(None)
    }

    fn tool_path_arg(&self, input: &str) -> String {
        // Strip @ prefix if present.
        input.trim().trim_start_matches('@').to_owned()
    }

    // ------------------------------------------------------------------
    // LLM decision call
    // ------------------------------------------------------------------

    fn call_llm_for_decision(
        &mut self,
        backend: &mut dyn LlmBackend,
        user_input: &str,
        state: &LoopState,
    ) -> io::Result<Decision> {
        let messages = self.build_decision_messages(user_input, state);
        if let Some(ref mut obs) = self.observer {
            obs.on_thinking("Deciding next action…");
        }
        let result = backend.generate_raw_json(&messages);
        match result {
            Ok((decision, _usage)) => self.parse_decision_value(&decision),
            Err(e) => {
                if let Some(ref mut obs) = self.observer {
                    obs.on_error(&e.to_string());
                }
                Ok(Decision {
                    action: "respond".into(),
                    text: format!("Error: {e}"),
                    tool_name: None,
                    arguments: None,
                    confidence: 1.0,
                })
            }
        }
    }

    /// Build the LLM message context for the decision call.
    fn build_decision_messages(&self, user_input: &str, state: &LoopState) -> Vec<ChatMessage> {
        let mut system = String::new();
        system.push_str("You are SubBake, a subtitle translation assistant.\n\n");
        system.push_str("Available tools:\n");
        for spec in ALL_TOOL_SPECS {
            system.push_str(&format!("- {}: {}", spec.name, spec.description));
            if spec.mutating {
                system.push_str(" (mutating)");
            }
            system.push('\n');
        }
        system.push_str("\nDecide the next action. Return JSON with keys:\n");
        system.push_str(r#"{"action": "respond" | "tool_call" | "ask_user", "text": "...", "tool_name": "...", "arguments": {...}}"#);
        system.push_str("\n- `respond`: reply to the user directly.\n");
        system.push_str("- `tool_call`: invoke a tool. Discovery tools feed observations back; mutating tools execute immediately.\n");
        system.push_str("- `ask_user`: ask the user for clarification.\n");
        system.push_str(
            "Keep confidence high (≥0.85) for direct tool calls, lower for clarification.\n",
        );
        system.push_str("Preserve subtitle id order, never merge or drop entries.\n");

        let mut user = String::new();
        user.push_str("User: ");
        user.push_str(user_input);
        user.push('\n');

        if !state.observations.is_empty() {
            user.push_str("\nObservations from earlier steps:\n");
            for (i, obs) in state.observations.iter().enumerate() {
                user.push_str(&format!("  [{i}] {}\n", obs.text));
            }
        }

        vec![ChatMessage::system(&system), ChatMessage::user(&user)]
    }

    fn parse_decision_value(&self, parsed: &JsonValue) -> io::Result<Decision> {
        Ok(Decision {
            action: parsed["action"].as_str().unwrap_or("respond").to_owned(),
            text: parsed["text"].as_str().unwrap_or("").to_owned(),
            tool_name: parsed["tool_name"].as_str().map(String::from),
            arguments: parsed.get("arguments").cloned(),
            confidence: parsed["confidence"].as_f64().unwrap_or(1.0),
        })
    }

    /// Gate: low-confidence `tool_call` → escalate to `ask_user` or `respond`.
    fn apply_confidence_gate(&self, decision: Decision, state: &LoopState) -> Decision {
        if decision.action != "tool_call" {
            return decision;
        }
        if decision.confidence >= CONFIDENCE_MEDIUM {
            return decision;
        }
        if decision.confidence < CONFIDENCE_LOW {
            return Decision {
                action: "respond".into(),
                text: "Could you clarify what you'd like me to do?".into(),
                tool_name: None,
                arguments: None,
                confidence: 1.0,
            };
        }
        // Medium confidence: ask_user if few observations.
        if state.observations.len() < MIN_OBSERVATIONS {
            return Decision {
                action: "ask_user".into(),
                text: format!(
                    "Shall I {} with {:?}?",
                    decision.tool_name.as_deref().unwrap_or("?"),
                    decision.arguments
                ),
                tool_name: decision.tool_name,
                arguments: decision.arguments,
                confidence: decision.confidence,
            };
        }
        decision
    }

    // ------------------------------------------------------------------
    // Plan mode check
    // ------------------------------------------------------------------

    fn is_in_plan_mode(&self) -> bool {
        self.session.as_ref().is_some_and(|s| s.mode == "plan")
    }

    // ------------------------------------------------------------------
    // Tool runner (stub — dispatches to real adapters)
    // ------------------------------------------------------------------

    fn execute_or_plan_tool(&mut self, tool_name: &str, args: &JsonValue) -> io::Result<String> {
        if let Some(ref mut obs) = self.observer {
            obs.on_tool_call(tool_name, args);
        }

        if self.tool_requires_approval(tool_name) || self.is_in_plan_mode() {
            let draft = crate::event::ToolCallDraft {
                tool_name: tool_name.to_owned(),
                arguments: args.clone(),
            };
            self.store_plan("", vec![draft])?;
            return Ok(
                "I've prepared a plan for your approval. Use `/approve` to execute.".to_owned(),
            );
        }

        self.run_tool(tool_name, args)
    }

    pub(crate) fn record_if_active(&mut self, kind: EventKind) -> io::Result<()> {
        if self.session.is_some() {
            self.record(kind)?;
        }
        Ok(())
    }

    /// Execute a tool by name with arguments. Returns a text summary.
    pub(crate) fn run_tool(&mut self, name: &str, args: &JsonValue) -> io::Result<String> {
        self.record_if_active(EventKind::ToolCall {
            tool_name: name.to_owned(),
            arguments: args.clone(),
        })?;

        match name {
            // -- Browse (FileGuard) --
            "list_files" => {
                let dir = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                let files = self.guard.list_files(PathBuf::from(dir).as_path())?;
                Ok(files
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
            "search_files" => {
                let dir = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                let pat = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
                let files = self.guard.search_files(PathBuf::from(dir).as_path(), pat)?;
                Ok(format_file_list(&files))
            }
            "read_file" => {
                let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
                self.guard.read_file(PathBuf::from(path).as_path())
            }
            "read_file_preview" => {
                let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
                let content = self.guard.read_file(PathBuf::from(path).as_path())?;
                let preview: String = content.chars().take(2000).collect();
                Ok(if preview.len() < content.len() {
                    format!("{preview}\n… (truncated)")
                } else {
                    preview
                })
            }
            "candidate_subtitles" => {
                let dir = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                let files = self
                    .guard
                    .search_files(PathBuf::from(dir).as_path(), ".srt")?;
                Ok(format_file_list(&files))
            }
            "recent_translations" => {
                let session = self.session.as_ref();
                let events = session
                    .map(|s| &s.events)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let mut out = Vec::new();
                for event in events.iter().rev().take(20) {
                    let tool_name = event.data.get("tool_name").and_then(|value| value.as_str());
                    if matches!(tool_name, Some("translate_file" | "translate_series")) {
                        out.push(event.text.clone());
                    }
                }
                Ok(out.join("\n"))
            }

            // -- File operations (FileGuard) --
            "create_file" => {
                let path = req_arg(args, "path")?;
                let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let r = self.guard.create_file(&path, content)?;
                self.record_file_operation(&r)?;
                Ok(format!("Created {}", r.path.display()))
            }
            "append_file" => {
                let path = req_arg(args, "path")?;
                let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let r = self.guard.append_file(&path, content)?;
                self.record_file_operation(&r)?;
                Ok(format!(
                    "Appended {} (backup: {})",
                    r.path.display(),
                    r.backup_path
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default()
                ))
            }
            "replace_in_file" => {
                let path = req_arg(args, "path")?;
                let old = args.get("old").and_then(|v| v.as_str()).unwrap_or("");
                let new = args.get("new").and_then(|v| v.as_str()).unwrap_or("");
                let r = self.guard.replace_in_file(&path, old, new)?;
                self.record_file_operation(&r)?;
                Ok(format!(
                    "Replaced in {} (backup: {})",
                    r.path.display(),
                    r.backup_path
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default()
                ))
            }
            "rename_path" => {
                let from = req_arg(args, "from")?;
                let to = req_arg(args, "to")?;
                let r = self.guard.rename_path(&from, &to)?;
                self.record_file_operation(&r)?;
                Ok(format!(
                    "Renamed {} → {}",
                    r.path.display(),
                    r.new_path
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default()
                ))
            }
            "delete_file" => {
                let path = req_arg(args, "path")?;
                let r = self.guard.delete_file(&path)?;
                self.record_file_operation(&r)?;
                Ok(format!(
                    "Deleted {} (backup: {})",
                    r.path.display(),
                    r.backup_path
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default()
                ))
            }

            // -- Translate --
            "translate_file" => {
                let path = req_arg(args, "path")?;
                let input = self.guard.resolve_path(&path)?;
                let request = TranslationRequest {
                    input_path: input,
                    output_path: None,
                    settings: TranslationSettings::default(),
                };
                let outcome = translate_subtitle(request)?;
                Ok(outcome
                    .output_path
                    .map(|p| format!("Translated: {}", p.display()))
                    .unwrap_or_default())
            }
            "translate_series" => {
                let dir = req_arg(args, "path")?;
                let input = self.guard.resolve_path(&dir)?;
                let request = subbake_adapters::BatchTranslationRequest {
                    root: input,
                    recursive: args
                        .get("recursive")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    overwrite: args
                        .get("overwrite")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    settings: TranslationSettings::default(),
                };
                let outcome = subbake_adapters::translate_subtitle_batch(request)?;
                Ok(format!(
                    "Translated {} files, skipped {}.",
                    outcome.processed,
                    outcome.skipped.len()
                ))
            }

            // -- Edit: read file, show content, suggest re-translate --
            "edit_subtitle" => {
                let path = req_arg(args, "path")?;
                let full = self.project_root.join(&path);
                if full.exists() {
                    let content = std::fs::read_to_string(&full)
                        .unwrap_or_else(|_| "(unreadable)".to_owned());
                    let preview: String = content.chars().take(500).collect();
                    Ok(format!(
                        "Current content of {}:\n{}\n\nTo edit, set `path` and `instructions` in arguments.",
                        path.display(),
                        preview
                    ))
                } else {
                    Ok(format!("File {} not found.", path.display()))
                }
            }

            // -- Transcribe --
            "transcribe_audio" => {
                let path = req_arg(args, "path")?;
                let input = self.guard.resolve_path(&path)?;
                let request = TranscriptionRequest {
                    media_path: input,
                    output_path: None,
                    settings: TranscriptionSettings::default(),
                };
                match transcribe_media(request) {
                    Ok(outcome) => Ok(format!("Transcribed: {}", outcome.output_path.display())),
                    Err(e) => Ok(format!("Transcription needs setup: {e}")),
                }
            }

            // -- Whisper management --
            "manage_whisper" => {
                let action_str = args
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("status");
                let action = match action_str {
                    "install" => WhisperAction::Install,
                    "status" => WhisperAction::Status,
                    "list-models" | "models" => WhisperAction::ListModels,
                    "download" => {
                        let name = args
                            .get("model")
                            .and_then(|v| v.as_str())
                            .unwrap_or("small");
                        WhisperAction::DownloadModel {
                            name: name.to_owned(),
                        }
                    }
                    other => return Ok(format!("unknown whisper action `{other}`")),
                };
                let request = WhisperRequest {
                    action,
                    binary_path: None,
                    models_dir: None,
                };
                match subbake_adapters::run_whisper(request) {
                    Ok(_) => Ok("whisper: done".to_owned()),
                    Err(e) => Ok(format!("whisper: {e}")),
                }
            }

            // -- Diagnose: read failure logs from a run directory --
            "diagnose_path" => {
                let path = req_arg(args, "path")?;
                let run_dir = self.project_root.join(&path).join(".subbake/runs");
                let mut results = Vec::new();
                if run_dir.exists() {
                    for entry in std::fs::read_dir(&run_dir).map_err(io::Error::other)? {
                        let entry = entry.map_err(io::Error::other)?;
                        let failures_dir = entry.path().join("failures");
                        if failures_dir.exists() {
                            for f in std::fs::read_dir(&failures_dir).map_err(io::Error::other)? {
                                let f = f.map_err(io::Error::other)?;
                                if let Ok(content) = std::fs::read_to_string(f.path()) {
                                    results.push(format!(
                                        "{}: {}",
                                        f.path().display(),
                                        &content[..content.len().min(200)]
                                    ));
                                }
                            }
                        }
                    }
                }
                if results.is_empty() {
                    Ok("No failure logs found.".to_owned())
                } else {
                    Ok(results.join("\n---\n"))
                }
            }
            "diagnose_text" => {
                let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
                Ok(format!(
                    "Diagnose text: received {} chars. For detailed diagnosis, set `path` to a run directory.",
                    text.len()
                ))
            }

            // -- Profile: read profiles from subbake.toml --
            "list_profiles" => {
                let config_path = self.project_root.join("subbake.toml");
                if !config_path.exists() {
                    return Ok("No subbake.toml found in project root.".to_owned());
                }
                let content = std::fs::read_to_string(&config_path)
                    .map_err(|e| io::Error::other(format!("read config: {e}")))?;
                let profiles = find_profile_names(&content);
                if profiles.is_empty() {
                    Ok(
                        "No profiles defined in subbake.toml. Create [profiles.<name>] sections."
                            .to_owned(),
                    )
                } else {
                    Ok(format!("Profiles: {}", profiles.join(", ")))
                }
            }
            "switch_profile" => {
                let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let config_path = self.project_root.join("subbake.toml");
                if !config_path.exists() {
                    return Ok(
                        "No subbake.toml. Create one with [profiles.<name>] sections.".to_owned(),
                    );
                }
                let content = std::fs::read_to_string(&config_path)
                    .map_err(|e| io::Error::other(format!("read config: {e}")))?;
                let profiles = find_profile_names(&content);
                if profiles.contains(&name.to_owned()) {
                    // Switching is done by the user editing their config default_profile.
                    Ok(format!(
                        "Profile `{name}` exists. Set `default_profile = \"{name}\"` in subbake.toml to activate it."
                    ))
                } else {
                    Ok(format!(
                        "Profile `{name}` not found. Available: {}",
                        profiles.join(", ")
                    ))
                }
            }

            _ => Ok(format!("[{name}: not yet wired]")),
        }
    }

    fn record_file_operation(&mut self, result: &FileOpResult) -> io::Result<()> {
        self.record_if_active(EventKind::FileOperation(FileOpEventData {
            action: file_action_label(result.action).to_owned(),
            path: self.event_path(&result.path),
            new_path: result.new_path.as_ref().map(|path| self.event_path(path)),
            backup_path: result
                .backup_path
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
            group_id: None,
            undone: false,
        }))
    }

    fn event_path(&self, path: &std::path::Path) -> String {
        let root = self
            .project_root
            .canonicalize()
            .unwrap_or_else(|_| self.project_root.clone());
        path.strip_prefix(&root)
            .or_else(|_| path.strip_prefix(&self.project_root))
            .unwrap_or(path)
            .to_string_lossy()
            .to_string()
    }
}

fn file_action_label(action: FileOpAction) -> &'static str {
    match action {
        FileOpAction::Create => "created",
        FileOpAction::Append => "appended",
        FileOpAction::Modified => "modified",
        FileOpAction::Renamed => "renamed",
        FileOpAction::Deleted => "deleted",
    }
}

/// Extract a required string argument from the LLM's tool args, or error.
fn req_arg(args: &JsonValue, key: &str) -> io::Result<PathBuf> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::other(format!("missing required argument `{key}`")))
}

fn format_file_list(files: &[PathBuf]) -> String {
    if files.is_empty() {
        return "(no files found)".to_owned();
    }
    files
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse lines like `[profiles.myprofile]` from TOML content.
fn find_profile_names(toml: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in toml.lines() {
        let trimmed = line.trim();
        if let Some(inner) = trimmed.strip_prefix("[profiles.")
            && let Some(name) = inner.strip_suffix(']')
        {
            let clean = name.trim();
            if !clean.is_empty() {
                names.push(clean.to_owned());
            }
        }
    }
    names.sort();
    names
}

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct QuickResult {
    output: String,
    response_text: Option<String>,
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn quick_path_translate_executes_tool() {
        let root = temp_root("quick-translate");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join("clip.txt"), "hello\n").expect("write subtitle");

        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = EchoDecisionBackend::new("test");

        let output = engine
            .run_line("translate @clip.txt", &mut backend)
            .expect("run line");

        assert!(output.contains("Translated:"), "{output}");
        assert!(root.join("clip.translated.txt").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn file_tool_records_event_and_undo_removes_created_file() {
        let root = temp_root("file-undo");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        let result = engine
            .run_tool(
                "create_file",
                &json!({"path": "notes.txt", "content": "hello"}),
            )
            .expect("create file");

        assert!(result.contains("Created"));
        assert!(root.join("notes.txt").exists());
        assert!(
            engine
                .session
                .as_ref()
                .expect("session")
                .events
                .iter()
                .any(|event| event.kind == "file_operation")
        );

        let undone = engine.undo_last().expect("undo");
        assert!(undone.contains("Undone"));
        assert!(!root.join("notes.txt").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-agent-decision-{label}-{nanos}"))
    }
}
