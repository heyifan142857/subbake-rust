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
    ConfigFile, SubtitleEditRequest, TranscriptionRequest, TranscriptionSettings,
    TranslationRequest, TranslationSettings, WhisperAction, WhisperRequest, default_output_path,
    diagnose_failure_path, edit_subtitle, is_supported_subtitle_path, load_diagnostic_reports,
    transcribe_media, translate_subtitle,
};
use subbake_core::diagnostics::{diagnose_text as diagnose_failure_text, format_diagnostic_report};
use subbake_core::entities::{BatchTranslationResult, TranslationLine, Usage};
use subbake_core::error::CoreResult;
use subbake_core::ports::{BackendJsonResult, BackendPayload, ChatMessage, LlmBackend};

use crate::discovery::{rank_subtitle_candidates, summarize_observation};
use crate::engine::AgentEngine;
use crate::event::{EventKind, FileOpEventData, ToolCallDraft};
use crate::guard::{FileOpAction, FileOpResult};
use crate::tools::{ranked_tool_specs, validate_tool_call};

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
    tool_name: String,
    text: String,
    summary: String,
}

#[derive(Debug, Clone)]
struct LoopState {
    step: usize,
    max_steps: usize,
    observations: Vec<Observation>,
}

/// The LLM's structured decision.
struct Decision {
    action: String, // "respond" | "tool_call" | "plan" | "ask_user"
    text: String,
    tool_name: Option<String>,
    arguments: Option<JsonValue>,
    tool_calls: Vec<ToolCallDraft>,
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
            return self.finish_response(result.output, false, result.response_text.is_some());
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
                    obs.on_step_limit();
                }
                return self.finish_response(msg, true, true);
            }
            state.step += 1;

            // Build context + call LLM.
            let decision = self.call_llm_for_decision(backend, input, &state)?;

            // Action gate: low-confidence respond → re-route.
            let decision = self.apply_confidence_gate(decision, &state);

            match decision.action.as_str() {
                "respond" => {
                    return self.finish_response(decision.text, false, true);
                }

                "ask_user" => {
                    return self.finish_response(decision.text, true, true);
                }

                "tool_call" => {
                    let tool_name = decision.tool_name.as_deref().unwrap_or("unknown");
                    let args = decision.arguments.unwrap_or(json!({}));

                    if self.is_discovery_tool(tool_name) {
                        // Discovery → run, append observation, continue.
                        let obs_text = self.run_tool(tool_name, &args)?;
                        let summary = summarize_observation(tool_name, &obs_text);
                        state.observations.push(Observation {
                            tool_name: tool_name.to_owned(),
                            text: obs_text.clone(),
                            summary,
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
                    return self.finish_response(result_text, false, true);
                }

                "plan" => {
                    self.store_plan(&decision.text, decision.tool_calls)?;
                    return self.finish_response(
                        "I've prepared a plan for your approval. Choose an action below."
                            .to_owned(),
                        false,
                        true,
                    );
                }

                other => {
                    let msg = format!("I'm not sure how to proceed (action={other}).");
                    return self.finish_response(msg, true, true);
                }
            }
        }
    }

    fn finish_response(
        &mut self,
        text: String,
        ask_user: bool,
        notify_observer: bool,
    ) -> io::Result<String> {
        let event = if ask_user {
            EventKind::AskUser { text: text.clone() }
        } else {
            EventKind::Assistant { text: text.clone() }
        };
        self.record_if_active(event)?;
        if notify_observer && let Some(ref mut observer) = self.observer {
            observer.on_response(&text);
        }
        Ok(text)
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
            Ok((decision, _usage)) => match self.parse_decision_value(&decision) {
                Ok(decision) => Ok(decision),
                Err(error) => {
                    if let Some(ref mut obs) = self.observer {
                        obs.on_error(&error.to_string());
                    }
                    Ok(Decision {
                        action: "ask_user".into(),
                        text: "I couldn't validate the proposed action. Could you clarify the file and operation?".into(),
                        tool_name: None,
                        arguments: None,
                        tool_calls: Vec::new(),
                        confidence: 1.0,
                    })
                }
            },
            Err(e) => {
                if let Some(ref mut obs) = self.observer {
                    obs.on_error(&e.to_string());
                }
                Ok(Decision {
                    action: "respond".into(),
                    text: format!("Error: {e}"),
                    tool_name: None,
                    arguments: None,
                    tool_calls: Vec::new(),
                    confidence: 1.0,
                })
            }
        }
    }

    /// Build the LLM message context for the decision call.
    fn build_decision_messages(&self, user_input: &str, state: &LoopState) -> Vec<ChatMessage> {
        let mut system = String::new();
        system.push_str("You are SubBake, a subtitle translation assistant.\n\n");
        system.push_str("Relevant available tools:\n");
        for spec in ranked_tool_specs(user_input) {
            system.push_str(&format!("- {}: {}", spec.name, spec.description));
            if spec.mutating {
                system.push_str(" (mutating)");
            }
            system.push('\n');
        }
        system.push_str("\nDecide the next action. Return JSON with keys:\n");
        system.push_str(r#"{"action": "respond" | "tool_call" | "plan" | "ask_user", "text": "...", "tool_name": "...", "arguments": {...}, "tool_calls": [{"tool_name": "...", "arguments": {...}}], "confidence": 0.0}"#);
        system.push_str("\n- `respond`: reply to the user directly.\n");
        system.push_str("- `tool_call`: invoke a tool. Discovery tools feed observations back; mutating tools execute immediately.\n");
        system.push_str(
            "- `plan`: propose one or more mutating tool calls that must wait for `/approve`.\n",
        );
        system.push_str("- `ask_user`: ask the user for clarification.\n");
        system.push_str(
            "Keep confidence high (≥0.85) for direct tool calls, lower for clarification.\n",
        );
        system.push_str("Preserve subtitle id order, never merge or drop entries.\n");

        let mut user = String::new();
        user.push_str("User: ");
        user.push_str(user_input);
        user.push('\n');

        if let Some(summary) = self.conversation_context_summary(12) {
            user.push_str("\nRecent session context:\n");
            user.push_str(&summary);
            user.push('\n');
        }

        if !state.observations.is_empty() {
            user.push_str("\nObservations from earlier steps:\n");
            for (i, obs) in state.observations.iter().enumerate() {
                user.push_str(&format!("  [{i}] {}: {}\n", obs.tool_name, obs.summary));
                for line in obs.text.lines().take(3) {
                    user.push_str(&format!("      {}\n", truncate_text(line, 240)));
                }
            }
        }

        vec![ChatMessage::system(&system), ChatMessage::user(&user)]
    }

    fn parse_decision_value(&self, parsed: &JsonValue) -> io::Result<Decision> {
        let object = parsed
            .as_object()
            .ok_or_else(|| io::Error::other("agent decision must be a JSON object"))?;
        let raw_action = object
            .get("action")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| io::Error::other("agent decision is missing `action`"))?;
        let action = match raw_action {
            "final_tool_call" => "tool_call",
            "respond" | "tool_call" | "plan" | "ask_user" => raw_action,
            other => {
                return Err(io::Error::other(format!(
                    "unsupported agent action `{other}`"
                )));
            }
        }
        .to_owned();
        let text = object
            .get("text")
            .and_then(JsonValue::as_str)
            .unwrap_or("")
            .to_owned();
        let confidence = object
            .get("confidence")
            .and_then(JsonValue::as_f64)
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let mut tool_name = None;
        let mut arguments = None;
        let mut tool_calls = Vec::new();
        if action == "tool_call" {
            let name = object
                .get("tool_name")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| io::Error::other("tool call is missing `tool_name`"))?;
            let args = object
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            validate_tool_call(name, &args).map_err(io::Error::other)?;
            tool_name = Some(name.to_owned());
            arguments = Some(args);
        } else if action == "plan" {
            let calls = object
                .get("tool_calls")
                .and_then(JsonValue::as_array)
                .ok_or_else(|| io::Error::other("plan is missing `tool_calls`"))?;
            if calls.is_empty() {
                return Err(io::Error::other("plan must contain at least one tool call"));
            }
            for call in calls {
                let name = call
                    .get("tool_name")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| io::Error::other("planned call is missing `tool_name`"))?;
                let args = call.get("arguments").cloned().unwrap_or_else(|| json!({}));
                validate_tool_call(name, &args).map_err(io::Error::other)?;
                if self.is_discovery_tool(name) {
                    return Err(io::Error::other(
                        "discovery tools must run before creating a plan",
                    ));
                }
                tool_calls.push(ToolCallDraft {
                    tool_name: name.to_owned(),
                    arguments: args,
                });
            }
        }
        Ok(Decision {
            action,
            text,
            tool_name,
            arguments,
            tool_calls,
            confidence,
        })
    }

    /// Gate: low-confidence `tool_call` → escalate to `ask_user` or `respond`.
    fn apply_confidence_gate(&self, decision: Decision, state: &LoopState) -> Decision {
        if !matches!(decision.action.as_str(), "tool_call" | "plan") {
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
                tool_calls: Vec::new(),
                confidence: 1.0,
            };
        }
        // Medium confidence: ask_user if few observations.
        if state.observations.len() < MIN_OBSERVATIONS {
            return Decision {
                action: "ask_user".into(),
                text: format!(
                    "Shall I proceed with {}?",
                    decision
                        .tool_name
                        .as_deref()
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!(
                            "{} planned operation(s)",
                            decision.tool_calls.len()
                        ))
                ),
                tool_name: decision.tool_name,
                arguments: decision.arguments,
                tool_calls: Vec::new(),
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
                "I've prepared a plan for your approval. Choose an action below.".to_owned(),
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
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let files = self.guard.search_files(PathBuf::from(dir).as_path(), "")?;
                let ranked = rank_subtitle_candidates(files, query, &self.project_root);
                Ok(format_file_list(&ranked))
            }
            "recent_translations" => {
                let session = self.session.as_ref();
                let events = session
                    .map(|s| &s.events)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let mut out = Vec::new();
                for event in events.iter().rev().take(20) {
                    if event.kind != "file_operation"
                        || event
                            .data
                            .get("undone")
                            .and_then(JsonValue::as_bool)
                            .unwrap_or(false)
                    {
                        continue;
                    }
                    let path = event.data.get("path").and_then(JsonValue::as_str);
                    if path.is_some_and(|path| {
                        path.contains(".translated.") || path.contains(".bilingual.")
                    }) {
                        out.push(path.unwrap_or_default().to_owned());
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
                let settings = self.active_translation_settings()?;
                let output_path =
                    default_output_path(&input, settings.output_format(), settings.bilingual)?;
                let undo_snapshot = self.guard.snapshot_write(&output_path)?;
                let request = TranslationRequest {
                    input_path: input,
                    output_path: None,
                    settings,
                };
                let outcome = translate_subtitle(request)?;
                if outcome.output_path.is_some() {
                    self.record_file_operation(&undo_snapshot)?;
                }
                Ok(outcome
                    .output_path
                    .map(|p| format!("Translated: {}", p.display()))
                    .unwrap_or_default())
            }
            "translate_series" => {
                let dir = req_arg(args, "path")?;
                let input = self.guard.resolve_path(&dir)?;
                let recursive = args
                    .get("recursive")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let overwrite = args
                    .get("overwrite")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let settings = self.active_translation_settings()?;
                let source_files = if recursive {
                    self.guard.search_files(&input, "")?
                } else {
                    self.guard.list_files(&input)?
                };
                let mut undo_snapshots = Vec::new();
                for source in source_files
                    .into_iter()
                    .filter(|path| path.is_file() && is_supported_subtitle_path(path))
                    .filter(|path| {
                        !path
                            .file_stem()
                            .and_then(|stem| stem.to_str())
                            .is_some_and(|stem| {
                                stem.ends_with(".translated") || stem.ends_with(".bilingual")
                            })
                    })
                {
                    let output =
                        default_output_path(&source, settings.output_format(), settings.bilingual)?;
                    if overwrite || !output.exists() {
                        let snapshot = self.guard.snapshot_write(&output)?;
                        undo_snapshots.push((output, snapshot));
                    }
                }
                let request = subbake_adapters::BatchTranslationRequest {
                    root: input,
                    recursive,
                    overwrite,
                    settings,
                };
                let outcome = subbake_adapters::translate_subtitle_batch(request)?;
                let group_id = format!("translate-series-{}", crate::session::iso_now());
                for output in &outcome.outputs {
                    if let Some((_, snapshot)) =
                        undo_snapshots.iter().find(|(path, _)| path == output)
                    {
                        self.record_file_operation_with_group(snapshot, Some(group_id.clone()))?;
                    }
                }
                Ok(format!(
                    "Translated {} files, skipped {}.",
                    outcome.processed,
                    outcome.skipped.len()
                ))
            }

            // -- Edit: targeted rewrite of a generated subtitle file --
            "edit_subtitle" => {
                let path = req_arg(args, "path")?;
                let instruction = req_string_arg(args, "instruction")?;
                let target_path = self.guard.resolve_path(&path)?;
                let snapshot = self.guard.snapshot_write(&target_path)?;
                let outcome = edit_subtitle(SubtitleEditRequest {
                    target_path: target_path.clone(),
                    instruction,
                    settings: self.active_translation_settings()?,
                    allow_non_generated: args
                        .get("allow_non_generated")
                        .and_then(|value| value.as_bool())
                        .unwrap_or(false),
                })?;
                self.record_file_operation(&snapshot)?;
                let mut lines = vec![format!("Edited: {}", target_path.display())];
                if !outcome.edit_notes.trim().is_empty() {
                    lines.push(outcome.edit_notes);
                }
                Ok(lines.join("\n"))
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
                    "update" => WhisperAction::Update,
                    "uninstall" => WhisperAction::Uninstall {
                        keep_models: args
                            .get("keep_models")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(false),
                    },
                    "status" => WhisperAction::Status,
                    "list-models" | "models" => WhisperAction::ListModels,
                    "download" | "download_model" => {
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

            // -- Diagnose: structured failure summary from file/run dir/text --
            "diagnose_path" => {
                let path = req_arg(args, "path")?;
                let full = self.guard.resolve_path(&path)?;
                let reports = if full.is_file() {
                    vec![diagnose_failure_path(&full)?]
                } else {
                    load_diagnostic_reports(&full)?
                };
                if reports.is_empty() {
                    Ok("No failure logs found.".to_owned())
                } else {
                    Ok(reports
                        .iter()
                        .map(format_diagnostic_report)
                        .collect::<Vec<_>>()
                        .join("\n\n---\n\n"))
                }
            }
            "diagnose_text" => {
                let text = req_string_arg(args, "text")?;
                Ok(format_diagnostic_report(&diagnose_failure_text(
                    &text,
                    "pasted diagnostic text",
                )))
            }

            // -- Profile: read and switch active session profile --
            "list_profiles" => {
                let Some((_, config)) = self.load_project_config()? else {
                    return Ok("No subbake config found in project root.".to_owned());
                };
                let mut profiles = config.profiles.keys().cloned().collect::<Vec<_>>();
                profiles.sort();
                if profiles.is_empty() {
                    Ok(
                        "No profiles defined in subbake.toml. Create [profiles.<name>] sections."
                            .to_owned(),
                    )
                } else {
                    let active = self
                        .session
                        .as_ref()
                        .and_then(|session| session.profile.clone());
                    Ok(format_profile_list(&profiles, active.as_deref()))
                }
            }
            "switch_profile" => {
                let name = req_string_arg(args, "name")?;
                let Some((config_path, config)) = self.load_project_config()? else {
                    return Ok(
                        "No subbake config found. Create one with [profiles.<name>] sections."
                            .to_owned(),
                    );
                };
                if !config.profiles.contains_key(&name) {
                    let mut profiles = config.profiles.keys().cloned().collect::<Vec<_>>();
                    profiles.sort();
                    return Ok(format!(
                        "Profile `{name}` not found. Available: {}",
                        profiles.join(", ")
                    ));
                }
                let settings = self.settings_for_profile(&config, Some(&name));
                let session = self
                    .session
                    .as_mut()
                    .ok_or_else(|| io::Error::other("no active session"))?;
                session.profile = Some(name.clone());
                session.config_path = Some(config_path.to_string_lossy().to_string());
                self.save()?;
                self.record_if_active(EventKind::Profile { name: name.clone() })?;
                Ok(format!(
                    "Profile switched: {name} ({}/{})",
                    settings.provider, settings.model
                ))
            }

            _ => Ok(format!("[{name}: not yet wired]")),
        }
    }

    fn record_file_operation(&mut self, result: &FileOpResult) -> io::Result<()> {
        self.record_file_operation_with_group(result, None)
    }

    fn record_file_operation_with_group(
        &mut self,
        result: &FileOpResult,
        group_id: Option<String>,
    ) -> io::Result<()> {
        self.record_if_active(EventKind::FileOperation(FileOpEventData {
            action: file_action_label(result.action).to_owned(),
            path: self.event_path(&result.path),
            new_path: result.new_path.as_ref().map(|path| self.event_path(path)),
            backup_path: result
                .backup_path
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
            group_id,
            undone: false,
        }))
    }

    pub(crate) fn load_project_config(&self) -> io::Result<Option<(PathBuf, ConfigFile)>> {
        let candidates = [
            self.session
                .as_ref()
                .and_then(|session| session.config_path.as_deref().map(PathBuf::from)),
            Some(self.project_root.join("subbake.toml")),
            Some(self.project_root.join(".subbake.toml")),
        ];
        for path in candidates.into_iter().flatten() {
            if path.exists() {
                return ConfigFile::load(&path).map(|config| Some((path, config)));
            }
        }
        Ok(None)
    }

    pub(crate) fn active_translation_settings(&self) -> io::Result<TranslationSettings> {
        let profile = self
            .session
            .as_ref()
            .and_then(|session| session.profile.as_deref());
        let Some((_, config)) = self.load_project_config()? else {
            return Ok(TranslationSettings::default());
        };
        Ok(self.settings_for_profile(&config, profile))
    }

    fn settings_for_profile(
        &self,
        config: &ConfigFile,
        profile: Option<&str>,
    ) -> TranslationSettings {
        TranslationSettings::default().with_patch(config.resolve(profile))
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

fn req_string_arg(args: &JsonValue, key: &str) -> io::Result<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_owned)
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

fn truncate_text(text: &str, limit: usize) -> String {
    let value = text.chars().take(limit).collect::<String>();
    if text.chars().count() > limit {
        format!("{value}...")
    } else {
        value
    }
}

fn format_profile_list(profiles: &[String], active: Option<&str>) -> String {
    let rendered = profiles
        .iter()
        .map(|name| {
            if Some(name.as_str()) == active {
                format!("{name} (active)")
            } else {
                name.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("Profiles: {rendered}")
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

    struct RawDecisionBackend {
        decision: JsonValue,
    }

    impl LlmBackend for RawDecisionBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "decision"
        }

        fn generate_json(&mut self, messages: &[ChatMessage]) -> CoreResult<BackendJsonResult> {
            EchoDecisionBackend::new("test").generate_json(messages)
        }

        fn generate_raw_json(
            &mut self,
            _messages: &[ChatMessage],
        ) -> CoreResult<(JsonValue, Usage)> {
            Ok((self.decision.clone(), Usage::default()))
        }
    }

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
        assert!(
            engine
                .session
                .as_ref()
                .expect("session")
                .events
                .iter()
                .any(|event| event.kind == "assistant")
        );
        let context = engine
            .conversation_context_summary(12)
            .expect("conversation summary");
        assert!(context.contains("User:"));
        assert!(context.contains("Assistant:"));

        engine.undo_last().expect("undo translation");
        assert!(!root.join("clip.translated.txt").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn undo_restores_overwritten_translation_output() {
        let root = temp_root("translate-overwrite-undo");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join("clip.txt"), "new\n").expect("write subtitle");
        std::fs::write(root.join("clip.translated.txt"), "old\n").expect("write old output");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = EchoDecisionBackend::new("test");

        engine
            .run_line("translate @clip.txt", &mut backend)
            .expect("translate");
        assert_ne!(
            std::fs::read_to_string(root.join("clip.translated.txt")).expect("translated"),
            "old\n"
        );

        engine.undo_last().expect("undo translation");
        assert_eq!(
            std::fs::read_to_string(root.join("clip.translated.txt")).expect("restored"),
            "old\n"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn undo_removes_series_outputs_as_one_group() {
        let root = temp_root("series-undo");
        let season = root.join("season");
        std::fs::create_dir_all(&season).expect("create season");
        std::fs::write(season.join("one.txt"), "one\n").expect("write one");
        std::fs::write(season.join("two.txt"), "two\n").expect("write two");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        engine
            .run_tool(
                "translate_series",
                &json!({"path": "season", "recursive": false}),
            )
            .expect("translate series");
        assert!(season.join("one.translated.txt").exists());
        assert!(season.join("two.translated.txt").exists());
        let recent = engine
            .run_tool("recent_translations", &json!({}))
            .expect("recent");
        assert!(recent.contains("one.translated.txt"));
        assert!(recent.contains("two.translated.txt"));

        let result = engine.undo_last().expect("undo series");
        assert!(result.contains("2 operations"));
        assert!(!season.join("one.translated.txt").exists());
        assert!(!season.join("two.translated.txt").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn llm_plan_waits_for_approval_and_then_executes() {
        let root = temp_root("llm-plan");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = RawDecisionBackend {
            decision: json!({
                "action": "plan",
                "text": "Create notes",
                "tool_calls": [{
                    "tool_name": "create_file",
                    "arguments": {"path": "notes.txt", "content": "hello"}
                }],
                "confidence": 0.95
            }),
        };

        let response = engine.run_line("create notes", &mut backend).expect("plan");
        assert!(response.contains("Choose an action below"));
        assert!(!root.join("notes.txt").exists());
        assert_eq!(
            engine
                .session
                .as_ref()
                .expect("session")
                .pending_plan
                .as_ref()
                .expect("pending")
                .tool_calls
                .len(),
            1
        );

        engine.approve_plan().expect("approve");
        assert_eq!(
            std::fs::read_to_string(root.join("notes.txt")).expect("created"),
            "hello"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn failed_plan_keeps_only_unfinished_tool_calls() {
        let root = temp_root("partial-plan");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        engine
            .store_plan(
                "Create then append",
                vec![
                    ToolCallDraft {
                        tool_name: "create_file".to_owned(),
                        arguments: json!({"path": "notes.txt", "content": "hello"}),
                    },
                    ToolCallDraft {
                        tool_name: "rename_path".to_owned(),
                        arguments: json!({"from": "notes.txt"}),
                    },
                ],
            )
            .expect("store plan");

        engine.approve_plan().expect_err("rename must fail");
        assert_eq!(
            std::fs::read_to_string(root.join("notes.txt")).expect("first call completed"),
            "hello"
        );
        let remaining = &engine
            .session
            .as_ref()
            .expect("session")
            .pending_plan
            .as_ref()
            .expect("pending")
            .tool_calls;
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].tool_name, "rename_path");

        engine.approve_plan().expect_err("retry still fails");
        assert_eq!(
            std::fs::read_to_string(root.join("notes.txt")).expect("not recreated"),
            "hello"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn explicit_plan_on_and_off_are_idempotent() {
        let root = temp_root("explicit-plan-mode");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        assert_eq!(
            engine.set_plan_mode(true).expect("on"),
            "Plan mode on. Mutating tools will wait for your approval."
        );
        assert_eq!(
            engine.set_plan_mode(true).expect("on again"),
            "Plan mode on. Mutating tools will wait for your approval."
        );
        assert_eq!(engine.set_plan_mode(false).expect("off"), "Plan mode off.");
        assert_eq!(
            engine.set_plan_mode(false).expect("off again"),
            "Plan mode off."
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn invalid_llm_tool_call_becomes_clarification() {
        let root = temp_root("invalid-decision");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = RawDecisionBackend {
            decision: json!({
                "action": "tool_call",
                "tool_name": "shell",
                "arguments": {"command": "rm -rf ."},
                "confidence": 1.0
            }),
        };

        let response = engine
            .run_line("do something", &mut backend)
            .expect("clarification");

        assert!(response.contains("clarify"));
        assert_eq!(
            engine
                .session
                .as_ref()
                .expect("session")
                .events
                .last()
                .expect("event")
                .kind,
            "ask_user"
        );
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

    #[test]
    fn edit_subtitle_updates_file_and_undo_restores_previous_content() {
        let root = temp_root("edit-undo");
        std::fs::create_dir_all(&root).expect("create root");
        let target = root.join("clip.translated.txt");
        std::fs::write(&target, "hello\n").expect("write target");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        let output = engine
            .run_tool(
                "edit_subtitle",
                &json!({"path": "clip.translated.txt", "instruction": "make it uppercase"}),
            )
            .expect("edit subtitle");
        assert!(output.contains("Edited:"));
        assert_eq!(
            std::fs::read_to_string(&target).expect("read edited"),
            "HELLO\n"
        );

        engine.undo_last().expect("undo edit");
        assert_eq!(
            std::fs::read_to_string(&target).expect("read restored"),
            "hello\n"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn diagnose_text_reports_structured_cause() {
        let root = temp_root("diagnose-text");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        let output = engine
            .run_tool(
                "diagnose_text",
                &json!({"text": "missing api key for provider"}),
            )
            .expect("diagnose text");

        assert!(output.contains("credentials"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn switch_profile_updates_active_session_profile() {
        let root = temp_root("switch-profile");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(
            root.join("subbake.toml"),
            "default_profile = \"fast\"\n\
             [defaults]\nprovider = \"mock\"\n\
             [profiles.fast]\nmodel = \"mock-fast\"\n\
             [profiles.strict]\nmodel = \"mock-strict\"\n",
        )
        .expect("write config");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        let output = engine
            .run_tool("switch_profile", &json!({"name": "strict"}))
            .expect("switch profile");

        assert!(output.contains("strict"));
        let session = engine.session.as_ref().expect("session");
        assert_eq!(session.profile.as_deref(), Some("strict"));
        assert_eq!(
            session.config_path.as_deref(),
            Some(root.join("subbake.toml").to_string_lossy().as_ref())
        );
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
