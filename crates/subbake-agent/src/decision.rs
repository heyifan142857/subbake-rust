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

use serde_json::{json, Value as JsonValue};
use subbake_adapters::{
    TranscriptionRequest, TranscriptionSettings, TranslationRequest, TranslationSettings,
    WhisperAction, WhisperRequest,
    transcribe_media, translate_subtitle,
};
use subbake_core::ports::{ChatMessage, LlmBackend};

use crate::engine::AgentEngine;
use crate::tools::ALL_TOOL_SPECS;

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
    action: String,      // "respond" | "tool_call" | "ask_user"
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
        // 1. Quick-path: keyword matching without LLM.
        if let Some(result) = self.try_quick_path(input)? {
            if let Some(ref text) = result.response_text
                && let Some(ref mut obs) = self.observer {
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
                    if self.tool_requires_approval(tool_name)
                        || self.is_in_plan_mode()
                    {
                        // Store as plan for later approval.
                        let draft = crate::event::ToolCallDraft {
                            tool_name: tool_name.to_owned(),
                            arguments: args.clone(),
                        };
                        self.store_plan("", vec![draft])?;
                        let msg = "I've prepared a plan for your approval. Use `/approve` to execute.".to_owned();
                        if let Some(ref mut obs) = self.observer {
                            obs.on_tool_call(tool_name, &args);
                            obs.on_response(&msg);
                        }
                        return Ok(msg);
                    }

                    if let Some(ref mut obs) = self.observer {
                        obs.on_tool_call(tool_name, &args);
                    }
                    let result_text = self.run_tool(tool_name, &args)?;
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

    fn try_quick_path(&self, input: &str) -> io::Result<Option<QuickResult>> {
        let trimmed = input.trim();

        // Pattern: "translate @<path>" or "translate <path>"
        if let Some(path) = trimmed.strip_prefix("translate ") {
            let p = self.resolve_tool_path(path);
            return Ok(Some(QuickResult {
                output: format!("Translate {}", p.display()),
                response_text: Some(format!("Translate {}", p.display())),
            }));
        }

        // Pattern: "transcribe @<path>"
        if let Some(path) = trimmed.strip_prefix("transcribe ") {
            let p = self.resolve_tool_path(path);
            return Ok(Some(QuickResult {
                output: format!("Transcribe {}", p.display()),
                response_text: Some(format!("Transcribing {}", p.display())),
            }));
        }

        // Pattern: "list files" or "ls"
        if matches!(trimmed, "list files" | "ls" | "list") {
            return Ok(Some(QuickResult {
                output: self.guard.list_files(std::path::Path::new("."))
                    .map(|files| files.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n"))
                    .unwrap_or_default(),
                response_text: None,
            }));
        }

        Ok(None)
    }

    fn resolve_tool_path(&self, input: &str) -> std::path::PathBuf {
        // Strip @ prefix if present.
        let cleaned = input.trim().trim_start_matches('@');
        self.project_root.join(cleaned)
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
        let result = backend.generate_json(&messages);
        match result {
            Ok(backend_result) => {
                let subbake_core::ports::BackendPayload::Translation(translation) = backend_result.payload;
                let text = translation.lines.first().map(|l| l.translation.clone()).unwrap_or_default();
                self.parse_decision(&text)
            }
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
        system.push_str("Keep confidence high (≥0.85) for direct tool calls, lower for clarification.\n");
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

        vec![
            ChatMessage::system(&system),
            ChatMessage::user(&user),
        ]
    }

    /// Parse the LLM's JSON response into a structured Decision.
    fn parse_decision(&self, text: &str) -> io::Result<Decision> {
        let trimmed = text.trim();
        let json_start = trimmed.find('{').unwrap_or(0);
        let json_str = &trimmed[json_start..];

        let parsed: JsonValue = serde_json::from_str(json_str)
            .unwrap_or_else(|_| json!({"action": "respond", "text": text}));

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
                text: format!("Shall I {} with {:?}?", decision.tool_name.as_deref().unwrap_or("?"), decision.arguments),
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
        self.session
            .as_ref()
            .is_some_and(|s| s.mode == "plan")
    }

    // ------------------------------------------------------------------
    // Tool runner (stub — dispatches to real adapters)
    // ------------------------------------------------------------------

    /// Execute a tool by name with arguments. Returns a text summary.
    fn run_tool(&self, name: &str, args: &JsonValue) -> io::Result<String> {
        match name {
            // -- Browse (FileGuard) --
            "list_files" => {
                let dir = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                let files = self.guard.list_files(PathBuf::from(dir).as_path())?;
                Ok(files.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n"))
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
                let files = self.guard.search_files(PathBuf::from(dir).as_path(), ".srt")?;
                Ok(format_file_list(&files))
            }
            "recent_translations" => {
                let session = self.session.as_ref();
                let events = session.map(|s| &s.events).map(|v| v.as_slice()).unwrap_or(&[]);
                let mut out = Vec::new();
                for event in events.iter().rev().take(20) {
                    if event.kind == "translate_file" || event.kind == "final_tool_call" {
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
                Ok(format!("Created {}", r.path.display()))
            }
            "append_file" => {
                let path = req_arg(args, "path")?;
                let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let r = self.guard.append_file(&path, content)?;
                Ok(format!("Appended {} (backup: {})", r.path.display(),
                    r.backup_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default()))
            }
            "replace_in_file" => {
                let path = req_arg(args, "path")?;
                let old = args.get("old").and_then(|v| v.as_str()).unwrap_or("");
                let new = args.get("new").and_then(|v| v.as_str()).unwrap_or("");
                let r = self.guard.replace_in_file(&path, old, new)?;
                Ok(format!("Replaced in {} (backup: {})", r.path.display(),
                    r.backup_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default()))
            }
            "rename_path" => {
                let from = req_arg(args, "from")?;
                let to = req_arg(args, "to")?;
                let r = self.guard.rename_path(&from, &to)?;
                Ok(format!("Renamed {} → {}", r.path.display(),
                    r.new_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default()))
            }
            "delete_file" => {
                let path = req_arg(args, "path")?;
                let r = self.guard.delete_file(&path)?;
                Ok(format!("Deleted {} (backup: {})", r.path.display(),
                    r.backup_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default()))
            }

            // -- Translate --
            "translate_file" => {
                let path = req_arg(args, "path")?;
                let input = self.project_root.join(&path);
                let request = TranslationRequest {
                    input_path: input,
                    output_path: None,
                    settings: TranslationSettings::default(),
                };
                let outcome = translate_subtitle(request)?;
                Ok(outcome.output_path.map(|p| format!("Translated: {}", p.display())).unwrap_or_default())
            }
            "translate_series" => {
                let dir = req_arg(args, "path")?;
                let input = self.project_root.join(&dir);
                let request = subbake_adapters::BatchTranslationRequest {
                    root: input,
                    recursive: args.get("recursive").and_then(|v| v.as_bool()).unwrap_or(false),
                    overwrite: args.get("overwrite").and_then(|v| v.as_bool()).unwrap_or(false),
                    settings: TranslationSettings::default(),
                };
                let outcome = subbake_adapters::translate_subtitle_batch(request)?;
                Ok(format!("Translated {} files, skipped {}.", outcome.processed, outcome.skipped.len()))
            }

            // -- Edit (requires editing pipeline) --
            "edit_subtitle" => {
                Ok("Edit tool: point me at a file and I'll re-translate specific lines.".to_owned())
            }

            // -- Transcribe --
            "transcribe_audio" => {
                let path = req_arg(args, "path")?;
                let input = self.project_root.join(&path);
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
                let action_str = args.get("action").and_then(|v| v.as_str()).unwrap_or("status");
                let action = match action_str {
                    "install" => WhisperAction::Install,
                    "status" => WhisperAction::Status,
                    "list-models" | "models" => WhisperAction::ListModels,
                    "download" => {
                        let name = args.get("model").and_then(|v| v.as_str()).unwrap_or("small");
                        WhisperAction::DownloadModel { name: name.to_owned() }
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

            // -- Diagnose (offline log reader — stub for now) --
            "diagnose_path" | "diagnose_text" => {
                Ok(format!("[{name}: not yet implemented. Translate failed runs produce log files I can read.]"))
            }

            // -- Profile (stub) --
            "switch_profile" | "list_profiles" => {
                Ok(format!("[{name}: configure profiles in subbake.toml]"))
            }

            _ => Ok(format!("[{name}: not yet wired]")),
        }
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
    files.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n")
}

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct QuickResult {
    output: String,
    response_text: Option<String>,
}
