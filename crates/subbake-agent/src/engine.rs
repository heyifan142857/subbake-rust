//! Headless agent engine — orchestrates sessions, tools, plan/approval, and undo.
//!
//! Design goals:
//! - Session, Guard, and Engine are separate structs (no 1000-line `_core.py`)
//! - Optional `EngineObserver` for streaming output (TUI, CLI, or test)
//! - Plan mode and approval are explicit state transitions, not side-effect-ridden if/else

use std::path::PathBuf;

use crate::event::{EventKind, PendingPlan, ToolCallDraft};
use crate::guard::FileGuard;
use crate::session::AgentSessionStore;
use crate::tools::{ALL_TOOL_SPECS, APPROVAL_REQUIRED_TOOL_NAMES, DISCOVERY_TOOL_NAMES, ToolKind};

// ---------------------------------------------------------------------------
// Observer trait — enables streaming output
// ---------------------------------------------------------------------------

/// Subscribe to engine lifecycle events for streaming display.
///
/// Every method has a default no-op implementation so observers only override
/// what they care about.
pub trait EngineObserver: Send {
    /// The LLM is "thinking" (producing reasoning text).
    fn on_thinking(&mut self, _text: &str) {}

    /// A tool is about to be called.
    fn on_tool_call(&mut self, _name: &str, _arguments: &serde_json::Value) {}

    /// A tool produced output (observation for the LLM context).
    fn on_observation(&mut self, _text: &str) {}

    /// An error occurred during tool execution.
    fn on_error(&mut self, _error: &str) {}

    /// A final response is ready (respond / ask_user).
    fn on_response(&mut self, _text: &str) {}

    /// The agent loop reached its step limit.
    fn on_step_limit(&mut self) {}
}

/// Observer that prints everything to stdout (mirrors Python `trace._AgentLoopTrace`).
pub struct StreamingObserver;

impl Default for StreamingObserver {
    fn default() -> Self {
        Self
    }
}

impl StreamingObserver {
    pub fn new() -> Self {
        Self
    }
}

impl EngineObserver for StreamingObserver {
    fn on_thinking(&mut self, text: &str) {
        println!("  ⎿  {}…", text.lines().next().unwrap_or(text));
    }

    fn on_tool_call(&mut self, name: &str, arguments: &serde_json::Value) {
        let args = serde_json::to_string(arguments).unwrap_or_default();
        if args.len() > 120 {
            println!("  ⚡ {name}  ({args:.120}…)");
        } else {
            println!("  ⚡ {name}  {args}");
        }
    }

    fn on_observation(&mut self, text: &str) {
        let preview = text.lines().next().unwrap_or(text);
        if preview.len() > 200 {
            println!("  ◀  {:.200}…", preview);
        } else {
            println!("  ◀  {preview}");
        }
    }

    fn on_error(&mut self, error: &str) {
        eprintln!("  ✖ {error}");
    }

    fn on_response(&mut self, text: &str) {
        println!("  ➔ {text}");
    }

    fn on_step_limit(&mut self) {
        println!("  ⚠ Agent loop reached step limit.");
    }
}

/// The headless agent engine.
pub struct AgentEngine {
    pub project_root: PathBuf,
    pub session_store: AgentSessionStore,
    pub guard: FileGuard,
    pub session: Option<crate::session::AgentSession>,
    pub observer: Option<Box<dyn EngineObserver>>,
}

impl AgentEngine {
    pub fn new(project_root: PathBuf) -> Self {
        let session_store = AgentSessionStore::new(project_root.clone());
        let guard = FileGuard::new(project_root.clone());
        Self {
            project_root,
            session_store,
            guard,
            session: None,
            observer: None,
        }
    }

    /// Attach an observer for streaming output.
    pub fn with_observer(mut self, observer: Box<dyn EngineObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    // ------------------------------------------------------------------
    // Session lifecycle
    // ------------------------------------------------------------------

    /// Create a new session and mark it active.
    pub fn start_session(&mut self) -> std::io::Result<()> {
        let session = self.session_store.create()?;
        self.session = Some(session);
        Ok(())
    }

    /// Resume an existing session by id (or the latest if `None`).
    pub fn resume_session(&mut self, id: Option<&str>) -> std::io::Result<()> {
        let session = match id {
            Some(sid) => self.session_store.load(sid)?,
            None => self
                .session_store
                .latest()?
                .ok_or_else(|| std::io::Error::other("no sessions to resume"))?,
        };
        self.session = Some(session);
        Ok(())
    }

    /// Save the active session to disk.
    pub fn save(&self) -> std::io::Result<()> {
        if let Some(ref session) = self.session {
            self.session_store.save(session)?;
        }
        Ok(())
    }

    /// Record an event in the active session and persist.
    pub fn record(&mut self, kind: EventKind) -> std::io::Result<()> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| std::io::Error::other("no active session"))?;

        let now = crate::session::iso_now();
        let (kind_str, text, data) = serialize_event(&kind);
        session.events.push(crate::session::AgentEvent {
            kind: kind_str,
            text,
            data,
            created_at: now,
        });
        session.updated_at = crate::session::iso_now();
        self.session_store.save(session)?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Plan mode
    // ------------------------------------------------------------------

    pub fn store_plan(
        &mut self,
        message: &str,
        tool_calls: Vec<ToolCallDraft>,
    ) -> std::io::Result<()> {
        let event_calls = tool_calls.clone();
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| std::io::Error::other("no active session"))?;
        session.mode = "plan".to_owned();
        session.pending_plan = Some(PendingPlan {
            message: message.to_owned(),
            tool_calls,
            created_at: crate::session::iso_now(),
        });
        self.session_store.save(session)?;
        self.record(EventKind::Plan {
            message: message.to_owned(),
            tool_calls: event_calls,
        })?;
        Ok(())
    }

    pub fn approve_plan(&mut self) -> std::io::Result<String> {
        let pending = self
            .session
            .as_ref()
            .and_then(|session| session.pending_plan.clone())
            .ok_or_else(|| std::io::Error::other("no pending plan to approve"))?;

        let mut outputs = Vec::new();
        for call in &pending.tool_calls {
            let result = self.run_tool(&call.tool_name, &call.arguments)?;
            outputs.push(format!("{}: {}", call.tool_name, result));
        }

        let session = self
            .session
            .as_mut()
            .ok_or_else(|| std::io::Error::other("no active session"))?;
        session.mode = "chat".to_owned();
        session.pending_plan = None;
        self.record(EventKind::Approve)?;

        if outputs.is_empty() {
            Ok("Approved an empty plan.".to_owned())
        } else {
            Ok(format!(
                "Approved and executed plan.\n{}",
                outputs.join("\n")
            ))
        }
    }

    pub fn reject_plan(&mut self) -> std::io::Result<String> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| std::io::Error::other("no active session"))?;
        session.mode = "chat".to_owned();
        session.pending_plan = None;
        self.record(EventKind::Reject)?;
        Ok("Rejected pending plan.".to_owned())
    }

    pub fn toggle_plan_mode(&mut self) -> std::io::Result<String> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| std::io::Error::other("no active session"))?;
        if session.mode == "plan" {
            session.mode = "chat".to_owned();
            session.pending_plan = None;
            self.session_store.save(session)?;
            Ok("Plan mode off.".to_owned())
        } else {
            session.mode = "plan".to_owned();
            self.session_store.save(session)?;
            Ok("Plan mode on. Mutating tools will wait for `/approve`.".to_owned())
        }
    }

    pub fn session_summary(&self) -> std::io::Result<String> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| std::io::Error::other("no active session"))?;
        let pending = if session.pending_plan.is_some() {
            "pending plan"
        } else {
            "no pending plan"
        };
        Ok(format!(
            "Session: {}\nMode: {}\nEvents: {}\n{}",
            session.id,
            session.mode,
            session.events.len(),
            pending
        ))
    }

    pub fn conversation_context_summary(&self, limit: usize) -> Option<String> {
        let session = self.session.as_ref()?;
        let mut lines = session
            .events
            .iter()
            .rev()
            .skip_while(|event| event.kind == "user")
            .filter_map(|event| {
                let label = match event.kind.as_str() {
                    "user" => "User",
                    "assistant" => "Assistant",
                    "ask_user" => "Assistant question",
                    "tool_call" => "Tool",
                    "file_operation" => "File operation",
                    "plan" => "Plan",
                    "error" => "Error",
                    _ => return None,
                };
                let text = if event.text.trim().is_empty() {
                    event.data.to_string()
                } else {
                    event.text.clone()
                };
                Some(format!("{label}: {}", truncate_summary(&text, 240)))
            })
            .take(limit)
            .collect::<Vec<_>>();
        lines.reverse();
        (!lines.is_empty()).then(|| lines.join("\n"))
    }

    pub fn handle_slash_command(&mut self, input: &str) -> std::io::Result<String> {
        let result = match input.trim() {
            "/plan" => self.toggle_plan_mode(),
            "/approve" => self.approve_plan(),
            "/reject" => self.reject_plan(),
            "/undo" => self.undo_last(),
            "/session" => self.session_summary(),
            other => Ok(format!("Unknown command `{other}`. Try /help.")),
        }?;
        self.record_if_active(EventKind::Assistant {
            text: result.clone(),
        })?;
        Ok(result)
    }

    // ------------------------------------------------------------------
    // Undo
    // ------------------------------------------------------------------

    /// Undo the last file_operation event (or group of events sharing a group_id).
    pub fn undo_last(&mut self) -> std::io::Result<String> {
        let events = {
            let session = self
                .session
                .as_ref()
                .ok_or_else(|| std::io::Error::other("no active session"))?;
            session.events.clone()
        };

        // Find the latest non-undone file_operation event.
        let target = events
            .iter()
            .rev()
            .find(|event| {
                event.kind == "file_operation"
                    && !event
                        .data
                        .get("undone")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
            })
            .cloned()
            .ok_or_else(|| std::io::Error::other("nothing to undo"))?;

        let group_id = target
            .data
            .get("group_id")
            .and_then(|v| v.as_str())
            .map(String::from);

        // Collect all events in this undo group.
        let targets: Vec<_> = if let Some(ref gid) = group_id {
            events
                .iter()
                .filter(|e| {
                    e.kind == "file_operation"
                        && e.data.get("group_id").and_then(|v| v.as_str()) == Some(gid.as_str())
                        && !e
                            .data
                            .get("undone")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                })
                .cloned()
                .collect()
        } else {
            vec![target.clone()]
        };

        let mut count = 0usize;
        for event in &targets {
            let action = event
                .data
                .get("action")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let path = event
                .data
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let backup = event.data.get("backup_path").and_then(|v| v.as_str());

            let target_path = self.project_root.join(path);

            match action {
                "created" => {
                    // Remove the created file/directory.
                    let _ = std::fs::remove_file(&target_path);
                    let _ = std::fs::remove_dir_all(&target_path);
                }
                "renamed" => {
                    if let Some(new_path) = event.data.get("new_path").and_then(|v| v.as_str()) {
                        let moved_path = self.project_root.join(new_path);
                        let _ = std::fs::remove_file(&moved_path);
                        let _ = std::fs::remove_dir_all(&moved_path);
                    }
                    if let Some(bp) = backup {
                        FileGuard::restore_backup(PathBuf::from(bp).as_path(), &target_path)?;
                    }
                }
                "deleted" | "modified" | "appended" => {
                    if let Some(bp) = backup {
                        FileGuard::restore_backup(PathBuf::from(bp).as_path(), &target_path)?;
                    }
                }
                _ => {}
            }

            // Mark as undone.
            if let Some(session) = self.session.as_mut() {
                for se in session.events.iter_mut().rev() {
                    if se.created_at == event.created_at && se.kind == "file_operation" {
                        if let Some(obj) = se.data.as_object_mut() {
                            obj.insert("undone".to_owned(), serde_json::Value::Bool(true));
                        }
                        break;
                    }
                }
            }
            count += 1;
        }

        self.save()?;
        self.record(EventKind::Undo)?;

        if count > 1 {
            Ok(format!("Undone {count} operations (series)."))
        } else {
            Ok("Undone 1 operation.".to_string())
        }
    }

    // ------------------------------------------------------------------
    // Tool dispatch helpers
    // ------------------------------------------------------------------

    /// Whether a tool requires explicit approval (plan mode or approval tool).
    pub fn tool_requires_approval(&self, tool_name: &str) -> bool {
        APPROVAL_REQUIRED_TOOL_NAMES.contains(&tool_name)
    }

    /// Whether a tool is a non-mutating discovery tool.
    pub fn is_discovery_tool(&self, tool_name: &str) -> bool {
        DISCOVERY_TOOL_NAMES.contains(&tool_name)
    }

    /// List tool specs filtered by category for the LLM context.
    pub fn tool_specs_for_llm(&self, categories: &[ToolKind]) -> Vec<&crate::tools::ToolSpec> {
        crate::tools::tool_specs_for_categories(ALL_TOOL_SPECS, categories)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn serialize_event(kind: &EventKind) -> (String, String, serde_json::Value) {
    match kind {
        EventKind::User { text } => ("user".into(), text.clone(), serde_json::json!({})),
        EventKind::Assistant { text } => ("assistant".into(), text.clone(), serde_json::json!({})),
        EventKind::AskUser { text } => ("ask_user".into(), text.clone(), serde_json::json!({})),
        EventKind::ToolCall {
            tool_name,
            arguments,
        } => (
            "tool_call".into(),
            tool_name.clone(),
            serde_json::json!({"tool_name": tool_name, "arguments": arguments}),
        ),
        EventKind::FinalToolCall {
            tool_name,
            arguments,
        } => (
            "final_tool_call".into(),
            tool_name.clone(),
            serde_json::json!({"tool_name": tool_name, "arguments": arguments}),
        ),
        EventKind::FileOperation(data) => (
            "file_operation".into(),
            format!("{} {}", data.action, data.path),
            serde_json::to_value(data).unwrap_or_default(),
        ),
        EventKind::Plan {
            message,
            tool_calls,
        } => (
            "plan".into(),
            message.clone(),
            serde_json::json!({"message": message, "tool_calls": tool_calls}),
        ),
        EventKind::Approve => ("approve".into(), String::new(), serde_json::json!({})),
        EventKind::Reject => ("reject".into(), String::new(), serde_json::json!({})),
        EventKind::Undo => ("undo".into(), String::new(), serde_json::json!({})),
        EventKind::Profile { name } => ("profile".into(), name.clone(), serde_json::json!({})),
        EventKind::Error { text } => ("error".into(), text.clone(), serde_json::json!({})),
    }
}

fn truncate_summary(text: &str, limit: usize) -> String {
    let value = text.chars().take(limit).collect::<String>();
    if text.chars().count() > limit {
        format!("{value}...")
    } else {
        value
    }
}
