//! Headless agent engine — orchestrates sessions, tools, plan/approval, and undo.
//!
//! Design goals:
//! - Session, Guard, and Engine are separate structs (no 1000-line `_core.py`)
//! - Optional `EngineObserver` for streaming output (TUI, CLI, or test)
//! - Plan mode and approval are explicit state transitions, not side-effect-ridden if/else

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use subbake_core::ports::{ModelToolResult, ToolContinuation};
use subbake_core::{AgentToolOutcome, CancellationGuard, CancellationToken, SharedProgress};

use crate::error::{AgentError, AgentResult};
use crate::event::{EventKind, PendingAgentTurn, PendingCommandApproval, ToolCallDraft};
use crate::guard::FileGuard;
use crate::plan_coordinator::PlanCoordinator;
use crate::presentation::ConversationPresenter;
pub use crate::presentation::{ProfileChoice, SessionChoice};
use crate::profile_coordinator::ProfileCoordinator;
use crate::session::{AgentSessionStore, SessionMode};
use crate::session_controller::SessionController;
use crate::tool_execution::render_tool_outcome;
use crate::tools::{ALL_TOOL_SPECS, ToolKind, find_tool_spec};
use crate::undo::UndoService;
use crate::{ConfigChange, ConfigEditorSnapshot};

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
    fn on_tool_call(&mut self, _call_id: &str, _name: &str, _arguments: &serde_json::Value) {}

    /// A tool completed successfully.
    fn on_tool_success(
        &mut self,
        _call_id: &str,
        _name: &str,
        _arguments: &serde_json::Value,
        outcome: &AgentToolOutcome,
    ) {
        self.on_observation(&render_tool_outcome(outcome));
    }

    /// A tool failed before producing a usable result.
    fn on_tool_failure(
        &mut self,
        _call_id: &str,
        name: &str,
        _arguments: &serde_json::Value,
        error: &str,
    ) {
        self.on_error(&format!("{name}: {error}"));
    }

    /// A running tool was cooperatively cancelled.
    fn on_tool_cancelled(&mut self, _call_id: &str, _name: &str) {}

    /// A tool produced output (observation for the LLM context).
    fn on_observation(&mut self, _text: &str) {}

    /// An error occurred during tool execution.
    fn on_error(&mut self, _error: &str) {}

    /// A final response is ready (respond / ask_user).
    fn on_response(&mut self, _text: &str) {}

    /// The agent loop reached its step limit.
    fn on_step_limit(&mut self) {}
}

/// Observer that prints every engine lifecycle event to stdout.
#[derive(Default)]
pub struct StreamingObserver {
    tool_started: HashMap<String, Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanDecision {
    Approve,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandDecision {
    Approve,
    Reject,
}

/// Returns whether `input` is a complete built-in slash command.
///
/// Inputs that merely start with `/` are ordinary chat text. This keeps
/// absolute paths and pasted shell-style instructions from being rejected as
/// unknown commands.
pub fn is_known_slash_command(input: &str) -> bool {
    let command = input.trim();
    matches!(
        command,
        "/help"
            | "/h"
            | "/plan"
            | "/plan on"
            | "/plan off"
            | "/approve"
            | "/reject"
            | "/undo"
            | "/sessions"
            | "/clear"
            | "/config"
            | "/profile"
            | "/history"
            | "/exit"
            | "/quit"
    ) || command
        .strip_prefix("/profile ")
        .is_some_and(|name| !name.trim().is_empty())
        || command
            .strip_prefix("/sessions ")
            .is_some_and(|id| !id.trim().is_empty())
        || command.starts_with("/history ")
}

impl StreamingObserver {
    pub fn new() -> Self {
        Self::default()
    }
}

impl EngineObserver for StreamingObserver {
    fn on_thinking(&mut self, text: &str) {
        println!("  ⎿  {}…", text.lines().next().unwrap_or(text));
    }

    fn on_tool_call(&mut self, call_id: &str, name: &str, arguments: &serde_json::Value) {
        self.tool_started.insert(call_id.to_owned(), Instant::now());
        let activity = crate::tool_presentation::running_activity(name, arguments);
        println!("  • {}", activity.headline);
        if let Some(detail) = activity.detail {
            println!("    {detail}");
        }
    }

    fn on_tool_success(
        &mut self,
        call_id: &str,
        name: &str,
        arguments: &serde_json::Value,
        outcome: &AgentToolOutcome,
    ) {
        let elapsed = self
            .tool_started
            .remove(call_id)
            .map_or(std::time::Duration::ZERO, |started| started.elapsed());
        let activity =
            crate::tool_presentation::completed_activity(name, arguments, outcome, elapsed);
        println!("  ✓ {}", activity.headline);
        if let Some(detail) = activity.detail {
            println!("    {detail}");
        }
    }

    fn on_tool_failure(
        &mut self,
        call_id: &str,
        name: &str,
        arguments: &serde_json::Value,
        error: &str,
    ) {
        self.tool_started.remove(call_id);
        let activity = crate::tool_presentation::failed_activity(name, arguments, error);
        eprintln!("  × {}", activity.headline);
        if let Some(detail) = activity.detail {
            eprintln!("    {detail}");
        }
    }

    fn on_tool_cancelled(&mut self, call_id: &str, _name: &str) {
        self.tool_started.remove(call_id);
    }

    fn on_error(&mut self, error: &str) {
        eprintln!("  × {error}");
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
    pub(crate) project_root: PathBuf,
    pub(crate) session_store: AgentSessionStore,
    pub(crate) guard: FileGuard,
    pub(crate) session: Option<crate::session::AgentSession>,
    pub(crate) observer: Option<Box<dyn EngineObserver>>,
    cancellation: CancellationToken,
    pub(crate) operation_guard: CancellationGuard,
    pub(crate) progress: Option<SharedProgress>,
    pub(crate) pending_native_continuation: Option<PendingNativeContinuation>,
    pub(crate) runtime_policy: AgentRuntimePolicy,
}

pub(crate) struct PendingNativeContinuation {
    pub(crate) continuation: ToolContinuation,
    pub(crate) results: Vec<ModelToolResult>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentRuntimePolicy {
    max_steps: usize,
    auto_approve_commands: bool,
}

impl AgentRuntimePolicy {
    pub fn new(max_steps: usize, auto_approve_commands: bool) -> AgentResult<Self> {
        if !(1..=128).contains(&max_steps) {
            return Err(AgentError::invalid_input(
                "agent max steps must be from 1 through 128",
            ));
        }
        Ok(Self {
            max_steps,
            auto_approve_commands,
        })
    }

    pub fn max_steps(self) -> usize {
        self.max_steps
    }

    pub fn auto_approve_commands(self) -> bool {
        self.auto_approve_commands
    }
}

impl Default for AgentRuntimePolicy {
    fn default() -> Self {
        Self {
            max_steps: crate::decision::AGENT_LOOP_MAX_STEPS,
            auto_approve_commands: false,
        }
    }
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
            cancellation: CancellationToken::default(),
            operation_guard: CancellationGuard::never(),
            progress: None,
            pending_native_continuation: None,
            runtime_policy: AgentRuntimePolicy::default(),
        }
    }

    pub fn set_runtime_policy(&mut self, policy: AgentRuntimePolicy) {
        self.runtime_policy = policy;
    }

    pub fn with_progress(mut self, progress: SharedProgress) -> Self {
        self.progress = Some(progress);
        self
    }

    pub fn project_root(&self) -> &std::path::Path {
        &self.project_root
    }

    pub fn active_profile(&self) -> Option<&str> {
        self.session
            .as_ref()
            .and_then(|session| session.profile.as_deref())
    }

    pub fn active_config_path(&self) -> Option<&str> {
        self.session
            .as_ref()
            .and_then(|session| session.config_path.as_deref())
    }

    pub fn is_plan_mode(&self) -> bool {
        self.session
            .as_ref()
            .is_some_and(|session| session.mode == SessionMode::Plan)
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    pub fn begin_operation(&mut self, guard: CancellationGuard) {
        self.operation_guard = guard;
    }

    pub(crate) fn check_cancelled(&self) -> AgentResult<()> {
        if self.operation_guard.is_cancelled() {
            Err(AgentError::Cancelled)
        } else {
            Ok(())
        }
    }

    pub fn check_operation_cancelled(&self) -> AgentResult<()> {
        self.check_cancelled()
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
    pub fn start_session(&mut self) -> AgentResult<()> {
        self.pending_native_continuation = None;
        SessionController::new(&self.session_store, &mut self.session).start()
    }

    /// Resume an existing session by id (or the latest if `None`).
    pub fn resume_session(&mut self, id: Option<&str>) -> AgentResult<()> {
        self.pending_native_continuation = None;
        SessionController::new(&self.session_store, &mut self.session).resume(id)
    }

    /// Save the active session to disk.
    pub fn save(&self) -> AgentResult<()> {
        if let Some(session) = self.session.as_ref() {
            self.session_store.save(session)?;
        }
        Ok(())
    }

    /// Pin configuration discovery to the path chosen by the composition
    /// layer so profile listing, model reporting, and backend construction use
    /// the same source of truth.
    pub fn set_config_path(&mut self, path: Option<&std::path::Path>) -> AgentResult<()> {
        SessionController::new(&self.session_store, &mut self.session).set_config_path(path)
    }

    /// Record an event in the active session and persist.
    pub fn record(&mut self, kind: EventKind) -> AgentResult<()> {
        SessionController::new(&self.session_store, &mut self.session).record(kind)
    }

    /// Persist an interactive operation error and return the session log path
    /// that now contains it.
    pub fn record_error(&mut self, error: &str) -> AgentResult<PathBuf> {
        SessionController::new(&self.session_store, &mut self.session).record_error(error)
    }

    // ------------------------------------------------------------------
    // Plan mode
    // ------------------------------------------------------------------

    pub fn store_plan(&mut self, message: &str, tool_calls: Vec<ToolCallDraft>) -> AgentResult<()> {
        let event_calls = tool_calls.clone();
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| std::io::Error::other("no active session"))?;
        PlanCoordinator::store(session, message, tool_calls);
        self.record(EventKind::Plan {
            message: message.to_owned(),
            tool_calls: event_calls,
        })?;
        Ok(())
    }

    pub fn store_command_approval(
        &mut self,
        tool_call: ToolCallDraft,
        call_id: String,
        remaining_tool_calls: Vec<crate::event::PendingToolCall>,
        reason: String,
        turn: PendingAgentTurn,
    ) -> AgentResult<()> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| std::io::Error::other("no active session"))?;
        if session.pending_plan.is_some() || session.pending_command_approval.is_some() {
            return Err(AgentError::invalid_state(
                "another plan or command approval is already pending",
            ));
        }
        session.pending_command_approval = Some(PendingCommandApproval {
            tool_call: tool_call.clone(),
            call_id,
            remaining_tool_calls,
            reason: reason.clone(),
            created_at: crate::session::iso_now(),
        });
        session.pending_agent_turn = Some(turn);
        self.record(EventKind::CommandApprovalRequested { tool_call, reason })
    }

    pub fn has_pending_command_approval(&self) -> bool {
        self.session
            .as_ref()
            .is_some_and(|session| session.pending_command_approval.is_some())
    }

    pub fn pending_command_approval_summary(&self) -> String {
        let Some(pending) = self
            .session
            .as_ref()
            .and_then(|session| session.pending_command_approval.as_ref())
        else {
            return "No command awaiting approval.".to_owned();
        };
        format!(
            "Command awaiting approval ({})\n{}",
            pending.reason, pending.tool_call.arguments
        )
    }

    pub fn approve_plan(&mut self) -> AgentResult<String> {
        let mut outputs = Vec::new();
        loop {
            let call = PlanCoordinator::next_call(
                self.session
                    .as_ref()
                    .ok_or_else(|| std::io::Error::other("no active session"))?,
            )?;
            let Some(call) = call else {
                break;
            };

            let result = self.run_observed_tool(&call.tool_name, &call.arguments)?;
            outputs.push(format!(
                "{}: {}",
                call.tool_name,
                render_tool_outcome(&result)
            ));

            let session = self
                .session
                .as_mut()
                .ok_or_else(|| std::io::Error::other("no active session"))?;
            PlanCoordinator::commit_completed_call(&self.session_store, session)?;
        }

        let session = self
            .session
            .as_mut()
            .ok_or_else(|| std::io::Error::other("no active session"))?;
        PlanCoordinator::finish(session);
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

    pub fn reject_plan(&mut self) -> AgentResult<String> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| std::io::Error::other("no active session"))?;
        PlanCoordinator::finish(session);
        self.record(EventKind::Reject)?;
        Ok("Rejected pending plan.".to_owned())
    }

    pub fn handle_plan_decision(&mut self, decision: PlanDecision) -> AgentResult<String> {
        let result = match decision {
            PlanDecision::Approve => self.approve_plan(),
            PlanDecision::Reject => self.reject_plan(),
        }?;
        self.record_if_active(EventKind::Assistant {
            text: result.clone(),
        })?;
        Ok(result)
    }

    pub fn select_profile(&mut self, name: &str) -> AgentResult<String> {
        let arguments = serde_json::json!({"name": name});
        let result = render_tool_outcome(&self.run_observed_tool("switch_profile", &arguments)?);
        self.record_if_active(EventKind::Assistant {
            text: result.clone(),
        })?;
        Ok(result)
    }

    pub fn toggle_plan_mode(&mut self) -> AgentResult<String> {
        let enabled = self
            .session
            .as_ref()
            .ok_or_else(|| std::io::Error::other("no active session"))?
            .mode
            != SessionMode::Plan;
        self.set_plan_mode(enabled)
    }

    pub fn set_plan_mode(&mut self, enabled: bool) -> AgentResult<String> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| std::io::Error::other("no active session"))?;
        PlanCoordinator::set_mode(&self.session_store, session, enabled)?;
        if !enabled {
            Ok("Plan mode off.".to_owned())
        } else {
            Ok("Plan mode on. Mutating tools will wait for your approval.".to_owned())
        }
    }

    pub fn handle_toggle_plan(&mut self) -> AgentResult<String> {
        self.toggle_plan_mode()
    }

    pub fn session_summary(&self) -> AgentResult<String> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| std::io::Error::other("no active session"))?;
        Ok(ConversationPresenter::session_summary(session))
    }

    pub fn sessions_summary(&self, limit: usize) -> AgentResult<String> {
        ConversationPresenter::sessions_summary(&self.session_store, self.session.as_ref(), limit)
    }

    pub fn session_choices(&self, limit: usize) -> AgentResult<Vec<SessionChoice>> {
        ConversationPresenter::session_choices(&self.session_store, self.session.as_ref(), limit)
    }

    pub fn session_profile(&self, id: &str) -> AgentResult<Option<String>> {
        self.session_store.load(id).map(|session| session.profile)
    }

    pub fn session_config(&self, id: &str) -> AgentResult<(Option<String>, Option<String>)> {
        self.session_store
            .load(id)
            .map(|session| (session.profile, session.config_path))
    }

    pub fn select_session(&mut self, id: &str) -> AgentResult<String> {
        self.resume_session(Some(id))?;
        let result = self.session_summary()?;
        self.record_if_active(EventKind::Assistant {
            text: result.clone(),
        })?;
        Ok(result)
    }

    pub fn history_summary(&self, limit: usize) -> AgentResult<String> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| std::io::Error::other("no active session"))?;
        Ok(ConversationPresenter::history_summary(session, limit))
    }

    pub fn clear_session(&mut self) -> AgentResult<String> {
        let config_path = self
            .session
            .as_ref()
            .and_then(|session| session.config_path.clone());
        self.start_session()?;
        if let Some(session) = self.session.as_mut() {
            session.config_path = config_path;
        }
        self.save()?;
        Ok("Started a new session.".to_owned())
    }

    pub fn has_pending_plan(&self) -> bool {
        self.session
            .as_ref()
            .is_some_and(|session| session.pending_plan.is_some())
    }

    pub fn pending_plan_summary(&self) -> String {
        let plan = self
            .session
            .as_ref()
            .and_then(|session| session.pending_plan.as_ref());
        ConversationPresenter::pending_plan_summary(plan)
    }

    pub fn profile_choices(&self) -> AgentResult<Vec<String>> {
        ProfileCoordinator::new(&self.project_root, self.session.as_ref()).names()
    }

    pub fn profile_picker_choices(&self) -> AgentResult<Vec<ProfileChoice>> {
        ProfileCoordinator::new(&self.project_root, self.session.as_ref()).picker_choices()
    }

    pub fn config_editor_snapshot(&self) -> AgentResult<ConfigEditorSnapshot> {
        ProfileCoordinator::new(&self.project_root, self.session.as_ref()).editor_snapshot()
    }

    pub fn prepare_config_update(
        &self,
        changes: Vec<ConfigChange>,
    ) -> AgentResult<(
        PathBuf,
        subbake_adapters::ConfigEditTarget,
        subbake_adapters::PreparedConfigUpdate,
    )> {
        ProfileCoordinator::new(&self.project_root, self.session.as_ref())
            .prepare_editor_update(changes)
    }

    pub fn conversation_context_summary(&self, limit: usize) -> Option<String> {
        ConversationPresenter::conversation_context(self.session.as_ref()?, limit)
    }

    pub fn input_history(&self) -> Vec<String> {
        ConversationPresenter::input_history(self.session.as_ref())
    }

    pub fn session_events(&self) -> Vec<crate::session::AgentEvent> {
        self.session
            .as_ref()
            .map(|session| session.events.clone())
            .unwrap_or_default()
    }

    pub fn handle_slash_command(&mut self, input: &str) -> AgentResult<String> {
        let trimmed = input.trim();
        let result = match trimmed {
            "/plan" => return self.handle_toggle_plan(),
            "/plan on" => return self.set_plan_mode(true),
            "/plan off" => return self.set_plan_mode(false),
            "/approve" => return self.handle_plan_decision(PlanDecision::Approve),
            "/reject" => return self.handle_plan_decision(PlanDecision::Reject),
            "/undo" => self.undo_last(),
            "/sessions" => self.sessions_summary(20),
            "/clear" => self.clear_session(),
            "/config" => Ok("Opening configuration.".to_owned()),
            "/profile" => self
                .run_tool("list_profiles", &serde_json::json!({}))
                .map(|outcome| render_tool_outcome(&outcome)),
            command if command.starts_with("/profile ") => {
                let name = command.trim_start_matches("/profile ").trim();
                if name.is_empty() {
                    self.run_tool("list_profiles", &serde_json::json!({}))
                        .map(|outcome| render_tool_outcome(&outcome))
                } else {
                    return self.select_profile(name);
                }
            }
            command if command.starts_with("/sessions ") => {
                let id = command.trim_start_matches("/sessions ").trim();
                return self.select_session(id);
            }
            command if command == "/history" || command.starts_with("/history ") => {
                let limit = command
                    .strip_prefix("/history")
                    .unwrap_or_default()
                    .trim()
                    .parse::<usize>()
                    .unwrap_or(20);
                self.history_summary(limit)
            }
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
    pub fn undo_last(&mut self) -> AgentResult<String> {
        let count = UndoService::undo_last(
            &self.project_root,
            &self.session_store,
            self.session
                .as_mut()
                .ok_or_else(|| std::io::Error::other("no active session"))?,
        )?;
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
        find_tool_spec(tool_name).is_some_and(|spec| spec.requires_approval)
    }

    /// Whether a tool is a non-mutating discovery tool.
    pub fn is_discovery_tool(&self, tool_name: &str) -> bool {
        find_tool_spec(tool_name).is_some_and(|spec| spec.discovery)
    }

    /// List tool specs filtered by category for the LLM context.
    pub fn tool_specs_for_llm(&self, categories: &[ToolKind]) -> Vec<&crate::tools::ToolSpec> {
        crate::tools::tool_specs_for_categories(ALL_TOOL_SPECS, categories)
    }
}

#[cfg(test)]
mod error_persistence_tests {
    use super::{AgentEngine, CommandDecision, is_known_slash_command};
    use crate::decision::EchoDecisionBackend;
    use crate::event::{PendingCommandApproval, ToolCallDraft};

    #[test]
    fn only_registered_slash_commands_take_the_command_path() {
        assert!(is_known_slash_command("/plan on"));
        assert!(is_known_slash_command("/profile subtitles"));
        assert!(is_known_slash_command("/config"));
        assert!(!is_known_slash_command("/model"));
        assert!(is_known_slash_command("/history 50"));
        assert!(!is_known_slash_command(
            "/home/azote/Downloads/Braveheart.fixed.translated.srt改为中英双语"
        ));
        assert!(!is_known_slash_command("/plan 改为中英双语"));
    }

    #[test]
    fn command_approval_without_a_v2_agent_turn_is_rejected() {
        let root = std::env::temp_dir().join(format!(
            "subbake-agent-command-approval-{}",
            crate::session::iso_now().replace([':', '.'], "-")
        ));
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        engine
            .session
            .as_mut()
            .expect("session")
            .pending_command_approval = Some(PendingCommandApproval {
            tool_call: ToolCallDraft {
                tool_name: "run_command".to_owned(),
                arguments: serde_json::json!({"command":"printf hello"}),
            },
            call_id: String::new(),
            remaining_tool_calls: Vec::new(),
            reason: "unknown command".to_owned(),
            created_at: crate::session::iso_now(),
        });
        assert!(engine.has_pending_command_approval());
        assert!(!engine.has_pending_plan());

        let mut backend = EchoDecisionBackend::new("test");
        let error = engine
            .handle_command_decision(CommandDecision::Reject, &mut backend)
            .expect_err("incomplete approval state must be rejected");
        assert!(
            error
                .to_string()
                .contains("missing its persisted agent turn")
        );
        assert!(engine.has_pending_command_approval());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn interactive_errors_are_persisted_in_the_active_session() {
        let root = std::env::temp_dir().join(format!(
            "subbake-agent-error-{}",
            crate::session::iso_now().replace([':', '.'], "-")
        ));
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        let path = engine
            .record_error("provider request failed")
            .expect("persist error");
        let saved = std::fs::read_to_string(&path).expect("read session log");

        assert!(path.starts_with(root.join(".subbake/agent/sessions")));
        assert!(saved.contains("\"kind\": \"error\""));
        assert!(saved.contains("provider request failed"));
        std::fs::remove_dir_all(root).expect("remove test root");
    }
}
