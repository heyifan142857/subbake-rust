//! Chat‑style TUI: scrollable output pane above, fixed input bar below.
//!
//! Layout (inspired by Codex / OpenCode):
//!
//! ┌─────────────────────────────────┐
//! │  Agent Output (scrollable)      │
//! │  [You] translate hello.srt      │
//! │  ⚡ translate_file ✓            │
//! │  ➔ Translated: out.srt         │
//! │  ...                            │
//! ├─────────────────────────────────┤
//! │ > _                             │
//! └─────────────────────────────────┘

use std::io;
use std::sync::mpsc;
use std::thread;

use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};
use unicode_width::UnicodeWidthStr;

use crate::engine::{EngineObserver, ProfileChoice, SessionChoice};
use crate::session::iso_now;
use subbake_core::{CancellationGuard, CancellationToken};

// ---------------------------------------------------------------------------
// Message types for the scrollback buffer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Msg {
    pub style: MsgStyle,
    pub text: String,
    pub stamp: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgStyle {
    User,
    Thinking,
    ToolCall,
    Observation,
    Response,
    Error,
    System,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartupInfo {
    pub provider: String,
    pub model: String,
    pub config: String,
    pub cache_enabled: bool,
    pub cwd: String,
}

impl Default for StartupInfo {
    fn default() -> Self {
        Self {
            provider: "mock".to_owned(),
            model: "mock-zh".to_owned(),
            config: "Not configured".to_owned(),
            cache_enabled: true,
            cwd: String::new(),
        }
    }
}

const STREAM_INTERVAL: std::time::Duration = std::time::Duration::from_millis(12);
const THINKING_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/help", "show commands"),
    ("/plan", "toggle plan mode; accepts on/off"),
    ("/model", "show the active model"),
    ("/profile", "list or switch profiles"),
    ("/undo", "undo the last file operation"),
    ("/sessions", "choose a saved session"),
    ("/history", "show conversation history"),
    ("/clear", "start a new session"),
    ("/exit", "exit SubBake"),
    ("/quit", "exit SubBake"),
];

const APPROVAL_OPTIONS: &[(&str, &str)] = &[
    ("approve", "execute the pending plan"),
    ("reject", "discard the pending plan"),
    (
        "tell agent what to do",
        "revise the plan with your instructions",
    ),
];
struct TuiPicker {
    pub options: Vec<ProfileChoice>,
}

struct SessionPicker {
    pub options: Vec<SessionChoice>,
}

pub enum TuiInteraction {
    Message {
        message: String,
        render: RenderPolicy,
    },
    PlanApproval {
        message: String,
    },
    ProfilePicker {
        message: String,
        options: Vec<ProfileChoice>,
    },
    SessionChanged {
        input_history: Vec<String>,
        events: Vec<crate::session::AgentEvent>,
        plan_mode: bool,
        model: String,
    },
    SessionPicker {
        message: String,
        options: Vec<SessionChoice>,
    },
    PlanModeChanged {
        enabled: bool,
    },
    ModelChanged {
        model: String,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPolicy {
    Stream,
    Immediate,
}

enum InputMode {
    Editing,
    BrowsingHistory { index: usize, draft: String },
    ChoosingProfile(TuiPicker),
    CreatingProfile,
    ChoosingSession(SessionPicker),
    AwaitingPlanDecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerticalNavigation {
    Selection(usize),
    History,
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuiAction {
    SubmitText(String),
    ApprovePlan,
    RejectPlan,
    SelectProfile(String),
    CreateProfile(String),
    SelectSession(String),
    TogglePlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ApprovalChoice {
    Submit(TuiAction),
    Revise,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ProfilePickerChoice {
    Select(String),
    Create,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EmptyModeChoice {
    Submit(TuiAction),
    RevisePlan,
    CreateProfile,
}

impl TuiAction {
    fn visible_text(&self) -> String {
        match self {
            Self::SubmitText(text) => text.clone(),
            Self::ApprovePlan => "approve".to_owned(),
            Self::RejectPlan => "reject".to_owned(),
            Self::SelectProfile(name) => format!("/profile {name}"),
            Self::CreateProfile(name) => format!("create profile {name}"),
            Self::SelectSession(id) => format!("/sessions {id}"),
            Self::TogglePlan => "toggle plan mode".to_owned(),
        }
    }
}

struct StreamingResponse {
    chars: Vec<char>,
    position: usize,
    message_index: usize,
    next_at: std::time::Instant,
}

struct TerminalSessionGuard {
    active: bool,
}

impl TerminalSessionGuard {
    fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        if let Err(error) = io::stdout().execute(EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(error);
        }
        Ok(Self { active: true })
    }

    fn restore(&mut self) -> io::Result<()> {
        if !self.active {
            return Ok(());
        }
        self.active = false;
        let screen_result = io::stdout().execute(LeaveAlternateScreen).map(|_| ());
        let raw_result = disable_raw_mode();
        screen_result.and(raw_result)
    }
}

impl Drop for TerminalSessionGuard {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

/// Bounded scrollback buffer that also serves as the `EngineObserver`.
#[derive(Debug, Clone)]
pub struct MsgView {
    messages: Vec<Msg>,
    max: usize,
}

impl MsgView {
    pub fn new(max: usize) -> Self {
        Self {
            messages: Vec::with_capacity(max),
            max,
        }
    }

    pub fn push(&mut self, style: MsgStyle, text: String) {
        let stamp = iso_now();
        if self.messages.len() >= self.max {
            self.messages.remove(0);
        }
        self.messages.push(Msg { style, text, stamp });
    }

    pub fn all(&self) -> &[Msg] {
        &self.messages
    }
}

/// Wraps a `MsgView` so the agent engine can push into it via the observer trait.
pub struct TuiObserver {
    pub view: std::sync::Arc<std::sync::Mutex<MsgView>>,
}

impl TuiObserver {
    pub fn new(view: std::sync::Arc<std::sync::Mutex<MsgView>>) -> Self {
        Self { view }
    }
}

impl EngineObserver for TuiObserver {
    fn on_thinking(&mut self, text: &str) {
        if let Ok(mut v) = self.view.lock()
            && !v
                .messages
                .last()
                .is_some_and(|message| message.style == MsgStyle::Thinking && message.text == text)
        {
            v.push(MsgStyle::Thinking, text.to_owned());
        }
    }

    fn on_tool_call(&mut self, name: &str, arguments: &serde_json::Value) {
        if let Ok(mut v) = self.view.lock() {
            let args = serde_json::to_string(arguments).unwrap_or_default();
            v.push(MsgStyle::ToolCall, format!("⚡ {name} {args}"));
        }
    }

    fn on_observation(&mut self, text: &str) {
        if let Ok(mut v) = self.view.lock() {
            v.push(MsgStyle::Observation, format!("◀ {text}"));
        }
    }

    fn on_error(&mut self, error: &str) {
        if let Ok(mut v) = self.view.lock() {
            v.push(MsgStyle::Error, format!("✖ {error}"));
        }
    }

    fn on_response(&mut self, text: &str) {
        // Final responses are returned to `SubBakeTui::run`, which animates
        // them. Pushing here as well would display every answer twice.
        let _ = text;
    }
}

// ---------------------------------------------------------------------------
// TUI App
// ---------------------------------------------------------------------------

pub struct SubBakeTui {
    terminal_session: TerminalSessionGuard,
    terminal: Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    msg_view: std::sync::Arc<std::sync::Mutex<MsgView>>,
    input: String,
    input_history: Vec<String>,
    input_mode: InputMode,
    scroll_offset: u16,
    running: bool,
    streaming: Option<StreamingResponse>,
    suggestion_index: usize,
    processing: bool,
    animation_started_at: std::time::Instant,
    cancellation: Option<CancellationToken>,
    cancellation_requested: bool,
    input_hint: &'static str,
    startup_info: StartupInfo,
    plan_mode: bool,
    transcript_page_lines: u16,
}

impl SubBakeTui {
    pub fn new() -> io::Result<Self> {
        let terminal_session = TerminalSessionGuard::enter()?;
        let backend = ratatui::backend::CrosstermBackend::new(io::stdout());
        let terminal = Terminal::new(backend)?;
        Ok(Self {
            terminal_session,
            terminal,
            msg_view: std::sync::Arc::new(std::sync::Mutex::new(MsgView::new(1000))),
            input: String::new(),
            input_history: Vec::new(),
            input_mode: InputMode::Editing,
            scroll_offset: 0,
            running: true,
            streaming: None,
            suggestion_index: 0,
            processing: false,
            animation_started_at: std::time::Instant::now(),
            cancellation: None,
            cancellation_requested: false,
            input_hint: session_input_hint(),
            startup_info: StartupInfo::default(),
            plan_mode: false,
            transcript_page_lines: 10,
        })
    }

    pub fn set_startup_info(&mut self, startup_info: StartupInfo) {
        self.startup_info = startup_info;
    }

    pub fn set_plan_mode(&mut self, enabled: bool) {
        self.plan_mode = enabled;
    }

    pub fn set_has_config_file(&mut self, has_config_file: bool) {
        if !has_config_file {
            self.input_hint = "Use /profile to create a model profile";
        }
    }

    pub fn observer(&self) -> TuiObserver {
        TuiObserver::new(self.msg_view.clone())
    }

    pub fn set_cancellation_token(&mut self, token: CancellationToken) {
        self.cancellation = Some(token);
    }

    pub fn set_input_history(&mut self, history: Vec<String>) {
        self.input_history = history;
        self.input_mode = InputMode::Editing;
    }

    pub fn set_session_replay(&mut self, events: Vec<crate::session::AgentEvent>) {
        self.finish_stream();
        if let Ok(mut view) = self.msg_view.lock() {
            view.messages.clear();
            for event in events {
                let (style, text) = match event.kind.as_str() {
                    "user" => (
                        MsgStyle::User,
                        format!("[{}] {}", event.created_at, event.text),
                    ),
                    "assistant" | "ask_user" => (MsgStyle::Response, format!("➔ {}", event.text)),
                    "tool_call" => (MsgStyle::ToolCall, format!("⚡ {}", event.text)),
                    "file_operation" => (MsgStyle::Observation, format!("◀ {}", event.text)),
                    "plan" => (MsgStyle::System, format!("Plan: {}", event.text)),
                    "error" => (MsgStyle::Error, format!("✖ {}", event.text)),
                    "cancelled" => (MsgStyle::System, "Cancelled.".to_owned()),
                    _ => continue,
                };
                if view.messages.len() >= view.max {
                    view.messages.remove(0);
                }
                view.messages.push(Msg {
                    style,
                    text,
                    stamp: event.created_at,
                });
            }
        }
        self.scroll_offset = 0;
    }

    /// Show the same resume picker used by the `/sessions` command on startup.
    pub fn open_session_picker(&mut self, options: Vec<SessionChoice>) {
        self.input.clear();
        self.input_mode = InputMode::ChoosingSession(SessionPicker { options });
        self.suggestion_index = 0;
    }

    /// Run the event loop. `process_fn` is called with the user's input each
    /// time they press Enter; it should run the agent engine and return the
    /// response text.
    pub fn run<F>(&mut self, mut process_fn: F) -> io::Result<()>
    where
        F: FnMut(TuiAction, CancellationGuard, &mut TuiObserver) -> io::Result<TuiInteraction>
            + Send
            + 'static,
    {
        let worker_observer = self.observer();
        let (request_tx, request_rx) = mpsc::channel::<(TuiAction, CancellationGuard)>();
        let (response_tx, response_rx) = mpsc::channel::<io::Result<TuiInteraction>>();
        let worker = thread::Builder::new()
            .name("subbake-agent-worker".to_owned())
            .spawn(move || {
                let mut observer = worker_observer;
                while let Ok((action, guard)) = request_rx.recv() {
                    let result = process_fn(action, guard, &mut observer);
                    if response_tx.send(result).is_err() {
                        break;
                    }
                }
            })?;

        let loop_result = (|| -> io::Result<()> {
            while self.running {
                if let Ok(result) = response_rx.try_recv() {
                    self.processing = false;
                    self.cancellation_requested = false;
                    match result {
                        Ok(TuiInteraction::Message { message, render }) => {
                            self.input_mode = InputMode::Editing;
                            self.suggestion_index = 0;
                            self.render_response(message, render);
                        }
                        Ok(TuiInteraction::PlanApproval { message }) => {
                            self.input_mode = InputMode::AwaitingPlanDecision;
                            self.suggestion_index = 0;
                            self.render_response(message, RenderPolicy::Immediate);
                        }
                        Ok(TuiInteraction::ProfilePicker { message, options }) => {
                            self.input_mode = InputMode::ChoosingProfile(TuiPicker { options });
                            self.suggestion_index = 0;
                            let _ = message;
                        }
                        Ok(TuiInteraction::SessionChanged {
                            input_history,
                            events,
                            plan_mode,
                            model,
                        }) => {
                            self.input_history = input_history;
                            self.input_mode = InputMode::Editing;
                            self.suggestion_index = 0;
                            self.set_session_replay(events);
                            self.plan_mode = plan_mode;
                            self.startup_info.model = model;
                        }
                        Ok(TuiInteraction::SessionPicker { message, options }) => {
                            self.input_mode = InputMode::ChoosingSession(SessionPicker { options });
                            self.suggestion_index = 0;
                            // `/sessions` opens a picker; its textual summary would only
                            // duplicate the rows already visible in that picker.
                            let _ = message;
                        }
                        Ok(TuiInteraction::PlanModeChanged { enabled }) => {
                            self.plan_mode = enabled;
                            self.input_mode = InputMode::Editing;
                            self.suggestion_index = 0;
                        }
                        Ok(TuiInteraction::ModelChanged { model, message }) => {
                            self.startup_info.model = model;
                            self.input_mode = InputMode::Editing;
                            self.suggestion_index = 0;
                            self.render_response(message, RenderPolicy::Immediate);
                        }
                        Err(error) => {
                            self.input_mode = InputMode::Editing;
                            if let Ok(mut view) = self.msg_view.lock() {
                                if error.kind() == io::ErrorKind::Interrupted {
                                    self.input_mode = InputMode::Editing;
                                    if let Some(index) = view
                                        .messages
                                        .iter()
                                        .rposition(|message| message.style == MsgStyle::Thinking)
                                    {
                                        view.messages.remove(index);
                                    }
                                    view.push(MsgStyle::System, "Cancelled.".to_owned());
                                } else {
                                    view.push(MsgStyle::Error, format!("Error: {error}"));
                                }
                            }
                        }
                    }
                }
                self.advance_stream();
                self.draw()?;
                self.handle_event(&request_tx)?;
            }
            Ok(())
        })();

        drop(request_tx);
        drop(response_rx);
        let terminal_result = self.terminal_session.restore();
        let worker_result = worker
            .join()
            .map_err(|_| io::Error::other("agent worker panicked"));

        loop_result?;
        terminal_result?;
        worker_result
    }

    fn handle_slash(&self, input: &str) -> String {
        match input {
            "/help" | "/h" => r#"Commands:
  /help /h  —  this menu
  /plan [on|off] — toggle or set plan mode
  /model    —  show active model
  /profile [NAME] — list or switch profiles
  /undo     —  undo last file operation
  /sessions [ID] — choose or resume a saved session
  /history [LIMIT] — show recent history
  /clear    —  start a new session
  /exit /quit — exit

Or just type what you want, e.g. "translate @clip.srt""#
                .to_owned(),
            "/plan" | "/model" | "/profile" | "/undo" | "/sessions" | "/history" | "/clear" => {
                format!(
                    "`{input}` is handled by the agent engine. When a real LLM backend is connected, these will route through the session."
                )
            }
            _ => {
                format!("Unknown command `{input}`. Try /help.")
            }
        }
    }

    fn start_stream(&mut self, text: String) {
        self.finish_stream();
        if let Ok(mut view) = self.msg_view.lock() {
            self.streaming = begin_stream(&mut view, text);
        }
    }

    fn render_response(&mut self, text: String, policy: RenderPolicy) {
        match policy {
            RenderPolicy::Stream => self.start_stream(text),
            RenderPolicy::Immediate => {
                self.finish_stream();
                if let Ok(mut view) = self.msg_view.lock() {
                    push_immediate_response(&mut view, text);
                }
            }
        }
    }

    fn finish_stream(&mut self) {
        let Some(stream) = self.streaming.take() else {
            return;
        };
        if let Ok(mut view) = self.msg_view.lock()
            && let Some(message) = view.messages.get_mut(stream.message_index)
        {
            message.text.extend(&stream.chars[stream.position..]);
        }
    }

    fn advance_stream(&mut self) {
        let Some(stream) = self.streaming.as_mut() else {
            return;
        };
        if std::time::Instant::now() < stream.next_at {
            return;
        }
        if let Ok(mut view) = self.msg_view.lock()
            && let Some(message) = view.messages.get_mut(stream.message_index)
            && let Some(ch) = stream.chars.get(stream.position)
        {
            message.text.push(*ch);
            stream.position += 1;
            stream.next_at = std::time::Instant::now() + STREAM_INTERVAL;
        }
        if stream.position >= stream.chars.len() {
            self.streaming = None;
        }
    }

    fn suggestions(&self) -> Vec<(String, String)> {
        suggestions_for(&self.input, &self.input_mode)
    }

    fn navigate_history_up(&mut self) {
        let Some((mode, input)) = history_up(&self.input_history, &self.input, &self.input_mode)
        else {
            return;
        };
        self.input_mode = mode;
        self.input = input;
    }

    fn navigate_history_down(&mut self) {
        let Some((mode, input)) = history_down(&self.input_history, &self.input_mode) else {
            return;
        };
        self.input_mode = mode;
        self.input = input;
    }

    fn draw(&mut self) -> io::Result<()> {
        let messages = self
            .msg_view
            .lock()
            .map(|v| v.all().to_vec())
            .unwrap_or_default();
        let scroll = self.scroll_offset;
        let suggestions = self.suggestions();
        let selected_suggestion = match &self.input_mode {
            InputMode::ChoosingSession(picker) => self
                .suggestion_index
                .min(picker.options.len().saturating_sub(1)),
            InputMode::ChoosingProfile(picker) => self
                .suggestion_index
                .min(picker.options.len().saturating_sub(1)),
            _ => self
                .suggestion_index
                .min(suggestions.len().saturating_sub(1)),
        };
        let processing = self.processing;
        let animation_tick = self.animation_started_at.elapsed().as_millis() / 80;
        let session_picker = match &self.input_mode {
            InputMode::ChoosingSession(picker) => Some(picker.options.clone()),
            _ => None,
        };
        let profile_picker = match &self.input_mode {
            InputMode::ChoosingProfile(picker) => Some(picker.options.clone()),
            _ => None,
        };
        let creating_profile = matches!(self.input_mode, InputMode::CreatingProfile);
        let profile_name_input = self.input.clone();
        let startup_info = self.startup_info.clone();
        let plan_mode = self.plan_mode;
        let mut transcript_page_lines = self.transcript_page_lines;

        self.terminal.draw(|frame| {
            let area = frame.area();
            if let Some(options) = &session_picker {
                frame.render_widget(Clear, area);
                let block = Block::default().borders(Borders::ALL);
                let inner = block.inner(area);
                frame.render_widget(block, area);
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(3),
                        Constraint::Min(0),
                        Constraint::Length(1),
                    ])
                    .split(inner);
                let header = vec![
                    Line::from(Span::styled(
                        "Resume a previous session",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )),
                    Line::from(Span::styled(
                        "Sessions for this project · newest activity first",
                        Style::default().fg(Color::DarkGray),
                    )),
                    Line::from(""),
                ];
                frame.render_widget(Paragraph::new(header), chunks[0]);
                let (start, end) =
                    picker_viewport(selected_suggestion, options.len(), chunks[1].height);
                let mut lines = Vec::new();
                for (index, session) in options.iter().enumerate().take(end).skip(start) {
                    let selected = index == selected_suggestion;
                    let style = if selected {
                        Style::default().fg(Color::Black).bg(Color::Cyan)
                    } else {
                        Style::default().fg(Color::White)
                    };
                    let metadata = format!(
                        "{}  {}  ·  {}  ·  {} events",
                        if session.active { "●" } else { " " },
                        session.updated_at,
                        session.cwd,
                        session.event_count,
                    );
                    lines.push(Line::from(Span::styled(metadata, style)));
                    lines.push(Line::from(Span::styled(
                        format!("   {}", session.title),
                        style.add_modifier(Modifier::BOLD),
                    )));
                    lines.push(Line::from(""));
                }
                frame.render_widget(Paragraph::new(lines), chunks[1]);
                let footer = format!(
                    "↑↓←→ navigate · Enter resume · Esc cancel  {}/{}",
                    selected_suggestion.saturating_add(1),
                    options.len()
                );
                frame.render_widget(
                    Paragraph::new(Line::from(Span::styled(
                        footer,
                        Style::default().fg(Color::DarkGray),
                    ))),
                    chunks[2],
                );
                return;
            }
            if let Some(options) = &profile_picker {
                frame.render_widget(Clear, area);
                let block = Block::default().borders(Borders::ALL);
                let inner = block.inner(area);
                frame.render_widget(block, area);
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(3),
                        Constraint::Min(0),
                        Constraint::Length(1),
                    ])
                    .split(inner);
                let header = vec![
                    Line::from(Span::styled(
                        "Choose a model profile",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )),
                    Line::from(Span::styled(
                        "Profiles from the active SubBake configuration",
                        Style::default().fg(Color::DarkGray),
                    )),
                    Line::from(""),
                ];
                frame.render_widget(Paragraph::new(header), chunks[0]);
                let (start, end) =
                    picker_viewport(selected_suggestion, options.len(), chunks[1].height);
                let mut lines = Vec::new();
                for (index, profile) in options.iter().enumerate().take(end).skip(start) {
                    let selected = index == selected_suggestion;
                    let style = if selected {
                        Style::default().fg(Color::Black).bg(Color::Cyan)
                    } else {
                        Style::default().fg(Color::White)
                    };
                    let marker = if profile.active { "●" } else { " " };
                    lines.push(Line::from(Span::styled(
                        format!("{marker}  {}", profile.name),
                        style.add_modifier(Modifier::BOLD),
                    )));
                    let details = if profile.create {
                        profile.model.clone()
                    } else {
                        format!("   {} / {}", profile.provider, profile.model)
                    };
                    lines.push(Line::from(Span::styled(details, style)));
                    lines.push(Line::from(""));
                }
                frame.render_widget(Paragraph::new(lines), chunks[1]);
                let footer = format!(
                    "↑↓←→ navigate · Enter select · Esc cancel  {}/{}",
                    selected_suggestion.saturating_add(1),
                    options.len()
                );
                frame.render_widget(
                    Paragraph::new(Line::from(Span::styled(
                        footer,
                        Style::default().fg(Color::DarkGray),
                    ))),
                    chunks[2],
                );
                return;
            }
            if creating_profile {
                frame.render_widget(Clear, area);
                let name = if profile_name_input.is_empty() {
                    "profile name…".to_owned()
                } else {
                    profile_name_input.clone()
                };
                let name_style = if profile_name_input.is_empty() {
                    Style::default().fg(Color::DarkGray)
                } else {
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD)
                };
                let lines = vec![
                    Line::from(Span::styled(
                        "Create a model profile",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )),
                    Line::from(Span::styled(
                        "Copy the active settings into a new profile",
                        Style::default().fg(Color::DarkGray),
                    )),
                    Line::from(""),
                    Line::from(Span::styled(
                        "Profile name",
                        Style::default().fg(Color::Cyan),
                    )),
                    Line::from(Span::styled(format!("> {name}"), name_style)),
                    Line::from(""),
                    Line::from(Span::styled(
                        "Allowed: letters, numbers, - and _",
                        Style::default().fg(Color::DarkGray),
                    )),
                    Line::from(Span::styled(
                        "Inline API keys and auth headers will not be copied.",
                        Style::default().fg(Color::Yellow),
                    )),
                    Line::from(""),
                    Line::from(Span::styled(
                        "Enter create · Esc cancel",
                        Style::default().fg(Color::DarkGray),
                    )),
                ];
                frame.render_widget(
                    Paragraph::new(lines)
                        .block(Block::default().borders(Borders::ALL))
                        .wrap(Wrap { trim: false }),
                    area,
                );
                frame.set_cursor_position((
                    area.x + 3 + terminal_width(&profile_name_input),
                    area.y + 5,
                ));
                return;
            }
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(1),
                    Constraint::Length(suggestions.len() as u16),
                    Constraint::Length(4),
                ])
                .split(area);

            // -- Output pane --
            let transcript_area = chunks[0];
            transcript_page_lines = transcript_area.height.max(1);
            let mut items =
                startup_panel_lines(&startup_info, transcript_area.width.saturating_sub(1));
            items.push(Line::from(""));
            items.extend(
                messages
                    .iter()
                    .enumerate()
                    .flat_map(|(index, msg)| {
                        let is_active_thinking = processing
                            && msg.style == MsgStyle::Thinking
                            && !messages[index + 1..]
                                .iter()
                                .any(|later| later.style == MsgStyle::Thinking);
                        let style = match msg.style {
                            MsgStyle::User => Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                            MsgStyle::Thinking if is_active_thinking => {
                                let colors = [Color::Yellow, Color::LightYellow, Color::White];
                                Style::default()
                                    .fg(colors[(animation_tick as usize / 2) % colors.len()])
                                    .add_modifier(Modifier::BOLD)
                            }
                            MsgStyle::Thinking => Style::default().fg(Color::DarkGray),
                            MsgStyle::ToolCall => Style::default().fg(Color::Green),
                            MsgStyle::Observation => Style::default().fg(Color::DarkGray),
                            MsgStyle::Response => Style::default().fg(Color::White),
                            MsgStyle::Error => Style::default().fg(Color::Red),
                            MsgStyle::System => {
                                Style::default().fg(Color::Blue).add_modifier(Modifier::DIM)
                            }
                        };
                        let display_text = if msg.style == MsgStyle::Thinking {
                            let marker = if is_active_thinking {
                                THINKING_FRAMES[animation_tick as usize % THINKING_FRAMES.len()]
                            } else {
                                "⎿"
                            };
                            format!("{marker} {}", msg.text)
                        } else {
                            msg.text.clone()
                        };
                        let lines = display_text
                            .split('\n')
                            .map(str::to_owned)
                            .collect::<Vec<_>>();
                        lines
                            .into_iter()
                            .map(move |line| Line::from(Span::styled(line, style)))
                    })
                    .collect::<Vec<Line<'_>>>(),
            );

            let visual_line_count = items
                .iter()
                .map(|line| {
                    let width = usize::from(transcript_area.width.max(1));
                    line.width().max(1).div_ceil(width)
                })
                .sum::<usize>();

            let paragraph = Paragraph::new(items)
                .block(Block::default().borders(Borders::NONE))
                .wrap(Wrap { trim: false });
            let max_scroll = visual_line_count
                .saturating_sub(usize::from(transcript_area.height))
                .min(usize::from(u16::MAX)) as u16;
            // `scroll_offset` is the number of visual lines above the newest
            // output, so zero follows a growing/streaming response.
            let offset = max_scroll.saturating_sub(scroll.min(max_scroll));
            let paragraph = paragraph.scroll((offset, 0));
            frame.render_widget(paragraph, transcript_area);
            if max_scroll > 0 {
                let mut scrollbar_state =
                    ScrollbarState::new(usize::from(max_scroll)).position(usize::from(offset));
                let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                    .begin_symbol(None)
                    .end_symbol(None)
                    .track_symbol(Some("│"))
                    .track_style(Style::default().fg(Color::Black))
                    .thumb_symbol("█")
                    .thumb_style(Style::default().fg(Color::DarkGray));
                frame.render_stateful_widget(scrollbar, transcript_area, &mut scrollbar_state);
            }

            // -- Slash command suggestions --
            let suggestion_area = chunks[1];
            let suggestion_lines =
                suggestions
                    .iter()
                    .enumerate()
                    .map(|(index, (command, description))| {
                        let selected = index == selected_suggestion;
                        let command_style = if selected {
                            Style::default()
                                .fg(Color::Black)
                                .bg(Color::Cyan)
                                .add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Color::Cyan)
                        };
                        let description_style = if selected {
                            Style::default().fg(Color::White).bg(Color::DarkGray)
                        } else {
                            Style::default().fg(Color::DarkGray)
                        };
                        Line::from(vec![
                            Span::styled(format!("› {command:<10}"), command_style),
                            Span::styled(description.clone(), description_style),
                        ])
                    });
            frame.render_widget(
                Paragraph::new(suggestion_lines.collect::<Vec<_>>()),
                suggestion_area,
            );

            // -- Input bar --
            let input_area = chunks[2];
            let input_entry_area = ratatui::layout::Rect::new(
                input_area.x,
                input_area.y,
                input_area.width,
                input_area.height.saturating_sub(1),
            );
            let status_area = ratatui::layout::Rect::new(
                input_area.x,
                input_area.bottom().saturating_sub(1),
                input_area.width,
                1,
            );
            let available_input_width = input_entry_area.width.saturating_sub(2);
            let input_line = input_line(
                &self.input,
                &self.input_mode,
                processing,
                self.input_hint,
                available_input_width,
            );
            let input_widget = Paragraph::new(input_line).block(
                Block::default()
                    .borders(Borders::TOP | Borders::BOTTOM)
                    .border_style(Style::default().fg(Color::DarkGray)),
            );
            frame.render_widget(input_widget, input_entry_area);
            let status_spans = vec![
                Span::styled(
                    startup_info.model.clone(),
                    Style::default().fg(Color::Yellow),
                ),
                Span::styled("  ", Style::default()),
                Span::styled(startup_info.cwd.clone(), Style::default().fg(Color::Green)),
            ];
            frame.render_widget(
                Paragraph::new(Line::from(status_spans)).wrap(Wrap { trim: false }),
                status_area,
            );
            if plan_mode && status_area.width >= 4 {
                let plan_area = ratatui::layout::Rect::new(
                    status_area.right().saturating_sub(4),
                    status_area.y,
                    4,
                    1,
                );
                frame.render_widget(
                    Paragraph::new(Span::styled(
                        "Plan",
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    ))
                    .alignment(ratatui::layout::Alignment::Right),
                    plan_area,
                );
            }

            // Set cursor position at end of input.
            frame.set_cursor_position((
                input_entry_area.x
                    + 2
                    + terminal_width(&self.input).min(available_input_width.saturating_sub(2)),
                input_entry_area.y + 1,
            ));
        })?;
        self.transcript_page_lines = transcript_page_lines;

        Ok(())
    }

    fn handle_event(
        &mut self,
        request_tx: &mpsc::Sender<(TuiAction, CancellationGuard)>,
    ) -> io::Result<()> {
        if !event::poll(STREAM_INTERVAL)? {
            return Ok(());
        }

        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.running = false;
                }
                KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.running = false;
                }
                KeyCode::Char('q') if self.input.is_empty() => {
                    self.running = false;
                }
                KeyCode::Esc => {
                    if self.processing {
                        if !self.cancellation_requested {
                            self.cancellation_requested = true;
                            if let Some(token) = &self.cancellation {
                                token.cancel();
                            }
                            if let Ok(mut view) = self.msg_view.lock() {
                                view.push(MsgStyle::System, "Cancellation requested…".to_owned());
                            }
                        }
                    } else {
                        self.input.clear();
                        self.input_mode = InputMode::Editing;
                        self.suggestion_index = 0;
                    }
                }
                KeyCode::Enter => {
                    if self.processing {
                        return Ok(());
                    }
                    let suggestions = self.suggestions();
                    let selected_action = if self.input.is_empty() {
                        match empty_mode_choice(&self.input_mode, self.suggestion_index) {
                            Some(EmptyModeChoice::Submit(action)) => Some(action),
                            Some(EmptyModeChoice::RevisePlan) => {
                                self.input_mode = InputMode::Editing;
                                self.suggestion_index = 0;
                                return Ok(());
                            }
                            Some(EmptyModeChoice::CreateProfile) => {
                                self.input_mode = InputMode::CreatingProfile;
                                self.suggestion_index = 0;
                                if let Ok(mut view) = self.msg_view.lock() {
                                    view.push(
                                        MsgStyle::System,
                                        "Enter a new profile name (letters, numbers, - and _)."
                                            .to_owned(),
                                    );
                                }
                                return Ok(());
                            }
                            None if !suggestions.is_empty() => {
                                let index = self.suggestion_index.min(suggestions.len() - 1);
                                self.input = suggestions[index].0.clone();
                                self.suggestion_index = 0;
                                return Ok(());
                            }
                            None => None,
                        }
                    } else if !suggestions.is_empty()
                        && !suggestions.iter().any(|item| item.0 == self.input)
                    {
                        let index = self.suggestion_index.min(suggestions.len() - 1);
                        self.input = suggestions[index].0.clone();
                        self.suggestion_index = 0;
                        return Ok(());
                    } else {
                        None
                    };

                    let action = if let Some(action) = selected_action {
                        action
                    } else {
                        let input = std::mem::take(&mut self.input);
                        let trimmed = input.trim().to_owned();

                        if trimmed.is_empty() {
                            return Ok(());
                        }

                        let creating_profile =
                            matches!(self.input_mode, InputMode::CreatingProfile);
                        self.input_mode = InputMode::Editing;
                        if self
                            .input_history
                            .last()
                            .is_none_or(|previous| previous != &trimmed)
                        {
                            self.input_history.push(trimmed.clone());
                        }

                        if trimmed == "/quit" || trimmed == "/exit" {
                            self.running = false;
                            return Ok(());
                        }

                        if matches!(trimmed.as_str(), "/help" | "/h") {
                            let response = self.handle_slash(&trimmed);
                            if let Ok(mut view) = self.msg_view.lock() {
                                view.push(MsgStyle::Response, response);
                            }
                            return Ok(());
                        }
                        if creating_profile {
                            TuiAction::CreateProfile(trimmed)
                        } else {
                            TuiAction::SubmitText(trimmed)
                        }
                    };
                    self.scroll_offset = 0;
                    let opens_session_picker =
                        matches!(&action, TuiAction::SubmitText(input) if input == "/sessions");
                    let changes_plan_mode = matches!(&action, TuiAction::TogglePlan)
                        || matches!(&action, TuiAction::SubmitText(input) if input.trim().starts_with("/plan"));

                    if !opens_session_picker && !changes_plan_mode {
                        // Paint the submitted message and initial thinking state
                        // before the potentially slow engine call begins.
                        if let Ok(mut view) = self.msg_view.lock() {
                            view.push(
                                MsgStyle::User,
                                format!("[{:?}] {}", iso_now(), action.visible_text()),
                            );
                            view.push(MsgStyle::Thinking, "Deciding next action…".to_owned());
                        }
                    }
                    self.processing = true;
                    self.cancellation_requested = false;
                    self.animation_started_at = std::time::Instant::now();
                    let guard = self
                        .cancellation
                        .as_ref()
                        .map(CancellationToken::guard)
                        .unwrap_or_default();
                    request_tx.send((action, guard)).map_err(|_| {
                        io::Error::new(io::ErrorKind::BrokenPipe, "agent worker stopped")
                    })?;
                }
                KeyCode::Char(ch) => {
                    if matches!(self.input_mode, InputMode::CreatingProfile)
                        && !is_profile_name_character(ch)
                    {
                        return Ok(());
                    }
                    if !matches!(self.input_mode, InputMode::CreatingProfile) {
                        self.input_mode = InputMode::Editing;
                    }
                    self.input.push(ch);
                    self.suggestion_index = 0;
                }
                KeyCode::Backspace => {
                    if !matches!(self.input_mode, InputMode::CreatingProfile) {
                        self.input_mode = InputMode::Editing;
                    }
                    self.input.pop();
                    self.suggestion_index = 0;
                }
                KeyCode::Up => {
                    match vertical_navigation(&self.input_mode, self.suggestions().len()) {
                        VerticalNavigation::Selection(count) => {
                            self.suggestion_index =
                                previous_suggestion(self.suggestion_index, count);
                        }
                        VerticalNavigation::History => self.navigate_history_up(),
                        VerticalNavigation::Disabled => {}
                    }
                }
                KeyCode::Down => {
                    match vertical_navigation(&self.input_mode, self.suggestions().len()) {
                        VerticalNavigation::Selection(count) => {
                            self.suggestion_index = (self.suggestion_index + 1) % count;
                        }
                        VerticalNavigation::History => self.navigate_history_down(),
                        VerticalNavigation::Disabled => {}
                    }
                }
                KeyCode::Left | KeyCode::Right => {
                    let option_count = match &self.input_mode {
                        InputMode::ChoosingSession(picker) => picker.options.len(),
                        InputMode::ChoosingProfile(picker) => picker.options.len(),
                        _ => 0,
                    };
                    if option_count > 0 {
                        self.suggestion_index = if key.code == KeyCode::Left {
                            previous_suggestion(self.suggestion_index, option_count)
                        } else {
                            (self.suggestion_index + 1) % option_count
                        };
                    }
                }
                KeyCode::PageUp => {
                    self.scroll_up(self.transcript_page_lines);
                }
                KeyCode::PageDown => {
                    self.scroll_down(self.transcript_page_lines);
                }
                KeyCode::BackTab => {
                    if self.processing {
                        return Ok(());
                    }
                    let action = TuiAction::TogglePlan;
                    self.processing = true;
                    self.cancellation_requested = false;
                    self.animation_started_at = std::time::Instant::now();
                    let guard = self
                        .cancellation
                        .as_ref()
                        .map(CancellationToken::guard)
                        .unwrap_or_default();
                    request_tx.send((action, guard)).map_err(|_| {
                        io::Error::new(io::ErrorKind::BrokenPipe, "agent worker stopped")
                    })?;
                }
                KeyCode::Tab => {
                    if self.input.starts_with('/') {
                        let matches = slash_suggestions(&self.input);
                        if !matches.is_empty() {
                            let index = self.suggestion_index.min(matches.len() - 1);
                            self.input = matches[index].0.to_owned();
                            self.suggestion_index = (index + 1) % matches.len();
                        }
                        return Ok(());
                    }
                    // Simple natural-language completion: "tra" → "translate "
                    let completions = [
                        "translate ",
                        "transcribe ",
                        "list files",
                        "read file",
                        "search files",
                        "whisper ",
                    ];
                    if !self.input.is_empty()
                        && let Some(c) = completions.iter().find(|c| c.starts_with(&self.input))
                    {
                        self.input = c.to_string();
                    }
                }
                _ => {}
            },
            Event::Resize(_, _) => {
                self.scroll_offset = 0;
            }
            _ => {}
        }

        Ok(())
    }

    fn scroll_up(&mut self, lines: u16) {
        self.scroll_offset = self.scroll_offset.saturating_add(lines);
    }

    fn scroll_down(&mut self, lines: u16) {
        self.scroll_offset = self.scroll_offset.saturating_sub(lines);
    }
}

const INPUT_HINTS: &[&str] = &[
    "Type a message or /help for commands",
    "Ask SubBake to translate, transcribe, or inspect a file",
    "Mention a subtitle file to get started",
    "Use /plan to review the next steps before changes",
    "Use /history to revisit earlier requests",
];

fn terminal_width(value: &str) -> u16 {
    u16::try_from(UnicodeWidthStr::width(value)).unwrap_or(u16::MAX)
}

fn startup_panel_lines(info: &StartupInfo, width: u16) -> Vec<Line<'_>> {
    let width = usize::from(width.max(4));
    let inner_width = width.saturating_sub(2);
    let value_style = Style::default().fg(Color::Cyan);
    let border_style = Style::default().fg(Color::DarkGray);
    let row = |label: &'static str, value: &str| {
        let prefix = format!("  {label:<10}");
        let available = inner_width.saturating_sub(prefix.chars().count());
        let value = truncate_with_ellipsis(value, available);
        let padding = " ".repeat(available.saturating_sub(value.chars().count()));
        Line::from(vec![
            Span::styled("│", border_style),
            Span::styled(prefix, Style::default().fg(Color::DarkGray)),
            Span::styled(value, value_style),
            Span::raw(padding),
            Span::styled("│", border_style),
        ])
    };
    let blank = || {
        Line::from(vec![
            Span::styled("│", border_style),
            Span::raw(" ".repeat(inner_width)),
            Span::styled("│", border_style),
        ])
    };
    let title = truncate_with_ellipsis(
        &format!("  SubBake v{}", env!("CARGO_PKG_VERSION")),
        inner_width,
    );
    let title_padding = " ".repeat(inner_width.saturating_sub(title.chars().count()));
    vec![
        Line::from(Span::styled(
            format!("╭{}╮", "─".repeat(inner_width)),
            border_style,
        )),
        Line::from(vec![
            Span::styled("│", border_style),
            Span::styled(title, Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(title_padding),
            Span::styled("│", border_style),
        ]),
        blank(),
        row("Provider", &info.provider),
        row("Model", &info.model),
        row("Config", &info.config),
        row(
            "Cache",
            if info.cache_enabled {
                "Enabled"
            } else {
                "Disabled"
            },
        ),
        Line::from(Span::styled(
            format!("╰{}╯", "─".repeat(inner_width)),
            border_style,
        )),
    ]
}

fn truncate_with_ellipsis(value: &str, width: usize) -> String {
    if value.chars().count() <= width {
        return value.to_owned();
    }
    if width == 0 {
        return String::new();
    }
    if width == 1 {
        return "…".to_owned();
    }
    value.chars().take(width - 1).chain(['…']).collect()
}

fn session_input_hint() -> &'static str {
    let index = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            duration.subsec_nanos() as usize % INPUT_HINTS.len()
        });
    INPUT_HINTS[index]
}

fn input_line<'a>(
    input: &'a str,
    mode: &InputMode,
    processing: bool,
    hint: &'a str,
    width: u16,
) -> Line<'a> {
    let truncate =
        |value: &str| truncate_with_ellipsis(value, usize::from(width.saturating_sub(2)));
    if input.is_empty() && matches!(mode, InputMode::Editing) && !processing {
        Line::from(vec![
            Span::styled("> ", Style::default().fg(Color::Cyan)),
            Span::styled(truncate(hint), Style::default().fg(Color::DarkGray)),
        ])
    } else {
        Line::from(Span::styled(
            format!("> {}", truncate(input)),
            Style::default().fg(Color::Cyan),
        ))
    }
}

fn suggestions_for(input: &str, mode: &InputMode) -> Vec<(String, String)> {
    match mode {
        InputMode::BrowsingHistory { .. } => Vec::new(),
        InputMode::AwaitingPlanDecision if input.is_empty() => APPROVAL_OPTIONS
            .iter()
            .map(|(label, description)| ((*label).to_owned(), (*description).to_owned()))
            .collect(),
        InputMode::ChoosingProfile(_) => Vec::new(),
        InputMode::CreatingProfile => Vec::new(),
        InputMode::ChoosingSession(_) => Vec::new(),
        _ => slash_suggestions(input)
            .into_iter()
            .map(|(command, description)| (command.to_owned(), description.to_owned()))
            .collect(),
    }
}

fn approval_choice(index: usize) -> ApprovalChoice {
    match index.min(APPROVAL_OPTIONS.len() - 1) {
        0 => ApprovalChoice::Submit(TuiAction::ApprovePlan),
        1 => ApprovalChoice::Submit(TuiAction::RejectPlan),
        _ => ApprovalChoice::Revise,
    }
}

fn vertical_navigation(mode: &InputMode, suggestion_count: usize) -> VerticalNavigation {
    match mode {
        InputMode::ChoosingProfile(picker) if !picker.options.is_empty() => {
            VerticalNavigation::Selection(picker.options.len())
        }
        InputMode::ChoosingSession(picker) if !picker.options.is_empty() => {
            VerticalNavigation::Selection(picker.options.len())
        }
        InputMode::AwaitingPlanDecision if suggestion_count > 0 => {
            VerticalNavigation::Selection(suggestion_count)
        }
        InputMode::Editing if suggestion_count > 0 => {
            VerticalNavigation::Selection(suggestion_count)
        }
        InputMode::Editing | InputMode::BrowsingHistory { .. } => VerticalNavigation::History,
        InputMode::ChoosingProfile(_)
        | InputMode::ChoosingSession(_)
        | InputMode::AwaitingPlanDecision
        | InputMode::CreatingProfile => VerticalNavigation::Disabled,
    }
}

fn profile_picker_choice(picker: &TuiPicker, index: usize) -> Option<ProfilePickerChoice> {
    let option = picker
        .options
        .get(index.min(picker.options.len().saturating_sub(1)))?;
    if option.create {
        Some(ProfilePickerChoice::Create)
    } else {
        Some(ProfilePickerChoice::Select(option.name.clone()))
    }
}

fn empty_mode_choice(mode: &InputMode, index: usize) -> Option<EmptyModeChoice> {
    match mode {
        InputMode::AwaitingPlanDecision => match approval_choice(index) {
            ApprovalChoice::Submit(action) => Some(EmptyModeChoice::Submit(action)),
            ApprovalChoice::Revise => Some(EmptyModeChoice::RevisePlan),
        },
        InputMode::ChoosingProfile(picker) => match profile_picker_choice(picker, index)? {
            ProfilePickerChoice::Select(name) => {
                Some(EmptyModeChoice::Submit(TuiAction::SelectProfile(name)))
            }
            ProfilePickerChoice::Create => Some(EmptyModeChoice::CreateProfile),
        },
        InputMode::ChoosingSession(picker) => picker
            .options
            .get(index.min(picker.options.len().saturating_sub(1)))
            .map(|session| EmptyModeChoice::Submit(TuiAction::SelectSession(session.id.clone()))),
        _ => None,
    }
}

fn begin_stream(view: &mut MsgView, text: String) -> Option<StreamingResponse> {
    if text.is_empty() {
        return None;
    }
    view.push(MsgStyle::Response, "➔ ".to_owned());
    Some(StreamingResponse {
        chars: text.chars().collect(),
        position: 0,
        message_index: view.messages.len() - 1,
        next_at: std::time::Instant::now(),
    })
}

fn push_immediate_response(view: &mut MsgView, text: String) {
    if !text.is_empty() {
        view.push(MsgStyle::Response, format!("➔ {text}"));
    }
}

fn history_up(history: &[String], input: &str, mode: &InputMode) -> Option<(InputMode, String)> {
    if history.is_empty() {
        return None;
    }
    let (index, draft) = match mode {
        InputMode::BrowsingHistory { index, draft } => (index.saturating_sub(1), draft.clone()),
        _ => (history.len() - 1, input.to_owned()),
    };
    Some((
        InputMode::BrowsingHistory { index, draft },
        history[index].clone(),
    ))
}

fn history_down(history: &[String], mode: &InputMode) -> Option<(InputMode, String)> {
    let InputMode::BrowsingHistory { index, draft } = mode else {
        return None;
    };
    if index + 1 < history.len() {
        let next = index + 1;
        Some((
            InputMode::BrowsingHistory {
                index: next,
                draft: draft.clone(),
            },
            history[next].clone(),
        ))
    } else {
        Some((InputMode::Editing, draft.clone()))
    }
}

const PICKER_ROW_HEIGHT: usize = 3;

fn picker_viewport(selected: usize, option_count: usize, height: u16) -> (usize, usize) {
    if option_count == 0 {
        return (0, 0);
    }
    let selected = selected.min(option_count - 1);
    let visible = (usize::from(height) / PICKER_ROW_HEIGHT).max(1);
    let start = selected
        .saturating_add(1)
        .saturating_sub(visible)
        .min(option_count.saturating_sub(visible));
    (start, start.saturating_add(visible).min(option_count))
}

fn slash_suggestions(input: &str) -> Vec<(&'static str, &'static str)> {
    if !input.starts_with('/') || input.contains(char::is_whitespace) {
        return Vec::new();
    }
    let query = input.to_ascii_lowercase();
    SLASH_COMMANDS
        .iter()
        .copied()
        .filter(|(command, _)| command.starts_with(&query))
        .collect()
}

fn previous_suggestion(current: usize, count: usize) -> usize {
    if current == 0 {
        count.saturating_sub(1)
    } else {
        current - 1
    }
}

fn is_profile_name_character(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '-' | '_')
}

#[cfg(test)]
mod tests {
    use crate::engine::ProfileChoice;

    use super::{
        ApprovalChoice, EmptyModeChoice, InputMode, ProfilePickerChoice, TuiAction, TuiPicker,
        VerticalNavigation, approval_choice, empty_mode_choice, history_down, history_up,
        input_line, is_profile_name_character, picker_viewport, previous_suggestion,
        profile_picker_choice, push_immediate_response, slash_suggestions, suggestions_for,
        terminal_width, vertical_navigation,
    };

    #[test]
    fn terminal_width_uses_display_columns_for_unicode_input() {
        assert_eq!(terminal_width("hello"), 5);
        assert_eq!(terminal_width("中文"), 4);
        assert_eq!(terminal_width("a中"), 3);
    }

    #[test]
    fn slash_displays_all_commands_and_filters_as_the_user_types() {
        assert_eq!(slash_suggestions("/").len(), 10);
        assert_eq!(
            slash_suggestions("/mod"),
            vec![("/model", "show the active model")]
        );
        assert!(slash_suggestions("hello /").is_empty());
    }

    #[test]
    fn input_hint_only_appears_for_idle_empty_editing() {
        let line = input_line("", &InputMode::Editing, false, "A helpful hint", 80);
        assert_eq!(line.spans.len(), 2);
        assert_eq!(line.spans[1].content, "A helpful hint");

        let typed = input_line("hello", &InputMode::Editing, false, "hidden", 80);
        assert_eq!(typed.spans.len(), 1);
        assert_eq!(typed.spans[0].content, "> hello");

        let processing = input_line("", &InputMode::Editing, true, "hidden", 80);
        assert_eq!(processing.spans.len(), 1);
        assert_eq!(processing.spans[0].content, "> ");
    }

    #[test]
    fn slash_selection_wraps_in_both_directions() {
        assert_eq!(previous_suggestion(0, 7), 6);
        assert_eq!(previous_suggestion(4, 7), 3);
        assert_eq!((6 + 1) % 7, 0);
    }

    #[test]
    fn visible_slash_and_approval_options_take_vertical_navigation_priority() {
        assert_eq!(
            vertical_navigation(&InputMode::Editing, slash_suggestions("/").len()),
            VerticalNavigation::Selection(10)
        );
        assert_eq!(
            vertical_navigation(&InputMode::AwaitingPlanDecision, 3),
            VerticalNavigation::Selection(3)
        );
    }

    #[test]
    fn history_is_only_active_in_editing_and_history_modes_without_suggestions() {
        assert_eq!(
            vertical_navigation(&InputMode::Editing, 0),
            VerticalNavigation::History
        );
        assert_eq!(
            vertical_navigation(
                &InputMode::BrowsingHistory {
                    index: 0,
                    draft: String::new(),
                },
                0,
            ),
            VerticalNavigation::History
        );
        assert_eq!(
            vertical_navigation(&InputMode::CreatingProfile, 0),
            VerticalNavigation::Disabled
        );
    }

    #[test]
    fn picker_viewport_keeps_the_selection_visible() {
        assert_eq!(picker_viewport(0, 20, 9), (0, 3));
        assert_eq!(picker_viewport(2, 20, 9), (0, 3));
        assert_eq!(picker_viewport(3, 20, 9), (1, 4));
        assert_eq!(picker_viewport(19, 20, 9), (17, 20));
    }

    #[test]
    fn picker_viewport_handles_empty_and_tiny_areas() {
        assert_eq!(picker_viewport(0, 0, 9), (0, 0));
        assert_eq!(picker_viewport(4, 5, 0), (4, 5));
        assert_eq!(picker_viewport(99, 5, 3), (4, 5));
    }

    #[test]
    fn typed_profile_action_has_a_stable_visible_form() {
        let action = TuiAction::SelectProfile("strict".to_owned());
        assert_eq!(action.visible_text(), "/profile strict");
    }

    #[test]
    fn profile_creation_is_a_typed_action_and_picker_choice() {
        let action = TuiAction::CreateProfile("review_copy".to_owned());
        assert_eq!(action.visible_text(), "create profile review_copy");
        let picker = TuiPicker {
            options: vec![ProfileChoice {
                name: "new profile…".to_owned(),
                provider: String::new(),
                model: "copy active settings without credentials".to_owned(),
                active: false,
                create: true,
            }],
        };
        assert_eq!(
            profile_picker_choice(&picker, 0),
            Some(ProfilePickerChoice::Create)
        );
        let mode = InputMode::ChoosingProfile(picker);
        assert!(suggestions_for("", &mode).is_empty());
        assert!(is_profile_name_character('_'));
        assert!(is_profile_name_character('9'));
        assert!(!is_profile_name_character('.'));
        assert!(!is_profile_name_character('中'));
    }

    #[test]
    fn existing_profile_picker_choice_submits_the_profile_name() {
        let picker = TuiPicker {
            options: vec![ProfileChoice {
                name: "strict".to_owned(),
                provider: "mock".to_owned(),
                model: "mock-strict".to_owned(),
                active: false,
                create: false,
            }],
        };
        assert_eq!(
            profile_picker_choice(&picker, 0),
            Some(ProfilePickerChoice::Select("strict".to_owned()))
        );
    }

    #[test]
    fn history_round_trip_restores_the_unsubmitted_draft() {
        let history = vec!["first".to_owned(), "/sessions".to_owned()];
        let (mode, input) = history_up(&history, "draft", &InputMode::Editing).expect("up");
        assert_eq!(input, "/sessions");
        let (mode, input) = history_up(&history, &input, &mode).expect("up again");
        assert_eq!(input, "first");
        let (mode, input) = history_down(&history, &mode).expect("down");
        assert_eq!(input, "/sessions");
        let (mode, input) = history_down(&history, &mode).expect("restore draft");
        assert!(matches!(mode, InputMode::Editing));
        assert_eq!(input, "draft");
    }

    #[test]
    fn active_picker_and_approval_modes_take_priority_over_slash_completion() {
        let profile = InputMode::ChoosingProfile(TuiPicker {
            options: vec![ProfileChoice {
                name: "fast".to_owned(),
                provider: "mock".to_owned(),
                model: "mock-fast".to_owned(),
                active: true,
                create: false,
            }],
        });
        assert!(suggestions_for("", &profile).is_empty());
        assert_eq!(
            suggestions_for("", &InputMode::AwaitingPlanDecision)[0].0,
            "approve"
        );
        let history = InputMode::BrowsingHistory {
            index: 0,
            draft: String::new(),
        };
        assert!(suggestions_for("/", &history).is_empty());
    }

    #[test]
    fn immediate_response_is_complete_without_a_stream_placeholder() {
        let mut view = super::MsgView::new(10);
        push_immediate_response(&mut view, "one.srt\ntwo.srt".to_owned());
        assert_eq!(view.all().len(), 1);
        assert_eq!(view.all()[0].text, "➔ one.srt\ntwo.srt");

        let stream = super::begin_stream(&mut view, "hello".to_owned()).expect("stream");
        assert_eq!(view.all()[stream.message_index].text, "➔ ");
        assert_eq!(stream.chars.iter().collect::<String>(), "hello");
    }

    #[test]
    fn all_plan_approval_choices_have_distinct_typed_outcomes() {
        assert_eq!(
            approval_choice(0),
            ApprovalChoice::Submit(TuiAction::ApprovePlan)
        );
        assert_eq!(
            approval_choice(1),
            ApprovalChoice::Submit(TuiAction::RejectPlan)
        );
        assert_eq!(approval_choice(2), ApprovalChoice::Revise);
        assert_eq!(
            empty_mode_choice(&InputMode::AwaitingPlanDecision, 0),
            Some(EmptyModeChoice::Submit(TuiAction::ApprovePlan))
        );
        assert_eq!(
            empty_mode_choice(&InputMode::AwaitingPlanDecision, 1),
            Some(EmptyModeChoice::Submit(TuiAction::RejectPlan))
        );
        assert_eq!(
            empty_mode_choice(&InputMode::AwaitingPlanDecision, 2),
            Some(EmptyModeChoice::RevisePlan)
        );
    }
}
