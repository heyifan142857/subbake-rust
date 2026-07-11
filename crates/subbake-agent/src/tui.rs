//! Chat-style inline TUI: committed output is written to the terminal's native
//! scrollback while the composer and active picker are redrawn below it.
//!
//! Layout (inspired by Codex / OpenCode):
//!
//! ┌─────────────────────────────────┐
//! │  Terminal-native scrollback     │
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
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, supports_keyboard_enhancement};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Widget, Wrap};
use ratatui::{Terminal, TerminalOptions, Viewport};
use unicode_width::UnicodeWidthStr;

use crate::engine::{EngineObserver, ProfileChoice, SessionChoice};
use crate::input_editor::InputEditor;
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

const EVENT_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);

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

struct TerminalSessionGuard {
    active: bool,
    keyboard_enhancement: bool,
    alternate_screen: bool,
}

impl TerminalSessionGuard {
    fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        let keyboard_enhancement = supports_keyboard_enhancement().unwrap_or(false);
        if keyboard_enhancement
            && let Err(error) = io::stdout().execute(PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES,
            ))
        {
            let _ = disable_raw_mode();
            return Err(error);
        }
        Ok(Self {
            active: true,
            keyboard_enhancement,
            alternate_screen: false,
        })
    }

    fn enter_alternate_screen(&mut self) -> io::Result<()> {
        if !self.alternate_screen {
            io::stdout().execute(EnterAlternateScreen)?;
            self.alternate_screen = true;
        }
        Ok(())
    }

    fn leave_alternate_screen(&mut self) -> io::Result<()> {
        if self.alternate_screen {
            io::stdout().execute(LeaveAlternateScreen)?;
            self.alternate_screen = false;
        }
        Ok(())
    }

    fn restore(&mut self) -> io::Result<()> {
        if !self.active {
            return Ok(());
        }
        self.active = false;
        let screen_result = self.leave_alternate_screen();
        let keyboard_result = if self.keyboard_enhancement {
            io::stdout()
                .execute(PopKeyboardEnhancementFlags)
                .map(|_| ())
        } else {
            Ok(())
        };
        let raw_result = disable_raw_mode();
        screen_result.and(keyboard_result).and(raw_result)
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
            messages: Vec::with_capacity(max.min(4096)),
            max,
        }
    }

    pub fn push(&mut self, style: MsgStyle, text: String) {
        let stamp = iso_now();
        if self.max != usize::MAX && self.messages.len() >= self.max {
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
        let _ = text;
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
    overlay_terminal: Option<Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>>,
    msg_view: std::sync::Arc<std::sync::Mutex<MsgView>>,
    input: InputEditor,
    input_history: Vec<String>,
    input_mode: InputMode,
    running: bool,
    suggestion_index: usize,
    processing: bool,
    pending_plan_toggle: Option<bool>,
    cancellation: Option<CancellationToken>,
    cancellation_requested: bool,
    input_hint: &'static str,
    startup_info: StartupInfo,
    plan_mode: bool,
    history_cursor: usize,
    startup_pending: bool,
    picker_cancel_exits: bool,
}

impl SubBakeTui {
    pub fn new() -> io::Result<Self> {
        let terminal_session = TerminalSessionGuard::enter()?;
        let backend = ratatui::backend::CrosstermBackend::new(io::stdout());
        let terminal = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(12),
            },
        )?;
        Ok(Self {
            terminal_session,
            terminal,
            overlay_terminal: None,
            // The terminal emulator owns scrollback retention. Keep the source
            // items for this process lifetime so the commit cursor stays stable.
            msg_view: std::sync::Arc::new(std::sync::Mutex::new(MsgView::new(usize::MAX))),
            input: InputEditor::default(),
            input_history: Vec::new(),
            input_mode: InputMode::Editing,
            running: true,
            suggestion_index: 0,
            processing: false,
            pending_plan_toggle: None,
            cancellation: None,
            cancellation_requested: false,
            input_hint: session_input_hint(),
            startup_info: StartupInfo::default(),
            plan_mode: false,
            history_cursor: 0,
            startup_pending: true,
            picker_cancel_exits: false,
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
        if let Ok(mut view) = self.msg_view.lock() {
            if !view.messages.is_empty() {
                view.push(
                    MsgStyle::System,
                    "──────── resumed session ────────".to_owned(),
                );
            }
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
    }

    /// Show the same resume picker used by the `/sessions` command on startup.
    pub fn open_session_picker(&mut self, options: Vec<SessionChoice>) -> io::Result<()> {
        self.open_fullscreen_overlay()?;
        self.picker_cancel_exits = true;
        self.input.clear();
        self.input_mode = InputMode::ChoosingSession(SessionPicker { options });
        self.suggestion_index = 0;
        Ok(())
    }

    fn open_fullscreen_overlay(&mut self) -> io::Result<()> {
        if self.overlay_terminal.is_none() {
            self.terminal_session.enter_alternate_screen()?;
            let backend = ratatui::backend::CrosstermBackend::new(io::stdout());
            self.overlay_terminal = Some(Terminal::new(backend)?);
        }
        Ok(())
    }

    fn close_fullscreen_overlay(&mut self) -> io::Result<()> {
        if let Some(mut terminal) = self.overlay_terminal.take() {
            terminal.clear()?;
            terminal.show_cursor()?;
        }
        self.terminal_session.leave_alternate_screen()
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
                        Ok(TuiInteraction::Message { message }) => {
                            self.input_mode = InputMode::Editing;
                            self.suggestion_index = 0;
                            self.render_response(message);
                        }
                        Ok(TuiInteraction::PlanApproval { message }) => {
                            self.input_mode = InputMode::AwaitingPlanDecision;
                            self.suggestion_index = 0;
                            self.render_response(message);
                        }
                        Ok(TuiInteraction::ProfilePicker { message, options }) => {
                            self.open_fullscreen_overlay()?;
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
                            self.open_fullscreen_overlay()?;
                            self.input_mode = InputMode::ChoosingSession(SessionPicker { options });
                            self.suggestion_index = 0;
                            // `/sessions` opens a picker; its textual summary would only
                            // duplicate the rows already visible in that picker.
                            let _ = message;
                        }
                        Ok(TuiInteraction::PlanModeChanged { enabled }) => {
                            self.plan_mode = enabled;
                            self.pending_plan_toggle = None;
                            self.input_mode = InputMode::Editing;
                            self.suggestion_index = 0;
                        }
                        Ok(TuiInteraction::ModelChanged { model, message }) => {
                            self.startup_info.model = model;
                            self.input_mode = InputMode::Editing;
                            self.suggestion_index = 0;
                            self.render_response(message);
                        }
                        Err(error) => {
                            if let Some(previous) = self.pending_plan_toggle.take() {
                                self.plan_mode = previous;
                            }
                            self.input_mode = InputMode::Editing;
                            if let Ok(mut view) = self.msg_view.lock() {
                                if error.kind() == io::ErrorKind::Interrupted {
                                    self.input_mode = InputMode::Editing;
                                    view.push(MsgStyle::System, "Cancelled.".to_owned());
                                } else {
                                    view.push(MsgStyle::Error, format!("Error: {error}"));
                                }
                            }
                        }
                    }
                }
                self.flush_history()?;
                self.draw()?;
                self.handle_event(&request_tx)?;
            }
            Ok(())
        })();

        drop(request_tx);
        drop(response_rx);
        let overlay_result = self.close_fullscreen_overlay();
        let clear_result = self.terminal.clear();
        let cursor_result = self.terminal.show_cursor();
        let terminal_result = self.terminal_session.restore();
        let worker_result = worker
            .join()
            .map_err(|_| io::Error::other("agent worker panicked"));

        loop_result?;
        overlay_result?;
        clear_result?;
        cursor_result?;
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

    fn render_response(&mut self, text: String) {
        if let Ok(mut view) = self.msg_view.lock() {
            push_immediate_response(&mut view, text);
        }
    }

    fn suggestions(&self) -> Vec<(String, String)> {
        suggestions_for(self.input.text(), &self.input_mode)
    }

    fn navigate_history_up(&mut self) {
        let Some((mode, input)) =
            history_up(&self.input_history, self.input.text(), &self.input_mode)
        else {
            return;
        };
        self.input_mode = mode;
        self.input.set_text(input);
    }

    fn navigate_history_down(&mut self) {
        let Some((mode, input)) = history_down(&self.input_history, &self.input_mode) else {
            return;
        };
        self.input_mode = mode;
        self.input.set_text(input);
    }

    /// Commit completed output above the inline viewport. Once inserted, these
    /// rows belong to the terminal emulator's native scrollback and are no
    /// longer redrawn by SubBake.
    fn flush_history(&mut self) -> io::Result<()> {
        if self.overlay_terminal.is_some() {
            return Ok(());
        }
        let width = self.terminal.size()?.width.max(1);
        if self.startup_pending {
            self.startup_pending = false;
            let lines = startup_panel_lines(&self.startup_info, width);
            self.insert_history_lines(lines, width)?;
        }

        let messages = self
            .msg_view
            .lock()
            .map(|view| view.messages[self.history_cursor.min(view.messages.len())..].to_vec())
            .unwrap_or_default();
        if messages.is_empty() {
            return Ok(());
        }
        self.history_cursor = self.history_cursor.saturating_add(messages.len());
        let lines = messages
            .iter()
            .flat_map(message_lines)
            .collect::<Vec<Line<'static>>>();
        self.insert_history_lines(lines, width)
    }

    fn insert_history_lines(&mut self, lines: Vec<Line<'static>>, width: u16) -> io::Result<()> {
        if lines.is_empty() {
            return Ok(());
        }
        let height = history_lines_height(&lines, width);
        self.terminal.insert_before(height.max(1), move |buffer| {
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .render(buffer.area, buffer);
        })
    }

    fn draw(&mut self) -> io::Result<()> {
        let terminal_area = self
            .overlay_terminal
            .as_ref()
            .map_or_else(|| self.terminal.size(), Terminal::size)?;
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
        let session_picker = match &self.input_mode {
            InputMode::ChoosingSession(picker) => Some(picker.options.clone()),
            _ => None,
        };
        let profile_picker = match &self.input_mode {
            InputMode::ChoosingProfile(picker) => Some(picker.options.clone()),
            _ => None,
        };
        let creating_profile = matches!(self.input_mode, InputMode::CreatingProfile);
        let profile_name_input = self.input.text().to_owned();
        let startup_info = self.startup_info.clone();
        let plan_mode = self.plan_mode;

        let input_width = terminal_area.width.saturating_sub(4).max(1);
        let max_input_lines = (terminal_area.height.saturating_mul(40) / 100).max(1);
        let input_line_count = self.input.desired_height(input_width).min(max_input_lines);
        let input_total_height = input_line_count.saturating_add(3);
        let suggestion_height = (suggestions.len() as u16).min(
            terminal_area
                .height
                .saturating_sub(input_total_height)
                .saturating_sub(1),
        );
        let visible_input_lines = self.input.visible_lines(input_width, input_line_count);
        let (input_cursor_x, input_cursor_y) = self.input.cursor_position(input_width);

        let draw_ui = |frame: &mut ratatui::Frame<'_>| {
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
                    Constraint::Length(suggestion_height),
                    Constraint::Length(input_total_height),
                ])
                .split(area);

            // -- Slash command suggestions --
            let suggestion_area = chunks[0];
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
            let input_area = chunks[1];
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
            let input_lines = if self.input.is_empty()
                && matches!(self.input_mode, InputMode::Editing)
                && !processing
            {
                vec![Line::from(vec![
                    Span::styled("> ", Style::default().fg(Color::Cyan)),
                    Span::styled(
                        truncate_with_ellipsis(self.input_hint, usize::from(input_width)),
                        Style::default().fg(Color::DarkGray),
                    ),
                ])]
            } else {
                visible_input_lines
                    .iter()
                    .enumerate()
                    .map(|(index, line)| {
                        let prefix = if index == 0 { "> " } else { "  " };
                        Line::from(Span::styled(
                            format!("{prefix}{line}"),
                            Style::default().fg(Color::Cyan),
                        ))
                    })
                    .collect()
            };
            let input_widget = Paragraph::new(input_lines).block(
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
                if processing && self.pending_plan_toggle.is_none() {
                    Span::styled("  Working · Esc cancel", Style::default().fg(Color::Yellow))
                } else {
                    Span::raw("")
                },
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

            // Keep the terminal cursor aligned with the logical editing cursor.
            frame.set_cursor_position((
                input_entry_area.x + 2 + input_cursor_x.min(input_width),
                input_entry_area.y + 1 + input_cursor_y.min(input_line_count.saturating_sub(1)),
            ));
        };
        if let Some(terminal) = self.overlay_terminal.as_mut() {
            terminal.draw(draw_ui)?;
        } else {
            self.terminal.draw(draw_ui)?;
        }

        Ok(())
    }

    fn handle_event(
        &mut self,
        request_tx: &mpsc::Sender<(TuiAction, CancellationGuard)>,
    ) -> io::Result<()> {
        if !event::poll(EVENT_POLL_INTERVAL)? {
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
                        let closes_overlay = matches!(
                            self.input_mode,
                            InputMode::ChoosingSession(_)
                                | InputMode::ChoosingProfile(_)
                                | InputMode::CreatingProfile
                        );
                        self.input.clear();
                        self.input_mode = InputMode::Editing;
                        self.suggestion_index = 0;
                        if closes_overlay {
                            self.close_fullscreen_overlay()?;
                        }
                        if self.picker_cancel_exits {
                            self.running = false;
                        }
                    }
                }
                _ if is_insert_newline_key(key)
                    && !matches!(self.input_mode, InputMode::CreatingProfile) =>
                {
                    self.input.insert_newline();
                    self.input_mode = InputMode::Editing;
                    self.suggestion_index = 0;
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
                                self.input.set_text(suggestions[index].0.clone());
                                self.suggestion_index = 0;
                                return Ok(());
                            }
                            None => None,
                        }
                    } else if !suggestions.is_empty()
                        && !suggestions.iter().any(|item| item.0 == self.input.text())
                    {
                        let index = self.suggestion_index.min(suggestions.len() - 1);
                        self.input.set_text(suggestions[index].0.clone());
                        self.suggestion_index = 0;
                        return Ok(());
                    } else {
                        None
                    };

                    let action = if let Some(action) = selected_action {
                        action
                    } else {
                        let input = self.input.take();
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
                    let opens_session_picker =
                        matches!(&action, TuiAction::SubmitText(input) if input == "/sessions");
                    let changes_plan_mode = matches!(&action, TuiAction::TogglePlan)
                        || matches!(&action, TuiAction::SubmitText(input) if input.trim().starts_with("/plan"));

                    if !opens_session_picker
                        && !changes_plan_mode
                        && let TuiAction::SubmitText(text) = &action
                        && let Ok(mut view) = self.msg_view.lock()
                    {
                        view.push(MsgStyle::User, format!("[{}] {text}", iso_now()));
                    }
                    if matches!(
                        &action,
                        TuiAction::SelectProfile(_)
                            | TuiAction::CreateProfile(_)
                            | TuiAction::SelectSession(_)
                    ) {
                        self.close_fullscreen_overlay()?;
                        self.picker_cancel_exits = false;
                    }
                    self.processing = true;
                    self.cancellation_requested = false;
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
                    self.input.insert_char(ch);
                    self.suggestion_index = 0;
                }
                KeyCode::Backspace => {
                    if !matches!(self.input_mode, InputMode::CreatingProfile) {
                        self.input_mode = InputMode::Editing;
                    }
                    self.input.backspace();
                    self.suggestion_index = 0;
                }
                KeyCode::Up => {
                    match vertical_navigation(&self.input_mode, self.suggestions().len()) {
                        VerticalNavigation::Selection(count) => {
                            self.suggestion_index =
                                previous_suggestion(self.suggestion_index, count);
                        }
                        VerticalNavigation::History => {
                            let width = self.terminal.size()?.width.saturating_sub(4).max(1);
                            if !self.input.move_up(width) {
                                self.navigate_history_up();
                            }
                        }
                        VerticalNavigation::Disabled => {}
                    }
                }
                KeyCode::Down => {
                    match vertical_navigation(&self.input_mode, self.suggestions().len()) {
                        VerticalNavigation::Selection(count) => {
                            self.suggestion_index = (self.suggestion_index + 1) % count;
                        }
                        VerticalNavigation::History => {
                            let width = self.terminal.size()?.width.saturating_sub(4).max(1);
                            if !self.input.move_down(width) {
                                self.navigate_history_down();
                            }
                        }
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
                    } else if !self.processing {
                        if key.code == KeyCode::Left {
                            self.input.move_left();
                        } else {
                            self.input.move_right();
                        }
                    }
                }
                KeyCode::BackTab => {
                    if self.processing {
                        return Ok(());
                    }
                    let previous = self.plan_mode;
                    self.plan_mode = !previous;
                    self.pending_plan_toggle = Some(previous);
                    let action = TuiAction::TogglePlan;
                    self.processing = true;
                    self.cancellation_requested = false;
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
                    if self.input.text().starts_with('/') {
                        let matches = slash_suggestions(self.input.text());
                        if !matches.is_empty() {
                            let index = self.suggestion_index.min(matches.len() - 1);
                            self.input.set_text(matches[index].0.to_owned());
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
                        && let Some(c) = completions
                            .iter()
                            .find(|c| c.starts_with(self.input.text()))
                    {
                        self.input.set_text(c.to_string());
                    }
                }
                _ => {}
            },
            Event::Resize(_, _) => {}
            _ => {}
        }

        Ok(())
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

fn message_lines(message: &Msg) -> Vec<Line<'static>> {
    let style = match message.style {
        MsgStyle::User => Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
        MsgStyle::ToolCall => Style::default().fg(Color::Green),
        MsgStyle::Observation => Style::default().fg(Color::DarkGray),
        MsgStyle::Response => Style::default().fg(Color::White),
        MsgStyle::Error => Style::default().fg(Color::Red),
        MsgStyle::System => Style::default().fg(Color::Blue).add_modifier(Modifier::DIM),
    };
    message
        .text
        .split('\n')
        .map(|line| Line::from(Span::styled(line.to_owned(), style)))
        .collect()
}

fn history_lines_height(lines: &[Line<'static>], width: u16) -> u16 {
    lines
        .iter()
        .map(|line| line.width().max(1).div_ceil(usize::from(width.max(1))))
        .sum::<usize>()
        .min(usize::from(u16::MAX)) as u16
}

fn startup_panel_lines(info: &StartupInfo, width: u16) -> Vec<Line<'static>> {
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

fn is_insert_newline_key(key: KeyEvent) -> bool {
    (key.code == KeyCode::Enter && key.modifiers.contains(KeyModifiers::SHIFT))
        || (key.code == KeyCode::Char('j') && key.modifiers.contains(KeyModifiers::CONTROL))
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
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    use crate::engine::ProfileChoice;

    use super::{
        ApprovalChoice, EmptyModeChoice, InputMode, Msg, MsgStyle, ProfilePickerChoice, TuiAction,
        TuiPicker, VerticalNavigation, approval_choice, empty_mode_choice, history_down,
        history_lines_height, history_up, is_insert_newline_key, is_profile_name_character,
        message_lines, picker_viewport, previous_suggestion, profile_picker_choice,
        push_immediate_response, slash_suggestions, suggestions_for, terminal_width,
        vertical_navigation,
    };

    #[test]
    fn terminal_width_uses_display_columns_for_unicode_input() {
        assert_eq!(terminal_width("hello"), 5);
        assert_eq!(terminal_width("中文"), 4);
        assert_eq!(terminal_width("a中"), 3);
    }

    #[test]
    fn history_height_uses_display_width_for_mixed_cjk_text() {
        let message = Msg {
            style: MsgStyle::Response,
            text: "➔ 翻译此文件：<i>[Robert, the 17th Earl of Bruce:]</i>".to_owned(),
            stamp: String::new(),
        };
        let lines = message_lines(&message);
        assert_eq!(history_lines_height(&lines, 40), 2);
        assert!(lines[0].width() > 40);
    }

    #[test]
    fn shift_enter_and_control_j_are_newline_keys() {
        assert!(is_insert_newline_key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::SHIFT,
        )));
        assert!(is_insert_newline_key(KeyEvent::new(
            KeyCode::Char('j'),
            KeyModifiers::CONTROL,
        )));
        assert!(!is_insert_newline_key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )));
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
    fn profile_creation_is_a_typed_action_and_picker_choice() {
        let action = TuiAction::CreateProfile("review_copy".to_owned());
        assert_eq!(action, TuiAction::CreateProfile("review_copy".to_owned()));
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
    fn response_is_committed_as_one_complete_message() {
        let mut view = super::MsgView::new(10);
        push_immediate_response(&mut view, "one.srt\ntwo.srt".to_owned());
        assert_eq!(view.all().len(), 1);
        assert_eq!(view.all()[0].text, "➔ one.srt\ntwo.srt");
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
