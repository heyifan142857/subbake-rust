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
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::engine::EngineObserver;
use crate::session::iso_now;

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

const STREAM_INTERVAL: std::time::Duration = std::time::Duration::from_millis(12);
const THINKING_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/help", "show commands"),
    ("/plan", "toggle plan mode"),
    ("/model", "show the active model"),
    ("/profile", "list or switch profiles"),
    ("/undo", "undo the last file operation"),
    ("/session", "show session info"),
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
    pub options: Vec<String>,
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
        options: Vec<String>,
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
    AwaitingPlanDecision,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuiAction {
    SubmitText(String),
    ApprovePlan,
    RejectPlan,
    SelectProfile(String),
}

impl TuiAction {
    fn visible_text(&self) -> String {
        match self {
            Self::SubmitText(text) => text.clone(),
            Self::ApprovePlan => "approve".to_owned(),
            Self::RejectPlan => "reject".to_owned(),
            Self::SelectProfile(name) => format!("/profile {name}"),
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
        let raw_result = disable_raw_mode();
        let screen_result = io::stdout().execute(LeaveAlternateScreen).map(|_| ());
        raw_result.and(screen_result)
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
        })
    }

    pub fn observer(&self) -> TuiObserver {
        TuiObserver::new(self.msg_view.clone())
    }

    pub fn set_input_history(&mut self, history: Vec<String>) {
        self.input_history = history;
        self.input_mode = InputMode::Editing;
    }

    /// Run the event loop. `process_fn` is called with the user's input each
    /// time they press Enter; it should run the agent engine and return the
    /// response text.
    pub fn run<F>(&mut self, mut process_fn: F) -> io::Result<()>
    where
        F: FnMut(TuiAction, &mut TuiObserver) -> io::Result<TuiInteraction> + Send + 'static,
    {
        let mut observer = self.observer();
        self.welcome(&mut observer)?;
        let worker_observer = self.observer();
        let (request_tx, request_rx) = mpsc::channel::<TuiAction>();
        let (response_tx, response_rx) = mpsc::channel::<io::Result<TuiInteraction>>();
        let worker = thread::Builder::new()
            .name("subbake-agent-worker".to_owned())
            .spawn(move || {
                let mut observer = worker_observer;
                while let Ok(action) = request_rx.recv() {
                    let result = process_fn(action, &mut observer);
                    if response_tx.send(result).is_err() {
                        break;
                    }
                }
            })?;

        let loop_result = (|| -> io::Result<()> {
            while self.running {
                if let Ok(result) = response_rx.try_recv() {
                    self.processing = false;
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
                            self.render_response(message, RenderPolicy::Immediate);
                        }
                        Err(error) => {
                            if let Ok(mut view) = self.msg_view.lock() {
                                view.push(MsgStyle::Error, format!("Error: {error}"));
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
  /plan     —  toggle plan mode
  /model    —  show active model
  /profile [NAME] — list or switch profiles
  /undo     —  undo last file operation
  /session  —  show session info
  /quit     —  exit

Or just type what you want, e.g. "translate @clip.srt""#
                .to_owned(),
            "/plan" | "/model" | "/profile" | "/undo" | "/session" => {
                format!(
                    "`{input}` is handled by the agent engine. When a real LLM backend is connected, these will route through the session."
                )
            }
            _ => {
                format!("Unknown command `{input}`. Try /help.")
            }
        }
    }

    fn welcome(&mut self, obs: &mut TuiObserver) -> io::Result<()> {
        if let Ok(mut v) = obs.view.lock() {
            v.push(
                MsgStyle::Response,
                "➔ SubBake agent ready. Type a message or /help for commands.".to_owned(),
            );
        }
        Ok(())
    }

    fn start_stream(&mut self, text: String) {
        if text.is_empty() {
            return;
        }
        self.finish_stream();
        if let Ok(mut view) = self.msg_view.lock() {
            view.push(MsgStyle::Response, "➔ ".to_owned());
            self.streaming = Some(StreamingResponse {
                chars: text.chars().collect(),
                position: 0,
                message_index: view.messages.len() - 1,
                next_at: std::time::Instant::now(),
            });
        }
    }

    fn render_response(&mut self, text: String, policy: RenderPolicy) {
        match policy {
            RenderPolicy::Stream => self.start_stream(text),
            RenderPolicy::Immediate => {
                self.finish_stream();
                if !text.is_empty()
                    && let Ok(mut view) = self.msg_view.lock()
                {
                    view.push(MsgStyle::Response, format!("➔ {text}"));
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
        match &self.input_mode {
            InputMode::BrowsingHistory { .. } => Vec::new(),
            InputMode::AwaitingPlanDecision if self.input.is_empty() => APPROVAL_OPTIONS
                .iter()
                .map(|(label, description)| ((*label).to_owned(), (*description).to_owned()))
                .collect(),
            InputMode::ChoosingProfile(picker) if self.input.is_empty() => picker
                .options
                .iter()
                .map(|option| (option.clone(), "configured profile".to_owned()))
                .collect(),
            _ => slash_suggestions(&self.input)
                .into_iter()
                .map(|(command, description)| (command.to_owned(), description.to_owned()))
                .collect(),
        }
    }

    fn draw(&mut self) -> io::Result<()> {
        let messages = self
            .msg_view
            .lock()
            .map(|v| v.all().to_vec())
            .unwrap_or_default();
        let scroll = self.scroll_offset;
        let suggestions = self.suggestions();
        let selected_suggestion = self
            .suggestion_index
            .min(suggestions.len().saturating_sub(1));
        let processing = self.processing;
        let animation_tick = self.animation_started_at.elapsed().as_millis() / 80;

        self.terminal.draw(|frame| {
            let area = frame.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(1),
                    Constraint::Length(suggestions.len() as u16),
                    Constraint::Length(3),
                ])
                .split(area);

            // -- Output pane --
            let output_area = chunks[0];
            let items: Vec<Line<'_>> = messages
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
                .collect();

            let visual_line_count = items
                .iter()
                .map(|line| {
                    let width = usize::from(output_area.width.max(1));
                    line.width().max(1).div_ceil(width)
                })
                .sum::<usize>();

            let paragraph = Paragraph::new(items)
                .block(Block::default().borders(Borders::NONE))
                .wrap(Wrap { trim: false });
            let max_scroll = (visual_line_count as u16).saturating_sub(output_area.height);
            // `scroll_offset` is the number of visual lines above the newest
            // output, so zero follows a growing/streaming response.
            let offset = max_scroll.saturating_sub(scroll.min(max_scroll));
            let paragraph = paragraph.scroll((offset, 0));
            frame.render_widget(paragraph, output_area);

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
            let input_style = Style::default().fg(Color::Cyan);
            let input_widget = Paragraph::new(Line::from(Span::styled(
                format!("> {}", self.input),
                input_style,
            )))
            .block(
                Block::default()
                    .borders(Borders::TOP)
                    .border_style(Style::default().fg(Color::DarkGray)),
            );
            frame.render_widget(input_widget, input_area);

            // Set cursor position at end of input.
            frame.set_cursor_position((
                input_area.x + 2 + self.input.len() as u16,
                input_area.y + 1,
            ));
        })?;

        Ok(())
    }

    fn handle_event(&mut self, request_tx: &mpsc::Sender<TuiAction>) -> io::Result<()> {
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
                KeyCode::Enter => {
                    if self.processing {
                        return Ok(());
                    }
                    let suggestions = self.suggestions();
                    let selected_action =
                        if matches!(self.input_mode, InputMode::AwaitingPlanDecision)
                            && self.input.is_empty()
                        {
                            match self.suggestion_index.min(APPROVAL_OPTIONS.len() - 1) {
                                0 => Some(TuiAction::ApprovePlan),
                                1 => Some(TuiAction::RejectPlan),
                                _ => {
                                    self.input_mode = InputMode::Editing;
                                    self.suggestion_index = 0;
                                    return Ok(());
                                }
                            }
                        } else if self.input.is_empty()
                            && let InputMode::ChoosingProfile(picker) = &self.input_mode
                            && !picker.options.is_empty()
                        {
                            let index = self.suggestion_index.min(picker.options.len() - 1);
                            Some(TuiAction::SelectProfile(picker.options[index].clone()))
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
                        TuiAction::SubmitText(trimmed)
                    };
                    self.scroll_offset = 0;

                    // Show user message.
                    if let Ok(mut v) = self.msg_view.lock() {
                        v.push(
                            MsgStyle::User,
                            format!("[{:?}] {}", iso_now(), action.visible_text()),
                        );
                    }

                    // Paint the submitted message and initial thinking state
                    // before the potentially slow engine call begins.
                    if let Ok(mut view) = self.msg_view.lock() {
                        view.push(MsgStyle::Thinking, "Deciding next action…".to_owned());
                    }
                    self.processing = true;
                    self.animation_started_at = std::time::Instant::now();
                    request_tx.send(action).map_err(|_| {
                        io::Error::new(io::ErrorKind::BrokenPipe, "agent worker stopped")
                    })?;
                }
                KeyCode::Char(ch) => {
                    self.input_mode = InputMode::Editing;
                    self.input.push(ch);
                    self.suggestion_index = 0;
                }
                KeyCode::Backspace => {
                    self.input_mode = InputMode::Editing;
                    self.input.pop();
                    self.suggestion_index = 0;
                }
                KeyCode::Up => {
                    let suggestions = self.suggestions();
                    if suggestions.is_empty() {
                        self.navigate_history_up();
                    } else {
                        self.suggestion_index =
                            previous_suggestion(self.suggestion_index, suggestions.len());
                    }
                }
                KeyCode::Down => {
                    let suggestions = self.suggestions();
                    if suggestions.is_empty() {
                        self.navigate_history_down();
                    } else {
                        self.suggestion_index = (self.suggestion_index + 1) % suggestions.len();
                    }
                }
                KeyCode::PageUp => {
                    self.scroll_offset = self.scroll_offset.saturating_add(10);
                }
                KeyCode::PageDown => {
                    self.scroll_offset = self.scroll_offset.saturating_sub(10);
                }
                KeyCode::Tab => {
                    if self.input.starts_with('/') {
                        let matches = slash_suggestions(&self.input);
                        if matches.len() == 1 {
                            self.input = matches[0].0.to_owned();
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

    fn navigate_history_up(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        let (index, draft) = match &self.input_mode {
            InputMode::BrowsingHistory { index, draft } => (index.saturating_sub(1), draft.clone()),
            _ => (self.input_history.len() - 1, self.input.clone()),
        };
        self.input_mode = InputMode::BrowsingHistory { index, draft };
        self.input.clone_from(&self.input_history[index]);
    }

    fn navigate_history_down(&mut self) {
        let InputMode::BrowsingHistory { index, draft } = &self.input_mode else {
            return;
        };
        let index = *index;
        let draft = draft.clone();
        if index + 1 < self.input_history.len() {
            let next = index + 1;
            self.input_mode = InputMode::BrowsingHistory { index: next, draft };
            self.input.clone_from(&self.input_history[next]);
        } else {
            self.input_mode = InputMode::Editing;
            self.input = draft;
        }
    }
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

#[cfg(test)]
mod tests {
    use super::{TuiAction, previous_suggestion, slash_suggestions};

    #[test]
    fn slash_displays_all_commands_and_filters_as_the_user_types() {
        assert_eq!(slash_suggestions("/").len(), 7);
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
    fn typed_profile_action_has_a_stable_visible_form() {
        let action = TuiAction::SelectProfile("strict".to_owned());
        assert_eq!(action.visible_text(), "/profile strict");
    }
}
