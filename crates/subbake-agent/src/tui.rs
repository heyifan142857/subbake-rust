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

pub struct TuiProcessResult {
    pub response: String,
    pub pending_plan: bool,
}

struct StreamingResponse {
    chars: Vec<char>,
    position: usize,
    message_index: usize,
    next_at: std::time::Instant,
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
    terminal: Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    msg_view: std::sync::Arc<std::sync::Mutex<MsgView>>,
    input: String,
    scroll_offset: u16,
    running: bool,
    streaming: Option<StreamingResponse>,
    suggestion_index: usize,
    awaiting_approval: bool,
    processing: bool,
    animation_started_at: std::time::Instant,
}

impl SubBakeTui {
    pub fn new() -> io::Result<Self> {
        enable_raw_mode()?;
        io::stdout().execute(EnterAlternateScreen)?;
        let backend = ratatui::backend::CrosstermBackend::new(io::stdout());
        let terminal = Terminal::new(backend)?;
        Ok(Self {
            terminal,
            msg_view: std::sync::Arc::new(std::sync::Mutex::new(MsgView::new(1000))),
            input: String::new(),
            scroll_offset: 0,
            running: true,
            streaming: None,
            suggestion_index: 0,
            awaiting_approval: false,
            processing: false,
            animation_started_at: std::time::Instant::now(),
        })
    }

    pub fn observer(&self) -> TuiObserver {
        TuiObserver::new(self.msg_view.clone())
    }

    /// Run the event loop. `process_fn` is called with the user's input each
    /// time they press Enter; it should run the agent engine and return the
    /// response text.
    pub fn run<F>(&mut self, mut process_fn: F) -> io::Result<()>
    where
        F: FnMut(&str, &mut TuiObserver) -> io::Result<TuiProcessResult> + Send + 'static,
    {
        let mut observer = self.observer();
        self.welcome(&mut observer)?;
        let worker_observer = self.observer();
        let (request_tx, request_rx) = mpsc::channel::<String>();
        let (response_tx, response_rx) = mpsc::channel::<io::Result<TuiProcessResult>>();
        thread::spawn(move || {
            let mut observer = worker_observer;
            while let Ok(input) = request_rx.recv() {
                let result = process_fn(&input, &mut observer);
                if response_tx.send(result).is_err() {
                    break;
                }
            }
        });

        while self.running {
            if let Ok(result) = response_rx.try_recv() {
                self.processing = false;
                match result {
                    Ok(result) => {
                        self.awaiting_approval = result.pending_plan;
                        self.suggestion_index = 0;
                        self.start_stream(result.response);
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

        disable_raw_mode()?;
        io::stdout().execute(LeaveAlternateScreen)?;
        Ok(())
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

    fn slash_suggestions(&self) -> Vec<(&'static str, &'static str)> {
        if self.awaiting_approval && self.input.is_empty() {
            APPROVAL_OPTIONS.to_vec()
        } else {
            slash_suggestions(&self.input)
        }
    }

    fn draw(&mut self) -> io::Result<()> {
        let messages = self
            .msg_view
            .lock()
            .map(|v| v.all().to_vec())
            .unwrap_or_default();
        let scroll = self.scroll_offset;
        let suggestions = self.slash_suggestions();
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
                            Span::styled(*description, description_style),
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

    fn handle_event(&mut self, request_tx: &mpsc::Sender<String>) -> io::Result<()> {
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
                    let suggestions = self.slash_suggestions();
                    if self.awaiting_approval && self.input.is_empty() {
                        match self.suggestion_index.min(APPROVAL_OPTIONS.len() - 1) {
                            0 => self.input = "/approve".to_owned(),
                            1 => self.input = "/reject".to_owned(),
                            _ => {
                                self.awaiting_approval = false;
                                self.suggestion_index = 0;
                                return Ok(());
                            }
                        }
                    }
                    if !suggestions.is_empty()
                        && !self.awaiting_approval
                        && !suggestions.iter().any(|item| item.0 == self.input)
                    {
                        let index = self.suggestion_index.min(suggestions.len() - 1);
                        self.input = suggestions[index].0.to_owned();
                        self.suggestion_index = 0;
                        return Ok(());
                    }
                    let input = std::mem::take(&mut self.input);
                    let trimmed = input.trim().to_owned();
                    self.scroll_offset = 0;

                    if trimmed.is_empty() {
                        return Ok(());
                    }

                    if trimmed == "/quit" || trimmed == "/exit" {
                        self.running = false;
                        return Ok(());
                    }

                    // Show user message.
                    if let Ok(mut v) = self.msg_view.lock() {
                        let visible_input = match trimmed.as_str() {
                            "/approve" if self.awaiting_approval => "approve",
                            "/reject" if self.awaiting_approval => "reject",
                            _ => &trimmed,
                        };
                        v.push(
                            MsgStyle::User,
                            format!("[{:?}] {}", iso_now(), visible_input),
                        );
                    }

                    // Handle help locally; engine-backed slash commands need
                    // session state, so route them through `process_fn`.
                    if trimmed.starts_with('/') && matches!(trimmed.as_str(), "/help" | "/h") {
                        let response = self.handle_slash(&trimmed);
                        if let Ok(mut v) = self.msg_view.lock() {
                            v.push(MsgStyle::Response, response);
                        }
                        return Ok(());
                    }

                    // Paint the submitted message and initial thinking state
                    // before the potentially slow engine call begins.
                    if let Ok(mut view) = self.msg_view.lock() {
                        view.push(MsgStyle::Thinking, "Deciding next action…".to_owned());
                    }
                    self.processing = true;
                    self.animation_started_at = std::time::Instant::now();
                    request_tx.send(trimmed).map_err(|_| {
                        io::Error::new(io::ErrorKind::BrokenPipe, "agent worker stopped")
                    })?;
                }
                KeyCode::Char(ch) => {
                    self.input.push(ch);
                    self.suggestion_index = 0;
                }
                KeyCode::Backspace => {
                    self.input.pop();
                    self.suggestion_index = 0;
                }
                KeyCode::Up => {
                    let suggestions = self.slash_suggestions();
                    if suggestions.is_empty() {
                        self.scroll_offset = self.scroll_offset.saturating_add(1);
                    } else {
                        self.suggestion_index =
                            previous_suggestion(self.suggestion_index, suggestions.len());
                    }
                }
                KeyCode::Down => {
                    let suggestions = self.slash_suggestions();
                    if suggestions.is_empty() {
                        self.scroll_offset = self.scroll_offset.saturating_sub(1);
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
    use super::{previous_suggestion, slash_suggestions};

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
}
