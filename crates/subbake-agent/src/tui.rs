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

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Terminal;

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
        if let Ok(mut v) = self.view.lock() {
            v.push(MsgStyle::Thinking, format!("⎿ {text}"));
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
        if let Ok(mut v) = self.view.lock() {
            v.push(MsgStyle::Response, format!("➔ {text}"));
        }
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
        F: FnMut(&str, &mut TuiObserver) -> io::Result<String>,
    {
        let mut observer = self.observer();
        self.welcome(&mut observer)?;

        while self.running {
            self.draw()?;
            self.handle_event(&mut observer, &mut process_fn)?;
        }

        disable_raw_mode()?;
        io::stdout().execute(LeaveAlternateScreen)?;
        Ok(())
    }

    fn handle_slash(&self, input: &str) -> String {
        match input {
            "/help" | "/h" => {
                r#"Commands:
  /help /h  —  this menu
  /plan     —  toggle plan mode
  /approve  —  approve pending plan
  /reject   —  reject pending plan
  /undo     —  undo last file operation
  /session  —  show session info
  /quit     —  exit

Or just type what you want, e.g. "translate @clip.srt""#
                    .to_owned()
            }
            "/plan" | "/approve" | "/reject" | "/undo" | "/session" => {
                format!("`{input}` is handled by the agent engine. When a real LLM backend is connected, these will route through the session.")
            }
            _ => {
                format!("Unknown command `{input}`. Try /help.")
            }
        }
    }

    fn welcome(&mut self, obs: &mut TuiObserver) -> io::Result<()> {
        obs.on_response("SubBake agent ready. Type a message or /help for commands.");
        Ok(())
    }

    fn draw(&mut self) -> io::Result<()> {
        let messages = self.msg_view.lock().map(|v| v.all().to_vec()).unwrap_or_default();
        let scroll = self.scroll_offset;

        self.terminal.draw(|frame| {
            let area = frame.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(area);

            // -- Output pane --
            let output_area = chunks[0];
            let items: Vec<Line<'_>> = messages.iter().map(|msg| {
                let style = match msg.style {
                    MsgStyle::User => Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                    MsgStyle::Thinking => Style::default().fg(Color::Yellow),
                    MsgStyle::ToolCall => Style::default().fg(Color::Green),
                    MsgStyle::Observation => Style::default().fg(Color::DarkGray),
                    MsgStyle::Response => Style::default().fg(Color::White),
                    MsgStyle::Error => Style::default().fg(Color::Red),
                    MsgStyle::System => Style::default().fg(Color::Blue).add_modifier(Modifier::DIM),
                };
                Line::from(Span::styled(&msg.text, style))
            }).collect();

            let max_scroll = (items.len() as u16).saturating_sub(output_area.height.saturating_sub(2));
            let offset = scroll.min(max_scroll);

            let paragraph = Paragraph::new(items)
                .block(Block::default().borders(Borders::NONE))
                .scroll((offset, 0));
            frame.render_widget(paragraph, output_area);

            // -- Input bar --
            let input_area = chunks[1];
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

    fn handle_event<F>(&mut self, observer: &mut TuiObserver, process_fn: &mut F) -> io::Result<()>
    where
        F: FnMut(&str, &mut TuiObserver) -> io::Result<String>,
    {
        if !event::poll(std::time::Duration::from_millis(100))? {
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
                        v.push(MsgStyle::User, format!("[{:?}] {}", iso_now(), trimmed));
                    }

                    // Handle slash commands locally.
                    if trimmed.starts_with('/') {
                        let response = self.handle_slash(&trimmed);
                        if let Ok(mut v) = self.msg_view.lock() {
                            v.push(MsgStyle::Response, response);
                        }
                        return Ok(());
                    }

                    // Process.
                    match process_fn(&trimmed, observer) {
                        Ok(response) => {
                            if let Ok(mut v) = self.msg_view.lock() {
                                v.push(MsgStyle::Response, response);
                            }
                        }
                        Err(e) => {
                            if let Ok(mut v) = self.msg_view.lock() {
                                v.push(MsgStyle::Error, format!("Error: {e}"));
                            }
                        }
                    }
                }
                KeyCode::Char(ch) => {
                    self.input.push(ch);
                }
                KeyCode::Backspace => {
                    self.input.pop();
                }
                KeyCode::Up => {
                    self.scroll_offset = self.scroll_offset.saturating_add(1);
                }
                KeyCode::Down => {
                    self.scroll_offset = self.scroll_offset.saturating_sub(1);
                }
                KeyCode::PageUp => {
                    self.scroll_offset = self.scroll_offset.saturating_add(10);
                }
                KeyCode::PageDown => {
                    self.scroll_offset = self.scroll_offset.saturating_sub(10);
                }
                KeyCode::Tab => {
                    // Simple completion: "tra" → "translate "
                    let completions = [
                        "translate ", "transcribe ", "list files", "read file",
                        "search files", "whisper ",
                    ];
                    if !self.input.is_empty()
                        && let Some(c) = completions.iter().find(|c| c.starts_with(&self.input)) {
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
