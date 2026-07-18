//! Chat-style inline TUI: committed output is written to the terminal's native
//! scrollback while the composer and active picker are redrawn below it.
//!
//! Layout (inspired by Codex / OpenCode):
//!
//! ┌─────────────────────────────────┐
//! │  Terminal-native scrollback     │
//! │  [You] translate hello.srt      │
//! │  ⚡ translate_file ✓             │
//! │  ➔ Translated: out.srt          │
//! │  ...                            │
//! ├─────────────────────────────────┤
//! │ > _                             │
//! └─────────────────────────────────┘

use std::io;
use std::sync::mpsc;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget, Wrap};
use ratatui::{Terminal, TerminalOptions, Viewport};
use unicode_width::UnicodeWidthStr;

use crate::engine::SessionChoice;
use crate::error::AgentResult;
use crate::input_editor::InputEditor;
use crate::tui_state::{
    APPROVAL_OPTIONS, EmptyModeChoice, InputMode, InteractionState, SessionPicker, TuiPicker,
    VerticalNavigation, empty_mode_choice, history_down, history_up, vertical_navigation,
};
use subbake_core::{CancellationGuard, CancellationToken};
use subbake_core::{ProgressEvent, TaskState};

mod history;
mod input_router;
mod progress;
mod protocol;
mod render;
mod terminal;
mod worker;

pub use history::{Msg, MsgStyle, MsgView, TuiObserver};
use progress::format_progress;
pub use protocol::{StartupInfo, TuiAction, TuiInteraction};
use terminal::TerminalSessionGuard;
use worker::{TuiWorker, WorkerRequest};

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

// ---------------------------------------------------------------------------
// TUI App
// ---------------------------------------------------------------------------

pub struct SubBakeTui {
    terminal_session: TerminalSessionGuard,
    terminal: Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    overlay_terminal: Option<Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>>,
    msg_view: std::sync::Arc<std::sync::Mutex<MsgView>>,
    progress: std::sync::Arc<std::sync::Mutex<Option<(ProgressEvent, std::time::Instant)>>>,
    input: InputEditor,
    input_history: Vec<String>,
    running: bool,
    suggestion_index: usize,
    interaction_state: InteractionState,
    cancellation: Option<CancellationToken>,
    input_hint: &'static str,
    startup_info: StartupInfo,
    plan_mode: bool,
    history_cursor: usize,
    startup_pending: bool,
}

impl SubBakeTui {
    pub fn new() -> io::Result<Self> {
        let terminal_session = TerminalSessionGuard::enter()?;
        let backend = ratatui::backend::CrosstermBackend::new(io::stdout());
        let terminal_rows = crossterm::terminal::size()?.1;
        let terminal = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(inline_viewport_height(terminal_rows)),
            },
        )?;
        Ok(Self {
            terminal_session,
            terminal,
            overlay_terminal: None,
            // The terminal emulator owns scrollback retention. Keep the source
            // items for this process lifetime so the commit cursor stays stable.
            msg_view: std::sync::Arc::new(std::sync::Mutex::new(MsgView::new(usize::MAX))),
            progress: std::sync::Arc::new(std::sync::Mutex::new(None)),
            input: InputEditor::default(),
            input_history: Vec::new(),
            running: true,
            suggestion_index: 0,
            interaction_state: InteractionState::default(),
            cancellation: None,
            input_hint: session_input_hint(),
            startup_info: StartupInfo::default(),
            plan_mode: false,
            history_cursor: 0,
            startup_pending: true,
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
        TuiObserver::new(self.msg_view.clone(), self.progress.clone())
    }

    pub fn set_cancellation_token(&mut self, token: CancellationToken) {
        self.cancellation = Some(token);
    }

    fn commit_progress_summary(&mut self) {
        let completed = self.progress.lock().ok().and_then(|value| value.clone());
        let Some((event, started)) = completed else {
            return;
        };
        if !matches!(
            event.state,
            TaskState::Completed | TaskState::Cancelled | TaskState::Failed
        ) {
            return;
        }
        let marker = match event.state {
            TaskState::Completed => "✓",
            TaskState::Cancelled => "■",
            TaskState::Failed => "✖",
            _ => return,
        };
        if let Ok(mut view) = self.msg_view.lock() {
            view.push(
                MsgStyle::System,
                format!("{marker} {}", format_progress(&event, started.elapsed())),
            );
        }
    }

    pub fn set_input_history(&mut self, history: Vec<String>) {
        self.input_history = history;
        self.interaction_state.set_input_mode(InputMode::Editing);
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
                let (style, text) = match event.tag() {
                    crate::session::EventTag::User => (
                        MsgStyle::User,
                        format!("[{}] {}", event.created_at, event.text),
                    ),
                    crate::session::EventTag::Assistant | crate::session::EventTag::AskUser => {
                        (MsgStyle::Response, format!("➔ {}", event.text))
                    }
                    crate::session::EventTag::ToolCall => {
                        (MsgStyle::ToolCall, format!("⚡ {}", event.text))
                    }
                    crate::session::EventTag::FileOperation => {
                        (MsgStyle::Observation, format!("◀ {}", event.text))
                    }
                    crate::session::EventTag::Plan => {
                        (MsgStyle::System, format!("Plan: {}", event.text))
                    }
                    crate::session::EventTag::Error => {
                        (MsgStyle::Error, format!("✖ {}", event.text))
                    }
                    crate::session::EventTag::Cancelled => {
                        (MsgStyle::System, "Cancelled.".to_owned())
                    }
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
        self.input.clear();
        self.interaction_state
            .set_input_mode(InputMode::ChoosingSession(SessionPicker {
                options,
                cancel_exits: true,
            }));
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
    pub fn run<F>(&mut self, process_fn: F) -> io::Result<()>
    where
        F: FnMut(TuiAction, CancellationGuard, &mut TuiObserver) -> AgentResult<TuiInteraction>
            + Send
            + 'static,
    {
        let mut worker = TuiWorker::spawn(process_fn, self.observer())?;

        let loop_result = (|| -> io::Result<()> {
            while self.running {
                if let Ok(result) = worker.try_recv() {
                    self.commit_progress_summary();
                    let plan_mode_rollback = self.interaction_state.finish();
                    match result {
                        Ok(TuiInteraction::Message { message }) => {
                            self.interaction_state.set_input_mode(InputMode::Editing);
                            self.suggestion_index = 0;
                            self.render_response(message);
                        }
                        Ok(TuiInteraction::PlanApproval { message }) => {
                            self.interaction_state
                                .set_input_mode(InputMode::AwaitingPlanDecision);
                            self.suggestion_index = 0;
                            self.render_response(message);
                        }
                        Ok(TuiInteraction::ProfilePicker { message, options }) => {
                            self.open_fullscreen_overlay()?;
                            self.interaction_state
                                .set_input_mode(InputMode::ChoosingProfile(TuiPicker { options }));
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
                            self.interaction_state.set_input_mode(InputMode::Editing);
                            self.suggestion_index = 0;
                            self.set_session_replay(events);
                            self.plan_mode = plan_mode;
                            self.startup_info.model = model;
                        }
                        Ok(TuiInteraction::SessionPicker { message, options }) => {
                            self.open_fullscreen_overlay()?;
                            self.interaction_state
                                .set_input_mode(InputMode::ChoosingSession(SessionPicker {
                                    options,
                                    cancel_exits: false,
                                }));
                            self.suggestion_index = 0;
                            // `/sessions` opens a picker; its textual summary would only
                            // duplicate the rows already visible in that picker.
                            let _ = message;
                        }
                        Ok(TuiInteraction::PlanModeChanged { enabled }) => {
                            self.plan_mode = enabled;
                            self.interaction_state.set_input_mode(InputMode::Editing);
                            self.suggestion_index = 0;
                        }
                        Ok(TuiInteraction::ModelChanged { model, message }) => {
                            self.startup_info.model = model;
                            self.interaction_state.set_input_mode(InputMode::Editing);
                            self.suggestion_index = 0;
                            self.render_response(message);
                        }
                        Err(error) => {
                            if let Some(previous) = plan_mode_rollback {
                                self.plan_mode = previous;
                            }
                            self.interaction_state.set_input_mode(InputMode::Editing);
                            if let Ok(mut view) = self.msg_view.lock() {
                                if error.is_cancelled() {
                                    self.interaction_state.set_input_mode(InputMode::Editing);
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
                self.handle_event(worker.sender()?)?;
            }
            Ok(())
        })();

        let overlay_result = self.close_fullscreen_overlay();
        let clear_result = self.terminal.clear();
        let cursor_result = self.terminal.show_cursor();
        let terminal_result = self.terminal_session.restore();
        let worker_result = worker.shutdown();

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
        suggestions_for(self.input.text(), self.interaction_state.input_mode())
    }

    fn navigate_history_up(&mut self) {
        let Some((mode, input)) = history_up(
            &self.input_history,
            self.input.text(),
            self.interaction_state.input_mode(),
        ) else {
            return;
        };
        self.interaction_state.set_input_mode(mode);
        self.input.set_text(input);
    }

    fn navigate_history_down(&mut self) {
        let Some((mode, input)) =
            history_down(&self.input_history, self.interaction_state.input_mode())
        else {
            return;
        };
        self.interaction_state.set_input_mode(mode);
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
        render::draw(self)
    }

    fn handle_event(&mut self, request_tx: &mpsc::Sender<WorkerRequest>) -> io::Result<()> {
        input_router::handle_event(self, request_tx)
    }
}

fn inline_viewport_height(terminal_rows: u16) -> u16 {
    terminal_rows.saturating_sub(1).clamp(1, 12)
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

fn push_immediate_response(view: &mut MsgView, text: String) {
    if !text.is_empty() {
        view.push(MsgStyle::Response, format!("➔ {text}"));
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
    use crate::tui_state::{
        ApprovalChoice, InteractionState, ProfilePickerChoice, approval_choice,
        profile_picker_choice,
    };

    use super::{
        EmptyModeChoice, InputMode, Msg, MsgStyle, TuiAction, TuiPicker, VerticalNavigation,
        empty_mode_choice, history_down, history_lines_height, history_up, is_insert_newline_key,
        is_profile_name_character, message_lines, picker_viewport, previous_suggestion,
        push_immediate_response, slash_suggestions, suggestions_for, terminal_width,
        vertical_navigation,
    };

    #[test]
    fn inline_viewport_leaves_room_for_native_scrollback() {
        assert_eq!(super::inline_viewport_height(40), 12);
        assert_eq!(super::inline_viewport_height(12), 11);
        assert_eq!(super::inline_viewport_height(2), 1);
        assert_eq!(super::inline_viewport_height(1), 1);
    }

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
    fn interaction_state_tracks_cancellation_as_a_typed_transition() {
        let mut phase = InteractionState::default();
        assert!(!phase.request_cancellation());

        phase.begin_processing(None);
        assert!(phase.is_processing());
        assert!(phase.request_cancellation());
        assert!(!phase.request_cancellation());
        assert_eq!(phase.finish(), None);
        assert!(matches!(phase, InteractionState::Idle { .. }));
    }

    #[test]
    fn interaction_state_returns_plan_mode_rollback_only_when_finishing() {
        let mut phase = InteractionState::default();
        phase.begin_processing(Some(false));

        assert_eq!(phase.finish(), Some(false));
        assert!(matches!(phase, InteractionState::Idle { .. }));
        assert_eq!(phase.finish(), None);
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

    #[test]
    fn file_preview_observation_is_not_added_to_visible_history() {
        use crate::engine::EngineObserver;
        let view = std::sync::Arc::new(std::sync::Mutex::new(super::MsgView::new(10)));
        let progress = std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut observer = super::TuiObserver::new(view.clone(), progress);
        observer.on_tool_call(
            "read_file_preview",
            &serde_json::json!({"path":"sample.srt"}),
        );
        observer.on_observation("subtitle body");
        let messages = view.lock().expect("view");
        assert_eq!(messages.all().len(), 1);
        assert!(!messages.all()[0].text.contains("subtitle body"));
    }

    #[test]
    fn progress_line_reports_resume_tokens_and_counts() {
        let mut event = subbake_core::ProgressEvent::running(
            subbake_core::TaskKind::Translation,
            "TRANSLATE",
            2,
            Some(4),
            subbake_core::ProgressUnit::Batches,
        );
        event.resumed = 1;
        event.usage.input_tokens = 20;
        event.usage.output_tokens = 10;
        let line = super::format_progress(&event, std::time::Duration::from_secs(3));
        assert!(line.contains("2/4"));
        assert!(line.contains("20/10 tok"));
        assert!(line.contains("resumed 1"));
    }

    #[test]
    fn duration_progress_line_reports_percentage_and_media_time() {
        let event = subbake_core::ProgressEvent::running(
            subbake_core::TaskKind::Transcription,
            "PREPARE_AUDIO",
            90_000,
            Some(180_000),
            subbake_core::ProgressUnit::Duration,
        );
        let line = super::format_progress(&event, std::time::Duration::from_secs(3));
        assert!(line.contains("50.0%"));
        assert!(line.contains("1:30/3:00"));
    }

    #[test]
    fn percent_progress_line_reports_a_percentage() {
        let event = subbake_core::ProgressEvent::running(
            subbake_core::TaskKind::Transcription,
            "TRANSCRIBE",
            25,
            Some(100),
            subbake_core::ProgressUnit::Percent,
        );
        let line = super::format_progress(&event, std::time::Duration::from_secs(3));
        assert!(line.contains("25.0%"));
        assert!(line.contains("[██────────]"));
    }
}
