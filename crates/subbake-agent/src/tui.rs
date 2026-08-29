//! Chat-style inline TUI: committed output is written to the terminal's native
//! scrollback while the composer and active picker are redrawn below it.
//!
//! Layout:
//!
//! ┌─────────────────────────────────┐
//! │  Terminal-native scrollback     │
//! │  › translate hello.srt          │
//! │  · I will translate the file.   │
//! │    └─ ✓ Translated hello.srt    │
//! │  • Output: hello.zh-CN.srt      │
//! │  ...                            │
//! ├─────────────────────────────────┤
//! │ > _                             │
//! └─────────────────────────────────┘

use std::collections::VecDeque;
use std::io;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crossterm::cursor::MoveTo;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{Clear, ClearType};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget, Wrap};
use ratatui::{Terminal, TerminalOptions, Viewport};

use crate::engine::{ApprovalPrompt, SessionChoice};
use crate::error::AgentResult;
use crate::input_editor::InputEditor;
use crate::steering::TurnSteering;
use crate::tui_state::{
    APPROVAL_OPTIONS, ConfigEditorState, EmptyModeChoice, InputMode, InteractionState,
    SessionPicker, TuiPicker, VerticalNavigation, empty_mode_choice, history_down, history_up,
    vertical_navigation,
};
use subbake_core::{CancellationGuard, CancellationToken};
use subbake_core::{ProgressEvent, TaskState};

mod history;
mod input_router;
mod layout;
mod main_view;
mod overlay_view;
mod progress;
mod protocol;
mod render;
mod terminal;
mod text;
mod transcript;
mod worker;

use history::ActiveTool;
pub use history::{
    Msg, MsgStyle, MsgView, ToolActivity, ToolActivityStatus, ToolGroup, TranscriptItem,
    TuiObserver,
};
use layout::ActiveLayout;
use progress::format_progress;
pub use protocol::{ConfigApplyAfter, StartupInfo, TuiAction, TuiInteraction};
use terminal::TerminalSessionGuard;
use text::{display_width, truncate_with_ellipsis};
use transcript::{history_lines_height, transcript_item_lines};
use worker::{TuiWorker, WorkerRequest};

const EVENT_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);
const RESIZE_REFLOW_DEBOUNCE: Duration = Duration::from_millis(75);

#[derive(Debug, Clone)]
struct ResizeReflowState {
    last_observed: (u16, u16),
    last_rebuilt: (u16, u16),
    pending: Option<((u16, u16), Instant)>,
}

impl ResizeReflowState {
    fn new(size: (u16, u16)) -> Self {
        Self {
            last_observed: size,
            last_rebuilt: size,
            pending: None,
        }
    }

    fn observe(&mut self, size: (u16, u16), now: Instant) {
        if size != self.last_observed {
            self.last_observed = size;
            self.pending =
                (size != self.last_rebuilt).then_some((size, now + RESIZE_REFLOW_DEBOUNCE));
        }
    }

    fn due_size(&self, now: Instant) -> Option<(u16, u16)> {
        self.pending
            .filter(|(size, deadline)| *size != self.last_rebuilt && now >= *deadline)
            .map(|(size, _)| size)
    }

    fn rebuilt(&mut self, size: (u16, u16)) {
        self.last_rebuilt = size;
        self.pending = None;
    }
}

// ---------------------------------------------------------------------------
// TUI App
// ---------------------------------------------------------------------------

pub struct SubBakeTui {
    terminal_session: TerminalSessionGuard,
    terminal: Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    resize_reflow: ResizeReflowState,
    overlay_terminal: Option<Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>>,
    msg_view: std::sync::Arc<std::sync::Mutex<MsgView>>,
    progress: std::sync::Arc<std::sync::Mutex<Option<(ProgressEvent, std::time::Instant)>>>,
    active_tool: std::sync::Arc<std::sync::Mutex<Option<ActiveTool>>>,
    input: InputEditor,
    input_history: Vec<String>,
    pending_inputs: VecDeque<String>,
    running: bool,
    suggestion_index: usize,
    interaction_state: InteractionState,
    cancellation: Option<CancellationToken>,
    turn_steering: Option<TurnSteering>,
    input_hint: &'static str,
    startup_info: StartupInfo,
    plan_mode: bool,
    history_cursor: usize,
    startup_pending: bool,
    config_editor: Option<ConfigEditorState>,
    approval_prompt: Option<ApprovalPrompt>,
    active_layout: Option<ActiveLayout>,
}

impl SubBakeTui {
    pub fn new() -> io::Result<Self> {
        let terminal_session = TerminalSessionGuard::enter()?;
        let inline_terminal_size = crossterm::terminal::size()?;
        let terminal = create_inline_terminal(inline_terminal_size.1)?;
        Ok(Self {
            terminal_session,
            terminal,
            resize_reflow: ResizeReflowState::new(inline_terminal_size),
            overlay_terminal: None,
            // The terminal emulator owns scrollback retention. Keep the source
            // items for this process lifetime so the commit cursor stays stable.
            msg_view: std::sync::Arc::new(std::sync::Mutex::new(MsgView::new(usize::MAX))),
            progress: std::sync::Arc::new(std::sync::Mutex::new(None)),
            active_tool: std::sync::Arc::new(std::sync::Mutex::new(None)),
            input: InputEditor::default(),
            input_history: Vec::new(),
            pending_inputs: VecDeque::new(),
            running: true,
            suggestion_index: 0,
            interaction_state: InteractionState::default(),
            cancellation: None,
            turn_steering: None,
            input_hint: session_input_hint(),
            startup_info: StartupInfo::default(),
            plan_mode: false,
            history_cursor: 0,
            startup_pending: true,
            config_editor: None,
            approval_prompt: None,
            active_layout: None,
        })
    }

    pub fn set_startup_info(&mut self, startup_info: StartupInfo) {
        self.startup_info = startup_info;
    }

    pub fn set_plan_mode(&mut self, enabled: bool) {
        self.plan_mode = enabled;
    }

    pub fn set_pending_approval(&mut self, prompt: Option<ApprovalPrompt>) {
        self.approval_prompt = prompt;
        if self.approval_prompt.is_some() {
            self.interaction_state
                .set_input_mode(InputMode::AwaitingApproval);
        }
    }

    pub fn set_has_config_file(&mut self, has_config_file: bool) {
        if !has_config_file {
            self.input_hint = "Use /profile to create a model profile";
        }
    }

    pub fn observer(&self) -> TuiObserver {
        TuiObserver::new(
            self.msg_view.clone(),
            self.progress.clone(),
            self.active_tool.clone(),
        )
    }

    pub fn set_cancellation_token(&mut self, token: CancellationToken) {
        self.cancellation = Some(token);
    }

    pub fn set_turn_steering(&mut self, steering: TurnSteering) {
        self.turn_steering = Some(steering);
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
            TaskState::Failed => "×",
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
            view.seal_tool_group();
            if !view.items.is_empty() {
                view.push(
                    MsgStyle::System,
                    "──────── resumed session ────────".to_owned(),
                );
            }
            view.replay(events);
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
        self.invalidate_layout();
        if self.overlay_terminal.is_none() {
            self.terminal_session.enter_alternate_screen()?;
            let backend = ratatui::backend::CrosstermBackend::new(io::stdout());
            self.overlay_terminal = Some(Terminal::new(backend)?);
        }
        Ok(())
    }

    fn close_fullscreen_overlay(&mut self) -> io::Result<()> {
        self.invalidate_layout();
        if let Some(mut terminal) = self.overlay_terminal.take() {
            terminal.clear()?;
            terminal.show_cursor()?;
        }
        self.terminal_session.leave_alternate_screen()
    }

    fn invalidate_layout(&mut self) {
        self.active_layout = None;
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
                    if let Ok(mut view) = self.msg_view.lock() {
                        view.seal_tool_group();
                    }
                    self.commit_progress_summary();
                    let plan_mode_rollback = self.interaction_state.finish();
                    match result {
                        Ok(TuiInteraction::Message { message }) => {
                            self.approval_prompt = None;
                            self.interaction_state.set_input_mode(InputMode::Editing);
                            self.suggestion_index = 0;
                            self.render_response(message);
                        }
                        Ok(TuiInteraction::Approval { prompt }) => {
                            self.approval_prompt = Some(prompt);
                            self.interaction_state
                                .set_input_mode(InputMode::AwaitingApproval);
                            self.suggestion_index = 0;
                        }
                        Ok(TuiInteraction::ProfilePicker { message, options }) => {
                            self.open_fullscreen_overlay()?;
                            self.interaction_state
                                .set_input_mode(InputMode::ChoosingProfile(TuiPicker { options }));
                            self.suggestion_index = 0;
                            let _ = message;
                        }
                        Ok(TuiInteraction::ConfigEditor {
                            message,
                            snapshot,
                            provider,
                            model,
                            cache_enabled,
                        }) => {
                            self.open_fullscreen_overlay()?;
                            self.startup_info.provider = provider;
                            self.startup_info.model = model;
                            self.startup_info.cache_enabled = cache_enabled;
                            self.startup_info.config = snapshot.path.display().to_string();
                            self.config_editor = Some(ConfigEditorState::new(snapshot));
                            self.interaction_state
                                .set_input_mode(InputMode::ConfigEditor);
                            self.suggestion_index = 0;
                            self.input.clear();
                            if !message.is_empty() {
                                self.render_response(message);
                            }
                        }
                        Ok(TuiInteraction::ConfigClosed {
                            message,
                            provider,
                            model,
                            cache_enabled,
                        }) => {
                            self.config_editor = None;
                            self.close_fullscreen_overlay()?;
                            self.startup_info.provider = provider;
                            self.startup_info.model = model;
                            self.startup_info.cache_enabled = cache_enabled;
                            self.interaction_state.set_input_mode(InputMode::Editing);
                            self.render_response(message);
                        }
                        Ok(TuiInteraction::SessionChanged {
                            input_history,
                            events,
                            plan_mode,
                            model,
                            approval,
                        }) => {
                            self.input_history = input_history;
                            self.approval_prompt = approval;
                            self.interaction_state.set_input_mode(
                                if self.approval_prompt.is_some() {
                                    InputMode::AwaitingApproval
                                } else {
                                    InputMode::Editing
                                },
                            );
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
                            self.interaction_state.set_input_mode(
                                if self.config_editor.is_some() {
                                    InputMode::ConfigEditor
                                } else {
                                    InputMode::Editing
                                },
                            );
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
                    input_router::submit_next_queued(self, worker.sender()?)?;
                }
                self.sync_inline_terminal_size()?;
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
  /profile [NAME] — list or switch profiles
  /model    — choose a model profile
  /config   —  edit configuration
  /undo     —  undo last file operation
  /sessions [ID] — choose or resume a saved session
  /history [LIMIT] — show recent history
  /clear    —  start a new session
  /exit /quit — exit

Or just type what you want, e.g. "translate @clip.srt""#
                .to_owned(),
            "/plan" | "/profile" | "/model" | "/config" | "/undo" | "/sessions" | "/history"
            | "/clear" => {
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
        self.sync_inline_terminal_size()?;
        if self.resize_reflow.pending.is_some() {
            return Ok(());
        }
        let width = self.terminal.size()?.width.max(1);
        if self.startup_pending {
            self.startup_pending = false;
            let lines = startup_panel_lines(&self.startup_info, width);
            self.insert_history_lines(lines, width)?;
        }

        let items = self
            .msg_view
            .lock()
            .map(|view| view.items[self.history_cursor.min(view.items.len())..].to_vec())
            .unwrap_or_default();
        if items.is_empty() {
            return Ok(());
        }
        self.history_cursor = self.history_cursor.saturating_add(items.len());
        let lines = items
            .iter()
            .flat_map(|item| transcript_item_lines(item, width, None))
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
        self.sync_inline_terminal_size()?;
        if self.overlay_terminal.is_none() && self.resize_reflow.pending.is_some() {
            return Ok(());
        }
        render::draw(self)
    }

    /// Rebuild the inline terminal from the saved transcript after resize
    /// events settle. Terminal-native wrapping cannot be reversed reliably,
    /// so the previous visible screen and scrollback are deliberately purged.
    fn sync_inline_terminal_size(&mut self) -> io::Result<()> {
        let size = crossterm::terminal::size()?;
        let now = Instant::now();
        self.resize_reflow.observe(size, now);
        if self.overlay_terminal.is_some() {
            return Ok(());
        }
        let Some(size) = self.resize_reflow.due_size(now) else {
            return Ok(());
        };
        crossterm::execute!(
            io::stdout(),
            Clear(ClearType::Purge),
            Clear(ClearType::All),
            MoveTo(0, 0)
        )?;
        self.terminal = create_inline_terminal(size.1)?;
        self.resize_reflow.rebuilt(size);
        self.history_cursor = 0;
        self.startup_pending = true;
        self.invalidate_layout();
        Ok(())
    }

    fn handle_event(&mut self, request_tx: &mpsc::Sender<WorkerRequest>) -> io::Result<()> {
        input_router::handle_event(self, request_tx)
    }
}

/// Inline terminal initialization asks the terminal for its cursor position.
/// A response can be lost while keyboard-protocol detection is handing the
/// input stream back to the normal event reader, so retry that handshake once.
/// Persistent terminal errors are still returned to the caller.
fn retry_terminal_initialization<T>(
    mut initialize: impl FnMut() -> io::Result<T>,
) -> io::Result<T> {
    initialize().or_else(|_| initialize())
}

fn create_inline_terminal(
    terminal_rows: u16,
) -> io::Result<Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>> {
    retry_terminal_initialization(|| {
        let backend = ratatui::backend::CrosstermBackend::new(io::stdout());
        Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(inline_viewport_height(terminal_rows)),
            },
        )
    })
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

fn startup_panel_lines(info: &StartupInfo, width: u16) -> Vec<Line<'static>> {
    let width = usize::from(width.max(4));
    let inner_width = width.saturating_sub(2);
    let value_style = Style::default().fg(Color::Cyan);
    let border_style = Style::default().fg(Color::DarkGray);
    let row = |label: &'static str, value: &str| {
        let prefix = format!("  {label:<10}");
        let available = inner_width.saturating_sub(display_width(&prefix));
        let value = truncate_with_ellipsis(value, available);
        let padding = " ".repeat(available.saturating_sub(display_width(&value)));
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
    let title = truncate_with_ellipsis(&format!("  SubBake v{}", info.version), inner_width);
    let title_padding = " ".repeat(inner_width.saturating_sub(display_width(&title)));
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
        InputMode::AwaitingApproval if input.is_empty() => APPROVAL_OPTIONS
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
        view.push_response(text);
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
    crate::engine::SLASH_COMMAND_SPECS
        .iter()
        .filter(|spec| spec.suggest && spec.command.starts_with(&query))
        .map(|spec| (spec.command, spec.description))
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

    use super::transcript::{message_lines, tool_group_lines};
    use super::{
        EmptyModeChoice, InputMode, Msg, MsgStyle, ResizeReflowState, StartupInfo,
        ToolActivityStatus, TranscriptItem, TuiAction, TuiPicker, VerticalNavigation,
        empty_mode_choice, history_down, history_lines_height, history_up, is_insert_newline_key,
        is_profile_name_character, picker_viewport, previous_suggestion, push_immediate_response,
        slash_suggestions, startup_panel_lines, suggestions_for, vertical_navigation,
    };

    #[test]
    fn inline_viewport_leaves_room_for_native_scrollback() {
        assert_eq!(super::inline_viewport_height(40), 12);
        assert_eq!(super::inline_viewport_height(12), 11);
        assert_eq!(super::inline_viewport_height(2), 1);
        assert_eq!(super::inline_viewport_height(1), 1);
    }

    #[test]
    fn resize_reflow_debounces_width_and_height_to_the_latest_size() {
        let start = std::time::Instant::now();
        let mut state = ResizeReflowState::new((100, 30));
        state.observe((40, 30), start);
        assert_eq!(
            state.due_size(start + std::time::Duration::from_millis(74)),
            None
        );
        state.observe((120, 30), start + std::time::Duration::from_millis(50));
        assert_eq!(
            state.due_size(start + std::time::Duration::from_millis(124)),
            None
        );
        assert_eq!(
            state.due_size(start + std::time::Duration::from_millis(125)),
            Some((120, 30))
        );
        state.rebuilt((120, 30));
        state.observe((120, 20), start + std::time::Duration::from_millis(130));
        assert_eq!(
            state.due_size(start + std::time::Duration::from_millis(205)),
            Some((120, 20))
        );
    }

    #[test]
    fn overdue_resize_remains_pending_until_the_overlay_can_close() {
        let start = std::time::Instant::now();
        let mut state = ResizeReflowState::new((80, 24));
        state.observe((40, 24), start);
        let due = start + std::time::Duration::from_millis(100);
        assert_eq!(state.due_size(due), Some((40, 24)));
        assert_eq!(
            state.due_size(due + std::time::Duration::from_secs(1)),
            Some((40, 24))
        );
    }

    #[test]
    fn resize_reflow_cancels_when_overlay_returns_to_the_rebuilt_size() {
        let start = std::time::Instant::now();
        let mut state = ResizeReflowState::new((120, 30));
        state.observe((40, 30), start);
        state.observe((120, 30), start + std::time::Duration::from_millis(50));

        assert_eq!(
            state.due_size(start + std::time::Duration::from_secs(1)),
            None
        );
        assert_eq!(state.pending, None);
    }

    #[test]
    fn terminal_initialization_retries_one_lost_query_response() {
        let mut attempts = 0;
        let value = super::retry_terminal_initialization(|| {
            attempts += 1;
            if attempts == 1 {
                Err(std::io::Error::other("lost terminal response"))
            } else {
                Ok("ready")
            }
        })
        .expect("second terminal initialization should succeed");

        assert_eq!(value, "ready");
        assert_eq!(attempts, 2);
    }

    #[test]
    fn terminal_initialization_returns_a_persistent_error_after_retry() {
        let mut attempts = 0;
        let error = super::retry_terminal_initialization::<()>(|| {
            attempts += 1;
            Err(std::io::Error::other("terminal unavailable"))
        })
        .expect_err("persistent terminal failure should be returned");

        assert_eq!(error.to_string(), "terminal unavailable");
        assert_eq!(attempts, 2);
    }

    #[test]
    fn startup_panel_uses_the_supplied_application_build_identity() {
        let info = StartupInfo {
            version: "0.2.0-alpha.1 (1234abcd, dirty)".to_owned(),
            ..StartupInfo::default()
        };
        let rendered = startup_panel_lines(&info, 80)
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(rendered.contains("SubBake v0.2.0-alpha.1 (1234abcd, dirty)"));
    }

    #[test]
    fn history_height_uses_display_width_for_mixed_cjk_text() {
        let message = Msg {
            style: MsgStyle::Response,
            text: "翻译此文件：<i>[Robert, the 17th Earl of Bruce:]</i>".to_owned(),
            stamp: String::new(),
        };
        let lines = message_lines(&message, 40);
        assert_eq!(history_lines_height(&lines, 40), 3);
        assert!(lines.iter().all(|line| line.width() <= 40));
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
        assert_eq!(slash_suggestions("/").len(), 11);
        assert_eq!(
            slash_suggestions("/con"),
            vec![("/config", "edit configuration")]
        );
        assert_eq!(
            slash_suggestions("/mod"),
            vec![("/model", "choose a model profile")]
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
            VerticalNavigation::Selection(11)
        );
        assert_eq!(
            vertical_navigation(&InputMode::AwaitingApproval, 3),
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
            suggestions_for("", &InputMode::AwaitingApproval)[0].0,
            "Approve once"
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
        let TranscriptItem::Message(message) = &view.all()[0] else {
            panic!("response message");
        };
        assert_eq!(message.text, "one.srt\ntwo.srt");
    }

    #[test]
    fn all_plan_approval_choices_have_distinct_typed_outcomes() {
        assert_eq!(
            approval_choice(0),
            ApprovalChoice::Submit(TuiAction::ApproveApproval)
        );
        assert_eq!(
            approval_choice(1),
            ApprovalChoice::Submit(TuiAction::RejectApproval)
        );
        assert_eq!(approval_choice(2), ApprovalChoice::Revise);
        assert_eq!(
            empty_mode_choice(&InputMode::AwaitingApproval, 0),
            Some(EmptyModeChoice::Submit(TuiAction::ApproveApproval))
        );
        assert_eq!(
            empty_mode_choice(&InputMode::AwaitingApproval, 1),
            Some(EmptyModeChoice::Submit(TuiAction::RejectApproval))
        );
        assert_eq!(
            empty_mode_choice(&InputMode::AwaitingApproval, 2),
            Some(EmptyModeChoice::ReviseApproval)
        );
    }

    #[test]
    fn file_preview_result_is_summarized_without_its_contents() {
        use crate::engine::EngineObserver;
        let view = std::sync::Arc::new(std::sync::Mutex::new(super::MsgView::new(10)));
        let progress = std::sync::Arc::new(std::sync::Mutex::new(None));
        let active_tool = std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut observer = super::TuiObserver::new(view.clone(), progress, active_tool.clone());
        observer.on_tool_call(
            "call-1",
            "read_file_preview",
            &serde_json::json!({"path":"sample.srt"}),
        );
        assert!(view.lock().expect("view").all().is_empty());
        assert_eq!(
            active_tool
                .lock()
                .expect("active tool")
                .as_ref()
                .map(|activity| activity.name.as_str()),
            Some("read_file_preview")
        );
        let outcome =
            subbake_core::AgentToolOutcome::Observation(subbake_core::ObservationToolOutcome {
                status: subbake_core::ToolExecutionStatus::Observed,
                observation: "read_file_preview".to_owned(),
                content: "subtitle body".to_owned(),
            });
        observer.on_tool_success(
            "call-1",
            "read_file_preview",
            &serde_json::json!({"path":"sample.srt"}),
            &outcome,
        );
        let messages = view.lock().expect("view");
        assert!(messages.all().is_empty());
        let group = messages.active_tool_group().expect("active tool group");
        assert_eq!(group.activities.len(), 1);
        assert_eq!(group.activities[0].status, ToolActivityStatus::Completed);
        assert!(group.activities[0].headline.contains("Read sample.srt"));
        assert!(
            !group.activities[0]
                .detail
                .as_deref()
                .unwrap_or_default()
                .contains("subtitle body")
        );
        let lines = tool_group_lines(group, 40, None);
        assert!(
            lines
                .iter()
                .any(|line| line.to_string().contains("└─ ✓ Read sample.srt"))
        );
        assert!(active_tool.lock().expect("active tool").is_none());
    }

    #[test]
    fn tool_group_uses_a_continuous_rail_and_width_aware_wrapping() {
        let mut view = super::MsgView::new(10);
        view.finish_tool(
            "command",
            "run_command",
            crate::tool_presentation::ToolActivityText {
                headline: "Ran iconv Captain.America.The.First.Avenger.source.srt".to_owned(),
                detail: Some("exit 0 · 0.1s".to_owned()),
            },
            ToolActivityStatus::Completed,
        );
        view.finish_tool(
            "translate",
            "translate_file",
            crate::tool_presentation::ToolActivityText {
                headline: "Translated Captain.America.The.First.Avenger.utf8.srt".to_owned(),
                detail: Some("→ output.zh-CN.srt · 1,842 cues · 42.8s".to_owned()),
            },
            ToolActivityStatus::Completed,
        );
        let group = view.active_tool_group().expect("active group");
        let lines = tool_group_lines(group, 30, None);
        let rendered = lines.iter().map(ToString::to_string).collect::<Vec<_>>();

        assert!(rendered[0].starts_with("  ┌─ ✓ Ran iconv"));
        assert!(
            rendered
                .iter()
                .any(|line| line.starts_with("  │    Captain"))
        );
        assert!(
            rendered
                .iter()
                .any(|line| line.starts_with("  └─ ✓ Translated"))
        );
        assert!(lines.iter().all(|line| line.width() <= 30));
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
