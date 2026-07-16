use std::io;
use std::sync::mpsc;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use subbake_core::{CancellationToken, TaskState};

use super::worker::WorkerRequest;
use super::{
    EmptyModeChoice, InputMode, MsgStyle, SessionPicker, SubBakeTui, TuiAction, VerticalNavigation,
    empty_mode_choice, is_insert_newline_key, is_profile_name_character, previous_suggestion,
    slash_suggestions, vertical_navigation,
};
use crate::session::iso_now;

pub(super) fn handle_event(
    app: &mut SubBakeTui,
    request_tx: &mpsc::Sender<WorkerRequest>,
) -> io::Result<()> {
    if !event::poll(super::EVENT_POLL_INTERVAL)? {
        return Ok(());
    }
    match event::read()? {
        Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.running = false;
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.running = false;
            }
            KeyCode::Char('q') if app.input.is_empty() => {
                app.running = false;
            }
            KeyCode::Esc => handle_escape(app)?,
            _ if is_insert_newline_key(key)
                && !matches!(
                    app.interaction_state.input_mode(),
                    InputMode::CreatingProfile
                ) =>
            {
                app.input.insert_newline();
                app.interaction_state.set_input_mode(InputMode::Editing);
                app.suggestion_index = 0;
            }
            KeyCode::Enter => handle_enter(app, request_tx)?,
            KeyCode::Char(character) => {
                if matches!(
                    app.interaction_state.input_mode(),
                    InputMode::CreatingProfile
                ) && !is_profile_name_character(character)
                {
                    return Ok(());
                }
                if !matches!(
                    app.interaction_state.input_mode(),
                    InputMode::CreatingProfile
                ) {
                    app.interaction_state.set_input_mode(InputMode::Editing);
                }
                app.input.insert_char(character);
                app.suggestion_index = 0;
            }
            KeyCode::Backspace => {
                if !matches!(
                    app.interaction_state.input_mode(),
                    InputMode::CreatingProfile
                ) {
                    app.interaction_state.set_input_mode(InputMode::Editing);
                }
                app.input.backspace();
                app.suggestion_index = 0;
            }
            KeyCode::Up => navigate_vertical(app, true)?,
            KeyCode::Down => navigate_vertical(app, false)?,
            KeyCode::Left | KeyCode::Right => navigate_horizontal(app, key.code),
            KeyCode::BackTab => toggle_plan(app, request_tx)?,
            KeyCode::Tab => complete_input(app),
            _ => {}
        },
        Event::Resize(_, _) => {}
        _ => {}
    }
    Ok(())
}

fn handle_escape(app: &mut SubBakeTui) -> io::Result<()> {
    if app.interaction_state.is_processing() {
        if app.interaction_state.request_cancellation() {
            if let Ok(mut progress) = app.progress.lock()
                && let Some((event, _)) = progress.as_mut()
            {
                event.state = TaskState::Cancelling;
                event.stage = "CANCELLING".to_owned();
            }
            if let Some(token) = &app.cancellation {
                token.cancel();
            }
            if let Ok(mut view) = app.msg_view.lock() {
                view.push(MsgStyle::System, "Cancellation requested…".to_owned());
            }
        }
        return Ok(());
    }
    let cancel_exits = matches!(
        app.interaction_state.input_mode(),
        InputMode::ChoosingSession(SessionPicker {
            cancel_exits: true,
            ..
        })
    );
    let closes_overlay = matches!(
        app.interaction_state.input_mode(),
        InputMode::ChoosingSession(_) | InputMode::ChoosingProfile(_) | InputMode::CreatingProfile
    );
    app.input.clear();
    app.interaction_state.set_input_mode(InputMode::Editing);
    app.suggestion_index = 0;
    if closes_overlay {
        app.close_fullscreen_overlay()?;
    }
    if cancel_exits {
        app.running = false;
    }
    Ok(())
}

fn handle_enter(app: &mut SubBakeTui, request_tx: &mpsc::Sender<WorkerRequest>) -> io::Result<()> {
    if app.interaction_state.is_processing() {
        return Ok(());
    }
    let suggestions = app.suggestions();
    let selected_action = if app.input.is_empty() {
        match empty_mode_choice(app.interaction_state.input_mode(), app.suggestion_index) {
            Some(EmptyModeChoice::Submit(action)) => Some(action),
            Some(EmptyModeChoice::RevisePlan) => {
                app.interaction_state.set_input_mode(InputMode::Editing);
                app.suggestion_index = 0;
                return Ok(());
            }
            Some(EmptyModeChoice::CreateProfile) => {
                app.interaction_state
                    .set_input_mode(InputMode::CreatingProfile);
                app.suggestion_index = 0;
                if let Ok(mut view) = app.msg_view.lock() {
                    view.push(
                        MsgStyle::System,
                        "Enter a new profile name (letters, numbers, - and _).".to_owned(),
                    );
                }
                return Ok(());
            }
            None if !suggestions.is_empty() => {
                let index = app.suggestion_index.min(suggestions.len() - 1);
                app.input.set_text(suggestions[index].0.clone());
                app.suggestion_index = 0;
                return Ok(());
            }
            None => None,
        }
    } else if !suggestions.is_empty() && !suggestions.iter().any(|item| item.0 == app.input.text())
    {
        let index = app.suggestion_index.min(suggestions.len() - 1);
        app.input.set_text(suggestions[index].0.clone());
        app.suggestion_index = 0;
        return Ok(());
    } else {
        None
    };

    let action = selected_action.unwrap_or_else(|| take_input_action(app));
    if matches!(&action, TuiAction::SubmitText(text) if text.is_empty()) {
        return Ok(());
    }
    if matches!(&action, TuiAction::SubmitText(text) if text == "/exit" || text == "/quit") {
        app.running = false;
        return Ok(());
    }
    if let TuiAction::SubmitText(text) = &action
        && matches!(text.as_str(), "/help" | "/h")
    {
        let response = app.handle_slash(text);
        if let Ok(mut view) = app.msg_view.lock() {
            view.push(MsgStyle::Response, response);
        }
        return Ok(());
    }
    submit_action(app, request_tx, action)
}

fn take_input_action(app: &mut SubBakeTui) -> TuiAction {
    let trimmed = app.input.take().trim().to_owned();
    let creating_profile = matches!(
        app.interaction_state.input_mode(),
        InputMode::CreatingProfile
    );
    app.interaction_state.set_input_mode(InputMode::Editing);
    if !trimmed.is_empty()
        && app
            .input_history
            .last()
            .is_none_or(|previous| previous != &trimmed)
    {
        app.input_history.push(trimmed.clone());
    }
    if creating_profile {
        TuiAction::CreateProfile(trimmed)
    } else {
        TuiAction::SubmitText(trimmed)
    }
}

fn submit_action(
    app: &mut SubBakeTui,
    request_tx: &mpsc::Sender<WorkerRequest>,
    action: TuiAction,
) -> io::Result<()> {
    let opens_session_picker =
        matches!(&action, TuiAction::SubmitText(input) if input == "/sessions");
    let changes_plan_mode = matches!(&action, TuiAction::TogglePlan)
        || matches!(
            &action,
            TuiAction::SubmitText(input)
                if matches!(input.trim(), "/plan" | "/plan on" | "/plan off")
        );
    if !opens_session_picker
        && !changes_plan_mode
        && let TuiAction::SubmitText(text) = &action
        && let Ok(mut view) = app.msg_view.lock()
    {
        view.push(MsgStyle::User, format!("[{}] {text}", iso_now()));
    }
    if matches!(
        &action,
        TuiAction::SelectProfile(_) | TuiAction::CreateProfile(_) | TuiAction::SelectSession(_)
    ) {
        app.close_fullscreen_overlay()?;
    }
    if let Ok(mut progress) = app.progress.lock() {
        *progress = None;
    }
    app.interaction_state.begin_processing(None);
    send(app, request_tx, action)
}

fn navigate_vertical(app: &mut SubBakeTui, up: bool) -> io::Result<()> {
    match vertical_navigation(app.interaction_state.input_mode(), app.suggestions().len()) {
        VerticalNavigation::Selection(count) => {
            app.suggestion_index = if up {
                previous_suggestion(app.suggestion_index, count)
            } else {
                (app.suggestion_index + 1) % count
            };
        }
        VerticalNavigation::History => {
            let width = app.terminal.size()?.width.saturating_sub(4).max(1);
            let moved = if up {
                app.input.move_up(width)
            } else {
                app.input.move_down(width)
            };
            if !moved {
                if up {
                    app.navigate_history_up();
                } else {
                    app.navigate_history_down();
                }
            }
        }
        VerticalNavigation::Disabled => {}
    }
    Ok(())
}

fn navigate_horizontal(app: &mut SubBakeTui, code: KeyCode) {
    let option_count = match app.interaction_state.input_mode() {
        InputMode::ChoosingSession(picker) => picker.options.len(),
        InputMode::ChoosingProfile(picker) => picker.options.len(),
        _ => 0,
    };
    if option_count > 0 {
        app.suggestion_index = if code == KeyCode::Left {
            previous_suggestion(app.suggestion_index, option_count)
        } else {
            (app.suggestion_index + 1) % option_count
        };
    } else if !app.interaction_state.is_processing() {
        if code == KeyCode::Left {
            app.input.move_left();
        } else {
            app.input.move_right();
        }
    }
}

fn toggle_plan(app: &mut SubBakeTui, request_tx: &mpsc::Sender<WorkerRequest>) -> io::Result<()> {
    if app.interaction_state.is_processing() {
        return Ok(());
    }
    let previous = app.plan_mode;
    app.plan_mode = !previous;
    app.interaction_state.begin_processing(Some(previous));
    send(app, request_tx, TuiAction::TogglePlan)
}

fn complete_input(app: &mut SubBakeTui) {
    if app.input.text().starts_with('/') {
        let matches = slash_suggestions(app.input.text());
        if !matches.is_empty() {
            let index = app.suggestion_index.min(matches.len() - 1);
            app.input.set_text(matches[index].0.to_owned());
            app.suggestion_index = (index + 1) % matches.len();
        }
        return;
    }
    let completions = [
        "translate ",
        "transcribe ",
        "list files",
        "read file",
        "search files",
        "whisper ",
    ];
    if !app.input.is_empty()
        && let Some(completion) = completions
            .iter()
            .find(|completion| completion.starts_with(app.input.text()))
    {
        app.input.set_text(completion.to_string());
    }
}

fn send(
    app: &SubBakeTui,
    request_tx: &mpsc::Sender<WorkerRequest>,
    action: TuiAction,
) -> io::Result<()> {
    let guard = app
        .cancellation
        .as_ref()
        .map(CancellationToken::guard)
        .unwrap_or_default();
    request_tx
        .send((action, guard))
        .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "agent worker stopped"))
}
