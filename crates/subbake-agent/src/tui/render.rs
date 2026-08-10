use std::io;

use ratatui::Terminal;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use super::progress::{format_progress, spinner_frame};
use super::{
    InputMode, InteractionState, SubBakeTui, picker_viewport, terminal_width,
    truncate_with_ellipsis,
};
use crate::ConfigFieldKind;
use crate::tui_state::{ConfigEditorState, ConfigFocus};

pub(super) fn draw(app: &mut SubBakeTui) -> io::Result<()> {
    let terminal_area = app
        .overlay_terminal
        .as_ref()
        .map_or_else(|| app.terminal.size(), Terminal::size)?;
    let suggestions = app.suggestions();
    let selected_suggestion = match app.interaction_state.input_mode() {
        InputMode::ChoosingSession(picker) => app
            .suggestion_index
            .min(picker.options.len().saturating_sub(1)),
        InputMode::ChoosingProfile(picker) => app
            .suggestion_index
            .min(picker.options.len().saturating_sub(1)),
        InputMode::ChoosingConfigProfile(picker) => app
            .suggestion_index
            .min(picker.options.len().saturating_sub(1)),
        _ => app
            .suggestion_index
            .min(suggestions.len().saturating_sub(1)),
    };
    let processing = app.interaction_state.is_processing();
    let progress = app.progress.lock().ok().and_then(|value| value.clone());
    let ambient_spinner = spinner_frame(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default(),
    );
    let session_picker = match app.interaction_state.input_mode() {
        InputMode::ChoosingSession(picker) => Some(picker.options.clone()),
        _ => None,
    };
    let profile_picker = match app.interaction_state.input_mode() {
        InputMode::ChoosingProfile(picker) | InputMode::ChoosingConfigProfile(picker) => {
            Some(picker.options.clone())
        }
        _ => None,
    };
    let creating_profile = matches!(
        app.interaction_state.input_mode(),
        InputMode::CreatingProfile | InputMode::CreatingConfigProfile
    );
    let config_editor = app.config_editor.clone();
    let profile_name_input = app.input.text().to_owned();
    let startup_info = app.startup_info.clone();
    let plan_mode = app.plan_mode;

    let input_width = terminal_area.width.saturating_sub(4).max(1);
    let max_input_lines = (terminal_area.height.saturating_mul(40) / 100).max(1);
    let input_line_count = app.input.desired_height(input_width).min(max_input_lines);
    let input_total_height = input_line_count.saturating_add(3);
    let toggling_plan_mode = matches!(
        app.interaction_state,
        InteractionState::Processing {
            plan_mode_rollback: Some(_),
            ..
        }
    );
    let progress_height = u16::from(processing && !toggling_plan_mode) * 2;
    let suggestion_height = (suggestions.len() as u16).min(
        terminal_area
            .height
            .saturating_sub(input_total_height)
            .saturating_sub(progress_height)
            .saturating_sub(1),
    );
    let visible_input_lines = app.input.visible_lines(input_width, input_line_count);
    let (input_cursor_x, input_cursor_y) = app.input.cursor_position(input_width);

    let draw_ui = |frame: &mut ratatui::Frame<'_>| {
        let area = frame.area();
        if let Some(editor) = &config_editor
            && !matches!(
                app.interaction_state.input_mode(),
                InputMode::ChoosingConfigProfile(_) | InputMode::CreatingConfigProfile
            )
        {
            render_config_editor(frame, area, editor, &profile_name_input);
            return;
        }
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
            frame.render_widget(
                Paragraph::new(vec![
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
                ]),
                chunks[0],
            );
            let (start, end) =
                picker_viewport(selected_suggestion, options.len(), chunks[1].height);
            let mut lines = Vec::new();
            for (index, session) in options.iter().enumerate().take(end).skip(start) {
                let selected = index == selected_suggestion;
                let style = selection_style(selected);
                lines.push(Line::from(Span::styled(
                    format!(
                        "{}  {}  ·  {}  ·  {} events",
                        if session.active { "●" } else { " " },
                        session.updated_at,
                        session.cwd,
                        session.event_count,
                    ),
                    style,
                )));
                lines.push(Line::from(Span::styled(
                    format!("   {}", session.title),
                    style.add_modifier(Modifier::BOLD),
                )));
                lines.push(Line::from(""));
            }
            frame.render_widget(Paragraph::new(lines), chunks[1]);
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    format!(
                        "↑↓←→ navigate · Enter resume · Esc cancel  {}/{}",
                        selected_suggestion.saturating_add(1),
                        options.len()
                    ),
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
            frame.render_widget(
                Paragraph::new(vec![
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
                ]),
                chunks[0],
            );
            let (start, end) =
                picker_viewport(selected_suggestion, options.len(), chunks[1].height);
            let mut lines = Vec::new();
            for (index, profile) in options.iter().enumerate().take(end).skip(start) {
                let style = selection_style(index == selected_suggestion);
                lines.push(Line::from(Span::styled(
                    format!(
                        "{}  {}",
                        if profile.active { "●" } else { " " },
                        profile.name
                    ),
                    style.add_modifier(Modifier::BOLD),
                )));
                lines.push(Line::from(Span::styled(
                    if profile.create {
                        profile.model.clone()
                    } else {
                        format!("   {} / {}", profile.provider, profile.model)
                    },
                    style,
                )));
                lines.push(Line::from(""));
            }
            frame.render_widget(Paragraph::new(lines), chunks[1]);
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    format!(
                        "↑↓←→ navigate · Enter select · Esc cancel  {}/{}",
                        selected_suggestion.saturating_add(1),
                        options.len()
                    ),
                    Style::default().fg(Color::DarkGray),
                ))),
                chunks[2],
            );
            return;
        }
        if creating_profile {
            frame.render_widget(Clear, area);
            let empty = profile_name_input.is_empty();
            let name = if empty {
                "profile name…".to_owned()
            } else {
                profile_name_input.clone()
            };
            frame.render_widget(
                Paragraph::new(vec![
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
                    Line::from(Span::styled(
                        format!("> {name}"),
                        if empty {
                            Style::default().fg(Color::DarkGray)
                        } else {
                            Style::default()
                                .fg(Color::White)
                                .add_modifier(Modifier::BOLD)
                        },
                    )),
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
                ])
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
                Constraint::Length(progress_height),
                Constraint::Length(input_total_height),
            ])
            .split(area);
        let suggestion_lines = suggestions
            .iter()
            .enumerate()
            .map(|(index, (command, description))| {
                let selected = index == selected_suggestion;
                Line::from(vec![
                    Span::styled(
                        format!("› {command:<10}"),
                        if selected {
                            Style::default()
                                .fg(Color::Black)
                                .bg(Color::Cyan)
                                .add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Color::Cyan)
                        },
                    ),
                    Span::styled(
                        description.clone(),
                        if selected {
                            Style::default().fg(Color::White).bg(Color::DarkGray)
                        } else {
                            Style::default().fg(Color::DarkGray)
                        },
                    ),
                ])
            })
            .collect::<Vec<_>>();
        frame.render_widget(Paragraph::new(suggestion_lines), chunks[0]);

        if progress_height > 0 {
            let progress_text = progress
                .as_ref()
                .map(|(event, started)| format_progress(event, started.elapsed()))
                .unwrap_or_else(|| format!("{ambient_spinner} Working"));
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(Span::styled(
                        format!("⚡ {progress_text}"),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    )),
                    Line::from(Span::styled(
                        "  Esc cancel",
                        Style::default().fg(Color::DarkGray),
                    )),
                ]),
                chunks[1],
            );
        }

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
        let input_lines = if app.input.is_empty()
            && matches!(app.interaction_state.input_mode(), InputMode::Editing)
            && !processing
        {
            vec![Line::from(vec![
                Span::styled("> ", Style::default().fg(Color::Cyan)),
                Span::styled(
                    truncate_with_ellipsis(app.input_hint, usize::from(input_width)),
                    Style::default().fg(Color::DarkGray),
                ),
            ])]
        } else {
            visible_input_lines
                .iter()
                .enumerate()
                .map(|(index, line)| {
                    Line::from(Span::styled(
                        format!("{}{line}", if index == 0 { "> " } else { "  " }),
                        Style::default().fg(Color::Cyan),
                    ))
                })
                .collect()
        };
        frame.render_widget(
            Paragraph::new(input_lines).block(
                Block::default()
                    .borders(Borders::TOP | Borders::BOTTOM)
                    .border_style(Style::default().fg(Color::DarkGray)),
            ),
            input_entry_area,
        );
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    startup_info.model.clone(),
                    Style::default().fg(Color::Yellow),
                ),
                Span::styled("  ", Style::default()),
                Span::styled(startup_info.cwd.clone(), Style::default().fg(Color::Green)),
            ]))
            .wrap(Wrap { trim: false }),
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
        frame.set_cursor_position((
            input_entry_area.x + 2 + input_cursor_x.min(input_width),
            input_entry_area.y + 1 + input_cursor_y.min(input_line_count.saturating_sub(1)),
        ));
    };
    if let Some(terminal) = app.overlay_terminal.as_mut() {
        terminal.draw(draw_ui)?;
    } else {
        app.terminal.draw(draw_ui)?;
    }
    Ok(())
}

fn render_config_editor(
    frame: &mut ratatui::Frame<'_>,
    area: ratatui::layout::Rect,
    editor: &ConfigEditorState,
    input: &str,
) {
    frame.render_widget(Clear, area);
    let outer = Block::default()
        .title(" SubBake configuration ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = outer.inner(area);
    frame.render_widget(outer, area);
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(4),
            Constraint::Length(2),
        ])
        .split(inner);
    let target = match &editor.snapshot.target {
        subbake_adapters::ConfigEditTarget::Defaults => "defaults".to_owned(),
        subbake_adapters::ConfigEditTarget::Profile(name) => format!("profile: {name}"),
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("Editing ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                target,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  ·  {}", editor.snapshot.path.display()),
                Style::default().fg(Color::DarkGray),
            ),
        ])),
        vertical[0],
    );
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(20), Constraint::Min(20)])
        .split(vertical[1]);
    let section_lines = crate::ConfigSection::ALL
        .iter()
        .enumerate()
        .map(|(index, section)| {
            let selected = index == editor.section_index;
            Line::from(Span::styled(
                format!("  {:<16}", section.label()),
                if selected {
                    selection_style(editor.focus == ConfigFocus::Sections)
                } else {
                    Style::default().fg(Color::DarkGray)
                },
            ))
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(section_lines).block(
            Block::default()
                .title(" Categories ")
                .borders(Borders::RIGHT)
                .border_style(Style::default().fg(Color::DarkGray)),
        ),
        columns[0],
    );

    let ids = editor.field_ids();
    let visible_height = usize::from(columns[1].height.saturating_sub(2)).max(1);
    let selected = editor.field_index.min(ids.len().saturating_sub(1));
    let start = selected
        .saturating_add(1)
        .saturating_sub(visible_height)
        .min(ids.len().saturating_sub(visible_height));
    let lines = ids
        .iter()
        .enumerate()
        .skip(start)
        .take(visible_height)
        .map(|(index, id)| {
            let view = editor.snapshot.fields.iter().find(|field| field.id == *id);
            let changed = editor.changes.iter().rev().find(|change| change.id == *id);
            let inherited = changed
                .map(|change| change.value.is_none())
                .or_else(|| view.map(|field| field.inherited))
                .unwrap_or(false);
            let configured = view.is_some_and(|field| field.configured);
            let raw_value = editor.value(*id);
            let value = match id.kind() {
                ConfigFieldKind::Secret if changed.is_some_and(|change| change.value.is_some()) => {
                    "•••••• (pending)".to_owned()
                }
                ConfigFieldKind::Secret if configured => "•••••• (configured)".to_owned(),
                ConfigFieldKind::Secret => "not configured".to_owned(),
                ConfigFieldKind::Boolean => {
                    if raw_value == "true" {
                        "● enabled".to_owned()
                    } else {
                        "○ disabled".to_owned()
                    }
                }
                _ if raw_value.is_empty() => "—".to_owned(),
                _ => raw_value,
            };
            let row_selected = index == selected;
            let row_style = if row_selected && editor.focus == ConfigFocus::Fields {
                selection_style(true)
            } else {
                Style::default().fg(Color::White)
            };
            Line::from(vec![
                Span::styled(format!("  {:<25}", id.label()), row_style),
                Span::styled(truncate_with_ellipsis(&value, 32), row_style),
                Span::styled(
                    if inherited {
                        "  inherited"
                    } else if changed.is_some() {
                        "  modified"
                    } else {
                        "  override"
                    },
                    if changed.is_some() {
                        Style::default().fg(Color::Yellow)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    },
                ),
            ])
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::default().title(format!(" {} ", editor.section().label()))),
        columns[1],
    );

    let footer = if let Some(id) = editor.editing_field {
        let shown = if matches!(id.kind(), ConfigFieldKind::Secret) {
            "•".repeat(input.chars().count())
        } else {
            input.to_owned()
        };
        format!("{}: {}  ·  Enter accept  Esc cancel", id.label(), shown)
    } else {
        "Tab focus  ↑↓ navigate  Enter edit  Space/←→ toggle  Del inherit  Ctrl+S save  Esc/q close"
            .to_owned()
    };
    frame.render_widget(
        Paragraph::new(Span::styled(footer, Style::default().fg(Color::DarkGray))),
        vertical[2],
    );

    if editor.confirm.is_some() {
        let popup = centered_rect(58, 7, area);
        frame.render_widget(Clear, popup);
        let options = ["Save changes", "Discard changes", "Cancel"];
        let lines = options
            .iter()
            .enumerate()
            .map(|(index, option)| {
                Line::from(Span::styled(
                    format!("  {option}"),
                    selection_style(index == editor.field_index.min(2)),
                ))
            })
            .collect::<Vec<_>>();
        frame.render_widget(
            Paragraph::new(lines).block(
                Block::default()
                    .title(" Unsaved configuration ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Yellow)),
            ),
            popup,
        );
    }
}

fn centered_rect(width: u16, height: u16, area: ratatui::layout::Rect) -> ratatui::layout::Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    ratatui::layout::Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

fn selection_style(selected: bool) -> Style {
    if selected {
        Style::default().fg(Color::Black).bg(Color::Cyan)
    } else {
        Style::default().fg(Color::White)
    }
}
