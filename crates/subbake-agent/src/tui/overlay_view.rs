use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Shadow};

use super::layout::OverlayLayout;
use super::render::{OverlaySnapshot, ViewSnapshot};
use super::text::{display_width, pad_to_width, tail_by_width, truncate_with_ellipsis};
use super::{ConfigEditorState, picker_viewport};
use crate::ConfigFieldKind;
use crate::tui_state::ConfigFocus;

pub(super) fn render(frame: &mut ratatui::Frame<'_>, layout: OverlayLayout, view: &ViewSnapshot) {
    match &view.overlay {
        Some(OverlaySnapshot::Sessions(options)) => {
            render_sessions(frame, layout, options, view.selected_suggestion);
        }
        Some(OverlaySnapshot::Profiles(options)) => {
            render_profiles(frame, layout, options, view.selected_suggestion);
        }
        Some(OverlaySnapshot::ProfileCreation) => {
            render_profile_creation(frame, layout, view);
        }
        Some(OverlaySnapshot::Config(editor)) => {
            render_config_editor(frame, layout, editor, view);
        }
        None => {}
    }
}

fn render_sessions(
    frame: &mut ratatui::Frame<'_>,
    layout: OverlayLayout,
    options: &[crate::engine::SessionChoice],
    selected: usize,
) {
    render_outer(frame, layout.outer, None);
    render_header(
        frame,
        layout.header,
        "Resume a previous session",
        "Sessions for this project · newest activity first",
    );
    let (start, end) = picker_viewport(selected, options.len(), layout.body.height);
    let width = usize::from(layout.body.width);
    let mut lines = Vec::new();
    for (index, session) in options.iter().enumerate().take(end).skip(start) {
        let style = selection_style(index == selected);
        lines.push(Line::from(Span::styled(
            truncate_with_ellipsis(
                &format!(
                    "{}  {}  ·  {}  ·  {} events",
                    if session.active { "●" } else { " " },
                    session.updated_at,
                    session.cwd,
                    session.event_count,
                ),
                width,
            ),
            style,
        )));
        lines.push(Line::from(Span::styled(
            truncate_with_ellipsis(&format!("   {}", session.title), width),
            style.add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
    }
    frame.render_widget(Paragraph::new(lines), layout.body);
    render_picker_footer(frame, layout.footer, selected, options.len(), "resume");
}

fn render_profiles(
    frame: &mut ratatui::Frame<'_>,
    layout: OverlayLayout,
    options: &[crate::engine::ProfileChoice],
    selected: usize,
) {
    render_outer(frame, layout.outer, None);
    render_header(
        frame,
        layout.header,
        "Choose a model profile",
        "Profiles from the active SubBake configuration",
    );
    let (start, end) = picker_viewport(selected, options.len(), layout.body.height);
    let width = usize::from(layout.body.width);
    let mut lines = Vec::new();
    for (index, profile) in options.iter().enumerate().take(end).skip(start) {
        let style = selection_style(index == selected);
        lines.push(Line::from(Span::styled(
            truncate_with_ellipsis(
                &format!(
                    "{}  {}",
                    if profile.active { "●" } else { " " },
                    profile.name
                ),
                width,
            ),
            style.add_modifier(Modifier::BOLD),
        )));
        let detail = if profile.create {
            profile.model.clone()
        } else {
            format!("   {} / {}", profile.provider, profile.model)
        };
        lines.push(Line::from(Span::styled(
            truncate_with_ellipsis(&detail, width),
            style,
        )));
        lines.push(Line::from(""));
    }
    frame.render_widget(Paragraph::new(lines), layout.body);
    render_picker_footer(frame, layout.footer, selected, options.len(), "select");
}

fn render_profile_creation(
    frame: &mut ratatui::Frame<'_>,
    layout: OverlayLayout,
    view: &ViewSnapshot,
) {
    render_outer(frame, layout.outer, None);
    let width = usize::from(layout.inner.width);
    let empty = view.input.is_empty();
    let available = width.saturating_sub(2);
    let visible_name = if empty {
        truncate_with_ellipsis("profile name…", available)
    } else {
        tail_by_width(view.input.text(), available)
    };
    let mut lines = [
        ("Create a model profile".to_owned(), Color::Cyan, true),
        (
            "Copy the active settings into a new profile".to_owned(),
            Color::DarkGray,
            false,
        ),
        (String::new(), Color::Reset, false),
        ("Profile name".to_owned(), Color::Cyan, false),
        (
            format!("> {visible_name}"),
            if empty { Color::DarkGray } else { Color::White },
            !empty,
        ),
        (String::new(), Color::Reset, false),
        (
            "Allowed: letters, numbers, - and _".to_owned(),
            Color::DarkGray,
            false,
        ),
        (
            "Inline API keys and auth headers will not be copied.".to_owned(),
            Color::Yellow,
            false,
        ),
        (String::new(), Color::Reset, false),
        (
            "Enter create · Esc cancel".to_owned(),
            Color::DarkGray,
            false,
        ),
    ]
    .into_iter()
    .map(|(text, color, bold)| {
        let mut style = Style::default().fg(color);
        if bold {
            style = style.add_modifier(Modifier::BOLD);
        }
        Line::from(Span::styled(truncate_with_ellipsis(&text, width), style))
    })
    .collect::<Vec<_>>();
    lines.truncate(usize::from(layout.inner.height));
    frame.render_widget(Paragraph::new(lines), layout.inner);

    if !layout.inner.is_empty() && layout.inner.height > 4 {
        let x = layout
            .inner
            .x
            .saturating_add(2)
            .saturating_add(u16::try_from(display_width(&visible_name)).unwrap_or(u16::MAX))
            .min(layout.inner.right().saturating_sub(1));
        frame.set_cursor_position((x, layout.inner.y.saturating_add(4)));
    }
}

fn render_config_editor(
    frame: &mut ratatui::Frame<'_>,
    layout: OverlayLayout,
    editor: &ConfigEditorState,
    view: &ViewSnapshot,
) {
    render_outer(frame, layout.outer, Some(" SubBake configuration "));
    let target = match &editor.snapshot.target {
        subbake_adapters::ConfigEditTarget::Defaults => "defaults".to_owned(),
        subbake_adapters::ConfigEditTarget::Profile(name) => format!("profile: {name}"),
    };
    let heading = format!("Editing {target}  ·  {}", editor.snapshot.path.display());
    frame.render_widget(
        Paragraph::new(truncate_with_ellipsis(
            &heading,
            usize::from(layout.header.width),
        )),
        layout.header,
    );

    if let Some(sidebar) = layout.sidebar {
        let lines = crate::ConfigSection::ALL
            .iter()
            .enumerate()
            .take(usize::from(sidebar.height))
            .map(|(index, section)| {
                let selected = index == editor.section_index;
                Line::from(Span::styled(
                    truncate_with_ellipsis(
                        &format!("  {}", section.label()),
                        usize::from(sidebar.width.saturating_sub(1)),
                    ),
                    if selected {
                        selection_style(editor.focus == ConfigFocus::Sections)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    },
                ))
            })
            .collect::<Vec<_>>();
        frame.render_widget(
            Paragraph::new(lines).block(
                Block::default()
                    .title(truncate_with_ellipsis(
                        " Categories ",
                        usize::from(sidebar.width),
                    ))
                    .borders(Borders::RIGHT)
                    .border_style(Style::default().fg(Color::DarkGray)),
            ),
            sidebar,
        );
    }

    render_config_fields(frame, layout.content, editor, layout.sidebar.is_none());
    render_config_footer(frame, layout.footer, editor, view);

    if editor.confirm.is_some() {
        frame.render_widget(Clear, layout.popup);
        let options = ["Save changes", "Discard changes", "Cancel"];
        let width = usize::from(layout.popup.width.saturating_sub(2));
        let lines = options
            .iter()
            .enumerate()
            .map(|(index, option)| {
                Line::from(Span::styled(
                    truncate_with_ellipsis(&format!("  {option}"), width),
                    selection_style(index == editor.field_index.min(2)),
                ))
            })
            .collect::<Vec<_>>();
        frame.render_widget(
            Paragraph::new(lines).block(
                Block::default()
                    .title(truncate_with_ellipsis(
                        " Unsaved configuration ",
                        usize::from(layout.popup.width),
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Yellow))
                    .shadow(Shadow::dark_shade().style(Color::DarkGray)),
            ),
            layout.popup,
        );
    }
}

fn render_config_fields(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    editor: &ConfigEditorState,
    narrow: bool,
) {
    let ids = editor.field_ids();
    let visible_height = usize::from(area.height).max(1);
    let selected = editor.field_index.min(ids.len().saturating_sub(1));
    let start = selected
        .saturating_add(1)
        .saturating_sub(visible_height)
        .min(ids.len().saturating_sub(visible_height));
    let width = usize::from(area.width);
    let status_width = if width >= 32 { (width / 4).min(12) } else { 0 };
    let label_width = if width >= 16 {
        (width.saturating_sub(status_width) * 2 / 5).min(
            ids.iter()
                .map(|id| display_width(id.label()).saturating_add(2))
                .max()
                .unwrap_or(0),
        )
    } else {
        width.min(8)
    };
    let value_width = width
        .saturating_sub(label_width)
        .saturating_sub(status_width);
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
                ConfigFieldKind::Boolean if raw_value == "true" => "● enabled".to_owned(),
                ConfigFieldKind::Boolean => "○ disabled".to_owned(),
                _ if raw_value.is_empty() => "—".to_owned(),
                _ => raw_value,
            };
            let status = if inherited {
                "inherited"
            } else if changed.is_some() {
                "modified"
            } else {
                "override"
            };
            let row_style = if index == selected && editor.focus == ConfigFocus::Fields {
                selection_style(true)
            } else {
                Style::default().fg(Color::White)
            };
            Line::from(vec![
                Span::styled(
                    pad_to_width(&format!("  {}", id.label()), label_width),
                    row_style,
                ),
                Span::styled(pad_to_width(&value, value_width), row_style),
                Span::styled(
                    pad_to_width(status, status_width),
                    if changed.is_some() {
                        Style::default().fg(Color::Yellow)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    },
                ),
            ])
        })
        .collect::<Vec<_>>();
    let title = if narrow {
        truncate_with_ellipsis(
            &format!(" {} ", editor.section().label()),
            usize::from(area.width),
        )
    } else {
        String::new()
    };
    frame.render_widget(
        Paragraph::new(lines).block(Block::default().title(title)),
        area,
    );
}

fn render_config_footer(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    editor: &ConfigEditorState,
    view: &ViewSnapshot,
) {
    let width = usize::from(area.width);
    let (lines, cursor_prefix, cursor_value) = if let Some(id) = editor.editing_field {
        let shown = if matches!(id.kind(), ConfigFieldKind::Secret) {
            "•".repeat(view.input.text().chars().count())
        } else {
            view.input.text().to_owned()
        };
        let prefix = format!("{}: ", id.label());
        let shown = tail_by_width(&shown, width.saturating_sub(display_width(&prefix)));
        (
            vec![
                truncate_with_ellipsis(&format!("{prefix}{shown}"), width),
                truncate_with_ellipsis("Enter accept  ·  Esc cancel", width),
            ],
            Some(prefix),
            Some(shown),
        )
    } else {
        let help = editor.selected_field().map_or_else(
            || "Select a configuration field".to_owned(),
            |id| {
                id.hint().map_or_else(
                    || id.toml_key(),
                    |hint| format!("{}  ·  {hint}", id.toml_key()),
                )
            },
        );
        (
            vec![
                truncate_with_ellipsis(&help, width),
                truncate_with_ellipsis(
                    "Tab focus  ↑↓ navigate  Enter edit  Space/←→ toggle  Del inherit  Ctrl+S save  Esc/q close",
                    width,
                ),
            ],
            None,
            None,
        )
    };
    frame.render_widget(
        Paragraph::new(lines.into_iter().map(Line::from).collect::<Vec<_>>())
            .style(Style::default().fg(Color::DarkGray)),
        area,
    );
    if let (Some(prefix), Some(value)) = (cursor_prefix, cursor_value)
        && !area.is_empty()
    {
        let x = area
            .x
            .saturating_add(
                u16::try_from(display_width(&prefix) + display_width(&value)).unwrap_or(u16::MAX),
            )
            .min(area.right().saturating_sub(1));
        frame.set_cursor_position((x, area.y));
    }
}

fn render_outer(frame: &mut ratatui::Frame<'_>, area: Rect, title: Option<&str>) {
    frame.render_widget(Clear, area);
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    if let Some(title) = title {
        block = block.title(truncate_with_ellipsis(title, usize::from(area.width)));
    }
    frame.render_widget(block, area);
}

fn render_header(frame: &mut ratatui::Frame<'_>, area: Rect, title: &str, subtitle: &str) {
    let width = usize::from(area.width);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                truncate_with_ellipsis(title, width),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                truncate_with_ellipsis(subtitle, width),
                Style::default().fg(Color::DarkGray),
            )),
        ]),
        area,
    );
}

fn render_picker_footer(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    selected: usize,
    count: usize,
    action: &str,
) {
    let footer = format!(
        "↑↓←→ navigate · Enter {action} · Esc cancel  {}/{}",
        selected.saturating_add(1).min(count),
        count
    );
    frame.render_widget(
        Paragraph::new(Span::styled(
            truncate_with_ellipsis(&footer, usize::from(area.width)),
            Style::default().fg(Color::DarkGray),
        )),
        area,
    );
}

fn selection_style(selected: bool) -> Style {
    if selected {
        Style::default().fg(Color::Black).bg(Color::Cyan)
    } else {
        Style::default().fg(Color::White)
    }
}
