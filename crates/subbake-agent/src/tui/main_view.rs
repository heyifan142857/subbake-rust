use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use super::layout::ComposerLayout;
use super::progress::format_progress;
use super::render::ViewSnapshot;
use super::text::{display_width, pad_to_width, truncate_with_ellipsis};

pub(super) fn render(frame: &mut ratatui::Frame<'_>, layout: ComposerLayout, view: &ViewSnapshot) {
    render_suggestions(frame, layout.suggestions, view);
    render_progress(frame, layout.progress, view);
    render_input(frame, layout, view);
}

fn render_suggestions(frame: &mut ratatui::Frame<'_>, area: Rect, view: &ViewSnapshot) {
    let width = usize::from(area.width);
    let lines = view
        .suggestions
        .iter()
        .take(usize::from(area.height))
        .enumerate()
        .map(|(index, (command, description))| {
            let selected = index == view.selected_suggestion;
            let command_width = view
                .suggestions
                .iter()
                .map(|(command, _)| display_width(command))
                .max()
                .unwrap_or(0)
                .saturating_add(3)
                .min(width);
            let command = pad_to_width(&format!("› {command}"), command_width);
            let description =
                truncate_with_ellipsis(description, width.saturating_sub(display_width(&command)));
            Line::from(vec![
                Span::styled(command, suggestion_command_style(selected)),
                Span::styled(description, suggestion_description_style(selected)),
            ])
        })
        .collect::<Vec<_>>();
    frame.render_widget(Paragraph::new(lines), area);
}

fn render_progress(frame: &mut ratatui::Frame<'_>, area: Rect, view: &ViewSnapshot) {
    if area.is_empty() {
        return;
    }
    let activity = view
        .active_tool
        .as_ref()
        .map(|active| crate::tool_presentation::running_activity(&active.name, &active.arguments));
    let key_hint = if view.input.is_empty() {
        "Enter queue · Esc cancel"
    } else {
        "Enter queue · Esc send now"
    };
    let detail = view
        .progress
        .as_ref()
        .map(|(event, started)| format_progress(event, started.elapsed()))
        .or_else(|| {
            activity
                .as_ref()
                .and_then(|activity| activity.detail.clone())
        })
        .map_or_else(
            || key_hint.to_owned(),
            |detail| format!("{detail} · {key_hint}"),
        );
    if let Some(group) = &view.active_tool_group {
        let mut lines = super::transcript::tool_group_lines(
            group,
            area.width,
            Some((view.spinner, detail.as_str())),
        );
        if !group
            .activities
            .iter()
            .any(|activity| activity.status == super::ToolActivityStatus::Running)
        {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {} ", view.spinner),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled("Working · ", Style::default().fg(Color::DarkGray)),
                Span::styled(key_hint, Style::default().fg(Color::DarkGray)),
            ]));
        }
        let visible_from = lines.len().saturating_sub(usize::from(area.height));
        frame.render_widget(Paragraph::new(lines.split_off(visible_from)), area);
        return;
    }

    let headline = activity.as_ref().map_or_else(
        || "Working".to_owned(),
        |activity| activity.headline.clone(),
    );
    let width = usize::from(area.width);
    let prefix = format!("  {} ", view.spinner);
    let lines = vec![
        Line::from(vec![
            Span::styled(
                truncate_with_ellipsis(&prefix, width),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                truncate_with_ellipsis(&headline, width.saturating_sub(display_width(&prefix))),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(Span::styled(
            truncate_with_ellipsis(&format!("    {detail}"), width),
            Style::default().fg(Color::DarkGray),
        )),
    ];
    frame.render_widget(Paragraph::new(lines), area);
}

fn render_input(frame: &mut ratatui::Frame<'_>, layout: ComposerLayout, view: &ViewSnapshot) {
    let mut input = view.input.clone();
    let input_width = layout.input_content_width;
    let input_lines = if view.input.is_empty() && view.editing {
        vec![Line::from(vec![
            Span::styled("> ", Style::default().fg(Color::Cyan)),
            Span::styled(
                truncate_with_ellipsis(
                    view.input_hint,
                    usize::from(layout.input_entry.width.saturating_sub(2)),
                ),
                Style::default().fg(Color::DarkGray),
            ),
        ])]
    } else if layout.input_line_count == 0 {
        Vec::new()
    } else {
        input
            .visible_lines(input_width, layout.input_line_count)
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
        layout.input_entry,
    );

    frame.render_widget(
        Paragraph::new(status_line(
            layout.status.width,
            view.plan_mode,
            &view.startup_info.model,
            &view.startup_info.cwd,
        )),
        layout.status,
    );

    if layout.input_line_count > 0 && layout.input_entry.width > 0 {
        let (cursor_x, cursor_y) = input.cursor_position(input_width);
        let right = layout.input_entry.right().saturating_sub(1);
        let bottom = layout.input_entry.bottom().saturating_sub(1);
        frame.set_cursor_position((
            layout
                .input_entry
                .x
                .saturating_add(2)
                .saturating_add(cursor_x)
                .min(right),
            layout
                .input_entry
                .y
                .saturating_add(1)
                .saturating_add(cursor_y)
                .min(bottom),
        ));
    }
}

fn status_line(width: u16, plan_mode: bool, model: &str, cwd: &str) -> Line<'static> {
    let mut remaining = usize::from(width);
    let mut spans = Vec::new();
    if plan_mode && remaining > 0 {
        let plan = truncate_with_ellipsis("Plan", remaining);
        remaining = remaining.saturating_sub(display_width(&plan));
        spans.push(Span::styled(
            plan,
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ));
    }
    append_status_part(&mut spans, &mut remaining, model, Color::Yellow);
    append_status_part(&mut spans, &mut remaining, cwd, Color::Green);
    Line::from(spans)
}

fn append_status_part(
    spans: &mut Vec<Span<'static>>,
    remaining: &mut usize,
    value: &str,
    color: Color,
) {
    if value.is_empty() || *remaining == 0 {
        return;
    }
    let separator = if spans.is_empty() { "" } else { "  " };
    if display_width(separator) >= *remaining {
        return;
    }
    spans.push(Span::raw(separator.to_owned()));
    *remaining = remaining.saturating_sub(display_width(separator));
    let value = truncate_with_ellipsis(value, *remaining);
    *remaining = remaining.saturating_sub(display_width(&value));
    spans.push(Span::styled(value, Style::default().fg(color)));
}

fn suggestion_command_style(selected: bool) -> Style {
    if selected {
        Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Cyan)
    }
}

fn suggestion_description_style(selected: bool) -> Style {
    if selected {
        Style::default().fg(Color::White).bg(Color::DarkGray)
    } else {
        Style::default().fg(Color::DarkGray)
    }
}

#[cfg(test)]
mod tests {
    use super::status_line;

    #[test]
    fn narrow_status_prioritizes_plan_then_model_then_cwd() {
        assert_eq!(status_line(4, true, "model", "/cwd").to_string(), "Plan");
        assert_eq!(
            status_line(10, true, "model", "/cwd").to_string(),
            "Plan  mod…"
        );
        assert_eq!(
            status_line(12, false, "model", "/cwd").to_string(),
            "model  /cwd"
        );
    }
}
