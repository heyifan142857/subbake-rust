use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::history::{Msg, MsgStyle, ToolActivityStatus, ToolGroup, TranscriptItem};
use super::text::{display_width, wrap_text};

pub(super) fn transcript_item_lines(
    item: &TranscriptItem,
    width: u16,
    running: Option<(char, &str)>,
) -> Vec<Line<'static>> {
    match item {
        TranscriptItem::Message(message) => message_lines(message, width),
        TranscriptItem::ToolGroup(group) => tool_group_lines(group, width, running),
    }
}

pub(super) fn message_lines(message: &Msg, width: u16) -> Vec<Line<'static>> {
    let (prefix, prefix_style, text_style, leading_blank) = match message.style {
        MsgStyle::User => (
            "› ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
            true,
        ),
        MsgStyle::Response => (
            "• ",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
            Style::default().fg(Color::White),
            true,
        ),
        MsgStyle::Commentary => (
            "· ",
            Style::default().fg(Color::DarkGray),
            Style::default().fg(Color::Gray),
            false,
        ),
        MsgStyle::Error => (
            "× ",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            Style::default().fg(Color::Red),
            true,
        ),
        MsgStyle::Observation => (
            "  ",
            Style::default(),
            Style::default().fg(Color::DarkGray),
            false,
        ),
        MsgStyle::System => (
            "  ",
            Style::default(),
            Style::default().fg(Color::Blue).add_modifier(Modifier::DIM),
            false,
        ),
    };
    let content_width = usize::from(width)
        .saturating_sub(display_width(prefix))
        .max(1);
    let wrapped = wrap_text(&message.text, content_width);
    let mut lines = Vec::with_capacity(wrapped.len() + usize::from(leading_blank));
    if leading_blank {
        lines.push(Line::default());
    }
    for (index, row) in wrapped.into_iter().enumerate() {
        lines.push(Line::from(vec![
            Span::styled(
                if index == 0 { prefix } else { "  " }.to_owned(),
                prefix_style,
            ),
            Span::styled(row, text_style),
        ]));
    }
    lines
}

pub(super) fn tool_group_lines(
    group: &ToolGroup,
    width: u16,
    running: Option<(char, &str)>,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let count = group.activities.len();
    for (index, activity) in group.activities.iter().enumerate() {
        let last = index + 1 == count;
        let connector = match (index, last) {
            (0, false) => "  ┌─ ",
            (_, false) => "  ├─ ",
            (_, true) => "  └─ ",
        };
        let continuation = if last { "       " } else { "  │    " };
        let (marker, marker_color) = match activity.status {
            ToolActivityStatus::Running => (
                running.map_or_else(|| "◐".to_owned(), |(spinner, _)| spinner.to_string()),
                Color::Cyan,
            ),
            ToolActivityStatus::Completed => ("✓".to_owned(), Color::Green),
            ToolActivityStatus::Failed => ("×".to_owned(), Color::Red),
            ToolActivityStatus::Cancelled => ("■".to_owned(), Color::Blue),
        };
        let headline_width = usize::from(width)
            .saturating_sub(display_width(connector) + display_width(&marker) + 1)
            .max(1);
        for (row_index, row) in wrap_text(&activity.headline, headline_width)
            .into_iter()
            .enumerate()
        {
            if row_index == 0 {
                lines.push(Line::from(vec![
                    Span::styled(connector.to_owned(), Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{marker} "),
                        Style::default()
                            .fg(marker_color)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        row,
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]));
            } else {
                lines.push(Line::from(vec![
                    Span::styled(
                        continuation.to_owned(),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(
                        row,
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]));
            }
        }
        let detail = if activity.status == ToolActivityStatus::Running {
            running
                .map(|(_, detail)| detail)
                .or(activity.detail.as_deref())
        } else {
            activity.detail.as_deref()
        };
        if let Some(detail) = detail {
            let detail_width = usize::from(width)
                .saturating_sub(display_width(continuation))
                .max(1);
            lines.extend(wrap_text(detail, detail_width).into_iter().map(|row| {
                Line::from(vec![
                    Span::styled(
                        continuation.to_owned(),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(row, Style::default().fg(Color::DarkGray)),
                ])
            }));
        }
    }
    lines
}

pub(super) fn history_lines_height(lines: &[Line<'static>], width: u16) -> u16 {
    lines
        .iter()
        .map(|line| line.width().max(1).div_ceil(usize::from(width.max(1))))
        .sum::<usize>()
        .min(usize::from(u16::MAX)) as u16
}
