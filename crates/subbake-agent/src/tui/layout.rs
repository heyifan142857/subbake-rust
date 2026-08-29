use ratatui::layout::Rect;

use crate::input_editor::InputEditor;

#[derive(Debug, Clone, Copy)]
pub(super) enum ActiveSurface {
    Composer,
    Picker,
    ProfileCreation,
    ConfigEditor,
    Approval,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct UiLayoutState<'a> {
    pub(super) surface: ActiveSurface,
    pub(super) input: &'a InputEditor,
    pub(super) suggestion_count: usize,
    pub(super) show_progress: bool,
    pub(super) progress_line_count: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ActiveLayout {
    pub(super) frame: Rect,
    pub(super) composer: Option<ComposerLayout>,
    pub(super) overlay: Option<OverlayLayout>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ComposerLayout {
    pub(super) suggestions: Rect,
    pub(super) progress: Rect,
    pub(super) input: Rect,
    pub(super) input_entry: Rect,
    pub(super) status: Rect,
    pub(super) input_content_width: u16,
    pub(super) input_line_count: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct OverlayLayout {
    pub(super) outer: Rect,
    pub(super) inner: Rect,
    pub(super) header: Rect,
    pub(super) body: Rect,
    pub(super) footer: Rect,
    pub(super) sidebar: Option<Rect>,
    pub(super) content: Rect,
    pub(super) popup: Rect,
}

impl ActiveLayout {
    pub(super) fn calculate(area: Rect, state: UiLayoutState<'_>) -> Self {
        match state.surface {
            ActiveSurface::Composer => Self {
                frame: area,
                composer: Some(ComposerLayout::calculate(area, state)),
                overlay: None,
            },
            surface => Self {
                frame: area,
                composer: None,
                overlay: Some(OverlayLayout::calculate(area, surface)),
            },
        }
    }
}

impl ComposerLayout {
    fn calculate(area: Rect, state: UiLayoutState<'_>) -> Self {
        let input_content_width = area.width.saturating_sub(2).max(1);
        let max_input_lines = (area.height.saturating_mul(40) / 100).max(1);
        let desired_input_lines = state
            .input
            .desired_height(input_content_width)
            .min(max_input_lines);
        let requested_input_height = desired_input_lines.saturating_add(3);
        let input_height = requested_input_height.min(area.height);
        let remaining = area.height.saturating_sub(input_height);
        let progress_height = if state.show_progress {
            state.progress_line_count.min(remaining)
        } else {
            0
        };
        let suggestion_height = u16::try_from(state.suggestion_count)
            .unwrap_or(u16::MAX)
            .min(remaining.saturating_sub(progress_height));
        let used_height = input_height
            .saturating_add(progress_height)
            .saturating_add(suggestion_height);
        let top_gap = u16::from(used_height < area.height);
        let top = area.y.saturating_add(top_gap);
        let suggestions = Rect::new(area.x, top, area.width, suggestion_height);
        let progress = Rect::new(area.x, suggestions.bottom(), area.width, progress_height);
        let input = Rect::new(area.x, progress.bottom(), area.width, input_height);
        let status_height = u16::from(input.height > 0);
        let status = Rect::new(
            input.x,
            input.bottom().saturating_sub(status_height),
            input.width,
            status_height,
        );
        let input_entry = Rect::new(
            input.x,
            input.y,
            input.width,
            input.height.saturating_sub(status_height),
        );
        let input_line_count = input_entry.height.saturating_sub(2);

        Self {
            suggestions,
            progress,
            input,
            input_entry,
            status,
            input_content_width,
            input_line_count,
        }
    }
}

impl OverlayLayout {
    fn calculate(area: Rect, surface: ActiveSurface) -> Self {
        let inner = if matches!(surface, ActiveSurface::Approval) {
            inset(area, 1, 0)
        } else {
            inset(area, 1, 1)
        };
        let (header_height, footer_height) = match surface {
            ActiveSurface::Picker => (3, 1),
            ActiveSurface::ConfigEditor => (2, 2),
            ActiveSurface::ProfileCreation => (3, 1),
            ActiveSurface::Approval => (1, 1),
            ActiveSurface::Composer => (0, 0),
        };
        let header_height = header_height.min(inner.height);
        let footer_height = footer_height.min(inner.height.saturating_sub(header_height));
        let body_height = inner
            .height
            .saturating_sub(header_height)
            .saturating_sub(footer_height);
        let header = Rect::new(inner.x, inner.y, inner.width, header_height);
        let body = Rect::new(inner.x, header.bottom(), inner.width, body_height);
        let footer = Rect::new(inner.x, body.bottom(), inner.width, footer_height);
        let sidebar = if matches!(surface, ActiveSurface::ConfigEditor) && body.width >= 64 {
            let width = (body.width / 4).clamp(16, 22);
            Some(Rect::new(body.x, body.y, width, body.height))
        } else {
            None
        };
        let content = sidebar.map_or(body, |sidebar| {
            Rect::new(
                sidebar.right(),
                body.y,
                body.width.saturating_sub(sidebar.width),
                body.height,
            )
        });

        Self {
            outer: area,
            inner,
            header,
            body,
            footer,
            sidebar,
            content,
            popup: centered_rect(58, 7, area),
        }
    }
}

fn inset(area: Rect, horizontal: u16, vertical: u16) -> Rect {
    let x_inset = horizontal.min(area.width / 2);
    let y_inset = vertical.min(area.height / 2);
    Rect::new(
        area.x.saturating_add(x_inset),
        area.y.saturating_add(y_inset),
        area.width.saturating_sub(x_inset.saturating_mul(2)),
        area.height.saturating_sub(y_inset.saturating_mul(2)),
    )
}

fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

#[cfg(test)]
mod tests {
    use super::{ActiveLayout, ActiveSurface, UiLayoutState};
    use crate::input_editor::InputEditor;
    use ratatui::layout::Rect;

    #[test]
    fn layouts_stay_inside_every_supported_width() {
        let input = InputEditor::default();
        for width in [0, 1, 20, 40, 63, 64, 80, 120] {
            let area = Rect::new(7, 3, width, 12);
            for surface in [
                ActiveSurface::Composer,
                ActiveSurface::Picker,
                ActiveSurface::ProfileCreation,
                ActiveSurface::ConfigEditor,
            ] {
                let layout = ActiveLayout::calculate(
                    area,
                    UiLayoutState {
                        surface,
                        input: &input,
                        suggestion_count: 12,
                        show_progress: true,
                        progress_line_count: 6,
                    },
                );
                for rect in rects(layout) {
                    assert!(contains(area, rect), "{width}: {rect:?} outside {area:?}");
                }
                if let Some(overlay) = layout.overlay {
                    assert!(overlay.header.bottom() <= overlay.body.y);
                    assert!(overlay.body.bottom() <= overlay.footer.y);
                    if let Some(sidebar) = overlay.sidebar {
                        assert!(sidebar.right() <= overlay.content.x);
                    }
                }
            }
        }
    }

    #[test]
    fn config_editor_switches_to_two_columns_at_64_content_columns() {
        let input = InputEditor::default();
        let layout = |width| {
            ActiveLayout::calculate(
                Rect::new(0, 0, width, 20),
                UiLayoutState {
                    surface: ActiveSurface::ConfigEditor,
                    input: &input,
                    suggestion_count: 0,
                    show_progress: false,
                    progress_line_count: 0,
                },
            )
            .overlay
            .expect("overlay")
        };
        assert!(layout(65).sidebar.is_none());
        assert!(layout(66).sidebar.is_some());
    }

    #[test]
    fn composer_regions_are_ordered_and_non_overlapping() {
        let input = InputEditor::default();
        for width in [0, 1, 20, 40, 63, 64, 80, 120] {
            let composer = ActiveLayout::calculate(
                Rect::new(0, 0, width, 12),
                UiLayoutState {
                    surface: ActiveSurface::Composer,
                    input: &input,
                    suggestion_count: 8,
                    show_progress: true,
                    progress_line_count: 6,
                },
            )
            .composer
            .expect("composer");
            assert!(composer.suggestions.bottom() <= composer.progress.y);
            assert!(composer.progress.bottom() <= composer.input.y);
            assert!(composer.input_entry.bottom() <= composer.status.y);
        }
    }

    #[test]
    fn idle_composer_stays_close_to_the_content_above() {
        let input = InputEditor::default();
        let area = Rect::new(0, 5, 80, 12);
        let composer = ActiveLayout::calculate(
            area,
            UiLayoutState {
                surface: ActiveSurface::Composer,
                input: &input,
                suggestion_count: 0,
                show_progress: false,
                progress_line_count: 0,
            },
        )
        .composer
        .expect("composer");

        assert_eq!(composer.input.y, area.y + 1);
        assert_eq!(composer.input.height, 4);
    }

    fn rects(layout: ActiveLayout) -> Vec<Rect> {
        let mut result = vec![layout.frame];
        if let Some(composer) = layout.composer {
            result.extend([
                composer.suggestions,
                composer.progress,
                composer.input,
                composer.input_entry,
                composer.status,
            ]);
        }
        if let Some(overlay) = layout.overlay {
            result.extend([
                overlay.outer,
                overlay.inner,
                overlay.header,
                overlay.body,
                overlay.footer,
                overlay.content,
                overlay.popup,
            ]);
            result.extend(overlay.sidebar);
        }
        result
    }

    fn contains(parent: Rect, child: Rect) -> bool {
        child.x >= parent.x
            && child.y >= parent.y
            && child.right() <= parent.right()
            && child.bottom() <= parent.bottom()
    }
}
