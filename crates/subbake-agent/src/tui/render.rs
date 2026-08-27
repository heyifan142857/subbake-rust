use std::io;

use subbake_core::ProgressEvent;

use super::history::ActiveTool;
use super::layout::{ActiveLayout, ActiveSurface, UiLayoutState};
use super::progress::spinner_frame;
use super::{InputMode, InteractionState, StartupInfo, SubBakeTui};
use crate::engine::{ProfileChoice, SessionChoice};
use crate::input_editor::InputEditor;
use crate::tui_state::ConfigEditorState;

pub(super) enum OverlaySnapshot {
    Sessions(Vec<SessionChoice>),
    Profiles(Vec<ProfileChoice>),
    ProfileCreation,
    Config(ConfigEditorState),
    Approval(crate::engine::ApprovalPrompt, bool),
}

pub(super) struct ViewSnapshot {
    pub(super) input: InputEditor,
    pub(super) input_hint: &'static str,
    pub(super) suggestions: Vec<(String, String)>,
    pub(super) selected_suggestion: usize,
    pub(super) editing: bool,
    pub(super) progress: Option<(ProgressEvent, std::time::Instant)>,
    pub(super) active_tool: Option<ActiveTool>,
    pub(super) spinner: char,
    pub(super) startup_info: StartupInfo,
    pub(super) plan_mode: bool,
    pub(super) overlay: Option<OverlaySnapshot>,
    show_progress: bool,
}

impl ViewSnapshot {
    fn new(app: &SubBakeTui) -> Self {
        let suggestions = app.suggestions();
        let overlay = match app.interaction_state.input_mode() {
            InputMode::ChoosingSession(picker) => {
                Some(OverlaySnapshot::Sessions(picker.options.clone()))
            }
            InputMode::ChoosingProfile(picker) | InputMode::ChoosingConfigProfile(picker) => {
                Some(OverlaySnapshot::Profiles(picker.options.clone()))
            }
            InputMode::CreatingProfile | InputMode::CreatingConfigProfile => {
                Some(OverlaySnapshot::ProfileCreation)
            }
            InputMode::AwaitingApproval | InputMode::RevisingApproval => {
                app.approval_prompt.clone().map(|prompt| {
                    OverlaySnapshot::Approval(
                        prompt,
                        matches!(
                            app.interaction_state.input_mode(),
                            InputMode::RevisingApproval
                        ),
                    )
                })
            }
            _ => app.config_editor.clone().map(OverlaySnapshot::Config),
        };
        let selected_count = match &overlay {
            Some(OverlaySnapshot::Sessions(options)) => options.len(),
            Some(OverlaySnapshot::Profiles(options)) => options.len(),
            _ => suggestions.len(),
        };
        let processing = app.interaction_state.is_processing();
        let toggling_plan_mode = matches!(
            app.interaction_state,
            InteractionState::Processing {
                plan_mode_rollback: Some(_),
                ..
            }
        );

        Self {
            input: app.input.clone(),
            input_hint: if processing {
                "Type a follow-up while SubBake works"
            } else if matches!(
                app.interaction_state.input_mode(),
                InputMode::RevisingApproval
            ) {
                "Tell the agent what to do instead"
            } else {
                app.input_hint
            },
            suggestions,
            selected_suggestion: app.suggestion_index.min(selected_count.saturating_sub(1)),
            editing: matches!(
                app.interaction_state.input_mode(),
                InputMode::Editing | InputMode::RevisingApproval
            ),
            progress: app.progress.lock().ok().and_then(|value| value.clone()),
            active_tool: app
                .active_tool
                .lock()
                .ok()
                .and_then(|activity| activity.clone()),
            spinner: spinner_frame(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default(),
            ),
            startup_info: app.startup_info.clone(),
            plan_mode: app.plan_mode,
            overlay,
            show_progress: processing && !toggling_plan_mode,
        }
    }

    fn surface(&self) -> ActiveSurface {
        match self.overlay {
            Some(OverlaySnapshot::Sessions(_) | OverlaySnapshot::Profiles(_)) => {
                ActiveSurface::Picker
            }
            Some(OverlaySnapshot::ProfileCreation) => ActiveSurface::ProfileCreation,
            Some(OverlaySnapshot::Config(_)) => ActiveSurface::ConfigEditor,
            Some(OverlaySnapshot::Approval(_, _)) => ActiveSurface::Approval,
            None => ActiveSurface::Composer,
        }
    }

    fn layout_state(&self) -> UiLayoutState<'_> {
        UiLayoutState {
            surface: self.surface(),
            input: &self.input,
            suggestion_count: self.suggestions.len(),
            show_progress: self.show_progress,
        }
    }
}

pub(super) fn draw(app: &mut SubBakeTui) -> io::Result<()> {
    app.invalidate_layout();
    let snapshot = ViewSnapshot::new(app);
    let mut drawn_layout = None;
    let draw_ui = |frame: &mut ratatui::Frame<'_>| {
        let layout = ActiveLayout::calculate(frame.area(), snapshot.layout_state());
        if let Some(composer) = layout.composer {
            super::main_view::render(frame, composer, &snapshot);
        } else if let Some(overlay) = layout.overlay {
            super::overlay_view::render(frame, overlay, &snapshot);
        }
        drawn_layout = Some(layout);
    };

    if let Some(terminal) = app.overlay_terminal.as_mut() {
        terminal.draw(draw_ui)?;
    } else {
        app.terminal.draw(draw_ui)?;
    }
    app.active_layout = drawn_layout;
    Ok(())
}

#[cfg(test)]
mod tests {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    use super::ViewSnapshot;
    use crate::input_editor::InputEditor;
    use crate::tui::layout::{ActiveLayout, ActiveSurface, UiLayoutState};

    fn render_snapshot(
        backend_width: u16,
        backend_height: u16,
        snapshot: &ViewSnapshot,
    ) -> (
        ActiveLayout,
        ratatui::buffer::Buffer,
        ratatui::layout::Position,
    ) {
        let backend = TestBackend::new(backend_width, backend_height);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let mut layout = None;
        terminal
            .draw(|frame| {
                let active = ActiveLayout::calculate(frame.area(), snapshot.layout_state());
                if let Some(composer) = active.composer {
                    crate::tui::main_view::render(frame, composer, snapshot);
                } else if let Some(overlay) = active.overlay {
                    crate::tui::overlay_view::render(frame, overlay, snapshot);
                }
                layout = Some(active);
            })
            .expect("draw");
        let cursor = terminal.get_cursor_position().expect("cursor");
        let buffer = terminal.backend().buffer().clone();
        (layout.expect("layout"), buffer, cursor)
    }

    fn composer_snapshot(text: &str) -> ViewSnapshot {
        let mut input = InputEditor::default();
        input.set_text(text.to_owned());
        ViewSnapshot {
            input,
            input_hint: "hint",
            suggestions: Vec::new(),
            selected_suggestion: 0,
            editing: true,
            progress: None,
            active_tool: None,
            spinner: '·',
            startup_info: super::StartupInfo {
                model: "very-long-model-name".to_owned(),
                cwd: "/a/very/long/current/directory".to_owned(),
                ..super::StartupInfo::default()
            },
            plan_mode: true,
            overlay: None,
            show_progress: false,
        }
    }

    #[test]
    fn test_backend_reflows_mixed_unicode_after_resize() {
        let text = "long English words 中文🙂 and another long segment\nsecond line";
        let narrow = composer_snapshot(text);
        let (narrow_layout, narrow_buffer, narrow_cursor) = render_snapshot(20, 12, &narrow);
        let wide = composer_snapshot(text);
        let (wide_layout, wide_buffer, wide_cursor) = render_snapshot(80, 12, &wide);

        assert!(
            narrow_layout.composer.expect("composer").input_line_count
                > wide_layout.composer.expect("composer").input_line_count
        );
        assert_eq!(narrow_buffer.area, Rect::new(0, 0, 20, 12));
        assert_eq!(wide_buffer.area, Rect::new(0, 0, 80, 12));
        assert!(narrow_cursor.x < 20 && narrow_cursor.y < 12);
        assert!(wide_cursor.x < 80 && wide_cursor.y < 12);
    }

    #[test]
    fn narrow_main_states_and_overlays_render_inside_the_test_backend() {
        use std::path::PathBuf;

        use super::OverlaySnapshot;
        use crate::engine::{ProfileChoice, SessionChoice};
        use crate::tui_state::ConfigEditorState;
        use crate::{ConfigEditorSnapshot, ConfigFieldId, ConfigFieldView};

        for width in [20, 40] {
            let mut approvals = composer_snapshot("");
            approvals.suggestions = vec![
                (
                    "approve".to_owned(),
                    "execute the pending plan 中文".to_owned(),
                ),
                ("reject".to_owned(), "discard the pending plan".to_owned()),
                (
                    "tell agent what to do".to_owned(),
                    "revise with instructions".to_owned(),
                ),
            ];
            approvals.show_progress = true;
            let (_, buffer, _) = render_snapshot(width, 12, &approvals);
            assert_eq!(buffer.area.width, width);

            let mut picker = composer_snapshot("");
            picker.overlay = Some(OverlaySnapshot::Profiles(vec![ProfileChoice {
                name: "一个非常长的-profile-name".to_owned(),
                provider: "a-provider-with-a-long-name".to_owned(),
                model: "a-model-with-a-long-name".to_owned(),
                active: true,
                create: false,
            }]));
            let (_, buffer, _) = render_snapshot(width, 12, &picker);
            assert_eq!(buffer.area.width, width);

            let mut sessions = composer_snapshot("");
            sessions.overlay = Some(OverlaySnapshot::Sessions(vec![SessionChoice {
                id: "session-id".to_owned(),
                title: "很长的 session title that must be clipped".to_owned(),
                updated_at: "2026-08-22T00:00:00Z".to_owned(),
                cwd: "/a/very/long/project/path".to_owned(),
                event_count: 123,
                active: true,
            }]));
            let (_, buffer, _) = render_snapshot(width, 12, &sessions);
            assert_eq!(buffer.area.width, width);

            let mut creation = composer_snapshot("profile_name_that_is_longer_than_the_screen");
            creation.overlay = Some(OverlaySnapshot::ProfileCreation);
            let (_, buffer, cursor) = render_snapshot(width, 12, &creation);
            assert_eq!(buffer.area.width, width);
            assert!(cursor.x < width && cursor.y < 12);

            let snapshot = ConfigEditorSnapshot {
                path: PathBuf::from("a/very/long/path/to/subbake.toml"),
                target: subbake_adapters::ConfigEditTarget::Defaults,
                active_profile: None,
                profiles: Vec::new(),
                fields: vec![ConfigFieldView {
                    id: ConfigFieldId::AgentMaxSteps,
                    value: "a value that is wider than the available editor".to_owned(),
                    inherited: true,
                    configured: true,
                }],
            };
            let mut editor = ConfigEditorState::new(snapshot);
            editor.section_index = crate::ConfigSection::ALL
                .iter()
                .position(|section| *section == crate::ConfigSection::Agent)
                .expect("agent section");
            editor.focus = crate::tui_state::ConfigFocus::Fields;
            editor.editing_field = Some(ConfigFieldId::AgentMaxSteps);
            let mut config = composer_snapshot("123456789012345678901234567890");
            config.overlay = Some(OverlaySnapshot::Config(editor));
            let (_, buffer, cursor) = render_snapshot(width, 12, &config);
            assert_eq!(buffer.area.width, width);
            assert!(cursor.x < width && cursor.y < 12);
        }
    }

    #[test]
    fn approval_panel_renders_typed_content_at_supported_widths() {
        use super::OverlaySnapshot;
        use crate::engine::{ApprovalKind, ApprovalPrompt};

        let operation = "ffmpeg -i movie.mkv -map 0 -map 1 -c copy -metadata:s:s:0 language=zho output-with-a-long-name.mkv";
        for width in [40, 80, 120] {
            let mut snapshot = composer_snapshot("");
            snapshot.suggestions = crate::tui_state::APPROVAL_OPTIONS
                .iter()
                .map(|(label, description)| (label.to_string(), description.to_string()))
                .collect();
            snapshot.overlay = Some(OverlaySnapshot::Approval(
                ApprovalPrompt {
                    kind: ApprovalKind::Command,
                    title: "Run this operation?".to_owned(),
                    purpose: "Embed the translated bilingual subtitle".to_owned(),
                    reason: "The command writes the requested MKV output".to_owned(),
                    operation: vec![operation.to_owned()],
                },
                false,
            ));
            let (_, buffer, cursor) = render_snapshot(width, 12, &snapshot);
            let rendered = buffer
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>();
            assert!(rendered.contains("Run this operation?"));
            assert!(rendered.contains("Purpose:"));
            assert!(rendered.contains("Reason:"));
            assert!(rendered.contains("ffmpeg"));
            assert!(rendered.contains("Approve once"));
            assert!(cursor.x < width && cursor.y < 12);

            let mut revising = snapshot;
            revising
                .input
                .set_text("Use the matching SRT instead".to_owned());
            let Some(OverlaySnapshot::Approval(_, mode)) = revising.overlay.as_mut() else {
                panic!("approval snapshot");
            };
            *mode = true;
            let (_, buffer, cursor) = render_snapshot(width, 12, &revising);
            let rendered = buffer
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>();
            assert!(rendered.contains("Feedback"));
            assert!(rendered.contains("matching SRT"));
            assert!(cursor.x < width && cursor.y < 12);
        }
    }

    #[test]
    fn zero_and_one_column_layouts_do_not_panic() {
        let input = InputEditor::default();
        for width in [0, 1] {
            let layout = ActiveLayout::calculate(
                Rect::new(0, 0, width, 1),
                UiLayoutState {
                    surface: ActiveSurface::Composer,
                    input: &input,
                    suggestion_count: 0,
                    show_progress: false,
                },
            );
            assert_eq!(layout.frame.width, width);
        }
    }

    #[test]
    fn config_confirmation_uses_native_block_shadow() {
        use std::path::PathBuf;

        use super::OverlaySnapshot;
        use crate::ConfigEditorSnapshot;
        use crate::tui_state::{ConfigConfirm, ConfigEditorState};

        let mut editor = ConfigEditorState::new(ConfigEditorSnapshot {
            path: PathBuf::from("subbake.toml"),
            target: subbake_adapters::ConfigEditTarget::Defaults,
            active_profile: None,
            profiles: Vec::new(),
            fields: Vec::new(),
        });
        editor.confirm = Some(ConfigConfirm::Close);
        let mut snapshot = composer_snapshot("");
        snapshot.overlay = Some(OverlaySnapshot::Config(editor));

        let (layout, buffer, _) = render_snapshot(80, 20, &snapshot);
        let popup = layout.overlay.expect("overlay").popup;
        let shadow = &buffer[(popup.right(), popup.y.saturating_add(1))];

        assert_eq!(shadow.symbol(), "▓");
        assert_eq!(shadow.fg, ratatui::style::Color::DarkGray);
    }
}
