use std::sync::{Arc, Mutex};
use std::time::Instant;

use subbake_core::{AgentToolOutcome, ProgressEvent, ProgressSink};

use crate::engine::EngineObserver;
use crate::session::iso_now;
use crate::tool_presentation::{ToolActivityText, completed_activity, failed_activity};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Msg {
    pub style: MsgStyle,
    pub text: String,
    pub stamp: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgStyle {
    User,
    Observation,
    Response,
    Commentary,
    Error,
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolActivityStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolActivity {
    pub call_id: String,
    pub tool_name: String,
    pub headline: String,
    pub detail: Option<String>,
    pub status: ToolActivityStatus,
}

impl ToolActivity {
    fn new(
        call_id: &str,
        tool_name: &str,
        text: ToolActivityText,
        status: ToolActivityStatus,
    ) -> Self {
        Self {
            call_id: call_id.to_owned(),
            tool_name: tool_name.to_owned(),
            headline: text.headline,
            detail: text.detail,
            status,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolGroup {
    pub activities: Vec<ToolActivity>,
}

impl ToolGroup {
    fn update(
        &mut self,
        call_id: &str,
        tool_name: &str,
        text: ToolActivityText,
        status: ToolActivityStatus,
    ) {
        if let Some(activity) = self
            .activities
            .iter_mut()
            .rev()
            .find(|activity| activity.call_id == call_id)
        {
            activity.tool_name = tool_name.to_owned();
            activity.headline = text.headline;
            activity.detail = text.detail;
            activity.status = status;
        } else {
            self.activities
                .push(ToolActivity::new(call_id, tool_name, text, status));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptItem {
    Message(Msg),
    ToolGroup(ToolGroup),
}

#[derive(Debug, Clone)]
pub(super) struct ActiveTool {
    pub(super) call_id: String,
    pub(super) name: String,
    pub(super) arguments: serde_json::Value,
    pub(super) started: Instant,
}

#[derive(Debug, Clone)]
pub struct MsgView {
    pub(super) items: Vec<TranscriptItem>,
    pub(super) active_tool_group: Option<ToolGroup>,
    pub(super) max: usize,
}

impl MsgView {
    pub fn new(max: usize) -> Self {
        Self {
            items: Vec::with_capacity(max.min(4096)),
            active_tool_group: None,
            max,
        }
    }

    pub fn push(&mut self, style: MsgStyle, text: String) {
        self.push_at(style, text, iso_now());
    }

    pub(super) fn push_at(&mut self, style: MsgStyle, text: String, stamp: String) {
        self.push_item(TranscriptItem::Message(Msg { style, text, stamp }));
    }

    pub(super) fn push_commentary(&mut self, text: String) {
        self.seal_tool_group();
        self.push(MsgStyle::Commentary, text);
    }

    pub(super) fn push_response(&mut self, text: String) {
        self.seal_tool_group();
        self.push(MsgStyle::Response, text);
    }

    pub(super) fn start_tool(&mut self, call_id: &str, tool_name: &str, text: ToolActivityText) {
        self.active_tool_group
            .get_or_insert_with(ToolGroup::default)
            .update(call_id, tool_name, text, ToolActivityStatus::Running);
    }

    pub(super) fn finish_tool(
        &mut self,
        call_id: &str,
        tool_name: &str,
        text: ToolActivityText,
        status: ToolActivityStatus,
    ) {
        self.active_tool_group
            .get_or_insert_with(ToolGroup::default)
            .update(call_id, tool_name, text, status);
    }

    pub(super) fn seal_tool_group(&mut self) {
        let Some(group) = self.active_tool_group.take() else {
            return;
        };
        if !group.activities.is_empty() {
            self.push_item(TranscriptItem::ToolGroup(group));
        }
    }

    fn push_item(&mut self, item: TranscriptItem) {
        if self.max == 0 {
            return;
        }
        if self.max != usize::MAX && self.items.len() >= self.max {
            self.items.remove(0);
        }
        self.items.push(item);
    }

    pub fn all(&self) -> &[TranscriptItem] {
        &self.items
    }

    pub fn active_tool_group(&self) -> Option<&ToolGroup> {
        self.active_tool_group.as_ref()
    }

    pub(super) fn replay(&mut self, events: Vec<crate::session::AgentEvent>) {
        let finished_calls = events
            .iter()
            .filter_map(|event| match event.typed() {
                Some(
                    crate::event::EventKind::ToolCompleted { call_id, .. }
                    | crate::event::EventKind::ToolFailed { call_id, .. }
                    | crate::event::EventKind::ToolCancelled { call_id, .. },
                ) => Some(call_id),
                _ => None,
            })
            .collect::<std::collections::HashSet<_>>();

        for event in events {
            let stamp = event.created_at.clone();
            match event.tag() {
                crate::session::EventTag::User => {
                    self.seal_tool_group();
                    self.push_at(MsgStyle::User, event.text, stamp);
                }
                crate::session::EventTag::Assistant | crate::session::EventTag::AskUser => {
                    self.seal_tool_group();
                    self.push_at(MsgStyle::Response, event.text, stamp);
                }
                crate::session::EventTag::Commentary => {
                    self.seal_tool_group();
                    self.push_at(MsgStyle::Commentary, event.text, stamp);
                }
                crate::session::EventTag::ToolStarted => {
                    let Some(crate::event::EventKind::ToolStarted {
                        call_id,
                        tool_name,
                        headline,
                        ..
                    }) = event.typed()
                    else {
                        continue;
                    };
                    if !finished_calls.contains(&call_id) {
                        self.finish_tool(
                            &call_id,
                            &tool_name,
                            ToolActivityText {
                                headline,
                                detail: Some("interrupted".to_owned()),
                            },
                            ToolActivityStatus::Cancelled,
                        );
                    }
                }
                crate::session::EventTag::ToolCompleted => {
                    let Some(crate::event::EventKind::ToolCompleted {
                        call_id,
                        tool_name,
                        headline,
                        detail,
                        ..
                    }) = event.typed()
                    else {
                        continue;
                    };
                    self.finish_tool(
                        &call_id,
                        &tool_name,
                        ToolActivityText { headline, detail },
                        ToolActivityStatus::Completed,
                    );
                }
                crate::session::EventTag::ToolFailed => {
                    let Some(crate::event::EventKind::ToolFailed {
                        call_id,
                        tool_name,
                        headline,
                        detail,
                        ..
                    }) = event.typed()
                    else {
                        continue;
                    };
                    self.finish_tool(
                        &call_id,
                        &tool_name,
                        ToolActivityText { headline, detail },
                        ToolActivityStatus::Failed,
                    );
                }
                crate::session::EventTag::ToolCancelled => {
                    let Some(crate::event::EventKind::ToolCancelled {
                        call_id,
                        tool_name,
                        headline,
                        detail,
                        duration_ms,
                    }) = event.typed()
                    else {
                        continue;
                    };
                    self.finish_tool(
                        &call_id,
                        &tool_name,
                        ToolActivityText {
                            headline,
                            detail: detail.or_else(|| {
                                Some(format!("cancelled · {:.1}s", duration_ms as f64 / 1_000.0))
                            }),
                        },
                        ToolActivityStatus::Cancelled,
                    );
                }
                crate::session::EventTag::FileOperation => {}
                crate::session::EventTag::Plan => {
                    self.seal_tool_group();
                    self.push_at(MsgStyle::System, format!("Plan: {}", event.text), stamp);
                }
                crate::session::EventTag::Error => {
                    self.seal_tool_group();
                    self.push_at(MsgStyle::Error, event.text, stamp);
                }
                crate::session::EventTag::Cancelled => {
                    self.seal_tool_group();
                    self.push_at(MsgStyle::System, "Cancelled.".to_owned(), stamp);
                }
                _ => {}
            }
        }
        self.seal_tool_group();
    }
}

#[derive(Clone)]
pub struct TuiObserver {
    pub view: Arc<Mutex<MsgView>>,
    progress: Arc<Mutex<Option<(ProgressEvent, Instant)>>>,
    active_tool: Arc<Mutex<Option<ActiveTool>>>,
}

impl TuiObserver {
    pub(super) fn new(
        view: Arc<Mutex<MsgView>>,
        progress: Arc<Mutex<Option<(ProgressEvent, Instant)>>>,
        active_tool: Arc<Mutex<Option<ActiveTool>>>,
    ) -> Self {
        Self {
            view,
            progress,
            active_tool,
        }
    }

    fn finish_active(&self, call_id: &str) -> (std::time::Duration, Option<ActiveTool>) {
        let active = self.active_tool.lock().ok().and_then(|mut active| {
            active
                .as_ref()
                .is_some_and(|activity| activity.call_id == call_id)
                .then(|| active.take())
                .flatten()
        });
        let elapsed = active
            .as_ref()
            .map_or(std::time::Duration::ZERO, |activity| {
                activity.started.elapsed()
            });
        if let Ok(mut progress) = self.progress.lock() {
            *progress = None;
        }
        (elapsed, active)
    }
}

impl EngineObserver for TuiObserver {
    fn on_thinking(&mut self, text: &str) {
        let _ = text;
    }

    fn on_commentary(&mut self, text: &str) {
        if let Ok(mut view) = self.view.lock() {
            view.push_commentary(text.to_owned());
        }
    }

    fn on_tool_call(&mut self, call_id: &str, name: &str, arguments: &serde_json::Value) {
        if let Ok(mut view) = self.view.lock() {
            view.start_tool(
                call_id,
                name,
                crate::tool_presentation::running_activity(name, arguments),
            );
        }
        if let Ok(mut active) = self.active_tool.lock() {
            *active = Some(ActiveTool {
                call_id: call_id.to_owned(),
                name: name.to_owned(),
                arguments: arguments.clone(),
                started: Instant::now(),
            });
        }
    }

    fn on_tool_success(
        &mut self,
        call_id: &str,
        name: &str,
        arguments: &serde_json::Value,
        outcome: &AgentToolOutcome,
    ) {
        let (elapsed, _) = self.finish_active(call_id);
        let activity = completed_activity(name, arguments, outcome, elapsed);
        if let Ok(mut view) = self.view.lock() {
            view.finish_tool(call_id, name, activity, ToolActivityStatus::Completed);
        }
    }

    fn on_tool_failure(
        &mut self,
        call_id: &str,
        name: &str,
        arguments: &serde_json::Value,
        error: &str,
    ) {
        self.finish_active(call_id);
        let activity = failed_activity(name, arguments, error);
        if let Ok(mut view) = self.view.lock() {
            view.finish_tool(call_id, name, activity, ToolActivityStatus::Failed);
        }
    }

    fn on_tool_cancelled(&mut self, call_id: &str, name: &str) {
        let (elapsed, active) = self.finish_active(call_id);
        let activity = active.map_or_else(
            || ToolActivityText {
                headline: format!("Cancelled {name}"),
                detail: Some(format!("cancelled · {:.1}s", elapsed.as_secs_f64())),
            },
            |active| crate::tool_presentation::cancelled_activity(name, &active.arguments, elapsed),
        );
        if let Ok(mut view) = self.view.lock() {
            view.finish_tool(call_id, name, activity, ToolActivityStatus::Cancelled);
        }
    }

    fn on_error(&mut self, error: &str) {
        if let Ok(mut view) = self.view.lock() {
            view.seal_tool_group();
            view.push(MsgStyle::Error, error.to_owned());
        }
    }

    fn on_response(&mut self, text: &str) {
        let _ = text;
    }
}

impl ProgressSink for TuiObserver {
    fn emit(&self, event: ProgressEvent) {
        if let Ok(mut progress) = self.progress.lock() {
            let started = progress.as_ref().map_or_else(Instant::now, |(_, at)| *at);
            *progress = Some((event, started));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MsgStyle, MsgView, ToolActivityStatus, TranscriptItem};
    use crate::tool_presentation::ToolActivityText;
    use crate::{event::EventKind, session::AgentEvent};

    fn activity(headline: &str, detail: &str) -> ToolActivityText {
        ToolActivityText {
            headline: headline.to_owned(),
            detail: Some(detail.to_owned()),
        }
    }

    #[test]
    fn tool_lifecycle_updates_one_structured_node_in_place() {
        let mut view = MsgView::new(10);
        view.start_tool(
            "call-1",
            "translate_file",
            activity("Translating a.srt", "zh-CN"),
        );
        view.finish_tool(
            "call-1",
            "translate_file",
            activity("Translated a.srt", "10 cues · 1.0s"),
            ToolActivityStatus::Completed,
        );

        assert!(view.all().is_empty());
        let group = view.active_tool_group().expect("active group");
        assert_eq!(group.activities.len(), 1);
        assert_eq!(group.activities[0].headline, "Translated a.srt");
        assert_eq!(group.activities[0].status, ToolActivityStatus::Completed);
    }

    #[test]
    fn commentary_seals_the_previous_tool_group_and_starts_a_new_phase() {
        let mut view = MsgView::new(10);
        view.start_tool("call-1", "run_command", activity("Running iconv", "cwd ."));
        view.finish_tool(
            "call-1",
            "run_command",
            activity("Ran iconv", "exit 0 · 0.1s"),
            ToolActivityStatus::Completed,
        );
        view.push_commentary("Now translate the UTF-8 copy.".to_owned());

        assert!(view.active_tool_group().is_none());
        assert!(matches!(view.all()[0], TranscriptItem::ToolGroup(_)));
        let TranscriptItem::Message(message) = &view.all()[1] else {
            panic!("commentary message");
        };
        assert_eq!(message.style, MsgStyle::Commentary);
    }

    #[test]
    fn zero_capacity_view_discards_committed_items_without_panicking() {
        let mut view = MsgView::new(0);
        view.push(MsgStyle::System, "discarded".to_owned());
        assert!(view.all().is_empty());
    }

    #[test]
    fn replay_rebuilds_the_same_contiguous_tool_group() {
        let events = [
            EventKind::Commentary {
                text: "Convert, then translate.".to_owned(),
            },
            EventKind::ToolStarted {
                call_id: "command".to_owned(),
                tool_name: "run_command".to_owned(),
                headline: "Running iconv".to_owned(),
                detail: None,
            },
            EventKind::ToolCompleted {
                call_id: "command".to_owned(),
                tool_name: "run_command".to_owned(),
                headline: "Ran iconv".to_owned(),
                detail: Some("exit 0 · 0.1s".to_owned()),
                duration_ms: 100,
            },
            EventKind::ToolStarted {
                call_id: "translate".to_owned(),
                tool_name: "translate_file".to_owned(),
                headline: "Translating a.srt".to_owned(),
                detail: None,
            },
            EventKind::ToolCompleted {
                call_id: "translate".to_owned(),
                tool_name: "translate_file".to_owned(),
                headline: "Translated a.srt".to_owned(),
                detail: Some("10 cues · 1.0s".to_owned()),
                duration_ms: 1_000,
            },
            EventKind::Assistant {
                text: "Done.".to_owned(),
            },
        ]
        .iter()
        .map(AgentEvent::from_kind)
        .collect();
        let mut view = MsgView::new(10);
        view.replay(events);

        assert_eq!(view.all().len(), 3);
        let TranscriptItem::ToolGroup(group) = &view.all()[1] else {
            panic!("tool group");
        };
        assert_eq!(group.activities.len(), 2);
        assert!(
            group
                .activities
                .iter()
                .all(|activity| activity.status == ToolActivityStatus::Completed)
        );
    }
}
