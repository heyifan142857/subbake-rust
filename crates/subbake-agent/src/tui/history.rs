use std::sync::{Arc, Mutex};
use std::time::Instant;

use subbake_core::{AgentToolOutcome, ProgressEvent, ProgressSink};

use crate::engine::EngineObserver;
use crate::session::iso_now;
use crate::tool_presentation::{completed_activity, failed_activity};

#[derive(Debug, Clone)]
pub struct Msg {
    pub style: MsgStyle,
    pub text: String,
    pub stamp: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgStyle {
    User,
    ToolCall,
    ToolFailure,
    ToolCancelled,
    Observation,
    Response,
    Commentary,
    Error,
    System,
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
    pub(super) messages: Vec<Msg>,
    pub(super) max: usize,
}

impl MsgView {
    pub fn new(max: usize) -> Self {
        Self {
            messages: Vec::with_capacity(max.min(4096)),
            max,
        }
    }

    pub fn push(&mut self, style: MsgStyle, text: String) {
        let stamp = iso_now();
        if self.max != usize::MAX && self.messages.len() >= self.max {
            self.messages.remove(0);
        }
        self.messages.push(Msg { style, text, stamp });
    }

    pub fn all(&self) -> &[Msg] {
        &self.messages
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

    fn finish_active(&self, call_id: &str) -> std::time::Duration {
        let elapsed = self
            .active_tool
            .lock()
            .ok()
            .and_then(|mut active| {
                active
                    .as_ref()
                    .is_some_and(|activity| activity.call_id == call_id)
                    .then(|| active.take())
                    .flatten()
            })
            .map_or(std::time::Duration::ZERO, |activity| {
                activity.started.elapsed()
            });
        if let Ok(mut progress) = self.progress.lock() {
            *progress = None;
        }
        elapsed
    }
}

impl EngineObserver for TuiObserver {
    fn on_thinking(&mut self, text: &str) {
        let _ = text;
    }

    fn on_commentary(&mut self, text: &str) {
        if let Ok(mut view) = self.view.lock() {
            view.push(MsgStyle::Commentary, format!("➔ {text}"));
        }
    }

    fn on_tool_call(&mut self, call_id: &str, name: &str, arguments: &serde_json::Value) {
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
        let activity = completed_activity(name, arguments, outcome, self.finish_active(call_id));
        if let Ok(mut view) = self.view.lock() {
            view.push(MsgStyle::ToolCall, activity_message("✓", activity));
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
            view.push(MsgStyle::ToolFailure, activity_message("×", activity));
        }
    }

    fn on_tool_cancelled(&mut self, call_id: &str, _name: &str) {
        self.finish_active(call_id);
    }

    fn on_error(&mut self, error: &str) {
        if let Ok(mut view) = self.view.lock() {
            view.push(MsgStyle::Error, format!("× {error}"));
        }
    }

    fn on_response(&mut self, text: &str) {
        let _ = text;
    }
}

fn activity_message(marker: &str, activity: crate::tool_presentation::ToolActivityText) -> String {
    activity.detail.map_or_else(
        || format!("  {marker} {}", activity.headline),
        |detail| format!("  {marker} {}\n    {detail}", activity.headline),
    )
}

impl ProgressSink for TuiObserver {
    fn emit(&self, event: ProgressEvent) {
        if let Ok(mut progress) = self.progress.lock() {
            let started = progress.as_ref().map_or_else(Instant::now, |(_, at)| *at);
            *progress = Some((event, started));
        }
    }
}
