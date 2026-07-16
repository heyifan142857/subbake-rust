use std::sync::{Arc, Mutex};
use std::time::Instant;

use subbake_core::{ProgressEvent, ProgressSink};

use crate::engine::EngineObserver;
use crate::session::iso_now;

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
    Observation,
    Response,
    Error,
    System,
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
    last_tool: Arc<Mutex<Option<String>>>,
}

impl TuiObserver {
    pub(super) fn new(
        view: Arc<Mutex<MsgView>>,
        progress: Arc<Mutex<Option<(ProgressEvent, Instant)>>>,
    ) -> Self {
        Self {
            view,
            progress,
            last_tool: Arc::new(Mutex::new(None)),
        }
    }
}

impl EngineObserver for TuiObserver {
    fn on_thinking(&mut self, text: &str) {
        let _ = text;
    }

    fn on_tool_call(&mut self, name: &str, arguments: &serde_json::Value) {
        if let Ok(mut last) = self.last_tool.lock() {
            *last = Some(name.to_owned());
        }
        if let Ok(mut view) = self.view.lock() {
            let args = serde_json::to_string(arguments).unwrap_or_default();
            view.push(MsgStyle::ToolCall, format!("⚡ {name} {args}"));
        }
    }

    fn on_observation(&mut self, text: &str) {
        if self
            .last_tool
            .lock()
            .ok()
            .and_then(|value| value.clone())
            .is_some_and(|name| matches!(name.as_str(), "read_file" | "read_file_preview"))
        {
            return;
        }
        if let Ok(mut view) = self.view.lock() {
            view.push(
                MsgStyle::Observation,
                format!("◀ {}", text.lines().next().unwrap_or(text)),
            );
        }
    }

    fn on_error(&mut self, error: &str) {
        if let Ok(mut view) = self.view.lock() {
            view.push(MsgStyle::Error, format!("✖ {error}"));
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
