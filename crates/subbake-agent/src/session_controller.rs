use std::path::{Path, PathBuf};

use crate::error::AgentResult;
use crate::event::EventKind;
use crate::session::{AgentEvent, AgentSession, AgentSessionStore, iso_now};

pub(crate) struct SessionController<'a> {
    store: &'a AgentSessionStore,
    active: &'a mut Option<AgentSession>,
}

impl<'a> SessionController<'a> {
    pub(crate) fn new(store: &'a AgentSessionStore, active: &'a mut Option<AgentSession>) -> Self {
        Self { store, active }
    }

    pub(crate) fn start(&mut self) -> AgentResult<()> {
        *self.active = Some(self.store.create()?);
        Ok(())
    }

    pub(crate) fn resume(&mut self, id: Option<&str>) -> AgentResult<()> {
        let session = match id {
            Some(id) => self.store.load(id)?,
            None => self
                .store
                .latest()?
                .ok_or_else(|| std::io::Error::other("no sessions to resume"))?,
        };
        *self.active = Some(session);
        Ok(())
    }

    pub(crate) fn set_config_path(&mut self, path: Option<&Path>) -> AgentResult<()> {
        let session = self
            .active
            .as_mut()
            .ok_or_else(|| std::io::Error::other("no active session"))?;
        session.config_path = path.map(|path| path.to_string_lossy().into_owned());
        self.store.save(session)
    }

    pub(crate) fn record(&mut self, kind: EventKind) -> AgentResult<()> {
        let session = self
            .active
            .as_mut()
            .ok_or_else(|| std::io::Error::other("no active session"))?;
        let (kind, text, data) = serialize_event(&kind);
        session.events.push(AgentEvent {
            kind,
            text,
            data,
            created_at: iso_now(),
        });
        session.updated_at = iso_now();
        self.store.save(session)
    }

    pub(crate) fn record_error(&mut self, error: &str) -> AgentResult<PathBuf> {
        self.record(EventKind::Error {
            text: error.to_owned(),
        })?;
        let session = self
            .active
            .as_ref()
            .ok_or_else(|| std::io::Error::other("no active session"))?;
        Ok(self.store.path_for(&session.id))
    }
}

fn serialize_event(kind: &EventKind) -> (String, String, serde_json::Value) {
    match kind {
        EventKind::User { text } => ("user".into(), text.clone(), serde_json::json!({})),
        EventKind::Assistant { text } => ("assistant".into(), text.clone(), serde_json::json!({})),
        EventKind::AskUser { text } => ("ask_user".into(), text.clone(), serde_json::json!({})),
        EventKind::ToolCall {
            tool_name,
            arguments,
        } => (
            "tool_call".into(),
            tool_name.clone(),
            serde_json::json!({"tool_name": tool_name, "arguments": arguments}),
        ),
        EventKind::FinalToolCall {
            tool_name,
            arguments,
        } => (
            "final_tool_call".into(),
            tool_name.clone(),
            serde_json::json!({"tool_name": tool_name, "arguments": arguments}),
        ),
        EventKind::FileOperation(data) => (
            "file_operation".into(),
            format!("{} {}", data.action, data.path),
            serde_json::to_value(data).unwrap_or_default(),
        ),
        EventKind::Plan {
            message,
            tool_calls,
        } => (
            "plan".into(),
            message.clone(),
            serde_json::json!({"message": message, "tool_calls": tool_calls}),
        ),
        EventKind::Approve => ("approve".into(), String::new(), serde_json::json!({})),
        EventKind::Reject => ("reject".into(), String::new(), serde_json::json!({})),
        EventKind::Undo => ("undo".into(), String::new(), serde_json::json!({})),
        EventKind::Profile { name } => ("profile".into(), name.clone(), serde_json::json!({})),
        EventKind::Error { text } => ("error".into(), text.clone(), serde_json::json!({})),
        EventKind::Cancelled => (
            "cancelled".into(),
            "Cancelled.".into(),
            serde_json::json!({}),
        ),
    }
}
