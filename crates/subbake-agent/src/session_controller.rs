use std::path::{Path, PathBuf};

use crate::error::AgentResult;
use crate::event::EventKind;
use crate::session::{AgentEvent, AgentSession, AgentSessionStore};

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
        session.events.push(AgentEvent::from_kind(&kind));
        session.updated_at = crate::session::iso_now();
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
