// Agent session — event log (the source of truth for undo, replay, and resume).
//
// Version 1 of the persisted agent-session JSON contract. The session JSON
// lives at `<project_root>/.subbake/agent/sessions/<id>.json`.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::event::PendingPlan;

pub const SESSION_VERSION: u64 = 1;

/// Stable discriminants for the v1 wire-format event kinds. `Unknown` keeps
/// older or future events readable without allowing ad-hoc comparisons in
/// runtime logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventTag {
    User,
    Assistant,
    AskUser,
    ToolCall,
    FinalToolCall,
    FileOperation,
    Plan,
    Approve,
    Reject,
    Undo,
    Profile,
    Error,
    Cancelled,
    Unknown,
}

impl EventTag {
    pub fn parse(value: &str) -> Self {
        match value {
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "ask_user" => Self::AskUser,
            "tool_call" => Self::ToolCall,
            "final_tool_call" => Self::FinalToolCall,
            "file_operation" => Self::FileOperation,
            "plan" => Self::Plan,
            "approve" => Self::Approve,
            "reject" => Self::Reject,
            "undo" => Self::Undo,
            "profile" => Self::Profile,
            "error" => Self::Error,
            "cancelled" => Self::Cancelled,
            _ => Self::Unknown,
        }
    }
}

/// The persisted session mode. Serde keeps the v1 JSON representation as the
/// existing lowercase string while preventing invalid in-memory modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SessionMode {
    #[default]
    Chat,
    Plan,
}

impl std::fmt::Display for SessionMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Chat => "chat",
            Self::Plan => "plan",
        })
    }
}

/// A single event recorded in a session. The `kind` field discriminates the
/// event type; `data` carries type-specific payloads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentEvent {
    pub kind: String,
    pub text: String,
    pub data: serde_json::Value,
    pub created_at: String,
}

impl AgentEvent {
    pub fn tag(&self) -> EventTag {
        EventTag::parse(&self.kind)
    }
}

/// An interactive agent session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentSession {
    pub version: u64,
    pub id: String,
    pub created_at: String,
    pub updated_at: String,
    pub cwd: String,
    pub profile: Option<String>,
    pub config_path: Option<String>,
    pub mode: SessionMode,
    pub pending_plan: Option<PendingPlan>,
    pub events: Vec<AgentEvent>,
}

impl AgentSession {
    pub fn new(id: String) -> Self {
        let now = iso_now();
        Self {
            version: SESSION_VERSION,
            id,
            created_at: now.clone(),
            updated_at: now,
            cwd: std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            profile: None,
            config_path: None,
            mode: SessionMode::Chat,
            pending_plan: None,
            events: Vec::new(),
        }
    }

    pub fn record_event(&mut self, kind: &str, text: &str, data: serde_json::Value) {
        self.events.push(AgentEvent {
            kind: kind.to_owned(),
            text: text.to_owned(),
            data,
            created_at: iso_now(),
        });
        self.updated_at = iso_now();
    }
}

// ---------------------------------------------------------------------------
// Session store — JSON file persistence
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AgentSessionStore {
    root: PathBuf,
}

impl AgentSessionStore {
    pub fn new(project_root: PathBuf) -> Self {
        Self {
            root: project_root.join(".subbake/agent/sessions"),
        }
    }

    pub fn path_for(&self, id: &str) -> PathBuf {
        self.root.join(format!("{id}.json"))
    }

    pub fn create(&self) -> std::io::Result<AgentSession> {
        let id = format!("{}-{}", iso_now(), hex_id());
        Ok(AgentSession::new(id))
    }

    pub fn save(&self, session: &AgentSession) -> std::io::Result<()> {
        let path = self.path_for(&session.id);
        if session.events.is_empty() {
            match std::fs::remove_file(path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| std::io::Error::other(format!("create session dir: {e}")))?;
        }
        let json = serde_json::to_string_pretty(session)
            .map_err(|e| std::io::Error::other(format!("serialize session: {e}")))?;
        let tmp = path.with_file_name(format!("{}.tmp", session.id));
        std::fs::write(&tmp, &json)
            .map_err(|e| std::io::Error::other(format!("write session: {e}")))?;
        std::fs::rename(&tmp, &path)
            .map_err(|e| std::io::Error::other(format!("rename session: {e}")))?;
        Ok(())
    }

    pub fn load(&self, id: &str) -> std::io::Result<AgentSession> {
        let path = self.path_for(id);
        let json = std::fs::read_to_string(&path)
            .map_err(|e| std::io::Error::other(format!("read session {id}: {e}")))?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::other(format!("parse session {id}: {e}")))
    }

    pub fn latest(&self) -> std::io::Result<Option<AgentSession>> {
        Ok(self.list(1)?.into_iter().next())
    }

    pub fn list(&self, limit: usize) -> std::io::Result<Vec<AgentSession>> {
        if !self.root.is_dir() {
            return Ok(Vec::new());
        }
        let mut sessions = Vec::new();
        for entry in std::fs::read_dir(&self.root)
            .map_err(|e| std::io::Error::other(format!("list sessions: {e}")))?
        {
            let entry = entry.map_err(|error| {
                std::io::Error::other(format!("read session directory entry: {error}"))
            })?;
            if !entry.path().extension().is_some_and(|ext| ext == "json") {
                continue;
            }
            let path = entry.path();
            let id = path.file_stem().and_then(|s| s.to_str()).ok_or_else(|| {
                std::io::Error::other(format!(
                    "session filename is not valid UTF-8: {}",
                    path.display()
                ))
            })?;
            let session = self.load(id).map_err(|error| {
                std::io::Error::other(format!("load session `{}`: {error}", path.display()))
            })?;
            sessions.push(session);
        }
        sessions.sort_by(|a, b| {
            b.updated_at
                .cmp(&a.updated_at)
                .then_with(|| b.id.cmp(&a.id))
        });
        sessions.truncate(limit);
        Ok(sessions)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(crate) fn iso_now() -> String {
    // Rough ISO-8601 UTC timestamp without pulling in chrono.
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after epoch");
    let secs = d.as_secs();
    // Compute date components using a simple days-since-epoch calculation.
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let h = time_secs / 3600;
    let m = (time_secs % 3600) / 60;
    let s = time_secs % 60;
    // Approximate Gregorian date (valid 1970-2100).
    let (y, mo, d) = civil_from_days(days as i64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    // Rata Die algorithm, from Howard Hinnant.
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    (y, mo, d)
}

fn hex_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    format!("{:016x}", nanos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_session_is_not_persisted() {
        let dir = std::env::temp_dir().join(format!("subbake-agent-sessions-{}", hex_id()));
        let store = AgentSessionStore::new(dir.clone());
        let session = store.create().expect("create session");
        assert_eq!(session.version, SESSION_VERSION);
        assert!(!session.id.is_empty());
        assert_eq!(session.mode, SessionMode::Chat);
        assert!(session.events.is_empty());

        assert!(!store.path_for(&session.id).exists());
        assert!(store.list(20).expect("list sessions").is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_mode_keeps_the_v1_string_wire_shape() {
        assert_eq!(
            serde_json::to_value(SessionMode::Plan).expect("serialize mode"),
            serde_json::json!("plan")
        );
        assert_eq!(
            serde_json::from_value::<SessionMode>(serde_json::json!("chat")).expect("read v1 mode"),
            SessionMode::Chat
        );
    }

    #[test]
    fn records_and_persists_events() {
        let dir = std::env::temp_dir().join(format!("subbake-agent-events-{}", hex_id()));
        let store = AgentSessionStore::new(dir.clone());
        let mut session = store.create().expect("create session");
        session.record_event(
            "user",
            "translate hello",
            serde_json::json!({"path": "hello.srt"}),
        );
        store.save(&session).expect("save with events");

        let loaded = store.load(&session.id).expect("load session");
        assert_eq!(loaded.events.len(), 1);
        assert_eq!(loaded.events[0].kind, "user");
        assert_eq!(loaded.events[0].data["path"], "hello.srt");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sessions_are_ordered_by_latest_activity() {
        let dir = std::env::temp_dir().join(format!("subbake-agent-latest-{}", hex_id()));
        let store = AgentSessionStore::new(dir.clone());
        let mut s1 = store.create().expect("session 1");
        s1.record_event("user", "first", serde_json::json!({}));
        s1.created_at = "2026-07-11T01:00:00Z".to_owned();
        s1.updated_at = "2026-07-11T03:00:00Z".to_owned();
        store.save(&s1).expect("save session 1");
        let mut s2 = store.create().expect("session 2");
        s2.record_event("user", "second", serde_json::json!({}));
        s2.created_at = "2026-07-11T02:00:00Z".to_owned();
        s2.updated_at = "2026-07-11T02:00:00Z".to_owned();
        store.save(&s2).expect("save session 2");

        let sessions = store.list(20).expect("list sessions");
        let latest = store.latest().expect("latest").expect("some session");

        assert_eq!(sessions[0].id, s1.id);
        assert_eq!(sessions[1].id, s2.id);
        assert_eq!(latest.id, s1.id);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_reports_corrupt_sessions_instead_of_hiding_them() {
        let dir = std::env::temp_dir().join(format!("subbake-agent-corrupt-{}", hex_id()));
        let store = AgentSessionStore::new(dir.clone());
        let session_dir = dir.join(".subbake/agent/sessions");
        std::fs::create_dir_all(&session_dir).expect("create session directory");
        std::fs::write(session_dir.join("broken.json"), "{not json")
            .expect("write corrupt session");

        let error = store
            .list(20)
            .expect_err("corrupt session must be reported");
        assert!(error.to_string().contains("broken.json"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_order_is_stable_when_update_times_match() {
        let dir = std::env::temp_dir().join(format!("subbake-agent-tie-{}", hex_id()));
        let store = AgentSessionStore::new(dir.clone());
        let mut first = AgentSession::new("session-a".to_owned());
        first.record_event("user", "first", serde_json::json!({}));
        first.updated_at = "2026-07-11T03:00:00Z".to_owned();
        store.save(&first).expect("save first session");
        let mut second = AgentSession::new("session-b".to_owned());
        second.record_event("user", "second", serde_json::json!({}));
        second.updated_at = first.updated_at.clone();
        store.save(&second).expect("save second session");

        let sessions = store.list(20).expect("list sessions");
        assert_eq!(sessions[0].id, "session-b");
        assert_eq!(sessions[1].id, "session-a");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn iso_now_format() {
        let s = iso_now();
        assert_eq!(s.len(), 20); // "2026-07-08T19:39:00Z"
        assert_eq!(&s[10..11], "T");
        assert_eq!(&s[19..20], "Z");
    }
}
