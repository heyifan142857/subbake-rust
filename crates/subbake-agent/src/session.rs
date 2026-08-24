// Agent session — event log (the source of truth for undo, replay, and resume).
//
// Version 2 of the persisted agent-session JSON contract. The session JSON
// lives at `<project_root>/.subbake/agent/sessions/<id>.json`.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AgentError, AgentResult};
use crate::event::{EventKind, PendingAgentTurn, PendingCommandApproval, PendingPlan};

pub const SESSION_VERSION: u64 = 2;

/// Stable discriminants for the v2 wire-format event kinds. `Unknown` keeps
/// future events readable without allowing ad-hoc comparisons in runtime logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventTag {
    User,
    Assistant,
    AskUser,
    ToolStarted,
    ToolCompleted,
    ToolFailed,
    ToolCancelled,
    FinalToolCall,
    FileOperation,
    Plan,
    Approve,
    Reject,
    Undo,
    Profile,
    Error,
    Cancelled,
    CommandApprovalRequested,
    CommandApproved,
    CommandRejected,
    Unknown,
}

impl EventTag {
    pub fn parse(value: &str) -> Self {
        match value {
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "ask_user" => Self::AskUser,
            "tool_started" => Self::ToolStarted,
            "tool_completed" => Self::ToolCompleted,
            "tool_failed" => Self::ToolFailed,
            "tool_cancelled" => Self::ToolCancelled,
            "final_tool_call" => Self::FinalToolCall,
            "file_operation" => Self::FileOperation,
            "plan" => Self::Plan,
            "approve" => Self::Approve,
            "reject" => Self::Reject,
            "undo" => Self::Undo,
            "profile" => Self::Profile,
            "error" => Self::Error,
            "cancelled" => Self::Cancelled,
            "command_approval_requested" => Self::CommandApprovalRequested,
            "command_approved" => Self::CommandApproved,
            "command_rejected" => Self::CommandRejected,
            _ => Self::Unknown,
        }
    }
}

/// The persisted session mode. Serde keeps the compact JSON representation as a
/// lowercase string while preventing invalid in-memory modes.
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

    pub(crate) fn from_kind(event: &EventKind) -> Self {
        let (kind, text, data) = match event {
            EventKind::User { text } => ("user", text.clone(), serde_json::json!({})),
            EventKind::Assistant { text } => ("assistant", text.clone(), serde_json::json!({})),
            EventKind::AskUser { text } => ("ask_user", text.clone(), serde_json::json!({})),
            EventKind::ToolStarted {
                call_id,
                tool_name,
                headline,
                detail,
            } => (
                "tool_started",
                tool_name.clone(),
                serde_json::json!({
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "headline": headline,
                    "detail": detail,
                }),
            ),
            EventKind::ToolCompleted {
                call_id,
                tool_name,
                headline,
                detail,
                duration_ms,
            } => (
                "tool_completed",
                tool_name.clone(),
                serde_json::json!({
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "headline": headline,
                    "detail": detail,
                    "duration_ms": duration_ms,
                }),
            ),
            EventKind::ToolFailed {
                call_id,
                tool_name,
                headline,
                detail,
                category,
                error,
                duration_ms,
            } => (
                "tool_failed",
                format!("{tool_name}: {error}"),
                serde_json::json!({
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "headline": headline,
                    "detail": detail,
                    "category": category,
                    "duration_ms": duration_ms,
                }),
            ),
            EventKind::ToolCancelled {
                call_id,
                tool_name,
                headline,
                detail,
                duration_ms,
            } => (
                "tool_cancelled",
                tool_name.clone(),
                serde_json::json!({
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "headline": headline,
                    "detail": detail,
                    "duration_ms": duration_ms,
                }),
            ),
            EventKind::FinalToolCall {
                tool_name,
                arguments,
            } => (
                "final_tool_call",
                tool_name.clone(),
                serde_json::json!({"tool_name": tool_name, "arguments": arguments}),
            ),
            EventKind::FileOperation(data) => (
                "file_operation",
                format!("{} {}", data.action, data.path),
                serde_json::to_value(data).unwrap_or_default(),
            ),
            EventKind::Plan {
                message,
                tool_calls,
            } => (
                "plan",
                message.clone(),
                serde_json::json!({"message": message, "tool_calls": tool_calls}),
            ),
            EventKind::CommandApprovalRequested { tool_call, reason } => (
                "command_approval_requested",
                reason.clone(),
                serde_json::json!({"tool_call": tool_call, "reason": reason}),
            ),
            EventKind::CommandApproved => {
                ("command_approved", String::new(), serde_json::json!({}))
            }
            EventKind::CommandRejected => {
                ("command_rejected", String::new(), serde_json::json!({}))
            }
            EventKind::Approve => ("approve", String::new(), serde_json::json!({})),
            EventKind::Reject => ("reject", String::new(), serde_json::json!({})),
            EventKind::Undo => ("undo", String::new(), serde_json::json!({})),
            EventKind::Profile { name } => ("profile", name.clone(), serde_json::json!({})),
            EventKind::Error { text } => ("error", text.clone(), serde_json::json!({})),
            EventKind::Cancelled => ("cancelled", "Cancelled.".to_owned(), serde_json::json!({})),
        };
        Self {
            kind: kind.to_owned(),
            text,
            data,
            created_at: iso_now(),
        }
    }

    /// Recover the typed runtime event while keeping unknown future v2 event
    /// kinds readable through the persisted `AgentEvent` representation.
    pub fn typed(&self) -> Option<EventKind> {
        let string = |key: &str| self.data.get(key)?.as_str().map(str::to_owned);
        match self.tag() {
            EventTag::User => Some(EventKind::User {
                text: self.text.clone(),
            }),
            EventTag::Assistant => Some(EventKind::Assistant {
                text: self.text.clone(),
            }),
            EventTag::AskUser => Some(EventKind::AskUser {
                text: self.text.clone(),
            }),
            EventTag::ToolStarted => Some(EventKind::ToolStarted {
                call_id: string("call_id")?,
                tool_name: string("tool_name")?,
                headline: string("headline")?,
                detail: self
                    .data
                    .get("detail")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned),
            }),
            EventTag::ToolCompleted => Some(EventKind::ToolCompleted {
                call_id: string("call_id")?,
                tool_name: string("tool_name")?,
                headline: string("headline")?,
                detail: self
                    .data
                    .get("detail")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned),
                duration_ms: self.data.get("duration_ms")?.as_u64()?,
            }),
            EventTag::FinalToolCall => Some(EventKind::FinalToolCall {
                tool_name: string("tool_name")?,
                arguments: self.data.get("arguments")?.clone(),
            }),
            EventTag::ToolFailed => {
                let tool_name = string("tool_name")?;
                let prefix = format!("{tool_name}: ");
                Some(EventKind::ToolFailed {
                    call_id: string("call_id")?,
                    tool_name,
                    headline: string("headline")?,
                    detail: self
                        .data
                        .get("detail")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned),
                    category: string("category")?,
                    error: self.text.strip_prefix(&prefix)?.to_owned(),
                    duration_ms: self.data.get("duration_ms")?.as_u64()?,
                })
            }
            EventTag::ToolCancelled => Some(EventKind::ToolCancelled {
                call_id: string("call_id")?,
                tool_name: string("tool_name")?,
                headline: string("headline")?,
                detail: self
                    .data
                    .get("detail")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned),
                duration_ms: self.data.get("duration_ms")?.as_u64()?,
            }),
            EventTag::FileOperation => serde_json::from_value(self.data.clone())
                .ok()
                .map(EventKind::FileOperation),
            EventTag::Plan => Some(EventKind::Plan {
                message: string("message").unwrap_or_else(|| self.text.clone()),
                tool_calls: serde_json::from_value(self.data.get("tool_calls")?.clone()).ok()?,
            }),
            EventTag::CommandApprovalRequested => Some(EventKind::CommandApprovalRequested {
                tool_call: serde_json::from_value(self.data.get("tool_call")?.clone()).ok()?,
                reason: string("reason").unwrap_or_else(|| self.text.clone()),
            }),
            EventTag::CommandApproved => Some(EventKind::CommandApproved),
            EventTag::CommandRejected => Some(EventKind::CommandRejected),
            EventTag::Approve => Some(EventKind::Approve),
            EventTag::Reject => Some(EventKind::Reject),
            EventTag::Undo => Some(EventKind::Undo),
            EventTag::Profile => Some(EventKind::Profile {
                name: self.text.clone(),
            }),
            EventTag::Error => Some(EventKind::Error {
                text: self.text.clone(),
            }),
            EventTag::Cancelled => Some(EventKind::Cancelled),
            EventTag::Unknown => None,
        }
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
    #[serde(default)]
    pub pending_command_approval: Option<PendingCommandApproval>,
    #[serde(default)]
    pub pending_agent_turn: Option<PendingAgentTurn>,
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
            pending_command_approval: None,
            pending_agent_turn: None,
            events: Vec::new(),
        }
    }

    #[cfg(test)]
    fn record_event(&mut self, kind: &str, text: &str, data: serde_json::Value) {
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
    project_root: PathBuf,
    root: PathBuf,
}

impl AgentSessionStore {
    pub fn new(project_root: PathBuf) -> Self {
        // Preserve the caller-owned logical spelling. Canonicalizing here
        // would leak macOS aliases and Windows verbatim paths into session
        // locations after FileGuard has already separated logical identity
        // from filesystem identity at the engine boundary.
        Self {
            root: project_root.join(".subbake/agent/sessions"),
            project_root,
        }
    }

    pub fn path_for(&self, id: &str) -> AgentResult<PathBuf> {
        validate_session_id(id)?;
        Ok(self.root.join(format!("{id}.json")))
    }

    pub fn create(&self) -> AgentResult<AgentSession> {
        let id = format!("{}-{}", iso_now().replace(':', "-"), hex_id());
        Ok(AgentSession::new(id))
    }

    pub fn save(&self, session: &AgentSession) -> AgentResult<()> {
        let path = self.path_for(&session.id)?;
        if session.version != SESSION_VERSION {
            return Err(AgentError::invalid_state(format!(
                "cannot save session version {}; expected version {SESSION_VERSION}",
                session.version
            )));
        }
        if session.events.is_empty() {
            self.ensure_storage_root(false)?;
            reject_symlink_or_non_file(&path)?;
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(source) => {
                    return Err(AgentError::SessionStorage {
                        operation: "remove empty session",
                        path: Some(path),
                        source,
                    });
                }
            }
            return Ok(());
        }
        self.ensure_storage_root(true)?;
        reject_symlink_or_non_file(&path)?;
        let json =
            serde_json::to_string_pretty(session).map_err(|source| AgentError::SessionData {
                operation: "serialize session",
                path: path.clone(),
                source,
            })?;
        subbake_adapters::write_file_atomically(&path, json.as_bytes())?;
        Ok(())
    }

    pub fn load(&self, id: &str) -> AgentResult<AgentSession> {
        let path = self.path_for(id)?;
        self.ensure_storage_root(false)?;
        reject_symlink_or_non_file(&path)?;
        let json = std::fs::read_to_string(&path).map_err(|source| AgentError::SessionStorage {
            operation: "read session",
            path: Some(path.clone()),
            source,
        })?;
        let session: AgentSession =
            serde_json::from_str(&json).map_err(|source| AgentError::SessionData {
                operation: "parse session",
                path: path.clone(),
                source,
            })?;
        if session.version != SESSION_VERSION {
            return Err(AgentError::invalid_state(format!(
                "session `{id}` uses unsupported version {}; expected version {SESSION_VERSION}",
                session.version
            )));
        }
        if session.id != id {
            return Err(AgentError::invalid_state(format!(
                "session file `{}` contains mismatched id `{}`",
                path.display(),
                session.id
            )));
        }
        Ok(session)
    }

    pub fn latest(&self) -> AgentResult<Option<AgentSession>> {
        Ok(self.list(1)?.into_iter().next())
    }

    pub fn list(&self, limit: usize) -> AgentResult<Vec<AgentSession>> {
        if !self.ensure_storage_root(false)? {
            return Ok(Vec::new());
        }
        let mut sessions = Vec::new();
        for entry in std::fs::read_dir(&self.root).map_err(|source| AgentError::SessionStorage {
            operation: "list sessions",
            path: Some(self.root.clone()),
            source,
        })? {
            let entry = entry.map_err(|source| AgentError::SessionStorage {
                operation: "read session directory entry",
                path: Some(self.root.clone()),
                source,
            })?;
            let file_type = entry
                .file_type()
                .map_err(|source| AgentError::SessionStorage {
                    operation: "inspect session directory entry",
                    path: Some(entry.path()),
                    source,
                })?;
            if file_type.is_symlink() {
                return Err(AgentError::invalid_state(format!(
                    "refusing symlinked session file `{}`",
                    entry.path().display()
                )));
            }
            if !file_type.is_file() {
                continue;
            }
            if !entry.path().extension().is_some_and(|ext| ext == "json") {
                continue;
            }
            let path = entry.path();
            let id = path.file_stem().and_then(|s| s.to_str()).ok_or_else(|| {
                AgentError::InvalidState {
                    message: format!("session filename is not valid UTF-8: {}", path.display()),
                }
            })?;
            let session = self.load(id)?;
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

    fn ensure_storage_root(&self, create: bool) -> AgentResult<bool> {
        if create && !self.project_root.exists() {
            std::fs::create_dir_all(&self.project_root).map_err(|source| {
                AgentError::SessionStorage {
                    operation: "create project directory",
                    path: Some(self.project_root.clone()),
                    source,
                }
            })?;
        }
        if !self.project_root.is_dir() {
            return if create {
                Err(AgentError::invalid_state(format!(
                    "session project root is not a directory: {}",
                    self.project_root.display()
                )))
            } else {
                Ok(false)
            };
        }

        let mut current = self.project_root.clone();
        for component in [".subbake", "agent", "sessions"] {
            current.push(component);
            match std::fs::symlink_metadata(&current) {
                Ok(metadata) if metadata.file_type().is_symlink() => {
                    return Err(AgentError::invalid_state(format!(
                        "refusing symlinked session directory `{}`",
                        current.display()
                    )));
                }
                Ok(metadata) if !metadata.is_dir() => {
                    return Err(AgentError::invalid_state(format!(
                        "session storage component is not a directory: {}",
                        current.display()
                    )));
                }
                Ok(_) => {}
                Err(source) if source.kind() == std::io::ErrorKind::NotFound && create => {
                    std::fs::create_dir(&current).map_err(|source| AgentError::SessionStorage {
                        operation: "create session directory",
                        path: Some(current.clone()),
                        source,
                    })?;
                }
                Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(false),
                Err(source) => {
                    return Err(AgentError::SessionStorage {
                        operation: "inspect session directory",
                        path: Some(current),
                        source,
                    });
                }
            }
        }
        Ok(true)
    }
}

fn validate_session_id(id: &str) -> AgentResult<()> {
    let is_single_component = matches!(
        Path::new(id).components().collect::<Vec<_>>().as_slice(),
        [Component::Normal(component)] if *component == id
    );
    if id.is_empty()
        || id.len() > 160
        || !is_single_component
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b':' | b'_' | b'-'))
    {
        return Err(AgentError::invalid_input(
            "session id must contain only ASCII letters, digits, ':', '_', or '-'",
        ));
    }
    Ok(())
}

fn reject_symlink_or_non_file(path: &Path) -> AgentResult<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(AgentError::invalid_state(
            format!("refusing symlinked session file `{}`", path.display()),
        )),
        Ok(metadata) if !metadata.is_file() => Err(AgentError::invalid_state(format!(
            "session path is not a regular file: {}",
            path.display()
        ))),
        Ok(_) => Ok(()),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(AgentError::SessionStorage {
            operation: "inspect session file",
            path: Some(path.to_path_buf()),
            source,
        }),
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

        assert!(!store.path_for(&session.id).expect("valid id").exists());
        assert!(store.list(20).expect("list sessions").is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_mode_keeps_the_v2_string_wire_shape() {
        assert_eq!(
            serde_json::to_value(SessionMode::Plan).expect("serialize mode"),
            serde_json::json!("plan")
        );
        assert_eq!(
            serde_json::from_value::<SessionMode>(serde_json::json!("chat")).expect("read v2 mode"),
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

    #[cfg(unix)]
    #[test]
    fn session_paths_preserve_an_aliased_project_root_spelling() {
        use std::os::unix::fs::symlink;

        let container =
            std::env::temp_dir().join(format!("subbake-agent-session-alias-{}", hex_id()));
        let actual = container.join("actual");
        let alias = container.join("alias");
        std::fs::create_dir_all(&actual).expect("create actual project root");
        symlink(&actual, &alias).expect("create project-root alias");

        let store = AgentSessionStore::new(alias.clone());
        let mut session = store.create().expect("create session");
        session.record_event("user", "hello", serde_json::json!({}));
        store
            .save(&session)
            .expect("save through project-root alias");

        let logical_path = store.path_for(&session.id).expect("session path");
        assert_eq!(
            logical_path,
            alias
                .join(".subbake/agent/sessions")
                .join(format!("{}.json", session.id))
        );
        assert!(
            actual
                .join(".subbake/agent/sessions")
                .join(format!("{}.json", session.id))
                .is_file()
        );

        let _ = std::fs::remove_dir_all(container);
    }

    #[test]
    fn rejects_session_ids_that_are_not_plain_safe_components() {
        let dir = std::env::temp_dir().join(format!("subbake-agent-id-{}", hex_id()));
        let store = AgentSessionStore::new(dir.clone());

        for id in ["../escape", "/tmp/escape", "nested/session", "session.json"] {
            assert!(store.path_for(id).is_err(), "unexpectedly accepted {id}");
            assert!(store.load(id).is_err(), "unexpectedly loaded {id}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_a_session_whose_embedded_id_does_not_match_its_filename() {
        let dir = std::env::temp_dir().join(format!("subbake-agent-id-mismatch-{}", hex_id()));
        let store = AgentSessionStore::new(dir.clone());
        let mut session = AgentSession::new("different-id".to_owned());
        session.record_event("user", "hello", serde_json::json!({}));
        let path = store.path_for("requested-id").expect("valid id");
        std::fs::create_dir_all(path.parent().expect("session directory"))
            .expect("create session directory");
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&session).expect("serialize session"),
        )
        .expect("write mismatched session");

        let error = store
            .load("requested-id")
            .expect_err("mismatched embedded id must fail");
        assert!(error.to_string().contains("mismatched id `different-id`"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_session_files() {
        use std::os::unix::fs::symlink;

        let dir = std::env::temp_dir().join(format!("subbake-agent-symlink-{}", hex_id()));
        let store = AgentSessionStore::new(dir.clone());
        let path = store.path_for("linked").expect("valid id");
        std::fs::create_dir_all(path.parent().expect("session directory"))
            .expect("create session directory");
        let target = dir.join("outside.json");
        std::fs::write(&target, "{}").expect("write target");
        symlink(&target, &path).expect("create symlink");

        let error = store.load("linked").expect_err("session symlink must fail");
        assert!(
            error
                .to_string()
                .contains("refusing symlinked session file")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_session_directories() {
        use std::os::unix::fs::symlink;

        let dir = std::env::temp_dir().join(format!("subbake-agent-dir-symlink-{}", hex_id()));
        let outside = std::env::temp_dir().join(format!("subbake-agent-outside-{}", hex_id()));
        std::fs::create_dir_all(&dir).expect("create project root");
        std::fs::create_dir_all(&outside).expect("create outside directory");
        symlink(&outside, dir.join(".subbake")).expect("create storage symlink");
        let store = AgentSessionStore::new(dir.clone());
        let mut session = store.create().expect("create session");
        session.record_event("user", "hello", serde_json::json!({}));

        let error = store
            .save(&session)
            .expect_err("session directory symlink must fail");
        assert!(
            error
                .to_string()
                .contains("refusing symlinked session directory")
        );
        assert!(
            std::fs::read_dir(&outside)
                .expect("read outside")
                .next()
                .is_none()
        );

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn rejects_an_older_session_version() {
        let dir = std::env::temp_dir().join(format!("subbake-agent-old-version-{}", hex_id()));
        let store = AgentSessionStore::new(dir.clone());
        let mut session = store.create().expect("create session");
        session.record_event("user", "old", serde_json::json!({}));
        session.version = 1;
        let path = store.path_for(&session.id).expect("valid id");
        std::fs::create_dir_all(path.parent().expect("session directory"))
            .expect("create session directory");
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&session).expect("serialize old session"),
        )
        .expect("write old session fixture");

        let error = store
            .load(&session.id)
            .expect_err("old session versions must be rejected");
        assert!(error.to_string().contains("unsupported version 1"));
        assert!(error.to_string().contains("expected version 2"));

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

    #[test]
    fn typed_tool_lifecycle_round_trips_through_the_v2_wire_shape() {
        let started = EventKind::ToolStarted {
            call_id: "call-1".to_owned(),
            tool_name: "read_file".to_owned(),
            headline: "Reading sample.srt".to_owned(),
            detail: None,
        };
        let persisted = AgentEvent::from_kind(&started);
        assert_eq!(persisted.kind, "tool_started");
        assert_eq!(persisted.typed(), Some(started));

        let completed = EventKind::ToolCompleted {
            call_id: "call-1".to_owned(),
            tool_name: "read_file".to_owned(),
            headline: "Read sample.srt".to_owned(),
            detail: Some("1 line · 0.0s".to_owned()),
            duration_ms: 42,
        };
        let persisted = AgentEvent::from_kind(&completed);
        assert_eq!(persisted.kind, "tool_completed");
        assert_eq!(persisted.typed(), Some(completed));

        let future = AgentEvent {
            kind: "future_event".to_owned(),
            text: String::new(),
            data: serde_json::json!({}),
            created_at: iso_now(),
        };
        assert_eq!(future.tag(), EventTag::Unknown);
        assert_eq!(future.typed(), None);
    }
}
