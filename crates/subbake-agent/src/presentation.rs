use crate::error::AgentResult;
use crate::event::PendingPlan;
use crate::session::{AgentSession, AgentSessionStore, EventTag};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionChoice {
    pub id: String,
    pub title: String,
    pub updated_at: String,
    pub cwd: String,
    pub event_count: usize,
    pub active: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileChoice {
    pub name: String,
    pub provider: String,
    pub model: String,
    pub active: bool,
    pub create: bool,
}

pub(crate) struct ConversationPresenter;

impl ConversationPresenter {
    pub(crate) fn session_summary(session: &AgentSession) -> String {
        let pending = if session.pending_plan.is_some() {
            "pending plan"
        } else {
            "no pending plan"
        };
        format!(
            "Session: {}\nMode: {}\nEvents: {}\n{}",
            session.id,
            session.mode,
            session.events.len(),
            pending
        )
    }

    pub(crate) fn sessions_summary(
        store: &AgentSessionStore,
        active: Option<&AgentSession>,
        limit: usize,
    ) -> AgentResult<String> {
        let sessions = store.list(limit)?;
        if sessions.is_empty() {
            return Ok("No saved sessions.".to_owned());
        }
        let active_id = active.map(|session| session.id.as_str());
        Ok(sessions
            .iter()
            .map(|session| {
                let marker = if Some(session.id.as_str()) == active_id {
                    "*"
                } else {
                    " "
                };
                format!(
                    "{marker} {}  {}  {} events",
                    session.id,
                    session.mode,
                    session.events.len()
                )
            })
            .collect::<Vec<_>>()
            .join("\n"))
    }

    pub(crate) fn session_choices(
        store: &AgentSessionStore,
        active: Option<&AgentSession>,
        limit: usize,
    ) -> AgentResult<Vec<SessionChoice>> {
        let active_id = active.map(|session| session.id.as_str());
        store.list(limit).map(|sessions| {
            sessions
                .into_iter()
                .map(|session| {
                    let title = session
                        .events
                        .iter()
                        .find(|event| {
                            event.tag() == EventTag::User && !event.text.trim().is_empty()
                        })
                        .map(|event| truncate(&event.text, 48))
                        .unwrap_or_else(|| "New session".to_owned());
                    let cwd = std::path::Path::new(&session.cwd)
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or(&session.cwd)
                        .to_owned();
                    SessionChoice {
                        active: Some(session.id.as_str()) == active_id,
                        id: session.id,
                        title,
                        updated_at: session.updated_at,
                        cwd,
                        event_count: session.events.len(),
                    }
                })
                .collect()
        })
    }

    pub(crate) fn history_summary(session: &AgentSession, limit: usize) -> String {
        let start = session.events.len().saturating_sub(limit);
        let lines = session.events[start..]
            .iter()
            .filter_map(|event| {
                let label = match event.tag() {
                    EventTag::User => "You",
                    EventTag::Assistant | EventTag::AskUser => "Agent",
                    EventTag::ToolCall => "Tool",
                    EventTag::Error => "Error",
                    _ => return None,
                };
                Some(format!("{label}: {}", event.text))
            })
            .collect::<Vec<_>>();
        if lines.is_empty() {
            "No conversation history.".to_owned()
        } else {
            lines.join("\n")
        }
    }

    pub(crate) fn pending_plan_summary(plan: Option<&PendingPlan>) -> String {
        let Some(plan) = plan else {
            return "No pending plan.".to_owned();
        };
        let mut lines = vec!["Plan awaiting approval:".to_owned()];
        if !plan.message.trim().is_empty() {
            lines.push(plan.message.trim().to_owned());
        }
        lines.extend(
            plan.tool_calls.iter().enumerate().map(|(index, call)| {
                format!("{}. {} {}", index + 1, call.tool_name, call.arguments)
            }),
        );
        lines.push("Choose an action below: approve, reject, or revise the plan.".to_owned());
        lines.join("\n")
    }

    pub(crate) fn conversation_context(session: &AgentSession, limit: usize) -> Option<String> {
        let mut lines = session
            .events
            .iter()
            .rev()
            .skip_while(|event| event.tag() == EventTag::User)
            .filter_map(|event| {
                let label = match event.tag() {
                    EventTag::User => "User",
                    EventTag::Assistant => "Assistant",
                    EventTag::AskUser => "Assistant question",
                    EventTag::ToolCall => "Tool",
                    EventTag::FileOperation => "File operation",
                    EventTag::Plan => "Plan",
                    EventTag::Error => "Error",
                    _ => return None,
                };
                let text = if event.text.trim().is_empty() {
                    event.data.to_string()
                } else {
                    event.text.clone()
                };
                Some(format!("{label}: {}", truncate(&text, 240)))
            })
            .take(limit)
            .collect::<Vec<_>>();
        lines.reverse();
        (!lines.is_empty()).then(|| lines.join("\n"))
    }

    pub(crate) fn input_history(session: Option<&AgentSession>) -> Vec<String> {
        session
            .map(|session| {
                session
                    .events
                    .iter()
                    .filter(|event| event.tag() == EventTag::User && !event.text.trim().is_empty())
                    .map(|event| event.text.clone())
                    .fold(Vec::<String>::new(), |mut history, input| {
                        if history.last() != Some(&input) {
                            history.push(input);
                        }
                        history
                    })
            })
            .unwrap_or_default()
    }
}

fn truncate(text: &str, max_chars: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= max_chars {
        normalized
    } else {
        format!(
            "{}…",
            normalized
                .chars()
                .take(max_chars.saturating_sub(1))
                .collect::<String>()
        )
    }
}
