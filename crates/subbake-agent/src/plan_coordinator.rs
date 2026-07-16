use crate::error::AgentResult;
use crate::event::{PendingPlan, ToolCallDraft};
use crate::session::{AgentSession, AgentSessionStore, SessionMode, iso_now};

pub(crate) struct PlanCoordinator;

impl PlanCoordinator {
    pub(crate) fn store(session: &mut AgentSession, message: &str, tool_calls: Vec<ToolCallDraft>) {
        session.mode = SessionMode::Plan;
        session.pending_plan = Some(PendingPlan {
            message: message.to_owned(),
            tool_calls,
            created_at: iso_now(),
        });
    }

    pub(crate) fn next_call(session: &AgentSession) -> AgentResult<Option<ToolCallDraft>> {
        session
            .pending_plan
            .as_ref()
            .ok_or_else(|| std::io::Error::other("no pending plan to approve").into())
            .map(|plan| plan.tool_calls.first().cloned())
    }

    pub(crate) fn commit_completed_call(
        store: &AgentSessionStore,
        session: &mut AgentSession,
    ) -> AgentResult<()> {
        let plan = session
            .pending_plan
            .as_mut()
            .ok_or_else(|| std::io::Error::other("pending plan disappeared"))?;
        if plan.tool_calls.is_empty() {
            return Err(std::io::Error::other("pending plan has no completed call").into());
        }
        plan.tool_calls.remove(0);
        store.save(session)
    }

    pub(crate) fn finish(session: &mut AgentSession) {
        session.mode = SessionMode::Chat;
        session.pending_plan = None;
    }

    pub(crate) fn set_mode(
        store: &AgentSessionStore,
        session: &mut AgentSession,
        enabled: bool,
    ) -> AgentResult<()> {
        session.mode = if enabled {
            SessionMode::Plan
        } else {
            SessionMode::Chat
        };
        if !enabled {
            session.pending_plan = None;
        }
        store.save(session)
    }
}
