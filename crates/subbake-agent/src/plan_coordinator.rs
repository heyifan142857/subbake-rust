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
            in_flight_call_id: None,
        });
    }

    pub(crate) fn next_call(session: &AgentSession) -> AgentResult<Option<ToolCallDraft>> {
        if session
            .pending_plan
            .as_ref()
            .is_some_and(|plan| plan.in_flight_call_id.is_some())
        {
            return Err(crate::AgentError::invalid_state(
                "the previous plan operation was interrupted or failed; its outcome may be partial. Inspect the files or use /undo, then revise or reject the plan before continuing",
            ));
        }
        session
            .pending_plan
            .as_ref()
            .ok_or_else(|| std::io::Error::other("no pending plan to approve").into())
            .map(|plan| plan.tool_calls.first().cloned())
    }

    pub(crate) fn begin_call(
        store: &AgentSessionStore,
        session: &mut AgentSession,
        call_id: String,
    ) -> AgentResult<()> {
        let mut candidate = session.clone();
        let plan = candidate
            .pending_plan
            .as_mut()
            .ok_or_else(|| std::io::Error::other("pending plan disappeared"))?;
        if plan.tool_calls.is_empty() {
            return Err(std::io::Error::other("pending plan has no next call").into());
        }
        plan.in_flight_call_id = Some(call_id);
        store.save(&candidate)?;
        *session = candidate;
        Ok(())
    }

    pub(crate) fn complete_call(session: &mut AgentSession, call_id: &str) {
        if let Some(plan) = session.pending_plan.as_mut()
            && plan.in_flight_call_id.as_deref() == Some(call_id)
        {
            if !plan.tool_calls.is_empty() {
                plan.tool_calls.remove(0);
            }
            plan.in_flight_call_id = None;
        }
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
