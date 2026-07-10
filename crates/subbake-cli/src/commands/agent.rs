use std::io;
use std::path::Path;

use subbake_adapters::{
    TranslationSettings, build_backend, discover_config_path, load_and_resolve,
};
use subbake_agent::event::EventKind;
use subbake_agent::{
    AgentEngine, AgentRequest, EchoDecisionBackend, PlanDecision, RenderPolicy, SubBakeTui,
    TuiAction, TuiInteraction,
};

use crate::args::AgentArgs;
use crate::output::print_agent_outcome;

pub fn run(args: AgentArgs) -> io::Result<()> {
    // No-args / resume → start TUI.
    match args.action.kind.as_str() {
        "start" | "" => start_interactive(),
        "resume" => {
            if let Some(sid) = &args.action.session_id {
                start_interactive_resume(Some(sid))
            } else {
                start_interactive_resume(None)
            }
        }
        _other => {
            // Legacy stub path (backwards compat).
            let outcome = subbake_agent::run_agent(AgentRequest {
                action: args.action,
            });
            print_agent_outcome(&outcome);
            Ok(())
        }
    }
}

fn start_interactive() -> io::Result<()> {
    let project_root = std::env::current_dir()?;
    let mut engine = AgentEngine::new(project_root);
    engine.start_session()?;

    run_tui_with_engine(engine)
}

fn start_interactive_resume(session_id: Option<&str>) -> io::Result<()> {
    let project_root = std::env::current_dir()?;
    let mut engine = AgentEngine::new(project_root);
    engine.resume_session(session_id)?;

    run_tui_with_engine(engine)
}

fn run_tui_with_engine(mut engine: AgentEngine) -> io::Result<()> {
    let config_path = discover_config_path();
    engine.set_config_path(config_path.as_deref())?;
    let initial_profile = engine
        .session
        .as_ref()
        .and_then(|session| session.profile.clone());
    let mut backend =
        build_agent_decision_backend(config_path.as_deref(), initial_profile.as_deref())?;

    // Create the TUI with an observer attached to the engine.
    let input_history = engine.input_history();
    let session_events = engine.session_events();
    let mut tui = SubBakeTui::new()?;
    tui.set_input_history(input_history);
    tui.set_session_replay(session_events);
    let observer = tui.observer();
    engine = engine.with_observer(Box::new(observer));

    tui.run(move |action, _obs| {
        let submitted_text = match &action {
            TuiAction::SubmitText(text) => Some(text.as_str()),
            _ => None,
        };
        let requested_profile = match &action {
            TuiAction::SelectProfile(name) => Some(name.as_str()),
            TuiAction::SubmitText(input) => input
                .trim()
                .strip_prefix("/profile ")
                .map(str::trim)
                .filter(|name| !name.is_empty()),
            _ => None,
        };
        let requested_session = match &action {
            TuiAction::SelectSession(id) => Some(id.as_str()),
            TuiAction::SubmitText(input) => input
                .trim()
                .strip_prefix("/session ")
                .map(str::trim)
                .filter(|id| !id.is_empty()),
            _ => None,
        };
        let candidate_backend = if let Some(profile) = requested_profile {
            if engine.profile_choices()?.iter().any(|name| name == profile) {
                Some(build_agent_decision_backend(
                    config_path.as_deref(),
                    Some(profile),
                )?)
            } else {
                None
            }
        } else {
            None
        };
        let candidate_session_backend = if let Some(id) = requested_session {
            let profile = engine.session_profile(id)?;
            Some(build_agent_decision_backend(
                config_path.as_deref(),
                profile.as_deref(),
            )?)
        } else {
            None
        };

        let result = match &action {
            TuiAction::SubmitText(input) if input.trim().starts_with('/') => {
                engine.record(EventKind::User {
                    text: input.clone(),
                })?;
                engine.handle_slash_command(input)?
            }
            TuiAction::SubmitText(input) => engine.run_line(input, &mut *backend)?,
            TuiAction::ApprovePlan => engine.handle_plan_decision(PlanDecision::Approve)?,
            TuiAction::RejectPlan => engine.handle_plan_decision(PlanDecision::Reject)?,
            TuiAction::SelectProfile(name) => engine.select_profile(name)?,
            TuiAction::SelectSession(id) => engine.select_session(id)?,
            TuiAction::TogglePlan => engine.handle_toggle_plan()?,
        };

        if let Some(candidate) = candidate_backend {
            backend = candidate;
        }
        if let Some(candidate) = candidate_session_backend {
            engine.set_config_path(config_path.as_deref())?;
            backend = candidate;
        }

        let changed_session =
            submitted_text.is_some_and(|input| matches!(input.trim(), "/clear" | "/resume"));
        if changed_session {
            engine.set_config_path(config_path.as_deref())?;
            let profile = engine
                .session
                .as_ref()
                .and_then(|session| session.profile.as_deref());
            backend = build_agent_decision_backend(config_path.as_deref(), profile)?;
        }

        // Save session after each interaction.
        let _ = engine.save();

        let profile_options = submitted_text
            .is_some_and(|input| matches!(input.trim(), "/model" | "/profile"))
            .then(|| engine.profile_choices())
            .transpose()?
            .filter(|options| !options.is_empty());
        let session_options = submitted_text
            .is_some_and(|input| input.trim() == "/session")
            .then(|| engine.session_choices(20))
            .transpose()?
            .filter(|options| !options.is_empty());

        if requested_session.is_some() || changed_session {
            Ok(TuiInteraction::SessionChanged {
                input_history: engine.input_history(),
                events: engine.session_events(),
            })
        } else if let Some(options) = session_options {
            Ok(TuiInteraction::SessionPicker {
                message: result,
                options,
            })
        } else if engine.has_pending_plan() {
            Ok(TuiInteraction::PlanApproval { message: result })
        } else if let Some(options) = profile_options {
            Ok(TuiInteraction::ProfilePicker {
                message: result,
                options,
            })
        } else {
            let render = render_policy(&action, &result);
            Ok(TuiInteraction::Message {
                message: result,
                render,
            })
        }
    })
}

fn render_policy(action: &TuiAction, response: &str) -> RenderPolicy {
    if !matches!(action, TuiAction::SubmitText(input) if !input.trim().starts_with('/'))
        || response.contains('\n')
    {
        RenderPolicy::Immediate
    } else {
        RenderPolicy::Stream
    }
}

fn build_agent_decision_backend(
    config_path: Option<&Path>,
    profile: Option<&str>,
) -> io::Result<Box<dyn subbake_core::ports::LlmBackend>> {
    let mut settings = TranslationSettings::default();
    if let Some(path) = config_path {
        let patch = load_and_resolve(path, profile)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("configuration disappeared: {}", path.display()),
            )
        })?;
        settings.apply_patch(patch);
    }

    if settings.provider == "mock" {
        return Ok(Box::new(EchoDecisionBackend::new("mock-decision")));
    }

    build_backend(&settings.backend_config())
        .map_err(|error| io::Error::other(format!("build agent backend: {error}")))
}

#[cfg(test)]
mod tests {
    use super::render_policy;
    use subbake_agent::{RenderPolicy, TuiAction};

    #[test]
    fn structured_and_command_responses_render_immediately() {
        assert_eq!(
            render_policy(&TuiAction::SubmitText("ls".to_owned()), "one.srt\ntwo.srt"),
            RenderPolicy::Immediate
        );
        assert_eq!(
            render_policy(&TuiAction::SubmitText("/session".to_owned()), "session"),
            RenderPolicy::Immediate
        );
        assert_eq!(
            render_policy(&TuiAction::SubmitText("hello".to_owned()), "hello!"),
            RenderPolicy::Stream
        );
    }
}
