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

    run_tui_with_engine(engine, false)
}

fn start_interactive_resume(session_id: Option<&str>) -> io::Result<()> {
    let project_root = std::env::current_dir()?;
    let mut engine = AgentEngine::new(project_root);
    engine.resume_session(session_id)?;

    run_tui_with_engine(engine, session_id.is_none())
}

fn run_tui_with_engine(mut engine: AgentEngine, open_session_picker: bool) -> io::Result<()> {
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
    tui.set_has_config_file(config_path.is_some());
    tui.set_cancellation_token(engine.cancellation_token());
    tui.set_input_history(input_history);
    tui.set_session_replay(session_events);
    if open_session_picker {
        tui.open_session_picker(engine.session_choices(20)?);
    }
    let observer = tui.observer();
    engine = engine.with_observer(Box::new(observer));

    tui.run(move |action, guard, _obs| {
        engine.begin_operation(guard);
        let submitted_text = match &action {
            TuiAction::SubmitText(text) => Some(text.as_str()),
            _ => None,
        };
        let requested_profile = match &action {
            TuiAction::SelectProfile(name) => Some(name.as_str()),
            TuiAction::CreateProfile(_) => None,
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
                .strip_prefix("/sessions ")
                .map(str::trim)
                .filter(|id| !id.is_empty()),
            _ => None,
        };
        let candidate_backend = if let Some(profile) = requested_profile {
            prepare_profile_backend(&engine, config_path.as_deref(), profile)?
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

        let operation_result = (|| -> io::Result<String> {
            Ok(match &action {
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
                TuiAction::CreateProfile(name) => engine.create_profile(name)?,
                TuiAction::SelectSession(id) => engine.select_session(id)?,
                TuiAction::TogglePlan => engine.handle_toggle_plan()?,
            })
        })();
        let result = match operation_result {
            Ok(result) => result,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {
                let _ = engine.record(EventKind::Cancelled);
                let _ = engine.save();
                return Err(error);
            }
            Err(error) => return Err(error),
        };

        if let Some(candidate) = candidate_backend {
            backend = candidate;
        }
        if let Some(candidate) = candidate_session_backend {
            engine.set_config_path(config_path.as_deref())?;
            backend = candidate;
        }

        let changed_session = submitted_text.is_some_and(|input| input.trim() == "/clear");
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
            .then(|| engine.profile_picker_choices())
            .transpose()?
            .filter(|options| !options.is_empty());
        let session_options = submitted_text
            .is_some_and(|input| input.trim() == "/sessions")
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

fn prepare_profile_backend(
    engine: &AgentEngine,
    config_path: Option<&Path>,
    profile: &str,
) -> io::Result<Option<Box<dyn subbake_core::ports::LlmBackend>>> {
    if !engine.profile_choices()?.iter().any(|name| name == profile) {
        return Ok(None);
    }
    build_agent_decision_backend(config_path, Some(profile)).map(Some)
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
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{build_agent_decision_backend, prepare_profile_backend, render_policy};
    use subbake_agent::{AgentEngine, RenderPolicy, TuiAction};

    #[test]
    fn structured_and_command_responses_render_immediately() {
        assert_eq!(
            render_policy(&TuiAction::SubmitText("ls".to_owned()), "one.srt\ntwo.srt"),
            RenderPolicy::Immediate
        );
        assert_eq!(
            render_policy(&TuiAction::SubmitText("/sessions".to_owned()), "session"),
            RenderPolicy::Immediate
        );
        assert_eq!(
            render_policy(&TuiAction::SubmitText("hello".to_owned()), "hello!"),
            RenderPolicy::Stream
        );
    }

    #[test]
    fn invalid_profile_backend_fails_instead_of_falling_back_to_mock() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("subbake-agent-bad-profile-{nonce}.toml"));
        std::fs::write(
            &path,
            "[profiles.bad]\nprovider = \"not-a-provider\"\nmodel = \"none\"\n",
        )
        .expect("write config");

        let error = build_agent_decision_backend(Some(&path), Some("bad"))
            .err()
            .expect("invalid provider must fail");
        assert!(error.to_string().contains("build agent backend"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn failed_profile_backend_preparation_does_not_mutate_the_session() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("subbake-profile-atomic-{nonce}"));
        std::fs::create_dir_all(&root).expect("create root");
        let path = root.join("subbake.toml");
        std::fs::write(
            &path,
            "[profiles.bad]\nprovider = \"not-a-provider\"\nmodel = \"none\"\n",
        )
        .expect("write config");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        engine.set_config_path(Some(&path)).expect("pin config");

        prepare_profile_backend(&engine, Some(&path), "bad")
            .err()
            .expect("invalid backend must fail before switching");
        assert_eq!(
            engine.session.as_ref().expect("session").profile,
            None,
            "backend preparation must not commit the requested profile"
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
