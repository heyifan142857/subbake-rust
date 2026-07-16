use std::path::{Path, PathBuf};

use subbake_adapters::{
    ConfigurationResolver, ResolveRequest, TranslationSettings, build_backend, discover_config_path,
};
use subbake_agent::event::EventKind;
use subbake_agent::{
    AgentActionKind, AgentEngine, AgentError, AgentResult, EchoDecisionBackend, PlanDecision,
    StartupInfo, SubBakeTui, TuiAction, TuiInteraction, is_known_slash_command,
};

use crate::CliResult;
use crate::args::AgentArgs;

pub fn run(args: AgentArgs) -> CliResult<()> {
    // No-args / resume → start TUI.
    match args.action.kind {
        AgentActionKind::Start => start_interactive(),
        AgentActionKind::Resume => {
            if let Some(sid) = &args.action.session_id {
                start_interactive_resume(Some(sid))
            } else {
                start_interactive_resume(None)
            }
        }
    }
}

fn start_interactive() -> CliResult<()> {
    let project_root = std::env::current_dir()?;
    let mut engine = AgentEngine::new(project_root);
    engine.start_session()?;

    run_tui_with_engine(engine, false)
}

fn start_interactive_resume(session_id: Option<&str>) -> CliResult<()> {
    let project_root = std::env::current_dir()?;
    let mut engine = AgentEngine::new(project_root);
    engine.resume_session(session_id)?;

    run_tui_with_engine(engine, session_id.is_none())
}

fn run_tui_with_engine(mut engine: AgentEngine, open_session_picker: bool) -> CliResult<()> {
    let project_root = engine.project_root().to_path_buf();
    let (mut config_path, needs_config_pin) = session_config_path(&engine);
    if needs_config_pin {
        engine.set_config_path(config_path.as_deref())?;
    }
    let initial_profile = engine.active_profile().map(str::to_owned);
    let mut backend =
        build_agent_decision_backend(config_path.as_deref(), initial_profile.as_deref())?;
    let startup_settings = resolved_settings(config_path.as_deref(), initial_profile.as_deref())?;

    // Create the TUI with an observer attached to the engine.
    let input_history = engine.input_history();
    let session_events = engine.session_events();
    let initial_plan_mode = engine.is_plan_mode();
    let mut tui = SubBakeTui::new()?;
    tui.set_startup_info(StartupInfo {
        provider: startup_settings.backend.id,
        model: startup_settings.backend.model,
        config: config_path
            .as_deref()
            .map(display_config_path)
            .unwrap_or_else(|| "Not configured".to_owned()),
        cache_enabled: startup_settings.translation.use_cache,
        cwd: project_root.to_string_lossy().into_owned(),
    });
    tui.set_has_config_file(config_path.is_some());
    tui.set_cancellation_token(engine.cancellation_token());
    tui.set_input_history(input_history);
    if !open_session_picker {
        tui.set_session_replay(session_events);
    }
    tui.set_plan_mode(initial_plan_mode);
    if open_session_picker {
        tui.open_session_picker(engine.session_choices(20)?)?;
    }
    let observer = tui.observer();
    engine = engine
        .with_progress(std::sync::Arc::new(observer.clone()))
        .with_observer(Box::new(observer));

    Ok(tui.run(move |action, guard, _obs| {
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
            let (profile, stored_config_path) = engine.session_config(id)?;
            let needs_config_pin = stored_config_path.is_none();
            let candidate_config_path = stored_config_path
                .map(PathBuf::from)
                .or_else(discover_config_path);
            let candidate_backend =
                build_agent_decision_backend(candidate_config_path.as_deref(), profile.as_deref())?;
            Some((candidate_backend, candidate_config_path, needs_config_pin))
        } else {
            None
        };

        let operation_result = (|| -> AgentResult<String> {
            Ok(match &action {
                TuiAction::SubmitText(input) if is_known_slash_command(input) => {
                    if !input.trim().starts_with("/plan") {
                        engine.record(EventKind::User {
                            text: input.clone(),
                        })?;
                    }
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
            Err(error) if error.is_cancelled() => {
                let _ = engine.record(EventKind::Cancelled);
                let _ = engine.save();
                return Err(error);
            }
            Err(error) => {
                let message = error.to_string();
                let displayed = match engine.record_error(&message) {
                    Ok(path) => format!("{message}\nError details saved to:\n{}", path.display()),
                    Err(save_error) => {
                        format!("{message}\nWarning: failed to save error details: {save_error}")
                    }
                };
                return Err(AgentError::Reported {
                    message: displayed,
                    source: Box::new(error),
                });
            }
        };

        if let Some(candidate) = candidate_backend {
            backend = candidate;
        }
        if let Some((candidate, candidate_config_path, needs_config_pin)) =
            candidate_session_backend
        {
            config_path = candidate_config_path;
            if needs_config_pin {
                engine.set_config_path(config_path.as_deref())?;
            }
            backend = candidate;
        }

        let changed_session = submitted_text.is_some_and(|input| input.trim() == "/clear");
        let changed_plan_mode = matches!(action, TuiAction::TogglePlan)
            || submitted_text.is_some_and(|input| input.trim().starts_with("/plan"));
        if changed_session {
            engine.set_config_path(config_path.as_deref())?;
            let profile = engine.active_profile();
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
            let profile = engine.active_profile();
            let model = resolved_settings(config_path.as_deref(), profile)?
                .backend
                .model;
            Ok(TuiInteraction::SessionChanged {
                input_history: engine.input_history(),
                events: engine.session_events(),
                plan_mode: engine.is_plan_mode(),
                model,
            })
        } else if changed_plan_mode {
            Ok(TuiInteraction::PlanModeChanged {
                enabled: engine.is_plan_mode(),
            })
        } else if matches!(
            action,
            TuiAction::SelectProfile(_) | TuiAction::CreateProfile(_)
        ) || requested_profile.is_some()
        {
            let profile = engine.active_profile();
            let settings = resolved_settings(config_path.as_deref(), profile)?;
            Ok(TuiInteraction::ModelChanged {
                model: settings.backend.model,
                message: result,
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
            Ok(TuiInteraction::Message { message: result })
        }
    })?)
}

fn prepare_profile_backend(
    engine: &AgentEngine,
    config_path: Option<&Path>,
    profile: &str,
) -> AgentResult<Option<Box<dyn subbake_core::ports::LlmBackend>>> {
    if !engine.profile_choices()?.iter().any(|name| name == profile) {
        return Ok(None);
    }
    build_agent_decision_backend(config_path, Some(profile)).map(Some)
}

fn session_config_path(engine: &AgentEngine) -> (Option<PathBuf>, bool) {
    let stored = engine.active_config_path().map(PathBuf::from);
    let needs_config_pin = stored.is_none();
    (stored.or_else(discover_config_path), needs_config_pin)
}

fn build_agent_decision_backend(
    config_path: Option<&Path>,
    profile: Option<&str>,
) -> AgentResult<Box<dyn subbake_core::ports::LlmBackend>> {
    let settings = resolved_settings(config_path, profile)?;

    if settings.backend.id == "mock" {
        return Ok(Box::new(EchoDecisionBackend::new("mock-decision")));
    }

    build_backend(&settings.backend_config()).map_err(|source| AgentError::AdapterContext {
        operation: "build agent backend",
        source: Box::new(source),
    })
}

fn resolved_settings(
    config_path: Option<&Path>,
    profile: Option<&str>,
) -> AgentResult<TranslationSettings> {
    ConfigurationResolver
        .resolve(ResolveRequest {
            pinned_path: config_path.map(Path::to_path_buf),
            profile: profile.map(str::to_owned),
            ..ResolveRequest::default()
        })
        .map(|resolved| resolved.settings)
        .map_err(Into::into)
}

fn display_config_path(path: &Path) -> String {
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
    home.as_deref()
        .and_then(|home| path.strip_prefix(home).ok())
        .map(|relative| format!("~/{}", relative.display()))
        .unwrap_or_else(|| path.display().to_string())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{build_agent_decision_backend, prepare_profile_backend, session_config_path};
    use subbake_agent::AgentEngine;

    #[test]
    fn invalid_profile_backend_fails_instead_of_falling_back_to_mock() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("subbake-agent-bad-profile-{nonce}.toml"));
        std::fs::write(
            &path,
            "version = 1\n\
             [profiles.bad.backend]\n\
             id = \"not-a-provider\"\n\
             model = \"none\"\n",
        )
        .expect("write config");

        let error = build_agent_decision_backend(Some(&path), Some("bad"))
            .err()
            .expect("invalid provider must fail");
        assert!(error.to_string().contains("api_format"));
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
            "version = 1\n\
             [profiles.bad.backend]\n\
             id = \"not-a-provider\"\n\
             model = \"none\"\n",
        )
        .expect("write config");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        engine.set_config_path(Some(&path)).expect("pin config");

        prepare_profile_backend(&engine, Some(&path), "bad")
            .err()
            .expect("invalid backend must fail before switching");
        assert_eq!(
            engine.active_profile(),
            None,
            "backend preparation must not commit the requested profile"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn resumed_session_keeps_its_pinned_config_path() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("subbake-pinned-config-{nonce}"));
        let pinned = root.join("original.toml");
        std::fs::create_dir_all(&root).expect("create root");

        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        engine.set_config_path(Some(&pinned)).expect("pin config");

        assert_eq!(session_config_path(&engine), (Some(pinned), false));
        let _ = std::fs::remove_dir_all(root);
    }
}
