use std::io;

use subbake_adapters::{
    TranslationSettings, build_backend, discover_config_path, load_and_resolve,
};
use subbake_agent::{
    AgentEngine, AgentRequest, EchoDecisionBackend, SubBakeTui, TuiPicker, TuiProcessResult,
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
    let initial_profile = engine
        .session
        .as_ref()
        .and_then(|session| session.profile.clone());
    let mut backend = build_agent_decision_backend(initial_profile.as_deref());

    // Create the TUI with an observer attached to the engine.
    let mut tui = SubBakeTui::new()?;
    let observer = tui.observer();
    engine = engine.with_observer(Box::new(observer));

    tui.run(move |input, _obs| {
        // Each user input: run through the engine's decision pipeline.
        let result = if input.trim().starts_with('/') {
            engine.handle_slash_command(input)?
        } else {
            engine.run_line(input, &mut *backend)?
        };

        if input.trim().starts_with("/profile ") {
            let profile = engine
                .session
                .as_ref()
                .and_then(|session| session.profile.clone());
            backend = build_agent_decision_backend(profile.as_deref());
        }

        // Save session after each interaction.
        let _ = engine.save();

        let picker = matches!(input.trim(), "/model" | "/profile")
            .then(|| engine.profile_choices())
            .transpose()?
            .and_then(|options| {
                (!options.is_empty()).then_some(TuiPicker {
                    command: "/profile".to_owned(),
                    options,
                })
            });

        Ok(TuiProcessResult {
            response: result,
            pending_plan: engine.has_pending_plan(),
            picker,
        })
    })
}

fn build_agent_decision_backend(profile: Option<&str>) -> Box<dyn subbake_core::ports::LlmBackend> {
    let mut settings = TranslationSettings::default();
    if let Some(path) = discover_config_path()
        && let Ok(Some(patch)) = load_and_resolve(&path, profile)
    {
        settings.apply_patch(patch);
    }

    if settings.provider == "mock" {
        return Box::new(EchoDecisionBackend::new("mock-decision"));
    }

    build_backend(&settings.backend_config())
        .unwrap_or_else(|_| Box::new(EchoDecisionBackend::new("mock-decision")))
}
