use std::io;

use subbake_agent::{AgentEngine, AgentRequest, EchoDecisionBackend, SubBakeTui};

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
            let outcome = subbake_agent::run_agent(AgentRequest { action: args.action });
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
    // Build a mock LLM backend for the agent decision loop.
    let mut backend: Box<dyn subbake_core::ports::LlmBackend> =
        Box::new(EchoDecisionBackend::new("mock-decision"));

    // Create the TUI with an observer attached to the engine.
    let mut tui = SubBakeTui::new()?;
    let observer = tui.observer();
    engine = engine.with_observer(Box::new(observer));

    tui.run(|input, _obs| {
        // Each user input: run through the engine's decision pipeline.
        let result = engine.run_line(input, &mut *backend)?;

        // Save session after each interaction.
        let _ = engine.save();

        Ok(result)
    })
}
