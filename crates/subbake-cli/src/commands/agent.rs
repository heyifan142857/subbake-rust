use std::io;

use subbake_agent::AgentRequest;

use crate::args::AgentArgs;
use crate::output::print_agent_outcome;

pub fn run(args: AgentArgs) -> io::Result<()> {
    let outcome = subbake_agent::run_agent(AgentRequest {
        action: args.action,
    });
    print_agent_outcome(&outcome);
    Ok(())
}
