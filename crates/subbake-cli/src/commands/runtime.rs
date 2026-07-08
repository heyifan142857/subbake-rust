use std::io;

use subbake_adapters::{RuntimeRequest, run_runtime};

use crate::args::RuntimeArgs;
use crate::output::print_runtime_outcome;

pub fn run(args: RuntimeArgs) -> io::Result<()> {
    let outcome = run_runtime(RuntimeRequest {
        action: args.action,
        target_path: args.target_path,
        runtime_dir: args.runtime_dir,
    })?;
    print_runtime_outcome(&outcome);
    Ok(())
}
