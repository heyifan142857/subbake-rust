use std::io;

use subbake_adapters::{PipelineRequest, run_pipeline};

use crate::args::TranslateArgs;
use crate::output::print_pipeline_outcome;

pub fn run(args: TranslateArgs) -> io::Result<()> {
    let outcome = run_pipeline(PipelineRequest {
        input_path: args.input_path,
        output_path: args.output,
        settings: args.settings,
    })?;
    print_pipeline_outcome(&outcome, args.json)?;
    Ok(())
}
