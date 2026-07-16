use subbake_adapters::{ProviderCheckRequest, check_provider};

use crate::CliResult;
use crate::args::ProviderArgs;
use crate::output::print_provider_check_outcome;

pub fn run(args: ProviderArgs) -> CliResult<()> {
    let outcome = check_provider(ProviderCheckRequest {
        config: args.config,
    })?;
    print_provider_check_outcome(&outcome);
    Ok(())
}
