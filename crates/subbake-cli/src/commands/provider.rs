use subbake_adapters::{ProviderCheckRequest, check_provider_cancellable};

use crate::CliResult;
use crate::args::ProviderArgs;
use crate::output::print_provider_check_outcome;

pub fn run(args: ProviderArgs) -> CliResult<()> {
    let cancellation = crate::cancellation::CliCancellation::new()?;
    let outcome = check_provider_cancellable(
        ProviderCheckRequest {
            config: args.config,
        },
        cancellation.guard(),
    )?;
    print_provider_check_outcome(&outcome);
    Ok(())
}
