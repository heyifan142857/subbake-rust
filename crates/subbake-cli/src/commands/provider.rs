use std::io;

use subbake_adapters::{ProviderCheckRequest, check_provider};

use crate::args::ProviderArgs;
use crate::output::print_provider_check_outcome;

pub fn run(args: ProviderArgs) -> io::Result<()> {
    let outcome = check_provider(ProviderCheckRequest {
        config: args.config,
    })
    .map_err(io::Error::other)?;
    print_provider_check_outcome(&outcome);
    Ok(())
}
