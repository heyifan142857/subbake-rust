use subbake_adapters::{MemoryRequest, manage_memory};

use crate::CliResult;
use crate::args::MemoryArgs;

pub fn run(args: MemoryArgs) -> CliResult<()> {
    let outcome = manage_memory(MemoryRequest {
        action: args.action,
        target_path: args.target_path,
        settings: args.settings,
    })?;
    println!("Glossary entries: {}", outcome.glossary_entries);
    println!(
        "Translation-memory entries: {}",
        outcome.translation_memory_entries
    );
    if outcome.changed_entries > 0 {
        println!("Changed entries: {}", outcome.changed_entries);
    }
    if let Some(path) = outcome.bundle_path {
        println!("Bundle: {}", path.display());
    }
    println!("Glossary: {}", outcome.glossary_path.display());
    println!(
        "Translation memory: {}",
        outcome.translation_memory_path.display()
    );
    Ok(())
}
