use subbake_adapters::{MemoryRequest, manage_memory};

use crate::CliResult;
use crate::args::MemoryArgs;
use crate::output::print_json_value;

pub fn run(args: MemoryArgs) -> CliResult<()> {
    let outcome = manage_memory(MemoryRequest {
        action: args.action,
        target_path: args.target_path,
        settings: args.settings,
    })?;
    if args.json {
        print_json_value(
            "memory_result",
            serde_json::json!({
                "glossary_entries": outcome.glossary_entries,
                "translation_memory_entries": outcome.translation_memory_entries,
                "changed_entries": outcome.changed_entries,
                "bundle_path": outcome.bundle_path,
                "glossary_path": outcome.glossary_path,
                "translation_memory_path": outcome.translation_memory_path,
            }),
        );
        return Ok(());
    }
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
