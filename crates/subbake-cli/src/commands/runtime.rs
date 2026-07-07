use std::fs;
use std::io;
use std::path::Path;

use subbake_core::storage::build_runtime_paths;

use crate::args::option_path_value;

pub fn run(args: &[String]) -> io::Result<()> {
    match args.first().map(String::as_str) {
        Some("inspect") => inspect(args),
        Some("clean") => clean(args),
        _ => Err(io::Error::other("runtime requires `inspect` or `clean`")),
    }
}

fn inspect(args: &[String]) -> io::Result<()> {
    let target = args
        .get(1)
        .ok_or_else(|| io::Error::other("runtime inspect requires a target"))?;
    let runtime_dir = option_path_value(args, "--runtime-dir")?;
    let paths = build_runtime_paths(
        Path::new(target),
        runtime_dir.as_deref(),
        None,
        "Auto",
        "Chinese",
        false,
    );
    println!("runtime: {}", paths.root_dir.display());
    println!("run: {}", paths.run_dir.display());
    println!("cache: {}", paths.cache_dir.display());
    println!("state: {}", paths.state_path.display());
    println!("glossary: {}", paths.glossary_path.display());
    Ok(())
}

fn clean(args: &[String]) -> io::Result<()> {
    let target = args
        .get(1)
        .ok_or_else(|| io::Error::other("runtime clean requires a target"))?;
    let yes = args.iter().any(|value| value == "--yes");
    let runtime_dir = option_path_value(args, "--runtime-dir")?;
    clean_runtime(Path::new(target), runtime_dir.as_deref(), yes)
}

fn clean_runtime(target: &Path, runtime_dir: Option<&Path>, yes: bool) -> io::Result<()> {
    let paths = build_runtime_paths(target, runtime_dir, None, "Auto", "Chinese", false);
    if !yes {
        return Err(io::Error::other(
            "runtime clean requires --yes in the current non-interactive implementation",
        ));
    }
    if paths.root_dir.exists() {
        fs::remove_dir_all(&paths.root_dir)?;
        println!("Removed: {}", paths.root_dir.display());
    } else {
        println!("Nothing removed: {}", paths.root_dir.display());
    }
    Ok(())
}
