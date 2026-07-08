use std::io;

use crate::args::{
    parse_batch_args, parse_pipeline_args, parse_provider_args, parse_transcribe_args,
    parse_translate_args, parse_whisper_args,
};

mod pipeline;
mod provider;
mod runtime;
mod transcribe;
mod translate;
mod whisper;

pub fn dispatch(args: Vec<String>) -> io::Result<()> {
    if args.is_empty() {
        println!("{}", subbake_agent::start_agent());
        return Ok(());
    }

    match args[0].as_str() {
        "agent" => run_agent(&args[1..]),
        "translate" => translate::translate_file(parse_translate_args(&args[1..])?).map(|_| ()),
        "batch" => translate::translate_batch(parse_batch_args(&args[1..])?),
        "transcribe" => transcribe::run(parse_transcribe_args(&args[1..])?),
        "pipeline" => pipeline::run(parse_pipeline_args(&args[1..])?),
        "provider" => provider::run(parse_provider_args(&args[1..])?),
        "runtime" => runtime::run(&args[1..]),
        "whisper" => whisper::run(parse_whisper_args(&args[1..])?),
        "--help" | "-h" => {
            print_help();
            Ok(())
        }
        "--version" | "-V" => {
            println!("sbake {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        other => Err(io::Error::other(format!(
            "unknown command `{other}`; run `sbake --help`"
        ))),
    }
}

fn print_help() {
    println!("sbake - agent-first subtitle translation CLI");
    println!();
    println!("Commands:");
    for name in crate::command_names() {
        println!("  {name}");
    }
}

fn run_agent(args: &[String]) -> io::Result<()> {
    if args.first().is_some_and(|value| value == "resume") {
        println!(
            "{}",
            subbake_agent::resume_agent(args.get(1).map(String::as_str))
        );
    } else if args.is_empty() {
        println!("{}", subbake_agent::start_agent());
    } else {
        return Err(io::Error::other(
            "unsupported agent command; use `agent resume [SESSION_ID]`",
        ));
    }
    Ok(())
}
