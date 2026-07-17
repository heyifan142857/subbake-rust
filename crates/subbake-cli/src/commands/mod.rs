use crate::args::{
    parse_agent_args, parse_batch_args, parse_pipeline_args, parse_provider_args,
    parse_resume_args, parse_runtime_args, parse_transcribe_args, parse_translate_args,
    parse_whisper_args,
};
use crate::{CliError, CliResult};

mod agent;
mod pipeline;
mod provider;
mod runtime;
mod transcribe;
mod translate;
mod whisper;

pub fn dispatch(args: Vec<String>) -> CliResult<()> {
    if args.is_empty() {
        return agent::run(parse_agent_args(&[])?);
    }

    if let Some(help) = requested_help(&args) {
        print!("{help}");
        return Ok(());
    }

    match args[0].as_str() {
        "agent" => agent::run(parse_agent_args(&args[1..])?),
        "resume" => agent::run(parse_resume_args(&args[1..])?),
        "translate" => translate::translate_file(parse_translate_args(&args[1..])?).map(|_| ()),
        "batch" => translate::translate_batch(parse_batch_args(&args[1..])?),
        "transcribe" => transcribe::run(parse_transcribe_args(&args[1..])?),
        "pipeline" => pipeline::run(parse_pipeline_args(&args[1..])?),
        "provider" => provider::run(parse_provider_args(&args[1..])?),
        "runtime" => runtime::run(parse_runtime_args(&args[1..])?),
        "whisper" => whisper::run(parse_whisper_args(&args[1..])?),
        "--help" | "-h" => {
            print!("{}", help_text(&[]));
            Ok(())
        }
        "--version" | "-V" => {
            println!("sbake {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        other => Err(CliError::usage(format!(
            "unknown command `{other}`; run `sbake --help`"
        ))),
    }
}

fn requested_help(args: &[String]) -> Option<&'static str> {
    let help_position = args
        .iter()
        .position(|arg| matches!(arg.as_str(), "--help" | "-h"))?;
    Some(help_text(&args[..help_position]))
}

pub(crate) fn help_text(command: &[String]) -> &'static str {
    match command
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] => TOP_LEVEL_HELP,
        ["agent"] => AGENT_HELP,
        ["resume"] => RESUME_HELP,
        ["translate"] => TRANSLATE_HELP,
        ["batch"] => BATCH_HELP,
        ["transcribe"] => TRANSCRIBE_HELP,
        ["pipeline"] => PIPELINE_HELP,
        ["provider"] | ["provider", "check"] => PROVIDER_HELP,
        ["runtime"] => RUNTIME_HELP,
        ["runtime", "inspect"] => RUNTIME_INSPECT_HELP,
        ["runtime", "clean"] => RUNTIME_CLEAN_HELP,
        ["whisper"] => WHISPER_HELP,
        ["whisper", "model"] => WHISPER_MODEL_HELP,
        _ => TOP_LEVEL_HELP,
    }
}

const TOP_LEVEL_HELP: &str = r#"Agent-first subtitle translation and transcription CLI

Usage: sbake [OPTIONS] [COMMAND]

Commands:
  agent       Start the interactive agent (also the default with no command)
  resume      Resume the latest or a specified agent session
  translate   Translate a subtitle or text file
  batch       Translate subtitle files in a directory
  transcribe  Transcribe audio or video into subtitles
  pipeline    Transcribe media when needed, then translate it
  provider    Validate a model provider configuration
  runtime     Inspect or clean runtime artifacts
  whisper     Install and manage whisper.cpp and its models
  help        Print help for a command

Options:
  -h, --help     Print help
  -V, --version  Print version

Examples:
  sbake
  sbake translate movie.srt --target-lang Chinese
  sbake pipeline movie.mp4 --target-lang English
  sbake resume
  sbake provider check
"#;

const AGENT_HELP: &str = r#"Start the interactive subtitle agent

Usage: sbake agent

The agent is also started when sbake is run without a command.
"#;
const RESUME_HELP: &str = r#"Resume an interactive agent session

Usage: sbake resume [SESSION_ID]

Omit SESSION_ID to resume the latest session.
"#;
const TRANSLATE_HELP: &str = r#"Translate a subtitle or text file

Usage: sbake translate <SUBTITLE> [OPTIONS]

Options:
  -o, --output <PATH>              Output file path
      --config <PATH>              Configuration file
      --profile <NAME>             Named provider profile
      --source-lang <LANGUAGE>     Source language
      --target-lang <LANGUAGE>     Target language
      --provider <NAME>            Provider name
      --model <NAME>               Model name
      --output-format <FORMAT>     Output subtitle format
      --bilingual                  Include source and translated text
      --mode <MODE>               Translation mode: economy, turbo, or cinema
      --review <POLICY>            Review policy: targeted or full (default: off)
      --no-review                  Disable review
      --fast                       Deprecated alias for --mode turbo
      --dry-run                    Prepare work without provider calls
      --json                       Emit structured JSON output
  -h, --help                       Print help

Additional provider, batching, concurrency, cache, retry, glossary, and runtime
options are accepted. Media input is not transcribed; use `sbake pipeline`.
"#;
const BATCH_HELP: &str = r#"Translate subtitle files in a directory

Usage: sbake batch <DIR> [OPTIONS]

Options:
      --recursive              Include nested directories
      --overwrite              Replace existing outputs
      --config <PATH>          Configuration file
      --profile <NAME>         Named provider profile
      --target-lang <LANGUAGE> Target language
      --bilingual              Include source and translated text
  -h, --help                   Print help

Translation provider, model, review, batching, cache, retry, and runtime options
accepted by `translate` are also available.
"#;
const TRANSCRIBE_HELP: &str = r#"Transcribe audio or video into subtitles

Usage: sbake transcribe <MEDIA> [OPTIONS]

Options:
  -o, --output <PATH>          Output file path
      --language <LANGUAGE>    Spoken language
      --provider <NAME>        Transcription provider
      --model <NAME>           Transcription model
      --format <FORMAT>        Output format: srt, vtt, or txt
      --sidecar <PATH>         Use a sidecar transcript
      --base-url <URL>         Provider base URL
      --api-key <KEY>          Provider API key
  -h, --help                   Print help
"#;
const PIPELINE_HELP: &str = r#"Transcribe media when needed, then translate it

Usage: sbake pipeline <MEDIA_OR_SUBTITLE> [OPTIONS]

Accepts all `translate` options plus:
      --transcriber <NAME>             Transcription provider
      --transcribe-language <LANGUAGE> Spoken language
      --transcribe-model <NAME>        Transcription model
      --transcribe-format <FORMAT>     srt, vtt, or txt
      --sidecar <PATH>                 Use a sidecar transcript
  -h, --help                           Print help
"#;
const PROVIDER_HELP: &str = r#"Validate a model provider configuration

Usage: sbake provider check [OPTIONS]

Options:
      --config <PATH>        Configuration file
      --profile <NAME>       Named runtime profile
      --provider <NAME>       Provider name
      --model <NAME>          Model name
      --api-format <FORMAT>   Provider wire protocol
      --base-url <URL>        Provider base URL
      --endpoint-url <URL>    Complete endpoint URL
      --api-key <KEY>         Inline API key
      --api-key-env <NAME>    API-key environment variable
      --auth-header <NAME>    Authorization header name
      --auth-prefix <PREFIX>  Authorization value prefix
  -h, --help                  Print help
"#;
const RUNTIME_HELP: &str = r#"Inspect or clean runtime artifacts

Usage: sbake runtime <COMMAND>

Commands:
  inspect  Inspect runtime artifacts for a target
  clean    Remove selected runtime artifacts

Run `sbake runtime <COMMAND> --help` for details.
"#;
const RUNTIME_INSPECT_HELP: &str = r#"Inspect runtime artifacts for a target

Usage: sbake runtime inspect <TARGET> [--runtime-dir <DIR>]
"#;
const RUNTIME_CLEAN_HELP: &str = r#"Remove runtime artifacts for a target

Usage: sbake runtime clean <TARGET> --yes [OPTIONS]

Options:
      --runs          Remove run state
      --cache         Remove request and review caches
      --glossary      Remove glossary data
      --all           Remove all runtime artifacts
      --runtime-dir <DIR>  Override the runtime directory
      --yes           Confirm deletion
  -h, --help          Print help
"#;
const WHISPER_HELP: &str = r#"Install and manage whisper.cpp and its models

Usage: sbake whisper [COMMAND] [OPTIONS]

Commands:
  status              Report installation status (default)
  install             Install whisper.cpp
  update              Update whisper.cpp
  uninstall           Uninstall whisper.cpp
  model list          List supported models
  model <NAME>        Download a model

Options:
      --bin <PATH>         Override the whisper binary path
      --models-dir <DIR>   Override the models directory
      --keep-models        Keep models when uninstalling
  -h, --help               Print help
"#;
const WHISPER_MODEL_HELP: &str = r#"List or download whisper.cpp models

Usage:
  sbake whisper model list
  sbake whisper model <NAME> [--models-dir <DIR>]
"#;
