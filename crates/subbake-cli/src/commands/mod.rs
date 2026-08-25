#[cfg(feature = "agent")]
use crate::args::{parse_agent_args, parse_resume_args};
use crate::args::{
    parse_batch_args, parse_edit_args, parse_evaluate_args, parse_memory_args,
    parse_overnight_args, parse_pipeline_args, parse_project_args, parse_provider_args,
    parse_qa_args, parse_runtime_args, parse_transcribe_args, parse_translate_args,
    parse_whisper_args,
};
use crate::{CliError, CliResult};

#[cfg(feature = "agent")]
mod agent;
mod completion;
mod edit;
mod evaluate;
mod memory;
mod overnight;
mod pipeline;
mod project;
mod provider;
mod qa;
mod runtime;
mod transcribe;
mod translate;
mod whisper;

pub(crate) struct CommandSpec {
    pub name: &'static str,
    pub summary: &'static str,
    pub help: &'static str,
    pub options: &'static [&'static str],
    pub subcommands: &'static [&'static str],
}

pub(crate) fn command_specs() -> &'static [CommandSpec] {
    COMMAND_SPECS
}

pub fn dispatch(args: Vec<String>) -> CliResult<()> {
    if args.is_empty() {
        #[cfg(feature = "agent")]
        {
            return agent::run(parse_agent_args(&[])?);
        }
        #[cfg(not(feature = "agent"))]
        {
            print!("{}", help_text(&[]));
            return Ok(());
        }
    }

    if let Some(help) = requested_help(&args) {
        print!("{help}");
        return Ok(());
    }

    if args[0] == "help" {
        print!("{}", help_text(&args[1..]));
        return Ok(());
    }

    match args[0].as_str() {
        #[cfg(feature = "agent")]
        "agent" => agent::run(parse_agent_args(&args[1..])?),
        #[cfg(feature = "agent")]
        "resume" => agent::run(parse_resume_args(&args[1..])?),
        "translate" => translate::translate_file(parse_translate_args(&args[1..])?).map(|_| ()),
        "edit" => edit::run(parse_edit_args(&args[1..])?),
        "batch" => translate::translate_batch(parse_batch_args(&args[1..])?),
        "evaluate" => evaluate::run(parse_evaluate_args(&args[1..])?),
        "memory" => memory::run(parse_memory_args(&args[1..])?),
        "qa" => qa::run(parse_qa_args(&args[1..])?),
        "project" => project::run(parse_project_args(&args[1..])?),
        "completion" => completion::run(&args[1..]),
        "transcribe" => transcribe::run(parse_transcribe_args(&args[1..])?),
        "pipeline" => pipeline::run(parse_pipeline_args(&args[1..])?),
        "overnight" => overnight::run(parse_overnight_args(&args[1..])?),
        "provider" => provider::run(parse_provider_args(&args[1..])?),
        "runtime" => runtime::run(parse_runtime_args(&args[1..])?),
        "whisper" => whisper::run(parse_whisper_args(&args[1..])?),
        "--help" | "-h" => {
            print!("{}", help_text(&[]));
            Ok(())
        }
        "--version" | "-V" => {
            println!("sbake {}", crate::version::build_identity());
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
    if let [name] = command
        && let Some(spec) = command_specs().iter().find(|spec| spec.name == name)
    {
        return spec.help;
    }
    match command
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] => TOP_LEVEL_HELP,
        #[cfg(feature = "agent")]
        ["agent"] => AGENT_HELP,
        #[cfg(feature = "agent")]
        ["resume"] => RESUME_HELP,
        ["translate"] => TRANSLATE_HELP,
        ["edit"] => EDIT_HELP,
        ["batch"] => BATCH_HELP,
        ["evaluate"] => EVALUATE_HELP,
        ["memory"] | ["memory", _] => MEMORY_HELP,
        ["qa"] => QA_HELP,
        ["project"] => PROJECT_HELP,
        ["transcribe"] => TRANSCRIBE_HELP,
        ["pipeline"] => PIPELINE_HELP,
        ["overnight"]
        | ["overnight", "submit"]
        | ["overnight", "status"]
        | ["overnight", "collect"] => OVERNIGHT_HELP,
        ["provider"] | ["provider", "check"] => PROVIDER_HELP,
        ["runtime"] => RUNTIME_HELP,
        ["runtime", "inspect"] => RUNTIME_INSPECT_HELP,
        ["runtime", "clean"] => RUNTIME_CLEAN_HELP,
        ["whisper"] => WHISPER_HELP,
        ["whisper", "model"] => WHISPER_MODEL_HELP,
        ["whisper", "vad-model"] => WHISPER_VAD_MODEL_HELP,
        _ => TOP_LEVEL_HELP,
    }
}

#[cfg(feature = "agent")]
const TOP_LEVEL_HELP: &str = r#"Agent-first subtitle translation and transcription CLI

Usage: sbake [OPTIONS] [COMMAND]

Commands:
  agent       Start the interactive agent (also the default with no command)
  resume      Resume the latest or a specified agent session
  translate   Translate a subtitle file
  edit        Safely refine an existing translated subtitle
  batch       Translate subtitle files in a directory
  evaluate    Compare a subtitle output with a reference offline
  memory      Inspect, export, import, or prune glossary and translation memory
  qa          Inspect subtitle timing and readability without a reference
  project     Build a project manifest and consistency report
  completion  Generate shell completion scripts
  transcribe  Transcribe audio or video into subtitles
  pipeline    Transcribe media when needed, then translate it
  overnight   Submit, check, and collect a provider-managed economy batch
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
  sbake overnight submit movie.srt --mode economy --profile openai
  sbake resume
  sbake provider check
"#;

#[cfg(not(feature = "agent"))]
const TOP_LEVEL_HELP: &str = r#"Subtitle translation and transcription CLI

Usage: sbake [OPTIONS] [COMMAND]

Commands:
  translate   Translate a subtitle file
  edit        Safely refine an existing translated subtitle
  batch       Translate subtitle files in a directory
  evaluate    Compare a subtitle output with a reference offline
  memory      Inspect, export, import, or prune glossary and translation memory
  qa          Inspect subtitle timing and readability without a reference
  project     Build a project manifest and consistency report
  completion  Generate shell completion scripts
  transcribe  Transcribe audio or video into subtitles
  pipeline    Transcribe media when needed, then translate it
  overnight   Submit, check, and collect a provider-managed economy batch
  provider    Validate a model provider configuration
  runtime     Inspect or clean runtime artifacts
  whisper     Install and manage whisper.cpp and its models
  help        Print help for a command

Options:
  -h, --help     Print help
  -V, --version  Print version

Examples:
  sbake translate movie.srt --target-lang Chinese
  sbake pipeline movie.mp4 --target-lang English
  sbake overnight submit movie.srt --mode economy --profile openai
  sbake provider check
"#;

#[cfg(feature = "agent")]
const AGENT_HELP: &str = r#"Start the interactive subtitle agent

Usage: sbake agent

The agent is also started when sbake is run without a command.
"#;
#[cfg(feature = "agent")]
const RESUME_HELP: &str = r#"Resume an interactive agent session

Usage: sbake resume [SESSION_ID]

Omit SESSION_ID to resume the latest session.
"#;
const TRANSLATE_HELP: &str = r#"Translate a subtitle file or an embedded subtitle track

Usage: sbake translate <SUBTITLE_OR_CONTAINER> [OPTIONS]

Options:
  -o, --output <PATH>              Output file path
      --overwrite                  Replace an existing output
      --config <PATH>              Configuration file
      --profile <NAME>             Named provider profile
      --source-lang <LANGUAGE>     Source language
      --target-lang <LANGUAGE>     Target language
      --subtitle-stream <INDEX>    Embedded text or bitmap stream index
      --preserve-source-container  Write a separate translated media file
      --in-place-container         Replace the source container (default)
      --provider <NAME>            Provider name
      --model <NAME>               Model name
      --output-format <FORMAT>     Output format: srt, vtt, txt, ass, ssa, or ttml
      --bilingual                  Include source and translated text
      --bilingual-font-scale <N>   Scale bilingual ASS fonts (default: 1.0)
      --online-terminology         Merge comprehensive terminology while translating (default in cinema)
      --no-online-terminology      Disable incremental terminology responses
      --allow-degraded-preflight   Continue when terminology preflight fails
      --strict-preflight           Fail when requested terminology preflight is unavailable or fails
      --preserve-names             Keep names in source spelling; disable Turbo name alignment
      --transliterate-names        Transliterate personal names (default)
      --mode <MODE>               Translation mode: economy, turbo, or cinema
      --request-token-budget <N>  Limit estimated prompt plus response tokens per request
      --confirmed-context-lines <N>
                                   Maximum confirmed lines carried into a later batch
      --confirmed-context-token-budget <N>
                                   Token budget for confirmed translation context
      --review <POLICY>            Review policy: targeted or full (default: off)
      --no-review                  Disable review
      --max-characters-per-second <N>
                                   Reject subtitles above this reading speed
      --max-characters-per-line <N>
                                   Reject subtitle lines longer than N characters
      --max-lines <N>              Reject subtitle entries with more than N lines
      --qa-fail-on <LEVEL>         Block publication on QA error or warning
      --dry-run                    Prepare work without provider calls
      --max-requests <N>           Stop before exceeding N provider requests
      --max-tokens <N>             Stop before the next call after N used tokens
      --json                       Emit structured JSON output
  -h, --help                       Print help

Additional provider, batching, concurrency, cache, retry, glossary, and runtime
options are accepted. MKV, MP4/M4V/MOV, and WebM inputs use an existing embedded
subtitle: text tracks are extracted directly, while PGS, VobSub, and DVB bitmap
tracks are OCRed with the installed Tesseract executable and matching language data. Audio is
never transcribed by this command; use `sbake transcribe` or `sbake pipeline`
when speech recognition is intended.
"#;
const EDIT_HELP: &str = r#"Safely refine an existing translated subtitle

Usage: sbake edit <SUBTITLE> --instruction <TEXT> [OPTIONS]

Options:
      --instruction <TEXT>      Requested subtitle edit
      --dry-run                 Preview the proposed diff without writing; long files use a distributed sample
      --allow-non-generated     Allow editing a file without a translated/bilingual name
      --config <PATH>           Configuration file
      --profile <NAME>          Named provider profile
      --source-lang <LANGUAGE>  Original source language
      --target-lang <LANGUAGE>  Edited subtitle language
      --provider <NAME>         Provider name
      --model <NAME>            Model name
      --glossary <PATH>         Required glossary used by deterministic validation
      --json                    Emit structured JSON output
  -h, --help                    Print help

The edit is rejected before publication if it changes IDs, formatting markers,
required terminology, or configured readability limits.
Long edits are processed in bounded batches and written only after every batch passes validation.
"#;
const BATCH_HELP: &str = r#"Translate subtitle files in a directory

Usage: sbake batch <DIR> [OPTIONS]

Options:
      --recursive              Include nested directories
      --overwrite              Replace existing outputs
      --fail-fast              Stop at the first file failure
      --retry-failed <MANIFEST> Process only failures from a prior batch manifest
      --config <PATH>          Configuration file
      --profile <NAME>         Named provider profile
      --target-lang <LANGUAGE> Target language
      --bilingual              Include source and translated text
      --bilingual-font-scale <N>
                              Scale bilingual ASS fonts (default: 1.0)
      --json                  Emit structured JSON output
      --qa-fail-on <LEVEL>    Block publication on QA error or warning
  -h, --help                   Print help

Translation provider, model, review, batching, cache, retry, and runtime options
accepted by `translate` are also available.
"#;
const EVALUATE_HELP: &str = r#"Compare a subtitle output with a reference offline

Usage: sbake evaluate <CANDIDATE> <REFERENCE> [--json]

Reports deterministic chrF and mechanical MQM-style structural findings.
Use it to track regressions; it does not replace human semantic evaluation.
"#;
const QA_HELP: &str = r#"Inspect subtitle timing and readability without a reference

Usage: sbake qa <SUBTITLE> [--json] [--fail-on <LEVEL>]

LEVEL is never (default), error, or warning. Checks empty text, invalid and
overlapping timing, reading speed, line length/count, and repeated segments.
"#;
const PROJECT_HELP: &str = r#"Inspect a subtitle project or season

Usage: sbake project <DIR> [OPTIONS]

Options:
      --recursive          Include nested episode directories
  -o, --output <PATH>      Atomically write the versioned JSON report
      --json               Emit the report as structured JSON
      --fail-on <LEVEL>    Fail on error or warning findings
  -h, --help               Print help

The report inventories pending, translated, and bilingual files; runs QA on
source and output subtitles; checks segment alignment; and reports identical
source lines that have divergent translations across episodes.
"#;
const MEMORY_HELP: &str = r#"Manage glossary and translation-memory data

Usage:
  sbake memory inspect <TARGET> [TRANSLATE OPTIONS]
  sbake memory export <TARGET> <BUNDLE> [TRANSLATE OPTIONS]
  sbake memory import <TARGET> <BUNDLE> [TRANSLATE OPTIONS]
  sbake memory prune <TARGET> --yes [TRANSLATE OPTIONS]

Bundles use a versioned JSON shape. Import merges entries without replacing
existing local values; prune removes blank mappings. Runtime, profile, source-language, and target-language
options are resolved the same way as translation. Pass --json for structured output.
"#;
const TRANSCRIBE_HELP: &str = r#"Transcribe audio or video into subtitles

Usage: sbake transcribe <MEDIA> [OPTIONS]

Options:
  -o, --output <PATH>          Output file path
      --overwrite              Replace an existing output
      --language <LANGUAGE>    Spoken language
      --model <NAME>           Transcription model
      --format <FORMAT>        Output format: srt, vtt, or txt
      --sidecar <PATH>         Use a sidecar transcript
      --vad / --no-vad        Enable or disable Silero voice activity detection
      --vad-model <NAME|PATH>  VAD model name or explicit model path
      --vad-threshold <0..1>   Speech detection probability threshold
      --vad-min-speech-duration-ms <MS> Minimum retained speech duration
      --vad-min-silence-duration-ms <MS> Minimum silence used to split speech
      --vad-speech-pad-ms <MS> Padding around detected speech
      --no-filter-hallucinations Keep repeated/silence marker segments
      --normalize-transcript / --no-normalize-transcript
                              Enable or disable whitespace/punctuation cleanup
      --speaker-labels / --no-speaker-labels
                              Detect and preserve Whisper speaker prefixes
      --config <PATH>          Configuration file
      --profile <NAME>         Named profile
      --runtime-dir <DIR>      Runtime storage root
      --whisper-bin <PATH>     Override whisper-cli path
      --whisper-models-dir <DIR> Override whisper model directory
      --json                   Emit structured JSON output
      --qa-fail-on <LEVEL>     Block publication on QA error or warning
  -h, --help                   Print help
"#;
const PIPELINE_HELP: &str = r#"Transcribe media when needed, then translate it

Usage: sbake pipeline <MEDIA_OR_SUBTITLE> [OPTIONS]

Accepts the translation settings from `translate` plus:
      --subtitle-stream <INDEX>       Explicit embedded text subtitle stream index
      --preserve-source-container     Write a separate translated media file
      --in-place-container            Atomically replace the source media (default)
      --transcribe-language <LANGUAGE> Spoken language
      --transcribe-model <NAME>        Transcription model
      --transcribe-format <FORMAT>     srt, vtt, or txt
      --sidecar <PATH>                 Use a sidecar transcript
      --transcribe-vad / --no-transcribe-vad Enable or disable Silero VAD
      --transcribe-vad-model <NAME|PATH> VAD model name or path
      --whisper-bin <PATH>             Override whisper-cli path
      --whisper-models-dir <DIR>       Override whisper model directory
      --json                            Emit structured JSON output
      --qa-fail-on <LEVEL>              Block publication on QA error or warning
  -h, --help                           Print help

For MKV, MP4/M4V/MOV, and WebM inputs, an existing text subtitle stream is
translated and added while the other streams are copied. If no translatable
text subtitle exists, the media is transcribed first. Use --subtitle-stream to
select a specific embedded stream.
"#;
const OVERNIGHT_HELP: &str = r#"Submit, check, and collect a provider-managed asynchronous translation batch

Usage:
  sbake overnight submit <SUBTITLE> --mode economy [TRANSLATE OPTIONS]
  sbake overnight status <MANIFEST> [PROVIDER OPTIONS]
  sbake overnight collect <MANIFEST> [PROVIDER OPTIONS] [--overwrite]

`submit` supports OpenAI Batch with `openai_chat` or `openai_responses`.
It saves a non-secret manifest under the subtitle runtime directory. Pass that
manifest path to `status` and `collect`; collection validates that the source
subtitle has not changed before writing the translated output. All actions
accept `--json`.
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
      --json                  Emit structured JSON output
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

Options:
      --json  Emit structured JSON output
"#;
const RUNTIME_CLEAN_HELP: &str = r#"Remove runtime artifacts for a target

Usage: sbake runtime clean <TARGET> --yes [OPTIONS]

Options:
      --runs          Remove run state
      --cache         Remove request and review caches
      --glossary      Remove glossary data
      --all           Remove all managed translation artifacts; preserve the runtime root and unrelated files
      --runtime-dir <DIR>  Override the runtime directory
      --yes           Confirm deletion
      --json          Emit structured JSON output
  -h, --help          Print help

At least one of --runs, --cache, --glossary, or --all is required.
Custom runtime directories must have been created by SubBake.
"#;
const WHISPER_HELP: &str = r#"Install and manage whisper.cpp and its models

Usage: sbake whisper [COMMAND] [OPTIONS]

Commands:
  status              Report installation status (default)
  versions            Fetch whisper.cpp release versions
  install             Install whisper.cpp
  update              Update whisper.cpp
  uninstall           Uninstall whisper.cpp
  model list          List supported models
  model <NAME>        Download a model
  vad-model list      List supported VAD models
  vad-model [NAME]    Download a VAD model (default: silero-v6.2.0)

Options:
      --bin <PATH>         Override the whisper binary path
      --models-dir <DIR>   Override the models directory
      --runtime-dir <DIR>  Runtime storage root
      --variant <VARIANT>  cpu, cuda, metal, vulkan, or openblas
      --config <PATH>      Configuration file
      --profile <NAME>     Named profile
      --keep-models        Keep models when uninstalling
      --json               Emit structured JSON output
  -h, --help               Print help
"#;
const WHISPER_MODEL_HELP: &str = r#"List or download whisper.cpp models

Usage:
  sbake whisper model list
  sbake whisper model <NAME> [--models-dir <DIR>]
"#;
const WHISPER_VAD_MODEL_HELP: &str = r#"List or download whisper.cpp VAD models

Usage:
  sbake whisper vad-model list
  sbake whisper vad-model [NAME] [--models-dir <DIR>]

When NAME is omitted, SubBake downloads silero-v6.2.0.
"#;

const COMPLETION_HELP: &str = r#"Generate a shell completion script

Usage: sbake completion <SHELL>

SHELL is bash, zsh, fish, or powershell. Print the script to standard output,
then source it or install it using your shell's normal completion directory.
"#;

const COMMON_TRANSLATION_OPTIONS: &[&str] = &[
    "--output",
    "--config",
    "--profile",
    "--source-lang",
    "--target-lang",
    "--provider",
    "--model",
    "--mode",
    "--review",
    "--no-review",
    "--dry-run",
    "--json",
    "--qa-fail-on",
    "--help",
];

static COMMAND_SPECS: &[CommandSpec] = &[
    #[cfg(feature = "agent")]
    CommandSpec {
        name: "agent",
        summary: "Start the interactive agent",
        help: AGENT_HELP,
        options: &["--help"],
        subcommands: &[],
    },
    #[cfg(feature = "agent")]
    CommandSpec {
        name: "resume",
        summary: "Resume an agent session",
        help: RESUME_HELP,
        options: &["--help"],
        subcommands: &[],
    },
    CommandSpec {
        name: "translate",
        summary: "Translate a subtitle file",
        help: TRANSLATE_HELP,
        options: COMMON_TRANSLATION_OPTIONS,
        subcommands: &[],
    },
    CommandSpec {
        name: "edit",
        summary: "Safely refine a translated subtitle",
        help: EDIT_HELP,
        options: &[
            "--instruction",
            "--dry-run",
            "--allow-non-generated",
            "--config",
            "--profile",
            "--json",
            "--help",
        ],
        subcommands: &[],
    },
    CommandSpec {
        name: "batch",
        summary: "Translate a directory",
        help: BATCH_HELP,
        options: &[
            "--recursive",
            "--overwrite",
            "--fail-fast",
            "--retry-failed",
            "--config",
            "--profile",
            "--qa-fail-on",
            "--json",
            "--help",
        ],
        subcommands: &[],
    },
    CommandSpec {
        name: "evaluate",
        summary: "Compare against a reference",
        help: EVALUATE_HELP,
        options: &["--json", "--help"],
        subcommands: &[],
    },
    CommandSpec {
        name: "memory",
        summary: "Manage glossary and translation memory",
        help: MEMORY_HELP,
        options: &["--json", "--help"],
        subcommands: &["inspect", "export", "import", "prune"],
    },
    CommandSpec {
        name: "qa",
        summary: "Inspect subtitle quality",
        help: QA_HELP,
        options: &["--json", "--fail-on", "--help"],
        subcommands: &[],
    },
    CommandSpec {
        name: "project",
        summary: "Inspect a subtitle project",
        help: PROJECT_HELP,
        options: &["--recursive", "--output", "--json", "--fail-on", "--help"],
        subcommands: &[],
    },
    CommandSpec {
        name: "transcribe",
        summary: "Transcribe audio or video",
        help: TRANSCRIBE_HELP,
        options: &[
            "--output",
            "--overwrite",
            "--language",
            "--model",
            "--format",
            "--sidecar",
            "--normalize-transcript",
            "--no-normalize-transcript",
            "--speaker-labels",
            "--no-speaker-labels",
            "--qa-fail-on",
            "--json",
            "--help",
        ],
        subcommands: &[],
    },
    CommandSpec {
        name: "pipeline",
        summary: "Transcribe then translate",
        help: PIPELINE_HELP,
        options: COMMON_TRANSLATION_OPTIONS,
        subcommands: &[],
    },
    CommandSpec {
        name: "overnight",
        summary: "Run provider-managed batches",
        help: OVERNIGHT_HELP,
        options: &["--config", "--profile", "--json", "--help"],
        subcommands: &["submit", "status", "collect"],
    },
    CommandSpec {
        name: "provider",
        summary: "Validate a provider",
        help: PROVIDER_HELP,
        options: &["--config", "--profile", "--json", "--help"],
        subcommands: &["check"],
    },
    CommandSpec {
        name: "runtime",
        summary: "Inspect or clean runtime data",
        help: RUNTIME_HELP,
        options: &["--runtime-dir", "--json", "--help"],
        subcommands: &["inspect", "clean"],
    },
    CommandSpec {
        name: "whisper",
        summary: "Manage whisper.cpp",
        help: WHISPER_HELP,
        options: &[
            "--bin",
            "--models-dir",
            "--variant",
            "--config",
            "--profile",
            "--json",
            "--help",
        ],
        subcommands: &[
            "status",
            "versions",
            "install",
            "update",
            "uninstall",
            "model",
            "vad-model",
        ],
    },
    CommandSpec {
        name: "completion",
        summary: "Generate shell completions",
        help: COMPLETION_HELP,
        options: &["--help"],
        subcommands: &["bash", "zsh", "fish", "powershell"],
    },
    CommandSpec {
        name: "help",
        summary: "Print command help",
        help: TOP_LEVEL_HELP,
        options: &[],
        subcommands: &[],
    },
];
