use std::path::PathBuf;

use subbake_adapters::{
    ApiFormat, BackendConfig, ConfigurationResolver, ResolveRequest, RuntimeAction,
    SettingsOverrides, TranscriptionFormat, TranscriptionSettings, TranslationSettings,
    WhisperAction, WhisperBuildVariant, apply_whisper_storage, default_whisper_binary_path_for,
    default_whisper_models_dir_for,
};
use subbake_agent::{AgentAction, AgentActionKind};

use crate::{CliError, CliResult};

#[derive(Debug, Clone)]
pub struct AgentArgs {
    pub action: AgentAction,
}

#[derive(Debug, Clone)]
pub struct TranslateArgs {
    pub input_path: PathBuf,
    pub output: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
    pub profile: Option<String>,
    pub settings: TranslationSettings,
    pub transcription_settings: TranscriptionSettings,
    pub json: bool,
}

#[derive(Debug, Clone, Default)]
pub struct BatchTranslateOptions {
    pub settings: TranslationSettings,
}

#[derive(Debug, Clone)]
pub struct BatchArgs {
    pub dir: PathBuf,
    pub recursive: bool,
    pub overwrite: bool,
    pub config_path: Option<PathBuf>,
    pub profile: Option<String>,
    pub translate: BatchTranslateOptions,
}

#[derive(Debug, Clone)]
pub struct TranscribeArgs {
    pub media_path: PathBuf,
    pub output: Option<PathBuf>,
    pub settings: TranscriptionSettings,
}

#[derive(Debug, Clone)]
pub struct ProviderArgs {
    pub config: BackendConfig,
}

#[derive(Debug, Clone)]
pub struct RuntimeArgs {
    pub action: RuntimeAction,
    pub target_path: PathBuf,
    pub runtime_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct WhisperArgs {
    pub action: WhisperAction,
    pub binary_path: Option<PathBuf>,
    pub models_dir: Option<PathBuf>,
    pub build_variant: WhisperBuildVariant,
}

#[derive(Debug, Clone)]
pub enum OvernightAction {
    Submit(TranslateArgs),
    Status(TranslateArgs),
    Collect {
        args: TranslateArgs,
        overwrite: bool,
    },
}

#[derive(Debug, Clone)]
pub struct OvernightArgs {
    pub action: OvernightAction,
}

#[derive(Debug, Clone)]
pub struct EvaluateArgs {
    pub candidate_path: PathBuf,
    pub reference_path: PathBuf,
    pub json: bool,
}

impl TranslateArgs {
    pub fn default_for(input_path: impl Into<PathBuf>) -> Self {
        Self {
            input_path: input_path.into(),
            output: None,
            config_path: None,
            profile: None,
            settings: TranslationSettings::default(),
            transcription_settings: TranscriptionSettings::default(),
            json: false,
        }
    }
}

pub fn parse_agent_args(args: &[String]) -> CliResult<AgentArgs> {
    if !args.is_empty() {
        return Err(CliError::usage(
            "unsupported agent command; start the agent with `sbake`",
        ));
    }

    let action = AgentAction {
        kind: AgentActionKind::Start,
        session_id: None,
    };

    Ok(AgentArgs { action })
}

pub fn parse_resume_args(args: &[String]) -> CliResult<AgentArgs> {
    if args.len() > 1 {
        return Err(CliError::usage("resume accepts at most one session id"));
    }
    Ok(AgentArgs {
        action: AgentAction {
            kind: AgentActionKind::Resume,
            session_id: args.first().cloned(),
        },
    })
}

pub fn parse_translate_args(args: &[String]) -> CliResult<TranslateArgs> {
    parse_file_translation_args(args, "translate requires a subtitle path", "translate")
}

pub fn parse_pipeline_args(args: &[String]) -> CliResult<TranslateArgs> {
    parse_file_translation_args(args, "pipeline requires an input path", "pipeline")
}

pub fn parse_overnight_args(args: &[String]) -> CliResult<OvernightArgs> {
    let action = args
        .first()
        .map(String::as_str)
        .ok_or_else(|| CliError::usage("overnight requires `submit`, `status`, or `collect`"))?;
    match action {
        "submit" => Ok(OvernightArgs {
            action: OvernightAction::Submit(parse_file_translation_args(
                &args[1..],
                "overnight submit requires a subtitle path",
                "overnight submit",
            )?),
        }),
        "status" => Ok(OvernightArgs {
            action: OvernightAction::Status(parse_file_translation_args(
                &args[1..],
                "overnight status requires a manifest path",
                "overnight status",
            )?),
        }),
        "collect" => {
            let mut filtered = Vec::new();
            let mut overwrite = false;
            for value in &args[1..] {
                if value == "--overwrite" {
                    overwrite = true;
                } else {
                    filtered.push(value.clone());
                }
            }
            Ok(OvernightArgs {
                action: OvernightAction::Collect {
                    args: parse_file_translation_args(
                        &filtered,
                        "overnight collect requires a manifest path",
                        "overnight collect",
                    )?,
                    overwrite,
                },
            })
        }
        other => Err(CliError::usage(format!(
            "unknown overnight command `{other}`; expected submit, status, or collect"
        ))),
    }
}

pub fn parse_evaluate_args(args: &[String]) -> CliResult<EvaluateArgs> {
    let candidate_path = args.first().ok_or_else(|| {
        CliError::usage("evaluate requires a candidate subtitle and a reference subtitle")
    })?;
    let reference_path = args.get(1).ok_or_else(|| {
        CliError::usage("evaluate requires a candidate subtitle and a reference subtitle")
    })?;
    let mut json = false;
    for option in &args[2..] {
        match option.as_str() {
            "--json" => json = true,
            other => {
                return Err(CliError::usage(format!(
                    "unknown evaluate option `{other}`"
                )));
            }
        }
    }
    Ok(EvaluateArgs {
        candidate_path: PathBuf::from(candidate_path),
        reference_path: PathBuf::from(reference_path),
        json,
    })
}

fn parse_file_translation_args(
    args: &[String],
    missing_input_message: &str,
    command_name: &str,
) -> CliResult<TranslateArgs> {
    let input_path = args
        .first()
        .ok_or_else(|| CliError::usage(missing_input_message))?;
    let mut parsed = TranslateArgs::default_for(input_path);

    // First pass: extract --config and --profile (store their values).
    let (explicit_config, explicit_profile) = extract_config_and_profile(args);

    let mut overrides = SettingsOverrides::default();

    // Second pass: all remaining CLI flags override.
    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "-o" | "--output" => parsed.output = Some(required_path(args, &mut index, "--output")?),
            "--config" | "--profile" => {
                // Skip flag + value (already consumed in first pass).
                skip_two(args, &mut index)?;
            }
            "--json" => parsed.json = true,
            value
                if command_name == "pipeline"
                    && parse_pipeline_transcription_option(
                        value,
                        args,
                        &mut index,
                        &mut parsed.transcription_settings,
                    )? => {}
            value if parse_translation_setting_option(value, args, &mut index, &mut overrides)? => {
            }
            other => {
                return Err(CliError::usage(format!(
                    "unknown {command_name} option `{other}`"
                )));
            }
        }
        index += 1;
    }
    let resolved = resolve_settings(explicit_config, explicit_profile, overrides)?;
    if command_name == "pipeline" {
        apply_whisper_storage(
            &mut parsed.transcription_settings,
            &resolved.settings.storage,
        );
    }
    parsed.settings = resolved.settings;
    parsed.config_path = resolved.config_path;
    parsed.profile = resolved.profile;
    Ok(parsed)
}

/// Scan only for `--config` and `--profile`, returning their values.
fn extract_config_and_profile(args: &[String]) -> (Option<PathBuf>, Option<String>) {
    let mut config_path = None;
    let mut profile = None;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--config" if i + 1 < args.len() => {
                config_path = Some(PathBuf::from(args[i + 1].clone()));
                i += 2;
                continue;
            }
            "--profile" if i + 1 < args.len() => {
                profile = Some(args[i + 1].clone());
                i += 2;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    (config_path, profile)
}

/// Skip a flag and its value (used to avoid re-consuming --config/--profile).
fn skip_two(args: &[String], index: &mut usize) -> CliResult<()> {
    if *index + 1 >= args.len() {
        return Err(CliError::usage(format!(
            "{} requires a value",
            args[*index]
        )));
    }
    *index += 1;
    Ok(())
}

pub fn parse_batch_args(args: &[String]) -> CliResult<BatchArgs> {
    let dir = args
        .first()
        .ok_or_else(|| CliError::usage("batch requires a directory"))?;
    let mut parsed = BatchArgs {
        dir: PathBuf::from(dir),
        recursive: false,
        overwrite: false,
        config_path: None,
        profile: None,
        translate: BatchTranslateOptions::default(),
    };

    let (explicit_config, explicit_profile) = extract_config_and_profile(args);
    let mut overrides = SettingsOverrides::default();

    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--recursive" => parsed.recursive = true,
            "--overwrite" => parsed.overwrite = true,
            "--config" | "--profile" => {
                skip_two(args, &mut index)?;
            }
            value if parse_translation_setting_option(value, args, &mut index, &mut overrides)? => {
            }
            other => return Err(CliError::usage(format!("unknown batch option `{other}`"))),
        }
        index += 1;
    }
    let resolved = resolve_settings(explicit_config, explicit_profile, overrides)?;
    parsed.translate.settings = resolved.settings;
    parsed.config_path = resolved.config_path;
    parsed.profile = resolved.profile;

    Ok(parsed)
}

fn resolve_settings(
    explicit_path: Option<PathBuf>,
    profile: Option<String>,
    cli_overrides: SettingsOverrides,
) -> CliResult<subbake_adapters::ResolvedConfiguration> {
    ConfigurationResolver
        .resolve(ResolveRequest {
            explicit_path,
            profile,
            cli_overrides,
            ..ResolveRequest::default()
        })
        .map_err(CliError::from)
}

pub fn parse_transcribe_args(args: &[String]) -> CliResult<TranscribeArgs> {
    let media_path = args
        .first()
        .ok_or_else(|| CliError::usage("transcribe requires a media path"))?;
    let mut parsed = TranscribeArgs {
        media_path: PathBuf::from(media_path),
        output: None,
        settings: TranscriptionSettings::default(),
    };
    let (explicit_config, explicit_profile) = extract_config_and_profile(args);
    let mut overrides = SettingsOverrides::default();

    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "-o" | "--output" => parsed.output = Some(required_path(args, &mut index, "--output")?),
            "--config" | "--profile" => skip_two(args, &mut index)?,
            "--runtime-dir" => {
                overrides.storage.runtime_dir =
                    Some(required_path(args, &mut index, "--runtime-dir")?)
            }
            "--whisper-bin" => {
                overrides.storage.whisper_binary_path =
                    Some(required_path(args, &mut index, "--whisper-bin")?)
            }
            "--whisper-models-dir" => {
                overrides.storage.whisper_models_dir =
                    Some(required_path(args, &mut index, "--whisper-models-dir")?)
            }
            "--language" => {
                parsed.settings.language = Some(required_value(args, &mut index, "--language")?)
            }
            "--model" => parsed.settings.model = Some(required_value(args, &mut index, "--model")?),
            "--sidecar" => {
                parsed.settings.sidecar_path = Some(required_path(args, &mut index, "--sidecar")?)
            }
            "--format" => {
                let value = required_value(args, &mut index, "--format")?;
                parsed.settings.output_format = TranscriptionFormat::parse(&value)
                    .ok_or_else(|| CliError::usage("--format must be one of: srt, vtt, txt"))?;
            }
            other => {
                return Err(CliError::usage(format!(
                    "unknown transcribe option `{other}`"
                )));
            }
        }
        index += 1;
    }

    let resolved = resolve_settings(explicit_config, explicit_profile, overrides)?;
    apply_whisper_storage(&mut parsed.settings, &resolved.settings.storage);
    Ok(parsed)
}

fn parse_pipeline_transcription_option(
    option: &str,
    args: &[String],
    index: &mut usize,
    settings: &mut TranscriptionSettings,
) -> CliResult<bool> {
    match option {
        "--transcribe-language" | "--language" => {
            settings.language = Some(required_value(args, index, option)?)
        }
        "--transcribe-model" | "--transcriber-model" => {
            settings.model = Some(required_value(args, index, option)?)
        }
        "--sidecar" => settings.sidecar_path = Some(required_path(args, index, option)?),
        "--transcribe-format" => {
            let value = required_value(args, index, option)?;
            settings.output_format = TranscriptionFormat::parse(&value).ok_or_else(|| {
                CliError::usage("--transcribe-format must be one of: srt, vtt, txt")
            })?;
        }
        _ => return Ok(false),
    }

    Ok(true)
}

pub fn parse_provider_args(args: &[String]) -> CliResult<ProviderArgs> {
    if args.first().is_none_or(|value| value != "check") {
        return Err(CliError::usage("provider requires `check`"));
    }

    let (explicit_config, explicit_profile) = extract_config_and_profile(args);
    let mut overrides = SettingsOverrides::default();
    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--config" | "--profile" => skip_two(args, &mut index)?,
            "--provider" => {
                overrides.backend.id = Some(required_value(args, &mut index, "--provider")?)
            }
            "--model" => {
                overrides.backend.model = Some(required_value(args, &mut index, "--model")?)
            }
            "--api-key" => {
                overrides.backend.api_key = Some(required_value(args, &mut index, "--api-key")?)
            }
            "--base-url" => {
                overrides.backend.base_url = Some(required_value(args, &mut index, "--base-url")?)
            }
            "--api-format" => {
                overrides.backend.api_format = Some(
                    ApiFormat::parse(&required_value(args, &mut index, "--api-format")?)
                        .map_err(|error| CliError::usage(error.to_string()))?,
                )
            }
            "--endpoint-url" => {
                overrides.backend.endpoint_url =
                    Some(required_value(args, &mut index, "--endpoint-url")?)
            }
            "--api-key-env" => {
                overrides.backend.api_key_env =
                    Some(required_value(args, &mut index, "--api-key-env")?)
            }
            "--auth-header" => {
                overrides.backend.auth_header =
                    Some(required_value(args, &mut index, "--auth-header")?)
            }
            "--auth-prefix" => {
                overrides.backend.auth_prefix =
                    Some(required_value(args, &mut index, "--auth-prefix")?)
            }
            other => {
                return Err(CliError::usage(format!(
                    "unknown provider option `{other}`"
                )));
            }
        }
        index += 1;
    }
    let resolved = resolve_settings(explicit_config, explicit_profile, overrides)?;

    Ok(ProviderArgs {
        config: resolved.settings.backend_config(),
    })
}

pub fn parse_runtime_args(args: &[String]) -> CliResult<RuntimeArgs> {
    let command = args
        .first()
        .ok_or_else(|| CliError::usage("runtime requires `inspect` or `clean`"))?;
    let target_path = args
        .get(1)
        .ok_or_else(|| CliError::usage(format!("runtime {command} requires a target")))?;
    let mut runtime_dir = None;
    let mut yes = false;
    let mut clean_runs = false;
    let mut clean_cache = false;
    let mut clean_glossary = false;
    let mut clean_all = false;
    let mut index = 2usize;
    while index < args.len() {
        match args[index].as_str() {
            "--runtime-dir" => {
                runtime_dir = Some(required_path(args, &mut index, "--runtime-dir")?)
            }
            "--yes" if command == "clean" => yes = true,
            "--runs" if command == "clean" => clean_runs = true,
            "--cache" if command == "clean" => clean_cache = true,
            "--glossary" if command == "clean" => clean_glossary = true,
            "--all" if command == "clean" => clean_all = true,
            other => {
                return Err(CliError::usage(format!("unknown runtime option `{other}`")));
            }
        }
        index += 1;
    }

    let action = match command.as_str() {
        "inspect" => RuntimeAction::Inspect,
        "clean" => RuntimeAction::Clean {
            yes,
            clean_runs,
            clean_cache,
            clean_glossary,
            all: clean_all,
        },
        _ => return Err(CliError::usage("runtime requires `inspect` or `clean`")),
    };

    Ok(RuntimeArgs {
        action,
        target_path: PathBuf::from(target_path),
        runtime_dir,
    })
}

pub fn parse_whisper_args(args: &[String]) -> CliResult<WhisperArgs> {
    let (explicit_config, explicit_profile) = extract_config_and_profile(args);
    let command = args.first().map(String::as_str).unwrap_or("status");
    let (action, mut index) = match command {
        "status" => (WhisperAction::Status, 1usize),
        "versions" | "list-versions" => (WhisperAction::ListVersions, 1usize),
        "install" => (WhisperAction::Install, 1usize),
        "update" => (WhisperAction::Update, 1usize),
        "uninstall" => (WhisperAction::Uninstall { keep_models: false }, 1usize),
        "models" | "list-models" => (WhisperAction::ListModels, 1usize),
        "model" if args.get(1).is_some_and(|value| value == "list") => {
            (WhisperAction::ListModels, 2usize)
        }
        "model" | "download-model" => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| CliError::usage("whisper model requires a model name"))?;
            (WhisperAction::DownloadModel { name }, 2usize)
        }
        other => {
            return Err(CliError::usage(format!(
                "unknown whisper command `{other}`"
            )));
        }
    };
    let mut parsed = WhisperArgs {
        action,
        binary_path: None,
        models_dir: None,
        build_variant: WhisperBuildVariant::Cpu,
    };
    let mut overrides = SettingsOverrides::default();

    while index < args.len() {
        match args[index].as_str() {
            "--bin" => parsed.binary_path = Some(required_path(args, &mut index, "--bin")?),
            "--models-dir" => {
                parsed.models_dir = Some(required_path(args, &mut index, "--models-dir")?)
            }
            "--keep-models" => {
                parsed.action = WhisperAction::Uninstall { keep_models: true };
            }
            "--variant" => {
                let value = required_value(args, &mut index, "--variant")?;
                parsed.build_variant = WhisperBuildVariant::parse(&value).ok_or_else(|| {
                    CliError::usage("--variant must be one of: cpu, cuda, metal, vulkan, openblas")
                })?;
            }
            "--runtime-dir" => {
                overrides.storage.runtime_dir =
                    Some(required_path(args, &mut index, "--runtime-dir")?)
            }
            "--config" | "--profile" => skip_two(args, &mut index)?,
            other => {
                return Err(CliError::usage(format!("unknown whisper option `{other}`")));
            }
        }
        index += 1;
    }

    let resolved = resolve_settings(explicit_config, explicit_profile, overrides)?;
    if parsed.binary_path.is_none() {
        parsed.binary_path = Some(
            resolved
                .settings
                .storage
                .whisper_binary_path
                .clone()
                .unwrap_or_else(|| {
                    default_whisper_binary_path_for(
                        resolved.settings.storage.runtime_dir.as_deref(),
                    )
                }),
        );
    }
    if parsed.models_dir.is_none() {
        parsed.models_dir = Some(
            resolved
                .settings
                .storage
                .whisper_models_dir
                .clone()
                .unwrap_or_else(|| {
                    default_whisper_models_dir_for(resolved.settings.storage.runtime_dir.as_deref())
                }),
        );
    }
    Ok(parsed)
}

fn parse_translation_setting_option(
    option: &str,
    args: &[String],
    index: &mut usize,
    overrides: &mut SettingsOverrides,
) -> CliResult<bool> {
    match option {
        "--output-format" => overrides.output.format = Some(required_value(args, index, option)?),
        "--provider" => overrides.backend.id = Some(required_value(args, index, option)?),
        "--model" => overrides.backend.model = Some(required_value(args, index, option)?),
        "--api-key" => overrides.backend.api_key = Some(required_value(args, index, option)?),
        "--base-url" => overrides.backend.base_url = Some(required_value(args, index, option)?),
        "--api-format" => {
            overrides.backend.api_format = Some(
                ApiFormat::parse(&required_value(args, index, option)?)
                    .map_err(|error| CliError::usage(error.to_string()))?,
            )
        }
        "--endpoint-url" => {
            overrides.backend.endpoint_url = Some(required_value(args, index, option)?)
        }
        "--api-key-env" => {
            overrides.backend.api_key_env = Some(required_value(args, index, option)?)
        }
        "--auth-header" => {
            overrides.backend.auth_header = Some(required_value(args, index, option)?)
        }
        "--auth-prefix" => {
            overrides.backend.auth_prefix = Some(required_value(args, index, option)?)
        }
        "--source-lang" => {
            overrides.translation.source_language = Some(required_value(args, index, option)?)
        }
        "--target-lang" => {
            overrides.translation.target_language = Some(required_value(args, index, option)?)
        }
        "--batch-size" => {
            overrides.translation.batch_size = Some(parse_batch_size(args, index, option)?)
        }
        "--batch-token-budget" => {
            overrides.translation.batch_token_budget = Some(parse_batch_size(args, index, option)?)
        }
        "--translation-concurrency" => {
            overrides.translation.translation_concurrency =
                Some(parse_batch_size(args, index, option)?)
        }
        "--review-concurrency" => {
            overrides.translation.review_concurrency = Some(parse_batch_size(args, index, option)?)
        }
        "--runtime-dir" => {
            overrides.storage.runtime_dir = Some(required_path(args, index, option)?)
        }
        "--whisper-bin" => {
            overrides.storage.whisper_binary_path = Some(required_path(args, index, option)?)
        }
        "--whisper-models-dir" => {
            overrides.storage.whisper_models_dir = Some(required_path(args, index, option)?)
        }
        "--glossary" => overrides.storage.glossary_path = Some(required_path(args, index, option)?),
        "--bilingual" => overrides.output.bilingual = Some(true),
        "--mode" => {
            overrides.translation.mode = Some(
                subbake_core::TranslationMode::parse(&required_value(args, index, option)?)
                    .map_err(|error| CliError::usage(error.to_string()))?,
            )
        }
        "--fast" => overrides.translation.fast_mode = Some(true),
        "--no-review" => {
            overrides.translation.review_policy = Some(subbake_core::ReviewPolicy::Off)
        }
        "--review" => {
            overrides.translation.review_policy = Some(
                subbake_core::ReviewPolicy::parse(&required_value(args, index, option)?)
                    .map_err(|error| CliError::usage(error.to_string()))?,
            )
        }
        "--dry-run" => overrides.translation.dry_run = Some(true),
        "--resume" => overrides.translation.resume = Some(true),
        "--no-resume" => overrides.translation.resume = Some(false),
        "--cache" => overrides.translation.use_cache = Some(true),
        "--no-cache" => overrides.translation.use_cache = Some(false),
        "--retries" => {
            overrides.translation.retries = Some(parse_nonnegative_usize(args, index, option)?)
        }
        "--agent" => overrides.translation.agent = Some(true),
        "--no-agent" => overrides.translation.agent = Some(false),
        "--agent-repair-attempts" => {
            overrides.translation.agent_repair_attempts =
                Some(parse_nonnegative_usize(args, index, option)?)
        }
        _ => return Ok(false),
    }

    Ok(true)
}

pub(crate) fn required_value(args: &[String], index: &mut usize, flag: &str) -> CliResult<String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| CliError::usage(format!("{flag} requires a value")))
}

fn required_path(args: &[String], index: &mut usize, flag: &str) -> CliResult<PathBuf> {
    required_value(args, index, flag).map(PathBuf::from)
}

fn parse_batch_size(args: &[String], index: &mut usize, flag: &str) -> CliResult<usize> {
    let raw = required_value(args, index, flag)?;
    let value = raw
        .parse::<usize>()
        .map_err(|_| CliError::usage(format!("{flag} must be a positive integer")))?;
    if value == 0 {
        return Err(CliError::usage(format!("{flag} must be greater than zero")));
    }
    Ok(value)
}

fn parse_nonnegative_usize(args: &[String], index: &mut usize, flag: &str) -> CliResult<usize> {
    required_value(args, index, flag)?
        .parse::<usize>()
        .map_err(|_| CliError::usage(format!("{flag} must be a non-negative integer")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_config(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "subbake-test-{}-{label}-empty.toml",
            std::process::id()
        ));
        std::fs::write(&path, "version = 1\n").expect("write empty config");
        path
    }

    #[test]
    fn parse_translate_rejects_zero_batch_size() {
        let args = vec![
            "clip.srt".to_owned(),
            "--batch-size".to_owned(),
            "0".to_owned(),
        ];
        let error = parse_translate_args(&args).expect_err("zero batch size should fail");
        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn parse_translate_accepts_resume_and_cache_switches() {
        let config = empty_config("translate-switches");
        let args = vec![
            "clip.srt".to_owned(),
            "--config".to_owned(),
            config.to_string_lossy().into_owned(),
            "--no-resume".to_owned(),
            "--no-cache".to_owned(),
            "--retries".to_owned(),
            "0".to_owned(),
            "--no-agent".to_owned(),
            "--agent-repair-attempts".to_owned(),
            "3".to_owned(),
        ];
        let parsed = parse_translate_args(&args).expect("translate args should parse");
        let _ = std::fs::remove_file(config);

        assert!(!parsed.settings.translation.resume);
        assert!(!parsed.settings.translation.use_cache);
        assert_eq!(parsed.settings.translation.retries, 0);
        assert!(!parsed.settings.translation.agent);
        assert_eq!(parsed.settings.translation.agent_repair_attempts, 3);
    }

    #[test]
    fn parse_translate_accepts_review_and_concurrency_options() {
        let config = empty_config("translation-options");
        let args = vec![
            "movie.srt".to_owned(),
            "--config".to_owned(),
            config.to_string_lossy().into_owned(),
            "--review".to_owned(),
            "full".to_owned(),
            "--translation-concurrency".to_owned(),
            "3".to_owned(),
            "--review-concurrency".to_owned(),
            "2".to_owned(),
            "--batch-token-budget".to_owned(),
            "1800".to_owned(),
        ];
        let parsed = parse_translate_args(&args).expect("translation options");
        let _ = std::fs::remove_file(config);
        assert_eq!(
            parsed.settings.translation.review_policy,
            subbake_core::ReviewPolicy::Full
        );
        assert_eq!(parsed.settings.translation.translation_concurrency, 3);
        assert_eq!(parsed.settings.translation.review_concurrency, 2);
        assert_eq!(parsed.settings.translation.batch_token_budget, 1_800);
    }

    #[test]
    fn parse_translate_mode_applies_mode_policy() {
        let config = empty_config("translation-mode");
        let args = vec![
            "movie.srt".to_owned(),
            "--config".to_owned(),
            config.to_string_lossy().into_owned(),
            "--mode".to_owned(),
            "cinema".to_owned(),
        ];
        let parsed = parse_translate_args(&args).expect("translation mode");
        let _ = std::fs::remove_file(config);
        assert_eq!(
            parsed.settings.translation.mode,
            subbake_core::TranslationMode::Cinema
        );
        assert_eq!(
            parsed.settings.translation.review_policy,
            subbake_core::ReviewPolicy::Full
        );
    }

    #[test]
    fn parse_resume_accepts_optional_session() {
        let args = vec!["abc".to_owned()];

        let parsed = parse_resume_args(&args).expect("resume should parse");

        assert_eq!(
            parsed.action,
            AgentAction {
                kind: AgentActionKind::Resume,
                session_id: Some("abc".to_owned()),
            }
        );
    }

    #[test]
    fn parse_batch_reuses_translation_options() {
        let config = empty_config("batch-options");
        let args = vec![
            "season".to_owned(),
            "--config".to_owned(),
            config.to_string_lossy().into_owned(),
            "--recursive".to_owned(),
            "--bilingual".to_owned(),
        ];
        let parsed = parse_batch_args(&args).expect("batch args should parse");
        let _ = std::fs::remove_file(config);

        assert!(parsed.recursive);
        assert!(parsed.translate.settings.output.bilingual);
    }

    #[test]
    fn parse_translate_reports_config_errors_with_path() {
        let path = std::env::temp_dir().join(format!(
            "subbake-test-{}-translate-invalid.toml",
            std::process::id()
        ));
        std::fs::write(
            &path,
            "version = 1\n[defaults.translation]\nbatch_size = \"nope\"\n",
        )
        .expect("write config");
        let args = vec![
            "clip.srt".to_owned(),
            "--config".to_owned(),
            path.to_string_lossy().into_owned(),
        ];

        let error = parse_translate_args(&args).expect_err("invalid config should fail");
        let _ = std::fs::remove_file(&path);

        assert!(error.to_string().contains(&path.display().to_string()));
        assert!(error.to_string().contains("failed to load config"));
    }

    #[test]
    fn parse_batch_reports_config_errors_with_path() {
        let path = std::env::temp_dir().join(format!(
            "subbake-test-{}-batch-invalid.toml",
            std::process::id()
        ));
        std::fs::write(
            &path,
            "version = 1\n[defaults.translation]\nunknown_setting = true\n",
        )
        .expect("write config");
        let args = vec![
            "season".to_owned(),
            "--config".to_owned(),
            path.to_string_lossy().into_owned(),
        ];

        let error = parse_batch_args(&args).expect_err("invalid config should fail");
        let _ = std::fs::remove_file(&path);

        assert!(error.to_string().contains(&path.display().to_string()));
        assert!(error.to_string().contains("failed to load config"));
    }

    #[test]
    fn explicit_missing_config_is_an_error() {
        let path =
            std::env::temp_dir().join(format!("subbake-test-{}-missing.toml", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let args = vec![
            "clip.srt".to_owned(),
            "--config".to_owned(),
            path.to_string_lossy().into_owned(),
        ];

        let error = parse_translate_args(&args).expect_err("explicit config must exist");
        assert!(error.to_string().contains("configuration"));
    }

    #[test]
    fn parse_batch_resolves_profile_config() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "subbake-test-{}-{}.toml",
            std::process::id(),
            "batch-profile"
        ));
        std::fs::write(
            &path,
            r#"
            version = 1

            [defaults.translation]
            target_language = "Japanese"

            [profiles.zh.translation]
            target_language = "Chinese"
            batch_size = 7
            "#,
        )
        .expect("write config");

        let args = vec![
            "season".to_owned(),
            "--config".to_owned(),
            path.to_string_lossy().to_string(),
            "--profile".to_owned(),
            "zh".to_owned(),
        ];
        let parsed = parse_batch_args(&args).expect("batch args should parse");
        let _ = std::fs::remove_file(&path);

        assert_eq!(parsed.profile.as_deref(), Some("zh"));
        assert_eq!(
            parsed.translate.settings.translation.target_language,
            "Chinese"
        );
        assert_eq!(parsed.translate.settings.translation.batch_size, 7);
    }

    #[test]
    fn cli_values_override_config_defaults() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "subbake-test-{}-{}.toml",
            std::process::id(),
            "cli-override"
        ));
        std::fs::write(
            &path,
            r#"
            version = 1

            [defaults.translation]
            target_language = "Japanese"
            batch_size = 9

            [defaults.output]
            bilingual = true
            "#,
        )
        .expect("write config");

        let args = vec![
            "clip.srt".to_owned(),
            "--config".to_owned(),
            path.to_string_lossy().to_string(),
            "--target-lang".to_owned(),
            "English".to_owned(),
        ];
        let parsed = parse_translate_args(&args).expect("translate args should parse");
        let _ = std::fs::remove_file(&path);

        assert_eq!(parsed.settings.translation.target_language, "English");
        assert_eq!(parsed.settings.translation.batch_size, 9);
        assert!(parsed.settings.output.bilingual);
    }

    #[test]
    fn parse_pipeline_reuses_file_translation_options() {
        let config = empty_config("pipeline-options");
        let args = vec![
            "movie.srt".to_owned(),
            "--config".to_owned(),
            config.to_string_lossy().into_owned(),
            "--output".to_owned(),
            "movie.zh.srt".to_owned(),
            "--json".to_owned(),
            "--no-review".to_owned(),
            "--transcribe-model".to_owned(),
            "base".to_owned(),
            "--language".to_owned(),
            "en".to_owned(),
        ];

        let parsed = parse_pipeline_args(&args).expect("pipeline args should parse");
        let _ = std::fs::remove_file(config);

        assert_eq!(parsed.input_path, PathBuf::from("movie.srt"));
        assert_eq!(parsed.output, Some(PathBuf::from("movie.zh.srt")));
        assert!(parsed.json);
        assert_eq!(
            parsed.settings.translation.review_policy,
            subbake_core::ReviewPolicy::Off
        );
        assert_eq!(parsed.transcription_settings.model.as_deref(), Some("base"));
        assert_eq!(
            parsed.transcription_settings.language.as_deref(),
            Some("en")
        );
    }

    #[test]
    fn parse_transcribe_accepts_local_whisper_options() {
        let args = vec![
            "movie.mp4".to_owned(),
            "--language".to_owned(),
            "en".to_owned(),
            "--model".to_owned(),
            "base".to_owned(),
            "--format".to_owned(),
            "vtt".to_owned(),
            "--sidecar".to_owned(),
            "movie.srt".to_owned(),
        ];

        let parsed = parse_transcribe_args(&args).expect("transcribe args should parse");

        assert_eq!(parsed.media_path, PathBuf::from("movie.mp4"));
        assert_eq!(parsed.settings.language.as_deref(), Some("en"));
        assert_eq!(parsed.settings.model.as_deref(), Some("base"));
        assert_eq!(parsed.settings.output_format, TranscriptionFormat::Vtt);
        assert_eq!(
            parsed.settings.sidecar_path,
            Some(PathBuf::from("movie.srt"))
        );
    }

    #[test]
    fn parse_provider_check_defaults_to_mock() {
        let config = empty_config("provider-default");
        let args = vec![
            "check".to_owned(),
            "--config".to_owned(),
            config.to_string_lossy().into_owned(),
        ];

        let parsed = parse_provider_args(&args).expect("provider check should parse");
        let _ = std::fs::remove_file(config);

        assert_eq!(parsed.config, BackendConfig::new("mock", "mock-zh"));
    }

    #[test]
    fn parse_provider_check_accepts_api_key_and_base_url() {
        let config = empty_config("provider-options");
        let args = vec![
            "check".to_owned(),
            "--config".to_owned(),
            config.to_string_lossy().into_owned(),
            "--provider".to_owned(),
            "openai".to_owned(),
            "--model".to_owned(),
            "gpt".to_owned(),
            "--api-format".to_owned(),
            "openai_chat".to_owned(),
            "--api-key".to_owned(),
            "sk-test".to_owned(),
            "--base-url".to_owned(),
            "https://example.test/v1".to_owned(),
        ];

        let parsed = parse_provider_args(&args).expect("provider check should parse");
        let _ = std::fs::remove_file(config);

        assert_eq!(parsed.config.id, "openai");
        assert_eq!(parsed.config.api_key.as_deref(), Some("sk-test"));
        assert_eq!(
            parsed.config.base_url.as_deref(),
            Some("https://example.test/v1")
        );
    }

    #[test]
    fn provider_check_uses_profile_then_cli_backend_overrides() {
        let path = std::env::temp_dir().join(format!(
            "subbake-test-{}-provider-profile.toml",
            std::process::id()
        ));
        std::fs::write(
            &path,
            r#"
            version = 1

            [profiles.remote.backend]
            id = "openai"
            model = "profile-model"
            api_format = "openai_chat"
            base_url = "https://profile.test/v1"
            "#,
        )
        .expect("write provider profile");
        let args = vec![
            "check".to_owned(),
            "--config".to_owned(),
            path.to_string_lossy().into_owned(),
            "--profile".to_owned(),
            "remote".to_owned(),
            "--model".to_owned(),
            "cli-model".to_owned(),
        ];

        let parsed = parse_provider_args(&args).expect("provider profile should resolve");
        let _ = std::fs::remove_file(path);

        assert_eq!(parsed.config.id, "openai");
        assert_eq!(parsed.config.model, "cli-model");
        assert_eq!(
            parsed.config.base_url.as_deref(),
            Some("https://profile.test/v1")
        );
        assert_eq!(parsed.config.api_format, Some(ApiFormat::OpenaiChat));
    }

    #[test]
    fn parse_runtime_clean_requires_explicit_action() {
        let args = vec![
            "clean".to_owned(),
            "clip.srt".to_owned(),
            "--yes".to_owned(),
            "--runtime-dir".to_owned(),
            ".subbake".to_owned(),
        ];

        let parsed = parse_runtime_args(&args).expect("runtime args should parse");

        assert_eq!(
            parsed.action,
            RuntimeAction::Clean {
                yes: true,
                clean_runs: false,
                clean_cache: false,
                clean_glossary: false,
                all: false,
            }
        );
        assert_eq!(parsed.target_path, PathBuf::from("clip.srt"));
        assert_eq!(parsed.runtime_dir, Some(PathBuf::from(".subbake")));
    }

    #[test]
    fn parse_whisper_model_accepts_paths() {
        let args = vec![
            "model".to_owned(),
            "base".to_owned(),
            "--bin".to_owned(),
            "tools/whisper-cli".to_owned(),
            "--models-dir".to_owned(),
            "models".to_owned(),
        ];

        let parsed = parse_whisper_args(&args).expect("whisper args should parse");

        assert_eq!(
            parsed.action,
            WhisperAction::DownloadModel {
                name: "base".to_owned()
            }
        );
        assert_eq!(parsed.binary_path, Some(PathBuf::from("tools/whisper-cli")));
        assert_eq!(parsed.models_dir, Some(PathBuf::from("models")));
    }

    #[test]
    fn parse_whisper_model_list_accepts_models_dir() {
        let args = vec![
            "model".to_owned(),
            "list".to_owned(),
            "--models-dir".to_owned(),
            "models".to_owned(),
        ];

        let parsed = parse_whisper_args(&args).expect("whisper args should parse");

        assert_eq!(parsed.action, WhisperAction::ListModels);
        assert_eq!(parsed.models_dir, Some(PathBuf::from("models")));
    }

    #[test]
    fn parse_whisper_uninstall_accepts_keep_models() {
        let args = vec!["uninstall".to_owned(), "--keep-models".to_owned()];

        let parsed = parse_whisper_args(&args).expect("whisper args should parse");

        assert_eq!(
            parsed.action,
            WhisperAction::Uninstall { keep_models: true }
        );
    }

    #[test]
    fn parse_whisper_update_is_supported() {
        let args = vec!["update".to_owned()];

        let parsed = parse_whisper_args(&args).expect("whisper args should parse");

        assert_eq!(parsed.action, WhisperAction::Update);
    }

    #[test]
    fn parse_whisper_versions_is_supported() {
        let parsed =
            parse_whisper_args(&["versions".to_owned()]).expect("whisper versions should parse");

        assert_eq!(parsed.action, WhisperAction::ListVersions);
    }

    #[test]
    fn whisper_paths_resolve_consistently_from_runtime_configuration() {
        let path = std::env::temp_dir().join(format!(
            "subbake-test-{}-whisper-storage.toml",
            std::process::id()
        ));
        std::fs::write(
            &path,
            "version = 2\n[defaults.storage]\nruntime_dir = \"runtime-data\"\n",
        )
        .expect("write config");
        let config = path.to_string_lossy().into_owned();

        let transcribe = parse_transcribe_args(&[
            "audio.wav".to_owned(),
            "--config".to_owned(),
            config.clone(),
        ])
        .expect("parse transcribe");
        let pipeline = parse_pipeline_args(&[
            "video.mp4".to_owned(),
            "--config".to_owned(),
            config.clone(),
        ])
        .expect("parse pipeline");
        let whisper = parse_whisper_args(&["status".to_owned(), "--config".to_owned(), config])
            .expect("parse whisper");
        let _ = std::fs::remove_file(path);

        let binary = PathBuf::from("runtime-data/whisper/bin").join(if cfg!(windows) {
            "whisper-cli.exe"
        } else {
            "whisper-cli"
        });
        let models = PathBuf::from("runtime-data/whisper/models");
        assert_eq!(
            transcribe.settings.whisper_binary_path,
            Some(binary.clone())
        );
        assert_eq!(
            pipeline.transcription_settings.whisper_binary_path,
            Some(binary.clone())
        );
        assert_eq!(whisper.binary_path, Some(binary));
        assert_eq!(transcribe.settings.whisper_models_dir, Some(models.clone()));
        assert_eq!(
            pipeline.transcription_settings.whisper_models_dir,
            Some(models.clone())
        );
        assert_eq!(whisper.models_dir, Some(models));
    }
}
