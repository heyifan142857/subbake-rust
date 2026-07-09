use std::io;
use std::path::PathBuf;

use subbake_adapters::{
    BackendConfig, RuntimeAction, TranscriptionFormat, TranscriptionSettings, TranslationSettings,
    WhisperAction, default_api_key_env, discover_config_path, load_and_resolve, resolve_env_var,
};
use subbake_agent::AgentAction;

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

pub fn parse_agent_args(args: &[String]) -> io::Result<AgentArgs> {
    let action = match args.first().map(String::as_str) {
        None => AgentAction {
            kind: "start".to_owned(),
            session_id: None,
        },
        Some("resume") => {
            if args.len() > 2 {
                return Err(io::Error::other(
                    "agent resume accepts at most one session id",
                ));
            }
            AgentAction {
                kind: "resume".to_owned(),
                session_id: args.get(1).cloned(),
            }
        }
        Some(_) => {
            return Err(io::Error::other(
                "unsupported agent command; use `agent resume [SESSION_ID]`",
            ));
        }
    };

    Ok(AgentArgs { action })
}

pub fn parse_translate_args(args: &[String]) -> io::Result<TranslateArgs> {
    parse_file_translation_args(args, "translate requires a subtitle path", "translate")
}

pub fn parse_pipeline_args(args: &[String]) -> io::Result<TranslateArgs> {
    parse_file_translation_args(args, "pipeline requires an input path", "pipeline")
}

fn parse_file_translation_args(
    args: &[String],
    missing_input_message: &str,
    command_name: &str,
) -> io::Result<TranslateArgs> {
    let input_path = args
        .first()
        .ok_or_else(|| io::Error::other(missing_input_message))?;
    let mut parsed = TranslateArgs::default_for(input_path);

    // First pass: extract --config and --profile (store their values).
    let (explicit_config, explicit_profile) = extract_config_and_profile(args);

    // Discover config file if none given via --config.
    let cfg_file = explicit_config.clone().unwrap_or_else(|| {
        discover_config_path().unwrap_or_else(|| PathBuf::from(".subbake.toml"))
    });

    // Load config + resolve profile as the baseline.
    if let Ok(Some(patch)) = load_and_resolve(&cfg_file, explicit_profile.as_deref()) {
        parsed.settings.apply_patch(patch);
    }
    if cfg_file.exists() {
        parsed.config_path = Some(cfg_file);
    }
    parsed.profile = explicit_profile;

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
            value
                if parse_translation_setting_option(
                    value,
                    args,
                    &mut index,
                    &mut parsed.settings,
                )? => {}
            other => {
                return Err(io::Error::other(format!(
                    "unknown {command_name} option `{other}`"
                )));
            }
        }
        index += 1;
    }
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
fn skip_two(args: &[String], index: &mut usize) -> io::Result<()> {
    if *index + 1 >= args.len() {
        return Err(io::Error::other(format!(
            "{} requires a value",
            args[*index]
        )));
    }
    *index += 1;
    Ok(())
}

pub fn parse_batch_args(args: &[String]) -> io::Result<BatchArgs> {
    let dir = args
        .first()
        .ok_or_else(|| io::Error::other("batch requires a directory"))?;
    let mut parsed = BatchArgs {
        dir: PathBuf::from(dir),
        recursive: false,
        overwrite: false,
        config_path: None,
        profile: None,
        translate: BatchTranslateOptions::default(),
    };

    let (explicit_config, explicit_profile) = extract_config_and_profile(args);
    let cfg_file = explicit_config.clone().unwrap_or_else(|| {
        discover_config_path().unwrap_or_else(|| PathBuf::from(".subbake.toml"))
    });
    if let Ok(Some(patch)) = load_and_resolve(&cfg_file, explicit_profile.as_deref()) {
        parsed.translate.settings.apply_patch(patch);
    }
    if cfg_file.exists() {
        parsed.config_path = Some(cfg_file);
    }
    parsed.profile = explicit_profile;

    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--recursive" => parsed.recursive = true,
            "--overwrite" => parsed.overwrite = true,
            "--config" | "--profile" => {
                skip_two(args, &mut index)?;
            }
            value
                if parse_translation_setting_option(
                    value,
                    args,
                    &mut index,
                    &mut parsed.translate.settings,
                )? => {}
            other => return Err(io::Error::other(format!("unknown batch option `{other}`"))),
        }
        index += 1;
    }

    Ok(parsed)
}

pub fn parse_transcribe_args(args: &[String]) -> io::Result<TranscribeArgs> {
    let media_path = args
        .first()
        .ok_or_else(|| io::Error::other("transcribe requires a media path"))?;
    let mut parsed = TranscribeArgs {
        media_path: PathBuf::from(media_path),
        output: None,
        settings: TranscriptionSettings::default(),
    };

    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "-o" | "--output" => parsed.output = Some(required_path(args, &mut index, "--output")?),
            "--language" => {
                parsed.settings.language = Some(required_value(args, &mut index, "--language")?)
            }
            "--provider" | "--transcriber" => {
                let flag = args[index].clone();
                parsed.settings.provider = required_value(args, &mut index, &flag)?
            }
            "--model" => parsed.settings.model = Some(required_value(args, &mut index, "--model")?),
            "--api-key" => {
                parsed.settings.api_key = Some(required_value(args, &mut index, "--api-key")?)
            }
            "--base-url" => {
                parsed.settings.base_url = Some(required_value(args, &mut index, "--base-url")?)
            }
            "--sidecar" => {
                parsed.settings.sidecar_path = Some(required_path(args, &mut index, "--sidecar")?)
            }
            "--format" => {
                let value = required_value(args, &mut index, "--format")?;
                parsed.settings.output_format = TranscriptionFormat::parse(&value)
                    .ok_or_else(|| io::Error::other("--format must be one of: srt, vtt, txt"))?;
            }
            other => {
                return Err(io::Error::other(format!(
                    "unknown transcribe option `{other}`"
                )));
            }
        }
        index += 1;
    }

    Ok(parsed)
}

fn parse_pipeline_transcription_option(
    option: &str,
    args: &[String],
    index: &mut usize,
    settings: &mut TranscriptionSettings,
) -> io::Result<bool> {
    match option {
        "--transcriber" | "--transcriber-provider" => {
            settings.provider = required_value(args, index, option)?
        }
        "--transcribe-language" | "--language" => {
            settings.language = Some(required_value(args, index, option)?)
        }
        "--transcribe-model" | "--transcriber-model" => {
            settings.model = Some(required_value(args, index, option)?)
        }
        "--transcribe-api-key" => settings.api_key = Some(required_value(args, index, option)?),
        "--transcribe-base-url" => settings.base_url = Some(required_value(args, index, option)?),
        "--sidecar" => settings.sidecar_path = Some(required_path(args, index, option)?),
        "--transcribe-format" => {
            let value = required_value(args, index, option)?;
            settings.output_format = TranscriptionFormat::parse(&value).ok_or_else(|| {
                io::Error::other("--transcribe-format must be one of: srt, vtt, txt")
            })?;
        }
        _ => return Ok(false),
    }

    Ok(true)
}

pub fn parse_provider_args(args: &[String]) -> io::Result<ProviderArgs> {
    if args.first().is_none_or(|value| value != "check") {
        return Err(io::Error::other("provider requires `check`"));
    }

    let mut provider = "mock".to_owned();
    let mut model = "mock-zh".to_owned();
    let mut api_key = None;
    let mut base_url = None;
    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--provider" => provider = required_value(args, &mut index, "--provider")?,
            "--model" => model = required_value(args, &mut index, "--model")?,
            "--api-key" => api_key = Some(required_value(args, &mut index, "--api-key")?),
            "--base-url" => base_url = Some(required_value(args, &mut index, "--base-url")?),
            other => {
                return Err(io::Error::other(format!(
                    "unknown provider option `{other}`"
                )));
            }
        }
        index += 1;
    }

    // Resolve API key: explicit value > provider-default env var
    let resolved_api_key = api_key.or_else(|| resolve_env_var(default_api_key_env(&provider)));

    Ok(ProviderArgs {
        config: BackendConfig {
            provider,
            model,
            api_key: resolved_api_key,
            base_url,
        },
    })
}

pub fn parse_runtime_args(args: &[String]) -> io::Result<RuntimeArgs> {
    let command = args
        .first()
        .ok_or_else(|| io::Error::other("runtime requires `inspect` or `clean`"))?;
    let target_path = args
        .get(1)
        .ok_or_else(|| io::Error::other(format!("runtime {command} requires a target")))?;
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
                return Err(io::Error::other(format!(
                    "unknown runtime option `{other}`"
                )));
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
        _ => return Err(io::Error::other("runtime requires `inspect` or `clean`")),
    };

    Ok(RuntimeArgs {
        action,
        target_path: PathBuf::from(target_path),
        runtime_dir,
    })
}

pub fn parse_whisper_args(args: &[String]) -> io::Result<WhisperArgs> {
    let command = args.first().map(String::as_str).unwrap_or("status");
    let (action, mut index) = match command {
        "status" => (WhisperAction::Status, 1usize),
        "install" => (WhisperAction::Install, 1usize),
        "models" | "list-models" => (WhisperAction::ListModels, 1usize),
        "model" if args.get(1).is_some_and(|value| value == "list") => {
            (WhisperAction::ListModels, 2usize)
        }
        "model" | "download-model" => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| io::Error::other("whisper model requires a model name"))?;
            (WhisperAction::DownloadModel { name }, 2usize)
        }
        other => {
            return Err(io::Error::other(format!(
                "unknown whisper command `{other}`"
            )));
        }
    };
    let mut parsed = WhisperArgs {
        action,
        binary_path: None,
        models_dir: None,
    };

    while index < args.len() {
        match args[index].as_str() {
            "--bin" => parsed.binary_path = Some(required_path(args, &mut index, "--bin")?),
            "--models-dir" => {
                parsed.models_dir = Some(required_path(args, &mut index, "--models-dir")?)
            }
            other => {
                return Err(io::Error::other(format!(
                    "unknown whisper option `{other}`"
                )));
            }
        }
        index += 1;
    }

    Ok(parsed)
}

fn parse_translation_setting_option(
    option: &str,
    args: &[String],
    index: &mut usize,
    settings: &mut TranslationSettings,
) -> io::Result<bool> {
    match option {
        "--output-format" => settings.output_format = Some(required_value(args, index, option)?),
        "--provider" => settings.provider = required_value(args, index, option)?,
        "--model" => settings.model = required_value(args, index, option)?,
        "--api-key" => settings.api_key = Some(required_value(args, index, option)?),
        "--base-url" => settings.base_url = Some(required_value(args, index, option)?),
        "--source-lang" => settings.source_language = required_value(args, index, option)?,
        "--target-lang" => settings.target_language = required_value(args, index, option)?,
        "--batch-size" => settings.batch_size = parse_batch_size(args, index)?,
        "--runtime-dir" => settings.runtime_dir = Some(required_path(args, index, option)?),
        "--glossary" => settings.glossary_path = Some(required_path(args, index, option)?),
        "--bilingual" => settings.bilingual = true,
        "--fast" => settings.fast_mode = true,
        "--no-review" => settings.final_review = false,
        "--dry-run" => settings.dry_run = true,
        "--resume" => settings.resume = true,
        "--no-resume" => settings.resume = false,
        "--cache" => settings.use_cache = true,
        "--no-cache" => settings.use_cache = false,
        "--retries" => settings.retries = parse_nonnegative_usize(args, index, option)?,
        "--agent" => settings.agent = true,
        "--no-agent" => settings.agent = false,
        "--agent-repair-attempts" => {
            settings.agent_repair_attempts = parse_nonnegative_usize(args, index, option)?
        }
        _ => return Ok(false),
    }

    Ok(true)
}

pub(crate) fn required_value(args: &[String], index: &mut usize, flag: &str) -> io::Result<String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| io::Error::other(format!("{flag} requires a value")))
}

fn required_path(args: &[String], index: &mut usize, flag: &str) -> io::Result<PathBuf> {
    required_value(args, index, flag).map(PathBuf::from)
}

fn parse_batch_size(args: &[String], index: &mut usize) -> io::Result<usize> {
    let raw = required_value(args, index, "--batch-size")?;
    let value = raw
        .parse::<usize>()
        .map_err(|_| io::Error::other("--batch-size must be a positive integer"))?;
    if value == 0 {
        return Err(io::Error::other("--batch-size must be greater than zero"));
    }
    Ok(value)
}

fn parse_nonnegative_usize(args: &[String], index: &mut usize, flag: &str) -> io::Result<usize> {
    required_value(args, index, flag)?
        .parse::<usize>()
        .map_err(|_| io::Error::other(format!("{flag} must be a non-negative integer")))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let args = vec![
            "clip.srt".to_owned(),
            "--no-resume".to_owned(),
            "--no-cache".to_owned(),
            "--retries".to_owned(),
            "0".to_owned(),
            "--no-agent".to_owned(),
            "--agent-repair-attempts".to_owned(),
            "3".to_owned(),
        ];
        let parsed = parse_translate_args(&args).expect("translate args should parse");

        assert!(!parsed.settings.resume);
        assert!(!parsed.settings.use_cache);
        assert_eq!(parsed.settings.retries, 0);
        assert!(!parsed.settings.agent);
        assert_eq!(parsed.settings.agent_repair_attempts, 3);
    }

    #[test]
    fn parse_agent_resume_accepts_optional_session() {
        let args = vec!["resume".to_owned(), "abc".to_owned()];

        let parsed = parse_agent_args(&args).expect("agent resume should parse");

        assert_eq!(
            parsed.action,
            AgentAction {
                kind: "resume".to_owned(),
                session_id: Some("abc".to_owned()),
            }
        );
    }

    #[test]
    fn parse_batch_reuses_translation_options() {
        let args = vec![
            "season".to_owned(),
            "--recursive".to_owned(),
            "--bilingual".to_owned(),
        ];
        let parsed = parse_batch_args(&args).expect("batch args should parse");

        assert!(parsed.recursive);
        assert!(parsed.translate.settings.bilingual);
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
            [defaults]
            target_language = "Japanese"

            [profiles.zh]
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
        assert_eq!(parsed.translate.settings.target_language, "Chinese");
        assert_eq!(parsed.translate.settings.batch_size, 7);
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
            [defaults]
            target_language = "Japanese"
            batch_size = 9
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

        assert_eq!(parsed.settings.target_language, "English");
        assert_eq!(parsed.settings.batch_size, 9);
        assert!(parsed.settings.bilingual);
    }

    #[test]
    fn parse_pipeline_reuses_file_translation_options() {
        let args = vec![
            "movie.srt".to_owned(),
            "--output".to_owned(),
            "movie.zh.srt".to_owned(),
            "--json".to_owned(),
            "--no-review".to_owned(),
            "--transcriber".to_owned(),
            "whisper_cpp".to_owned(),
            "--transcribe-model".to_owned(),
            "base".to_owned(),
            "--language".to_owned(),
            "en".to_owned(),
        ];

        let parsed = parse_pipeline_args(&args).expect("pipeline args should parse");

        assert_eq!(parsed.input_path, PathBuf::from("movie.srt"));
        assert_eq!(parsed.output, Some(PathBuf::from("movie.zh.srt")));
        assert!(parsed.json);
        assert!(!parsed.settings.final_review);
        assert_eq!(parsed.transcription_settings.provider, "whisper_cpp");
        assert_eq!(parsed.transcription_settings.model.as_deref(), Some("base"));
        assert_eq!(
            parsed.transcription_settings.language.as_deref(),
            Some("en")
        );
    }

    #[test]
    fn parse_transcribe_accepts_backend_options() {
        let args = vec![
            "movie.mp4".to_owned(),
            "--language".to_owned(),
            "en".to_owned(),
            "--provider".to_owned(),
            "whisper_cpp".to_owned(),
            "--model".to_owned(),
            "base".to_owned(),
            "--api-key".to_owned(),
            "sk-test".to_owned(),
            "--base-url".to_owned(),
            "https://example.test/v1".to_owned(),
            "--format".to_owned(),
            "vtt".to_owned(),
            "--sidecar".to_owned(),
            "movie.srt".to_owned(),
        ];

        let parsed = parse_transcribe_args(&args).expect("transcribe args should parse");

        assert_eq!(parsed.media_path, PathBuf::from("movie.mp4"));
        assert_eq!(parsed.settings.language.as_deref(), Some("en"));
        assert_eq!(parsed.settings.provider, "whisper_cpp");
        assert_eq!(parsed.settings.model.as_deref(), Some("base"));
        assert_eq!(parsed.settings.api_key.as_deref(), Some("sk-test"));
        assert_eq!(
            parsed.settings.base_url.as_deref(),
            Some("https://example.test/v1")
        );
        assert_eq!(parsed.settings.output_format, TranscriptionFormat::Vtt);
        assert_eq!(
            parsed.settings.sidecar_path,
            Some(PathBuf::from("movie.srt"))
        );
    }

    #[test]
    fn parse_provider_check_defaults_to_mock() {
        let args = vec!["check".to_owned()];

        let parsed = parse_provider_args(&args).expect("provider check should parse");

        assert_eq!(parsed.config, BackendConfig::new("mock", "mock-zh"));
    }

    #[test]
    fn parse_provider_check_accepts_api_key_and_base_url() {
        let args = vec![
            "check".to_owned(),
            "--provider".to_owned(),
            "openai".to_owned(),
            "--model".to_owned(),
            "gpt".to_owned(),
            "--api-key".to_owned(),
            "sk-test".to_owned(),
            "--base-url".to_owned(),
            "https://example.test/v1".to_owned(),
        ];

        let parsed = parse_provider_args(&args).expect("provider check should parse");

        assert_eq!(parsed.config.provider, "openai");
        assert_eq!(parsed.config.api_key.as_deref(), Some("sk-test"));
        assert_eq!(
            parsed.config.base_url.as_deref(),
            Some("https://example.test/v1")
        );
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
}
