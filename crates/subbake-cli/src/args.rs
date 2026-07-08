use std::io;
use std::path::PathBuf;

use subbake_adapters::{
    BackendConfig, RuntimeAction, TranscriptionFormat, TranscriptionSettings, TranslationSettings,
    WhisperAction, load_translation_settings_patch,
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
    pub settings: TranslationSettings,
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
            settings: TranslationSettings::default(),
            json: false,
        }
    }
}

pub fn parse_agent_args(args: &[String]) -> io::Result<AgentArgs> {
    let action = match args.first().map(String::as_str) {
        None => AgentAction::Start,
        Some("resume") => {
            if args.len() > 2 {
                return Err(io::Error::other(
                    "agent resume accepts at most one session id",
                ));
            }
            AgentAction::Resume {
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
    apply_config_if_present(args, 1, &mut parsed.config_path, &mut parsed.settings)?;
    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "-o" | "--output" => parsed.output = Some(required_path(args, &mut index, "--output")?),
            "--config" => parsed.config_path = Some(required_path(args, &mut index, "--config")?),
            "--output-format" => {
                parsed.settings.output_format =
                    Some(required_value(args, &mut index, "--output-format")?)
            }
            "--provider" => {
                parsed.settings.provider = required_value(args, &mut index, "--provider")?
            }
            "--model" => parsed.settings.model = required_value(args, &mut index, "--model")?,
            "--source-lang" => {
                parsed.settings.source_language = required_value(args, &mut index, "--source-lang")?
            }
            "--target-lang" => {
                parsed.settings.target_language = required_value(args, &mut index, "--target-lang")?
            }
            "--batch-size" => parsed.settings.batch_size = parse_batch_size(args, &mut index)?,
            "--bilingual" => parsed.settings.bilingual = true,
            "--fast" => parsed.settings.fast_mode = true,
            "--no-review" => parsed.settings.final_review = false,
            "--dry-run" => parsed.settings.dry_run = true,
            "--runtime-dir" => {
                parsed.settings.runtime_dir =
                    Some(required_path(args, &mut index, "--runtime-dir")?)
            }
            "--glossary" => {
                parsed.settings.glossary_path = Some(required_path(args, &mut index, "--glossary")?)
            }
            "--json" => parsed.json = true,
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

pub fn parse_batch_args(args: &[String]) -> io::Result<BatchArgs> {
    let dir = args
        .first()
        .ok_or_else(|| io::Error::other("batch requires a directory"))?;
    let mut parsed = BatchArgs {
        dir: PathBuf::from(dir),
        recursive: false,
        overwrite: false,
        config_path: None,
        translate: BatchTranslateOptions::default(),
    };
    apply_config_if_present(
        args,
        1,
        &mut parsed.config_path,
        &mut parsed.translate.settings,
    )?;

    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--recursive" => parsed.recursive = true,
            "--overwrite" => parsed.overwrite = true,
            "--config" => parsed.config_path = Some(required_path(args, &mut index, "--config")?),
            "--output-format" => {
                parsed.translate.settings.output_format =
                    Some(required_value(args, &mut index, "--output-format")?)
            }
            "--provider" => {
                parsed.translate.settings.provider = required_value(args, &mut index, "--provider")?
            }
            "--model" => {
                parsed.translate.settings.model = required_value(args, &mut index, "--model")?
            }
            "--source-lang" => {
                parsed.translate.settings.source_language =
                    required_value(args, &mut index, "--source-lang")?
            }
            "--target-lang" => {
                parsed.translate.settings.target_language =
                    required_value(args, &mut index, "--target-lang")?
            }
            "--batch-size" => {
                parsed.translate.settings.batch_size = parse_batch_size(args, &mut index)?
            }
            "--bilingual" => parsed.translate.settings.bilingual = true,
            "--fast" => parsed.translate.settings.fast_mode = true,
            "--no-review" => parsed.translate.settings.final_review = false,
            "--dry-run" => parsed.translate.settings.dry_run = true,
            "--runtime-dir" => {
                parsed.translate.settings.runtime_dir =
                    Some(required_path(args, &mut index, "--runtime-dir")?)
            }
            "--glossary" => {
                parsed.translate.settings.glossary_path =
                    Some(required_path(args, &mut index, "--glossary")?)
            }
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
            "--model" => parsed.settings.model = Some(required_value(args, &mut index, "--model")?),
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

pub fn parse_provider_args(args: &[String]) -> io::Result<ProviderArgs> {
    if args.first().is_none_or(|value| value != "check") {
        return Err(io::Error::other("provider requires `check`"));
    }

    let mut provider = "mock".to_owned();
    let mut model = "mock-zh".to_owned();
    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--provider" => provider = required_value(args, &mut index, "--provider")?,
            "--model" => model = required_value(args, &mut index, "--model")?,
            other => {
                return Err(io::Error::other(format!(
                    "unknown provider option `{other}`"
                )));
            }
        }
        index += 1;
    }

    Ok(ProviderArgs {
        config: BackendConfig::new(provider, model),
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
    let mut index = 2usize;
    while index < args.len() {
        match args[index].as_str() {
            "--runtime-dir" => {
                runtime_dir = Some(required_path(args, &mut index, "--runtime-dir")?)
            }
            "--yes" if command == "clean" => yes = true,
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
        "clean" => RuntimeAction::Clean { yes },
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

fn apply_config_if_present(
    args: &[String],
    start_index: usize,
    config_path: &mut Option<PathBuf>,
    settings: &mut TranslationSettings,
) -> io::Result<()> {
    let Some(path) = option_path_value_from(args, start_index, "--config")? else {
        return Ok(());
    };
    let patch = load_translation_settings_patch(&path)?;
    settings.apply_patch(patch);
    *config_path = Some(path);
    Ok(())
}

fn option_path_value_from(
    args: &[String],
    start_index: usize,
    flag: &str,
) -> io::Result<Option<PathBuf>> {
    let mut index = start_index;
    while index < args.len() {
        if args[index] == flag {
            return required_path(args, &mut index, flag).map(Some);
        }
        index += 1;
    }
    Ok(None)
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
    fn parse_agent_resume_accepts_optional_session() {
        let args = vec!["resume".to_owned(), "abc".to_owned()];

        let parsed = parse_agent_args(&args).expect("agent resume should parse");

        assert_eq!(
            parsed.action,
            AgentAction::Resume {
                session_id: Some("abc".to_owned())
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
        ];

        let parsed = parse_pipeline_args(&args).expect("pipeline args should parse");

        assert_eq!(parsed.input_path, PathBuf::from("movie.srt"));
        assert_eq!(parsed.output, Some(PathBuf::from("movie.zh.srt")));
        assert!(parsed.json);
        assert!(!parsed.settings.final_review);
    }

    #[test]
    fn parse_transcribe_accepts_backend_options() {
        let args = vec![
            "movie.mp4".to_owned(),
            "--language".to_owned(),
            "en".to_owned(),
            "--model".to_owned(),
            "base".to_owned(),
            "--format".to_owned(),
            "vtt".to_owned(),
        ];

        let parsed = parse_transcribe_args(&args).expect("transcribe args should parse");

        assert_eq!(parsed.media_path, PathBuf::from("movie.mp4"));
        assert_eq!(parsed.settings.language.as_deref(), Some("en"));
        assert_eq!(parsed.settings.model.as_deref(), Some("base"));
        assert_eq!(parsed.settings.output_format, TranscriptionFormat::Vtt);
    }

    #[test]
    fn parse_provider_check_defaults_to_mock() {
        let args = vec!["check".to_owned()];

        let parsed = parse_provider_args(&args).expect("provider check should parse");

        assert_eq!(parsed.config, BackendConfig::new("mock", "mock-zh"));
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

        assert_eq!(parsed.action, RuntimeAction::Clean { yes: true });
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
}
