use std::io;
use std::path::PathBuf;

use subbake_adapters::TranslationSettings;

#[derive(Debug, Clone)]
pub struct TranslateArgs {
    pub subtitle: PathBuf,
    pub output: Option<PathBuf>,
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
    pub translate: BatchTranslateOptions,
}

impl TranslateArgs {
    pub fn default_for(subtitle: impl Into<PathBuf>) -> Self {
        Self {
            subtitle: subtitle.into(),
            output: None,
            settings: TranslationSettings::default(),
            json: false,
        }
    }
}

pub fn parse_translate_args(args: &[String]) -> io::Result<TranslateArgs> {
    let subtitle = args
        .first()
        .ok_or_else(|| io::Error::other("translate requires a subtitle path"))?;
    let mut parsed = TranslateArgs::default_for(subtitle);
    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "-o" | "--output" => parsed.output = Some(required_path(args, &mut index, "--output")?),
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
                    "unknown translate option `{other}`"
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
        translate: BatchTranslateOptions::default(),
    };

    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--recursive" => parsed.recursive = true,
            "--overwrite" => parsed.overwrite = true,
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

pub(crate) fn option_path_value(args: &[String], flag: &str) -> io::Result<Option<PathBuf>> {
    let mut index = 0usize;
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
}
