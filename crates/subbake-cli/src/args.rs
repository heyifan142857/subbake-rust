use std::io;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct TranslateArgs {
    pub subtitle: PathBuf,
    pub output: Option<PathBuf>,
    pub output_format: Option<String>,
    pub provider: String,
    pub model: String,
    pub source_lang: String,
    pub target_lang: String,
    pub batch_size: usize,
    pub bilingual: bool,
    pub fast: bool,
    pub no_review: bool,
    pub dry_run: bool,
    pub runtime_dir: Option<PathBuf>,
    pub glossary: Option<PathBuf>,
    pub json: bool,
}

#[derive(Debug, Clone)]
pub struct BatchTranslateOptions {
    pub output_format: Option<String>,
    pub provider: String,
    pub model: String,
    pub source_lang: String,
    pub target_lang: String,
    pub batch_size: usize,
    pub bilingual: bool,
    pub fast: bool,
    pub no_review: bool,
    pub dry_run: bool,
    pub runtime_dir: Option<PathBuf>,
    pub glossary: Option<PathBuf>,
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
            output_format: None,
            provider: "mock".to_owned(),
            model: "mock-zh".to_owned(),
            source_lang: "Auto".to_owned(),
            target_lang: "Chinese".to_owned(),
            batch_size: 30,
            bilingual: false,
            fast: false,
            no_review: false,
            dry_run: false,
            runtime_dir: None,
            glossary: None,
            json: false,
        }
    }
}

impl BatchTranslateOptions {
    pub fn default() -> Self {
        Self {
            output_format: None,
            provider: "mock".to_owned(),
            model: "mock-zh".to_owned(),
            source_lang: "Auto".to_owned(),
            target_lang: "Chinese".to_owned(),
            batch_size: 30,
            bilingual: false,
            fast: false,
            no_review: false,
            dry_run: false,
            runtime_dir: None,
            glossary: None,
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
                parsed.output_format = Some(required_value(args, &mut index, "--output-format")?)
            }
            "--provider" => parsed.provider = required_value(args, &mut index, "--provider")?,
            "--model" => parsed.model = required_value(args, &mut index, "--model")?,
            "--source-lang" => parsed.source_lang = required_value(args, &mut index, "--source-lang")?,
            "--target-lang" => parsed.target_lang = required_value(args, &mut index, "--target-lang")?,
            "--batch-size" => parsed.batch_size = parse_batch_size(args, &mut index)?,
            "--bilingual" => parsed.bilingual = true,
            "--fast" => parsed.fast = true,
            "--no-review" => parsed.no_review = true,
            "--dry-run" => parsed.dry_run = true,
            "--runtime-dir" => parsed.runtime_dir = Some(required_path(args, &mut index, "--runtime-dir")?),
            "--glossary" => parsed.glossary = Some(required_path(args, &mut index, "--glossary")?),
            "--json" => parsed.json = true,
            other => return Err(io::Error::other(format!("unknown translate option `{other}`"))),
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
                parsed.translate.output_format = Some(required_value(args, &mut index, "--output-format")?)
            }
            "--provider" => parsed.translate.provider = required_value(args, &mut index, "--provider")?,
            "--model" => parsed.translate.model = required_value(args, &mut index, "--model")?,
            "--source-lang" => {
                parsed.translate.source_lang = required_value(args, &mut index, "--source-lang")?
            }
            "--target-lang" => {
                parsed.translate.target_lang = required_value(args, &mut index, "--target-lang")?
            }
            "--batch-size" => parsed.translate.batch_size = parse_batch_size(args, &mut index)?,
            "--bilingual" => parsed.translate.bilingual = true,
            "--fast" => parsed.translate.fast = true,
            "--no-review" => parsed.translate.no_review = true,
            "--dry-run" => parsed.translate.dry_run = true,
            "--runtime-dir" => {
                parsed.translate.runtime_dir = Some(required_path(args, &mut index, "--runtime-dir")?)
            }
            "--glossary" => parsed.translate.glossary = Some(required_path(args, &mut index, "--glossary")?),
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
        let args = vec!["clip.srt".to_owned(), "--batch-size".to_owned(), "0".to_owned()];
        let error = parse_translate_args(&args).expect_err("zero batch size should fail");
        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn parse_batch_reuses_translation_options() {
        let args = vec!["season".to_owned(), "--recursive".to_owned(), "--bilingual".to_owned()];
        let parsed = parse_batch_args(&args).expect("batch args should parse");

        assert!(parsed.recursive);
        assert!(parsed.translate.bilingual);
    }
}
