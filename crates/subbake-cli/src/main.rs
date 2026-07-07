use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use subbake_adapters::{
    MockBackend, default_output_path, is_supported_subtitle_path, read_document,
    render_and_write_document,
};
use subbake_core::entities::{BatchPlanEntry, PipelineOptions, PipelineResult};
use subbake_core::formats::RenderOptions;
use subbake_core::pipeline::SubtitlePipeline;
use subbake_core::ports::{LlmBackend, NoopDashboard};
use subbake_core::storage::build_runtime_paths;

#[derive(Debug, Clone)]
struct TranslateArgs {
    subtitle: PathBuf,
    output: Option<PathBuf>,
    output_format: Option<String>,
    provider: String,
    model: String,
    source_lang: String,
    target_lang: String,
    batch_size: usize,
    bilingual: bool,
    fast: bool,
    no_review: bool,
    dry_run: bool,
    runtime_dir: Option<PathBuf>,
    glossary: Option<PathBuf>,
    json: bool,
}

#[derive(Debug, Clone)]
struct BatchTranslateOptions {
    output_format: Option<String>,
    provider: String,
    model: String,
    source_lang: String,
    target_lang: String,
    batch_size: usize,
    bilingual: bool,
    fast: bool,
    no_review: bool,
    dry_run: bool,
    runtime_dir: Option<PathBuf>,
    glossary: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct BatchArgs {
    dir: PathBuf,
    recursive: bool,
    overwrite: bool,
    translate: BatchTranslateOptions,
}

fn main() {
    if let Err(error) = run(env::args().skip(1).collect()) {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

fn run(args: Vec<String>) -> io::Result<()> {
    if args.is_empty() {
        println!("{}", subbake_agent::start_agent());
        return Ok(());
    }

    match args[0].as_str() {
        "agent" => run_agent(&args[1..]),
        "translate" => translate_file(parse_translate_args(&args[1..])?).map(|_| ()),
        "batch" => translate_batch(parse_batch_args(&args[1..])?),
        "transcribe" => run_transcribe(&args[1..]),
        "pipeline" => run_pipeline(&args[1..]),
        "provider" => run_provider(&args[1..]),
        "runtime" => run_runtime(&args[1..]),
        "whisper" => run_whisper(&args[1..]),
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
    for name in subbake_cli::command_names() {
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

fn run_transcribe(args: &[String]) -> io::Result<()> {
    let media = args
        .first()
        .ok_or_else(|| io::Error::other("transcribe requires a media path"))?;
    println!("Transcription adapter is pending migration: {media}");
    Ok(())
}

fn run_pipeline(args: &[String]) -> io::Result<()> {
    let input = args
        .first()
        .ok_or_else(|| io::Error::other("pipeline requires an input path"))?;
    println!(
        "Pipeline adapter is pending migration for {input}. Use `translate` for subtitle files."
    );
    Ok(())
}

fn run_provider(args: &[String]) -> io::Result<()> {
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
            other => return Err(io::Error::other(format!("unknown provider option `{other}`"))),
        }
        index += 1;
    }

    if provider != "mock" {
        return Err(io::Error::other(format!(
            "provider `{provider}` adapter is pending migration"
        )));
    }
    let backend = MockBackend::new(model);
    let (ok, message) = backend.check_credentials().map_err(io::Error::other)?;
    if ok {
        println!("Provider check passed.");
        println!("{message}");
        Ok(())
    } else {
        Err(io::Error::other(message))
    }
}

fn run_runtime(args: &[String]) -> io::Result<()> {
    match args.first().map(String::as_str) {
        Some("inspect") => {
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
        Some("clean") => {
            let target = args
                .get(1)
                .ok_or_else(|| io::Error::other("runtime clean requires a target"))?;
            let yes = args.iter().any(|value| value == "--yes");
            let runtime_dir = option_path_value(args, "--runtime-dir")?;
            clean_runtime(Path::new(target), runtime_dir.as_deref(), yes)
        }
        _ => Err(io::Error::other("runtime requires `inspect` or `clean`")),
    }
}

fn run_whisper(args: &[String]) -> io::Result<()> {
    let command = args.first().map(String::as_str).unwrap_or("status");
    println!("whisper command `{command}` is pending adapter migration.");
    Ok(())
}

fn parse_translate_args(args: &[String]) -> io::Result<TranslateArgs> {
    let subtitle = args
        .first()
        .ok_or_else(|| io::Error::other("translate requires a subtitle path"))?;
    let mut parsed = TranslateArgs {
        subtitle: PathBuf::from(subtitle),
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
    };
    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "-o" | "--output" => parsed.output = Some(required_path(args, &mut index, "--output")?),
            "--output-format" => parsed.output_format = Some(required_value(args, &mut index, "--output-format")?),
            "--provider" => parsed.provider = required_value(args, &mut index, "--provider")?,
            "--model" => parsed.model = required_value(args, &mut index, "--model")?,
            "--source-lang" => parsed.source_lang = required_value(args, &mut index, "--source-lang")?,
            "--target-lang" => parsed.target_lang = required_value(args, &mut index, "--target-lang")?,
            "--batch-size" => parsed.batch_size = required_value(args, &mut index, "--batch-size")?.parse().map_err(|_| io::Error::other("--batch-size must be a positive integer"))?,
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

fn parse_batch_args(args: &[String]) -> io::Result<BatchArgs> {
    let dir = args
        .first()
        .ok_or_else(|| io::Error::other("batch requires a directory"))?;
    let mut parsed = BatchArgs {
        dir: PathBuf::from(dir),
        recursive: false,
        overwrite: false,
        translate: BatchTranslateOptions {
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
        },
    };

    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--recursive" => parsed.recursive = true,
            "--overwrite" => parsed.overwrite = true,
            "--output-format" => parsed.translate.output_format = Some(required_value(args, &mut index, "--output-format")?),
            "--provider" => parsed.translate.provider = required_value(args, &mut index, "--provider")?,
            "--model" => parsed.translate.model = required_value(args, &mut index, "--model")?,
            "--source-lang" => parsed.translate.source_lang = required_value(args, &mut index, "--source-lang")?,
            "--target-lang" => parsed.translate.target_lang = required_value(args, &mut index, "--target-lang")?,
            "--batch-size" => parsed.translate.batch_size = required_value(args, &mut index, "--batch-size")?.parse().map_err(|_| io::Error::other("--batch-size must be a positive integer"))?,
            "--bilingual" => parsed.translate.bilingual = true,
            "--fast" => parsed.translate.fast = true,
            "--no-review" => parsed.translate.no_review = true,
            "--dry-run" => parsed.translate.dry_run = true,
            "--runtime-dir" => parsed.translate.runtime_dir = Some(required_path(args, &mut index, "--runtime-dir")?),
            "--glossary" => parsed.translate.glossary = Some(required_path(args, &mut index, "--glossary")?),
            other => return Err(io::Error::other(format!("unknown batch option `{other}`"))),
        }
        index += 1;
    }

    Ok(parsed)
}

fn translate_file(args: TranslateArgs) -> io::Result<Option<PathBuf>> {
    if args.provider != "mock" {
        return Err(io::Error::other(format!(
            "provider `{}` adapter is pending migration",
            args.provider
        )));
    }
    if !is_supported_subtitle_path(&args.subtitle) {
        return Err(io::Error::other(
            "`translate` accepts subtitle files only; use `pipeline` for combined media workflows",
        ));
    }

    let document = read_document(&args.subtitle)?;
    let output_path = match args.output.clone() {
        Some(path) => path,
        None => default_output_path(&args.subtitle, args.output_format.as_deref(), args.bilingual)?,
    };

    let mut options = PipelineOptions::new(args.subtitle.clone());
    options.output_path = Some(output_path.clone());
    options.output_format = args.output_format.clone();
    options.provider = args.provider.clone();
    options.model = args.model.clone();
    options.source_language = args.source_lang.clone();
    options.target_language = args.target_lang.clone();
    options.batch_size = args.batch_size;
    options.bilingual = args.bilingual;
    options.fast_mode = args.fast;
    options.final_review = !args.no_review;
    options.dry_run = args.dry_run;
    options.runtime_dir = args.runtime_dir.clone();
    options.glossary_path = args.glossary.clone();

    let backend = MockBackend::new(&options.model);
    let mut pipeline = SubtitlePipeline::new(backend, NoopDashboard, options);
    let run = pipeline.run_document(&document).map_err(io::Error::other)?;

    if args.dry_run {
        if args.json {
            println!("{}", result_json(&run.result));
        } else {
            println!("Dry run: no model calls were made.");
            println!("Planned batches: {}", run.result.planned_batches.len());
            for batch in run.result.planned_batches {
                println!(
                    "  batch {}: {} line(s), {} -> {}",
                    batch.index, batch.size, batch.first_id, batch.last_id
                );
            }
        }
        return Ok(None);
    }

    let render_options = RenderOptions::new(args.bilingual, args.output_format.clone());
    render_and_write_document(
        &document,
        &run.translated_segments,
        &output_path,
        &render_options,
    )?;

    if args.json {
        let mut result = run.result;
        result.output_path = Some(output_path.clone());
        println!("{}", result_json(&result));
    } else {
        println!("Output: {}", output_path.display());
        println!(
            "Usage: {} in / {} out / {} total",
            run.result.usage.input_tokens,
            run.result.usage.output_tokens,
            run.result.usage.total_tokens
        );
        println!("Batches: {} translated", run.result.batches_translated);
    }

    Ok(Some(output_path))
}

fn translate_batch(args: BatchArgs) -> io::Result<()> {
    let files = discover_subtitle_files(&args.dir, args.recursive)?;
    if files.is_empty() {
        println!("No subtitle files found.");
        return Ok(());
    }

    let mut processed = 0usize;
    let mut skipped = 0usize;
    for file in files {
        let translate_args = TranslateArgs {
            subtitle: file.clone(),
            output: None,
            output_format: args.translate.output_format.clone(),
            provider: args.translate.provider.clone(),
            model: args.translate.model.clone(),
            source_lang: args.translate.source_lang.clone(),
            target_lang: args.translate.target_lang.clone(),
            batch_size: args.translate.batch_size,
            bilingual: args.translate.bilingual,
            fast: args.translate.fast,
            no_review: args.translate.no_review,
            dry_run: args.translate.dry_run,
            runtime_dir: args.translate.runtime_dir.clone(),
            glossary: args.translate.glossary.clone(),
            json: false,
        };
        let output_path = default_output_path(
            &translate_args.subtitle,
            translate_args.output_format.as_deref(),
            translate_args.bilingual,
        )?;
        if output_path.exists() && !args.overwrite && !args.translate.dry_run {
            println!("Skipped existing output: {}", output_path.display());
            skipped += 1;
            continue;
        }
        translate_file(translate_args)?;
        processed += 1;
    }

    println!("Batch result: {processed} processed, {skipped} skipped, 0 failed");
    Ok(())
}

fn discover_subtitle_files(dir: &Path, recursive: bool) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    discover_subtitle_files_inner(dir, recursive, &mut files)?;
    files.sort();
    Ok(files)
}

fn discover_subtitle_files_inner(
    dir: &Path,
    recursive: bool,
    files: &mut Vec<PathBuf>,
) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && recursive {
            discover_subtitle_files_inner(&path, recursive, files)?;
        } else if path.is_file() && is_supported_subtitle_path(&path) && !is_generated_output(&path)
        {
            files.push(path);
        }
    }
    Ok(())
}

fn is_generated_output(path: &Path) -> bool {
    path.file_stem()
        .and_then(|value| value.to_str())
        .is_some_and(|stem| stem.ends_with(".translated") || stem.ends_with(".bilingual"))
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

fn required_value(args: &[String], index: &mut usize, flag: &str) -> io::Result<String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| io::Error::other(format!("{flag} requires a value")))
}

fn required_path(args: &[String], index: &mut usize, flag: &str) -> io::Result<PathBuf> {
    required_value(args, index, flag).map(PathBuf::from)
}

fn option_path_value(args: &[String], flag: &str) -> io::Result<Option<PathBuf>> {
    let mut index = 0usize;
    while index < args.len() {
        if args[index] == flag {
            return required_path(args, &mut index, flag).map(Some);
        }
        index += 1;
    }
    Ok(None)
}

fn result_json(result: &PipelineResult) -> String {
    let output_path = result
        .output_path
        .as_ref()
        .map(|path| quote_json(&path.to_string_lossy()))
        .unwrap_or_else(|| "null".to_owned());
    let glossary_path = result
        .glossary_path
        .as_ref()
        .map(|path| quote_json(&path.to_string_lossy()))
        .unwrap_or_else(|| "null".to_owned());
    let planned_batches = result
        .planned_batches
        .iter()
        .map(batch_json)
        .collect::<Vec<_>>()
        .join(",");

    format!(
        "{{\"output_path\":{output_path},\"batches_translated\":{},\"review_batches\":{},\"usage\":{{\"input_tokens\":{},\"output_tokens\":{},\"total_tokens\":{}}},\"dry_run\":{},\"planned_batches\":[{}],\"glossary_path\":{glossary_path}}}",
        result.batches_translated,
        result.review_batches,
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.total_tokens,
        result.dry_run,
        planned_batches
    )
}

fn batch_json(batch: &BatchPlanEntry) -> String {
    format!(
        "{{\"index\":{},\"size\":{},\"first_id\":{},\"last_id\":{}}}",
        batch.index,
        batch.size,
        quote_json(&batch.first_id),
        quote_json(&batch.last_id)
    )
}

fn quote_json(value: &str) -> String {
    let mut output = String::new();
    output.push('"');
    for ch in value.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            ch => output.push(ch),
        }
    }
    output.push('"');
    output
}
