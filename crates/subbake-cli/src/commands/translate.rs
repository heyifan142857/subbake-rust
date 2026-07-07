use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use subbake_adapters::{
    BackendConfig, build_backend, default_output_path, is_supported_subtitle_path, read_document,
    render_and_write_document,
};
use subbake_core::entities::PipelineOptions;
use subbake_core::formats::RenderOptions;
use subbake_core::pipeline::SubtitlePipeline;
use subbake_core::ports::NoopDashboard;

use crate::args::{BatchArgs, TranslateArgs};
use crate::output::result_json;

pub fn translate_file(args: TranslateArgs) -> io::Result<Option<PathBuf>> {
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

    let options = build_pipeline_options(&args, output_path.clone());
    let backend = build_backend(&BackendConfig::new(&options.provider, &options.model))
        .map_err(io::Error::other)?;
    let mut pipeline = SubtitlePipeline::new(backend, NoopDashboard, options);
    let run = pipeline.run_document(&document).map_err(io::Error::other)?;

    if args.dry_run {
        print_dry_run(&args, &run.result.planned_batches);
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

pub fn translate_batch(args: BatchArgs) -> io::Result<()> {
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

fn build_pipeline_options(args: &TranslateArgs, output_path: PathBuf) -> PipelineOptions {
    let mut options = PipelineOptions::new(args.subtitle.clone());
    options.output_path = Some(output_path);
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
    options
}

fn print_dry_run(args: &TranslateArgs, planned_batches: &[subbake_core::BatchPlanEntry]) {
    if args.json {
        let result = subbake_core::PipelineResult {
            output_path: None,
            batches_translated: 0,
            review_batches: 0,
            usage: subbake_core::Usage::default(),
            dry_run: true,
            planned_batches: planned_batches.to_vec(),
            cache_hits: 0,
            resumed_translation_batches: 0,
            resumed_review_batches: 0,
            translation_memory_hits: 0,
            state_path: None,
            glossary_path: args.glossary.clone(),
            agent_repairs: Vec::new(),
        };
        println!("{}", result_json(&result));
        return;
    }

    println!("Dry run: no model calls were made.");
    println!("Planned batches: {}", planned_batches.len());
    for batch in planned_batches {
        println!(
            "  batch {}: {} line(s), {} -> {}",
            batch.index, batch.size, batch.first_id, batch.last_id
        );
    }
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
        } else if path.is_file() && is_supported_subtitle_path(&path) && !is_generated_output(&path) {
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
