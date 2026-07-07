use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use subbake_adapters::{
    TranslationRequest, default_output_path, is_supported_subtitle_path, translate_subtitle,
};

use crate::args::{BatchArgs, TranslateArgs};
use crate::output::result_json;

pub fn translate_file(args: TranslateArgs) -> io::Result<Option<PathBuf>> {
    let outcome = translate_subtitle(TranslationRequest {
        input_path: args.subtitle.clone(),
        output_path: args.output.clone(),
        settings: args.settings.clone(),
    })?;
    if args.settings.dry_run {
        print_dry_run(&args, &outcome.result.planned_batches);
        return Ok(None);
    }

    let output_path = outcome
        .output_path
        .clone()
        .ok_or_else(|| io::Error::other("translation completed without an output path"))?;

    if args.json {
        println!("{}", result_json(&outcome.result));
    } else {
        println!("Output: {}", output_path.display());
        println!(
            "Usage: {} in / {} out / {} total",
            outcome.result.usage.input_tokens,
            outcome.result.usage.output_tokens,
            outcome.result.usage.total_tokens
        );
        println!("Batches: {} translated", outcome.result.batches_translated);
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
            config_path: args.config_path.clone(),
            settings: args.translate.settings.clone(),
            json: false,
        };
        let output_path = default_output_path(
            &translate_args.subtitle,
            translate_args.settings.output_format(),
            translate_args.settings.bilingual,
        )?;
        if output_path.exists() && !args.overwrite && !args.translate.settings.dry_run {
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
            glossary_path: args.settings.glossary_path.clone(),
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
