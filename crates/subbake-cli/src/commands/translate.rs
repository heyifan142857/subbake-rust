use std::io;
use std::path::PathBuf;

use subbake_adapters::{
    BatchTranslationRequest, TranslationRequest, translate_subtitle, translate_subtitle_batch,
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
    let outcome = translate_subtitle_batch(BatchTranslationRequest {
        root: args.dir,
        recursive: args.recursive,
        overwrite: args.overwrite,
        settings: args.translate.settings,
    })?;
    if outcome.processed == 0 && outcome.skipped.is_empty() {
        println!("No subtitle files found.");
        return Ok(());
    }
    for path in &outcome.skipped {
        println!("Skipped existing output for: {}", path.display());
    }

    println!(
        "Batch result: {} processed, {} skipped, 0 failed",
        outcome.processed,
        outcome.skipped.len()
    );
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
