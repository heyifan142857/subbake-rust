use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use subbake_core::PipelineResult;
use subbake_core::formats::RenderOptions;
use subbake_core::pipeline::SubtitlePipeline;
use subbake_core::ports::NoopDashboard;

use crate::fs::{
    default_output_path, is_supported_subtitle_path, read_document, render_and_write_document,
};
use crate::providers::build_backend;
use crate::settings::TranslationSettings;

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationRequest {
    pub input_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub settings: TranslationSettings,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationOutcome {
    pub result: PipelineResult,
    pub output_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchTranslationRequest {
    pub root: PathBuf,
    pub recursive: bool,
    pub overwrite: bool,
    pub settings: TranslationSettings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchTranslationOutcome {
    pub processed: usize,
    pub skipped: Vec<PathBuf>,
    pub outputs: Vec<PathBuf>,
}

pub fn translate_subtitle(request: TranslationRequest) -> io::Result<TranslationOutcome> {
    if !is_supported_subtitle_path(&request.input_path) {
        return Err(io::Error::other(
            "`translate` accepts subtitle files only; use `pipeline` for combined media workflows",
        ));
    }

    let document = read_document(&request.input_path)?;
    let output_path = match request.output_path.clone() {
        Some(path) => path,
        None => default_output_path(
            &request.input_path,
            request.settings.output_format(),
            request.settings.bilingual,
        )?,
    };

    let options = request
        .settings
        .to_pipeline_options(request.input_path.clone(), Some(output_path.clone()));
    let backend = build_backend(&request.settings.backend_config()).map_err(io::Error::other)?;
    let mut pipeline = SubtitlePipeline::new(backend, NoopDashboard, options);
    let run = pipeline.run_document(&document).map_err(io::Error::other)?;

    if request.settings.dry_run {
        return Ok(TranslationOutcome {
            result: run.result,
            output_path: None,
        });
    }

    let render_options = RenderOptions::new(
        request.settings.bilingual,
        request.settings.output_format.clone(),
    );
    render_and_write_document(
        &document,
        &run.translated_segments,
        &output_path,
        &render_options,
    )?;

    let mut result = run.result;
    result.output_path = Some(output_path.clone());
    Ok(TranslationOutcome {
        result,
        output_path: Some(output_path),
    })
}

pub fn translate_subtitle_batch(
    request: BatchTranslationRequest,
) -> io::Result<BatchTranslationOutcome> {
    let files = discover_subtitle_files(&request.root, request.recursive)?;
    let mut processed = 0usize;
    let mut skipped = Vec::new();
    let mut outputs = Vec::new();

    for input_path in files {
        let output_path = default_output_path(
            &input_path,
            request.settings.output_format(),
            request.settings.bilingual,
        )?;
        if output_path.exists() && !request.overwrite && !request.settings.dry_run {
            skipped.push(input_path);
            continue;
        }

        let outcome = translate_subtitle(TranslationRequest {
            input_path,
            output_path: None,
            settings: request.settings.clone(),
        })?;
        if let Some(output_path) = outcome.output_path {
            outputs.push(output_path);
        }
        processed += 1;
    }

    Ok(BatchTranslationOutcome {
        processed,
        skipped,
        outputs,
    })
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn translates_txt_with_mock_backend() {
        let root = temp_root("translate");
        fs::create_dir_all(&root).expect("create temp root");
        let input_path = root.join("clip.txt");
        fs::write(&input_path, "hello\n").expect("write input");

        let mut settings = TranslationSettings {
            target_language: "English".to_owned(),
            ..TranslationSettings::default()
        };
        settings.final_review = false;
        let outcome = translate_subtitle(TranslationRequest {
            input_path: input_path.clone(),
            output_path: None,
            settings,
        })
        .expect("translate");
        let output_path = outcome.output_path.expect("output path");
        let output = fs::read_to_string(&output_path).expect("read output");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(output, "[MOCK-EN] hello\n");
        assert_eq!(outcome.result.batches_translated, 1);
    }

    #[test]
    fn dry_run_does_not_write_output() {
        let root = temp_root("dry-run");
        fs::create_dir_all(&root).expect("create temp root");
        let input_path = root.join("clip.txt");
        fs::write(&input_path, "hello\n").expect("write input");
        let output_path = root.join("custom.txt");
        let settings = TranslationSettings {
            dry_run: true,
            ..TranslationSettings::default()
        };

        let outcome = translate_subtitle(TranslationRequest {
            input_path,
            output_path: Some(output_path.clone()),
            settings,
        })
        .expect("dry run");
        let output_exists = output_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(outcome.result.dry_run);
        assert!(!output_exists);
    }

    #[test]
    fn batch_skips_existing_outputs() {
        let root = temp_root("batch-skip");
        fs::create_dir_all(&root).expect("create temp root");
        let input_path = root.join("clip.txt");
        fs::write(&input_path, "hello\n").expect("write input");
        let output_path = root.join("clip.translated.txt");
        fs::write(&output_path, "existing\n").expect("write output");

        let outcome = translate_subtitle_batch(BatchTranslationRequest {
            root: root.clone(),
            recursive: false,
            overwrite: false,
            settings: TranslationSettings::default(),
        })
        .expect("batch translate");
        let output = fs::read_to_string(&output_path).expect("read output");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(outcome.processed, 0);
        assert_eq!(outcome.skipped, vec![input_path]);
        assert_eq!(output, "existing\n");
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-translation-{label}-{nanos}"))
    }
}
