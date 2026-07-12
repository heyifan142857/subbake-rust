use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use subbake_core::formats::RenderOptions;
use subbake_core::pipeline::SubtitlePipeline;
use subbake_core::ports::NoopDashboard;
use subbake_core::ports::RuntimeStore;
use subbake_core::storage::{build_runtime_paths, input_signature_from_bytes};
use subbake_core::{CancellationGuard, CoreError, NoopProgress, PipelineResult, SharedProgress};

use crate::fs::{
    default_output_path, is_supported_subtitle_path, read_document, render_and_write_document,
};
use crate::providers::build_backend;
use crate::runtime_store::FileRuntimeStore;
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
    translate_subtitle_cancellable(request, &CancellationGuard::never())
}

pub fn translate_subtitle_cancellable(
    request: TranslationRequest,
    cancellation: &CancellationGuard,
) -> io::Result<TranslationOutcome> {
    translate_subtitle_cancellable_with_progress(
        request,
        cancellation,
        std::sync::Arc::new(NoopProgress),
    )
}

pub fn translate_subtitle_cancellable_with_progress(
    request: TranslationRequest,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
) -> io::Result<TranslationOutcome> {
    check_cancelled(cancellation)?;
    if !is_supported_subtitle_path(&request.input_path) {
        return Err(io::Error::other(
            "`translate` accepts subtitle files only; use `pipeline` for combined media workflows",
        ));
    }

    let input_bytes = fs::read(&request.input_path)?;
    let metadata = fs::metadata(&request.input_path)?;
    let mtime_ns = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos());
    let input_signature = input_signature_from_bytes(&input_bytes, mtime_ns);
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

    // Wire runtime store for glossary/TM persistence.
    let paths = build_runtime_paths(
        &request.input_path,
        request.settings.runtime_dir(),
        request.settings.glossary_path(),
        &request.settings.source_language,
        &request.settings.target_language,
        request.settings.fast_mode,
    );
    let store = FileRuntimeStore::new(paths);
    store.ensure_layout().map_err(io::Error::other)?;

    let terminal_progress = progress.clone();
    let mut pipeline = SubtitlePipeline::new(backend, NoopDashboard, options)
        .with_input_signature(input_signature)
        .with_cancellation(cancellation.clone())
        .with_progress(Box::new(progress));
    pipeline = pipeline.with_store(Box::new(store));
    let run = match pipeline.run_document(&document) {
        Ok(run) => run,
        Err(error) => {
            let mut event = subbake_core::ProgressEvent::running(
                subbake_core::TaskKind::Translation,
                "TRANSLATE",
                0,
                None,
                subbake_core::ProgressUnit::Batches,
            );
            event.state = if matches!(error, CoreError::Cancelled) {
                subbake_core::TaskState::Cancelled
            } else {
                subbake_core::TaskState::Failed
            };
            event.message = Some(error.to_string());
            terminal_progress.emit(event);
            return Err(core_error(error));
        }
    };

    if request.settings.dry_run {
        let mut event = subbake_core::ProgressEvent::running(
            subbake_core::TaskKind::Translation,
            "DRY_RUN",
            1,
            Some(1),
            subbake_core::ProgressUnit::Steps,
        );
        event.state = subbake_core::TaskState::Completed;
        terminal_progress.emit(event);
        return Ok(TranslationOutcome {
            result: run.result,
            output_path: None,
        });
    }

    let render_options = RenderOptions::new(
        request.settings.bilingual,
        request.settings.output_format.clone(),
    );
    check_cancelled(cancellation)?;
    render_and_write_document(
        &document,
        &run.translated_segments,
        &output_path,
        &render_options,
    )?;

    let mut result = run.result;
    result.output_path = Some(output_path.clone());
    let mut event = subbake_core::ProgressEvent::running(
        subbake_core::TaskKind::Translation,
        "COMPLETE",
        result.batches_translated as u64,
        Some(result.batches_translated as u64),
        subbake_core::ProgressUnit::Batches,
    );
    event.state = subbake_core::TaskState::Completed;
    event.resumed = result.resumed_translation_batches as u64;
    event.usage = result.usage;
    terminal_progress.emit(event);
    Ok(TranslationOutcome {
        result,
        output_path: Some(output_path),
    })
}

pub fn translate_subtitle_batch(
    request: BatchTranslationRequest,
) -> io::Result<BatchTranslationOutcome> {
    translate_subtitle_batch_cancellable(request, &CancellationGuard::never())
}

pub fn translate_subtitle_batch_cancellable(
    request: BatchTranslationRequest,
    cancellation: &CancellationGuard,
) -> io::Result<BatchTranslationOutcome> {
    translate_subtitle_batch_with_progress(request, cancellation, std::sync::Arc::new(NoopProgress))
}

pub fn translate_subtitle_batch_with_progress(
    request: BatchTranslationRequest,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
) -> io::Result<BatchTranslationOutcome> {
    let files = discover_subtitle_files(&request.root, request.recursive)?;
    let total_files = files.len();
    let mut processed = 0usize;
    let mut skipped = Vec::new();
    let mut outputs = Vec::new();

    for input_path in files {
        check_cancelled(cancellation)?;
        let output_path = default_output_path(
            &input_path,
            request.settings.output_format(),
            request.settings.bilingual,
        )?;
        if output_path.exists() && !request.overwrite && !request.settings.dry_run {
            skipped.push(input_path);
            continue;
        }

        progress.emit(subbake_core::ProgressEvent::running(
            subbake_core::TaskKind::BatchTranslation,
            "FILES",
            processed as u64,
            Some(total_files as u64),
            subbake_core::ProgressUnit::Files,
        ));
        let outcome = translate_subtitle_cancellable_with_progress(
            TranslationRequest {
                input_path,
                output_path: None,
                settings: request.settings.clone(),
            },
            cancellation,
            progress.clone(),
        )?;
        if let Some(output_path) = outcome.output_path {
            outputs.push(output_path);
        }
        processed += 1;
    }

    let mut done = subbake_core::ProgressEvent::running(
        subbake_core::TaskKind::BatchTranslation,
        "FILES",
        processed as u64,
        Some(total_files as u64),
        subbake_core::ProgressUnit::Files,
    );
    done.state = subbake_core::TaskState::Completed;
    progress.emit(done);

    Ok(BatchTranslationOutcome {
        processed,
        skipped,
        outputs,
    })
}

fn check_cancelled(cancellation: &CancellationGuard) -> io::Result<()> {
    cancellation.check().map_err(core_error)
}

fn core_error(error: CoreError) -> io::Error {
    if matches!(error, CoreError::Cancelled) {
        io::Error::new(io::ErrorKind::Interrupted, "operation cancelled")
    } else {
        io::Error::other(error)
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
            target_language: "en".to_owned(),
            ..TranslationSettings::default()
        };
        settings.review_policy = subbake_core::ReviewPolicy::Off;
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

    #[test]
    fn batch_parse_error_identifies_the_malformed_file_and_block() {
        let root = temp_root("batch-malformed");
        fs::create_dir_all(&root).expect("create temp root");
        let input_path = root.join("broken.srt");
        fs::write(
            &input_path,
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n\nas a warning.\n",
        )
        .expect("write malformed subtitle");

        let error = translate_subtitle_batch(BatchTranslationRequest {
            root: root.clone(),
            recursive: false,
            overwrite: false,
            settings: TranslationSettings::default(),
        })
        .expect_err("malformed subtitle should stop the batch");
        let message = error.to_string();
        let _ = fs::remove_dir_all(&root);

        assert!(message.contains(&input_path.display().to_string()));
        assert!(message.contains("Malformed SRT block:\nas a warning."));
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-translation-{label}-{nanos}"))
    }
}
