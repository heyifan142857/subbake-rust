use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use subbake_core::formats::RenderOptions;
use subbake_core::languages::normalize_language;
use subbake_core::pipeline::SubtitlePipeline;
use subbake_core::ports::NoopDashboard;
use subbake_core::ports::RuntimeStore;
use subbake_core::storage::{InputSignature, build_runtime_paths, input_signature_from_bytes};
use subbake_core::{CancellationGuard, CoreError, NoopProgress, PipelineResult, SharedProgress};

use crate::embedded_subtitles::{
    default_embedded_translation_output_path, is_supported_subtitle_container_path,
    translate_embedded_subtitle_cancellable_with_progress,
};
use crate::error::{AdapterError, AdapterResult};
use crate::fs::{
    default_output_path_with_language, is_supported_subtitle_path, read_document,
    render_and_write_document, stable_runtime_input_path,
};
use crate::providers::build_backend;
use crate::runtime_store::FileRuntimeStore;
use crate::settings::TranslationSettings;

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationRequest {
    pub input_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub output_language_tag: Option<String>,
    pub overwrite: bool,
    pub settings: TranslationSettings,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationOutcome {
    pub result: PipelineResult,
    pub output_path: Option<PathBuf>,
    pub subtitle_entries: usize,
    pub container_change: Option<ContainerTranslationChange>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerTranslationChange {
    pub in_place: bool,
    pub subtitle_title: String,
    /// Previous SubBake-managed subtitle contents when this translation
    /// replaced a track for the same target language. Interactive undo can
    /// persist this small payload instead of copying the media container.
    pub previous_subtitle: Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TranslationInputIdentity {
    pub path: PathBuf,
    pub signature: InputSignature,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchTranslationRequest {
    pub root: PathBuf,
    pub recursive: bool,
    pub overwrite: bool,
    pub output_dir: Option<PathBuf>,
    pub output_language_tag: Option<String>,
    pub settings: TranslationSettings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchTranslationOutcome {
    pub processed: usize,
    pub inputs: Vec<PathBuf>,
    pub skipped: Vec<PathBuf>,
    pub outputs: Vec<PathBuf>,
    pub subtitle_entries: usize,
    pub dry_run: bool,
    pub cache_hits: usize,
    pub resumed_translation_batches: usize,
    pub resumed_review_batches: usize,
    pub translation_memory_hits: usize,
}

pub fn translate_subtitle(request: TranslationRequest) -> AdapterResult<TranslationOutcome> {
    translate_subtitle_cancellable(request, &CancellationGuard::never())
}

/// Translate a standalone subtitle or a text subtitle stream embedded in a
/// supported container. Container inputs never fall back to transcription.
pub fn translate_input(request: TranslationRequest) -> AdapterResult<TranslationOutcome> {
    translate_input_cancellable_with_progress(
        request,
        &CancellationGuard::never(),
        std::sync::Arc::new(NoopProgress),
    )
}

pub fn translate_input_cancellable(
    request: TranslationRequest,
    cancellation: &CancellationGuard,
) -> AdapterResult<TranslationOutcome> {
    translate_input_cancellable_with_progress(
        request,
        cancellation,
        std::sync::Arc::new(NoopProgress),
    )
}

pub fn translate_input_cancellable_with_progress(
    request: TranslationRequest,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
) -> AdapterResult<TranslationOutcome> {
    if is_supported_subtitle_path(&request.input_path) {
        translate_subtitle_cancellable_with_progress(request, cancellation, progress)
    } else if is_supported_subtitle_container_path(&request.input_path) {
        translate_embedded_subtitle_cancellable_with_progress(request, cancellation, progress)
    } else {
        Err(AdapterError::invalid_input(
            "`translate` accepts subtitle files or MKV, MP4/M4V/MOV, and WebM containers with text subtitle streams; use `pipeline` when transcription is required",
        ))
    }
}

pub fn default_translation_output_path(
    input_path: &Path,
    output_format: Option<&str>,
    bilingual: bool,
    language_tag: Option<&str>,
    preserve_source_container: bool,
) -> AdapterResult<PathBuf> {
    if is_supported_subtitle_container_path(input_path) {
        default_embedded_translation_output_path(
            input_path,
            bilingual,
            language_tag,
            preserve_source_container,
        )
    } else {
        default_output_path_with_language(input_path, output_format, bilingual, language_tag)
    }
}

pub fn translate_subtitle_cancellable(
    request: TranslationRequest,
    cancellation: &CancellationGuard,
) -> AdapterResult<TranslationOutcome> {
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
) -> AdapterResult<TranslationOutcome> {
    translate_subtitle_cancellable_with_progress_and_identity(request, cancellation, progress, None)
}

pub(crate) fn translate_subtitle_cancellable_with_progress_and_identity(
    mut request: TranslationRequest,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
    identity: Option<TranslationInputIdentity>,
) -> AdapterResult<TranslationOutcome> {
    check_cancelled(cancellation)?;
    normalize_translation_languages(&mut request.settings)?;
    request.output_language_tag = request
        .output_language_tag
        .as_deref()
        .map(|value| normalize_language(value, false))
        .transpose()
        .map_err(|error| AdapterError::invalid_input(error.to_string()))?;
    if !is_supported_subtitle_path(&request.input_path) {
        return Err(AdapterError::invalid_input(
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
    let input_signature = identity
        .as_ref()
        .map(|identity| identity.signature.clone())
        .unwrap_or_else(|| input_signature_from_bytes(&input_bytes, mtime_ns));
    let document = read_document(&request.input_path)?;
    let output_path = match request.output_path.clone() {
        Some(path) => path,
        None => default_output_path_with_language(
            &request.input_path,
            request.settings.output_format(),
            request.settings.output.bilingual,
            request.output_language_tag.as_deref(),
        )?,
    };
    if output_path.exists() && !request.overwrite && !request.settings.translation.dry_run {
        return Err(AdapterError::invalid_input(format!(
            "output already exists and overwrite is false: {}",
            output_path.display()
        )));
    }

    let runtime_input_path = identity
        .as_ref()
        .map(|identity| identity.path.as_path())
        .unwrap_or(&request.input_path);
    let runtime_output_path = identity
        .as_ref()
        .map(|identity| identity.output_path.clone())
        .unwrap_or_else(|| output_path.clone());
    let options = request
        .settings
        .to_pipeline_options(runtime_input_path.to_path_buf(), Some(runtime_output_path));
    let backend = build_backend(&request.settings.backend_config())?;
    let needs_reviewer = request.settings.translation.mode == subbake_core::TranslationMode::Cinema
        || request.settings.translation.review_policy != subbake_core::ReviewPolicy::Off
        || request.settings.translation.terminology_preflight;
    let reviewer = needs_reviewer
        .then(|| request.settings.reviewer_backend_config())
        .flatten()
        .map(|config| build_backend(&config))
        .transpose()?;

    // Wire runtime store for glossary/TM persistence.
    let stable_input_path = stable_runtime_input_path(runtime_input_path)?;
    let paths = build_runtime_paths(
        runtime_input_path,
        &stable_input_path,
        request.settings.runtime_dir(),
        request.settings.glossary_path(),
        &request.settings.translation.source_language,
        &request.settings.translation.target_language,
        request.settings.translation.mode == subbake_core::TranslationMode::Economy,
    );
    let store = FileRuntimeStore::new(paths);
    store.ensure_layout().map_err(AdapterError::from)?;

    let terminal_progress = progress.clone();
    let mut pipeline = SubtitlePipeline::new(backend, NoopDashboard, options)
        .with_input_signature(input_signature)
        .with_cancellation(cancellation.clone())
        .with_progress(Box::new(progress));
    if let Some(reviewer) = reviewer {
        pipeline = pipeline.with_reviewer(reviewer);
    }
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
            return Err(AdapterError::from(error));
        }
    };

    if request.settings.translation.dry_run {
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
            subtitle_entries: document.segments.len(),
            container_change: None,
        });
    }

    let render_options = RenderOptions::new(
        request.settings.output.bilingual,
        request.settings.output.format.clone(),
    )
    .with_bilingual_order(request.settings.output.bilingual_order);
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
        subtitle_entries: document.segments.len(),
        container_change: None,
    })
}

pub fn translate_subtitle_batch(
    request: BatchTranslationRequest,
) -> AdapterResult<BatchTranslationOutcome> {
    translate_subtitle_batch_cancellable(request, &CancellationGuard::never())
}

pub fn translate_subtitle_batch_cancellable(
    request: BatchTranslationRequest,
    cancellation: &CancellationGuard,
) -> AdapterResult<BatchTranslationOutcome> {
    translate_subtitle_batch_with_progress(request, cancellation, std::sync::Arc::new(NoopProgress))
}

pub fn translate_subtitle_batch_with_progress(
    mut request: BatchTranslationRequest,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
) -> AdapterResult<BatchTranslationOutcome> {
    normalize_translation_languages(&mut request.settings)?;
    request.output_language_tag = request
        .output_language_tag
        .as_deref()
        .map(|value| normalize_language(value, false))
        .transpose()
        .map_err(|error| AdapterError::invalid_input(error.to_string()))?;
    let files = discover_subtitle_files(&request.root, request.recursive)?;
    let total_files = files.len();
    let mut processed = 0usize;
    let mut processed_inputs = Vec::new();
    let mut skipped = Vec::new();
    let mut outputs = Vec::new();
    let mut subtitle_entries = 0usize;
    let mut cache_hits = 0usize;
    let mut resumed_translation_batches = 0usize;
    let mut resumed_review_batches = 0usize;
    let mut translation_memory_hits = 0usize;

    for input_path in files {
        check_cancelled(cancellation)?;
        let output_path = batch_translation_output_path(&request, &input_path)?;
        if output_path.exists() && !request.overwrite && !request.settings.translation.dry_run {
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
                input_path: input_path.clone(),
                output_path: Some(output_path),
                output_language_tag: request.output_language_tag.clone(),
                overwrite: request.overwrite,
                settings: request.settings.clone(),
            },
            cancellation,
            progress.clone(),
        )?;
        if let Some(output_path) = outcome.output_path {
            outputs.push(output_path);
        }
        subtitle_entries += outcome.subtitle_entries;
        cache_hits += outcome.result.cache_hits;
        resumed_translation_batches += outcome.result.resumed_translation_batches;
        resumed_review_batches += outcome.result.resumed_review_batches;
        translation_memory_hits += outcome.result.translation_memory_hits;
        processed_inputs.push(input_path);
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
        inputs: processed_inputs,
        skipped,
        outputs,
        subtitle_entries,
        dry_run: request.settings.translation.dry_run,
        cache_hits,
        resumed_translation_batches,
        resumed_review_batches,
        translation_memory_hits,
    })
}

pub fn batch_translation_output_path(
    request: &BatchTranslationRequest,
    input_path: &Path,
) -> AdapterResult<PathBuf> {
    let output_input = if let Some(output_dir) = &request.output_dir {
        let relative = input_path.strip_prefix(&request.root).map_err(|_| {
            AdapterError::invalid_input(format!(
                "batch input {} is outside root {}",
                input_path.display(),
                request.root.display()
            ))
        })?;
        output_dir.join(relative)
    } else {
        input_path.to_path_buf()
    };
    default_output_path_with_language(
        &output_input,
        request.settings.output_format(),
        request.settings.output.bilingual,
        request.output_language_tag.as_deref(),
    )
}

pub(crate) fn normalize_translation_languages(
    settings: &mut TranslationSettings,
) -> AdapterResult<()> {
    settings.translation.source_language =
        normalize_language(&settings.translation.source_language, true)
            .map_err(|error| AdapterError::invalid_input(error.to_string()))?;
    settings.translation.target_language =
        normalize_language(&settings.translation.target_language, false)
            .map_err(|error| AdapterError::invalid_input(error.to_string()))?;
    Ok(())
}

fn check_cancelled(cancellation: &CancellationGuard) -> AdapterResult<()> {
    cancellation.check().map_err(AdapterError::from)
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

        let mut settings = TranslationSettings::default();
        settings.translation.target_language = "en".to_owned();
        settings.translation.review_policy = subbake_core::ReviewPolicy::Off;
        let outcome = translate_subtitle(TranslationRequest {
            input_path: input_path.clone(),
            output_path: None,
            output_language_tag: None,
            overwrite: true,
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
    fn cached_translation_can_be_rerendered_as_bilingual_output() {
        let root = temp_root("bilingual-cache-rerender");
        fs::create_dir_all(&root).expect("create temp root");
        let input_path = root.join("clip.txt");
        fs::write(&input_path, "hello\n").expect("write input");

        let mut translated_settings = TranslationSettings::default();
        translated_settings.translation.target_language = "en".to_owned();
        translated_settings.translation.review_policy = subbake_core::ReviewPolicy::Off;
        let translated = translate_subtitle(TranslationRequest {
            input_path: input_path.clone(),
            output_path: None,
            output_language_tag: None,
            overwrite: true,
            settings: translated_settings,
        })
        .expect("translate source subtitle");

        let mut bilingual_settings = TranslationSettings::default();
        bilingual_settings.translation.target_language = "en".to_owned();
        bilingual_settings.translation.review_policy = subbake_core::ReviewPolicy::Off;
        bilingual_settings.output.bilingual = true;
        let bilingual = translate_subtitle(TranslationRequest {
            input_path,
            output_path: None,
            output_language_tag: None,
            overwrite: true,
            settings: bilingual_settings,
        })
        .expect("render bilingual subtitle from cache");
        let bilingual_path = bilingual.output_path.expect("bilingual output path");
        let bilingual_text = fs::read_to_string(&bilingual_path).expect("read bilingual output");
        let _ = fs::remove_dir_all(&root);

        assert!(
            translated
                .output_path
                .expect("translated output path")
                .ends_with("clip.translated.txt")
        );
        assert!(bilingual_path.ends_with("clip.bilingual.txt"));
        assert_eq!(bilingual.result.resumed_translation_batches, 1);
        assert_eq!(bilingual_text, "[MOCK-EN] hello\nhello\n");
    }

    #[test]
    fn dry_run_does_not_write_output() {
        let root = temp_root("dry-run");
        fs::create_dir_all(&root).expect("create temp root");
        let input_path = root.join("clip.txt");
        fs::write(&input_path, "hello\n").expect("write input");
        let output_path = root.join("custom.txt");
        let mut settings = TranslationSettings::default();
        settings.translation.dry_run = true;

        let outcome = translate_subtitle(TranslationRequest {
            input_path,
            output_path: Some(output_path.clone()),
            output_language_tag: None,
            overwrite: true,
            settings,
        })
        .expect("dry run");
        let output_exists = output_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(outcome.result.dry_run);
        assert!(!output_exists);
    }

    #[test]
    fn unified_translation_does_not_treat_arbitrary_media_as_subtitles() {
        let error = translate_input(TranslationRequest {
            input_path: PathBuf::from("movie.avi"),
            output_path: None,
            output_language_tag: None,
            overwrite: true,
            settings: TranslationSettings::default(),
        })
        .expect_err("unsupported media translation must require the explicit pipeline");

        assert!(error.to_string().contains("when transcription is required"));
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
            output_dir: None,
            output_language_tag: None,
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
            output_dir: None,
            output_language_tag: None,
            settings: TranslationSettings::default(),
        })
        .expect_err("malformed subtitle should stop the batch");
        let message = error.to_string();
        let _ = fs::remove_dir_all(&root);

        assert!(message.contains(&input_path.display().to_string()));
        assert!(message.contains("Malformed SRT block:\nas a warning."));
    }

    #[test]
    fn single_translation_rejects_existing_output_without_overwrite() {
        let root = temp_root("single-overwrite");
        fs::create_dir_all(&root).expect("create root");
        let input = root.join("clip.txt");
        let output = root.join("clip.translated.txt");
        fs::write(&input, "hello\n").expect("write input");
        fs::write(&output, "existing\n").expect("write output");

        let error = translate_subtitle(TranslationRequest {
            input_path: input,
            output_path: Some(output.clone()),
            output_language_tag: None,
            overwrite: false,
            settings: TranslationSettings::default(),
        })
        .expect_err("existing output must fail");
        let content = fs::read_to_string(&output).expect("read output");
        let _ = fs::remove_dir_all(&root);

        assert!(error.to_string().contains("overwrite is false"));
        assert_eq!(content, "existing\n");
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-translation-{label}-{nanos}"))
    }
}
