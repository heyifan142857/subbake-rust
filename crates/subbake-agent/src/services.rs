use std::path::{Path, PathBuf};

use subbake_adapters::{
    AdapterResult, BatchTranslationOutcome, BatchTranslationRequest, SandboxedCommandOutput,
    SandboxedCommandRequest, SubtitleEditOutcome, SubtitleEditRequest, TranscriptionOutcome,
    TranscriptionRequest, TranslationOutcome, TranslationRequest, WhisperOutcome, WhisperRequest,
};
use subbake_core::{CancellationGuard, SharedProgress};

/// Narrow boundary between agent tool decisions and side-effecting adapter services.
pub(crate) trait AgentServices: Send + Sync {
    fn run_command(
        &self,
        request: &SandboxedCommandRequest,
        cancellation: &CancellationGuard,
    ) -> AdapterResult<SandboxedCommandOutput>;

    fn transcribe(
        &self,
        request: TranscriptionRequest,
        cancellation: &CancellationGuard,
        progress: Option<SharedProgress>,
    ) -> AdapterResult<TranscriptionOutcome>;

    fn manage_whisper(
        &self,
        request: WhisperRequest,
        cancellation: &CancellationGuard,
        progress: Option<SharedProgress>,
    ) -> AdapterResult<WhisperOutcome>;

    fn diagnose_path(&self, path: &Path) -> AdapterResult<String>;

    fn default_translation_output_path(
        &self,
        input_path: &Path,
        output_format: Option<&str>,
        bilingual: bool,
        language_tag: Option<&str>,
        preserve_source_container: bool,
    ) -> AdapterResult<PathBuf>;

    fn batch_translation_output_path(
        &self,
        request: &BatchTranslationRequest,
        input_path: &Path,
    ) -> AdapterResult<PathBuf>;

    fn translate(
        &self,
        request: TranslationRequest,
        cancellation: &CancellationGuard,
        progress: Option<SharedProgress>,
    ) -> AdapterResult<TranslationOutcome>;

    fn translate_batch(
        &self,
        request: BatchTranslationRequest,
        cancellation: &CancellationGuard,
        progress: Option<SharedProgress>,
    ) -> AdapterResult<BatchTranslationOutcome>;

    fn edit_subtitle(
        &self,
        request: SubtitleEditRequest,
        cancellation: &CancellationGuard,
    ) -> AdapterResult<SubtitleEditOutcome>;
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct DefaultAgentServices;

impl AgentServices for DefaultAgentServices {
    fn run_command(
        &self,
        request: &SandboxedCommandRequest,
        cancellation: &CancellationGuard,
    ) -> AdapterResult<SandboxedCommandOutput> {
        subbake_adapters::run_sandboxed_command(request, cancellation)
    }

    fn transcribe(
        &self,
        request: TranscriptionRequest,
        cancellation: &CancellationGuard,
        progress: Option<SharedProgress>,
    ) -> AdapterResult<TranscriptionOutcome> {
        if let Some(progress) = progress {
            subbake_adapters::transcribe_media_cancellable_with_progress(
                request,
                cancellation,
                progress,
            )
        } else {
            subbake_adapters::transcribe_media_cancellable(request, cancellation)
        }
    }

    fn manage_whisper(
        &self,
        request: WhisperRequest,
        cancellation: &CancellationGuard,
        progress: Option<SharedProgress>,
    ) -> AdapterResult<WhisperOutcome> {
        if let Some(progress) = progress {
            subbake_adapters::run_whisper_cancellable_with_progress(request, cancellation, progress)
        } else {
            subbake_adapters::run_whisper_cancellable(request, cancellation)
        }
    }

    fn diagnose_path(&self, path: &Path) -> AdapterResult<String> {
        let reports = if path.is_file() {
            vec![subbake_adapters::diagnose_failure_path(path)?]
        } else {
            subbake_adapters::load_diagnostic_reports(path)?
        };
        if reports.is_empty() {
            Ok("No failure logs found.".to_owned())
        } else {
            Ok(reports
                .iter()
                .map(subbake_adapters::format_diagnostic_report)
                .collect::<Vec<_>>()
                .join("\n\n---\n\n"))
        }
    }

    fn default_translation_output_path(
        &self,
        input_path: &Path,
        output_format: Option<&str>,
        bilingual: bool,
        language_tag: Option<&str>,
        preserve_source_container: bool,
    ) -> AdapterResult<PathBuf> {
        subbake_adapters::default_translation_output_path(
            input_path,
            output_format,
            bilingual,
            language_tag,
            preserve_source_container,
        )
    }

    fn batch_translation_output_path(
        &self,
        request: &BatchTranslationRequest,
        input_path: &Path,
    ) -> AdapterResult<PathBuf> {
        subbake_adapters::batch_translation_output_path(request, input_path)
    }

    fn translate(
        &self,
        request: TranslationRequest,
        cancellation: &CancellationGuard,
        progress: Option<SharedProgress>,
    ) -> AdapterResult<TranslationOutcome> {
        if let Some(progress) = progress {
            subbake_adapters::translate_input_cancellable_with_progress(
                request,
                cancellation,
                progress,
            )
        } else {
            subbake_adapters::translate_input_cancellable(request, cancellation)
        }
    }

    fn translate_batch(
        &self,
        request: BatchTranslationRequest,
        cancellation: &CancellationGuard,
        progress: Option<SharedProgress>,
    ) -> AdapterResult<BatchTranslationOutcome> {
        if let Some(progress) = progress {
            subbake_adapters::translate_subtitle_batch_with_progress(
                request,
                cancellation,
                progress,
            )
        } else {
            subbake_adapters::translate_subtitle_batch_cancellable(request, cancellation)
        }
    }

    fn edit_subtitle(
        &self,
        request: SubtitleEditRequest,
        cancellation: &CancellationGuard,
    ) -> AdapterResult<SubtitleEditOutcome> {
        subbake_adapters::edit_subtitle_cancellable(request, cancellation)
    }
}
