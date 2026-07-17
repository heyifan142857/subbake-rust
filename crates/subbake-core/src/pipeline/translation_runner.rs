use crate::entities::{ReviewPolicy, SubtitleSegment, TerminologyStats, Usage};
use crate::error::{CoreError, CoreResult};
use crate::ports::{BatchShardKind, DashboardSink, LlmBackend};
use crate::progress::TaskState;
use crate::storage::ResumeSnapshot;
use crate::validation::validate_full_alignment;

use super::SubtitlePipeline;
use super::support::{update_translation_memory, validate_window_terminology};
use super::translation_stage::TranslationStage;

pub(super) struct TranslationRun {
    pub batches: Vec<Vec<SubtitleSegment>>,
    pub segments: Vec<SubtitleSegment>,
    pub usage: Usage,
}

pub(super) fn run<B, D>(
    pipeline: &mut SubtitlePipeline<B, D>,
    document: &crate::entities::SubtitleDocument,
    batches: Vec<Vec<SubtitleSegment>>,
    resume: &ResumeSnapshot,
    terminology: &TerminologyStats,
) -> CoreResult<TranslationRun>
where
    B: LlmBackend,
    D: DashboardSink,
{
    pipeline.report(
        "TRANSLATE",
        if resume.translation_batches_completed > 0 {
            TaskState::Resuming
        } else {
            TaskState::Running
        },
        resume.translation_batches_completed,
        Some(batches.len()),
        resume.translation_batches_completed,
        Usage::default(),
    );
    let mut stage = TranslationStage::new(
        batches,
        resume.translation_batches_completed,
        resume.translated_segments.clone(),
    )?;
    let mut usage = resume.usage;
    if resume.translation_batches_completed == 0 {
        usage.add(terminology.usage);
    }
    if usage != Usage::default() {
        pipeline.dashboard.add_usage(usage);
    }

    while !stage.is_complete() {
        pipeline.cancellation.check()?;
        let concurrency = if pipeline.backend.supports_parallel_generation() {
            pipeline.options.translation_concurrency.max(1)
        } else {
            1
        };
        // Turbo keeps the transport saturated across a second queued wave;
        // adapters still enforce the configured in-flight limit.
        let window_size = if pipeline.options.mode == crate::entities::TranslationMode::Turbo
            && pipeline.backend.supports_parallel_generation()
        {
            concurrency.saturating_mul(2)
        } else {
            concurrency
        };
        let prepared = stage.prepare_window(
            window_size,
            pipeline.options.use_cache,
            &pipeline.translation_memory,
        );
        pipeline.report(
            "TRANSLATE",
            TaskState::Running,
            stage.next_batch(),
            Some(stage.len()),
            resume.translation_batches_completed,
            usage,
        );
        let pending = prepared
            .iter()
            .filter(|batch| !batch.pending.is_empty())
            .map(|batch| (batch.index + 1, batch.pending.clone()))
            .collect::<Vec<_>>();
        pipeline.report_translation_window(
            stage.batches(),
            stage.next_batch(),
            pending.len(),
            resume.translation_batches_completed,
            usage,
        );
        let mut generated = pipeline.translate_window(&pending)?;
        validate_window_terminology(
            &prepared,
            &generated,
            &pipeline.required_glossary,
            pipeline.options.review_policy != ReviewPolicy::Off,
        )?;
        for prepared_batch in prepared {
            let result = if prepared_batch.pending.is_empty() {
                None
            } else {
                Some(
                    generated
                        .remove(&(prepared_batch.index + 1))
                        .ok_or_else(|| {
                            CoreError::DataInvariant(format!(
                                "translation window omitted batch {}",
                                prepared_batch.index + 1
                            ))
                        })?,
                )
            };
            let applied = stage.apply(prepared_batch, result)?;
            if let Some(result) = applied.result.as_ref() {
                usage.add(result.usage);
                pipeline.dashboard.add_usage(result.usage);
                pipeline
                    .memory
                    .update(&result.summary, &result.glossary_updates);
            }
            pipeline.translation_memory_hits = stage.memory_hits();
            if pipeline.options.use_cache {
                update_translation_memory(
                    &mut pipeline.translation_memory,
                    &applied.source,
                    &applied.translated,
                );
            }
            if let Some(store) = pipeline.store.as_ref() {
                pipeline.cancellation.check()?;
                store.save_glossary(
                    &pipeline
                        .memory
                        .glossary
                        .iter()
                        .map(|(source, target)| (source.clone(), target.clone()))
                        .collect::<Vec<_>>(),
                )?;
                store.save_batch_segments(
                    BatchShardKind::Translated,
                    applied.index + 1,
                    &applied.translated,
                )?;
                if pipeline.options.use_cache {
                    store.save_translation_memory(
                        &pipeline
                            .translation_memory
                            .iter()
                            .map(|(key, text)| (key.clone(), text.clone()))
                            .collect::<Vec<_>>(),
                    )?;
                }
            }
            pipeline.cancellation.check()?;
            pipeline.save_run_state(
                applied.index + 1,
                resume.review_batches_completed,
                false,
                usage,
            )?;
            pipeline.report(
                "TRANSLATE",
                TaskState::Running,
                applied.index + 1,
                Some(stage.len()),
                resume.translation_batches_completed,
                usage,
            );
        }
        pipeline.report_translation_window(
            stage.batches(),
            stage.next_batch(),
            0,
            resume.translation_batches_completed,
            usage,
        );
    }

    validate_full_alignment(&document.segments, stage.output())?;
    pipeline.cancellation.check()?;
    pipeline.save_run_state(stage.len(), resume.review_batches_completed, true, usage)?;
    pipeline.dashboard.mark_done("TRANSLATE");
    let batches = stage.batches().to_vec();
    Ok(TranslationRun {
        batches,
        segments: stage.finish(),
        usage,
    })
}
