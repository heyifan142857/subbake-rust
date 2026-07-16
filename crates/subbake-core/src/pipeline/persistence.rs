use crate::entities::{PipelineOptions, SubtitleSegment, Usage};
use crate::error::{CoreError, CoreResult};
use crate::memory::ContextMemory;
use crate::ports::{BatchShardKind, RuntimeStore};
use crate::storage::{InputSignature, ResumeSnapshot, RunState, build_translation_fingerprint};
use crate::validation::validate_full_alignment;

pub(super) struct PipelinePersistence<'a> {
    pub options: &'a PipelineOptions,
    pub store: Option<&'a dyn RuntimeStore>,
    pub input_signature: Option<&'a InputSignature>,
}

impl PipelinePersistence<'_> {
    pub(super) fn load_resume_snapshot(
        &self,
        batches: &[Vec<SubtitleSegment>],
        memory: &mut ContextMemory,
    ) -> CoreResult<ResumeSnapshot> {
        if !self.options.resume {
            return Ok(ResumeSnapshot::default());
        }
        let (Some(input_signature), Some(store)) = (self.input_signature, self.store) else {
            return Ok(ResumeSnapshot::default());
        };
        let expected_fingerprint = build_translation_fingerprint(self.options, input_signature);
        let Some(state) = store.load_run_state()? else {
            return Ok(ResumeSnapshot::default());
        };
        let Some(mut snapshot) = state.resume_snapshot(&expected_fingerprint) else {
            return Ok(ResumeSnapshot::default());
        };
        if snapshot.translation_batches_completed > batches.len() {
            return Err(CoreError::DataInvariant(format!(
                "resume state has {} translated batches, but the current input has only {}",
                snapshot.translation_batches_completed,
                batches.len()
            )));
        }

        let expected_segment_count = batches
            .iter()
            .take(snapshot.translation_batches_completed)
            .map(Vec::len)
            .sum::<usize>();
        if snapshot.translation_batches_completed > 0 && snapshot.translated_segments.is_empty() {
            snapshot.translated_segments = store.load_batch_segments(
                BatchShardKind::Translated,
                snapshot.translation_batches_completed,
            )?;
        }
        if snapshot.translated_segments.len() != expected_segment_count {
            return Err(CoreError::DataInvariant(format!(
                "resume state expected {expected_segment_count} translated segments across {} batches, but loaded {}",
                snapshot.translation_batches_completed,
                snapshot.translated_segments.len()
            )));
        }
        let resumed_source = batches
            .iter()
            .take(snapshot.translation_batches_completed)
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        validate_full_alignment(&resumed_source, &snapshot.translated_segments)?;

        if snapshot.review_batches_completed > 0 && snapshot.reviewed_segments.is_empty() {
            snapshot.reviewed_segments = store
                .load_batch_segments(BatchShardKind::Reviewed, snapshot.review_batches_completed)?;
        }
        *memory = snapshot.memory.clone();
        Ok(snapshot)
    }

    pub(super) fn save_run_state(
        &self,
        memory: &ContextMemory,
        translation_batches_completed: usize,
        review_batches_completed: usize,
        validation_completed: bool,
        usage: Usage,
    ) -> CoreResult<()> {
        let (Some(store), Some(input_signature)) = (self.store, self.input_signature) else {
            return Ok(());
        };
        store.save_run_state(&RunState::new(
            self.options,
            input_signature.clone(),
            usage,
            memory.clone(),
            translation_batches_completed,
            review_batches_completed,
            validation_completed,
        ))
    }
}
