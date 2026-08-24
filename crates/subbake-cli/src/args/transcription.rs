use subbake_adapters::{TranscriptionFormat, TranscriptionSettings};

use super::{
    parse_nonnegative_u64, parse_positive_u64, parse_unit_f32, required_path, required_value,
};
use crate::{CliError, CliResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TranscriptionOptionSurface {
    Direct,
    Pipeline,
}

/// Typed accumulator shared by `transcribe` and the transcription half of `pipeline`.
#[derive(Debug, Default)]
pub(super) struct TranscriptionOptionGroup {
    settings: TranscriptionSettings,
}

impl TranscriptionOptionGroup {
    pub(super) fn parse(
        &mut self,
        surface: TranscriptionOptionSurface,
        option: &str,
        args: &[String],
        index: &mut usize,
    ) -> CliResult<bool> {
        let settings = &mut self.settings;
        match (surface, option) {
            (TranscriptionOptionSurface::Direct, "--language")
            | (TranscriptionOptionSurface::Pipeline, "--transcribe-language" | "--language") => {
                settings.language = Some(required_value(args, index, option)?)
            }
            (TranscriptionOptionSurface::Direct, "--model")
            | (
                TranscriptionOptionSurface::Pipeline,
                "--transcribe-model" | "--transcriber-model",
            ) => settings.model = Some(required_value(args, index, option)?),
            (_, "--sidecar") => settings.sidecar_path = Some(required_path(args, index, option)?),
            (TranscriptionOptionSurface::Direct, "--format")
            | (TranscriptionOptionSurface::Pipeline, "--transcribe-format") => {
                let value = required_value(args, index, option)?;
                settings.output_format = TranscriptionFormat::parse(&value).ok_or_else(|| {
                    let flag = if surface == TranscriptionOptionSurface::Direct {
                        "--format"
                    } else {
                        "--transcribe-format"
                    };
                    CliError::usage(format!("{flag} must be one of: srt, vtt, txt"))
                })?;
            }
            (_, "--filter-hallucinations") => settings.filter_hallucinations = true,
            (_, "--no-filter-hallucinations") => settings.filter_hallucinations = false,
            (_, "--normalize-transcript") => settings.normalize_text = Some(true),
            (_, "--no-normalize-transcript") => settings.normalize_text = Some(false),
            (_, "--speaker-labels") => settings.speaker_labels = Some(true),
            (_, "--no-speaker-labels") => settings.speaker_labels = Some(false),
            (TranscriptionOptionSurface::Direct, "--vad")
            | (TranscriptionOptionSurface::Pipeline, "--vad" | "--transcribe-vad") => {
                settings.vad_enabled = Some(true)
            }
            (TranscriptionOptionSurface::Direct, "--no-vad")
            | (TranscriptionOptionSurface::Pipeline, "--no-vad" | "--no-transcribe-vad") => {
                settings.vad_enabled = Some(false)
            }
            (TranscriptionOptionSurface::Direct, "--vad-model")
            | (TranscriptionOptionSurface::Pipeline, "--vad-model" | "--transcribe-vad-model") => {
                settings.vad_model = Some(required_value(args, index, option)?)
            }
            (TranscriptionOptionSurface::Direct, "--vad-threshold")
            | (
                TranscriptionOptionSurface::Pipeline,
                "--vad-threshold" | "--transcribe-vad-threshold",
            ) => settings.vad_threshold = Some(parse_unit_f32(args, index, option)?),
            (_, "--vad-min-speech-duration-ms") => {
                settings.vad_min_speech_duration_ms = Some(parse_positive_u64(args, index, option)?)
            }
            (_, "--vad-min-silence-duration-ms") => {
                settings.vad_min_silence_duration_ms =
                    Some(parse_positive_u64(args, index, option)?)
            }
            (_, "--vad-speech-pad-ms") => {
                settings.vad_speech_pad_ms = Some(parse_nonnegative_u64(args, index, option)?)
            }
            _ => return Ok(false),
        }

        Ok(true)
    }

    pub(super) fn settings_mut(&mut self) -> &mut TranscriptionSettings {
        &mut self.settings
    }

    pub(super) fn into_settings(self) -> TranscriptionSettings {
        self.settings
    }
}
