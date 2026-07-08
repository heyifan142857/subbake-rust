use std::io;
use std::path::PathBuf;

use crate::fs::is_supported_subtitle_path;
use crate::settings::TranslationSettings;
use crate::transcription::{
    TranscriptionRequest, TranscriptionSettings, transcribe_media,
};
use crate::translation::{TranslationOutcome, TranslationRequest, translate_subtitle};

#[derive(Debug, Clone, PartialEq)]
pub struct PipelineRequest {
    pub input_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub settings: TranslationSettings,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineOutcome {
    Subtitle(TranslationOutcome),
}

pub fn run_pipeline(request: PipelineRequest) -> io::Result<PipelineOutcome> {
    if is_supported_subtitle_path(&request.input_path) {
        let outcome = translate_subtitle(TranslationRequest {
            input_path: request.input_path,
            output_path: request.output_path,
            settings: request.settings,
        })?;
        return Ok(PipelineOutcome::Subtitle(outcome));
    }

    // Media input: transcribe first, then translate the result.
    let transcription_out = transcribe_media(TranscriptionRequest {
        media_path: request.input_path.clone(),
        output_path: None,
        settings: TranscriptionSettings::default(),
    })?;
    let transcribed_path = transcription_out.output_path;

    // Build TranslationRequest — use the transcription output as input.
    let translation_out = translate_subtitle(TranslationRequest {
        input_path: transcribed_path,
        output_path: request.output_path,
        settings: request.settings,
    })?;

    Ok(PipelineOutcome::Subtitle(translation_out))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn subtitle_inputs_use_translation_service() {
        let root = temp_root("subtitle");
        fs::create_dir_all(&root).expect("create temp root");
        let input_path = root.join("clip.txt");
        fs::write(&input_path, "hello\n").expect("write input");
        let settings = TranslationSettings {
            target_language: "English".to_owned(),
            final_review: false,
            ..TranslationSettings::default()
        };

        let outcome = run_pipeline(PipelineRequest {
            input_path,
            output_path: None,
            settings,
        })
        .expect("run pipeline");
        let output_path = match outcome {
            PipelineOutcome::Subtitle(outcome) => outcome.output_path.expect("output path"),
        };
        let output = fs::read_to_string(&output_path).expect("read output");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(output, "[MOCK-EN] hello\n");
    }

    #[test]
    fn media_input_chains_transcription_then_translation() {
        // Use .wav to skip ffmpeg; a non-existent file means the pipeline tries
        // transcription (= no longer "pending migration").
        let error = run_pipeline(PipelineRequest {
            input_path: PathBuf::from("audio.wav"),
            output_path: None,
            settings: TranslationSettings::default(),
        })
        .expect_err("media pipeline should try transcription");

        let msg = error.to_string();
        assert!(!msg.contains("pending migration"),
            "media path should no longer return pending migration: {msg}");
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-pipeline-service-{label}-{nanos}"))
    }
}
