use subbake_adapters::{ApiFormat, SettingsOverrides};

use super::{
    parse_batch_size, parse_bilingual_font_scale, parse_nonnegative_usize, parse_positive_f64,
    parse_timeout_seconds, required_path, required_value,
};
use crate::{CliError, CliResult};

/// Typed accumulator shared by translate, batch, pipeline, and memory parsing.
#[derive(Debug, Default)]
pub(super) struct TranslationOptionGroup {
    overrides: SettingsOverrides,
}

impl TranslationOptionGroup {
    pub(super) fn parse(
        &mut self,
        option: &str,
        args: &[String],
        index: &mut usize,
    ) -> CliResult<bool> {
        let overrides = &mut self.overrides;
        match option {
            "--output-format" => {
                overrides.output.format = Some(required_value(args, index, option)?)
            }
            "--bilingual-font-scale" => {
                overrides.output.bilingual_font_scale =
                    Some(parse_bilingual_font_scale(args, index, option)?)
            }
            "--provider" => overrides.backend.id = Some(required_value(args, index, option)?),
            "--model" => overrides.backend.model = Some(required_value(args, index, option)?),
            "--api-key" => overrides.backend.api_key = Some(required_value(args, index, option)?),
            "--base-url" => overrides.backend.base_url = Some(required_value(args, index, option)?),
            "--api-format" => {
                overrides.backend.api_format = Some(
                    ApiFormat::parse(&required_value(args, index, option)?)
                        .map_err(|error| CliError::usage(error.to_string()))?,
                )
            }
            "--endpoint-url" => {
                overrides.backend.endpoint_url = Some(required_value(args, index, option)?)
            }
            "--api-key-env" => {
                overrides.backend.api_key_env = Some(required_value(args, index, option)?)
            }
            "--auth-header" => {
                overrides.backend.auth_header = Some(required_value(args, index, option)?)
            }
            "--auth-prefix" => {
                overrides.backend.auth_prefix = Some(required_value(args, index, option)?)
            }
            "--timeout-seconds" => {
                overrides.backend.timeout_seconds =
                    Some(parse_timeout_seconds(args, index, option)?)
            }
            "--source-lang" => {
                overrides.translation.source_language = Some(required_value(args, index, option)?)
            }
            "--target-lang" => {
                overrides.translation.target_language = Some(required_value(args, index, option)?)
            }
            "--subtitle-stream" => {
                overrides.translation.subtitle_stream_index =
                    Some(parse_nonnegative_usize(args, index, option)?)
            }
            "--batch-size" => {
                overrides.translation.batch_size = Some(parse_batch_size(args, index, option)?)
            }
            "--batch-token-budget" => {
                overrides.translation.batch_token_budget =
                    Some(parse_batch_size(args, index, option)?)
            }
            "--request-token-budget" => {
                overrides.translation.request_token_budget =
                    Some(parse_batch_size(args, index, option)?)
            }
            "--confirmed-context-lines" => {
                overrides.translation.confirmed_context_lines =
                    Some(parse_nonnegative_usize(args, index, option)?)
            }
            "--confirmed-context-token-budget" => {
                overrides.translation.confirmed_context_token_budget =
                    Some(parse_nonnegative_usize(args, index, option)?)
            }
            "--translation-concurrency" => {
                overrides.translation.translation_concurrency =
                    Some(parse_batch_size(args, index, option)?)
            }
            "--review-concurrency" => {
                overrides.translation.review_concurrency =
                    Some(parse_batch_size(args, index, option)?)
            }
            "--max-characters-per-second" => {
                overrides.translation.max_characters_per_second =
                    Some(parse_positive_f64(args, index, option)?)
            }
            "--max-characters-per-line" => {
                overrides.translation.max_characters_per_line =
                    Some(parse_batch_size(args, index, option)?)
            }
            "--max-lines" => {
                overrides.translation.max_lines = Some(parse_batch_size(args, index, option)?)
            }
            "--runtime-dir" => {
                overrides.storage.runtime_dir = Some(required_path(args, index, option)?)
            }
            "--whisper-bin" => {
                overrides.storage.whisper_binary_path = Some(required_path(args, index, option)?)
            }
            "--whisper-models-dir" => {
                overrides.storage.whisper_models_dir = Some(required_path(args, index, option)?)
            }
            "--glossary" => {
                overrides.storage.glossary_path = Some(required_path(args, index, option)?)
            }
            "--bilingual" => overrides.output.bilingual = Some(true),
            "--online-terminology" => overrides.translation.online_terminology = Some(true),
            "--no-online-terminology" => overrides.translation.online_terminology = Some(false),
            "--allow-degraded-preflight" => {
                overrides.translation.allow_degraded_preflight = Some(true)
            }
            "--strict-preflight" => overrides.translation.allow_degraded_preflight = Some(false),
            "--preserve-names" => overrides.translation.preserve_names = Some(true),
            "--transliterate-names" => overrides.translation.preserve_names = Some(false),
            "--preserve-source-container" => {
                overrides.output.preserve_source_container = Some(true)
            }
            "--in-place-container" => overrides.output.preserve_source_container = Some(false),
            "--mode" => {
                overrides.translation.mode = Some(
                    subbake_core::TranslationMode::parse(&required_value(args, index, option)?)
                        .map_err(|error| CliError::usage(error.to_string()))?,
                )
            }
            "--ocr-correction" => {
                overrides.translation.ocr_correction = Some(
                    subbake_core::OcrCorrectionMode::parse(&required_value(args, index, option)?)
                        .map_err(|error| CliError::usage(error.to_string()))?,
                )
            }
            "--no-review" => {
                overrides.translation.review_policy = Some(subbake_core::ReviewPolicy::Off)
            }
            "--review" => {
                overrides.translation.review_policy = Some(
                    subbake_core::ReviewPolicy::parse(&required_value(args, index, option)?)
                        .map_err(|error| CliError::usage(error.to_string()))?,
                )
            }
            "--dry-run" => overrides.translation.dry_run = Some(true),
            "--resume" => overrides.translation.resume = Some(true),
            "--no-resume" => overrides.translation.resume = Some(false),
            "--cache" => overrides.translation.use_cache = Some(true),
            "--no-cache" => overrides.translation.use_cache = Some(false),
            "--retries" => {
                overrides.translation.retries = Some(parse_nonnegative_usize(args, index, option)?)
            }
            "--model-repair" => overrides.translation.model_repair = Some(true),
            "--no-model-repair" => overrides.translation.model_repair = Some(false),
            "--model-repair-attempts" => {
                overrides.translation.model_repair_attempts =
                    Some(parse_nonnegative_usize(args, index, option)?)
            }
            "--max-requests" => {
                overrides.translation.max_requests = Some(parse_batch_size(args, index, option)?)
            }
            "--max-tokens" => {
                overrides.translation.max_tokens = Some(parse_batch_size(args, index, option)?)
            }
            _ => return Ok(false),
        }

        Ok(true)
    }

    pub(super) fn into_overrides(self) -> SettingsOverrides {
        self.overrides
    }
}
