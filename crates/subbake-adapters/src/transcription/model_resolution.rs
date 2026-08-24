use std::path::PathBuf;

use crate::error::{AdapterError, AdapterResult};
use crate::whisper::{
    default_whisper_binary_path_for, default_whisper_models_dir_for, installed_models_in,
};

use super::{MultipleModelPolicy, TranscriptionSettings};

pub(super) fn locate_whisper_binary(settings: &TranscriptionSettings) -> AdapterResult<PathBuf> {
    let path = settings
        .whisper_binary_path
        .clone()
        .unwrap_or_else(|| default_whisper_binary_path_for(None));
    if path.exists() {
        Ok(path)
    } else {
        Err(AdapterError::invalid_input(format!(
            "whisper-cli not found at `{}`; set `storage.whisper_binary_path` to an existing executable or run `sbake whisper install`",
            path.display()
        )))
    }
}

#[derive(Debug)]
pub(super) struct ResolvedWhisperModel {
    pub(super) name: String,
    pub(super) path: PathBuf,
    pub(super) auto_selected: bool,
}

pub(super) fn resolve_whisper_model(
    settings: &TranscriptionSettings,
) -> AdapterResult<ResolvedWhisperModel> {
    let models_dir = settings
        .whisper_models_dir
        .clone()
        .unwrap_or_else(|| default_whisper_models_dir_for(None));
    let mut installed = if models_dir.is_dir() {
        installed_models_in(&models_dir).map_err(|source| {
            AdapterError::external_io(
                "list installed whisper models",
                Some(models_dir.clone()),
                source,
            )
        })?
    } else {
        Vec::new()
    };
    installed.sort_by(|left, right| model_rank(&left.name).cmp(&model_rank(&right.name)));
    let installed_names = installed
        .iter()
        .map(|model| model.name.clone())
        .collect::<Vec<_>>();

    if let Some(requested) = settings.model.as_deref() {
        return installed
            .into_iter()
            .find(|model| model.name == requested)
            .map(|model| ResolvedWhisperModel {
                name: model.name,
                path: model.path,
                auto_selected: false,
            })
            .ok_or_else(|| {
                AdapterError::invalid_input(format!(
                    "model `{requested}` was not found in `{}`; available models: {}. Set `storage.whisper_models_dir` to your existing model directory or run `sbake whisper model {requested}`.",
                    models_dir.display(),
                    display_model_names(&installed_names)
                ))
            });
    }

    let selected = match installed.as_slice() {
        [] => {
            return Err(AdapterError::invalid_input(format!(
                "no Whisper models were found in `{}`; set `storage.whisper_models_dir` to a directory containing ggml-*.bin or ggml-*.gguf files, or run `sbake whisper model list` followed by `sbake whisper model <NAME>`",
                models_dir.display()
            )));
        }
        [only] => only,
        many => {
            if let Some(small) = many.iter().find(|model| model.name == "small") {
                small
            } else if settings.multiple_model_policy == MultipleModelPolicy::PreferRanked {
                &many[0]
            } else {
                return Err(AdapterError::invalid_input(format!(
                    "multiple whisper.cpp models are installed; specify `--model <NAME>`. Available: {}",
                    display_model_names(&installed_names)
                )));
            }
        }
    };
    Ok(ResolvedWhisperModel {
        name: selected.name.clone(),
        path: selected.path.clone(),
        auto_selected: true,
    })
}

fn display_model_names(names: &[String]) -> String {
    if names.is_empty() {
        "none".to_owned()
    } else {
        names.join(", ")
    }
}

fn model_rank(name: &str) -> (usize, usize, usize, &str) {
    const FAMILIES: &[&str] = &[
        "small",
        "base",
        "medium",
        "large-v3-turbo",
        "large-v3",
        "large-v2",
        "large-v1",
        "tiny",
    ];
    let family = FAMILIES
        .iter()
        .position(|family| {
            name == *family
                || name
                    .strip_prefix(*family)
                    .is_some_and(|suffix| suffix.starts_with('-') || suffix.starts_with('.'))
        })
        .unwrap_or(FAMILIES.len());
    let english_only = usize::from(name.contains(".en"));
    let quantization = if name.contains("q8_") {
        1
    } else if name.contains("q5_") {
        2
    } else {
        0
    };
    (family, english_only, quantization, name)
}
