use std::path::{Path, PathBuf};

use subbake_core::CancellationGuard;
use subbake_core::editing::{build_subtitle_edit_messages, parse_subtitle_edit_payload};
use subbake_core::entities::SubtitleSegment;
use subbake_core::formats::RenderOptions;
use subbake_core::languages::{is_language_tag, normalize_language};
use subbake_core::ports::{GenerationRequest, LlmBackend};

use crate::error::{AdapterError, AdapterResult};
use crate::fs::{is_supported_subtitle_path, read_document, render_and_write_document};
use crate::providers::build_backend;
use crate::settings::TranslationSettings;

#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleEditRequest {
    pub target_path: PathBuf,
    pub instruction: String,
    pub settings: TranslationSettings,
    pub allow_non_generated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubtitleEditOutcome {
    pub target_path: PathBuf,
    pub target_language: String,
    pub modified_entries: usize,
    pub edit_notes: String,
}

pub fn edit_subtitle(request: SubtitleEditRequest) -> AdapterResult<SubtitleEditOutcome> {
    edit_subtitle_cancellable(request, &CancellationGuard::never())
}

pub fn edit_subtitle_cancellable(
    mut request: SubtitleEditRequest,
    cancellation: &CancellationGuard,
) -> AdapterResult<SubtitleEditOutcome> {
    cancellation.check().map_err(AdapterError::from)?;
    request.settings.translation.target_language =
        normalize_language(&request.settings.translation.target_language, false)
            .map_err(|error| AdapterError::invalid_input(error.to_string()))?;
    if !request.target_path.exists() {
        return Err(AdapterError::invalid_input(format!(
            "subtitle not found: {}",
            request.target_path.display()
        )));
    }
    if !is_supported_subtitle_path(&request.target_path) {
        return Err(AdapterError::invalid_input(format!(
            "unsupported subtitle format: {}",
            request.target_path.display()
        )));
    }
    if !request.allow_non_generated && !is_generated_output(&request.target_path) {
        return Err(AdapterError::invalid_input(
            "edit_subtitle only edits generated `.translated.*` or `.bilingual.*` files",
        ));
    }

    let document = read_document(&request.target_path)?;
    let source_document = infer_source_document(&request.target_path, document.segments.len())?;
    let messages = build_subtitle_edit_messages(
        &document.segments,
        source_document.as_ref().map(|doc| doc.segments.as_slice()),
        &request.instruction,
        &request.settings.translation.target_language,
    )
    .map_err(AdapterError::from)?;

    let mut backend = build_backend(&request.settings.backend_config())?;
    let (payload, _) = backend
        .execute(GenerationRequest::json(messages), cancellation)
        .map_err(AdapterError::from)?
        .into_json()
        .map_err(AdapterError::from)?;
    let payload =
        parse_subtitle_edit_payload(payload, &document.segments).map_err(AdapterError::from)?;

    let modified_entries = document
        .segments
        .iter()
        .zip(&payload.lines)
        .filter(|(segment, line)| segment.text != line.translation)
        .count();
    let translations = merge_segments(&document.segments, &payload.lines);
    cancellation.check().map_err(AdapterError::from)?;
    render_and_write_document(
        &document,
        &translations,
        &request.target_path,
        &RenderOptions::new(false, Some(document.format.clone())),
    )?;

    Ok(SubtitleEditOutcome {
        target_path: request.target_path,
        target_language: request.settings.translation.target_language,
        modified_entries,
        edit_notes: payload.edit_notes,
    })
}

fn merge_segments(
    target_segments: &[SubtitleSegment],
    edited_lines: &[subbake_core::entities::TranslationLine],
) -> Vec<SubtitleSegment> {
    target_segments
        .iter()
        .zip(edited_lines)
        .map(|(segment, line)| SubtitleSegment {
            id: segment.id.clone(),
            text: line.translation.clone(),
            start: segment.start.clone(),
            end: segment.end.clone(),
            identifier: segment.identifier.clone(),
            settings: segment.settings.clone(),
        })
        .collect()
}

fn infer_source_document(
    target_path: &Path,
    expected_segments: usize,
) -> AdapterResult<Option<subbake_core::entities::SubtitleDocument>> {
    for source_path in infer_source_paths(target_path) {
        if !source_path.exists() || !is_supported_subtitle_path(&source_path) {
            continue;
        }
        let source = read_document(&source_path)?;
        if source.segments.len() == expected_segments {
            return Ok(Some(source));
        }
    }
    Ok(None)
}

fn infer_source_paths(target_path: &Path) -> Vec<PathBuf> {
    let Some(file_name) = target_path.file_name().and_then(|value| value.to_str()) else {
        return Vec::new();
    };
    let marker = if file_name.contains(".translated.") {
        ".translated."
    } else if file_name.contains(".bilingual.") {
        ".bilingual."
    } else {
        return Vec::new();
    };
    let Some((prefix, extension)) = file_name.split_once(marker) else {
        return Vec::new();
    };
    let mut candidates = vec![target_path.with_file_name(format!("{prefix}.{extension}"))];
    if let Some((base, possible_language)) = prefix.rsplit_once('.')
        && is_language_tag(possible_language)
    {
        candidates.push(target_path.with_file_name(format!("{base}.{extension}")));
    }
    candidates
}

fn is_generated_output(path: &Path) -> bool {
    path.file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|name| name.contains(".translated.") || name.contains(".bilingual."))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn edits_generated_txt_with_mock_backend() {
        let root = temp_root("edit");
        fs::create_dir_all(&root).expect("create root");
        let path = root.join("clip.translated.txt");
        fs::write(&path, "hello\n").expect("write target");

        let outcome = edit_subtitle(SubtitleEditRequest {
            target_path: path.clone(),
            instruction: "make it uppercase".to_owned(),
            settings: TranslationSettings::default(),
            allow_non_generated: false,
        })
        .expect("edit subtitle");
        let content = fs::read_to_string(&path).expect("read edited file");
        let _ = fs::remove_dir_all(&root);

        assert!(content.contains("HELLO"));
        assert!(!outcome.edit_notes.is_empty());
        assert_eq!(outcome.modified_entries, 1);
    }

    #[test]
    fn rejects_non_generated_input_by_default() {
        let root = temp_root("edit-source");
        fs::create_dir_all(&root).expect("create root");
        let path = root.join("clip.txt");
        fs::write(&path, "hello\n").expect("write source");

        let error = edit_subtitle(SubtitleEditRequest {
            target_path: path,
            instruction: "rewrite".to_owned(),
            settings: TranslationSettings::default(),
            allow_non_generated: false,
        })
        .expect_err("source subtitle should fail");
        let _ = fs::remove_dir_all(&root);

        assert!(error.to_string().contains("generated"));
    }

    #[test]
    fn language_tagged_output_infers_the_original_source() {
        let root = temp_root("edit-language-output");
        fs::create_dir_all(&root).expect("create root");
        let source = root.join("clip.srt");
        let target = root.join("clip.ja.translated.srt");
        fs::write(&source, "1\n00:00:00,000 --> 00:00:01,000\nhello\n").expect("write source");
        fs::write(&target, "1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n").expect("write target");

        let source_document = infer_source_document(&target, 1).expect("infer source document");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(source_document.expect("source document").path, source);
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-{label}-{nanos}"))
    }
}
