use std::io;
use std::path::{Path, PathBuf};

use subbake_core::editing::{build_subtitle_edit_messages, parse_subtitle_edit_payload};
use subbake_core::entities::SubtitleSegment;
use subbake_core::formats::RenderOptions;
use subbake_core::ports::LlmBackend;

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
    pub edit_notes: String,
}

pub fn edit_subtitle(request: SubtitleEditRequest) -> io::Result<SubtitleEditOutcome> {
    if !request.target_path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("subtitle not found: {}", request.target_path.display()),
        ));
    }
    if !is_supported_subtitle_path(&request.target_path) {
        return Err(io::Error::other(format!(
            "unsupported subtitle format: {}",
            request.target_path.display()
        )));
    }
    if !request.allow_non_generated && !is_generated_output(&request.target_path) {
        return Err(io::Error::other(
            "edit_subtitle only edits generated `.translated.*` or `.bilingual.*` files",
        ));
    }

    let document = read_document(&request.target_path)?;
    let source_document = infer_source_document(&request.target_path, document.segments.len())?;
    let messages = build_subtitle_edit_messages(
        &document.segments,
        source_document.as_ref().map(|doc| doc.segments.as_slice()),
        &request.instruction,
        &request.settings.target_language,
    )
    .map_err(io::Error::other)?;

    let mut backend =
        build_backend(&request.settings.backend_config()).map_err(io::Error::other)?;
    let (payload, _) = backend
        .generate_raw_json(&messages)
        .map_err(io::Error::other)?;
    let payload =
        parse_subtitle_edit_payload(payload, &document.segments).map_err(io::Error::other)?;

    let translations = merge_segments(&document.segments, &payload.lines);
    render_and_write_document(
        &document,
        &translations,
        &request.target_path,
        &RenderOptions::new(false, Some(document.format.clone())),
    )?;

    Ok(SubtitleEditOutcome {
        target_path: request.target_path,
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
) -> io::Result<Option<subbake_core::entities::SubtitleDocument>> {
    let Some(source_path) = infer_source_path(target_path) else {
        return Ok(None);
    };
    if !source_path.exists() || !is_supported_subtitle_path(&source_path) {
        return Ok(None);
    }
    let source = read_document(&source_path)?;
    if source.segments.len() == expected_segments {
        Ok(Some(source))
    } else {
        Ok(None)
    }
}

fn infer_source_path(target_path: &Path) -> Option<PathBuf> {
    let file_name = target_path.file_name()?.to_str()?;
    let source_name = file_name.replace(".translated.", ".");
    let source_name = if source_name == file_name {
        file_name.replace(".bilingual.", ".")
    } else {
        source_name
    };
    if source_name == file_name {
        None
    } else {
        Some(target_path.with_file_name(source_name))
    }
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

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-{label}-{nanos}"))
    }
}
