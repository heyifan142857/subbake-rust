use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use subbake_core::CancellationGuard;
use subbake_core::editing::{
    SUBTITLE_EDIT_MAX_BATCH_ENTRIES, SUBTITLE_EDIT_RESPONSE_TOKEN_BUDGET,
    SubtitleEditTokenEstimate, build_subtitle_edit_messages, distributed_subtitle_edit_indices,
    estimate_subtitle_edit_tokens, parse_subtitle_edit_payload,
};
use subbake_core::entities::{SubtitleSegment, TranslationLine};
use subbake_core::error::CoreError;
use subbake_core::formats::RenderOptions;
use subbake_core::languages::{is_language_tag, normalize_language};
use subbake_core::ports::{GenerationRequest, LlmBackend, RuntimeMemoryStore};
use subbake_core::storage::build_runtime_paths;
use subbake_core::validation::{FinalValidationPolicy, validate_final_output};

use crate::error::{AdapterError, AdapterResult};
use crate::fs::{
    is_supported_subtitle_path, read_document, render_and_write_document, stable_runtime_input_path,
};
use crate::providers::build_backend;
use crate::runtime_store::FileRuntimeStore;
use crate::settings::ResolvedSettings;

#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleEditRequest {
    pub target_path: PathBuf,
    pub instruction: String,
    pub settings: ResolvedSettings,
    pub allow_non_generated: bool,
    pub dry_run: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubtitleEditChange {
    pub id: String,
    pub before: String,
    pub after: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubtitleEditOutcome {
    pub target_path: PathBuf,
    pub target_language: String,
    pub modified_entries: usize,
    pub edit_notes: String,
    pub dry_run: bool,
    #[serde(default)]
    pub processed_entries: usize,
    #[serde(default)]
    pub total_entries: usize,
    #[serde(default)]
    pub partial_preview: bool,
    pub changes: Vec<SubtitleEditChange>,
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
    let source_segments = source_document
        .as_ref()
        .map(|document| document.segments.as_slice());
    let full_messages = build_subtitle_edit_messages(
        &document.segments,
        source_segments,
        &request.instruction,
        &request.settings.translation.target_language,
    )
    .map_err(AdapterError::from)?;
    let full_estimate = estimate_subtitle_edit_tokens(&full_messages, &document.segments);
    let partial_preview = request.dry_run
        && !edit_request_fits(
            full_estimate,
            request.settings.translation.request_token_budget,
        );
    let selected_indices = if partial_preview {
        preview_indices(
            &document.segments,
            source_segments,
            &request.instruction,
            &request.settings.translation.target_language,
            request.settings.translation.request_token_budget,
        )?
    } else {
        (0..document.segments.len()).collect()
    };
    let target_segments = select_segments(&document.segments, &selected_indices);
    let selected_source = source_segments.map(|source| select_segments(source, &selected_indices));

    let mut backend = build_backend(&request.settings.backend_config())?;
    let batches = if partial_preview {
        std::iter::once(0..target_segments.len()).collect()
    } else {
        plan_edit_batches(
            &target_segments,
            selected_source.as_deref(),
            &request.instruction,
            &request.settings.translation.target_language,
            request.settings.translation.request_token_budget,
        )?
    };
    let mut edited_lines = Vec::with_capacity(target_segments.len());
    let mut edit_notes = BTreeSet::new();
    for range in &batches {
        cancellation.check().map_err(AdapterError::from)?;
        let source_batch = selected_source
            .as_deref()
            .map(|source| &source[range.clone()]);
        let payload = execute_edit_batch(
            backend.as_mut(),
            &target_segments[range.clone()],
            source_batch,
            &request.instruction,
            &request.settings.translation.target_language,
            cancellation,
        )?;
        edited_lines.extend(payload.lines);
        if !payload.edit_notes.trim().is_empty() {
            edit_notes.insert(payload.edit_notes.trim().to_owned());
        }
    }

    let changes = target_segments
        .iter()
        .zip(&edited_lines)
        .filter(|(segment, line)| segment.text != line.translation)
        .map(|(segment, line)| SubtitleEditChange {
            id: segment.id.clone(),
            before: segment.text.clone(),
            after: line.translation.clone(),
        })
        .collect::<Vec<_>>();
    let modified_entries = changes.len();
    let translations = merge_segments(&target_segments, &edited_lines);
    let required_glossary = load_required_glossary(&request)?;
    let (validation_source, validation_source_language) = selected_source
        .as_deref()
        .map(|source| {
            (
                source,
                request.settings.translation.source_language.as_str(),
            )
        })
        .unwrap_or((
            target_segments.as_slice(),
            request.settings.translation.target_language.as_str(),
        ));
    validate_final_output(
        validation_source,
        &translations,
        &required_glossary,
        validation_source_language,
        &request.settings.translation.target_language,
        FinalValidationPolicy {
            max_characters_per_second: request.settings.translation.max_characters_per_second,
            max_characters_per_line: request.settings.translation.max_characters_per_line,
            max_lines: request.settings.translation.max_lines,
        },
    )
    .map_err(AdapterError::from)?;
    cancellation.check().map_err(AdapterError::from)?;
    if !request.dry_run {
        debug_assert!(!partial_preview);
        render_and_write_document(
            &document,
            &translations,
            &request.target_path,
            &RenderOptions::new(false, Some(document.format.clone())),
        )?;
    }

    Ok(SubtitleEditOutcome {
        target_path: request.target_path,
        target_language: request.settings.translation.target_language,
        modified_entries,
        edit_notes: edit_notes.into_iter().collect::<Vec<_>>().join(" "),
        dry_run: request.dry_run,
        processed_entries: target_segments.len(),
        total_entries: document.segments.len(),
        partial_preview,
        changes,
    })
}

fn execute_edit_batch(
    backend: &mut dyn LlmBackend,
    target_segments: &[SubtitleSegment],
    source_segments: Option<&[SubtitleSegment]>,
    instruction: &str,
    target_language: &str,
    cancellation: &CancellationGuard,
) -> AdapterResult<subbake_core::SubtitleEditPayload> {
    let messages = build_subtitle_edit_messages(
        target_segments,
        source_segments,
        instruction,
        target_language,
    )
    .map_err(AdapterError::from)?;
    let (payload, _) = backend
        .execute(
            GenerationRequest::json(messages).without_reasoning(),
            cancellation,
        )
        .map_err(AdapterError::from)?
        .into_json()
        .map_err(AdapterError::from)?;
    parse_subtitle_edit_payload(payload, target_segments).map_err(AdapterError::from)
}

fn plan_edit_batches(
    target_segments: &[SubtitleSegment],
    source_segments: Option<&[SubtitleSegment]>,
    instruction: &str,
    target_language: &str,
    request_token_budget: usize,
) -> AdapterResult<Vec<Range<usize>>> {
    let mut batches = Vec::new();
    let mut start = 0usize;
    while start < target_segments.len() {
        let limit = target_segments
            .len()
            .min(start.saturating_add(SUBTITLE_EDIT_MAX_BATCH_ENTRIES));
        let mut accepted_end = start;
        for end in start + 1..=limit {
            let source = source_segments.map(|segments| &segments[start..end]);
            let messages = build_subtitle_edit_messages(
                &target_segments[start..end],
                source,
                instruction,
                target_language,
            )
            .map_err(AdapterError::from)?;
            let estimate = estimate_subtitle_edit_tokens(&messages, &target_segments[start..end]);
            if !edit_request_fits(estimate, request_token_budget) {
                break;
            }
            accepted_end = end;
        }
        if accepted_end == start {
            return Err(AdapterError::from(CoreError::ResourceBudgetExceeded(
                format!(
                    "subtitle edit entry `{}` cannot fit within the configured request/output token budget",
                    target_segments[start].id
                ),
            )));
        }
        batches.push(start..accepted_end);
        start = accepted_end;
    }
    Ok(batches)
}

fn preview_indices(
    target_segments: &[SubtitleSegment],
    source_segments: Option<&[SubtitleSegment]>,
    instruction: &str,
    target_language: &str,
    request_token_budget: usize,
) -> AdapterResult<Vec<usize>> {
    let maximum = target_segments.len().min(SUBTITLE_EDIT_MAX_BATCH_ENTRIES);
    for count in (1..=maximum).rev() {
        let indices = distributed_subtitle_edit_indices(target_segments.len(), count);
        let target = select_segments(target_segments, &indices);
        let source = source_segments.map(|segments| select_segments(segments, &indices));
        let messages =
            build_subtitle_edit_messages(&target, source.as_deref(), instruction, target_language)
                .map_err(AdapterError::from)?;
        if edit_request_fits(
            estimate_subtitle_edit_tokens(&messages, &target),
            request_token_budget,
        ) {
            return Ok(indices);
        }
    }
    Err(AdapterError::from(CoreError::ResourceBudgetExceeded(
        "even one sampled subtitle edit entry cannot fit within the configured request/output token budget"
            .to_owned(),
    )))
}

fn select_segments(segments: &[SubtitleSegment], indices: &[usize]) -> Vec<SubtitleSegment> {
    indices
        .iter()
        .filter_map(|index| segments.get(*index).cloned())
        .collect()
}

fn edit_request_fits(estimate: SubtitleEditTokenEstimate, request_token_budget: usize) -> bool {
    estimate.response <= SUBTITLE_EDIT_RESPONSE_TOKEN_BUDGET
        && (request_token_budget == 0
            || estimate.request.saturating_add(estimate.response) <= request_token_budget)
}

fn load_required_glossary(
    request: &SubtitleEditRequest,
) -> AdapterResult<BTreeMap<String, String>> {
    if request.settings.storage.glossary_path.is_none() {
        return Ok(BTreeMap::new());
    }
    let stable_path = stable_runtime_input_path(&request.target_path)?;
    let paths = build_runtime_paths(
        &request.target_path,
        &stable_path,
        request.settings.storage.runtime_dir.as_deref(),
        request.settings.storage.glossary_path.as_deref(),
        &request.settings.translation.source_language,
        &request.settings.translation.target_language,
        false,
    );
    FileRuntimeStore::new(paths)
        .load_glossary()
        .map(|entries| entries.into_iter().collect())
        .map_err(AdapterError::from)
}

fn merge_segments(
    target_segments: &[SubtitleSegment],
    edited_lines: &[TranslationLine],
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
            semantic: segment.semantic.clone(),
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
            settings: ResolvedSettings::default(),
            allow_non_generated: false,
            dry_run: false,
        })
        .expect("edit subtitle");
        let content = fs::read_to_string(&path).expect("read edited file");
        let _ = fs::remove_dir_all(&root);

        assert!(content.contains("HELLO"));
        assert!(!outcome.edit_notes.is_empty());
        assert_eq!(outcome.modified_entries, 1);
        assert_eq!(outcome.processed_entries, 1);
        assert_eq!(outcome.total_entries, 1);
        assert!(!outcome.partial_preview);
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
            settings: ResolvedSettings::default(),
            allow_non_generated: false,
            dry_run: false,
        })
        .expect_err("source subtitle should fail");
        let _ = fs::remove_dir_all(&root);

        assert!(error.to_string().contains("generated"));
    }

    #[test]
    fn dry_run_returns_diff_without_writing() {
        let root = temp_root("edit-dry-run");
        fs::create_dir_all(&root).expect("create root");
        let path = root.join("clip.translated.txt");
        fs::write(&path, "hello\n").expect("write target");

        let outcome = edit_subtitle(SubtitleEditRequest {
            target_path: path.clone(),
            instruction: "make it uppercase".to_owned(),
            settings: ResolvedSettings::default(),
            allow_non_generated: false,
            dry_run: true,
        })
        .expect("preview subtitle edit");

        assert!(outcome.dry_run);
        assert!(!outcome.partial_preview);
        assert_eq!(outcome.changes.len(), 1);
        assert_eq!(outcome.changes[0].before, "hello");
        assert_eq!(outcome.changes[0].after, "HELLO");
        assert_eq!(
            fs::read_to_string(&path).expect("unchanged target"),
            "hello\n"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn long_dry_run_edits_a_distributed_sample_without_writing() {
        let root = temp_root("edit-long-preview");
        fs::create_dir_all(&root).expect("create root");
        let path = root.join("movie.translated.srt");
        let content = subtitle_with_entries(500);
        fs::write(&path, &content).expect("write target");

        let outcome = edit_subtitle(SubtitleEditRequest {
            target_path: path.clone(),
            instruction: "make it uppercase".to_owned(),
            settings: ResolvedSettings::default(),
            allow_non_generated: false,
            dry_run: true,
        })
        .expect("preview long subtitle edit");

        assert!(outcome.partial_preview);
        assert_eq!(outcome.total_entries, 500);
        assert!(outcome.processed_entries <= SUBTITLE_EDIT_MAX_BATCH_ENTRIES);
        assert!(outcome.processed_entries < outcome.total_entries);
        assert_eq!(
            outcome.changes.first().map(|change| change.id.as_str()),
            Some("1")
        );
        assert_eq!(
            outcome.changes.last().map(|change| change.id.as_str()),
            Some("500")
        );
        assert_eq!(
            fs::read_to_string(&path).expect("unchanged target"),
            content
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn full_edit_planner_splits_large_documents_into_bounded_batches() {
        let segments = (1..=120)
            .map(|index| SubtitleSegment {
                id: index.to_string(),
                text: format!("line {index}"),
                start: None,
                end: None,
                identifier: None,
                settings: None,
                semantic: Default::default(),
            })
            .collect::<Vec<_>>();

        let batches = plan_edit_batches(
            &segments,
            None,
            "rewrite",
            "Chinese",
            ResolvedSettings::default().translation.request_token_budget,
        )
        .expect("plan edit batches");

        assert!(batches.len() >= 3);
        assert_eq!(batches.first(), Some(&(0..SUBTITLE_EDIT_MAX_BATCH_ENTRIES)));
        assert_eq!(batches.last().map(|range| range.end), Some(120));
        assert!(
            batches
                .iter()
                .all(|range| range.len() <= SUBTITLE_EDIT_MAX_BATCH_ENTRIES)
        );
    }

    #[test]
    fn long_real_edit_writes_every_bounded_batch_only_after_completion() {
        let root = temp_root("edit-long-full");
        fs::create_dir_all(&root).expect("create root");
        let path = root.join("movie.translated.srt");
        fs::write(&path, subtitle_with_entries(60)).expect("write target");

        let outcome = edit_subtitle(SubtitleEditRequest {
            target_path: path.clone(),
            instruction: "make it uppercase".to_owned(),
            settings: ResolvedSettings::default(),
            allow_non_generated: false,
            dry_run: false,
        })
        .expect("edit all subtitle batches");
        let edited = fs::read_to_string(&path).expect("read edited target");

        assert!(!outcome.partial_preview);
        assert_eq!(outcome.processed_entries, 60);
        assert_eq!(outcome.total_entries, 60);
        assert_eq!(outcome.modified_entries, 60);
        assert!(edited.contains("LINE NUMBER 1"));
        assert!(edited.contains("LINE NUMBER 60"));
        assert!(!edited.contains("line number"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn editing_leaves_number_changes_to_the_requested_model_edit() {
        let root = temp_root("edit-facts");
        fs::create_dir_all(&root).expect("create root");
        let path = root.join("clip.translated.txt");
        fs::write(&path, "It costs 10 dollars.\n").expect("write target");

        edit_subtitle(SubtitleEditRequest {
            target_path: path.clone(),
            instruction: "change number".to_owned(),
            settings: ResolvedSettings::default(),
            allow_non_generated: false,
            dry_run: false,
        })
        .expect("requested numeric edit should be written");

        assert_eq!(
            fs::read_to_string(&path).expect("edited target"),
            "It costs 11 dollars.\n"
        );
        let _ = fs::remove_dir_all(&root);
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

    fn subtitle_with_entries(count: usize) -> String {
        (1..=count)
            .map(|index| format!("{index}\n00:00:00,000 --> 00:00:01,000\nline number {index}\n"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}
