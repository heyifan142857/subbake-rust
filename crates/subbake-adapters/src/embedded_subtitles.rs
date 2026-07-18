//! Translation of text subtitle streams embedded in common media containers.
//!
//! The original container is never modified in place while FFmpeg is running:
//! a translated subtitle is extracted into a temporary workspace, translated
//! through the normal subtitle pipeline, then appended to a temporary sibling
//! container before an atomic rename.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde::Deserialize;
use subbake_core::storage::input_signature_from_bytes;
use subbake_core::{
    CancellationGuard, NoopProgress, ProgressEvent, ProgressUnit, SharedProgress, TaskKind,
    TaskState,
};

use crate::error::{AdapterError, AdapterResult};
use crate::process::run_command_cancellable;
use crate::translation::{
    TranslationInputIdentity, TranslationOutcome, TranslationRequest,
    normalize_translation_languages, translate_subtitle_cancellable_with_progress_and_identity,
};

const FFMPEG: &str = "ffmpeg";
const FFPROBE: &str = "ffprobe";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubtitleContainerKind {
    Matroska,
    Mp4,
    Mov,
    WebM,
}

struct EmbedSubtitleRequest<'a> {
    input_path: &'a Path,
    streams: &'a [SubtitleStream],
    subtitle_path: &'a Path,
    output_path: &'a Path,
    target_language: &'a str,
    container_kind: SubtitleContainerKind,
}

impl SubtitleContainerKind {
    fn from_path(path: &Path) -> Option<Self> {
        match path
            .extension()
            .and_then(OsStr::to_str)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("mkv") => Some(Self::Matroska),
            Some("mp4" | "m4v") => Some(Self::Mp4),
            Some("mov") => Some(Self::Mov),
            Some("webm") => Some(Self::WebM),
            _ => None,
        }
    }

    fn subtitle_codec(self) -> &'static str {
        match self {
            Self::Matroska => "srt",
            Self::Mp4 | Self::Mov => "mov_text",
            Self::WebM => "webvtt",
        }
    }

    fn muxer(self) -> &'static str {
        match self {
            Self::Matroska => "matroska",
            Self::Mp4 => "mp4",
            Self::Mov => "mov",
            Self::WebM => "webm",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SubtitleStream {
    index: usize,
    codec: String,
    language: Option<String>,
    title: Option<String>,
    default: bool,
    forced: bool,
}

#[derive(Debug, Deserialize)]
struct ProbeResponse {
    #[serde(default)]
    streams: Vec<ProbeStream>,
}

#[derive(Debug, Deserialize)]
struct ProbeStream {
    index: usize,
    codec_name: Option<String>,
    codec_type: Option<String>,
    #[serde(default)]
    tags: ProbeTags,
    #[serde(default)]
    disposition: ProbeDisposition,
}

#[derive(Debug, Default, Deserialize)]
struct ProbeTags {
    language: Option<String>,
    title: Option<String>,
    handler_name: Option<String>,
    name: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ProbeDisposition {
    default: i64,
    forced: i64,
}

pub fn is_supported_subtitle_container_path(path: &Path) -> bool {
    SubtitleContainerKind::from_path(path).is_some()
}

pub fn default_embedded_translation_output_path(
    input_path: &Path,
    bilingual: bool,
    language_tag: Option<&str>,
    preserve_source_container: bool,
) -> AdapterResult<PathBuf> {
    if !is_supported_subtitle_container_path(input_path) {
        return Err(AdapterError::invalid_input(format!(
            "unsupported subtitle container: {}",
            input_path.display()
        )));
    }
    if !preserve_source_container {
        return Ok(input_path.to_path_buf());
    }
    let stem = input_path
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("output");
    let language = language_tag
        .filter(|value| !value.trim().is_empty())
        .map(|value| format!(".{}", value.trim()))
        .unwrap_or_default();
    let suffix = if bilingual { "bilingual" } else { "translated" };
    let extension = input_path
        .extension()
        .and_then(OsStr::to_str)
        .unwrap_or("mkv");
    Ok(input_path.with_file_name(format!("{stem}{language}.{suffix}.{extension}")))
}

pub fn translate_embedded_subtitle(
    request: TranslationRequest,
) -> AdapterResult<TranslationOutcome> {
    translate_embedded_subtitle_cancellable_with_progress(
        request,
        &CancellationGuard::never(),
        std::sync::Arc::new(NoopProgress),
    )
}

pub fn translate_embedded_subtitle_cancellable(
    request: TranslationRequest,
    cancellation: &CancellationGuard,
) -> AdapterResult<TranslationOutcome> {
    translate_embedded_subtitle_cancellable_with_progress(
        request,
        cancellation,
        std::sync::Arc::new(NoopProgress),
    )
}

pub fn translate_embedded_subtitle_cancellable_with_progress(
    request: TranslationRequest,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
) -> AdapterResult<TranslationOutcome> {
    translate_embedded_subtitle_with_programs(
        request,
        cancellation,
        progress,
        Path::new(FFMPEG),
        Path::new(FFPROBE),
    )
}

/// Remove a SubBake-managed subtitle track without retaining a full media
/// backup. This powers the interactive agent's semantic undo operation.
pub fn remove_embedded_subtitle_by_title(
    input_path: &Path,
    title: &str,
    cancellation: &CancellationGuard,
) -> AdapterResult<()> {
    let container_kind = SubtitleContainerKind::from_path(input_path).ok_or_else(|| {
        AdapterError::invalid_input(format!(
            "unsupported subtitle container: {}",
            input_path.display()
        ))
    })?;
    let streams = probe_subtitle_streams(Path::new(FFPROBE), input_path, cancellation)?;
    let removed = streams
        .iter()
        .filter(|stream| stream.title.as_deref() == Some(title))
        .map(|stream| stream.index)
        .collect::<Vec<_>>();
    if removed.is_empty() {
        return Err(AdapterError::invalid_input(format!(
            "container no longer contains the SubBake subtitle track `{title}`"
        )));
    }
    let staged = temporary_output_path(input_path);
    let mut cleanup = TemporaryOutput::new(staged.clone());
    remux_without_streams(
        Path::new(FFMPEG),
        input_path,
        &staged,
        &removed,
        container_kind,
        cancellation,
    )?;
    let remaining = probe_subtitle_streams(Path::new(FFPROBE), &staged, cancellation)?;
    if remaining
        .iter()
        .any(|stream| stream.title.as_deref() == Some(title))
    {
        return Err(AdapterError::ChildProcess {
            program: "ffprobe",
            status: None,
            message: "translated subtitle track remained after container undo".to_owned(),
        });
    }
    preserve_permissions(input_path, &staged)?;
    cancellation.check().map_err(AdapterError::from)?;
    replace_file(&staged, input_path)?;
    cleanup.disarm();
    Ok(())
}

/// Restore a previously replaced SubBake-managed track from an SRT sidecar.
/// The container itself is staged, verified, and atomically replaced.
pub fn restore_embedded_subtitle_from_srt(
    input_path: &Path,
    title: &str,
    subtitle_path: &Path,
    cancellation: &CancellationGuard,
) -> AdapterResult<()> {
    let container_kind = SubtitleContainerKind::from_path(input_path).ok_or_else(|| {
        AdapterError::invalid_input(format!(
            "unsupported subtitle container: {}",
            input_path.display()
        ))
    })?;
    let target_language = title
        .strip_suffix(" (SubBake translation)")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| AdapterError::invalid_input("invalid SubBake subtitle title"))?;
    if !subtitle_path.is_file() {
        return Err(AdapterError::invalid_input(format!(
            "embedded subtitle undo payload does not exist: {}",
            subtitle_path.display()
        )));
    }
    let streams = probe_subtitle_streams(Path::new(FFPROBE), input_path, cancellation)?;
    let staged = temporary_output_path(input_path);
    let mut cleanup = TemporaryOutput::new(staged.clone());
    embed_subtitle(
        Path::new(FFMPEG),
        cancellation,
        &EmbedSubtitleRequest {
            input_path,
            streams: &streams,
            subtitle_path,
            output_path: &staged,
            target_language,
            container_kind,
        },
    )?;
    let restored = probe_subtitle_streams(Path::new(FFPROBE), &staged, cancellation)?;
    if !restored
        .iter()
        .any(|stream| stream.title.as_deref() == Some(title))
    {
        return Err(AdapterError::ChildProcess {
            program: "ffprobe",
            status: None,
            message: "restored container is missing the previous subtitle track".to_owned(),
        });
    }
    preserve_permissions(input_path, &staged)?;
    cancellation.check().map_err(AdapterError::from)?;
    replace_file(&staged, input_path)?;
    cleanup.disarm();
    Ok(())
}

fn translate_embedded_subtitle_with_programs(
    mut request: TranslationRequest,
    cancellation: &CancellationGuard,
    progress: SharedProgress,
    ffmpeg: &Path,
    ffprobe: &Path,
) -> AdapterResult<TranslationOutcome> {
    cancellation.check().map_err(AdapterError::from)?;
    normalize_translation_languages(&mut request.settings)?;
    request.output_language_tag = request
        .output_language_tag
        .as_deref()
        .map(|value| subbake_core::languages::normalize_language(value, false))
        .transpose()
        .map_err(|error| AdapterError::invalid_input(error.to_string()))?;
    if !is_supported_subtitle_container_path(&request.input_path) {
        return Err(AdapterError::invalid_input(format!(
            "unsupported subtitle container: {}",
            request.input_path.display()
        )));
    }

    let container_input = request.input_path.clone();
    let container_kind = SubtitleContainerKind::from_path(&container_input).ok_or_else(|| {
        AdapterError::invalid_input(format!(
            "unsupported subtitle container: {}",
            container_input.display()
        ))
    })?;
    let final_output = match request.output_path.clone() {
        Some(path) => path,
        None => default_embedded_translation_output_path(
            &request.input_path,
            request.settings.output.bilingual,
            request.output_language_tag.as_deref(),
            request.settings.output.preserve_source_container,
        )?,
    };
    if SubtitleContainerKind::from_path(&final_output) != Some(container_kind) {
        return Err(AdapterError::invalid_input(format!(
            "embedded subtitle output must keep the source container format: {}",
            final_output.display()
        )));
    }
    let in_place = same_path(&container_input, &final_output);
    if final_output.exists()
        && !in_place
        && !request.overwrite
        && !request.settings.translation.dry_run
    {
        return Err(AdapterError::invalid_input(format!(
            "output already exists and overwrite is false: {}",
            final_output.display()
        )));
    }

    emit_stage(&progress, "INSPECT_SUBTITLES", TaskState::Running);
    let streams = probe_subtitle_streams(ffprobe, &request.input_path, cancellation)?;
    let source = select_text_stream(&streams, &request.settings.translation.source_language)?;
    emit_stage(&progress, "INSPECT_SUBTITLES", TaskState::Completed);

    let temporary = unique_temp_dir()?;
    let extracted_path = temporary.path().join("source.srt");
    emit_stage(&progress, "EXTRACT_SUBTITLE", TaskState::Running);
    extract_subtitle(
        ffmpeg,
        &request.input_path,
        source.index,
        &extracted_path,
        cancellation,
    )?;
    emit_stage(&progress, "EXTRACT_SUBTITLE", TaskState::Completed);

    let extracted_bytes = fs::read(&extracted_path).map_err(|source| {
        AdapterError::external_io(
            "read extracted subtitle",
            Some(extracted_path.clone()),
            source,
        )
    })?;
    let identity = TranslationInputIdentity {
        path: embedded_stream_identity(&request.input_path, source.index)?,
        signature: input_signature_from_bytes(&extracted_bytes, None),
        output_path: final_output.clone(),
    };
    let translated_path = temporary.path().join("translated.srt");
    request.input_path = extracted_path;
    request.output_path = Some(translated_path.clone());
    request.settings.output.format = Some("srt".to_owned());
    let target_language = request.settings.translation.target_language.clone();
    let mut outcome = translate_subtitle_cancellable_with_progress_and_identity(
        request,
        cancellation,
        progress.clone(),
        Some(identity),
    )?;

    if outcome.subtitle_entries == 0 {
        return Err(AdapterError::invalid_input(format!(
            "selected embedded subtitle stream {} contains no subtitle entries",
            source.index
        )));
    }

    if outcome.result.dry_run {
        outcome.output_path = None;
        outcome.result.output_path = None;
        return Ok(outcome);
    }

    let expected_title = translated_subtitle_title(&target_language);
    let previous_streams = streams
        .iter()
        .filter(|stream| stream.title.as_deref() == Some(expected_title.as_str()))
        .collect::<Vec<_>>();
    if previous_streams.len() > 1 {
        return Err(AdapterError::invalid_input(format!(
            "container contains multiple SubBake tracks titled `{expected_title}`; remove duplicates before translating so the operation remains safely undoable"
        )));
    }
    let previous_subtitle = if let Some(previous) = previous_streams.first() {
        let path = temporary.path().join("previous-translation.srt");
        extract_subtitle(
            ffmpeg,
            &container_input,
            previous.index,
            &path,
            cancellation,
        )?;
        Some(fs::read(&path).map_err(|source| {
            AdapterError::external_io("read previous translated subtitle", Some(path), source)
        })?)
    } else {
        None
    };

    let parent = final_output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent).map_err(|source| {
            AdapterError::external_io(
                "create container output directory",
                Some(parent.to_path_buf()),
                source,
            )
        })?;
    }
    let mut staged_output = TemporaryOutput::new(temporary_output_path(&final_output));
    emit_stage(&progress, "EMBED_SUBTITLE", TaskState::Running);
    embed_subtitle(
        ffmpeg,
        cancellation,
        &EmbedSubtitleRequest {
            input_path: &container_input,
            streams: &streams,
            subtitle_path: &translated_path,
            output_path: &staged_output.path,
            target_language: &target_language,
            container_kind,
        },
    )?;
    let embedded_streams = probe_subtitle_streams(ffprobe, &staged_output.path, cancellation)?;
    if !embedded_streams
        .iter()
        .any(|stream| stream.title.as_deref() == Some(expected_title.as_str()))
    {
        return Err(AdapterError::ChildProcess {
            program: "ffprobe",
            status: None,
            message: "translated container is missing the appended subtitle track".to_owned(),
        });
    }
    preserve_permissions(&container_input, &staged_output.path)?;
    cancellation.check().map_err(AdapterError::from)?;
    replace_file(&staged_output.path, &final_output)?;
    staged_output.disarm();
    emit_stage(&progress, "EMBED_SUBTITLE", TaskState::Completed);

    outcome.output_path = Some(final_output.clone());
    outcome.result.output_path = Some(final_output);
    outcome.container_change = Some(crate::translation::ContainerTranslationChange {
        in_place,
        subtitle_title: translated_subtitle_title(&target_language),
        previous_subtitle,
    });
    Ok(outcome)
}

fn probe_subtitle_streams(
    ffprobe: &Path,
    input_path: &Path,
    cancellation: &CancellationGuard,
) -> AdapterResult<Vec<SubtitleStream>> {
    let output = run_command_cancellable(
        Command::new(ffprobe).args([
            OsStr::new("-v"),
            OsStr::new("error"),
            OsStr::new("-show_entries"),
            OsStr::new(
                "stream=index,codec_name,codec_type:stream_tags=language,title,handler_name,name:stream_disposition=default,forced",
            ),
            OsStr::new("-of"),
            OsStr::new("json"),
            input_path.as_os_str(),
        ]),
        cancellation,
        "ffprobe embedded subtitle streams",
    )?;
    if !output.status.success() {
        return Err(child_process_error(
            "ffprobe",
            &output,
            "failed to inspect embedded subtitle streams",
        ));
    }
    let response: ProbeResponse =
        serde_json::from_slice(&output.stdout).map_err(|source| AdapterError::Serialization {
            context: "parse ffprobe subtitle stream response",
            source,
        })?;
    Ok(response
        .streams
        .into_iter()
        .filter(|stream| stream.codec_type.as_deref() == Some("subtitle"))
        .map(|stream| SubtitleStream {
            index: stream.index,
            codec: stream.codec_name.unwrap_or_else(|| "unknown".to_owned()),
            language: stream.tags.language,
            title: stream
                .tags
                .title
                .or(stream.tags.handler_name)
                .or(stream.tags.name),
            default: stream.disposition.default != 0,
            forced: stream.disposition.forced != 0,
        })
        .collect())
}

fn select_text_stream<'a>(
    streams: &'a [SubtitleStream],
    source_language: &str,
) -> AdapterResult<&'a SubtitleStream> {
    let text_streams = streams
        .iter()
        .filter(|stream| {
            is_text_subtitle_codec(&stream.codec)
                && !stream
                    .title
                    .as_deref()
                    .is_some_and(is_subbake_translation_title)
        })
        .collect::<Vec<_>>();
    if text_streams.is_empty() {
        let message = if streams.is_empty() {
            "container contains no embedded subtitle streams".to_owned()
        } else {
            format!(
                "container contains no translatable text subtitle stream; found only: {}",
                streams
                    .iter()
                    .map(|stream| {
                        stream.title.as_ref().map_or_else(
                            || stream.codec.clone(),
                            |title| format!("{} ({title})", stream.codec),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        return Err(AdapterError::invalid_input(message));
    }

    let requested_language = normalized_primary_language(source_language);
    let candidates = if requested_language
        .as_deref()
        .is_some_and(|value| value != "auto")
    {
        let matching = text_streams
            .iter()
            .copied()
            .filter(|stream| {
                stream
                    .language
                    .as_deref()
                    .and_then(normalized_primary_language)
                    .as_deref()
                    == requested_language.as_deref()
            })
            .collect::<Vec<_>>();
        if matching.is_empty() {
            let available = text_streams
                .iter()
                .map(|stream| {
                    format!(
                        "{}:{}",
                        stream.index,
                        stream.language.as_deref().unwrap_or("und")
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            return Err(AdapterError::invalid_input(format!(
                "no text subtitle stream matches source language `{source_language}`; available streams: {available}"
            )));
        }
        matching
    } else {
        text_streams
    };
    candidates
        .iter()
        .copied()
        .find(|stream| stream.default && !stream.forced)
        .or_else(|| candidates.iter().copied().find(|stream| !stream.forced))
        .or_else(|| candidates.first().copied())
        .ok_or_else(|| AdapterError::invalid_input("container contains no selectable subtitle"))
}

fn is_text_subtitle_codec(codec: &str) -> bool {
    matches!(
        codec,
        "subrip"
            | "srt"
            | "ass"
            | "ssa"
            | "webvtt"
            | "mov_text"
            | "text"
            | "microdvd"
            | "mpl2"
            | "jacosub"
            | "sami"
            | "realtext"
            | "subviewer"
            | "subviewer1"
    )
}

fn normalized_primary_language(value: &str) -> Option<String> {
    let normalized = subbake_core::languages::normalize_language_name(value, true);
    (normalized != "und").then(|| {
        normalized
            .split('-')
            .next()
            .unwrap_or(normalized.as_str())
            .to_ascii_lowercase()
    })
}

fn extract_subtitle(
    ffmpeg: &Path,
    input_path: &Path,
    stream_index: usize,
    output_path: &Path,
    cancellation: &CancellationGuard,
) -> AdapterResult<()> {
    let map = format!("0:{stream_index}");
    let output = run_command_cancellable(
        Command::new(ffmpeg).args([
            OsStr::new("-nostdin"),
            OsStr::new("-hide_banner"),
            OsStr::new("-v"),
            OsStr::new("error"),
            OsStr::new("-y"),
            OsStr::new("-i"),
            input_path.as_os_str(),
            OsStr::new("-map"),
            OsStr::new(&map),
            OsStr::new("-vn"),
            OsStr::new("-an"),
            OsStr::new("-dn"),
            OsStr::new("-c:s"),
            OsStr::new("srt"),
            output_path.as_os_str(),
        ]),
        cancellation,
        "ffmpeg embedded subtitle extraction",
    )?;
    if !output.status.success() {
        return Err(child_process_error(
            "ffmpeg",
            &output,
            "failed to extract embedded subtitle as SRT",
        ));
    }
    if !output_path.is_file() {
        return Err(AdapterError::ChildProcess {
            program: "ffmpeg",
            status: output.status.code(),
            message: "ffmpeg did not create the extracted subtitle".to_owned(),
        });
    }
    Ok(())
}

fn embed_subtitle(
    ffmpeg: &Path,
    cancellation: &CancellationGuard,
    request: &EmbedSubtitleRequest<'_>,
) -> AdapterResult<()> {
    let title = translated_subtitle_title(request.target_language);
    let replaced_streams = request
        .streams
        .iter()
        .filter(|stream| stream.title.as_deref() == Some(title.as_str()))
        .map(|stream| stream.index)
        .collect::<Vec<_>>();
    let subtitle_ordinal = request.streams.len().saturating_sub(replaced_streams.len());
    let mut args = embed_subtitle_args(request, subtitle_ordinal, &title, &replaced_streams);
    let mut command = Command::new(ffmpeg);
    command.args(args.drain(..));
    let output = run_command_cancellable(
        &mut command,
        cancellation,
        "ffmpeg translated subtitle embedding",
    )?;
    if !output.status.success() {
        return Err(child_process_error(
            "ffmpeg",
            &output,
            "failed to embed translated subtitle",
        ));
    }
    if !request.output_path.is_file() {
        return Err(AdapterError::ChildProcess {
            program: "ffmpeg",
            status: output.status.code(),
            message: "ffmpeg did not create the translated container".to_owned(),
        });
    }
    Ok(())
}

fn embed_subtitle_args(
    request: &EmbedSubtitleRequest<'_>,
    subtitle_ordinal: usize,
    title: &str,
    replaced_streams: &[usize],
) -> Vec<OsString> {
    let language_metadata = format!("language={}", matroska_language(request.target_language));
    let title_metadata = format!("title={title}");
    let handler_metadata = format!("handler_name={title}");
    let mut args = vec![
        "-nostdin".into(),
        "-hide_banner".into(),
        "-v".into(),
        "error".into(),
        "-y".into(),
        "-i".into(),
        request.input_path.as_os_str().to_owned(),
        "-i".into(),
        request.subtitle_path.as_os_str().to_owned(),
        "-map".into(),
        "0".into(),
    ];
    for stream_index in replaced_streams {
        args.push("-map".into());
        args.push(format!("-0:{stream_index}").into());
    }
    args.extend([
        "-map".into(),
        "1:0".into(),
        "-map_metadata".into(),
        "0".into(),
        "-map_chapters".into(),
        "0".into(),
        "-c".into(),
        "copy".into(),
        format!("-metadata:s:s:{subtitle_ordinal}").into(),
        language_metadata.into(),
        format!("-metadata:s:s:{subtitle_ordinal}").into(),
        title_metadata.into(),
        format!("-metadata:s:s:{subtitle_ordinal}").into(),
        handler_metadata.into(),
        format!("-disposition:s:{subtitle_ordinal}").into(),
        "0".into(),
        format!("-c:s:{subtitle_ordinal}").into(),
        request.container_kind.subtitle_codec().into(),
        "-f".into(),
        request.container_kind.muxer().into(),
        request.output_path.as_os_str().to_owned(),
    ]);
    args
}

fn remux_without_streams(
    ffmpeg: &Path,
    input_path: &Path,
    output_path: &Path,
    removed_streams: &[usize],
    container_kind: SubtitleContainerKind,
    cancellation: &CancellationGuard,
) -> AdapterResult<()> {
    let mut args = vec![
        OsString::from("-nostdin"),
        "-hide_banner".into(),
        "-v".into(),
        "error".into(),
        "-y".into(),
        "-i".into(),
        input_path.as_os_str().to_owned(),
        "-map".into(),
        "0".into(),
    ];
    for stream_index in removed_streams {
        args.push("-map".into());
        args.push(format!("-0:{stream_index}").into());
    }
    args.extend([
        "-map_metadata".into(),
        "0".into(),
        "-map_chapters".into(),
        "0".into(),
        "-c".into(),
        "copy".into(),
        "-f".into(),
        container_kind.muxer().into(),
        output_path.as_os_str().to_owned(),
    ]);
    let output = run_command_cancellable(
        Command::new(ffmpeg).args(args),
        cancellation,
        "ffmpeg embedded subtitle undo",
    )?;
    if !output.status.success() {
        return Err(child_process_error(
            "ffmpeg",
            &output,
            "failed to remove embedded subtitle track",
        ));
    }
    if !output_path.is_file() {
        return Err(AdapterError::ChildProcess {
            program: "ffmpeg",
            status: output.status.code(),
            message: "ffmpeg did not create the container undo output".to_owned(),
        });
    }
    Ok(())
}

fn matroska_language(language: &str) -> &str {
    match language.split('-').next().unwrap_or(language) {
        "zh" => "chi",
        "en" => "eng",
        "ja" => "jpn",
        "ko" => "kor",
        "fr" => "fre",
        "es" => "spa",
        "de" => "ger",
        "pt" => "por",
        "ru" => "rus",
        "it" => "ita",
        "ar" => "ara",
        "hi" => "hin",
        "nl" => "dut",
        "pl" => "pol",
        "tr" => "tur",
        "uk" => "ukr",
        "vi" => "vie",
        "th" => "tha",
        "id" => "ind",
        _ => language,
    }
}

fn translated_subtitle_title(target_language: &str) -> String {
    format!("{target_language} (SubBake translation)")
}

fn is_subbake_translation_title(title: &str) -> bool {
    title.ends_with(" (SubBake translation)")
}

fn same_path(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn embedded_stream_identity(input_path: &Path, stream_index: usize) -> AdapterResult<PathBuf> {
    let stable_input = input_path.canonicalize().or_else(|_| {
        if input_path.is_absolute() {
            Ok(input_path.to_path_buf())
        } else {
            std::env::current_dir().map(|dir| dir.join(input_path))
        }
    })?;
    let name = stable_input
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("container.mkv");
    Ok(stable_input.with_file_name(format!(
        ".{name}.subbake-subtitle-stream-{stream_index}.srt"
    )))
}

fn emit_stage(progress: &SharedProgress, stage: &str, state: TaskState) {
    let mut event =
        ProgressEvent::running(TaskKind::Pipeline, stage, 1, Some(1), ProgressUnit::Steps);
    event.state = state;
    progress.emit(event);
}

fn child_process_error(program: &'static str, output: &Output, fallback: &str) -> AdapterError {
    AdapterError::ChildProcess {
        program,
        status: output.status.code(),
        message: child_diagnostics(output, fallback),
    }
}

fn child_diagnostics(output: &Output, fallback: &str) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if stderr.is_empty() {
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if stdout.is_empty() {
            fallback.to_owned()
        } else {
            stdout
        }
    } else {
        stderr
    }
}

fn replace_file(source: &Path, destination: &Path) -> AdapterResult<()> {
    fs::rename(source, destination).map_err(|source| {
        AdapterError::external_io(
            "commit translated container",
            Some(destination.to_path_buf()),
            source,
        )
    })
}

fn preserve_permissions(source: &Path, destination: &Path) -> AdapterResult<()> {
    let permissions = fs::metadata(source)
        .map_err(|error| {
            AdapterError::external_io(
                "read source container permissions",
                Some(source.to_path_buf()),
                error,
            )
        })?
        .permissions();
    fs::set_permissions(destination, permissions).map_err(|error| {
        AdapterError::external_io(
            "preserve container permissions",
            Some(destination.to_path_buf()),
            error,
        )
    })
}

fn temporary_output_path(destination: &Path) -> PathBuf {
    static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = destination
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("output");
    let extension = destination
        .extension()
        .and_then(OsStr::to_str)
        .unwrap_or("mkv");
    destination.with_file_name(format!(
        ".{name}.subbake-{}-{nonce}.{extension}",
        std::process::id(),
    ))
}

fn unique_temp_dir() -> AdapterResult<TemporaryDirectory> {
    static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let root = std::env::temp_dir();
    for _ in 0..100 {
        let nonce = NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = root.join(format!(
            "subbake-embedded-subtitle-{}-{nonce}",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(TemporaryDirectory(path)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "unable to allocate an embedded subtitle temporary directory",
    )
    .into())
}

struct TemporaryDirectory(PathBuf);

impl TemporaryDirectory {
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct TemporaryOutput {
    path: PathBuf,
    armed: bool,
}

impl TemporaryOutput {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(
        index: usize,
        codec: &str,
        language: Option<&str>,
        default: bool,
        forced: bool,
    ) -> SubtitleStream {
        SubtitleStream {
            index,
            codec: codec.to_owned(),
            language: language.map(str::to_owned),
            title: None,
            default,
            forced,
        }
    }

    #[test]
    fn auto_source_prefers_default_non_forced_text_stream() {
        let streams = vec![
            stream(2, "subrip", Some("spa"), false, false),
            stream(3, "subrip", Some("eng"), true, false),
            stream(4, "hdmv_pgs_subtitle", Some("eng"), false, false),
        ];

        let selected = select_text_stream(&streams, "Auto").expect("select stream");

        assert_eq!(selected.index, 3);
    }

    #[test]
    fn explicit_source_language_matches_iso_639_three_letter_tag() {
        let streams = vec![
            stream(2, "subrip", Some("spa"), true, false),
            stream(3, "ass", Some("eng"), false, false),
        ];

        let selected = select_text_stream(&streams, "en").expect("select English");

        assert_eq!(selected.index, 3);
    }

    #[test]
    fn bitmap_only_container_reports_supported_boundary() {
        let streams = vec![stream(3, "hdmv_pgs_subtitle", Some("eng"), true, false)];

        let error = select_text_stream(&streams, "Auto").expect_err("bitmap must fail");

        assert!(error.to_string().contains("no translatable text"));
        assert!(error.to_string().contains("hdmv_pgs_subtitle"));
    }

    #[test]
    fn default_output_is_in_place_unless_source_container_is_preserved() {
        let in_place = default_embedded_translation_output_path(
            Path::new("movie.mkv"),
            false,
            Some("zh-Hans"),
            false,
        )
        .expect("in-place output path");
        let translated = default_embedded_translation_output_path(
            Path::new("movie.mkv"),
            false,
            Some("zh-Hans"),
            true,
        )
        .expect("output path");
        let bilingual =
            default_embedded_translation_output_path(Path::new("movie.mp4"), true, None, true)
                .expect("output path");

        assert_eq!(in_place, PathBuf::from("movie.mkv"));
        assert_eq!(translated, PathBuf::from("movie.zh-Hans.translated.mkv"));
        assert_eq!(bilingual, PathBuf::from("movie.bilingual.mp4"));
    }

    #[test]
    fn appended_subtitle_metadata_uses_next_subtitle_ordinal() {
        let streams = [];
        let request = EmbedSubtitleRequest {
            input_path: Path::new("input.mkv"),
            streams: &streams,
            subtitle_path: Path::new("translated.srt"),
            output_path: Path::new("output.mkv"),
            target_language: "zh-Hans",
            container_kind: SubtitleContainerKind::Matroska,
        };
        let args = embed_subtitle_args(&request, 2, "zh-Hans (SubBake translation)", &[]);
        let rendered = args
            .iter()
            .map(|arg| arg.to_string_lossy())
            .collect::<Vec<_>>();

        assert!(rendered.iter().any(|arg| arg == "-metadata:s:s:2"));
        assert!(rendered.iter().any(|arg| arg == "-disposition:s:2"));
        assert!(rendered.iter().any(|arg| arg == "language=chi"));
    }

    #[test]
    fn recognizes_common_text_subtitle_containers_and_their_codecs() {
        for (path, codec, muxer) in [
            ("movie.mkv", "srt", "matroska"),
            ("movie.mp4", "mov_text", "mp4"),
            ("movie.m4v", "mov_text", "mp4"),
            ("movie.mov", "mov_text", "mov"),
            ("movie.webm", "webvtt", "webm"),
        ] {
            let kind = SubtitleContainerKind::from_path(Path::new(path)).expect("container");
            assert_eq!(kind.subtitle_codec(), codec);
            assert_eq!(kind.muxer(), muxer);
        }
        assert!(SubtitleContainerKind::from_path(Path::new("movie.avi")).is_none());
    }
}
