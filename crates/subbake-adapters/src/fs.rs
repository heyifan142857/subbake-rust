use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use subbake_core::entities::{SubtitleDocument, SubtitleSegment};
use subbake_core::formats::{
    RenderOptions, normalize_format, parse_document_text, render_document,
    supported_format_from_path,
};

use crate::error::{AdapterError, AdapterResult};

pub fn is_supported_subtitle_path(path: &Path) -> bool {
    supported_format_from_path(path).is_some()
}

/// Resolve the filesystem-dependent identity used to isolate runtime data.
///
/// Existing paths are canonicalized to preserve historical run keys. Missing
/// paths retain their absolute spelling, or are anchored to the current
/// directory when relative.
pub fn stable_runtime_input_path(path: &Path) -> AdapterResult<PathBuf> {
    match path.canonicalize() {
        Ok(canonical) => Ok(canonical),
        Err(_) if path.is_absolute() => Ok(path.to_path_buf()),
        Err(_) => std::env::current_dir()
            .map(|current_dir| current_dir.join(path))
            .map_err(|source| AdapterError::external_io("resolve current directory", None, source)),
    }
}

pub fn read_document(path: &Path) -> AdapterResult<SubtitleDocument> {
    let text = fs::read_to_string(path).map_err(|source| {
        AdapterError::external_io("read subtitle", Some(path.to_path_buf()), source)
    })?;
    parse_document_text(path, &text, None).map_err(|source| AdapterError::CoreContext {
        operation: "parse subtitle",
        path: Some(path.to_path_buf()),
        source,
    })
}

pub fn render_and_write_document(
    document: &SubtitleDocument,
    translations: &[SubtitleSegment],
    output_path: &Path,
    options: &RenderOptions,
) -> AdapterResult<String> {
    let rendered = render_document(document, translations, options).map_err(AdapterError::from)?;
    write_verified_atomically(output_path, rendered.as_bytes())?;
    Ok(rendered)
}

pub(crate) fn render_and_write_document_atomic(
    document: &SubtitleDocument,
    translations: &[SubtitleSegment],
    output_path: &Path,
    options: &RenderOptions,
) -> AdapterResult<String> {
    render_and_write_document(document, translations, output_path, options)
}

fn write_verified_atomically(output_path: &Path, bytes: &[u8]) -> AdapterResult<()> {
    let parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|source| {
        AdapterError::external_io(
            "create subtitle output directory",
            Some(parent.to_path_buf()),
            source,
        )
    })?;
    let file_name = output_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("subtitle");
    static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{file_name}.subbake-tmp-{}-{nonce}",
        std::process::id()
    ));
    let staged = (|| -> AdapterResult<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|source| {
                AdapterError::external_io("create staged subtitle", Some(temporary.clone()), source)
            })?;
        file.write_all(bytes).map_err(|source| {
            AdapterError::external_io("write staged subtitle", Some(temporary.clone()), source)
        })?;
        file.sync_all().map_err(|source| {
            AdapterError::external_io("sync staged subtitle", Some(temporary.clone()), source)
        })?;
        if let Ok(metadata) = fs::metadata(output_path) {
            fs::set_permissions(&temporary, metadata.permissions()).map_err(|source| {
                AdapterError::external_io(
                    "preserve subtitle permissions",
                    Some(temporary.clone()),
                    source,
                )
            })?;
        }
        Ok(())
    })();
    if let Err(error) = staged {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }

    publish_staged_file(&temporary, output_path)?;
    let written = fs::read(output_path).map_err(|source| {
        AdapterError::external_io(
            "verify rendered subtitle",
            Some(output_path.to_path_buf()),
            source,
        )
    })?;
    if written != bytes {
        return Err(AdapterError::Core(subbake_core::CoreError::DataInvariant(
            format!("write verification failed for {}", output_path.display()),
        )));
    }
    Ok(())
}

#[cfg(not(windows))]
fn publish_staged_file(temporary: &Path, output_path: &Path) -> AdapterResult<()> {
    fs::rename(temporary, output_path).map_err(|source| {
        let _ = fs::remove_file(temporary);
        AdapterError::external_io(
            "publish rendered subtitle",
            Some(output_path.to_path_buf()),
            source,
        )
    })
}

#[cfg(windows)]
fn publish_staged_file(temporary: &Path, output_path: &Path) -> AdapterResult<()> {
    if !output_path.exists() {
        return fs::rename(temporary, output_path).map_err(|source| {
            let _ = fs::remove_file(temporary);
            AdapterError::external_io(
                "publish rendered subtitle",
                Some(output_path.to_path_buf()),
                source,
            )
        });
    }
    let prior = output_path.with_file_name(format!(
        ".{}.subbake-prior-{}",
        output_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("subtitle"),
        std::process::id()
    ));
    fs::rename(output_path, &prior).map_err(|source| {
        let _ = fs::remove_file(temporary);
        AdapterError::external_io(
            "stage previous subtitle",
            Some(output_path.to_path_buf()),
            source,
        )
    })?;
    if let Err(source) = fs::rename(temporary, output_path) {
        let _ = fs::rename(&prior, output_path);
        let _ = fs::remove_file(temporary);
        return Err(AdapterError::external_io(
            "publish rendered subtitle",
            Some(output_path.to_path_buf()),
            source,
        ));
    }
    let _ = fs::remove_file(prior);
    Ok(())
}

pub fn default_output_path(
    input_path: &Path,
    output_format: Option<&str>,
    bilingual: bool,
) -> AdapterResult<PathBuf> {
    default_output_path_with_language(input_path, output_format, bilingual, None)
}

pub fn default_output_path_with_language(
    input_path: &Path,
    output_format: Option<&str>,
    bilingual: bool,
    language_tag: Option<&str>,
) -> AdapterResult<PathBuf> {
    let target_format = match output_format {
        Some(value) => normalize_format(value).map_err(AdapterError::from)?,
        None => supported_format_from_path(input_path)
            .ok_or_else(|| {
                AdapterError::invalid_input(format!("unsupported format: {}", input_path.display()))
            })?
            .to_owned(),
    };
    let suffix = if bilingual { "bilingual" } else { "translated" };
    let stem = input_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    let language = language_tag
        .filter(|value| !value.trim().is_empty())
        .map(|value| format!(".{}", value.trim()))
        .unwrap_or_default();
    Ok(input_path.with_file_name(format!("{stem}{language}.{suffix}.{target_format}")))
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn stable_runtime_path_canonicalizes_an_existing_path() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("subbake-stable-path-{nonce}"));
        fs::create_dir_all(&root).expect("create root");
        let input = root.join("clip.srt");
        fs::write(&input, b"1\n").expect("write input");

        let stable = stable_runtime_input_path(&input).expect("stable path");
        let expected = input.canonicalize().expect("canonical path");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(stable, expected);
    }

    #[test]
    fn stable_runtime_path_anchors_a_missing_relative_path() {
        let relative = Path::new("missing/subtitle.srt");
        let stable = stable_runtime_input_path(relative).expect("stable path");

        assert_eq!(
            stable,
            std::env::current_dir()
                .expect("current directory")
                .join(relative)
        );
    }

    #[test]
    fn stable_runtime_path_preserves_a_missing_absolute_path() {
        let absolute = std::env::temp_dir().join("subbake-path-that-does-not-exist.srt");
        let stable = stable_runtime_input_path(&absolute).expect("stable path");

        assert_eq!(stable, absolute);
    }

    #[test]
    fn explicit_language_tag_is_included_in_default_output_name() {
        let path = default_output_path_with_language(
            Path::new("/work/sample.srt"),
            Some("srt"),
            false,
            Some("ja"),
        )
        .expect("output path");
        assert_eq!(path, PathBuf::from("/work/sample.ja.translated.srt"));

        let bilingual = default_output_path_with_language(
            Path::new("/work/sample.srt"),
            Some("vtt"),
            true,
            Some("zh-Hans"),
        )
        .expect("bilingual output path");
        assert_eq!(
            bilingual,
            PathBuf::from("/work/sample.zh-Hans.bilingual.vtt")
        );
    }

    #[test]
    fn ass_is_a_supported_subtitle_with_ass_default_output() {
        assert!(is_supported_subtitle_path(Path::new("movie.ass")));
        assert_eq!(
            default_output_path(Path::new("movie.ass"), None, false).expect("ASS output path"),
            PathBuf::from("movie.translated.ass")
        );
    }

    #[cfg(unix)]
    #[test]
    fn atomic_writer_replaces_contents_and_preserves_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("subbake-atomic-output-{nonce}"));
        fs::create_dir_all(&root).expect("create root");
        let output = root.join("translated.srt");
        fs::write(&output, b"old").expect("write old output");
        fs::set_permissions(&output, fs::Permissions::from_mode(0o640))
            .expect("set output permissions");

        write_verified_atomically(&output, b"new subtitle").expect("publish replacement");

        assert_eq!(
            fs::read(&output).expect("read replacement"),
            b"new subtitle"
        );
        assert_eq!(
            fs::metadata(&output)
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777,
            0o640
        );
        assert_eq!(
            fs::read_dir(&root)
                .expect("list root")
                .filter_map(Result::ok)
                .count(),
            1
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn failed_atomic_publish_removes_its_staged_file() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("subbake-atomic-failure-{nonce}"));
        let output = root.join("directory-target.srt");
        fs::create_dir_all(&output).expect("create conflicting directory");

        write_verified_atomically(&output, b"subtitle")
            .expect_err("a directory cannot be replaced as a subtitle file");

        let staged = fs::read_dir(&root)
            .expect("list root")
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().contains("subbake-tmp"))
            .count();
        assert_eq!(staged, 0);
        assert!(output.is_dir());
        let _ = fs::remove_dir_all(root);
    }
}
