use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use subbake_core::entities::{SubtitleDocument, SubtitleSegment};
use subbake_core::formats::{
    RenderOptions, normalize_format, parse_document_text, render_document,
    supported_format_from_path,
};

use crate::error::{AdapterError, AdapterResult};
use crate::platform::PlatformPaths;

static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

pub fn is_supported_subtitle_path(path: &Path) -> bool {
    supported_format_from_path(path).is_some()
}

/// Resolve the filesystem-dependent identity used to isolate runtime data.
///
/// Existing paths are canonicalized to preserve historical run keys. Missing
/// tails are appended to the resolved identity of their deepest existing
/// ancestor, so aliases cannot create distinct runtime keys.
pub fn stable_runtime_input_path(path: &Path) -> AdapterResult<PathBuf> {
    PlatformPaths::identify_with_missing_tail(path)
        .map(|identity| identity.resolved().to_path_buf())
}

pub fn read_document(path: &Path) -> AdapterResult<SubtitleDocument> {
    let text = fs::read_to_string(path).map_err(|source| {
        AdapterError::external_io("read subtitle", Some(path.to_path_buf()), source)
    })?;
    parse_document_text(path, &text, None).map_err(|source| AdapterError::CoreContext {
        operation: "parse subtitle",
        path: Some(path.to_path_buf()),
        source: Box::new(source),
    })
}

pub fn render_and_write_document(
    document: &SubtitleDocument,
    translations: &[SubtitleSegment],
    output_path: &Path,
    options: &RenderOptions,
) -> AdapterResult<String> {
    let rendered = render_document(document, translations, options).map_err(AdapterError::from)?;
    write_file_atomically(output_path, rendered.as_bytes())?;
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

pub fn write_file_atomically(output_path: &Path, bytes: &[u8]) -> AdapterResult<()> {
    write_file_atomically_with_permissions(output_path, bytes, None)
}

pub(crate) fn write_file_atomically_with_permissions(
    output_path: &Path,
    bytes: &[u8],
    new_file_permissions: Option<fs::Permissions>,
) -> AdapterResult<()> {
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
    let write_lock = AtomicWriteLock::acquire(output_path)?;
    reject_directory_target(output_path)?;
    let file_name = output_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("subtitle");
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
                AdapterError::external_io("create staged file", Some(temporary.clone()), source)
            })?;
        file.write_all(bytes).map_err(|source| {
            AdapterError::external_io("write staged file", Some(temporary.clone()), source)
        })?;
        file.sync_all().map_err(|source| {
            AdapterError::external_io("sync staged file", Some(temporary.clone()), source)
        })?;
        if let Ok(metadata) = fs::metadata(output_path) {
            fs::set_permissions(&temporary, metadata.permissions()).map_err(|source| {
                AdapterError::external_io(
                    "preserve file permissions",
                    Some(temporary.clone()),
                    source,
                )
            })?;
        } else if let Some(permissions) = new_file_permissions {
            fs::set_permissions(&temporary, permissions).map_err(|source| {
                AdapterError::external_io(
                    "set staged file permissions",
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
            "verify atomic write",
            Some(output_path.to_path_buf()),
            source,
        )
    })?;
    if written != bytes {
        return Err(AdapterError::Core(subbake_core::CoreError::DataInvariant(
            format!("write verification failed for {}", output_path.display()),
        )));
    }
    drop(write_lock);
    sync_parent_directory(parent)?;
    Ok(())
}

struct AtomicWriteLock {
    path: PathBuf,
}

impl AtomicWriteLock {
    fn acquire(target: &Path) -> AdapterResult<Self> {
        let file_name = target
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("file");
        let path = target.with_file_name(format!(".{file_name}.subbake-write-lock"));
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut file) => {
                    file.write_all(std::process::id().to_string().as_bytes())
                        .and_then(|()| file.sync_all())
                        .map_err(|source| {
                            let _ = fs::remove_file(&path);
                            AdapterError::external_io(
                                "initialize atomic write lock",
                                Some(path.clone()),
                                source,
                            )
                        })?;
                    return Ok(Self { path });
                }
                Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {
                    if lock_is_stale(&path) {
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    if std::time::Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(10));
                        continue;
                    }
                    return Err(AdapterError::external_io(
                        "acquire atomic write lock",
                        Some(path),
                        std::io::Error::new(
                            std::io::ErrorKind::WouldBlock,
                            "another process is writing the same file",
                        ),
                    ));
                }
                Err(source) => {
                    return Err(AdapterError::external_io(
                        "create atomic write lock",
                        Some(path),
                        source,
                    ));
                }
            }
        }
    }
}

impl Drop for AtomicWriteLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn lock_is_stale(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .and_then(|metadata| metadata.modified())
        .and_then(|modified| modified.elapsed().map_err(std::io::Error::other))
        .is_ok_and(|elapsed| elapsed > Duration::from_secs(300))
}

fn reject_directory_target(path: &Path) -> AdapterResult<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => Err(AdapterError::external_io(
            "inspect atomic write target",
            Some(path.to_path_buf()),
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "atomic file output cannot replace a directory",
            ),
        )),
        Ok(_) => Ok(()),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(AdapterError::external_io(
            "inspect atomic write target",
            Some(path.to_path_buf()),
            source,
        )),
    }
}

#[cfg(not(windows))]
fn publish_staged_file(temporary: &Path, output_path: &Path) -> AdapterResult<()> {
    fs::rename(temporary, output_path).map_err(|source| {
        let _ = fs::remove_file(temporary);
        AdapterError::external_io(
            "publish staged file",
            Some(output_path.to_path_buf()),
            source,
        )
    })
}

#[cfg(windows)]
fn publish_staged_file(temporary: &Path, output_path: &Path) -> AdapterResult<()> {
    if let Err(error) = reject_directory_target(output_path) {
        let _ = fs::remove_file(temporary);
        return Err(error);
    }
    if !output_path.exists() {
        return fs::rename(temporary, output_path).map_err(|source| {
            let _ = fs::remove_file(temporary);
            AdapterError::external_io(
                "publish staged file",
                Some(output_path.to_path_buf()),
                source,
            )
        });
    }
    let prior_nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let prior = output_path.with_file_name(format!(
        ".{}.subbake-prior-{}-{prior_nonce}",
        output_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("subtitle"),
        std::process::id()
    ));
    fs::rename(output_path, &prior).map_err(|source| {
        let _ = fs::remove_file(temporary);
        AdapterError::external_io(
            "stage previous file",
            Some(output_path.to_path_buf()),
            source,
        )
    })?;
    if let Err(source) = fs::rename(temporary, output_path) {
        let _ = fs::rename(&prior, output_path);
        let _ = fs::remove_file(temporary);
        return Err(AdapterError::external_io(
            "publish staged file",
            Some(output_path.to_path_buf()),
            source,
        ));
    }
    let _ = fs::remove_file(prior);
    Ok(())
}

#[cfg(unix)]
fn sync_parent_directory(parent: &Path) -> AdapterResult<()> {
    fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| {
            AdapterError::external_io(
                "sync atomic write directory",
                Some(parent.to_path_buf()),
                source,
            )
        })
}

#[cfg(not(unix))]
fn sync_parent_directory(_parent: &Path) -> AdapterResult<()> {
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
    use std::sync::{Arc, Barrier};

    use super::*;

    #[test]
    fn stable_runtime_path_canonicalizes_an_existing_path() {
        let temporary = tempfile::Builder::new()
            .prefix("subbake-stable-path-")
            .tempdir()
            .expect("create temporary directory");
        let root = temporary.path();
        let input = root.join("clip.srt");
        fs::write(&input, b"1\n").expect("write input");

        let stable = stable_runtime_input_path(&input).expect("stable path");
        let expected = input.canonicalize().expect("canonical path");
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
    fn stable_runtime_path_resolves_the_parent_of_a_missing_absolute_path() {
        let temporary = tempfile::tempdir().expect("create temporary directory");
        let absolute = temporary.path().join("missing.srt");
        let stable = stable_runtime_input_path(&absolute).expect("stable path");

        assert_eq!(
            stable,
            temporary
                .path()
                .canonicalize()
                .expect("resolve temporary directory")
                .join("missing.srt")
        );
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

    #[test]
    fn atomic_writer_replaces_existing_file_without_leaving_staging_files() {
        let temporary = tempfile::Builder::new()
            .prefix("subbake-atomic-replace-")
            .tempdir()
            .expect("create temporary directory");
        let root = temporary.path();
        let output = root.join("translated.srt");
        fs::write(&output, b"old subtitle").expect("write old output");

        write_file_atomically(&output, b"new subtitle").expect("publish replacement");

        assert_eq!(fs::read(&output).expect("read output"), b"new subtitle");
        assert_eq!(
            fs::read_dir(root)
                .expect("list root")
                .filter_map(Result::ok)
                .count(),
            1,
            "atomic replacement left a lock, backup, or staging file"
        );
    }

    #[cfg(unix)]
    #[test]
    fn atomic_writer_replaces_contents_and_preserves_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let temporary = tempfile::Builder::new()
            .prefix("subbake-atomic-output-")
            .tempdir()
            .expect("create temporary directory");
        let root = temporary.path();
        let output = root.join("translated.srt");
        fs::write(&output, b"old").expect("write old output");
        fs::set_permissions(&output, fs::Permissions::from_mode(0o640))
            .expect("set output permissions");

        write_file_atomically(&output, b"new subtitle").expect("publish replacement");

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
            fs::read_dir(root)
                .expect("list root")
                .filter_map(Result::ok)
                .count(),
            1
        );
    }

    #[test]
    fn atomic_writer_serializes_concurrent_replacements() {
        let temporary = tempfile::Builder::new()
            .prefix("subbake-atomic-concurrent-")
            .tempdir()
            .expect("create temporary directory");
        let root = temporary.path();
        let output = Arc::new(root.join("state.json"));
        fs::write(output.as_ref(), b"initial").expect("write initial file");
        let barrier = Arc::new(Barrier::new(8));
        let writers = (0..8)
            .map(|index| {
                let output = Arc::clone(&output);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let bytes = format!("writer-{index}").into_bytes();
                    barrier.wait();
                    write_file_atomically(&output, &bytes)
                })
            })
            .collect::<Vec<_>>();

        for writer in writers {
            writer.join().expect("writer thread").expect("atomic write");
        }

        let content = fs::read_to_string(output.as_ref()).expect("read final file");
        assert!(content.starts_with("writer-"));
        assert_eq!(
            fs::read_dir(root)
                .expect("list root")
                .filter_map(Result::ok)
                .count(),
            1
        );
    }

    #[test]
    fn atomic_writer_waits_for_a_slow_concurrent_writer() {
        let temporary = tempfile::Builder::new()
            .prefix("subbake-atomic-slow-writer-")
            .tempdir()
            .expect("create temporary directory");
        let root = temporary.path();
        let output = root.join("state.json");
        fs::write(&output, b"initial").expect("write initial file");
        let lock = AtomicWriteLock::acquire(&output).expect("acquire competing lock");
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(1_200));
            drop(lock);
        });

        write_file_atomically(&output, b"replacement").expect("wait for competing writer");
        releaser.join().expect("release competing lock");

        assert_eq!(fs::read(&output).expect("read replacement"), b"replacement");
    }

    #[test]
    fn failed_atomic_publish_removes_its_staged_file() {
        let temporary = tempfile::Builder::new()
            .prefix("subbake-atomic-failure-")
            .tempdir()
            .expect("create temporary directory");
        let root = temporary.path();
        let output = root.join("directory-target.srt");
        fs::create_dir_all(&output).expect("create conflicting directory");

        write_file_atomically(&output, b"subtitle")
            .expect_err("a directory cannot be replaced as a subtitle file");

        let staged = fs::read_dir(root)
            .expect("list root")
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().contains("subbake-tmp"))
            .count();
        assert_eq!(staged, 0);
        assert!(output.is_dir());
    }
}
