use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use subbake_core::entities::{SubtitleDocument, SubtitleSegment};
use subbake_core::formats::{
    RenderOptions, normalize_format, parse_document_text, render_document,
    supported_format_from_path,
};

pub fn is_supported_subtitle_path(path: &Path) -> bool {
    supported_format_from_path(path).is_some()
}

/// Resolve the filesystem-dependent identity used to isolate runtime data.
///
/// Existing paths are canonicalized to preserve historical run keys. Missing
/// paths retain their absolute spelling, or are anchored to the current
/// directory when relative.
pub fn stable_runtime_input_path(path: &Path) -> io::Result<PathBuf> {
    match path.canonicalize() {
        Ok(canonical) => Ok(canonical),
        Err(_) if path.is_absolute() => Ok(path.to_path_buf()),
        Err(_) => std::env::current_dir()
            .map(|current_dir| current_dir.join(path))
            .map_err(|error| io::Error::other(format!("resolve current directory: {error}"))),
    }
}

pub fn read_document(path: &Path) -> io::Result<SubtitleDocument> {
    let text = fs::read_to_string(path)?;
    parse_document_text(path, &text, None)
        .map_err(|error| io::Error::other(format!("failed to parse {}: {error}", path.display())))
}

pub fn render_and_write_document(
    document: &SubtitleDocument,
    translations: &[SubtitleSegment],
    output_path: &Path,
    options: &RenderOptions,
) -> io::Result<String> {
    let rendered = render_document(document, translations, options).map_err(io::Error::other)?;
    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(output_path, rendered.as_bytes())?;
    let written = fs::read_to_string(output_path)?;
    if written != rendered {
        return Err(io::Error::other(format!(
            "write verification failed for {}",
            output_path.display()
        )));
    }
    Ok(rendered)
}

pub fn default_output_path(
    input_path: &Path,
    output_format: Option<&str>,
    bilingual: bool,
) -> io::Result<PathBuf> {
    let target_format = match output_format {
        Some(value) => normalize_format(value).map_err(io::Error::other)?,
        None => supported_format_from_path(input_path)
            .ok_or_else(|| {
                io::Error::other(format!("unsupported format: {}", input_path.display()))
            })?
            .to_owned(),
    };
    let suffix = if bilingual { "bilingual" } else { "translated" };
    let stem = input_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    Ok(input_path.with_file_name(format!("{stem}.{suffix}.{target_format}")))
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
}
