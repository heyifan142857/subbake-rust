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

pub fn read_document(path: &Path) -> io::Result<SubtitleDocument> {
    let text = fs::read_to_string(path)?;
    parse_document_text(path, &text, None).map_err(io::Error::other)
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
