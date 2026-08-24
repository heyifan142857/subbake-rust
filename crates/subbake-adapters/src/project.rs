//! Filesystem adapter for project-level subtitle inspection.

use std::fs;
use std::path::{Path, PathBuf};

use subbake_core::{ProjectDocumentPair, ProjectReport, QualityPolicy, inspect_project};

use crate::error::{AdapterError, AdapterResult};
use crate::fs::{is_supported_subtitle_path, read_document};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectInspectionRequest {
    pub root: PathBuf,
    pub recursive: bool,
}

pub fn inspect_subtitle_project(request: ProjectInspectionRequest) -> AdapterResult<ProjectReport> {
    if !request.root.is_dir() {
        return Err(AdapterError::invalid_input(format!(
            "project root is not a directory: {}",
            request.root.display()
        )));
    }
    let mut paths = Vec::new();
    discover_subtitles(&request.root, request.recursive, &mut paths)?;
    paths.sort();

    let sources = paths
        .iter()
        .filter(|path| generated_kind(path).is_none())
        .cloned()
        .collect::<Vec<_>>();
    let mut pairs = Vec::with_capacity(sources.len());
    for source_path in sources {
        let output_path = matching_output(&source_path, &paths);
        let bilingual = output_path
            .as_deref()
            .and_then(generated_kind)
            .is_some_and(|kind| kind == GeneratedKind::Bilingual);
        let output = output_path.as_deref().map(read_document).transpose()?;
        pairs.push(ProjectDocumentPair {
            source_path: relative_display(&request.root, &source_path),
            source: read_document(&source_path)?,
            output_path: output_path
                .as_deref()
                .map(|path| relative_display(&request.root, path)),
            output,
            bilingual,
        });
    }

    Ok(inspect_project(
        request.root.to_string_lossy(),
        pairs,
        QualityPolicy::default(),
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeneratedKind {
    Translated,
    Bilingual,
}

fn generated_kind(path: &Path) -> Option<GeneratedKind> {
    let stem = path.file_stem()?.to_str()?;
    if stem.ends_with(".translated") {
        Some(GeneratedKind::Translated)
    } else if stem.ends_with(".bilingual") {
        Some(GeneratedKind::Bilingual)
    } else {
        None
    }
}

fn matching_output(source: &Path, paths: &[PathBuf]) -> Option<PathBuf> {
    let source_stem = source.file_stem()?.to_str()?;
    let source_parent = source.parent();
    paths
        .iter()
        .filter(|candidate| candidate.parent() == source_parent)
        .filter_map(|candidate| {
            let kind = generated_kind(candidate)?;
            let stem = candidate.file_stem()?.to_str()?;
            let base = match kind {
                GeneratedKind::Translated => stem.strip_suffix(".translated")?,
                GeneratedKind::Bilingual => stem.strip_suffix(".bilingual")?,
            };
            (base == source_stem).then_some((kind, candidate))
        })
        .min_by_key(|(kind, _)| match kind {
            GeneratedKind::Translated => 0,
            GeneratedKind::Bilingual => 1,
        })
        .map(|(_, path)| path.clone())
}

fn discover_subtitles(root: &Path, recursive: bool, paths: &mut Vec<PathBuf>) -> AdapterResult<()> {
    let entries = fs::read_dir(root).map_err(|source| {
        AdapterError::external_io("read project directory", Some(root.to_path_buf()), source)
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| {
            AdapterError::external_io(
                "read project directory entry",
                Some(root.to_path_buf()),
                source,
            )
        })?;
        let path = entry.path();
        if path.is_dir() && recursive {
            discover_subtitles(&path, true, paths)?;
        } else if path.is_file() && is_supported_subtitle_path(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

fn relative_display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn builds_inventory_and_detects_cross_episode_inconsistency() {
        let root = std::env::temp_dir().join(format!(
            "subbake-project-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("create project");
        fs::write(root.join("e1.srt"), srt("Hello")).expect("source one");
        fs::write(root.join("e1.translated.srt"), srt("你好")).expect("output one");
        fs::write(root.join("e2.srt"), srt("Hello")).expect("source two");
        fs::write(root.join("e2.translated.srt"), srt("您好")).expect("output two");
        fs::write(root.join("e3.srt"), srt("Later")).expect("pending source");

        let report = inspect_subtitle_project(ProjectInspectionRequest {
            root: root.clone(),
            recursive: false,
        })
        .expect("inspect project");

        assert_eq!(report.summary.files, 3);
        assert_eq!(report.summary.pending, 1);
        assert_eq!(report.summary.consistency_issues, 1);
        let _ = fs::remove_dir_all(root);
    }

    fn srt(text: &str) -> String {
        format!("1\n00:00:00,000 --> 00:00:02,000\n{text}\n")
    }
}
