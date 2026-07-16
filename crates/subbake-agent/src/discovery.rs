use std::path::{Path, PathBuf};

pub(crate) fn rank_subtitle_candidates(
    paths: Vec<PathBuf>,
    query: &str,
    project_root: &Path,
) -> Vec<PathBuf> {
    let query = query.to_lowercase();
    let tokens = query
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    let mut candidates = paths
        .into_iter()
        .filter(|path| {
            path.extension()
                .and_then(|suffix| suffix.to_str())
                .is_some_and(|suffix| {
                    matches!(suffix.to_lowercase().as_str(), "srt" | "vtt" | "txt")
                })
        })
        .filter(|path| {
            !path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .is_some_and(|stem| stem.contains(".translated") || stem.contains(".bilingual"))
        })
        .map(|path| {
            let relative = path.strip_prefix(project_root).unwrap_or(&path);
            let name = relative.to_string_lossy().to_lowercase();
            let token_hits = tokens.iter().filter(|token| name.contains(**token)).count();
            let exact = !query.is_empty() && name.contains(&query);
            let score = token_hits * 10 + usize::from(exact) * 25;
            (score, name, path)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    candidates
        .into_iter()
        .take(20)
        .map(|(_, _, path)| path)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_matching_subtitles_before_unrelated_files() {
        let root = Path::new("/project");
        let ranked = rank_subtitle_candidates(
            vec![
                root.join("other.srt"),
                root.join("The Matrix.srt"),
                root.join("The Matrix.translated.srt"),
            ],
            "matrix",
            root,
        );
        assert_eq!(ranked[0], root.join("The Matrix.srt"));
        assert_eq!(ranked.len(), 2);
    }
}
