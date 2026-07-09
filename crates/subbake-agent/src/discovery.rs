use std::path::{Path, PathBuf};

pub(crate) fn summarize_observation(tool_name: &str, text: &str) -> String {
    let lines = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return match tool_name {
            "candidate_subtitles" => "no subtitle candidates".to_owned(),
            "recent_translations" => "no recent translations".to_owned(),
            _ => "no matches".to_owned(),
        };
    }
    match tool_name {
        "read_file" | "read_file_preview" => {
            format!(
                "preview ({} chars): {}",
                text.chars().count(),
                truncate(lines[0], 160)
            )
        }
        "list_files" | "search_files" | "candidate_subtitles" => {
            let top = lines
                .iter()
                .take(3)
                .map(|line| line.trim())
                .collect::<Vec<_>>()
                .join(", ");
            format!("{} item(s), top: {top}", lines.len())
        }
        "recent_translations" => {
            format!("{} recent: {}", lines.len(), truncate(lines[0], 160))
        }
        _ => truncate(text, 300),
    }
}

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

fn truncate(text: &str, limit: usize) -> String {
    let truncated = text.chars().take(limit).collect::<String>();
    if text.chars().count() > limit {
        format!("{truncated}...")
    } else {
        truncated
    }
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
