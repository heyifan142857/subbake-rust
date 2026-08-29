use ratatui::buffer::CellWidth;
use unicode_segmentation::UnicodeSegmentation;

pub(super) fn display_width(value: &str) -> usize {
    usize::from(value.cell_width())
}

pub(super) fn truncate_with_ellipsis(value: &str, width: usize) -> String {
    if display_width(value) <= width {
        return value.to_owned();
    }
    if width == 0 {
        return String::new();
    }
    if width == 1 {
        return "…".to_owned();
    }

    let content_width = width - 1;
    let mut rendered = String::new();
    let mut used = 0;
    for grapheme in value.graphemes(true) {
        let grapheme_width = display_width(grapheme).max(1);
        if used + grapheme_width > content_width {
            break;
        }
        rendered.push_str(grapheme);
        used += grapheme_width;
    }
    rendered.push('…');
    rendered
}

pub(super) fn tail_by_width(value: &str, width: usize) -> String {
    if display_width(value) <= width {
        return value.to_owned();
    }
    let mut suffix = Vec::new();
    let mut used = 0;
    for grapheme in value.graphemes(true).rev() {
        let grapheme_width = display_width(grapheme).max(1);
        if used + grapheme_width > width {
            break;
        }
        suffix.push(grapheme);
        used += grapheme_width;
    }
    suffix.into_iter().rev().collect()
}

pub(super) fn pad_to_width(value: &str, width: usize) -> String {
    let value = truncate_with_ellipsis(value, width);
    let padding = width.saturating_sub(display_width(&value));
    format!("{value}{}", " ".repeat(padding))
}

/// Wrap text into terminal display columns while preserving explicit line
/// breaks. Break at whitespace when possible and fall back to grapheme
/// boundaries for long paths, URLs, and CJK text.
pub(super) fn wrap_text(value: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return Vec::new();
    }
    value
        .split('\n')
        .flat_map(|line| wrap_logical_line(line, width))
        .collect()
}

fn wrap_logical_line(line: &str, width: usize) -> Vec<String> {
    if line.is_empty() {
        return vec![String::new()];
    }
    let graphemes = line.graphemes(true).collect::<Vec<_>>();
    let mut rows = Vec::new();
    let mut start = 0;
    while start < graphemes.len() {
        while start < graphemes.len() && graphemes[start].chars().all(char::is_whitespace) {
            start += 1;
        }
        if start == graphemes.len() {
            break;
        }
        let mut end = start;
        let mut used = 0;
        let mut whitespace_break = None;
        while end < graphemes.len() {
            let grapheme_width = display_width(graphemes[end]).max(1);
            if used > 0 && used + grapheme_width > width {
                break;
            }
            used += grapheme_width;
            end += 1;
            if graphemes[end - 1].chars().all(char::is_whitespace) {
                whitespace_break = Some(end - 1);
            }
            if used >= width {
                break;
            }
        }
        let row_end = if end < graphemes.len() {
            whitespace_break
                .filter(|break_at| *break_at > start)
                .unwrap_or(end)
        } else {
            end
        };
        rows.push(graphemes[start..row_end].concat().trim_end().to_owned());
        start = if row_end < end { row_end + 1 } else { end };
    }
    if rows.is_empty() {
        rows.push(String::new());
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::{display_width, pad_to_width, tail_by_width, truncate_with_ellipsis, wrap_text};

    #[test]
    fn helpers_use_unicode_display_columns() {
        assert_eq!(display_width("a中文"), 5);
        // Ratatui accounts for the way terminals render halfwidth katakana
        // sound marks, unlike unicode-width's generic Unicode calculation.
        assert_eq!(display_width("ｶﾞ"), 2);
        assert_eq!(truncate_with_ellipsis("a中文", 4), "a中…");
        assert_eq!(tail_by_width("ab中文", 4), "中文");
        assert_eq!(display_width(&pad_to_width("中", 4)), 4);
    }

    #[test]
    fn wrapping_prefers_words_and_falls_back_to_graphemes() {
        assert_eq!(wrap_text("alpha beta gamma", 10), ["alpha", "beta gamma"]);
        assert_eq!(wrap_text("字幕翻译完成", 6), ["字幕翻", "译完成"]);
        assert_eq!(wrap_text("a\n\nb", 4), ["a", "", "b"]);
        assert!(wrap_text("🙂🙂", 1).iter().all(|row| !row.is_empty()));
    }
}
