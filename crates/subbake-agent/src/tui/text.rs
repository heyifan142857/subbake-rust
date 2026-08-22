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

#[cfg(test)]
mod tests {
    use super::{display_width, pad_to_width, tail_by_width, truncate_with_ellipsis};

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
}
