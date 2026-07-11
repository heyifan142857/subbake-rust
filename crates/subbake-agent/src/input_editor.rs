use std::ops::Range;

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

#[derive(Debug, Clone, Default)]
pub(crate) struct InputEditor {
    text: String,
    cursor: usize,
    preferred_column: Option<usize>,
    scroll: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VisualLine {
    pub(crate) range: Range<usize>,
}

impl InputEditor {
    pub(crate) fn text(&self) -> &str {
        &self.text
    }
    #[cfg(test)]
    pub(crate) fn cursor(&self) -> usize {
        self.cursor
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
    pub(crate) fn clear(&mut self) {
        self.set_text(String::new());
    }

    pub(crate) fn take(&mut self) -> String {
        self.cursor = 0;
        self.preferred_column = None;
        self.scroll = 0;
        std::mem::take(&mut self.text)
    }

    pub(crate) fn set_text(&mut self, text: String) {
        self.text = text;
        self.cursor = self.text.len();
        self.preferred_column = None;
        self.scroll = 0;
    }

    pub(crate) fn insert_char(&mut self, character: char) {
        self.text.insert(self.cursor, character);
        self.cursor += character.len_utf8();
        self.preferred_column = None;
    }

    pub(crate) fn insert_newline(&mut self) {
        self.insert_char('\n');
    }

    pub(crate) fn backspace(&mut self) {
        if let Some(previous) = self.previous_boundary() {
            self.text.drain(previous..self.cursor);
            self.cursor = previous;
            self.preferred_column = None;
        }
    }

    pub(crate) fn move_left(&mut self) {
        if let Some(previous) = self.previous_boundary() {
            self.cursor = previous;
        }
        self.preferred_column = None;
    }

    pub(crate) fn move_right(&mut self) {
        if let Some(grapheme) = self.text[self.cursor..].graphemes(true).next() {
            self.cursor += grapheme.len();
        }
        self.preferred_column = None;
    }

    pub(crate) fn move_up(&mut self, width: u16) -> bool {
        self.move_vertical(width, -1)
    }
    pub(crate) fn move_down(&mut self, width: u16) -> bool {
        self.move_vertical(width, 1)
    }

    fn move_vertical(&mut self, width: u16, direction: isize) -> bool {
        let lines = self.visual_lines(width);
        let current = line_index(&lines, self.cursor);
        let target = current as isize + direction;
        if target < 0 || target >= lines.len() as isize {
            return false;
        }
        let column = self.preferred_column.unwrap_or_else(|| {
            UnicodeWidthStr::width(&self.text[lines[current].range.start..self.cursor])
        });
        self.cursor = cursor_at_column(&self.text, &lines[target as usize].range, column);
        self.preferred_column = Some(column);
        true
    }

    pub(crate) fn visual_lines(&self, width: u16) -> Vec<VisualLine> {
        let width = usize::from(width.max(1));
        let mut lines = Vec::new();
        let mut start = 0;
        let mut columns = 0;
        for (index, grapheme) in self.text.grapheme_indices(true) {
            if grapheme == "\n" {
                lines.push(VisualLine {
                    range: start..index,
                });
                start = index + 1;
                columns = 0;
                continue;
            }
            let grapheme_width = UnicodeWidthStr::width(grapheme).max(1);
            if columns > 0 && columns + grapheme_width > width {
                lines.push(VisualLine {
                    range: start..index,
                });
                start = index;
                columns = 0;
            }
            columns += grapheme_width;
        }
        lines.push(VisualLine {
            range: start..self.text.len(),
        });
        lines
    }

    pub(crate) fn desired_height(&self, width: u16) -> u16 {
        u16::try_from(self.visual_lines(width).len()).unwrap_or(u16::MAX)
    }

    pub(crate) fn visible_lines(&mut self, width: u16, height: u16) -> Vec<String> {
        let lines = self.visual_lines(width);
        let cursor_line = line_index(&lines, self.cursor);
        let height = usize::from(height.max(1));
        if cursor_line < self.scroll {
            self.scroll = cursor_line;
        }
        if cursor_line >= self.scroll + height {
            self.scroll = cursor_line + 1 - height;
        }
        self.scroll = self.scroll.min(lines.len().saturating_sub(height));
        lines
            .iter()
            .skip(self.scroll)
            .take(height)
            .map(|line| self.text[line.range.clone()].to_owned())
            .collect()
    }

    pub(crate) fn cursor_position(&self, width: u16) -> (u16, u16) {
        let lines = self.visual_lines(width);
        let index = line_index(&lines, self.cursor);
        let column = UnicodeWidthStr::width(&self.text[lines[index].range.start..self.cursor]);
        (
            u16::try_from(column).unwrap_or(u16::MAX),
            u16::try_from(index.saturating_sub(self.scroll)).unwrap_or(u16::MAX),
        )
    }

    fn previous_boundary(&self) -> Option<usize> {
        self.text[..self.cursor]
            .grapheme_indices(true)
            .next_back()
            .map(|(index, _)| index)
    }
}

fn line_index(lines: &[VisualLine], cursor: usize) -> usize {
    lines
        .partition_point(|line| line.range.start <= cursor)
        .saturating_sub(1)
}

fn cursor_at_column(text: &str, range: &Range<usize>, target: usize) -> usize {
    let mut columns = 0;
    for (offset, grapheme) in text[range.clone()].grapheme_indices(true) {
        let next = columns + UnicodeWidthStr::width(grapheme).max(1);
        if next > target {
            return range.start + offset;
        }
        columns = next;
    }
    range.end
}

#[cfg(test)]
mod tests {
    use super::InputEditor;

    #[test]
    fn wraps_unicode_and_moves_between_visual_lines() {
        let mut editor = InputEditor::default();
        editor.set_text("ab中文🙂z".to_owned());
        assert_eq!(
            editor
                .visual_lines(4)
                .iter()
                .map(|l| &editor.text()[l.range.clone()])
                .collect::<Vec<_>>(),
            vec!["ab中", "文🙂", "z"]
        );
        assert!(editor.move_up(4));
        assert_eq!(editor.cursor(), "ab中".len());
    }

    #[test]
    fn explicit_newlines_and_scrolling_keep_cursor_visible() {
        let mut editor = InputEditor::default();
        editor.set_text("one\ntwo\nthree".to_owned());
        assert_eq!(editor.desired_height(20), 3);
        assert_eq!(editor.visible_lines(20, 2), vec!["two", "three"]);
        assert_eq!(editor.cursor_position(20), (5, 1));
    }

    #[test]
    fn backspace_removes_a_complete_grapheme() {
        let mut editor = InputEditor::default();
        editor.set_text("e\u{301}".to_owned());
        editor.backspace();
        assert!(editor.is_empty());
    }
}
