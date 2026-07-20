use crate::entities::{SubtitleSegment, TranslationLine};

const MARKER_PREFIX: &str = "⟦SBK_FMT_";
const MARKER_SUFFIX: &str = "⟧";

#[derive(Debug, Clone)]
struct FormatToken<'a> {
    start: usize,
    end: usize,
    raw: &'a str,
    visible_before: usize,
}

pub(crate) fn protect_formatting(text: &str) -> String {
    let tokens = format_tokens(text);
    if tokens.is_empty() {
        return text.to_owned();
    }
    let mut output = String::with_capacity(text.len() + tokens.len() * 12);
    let mut cursor = 0;
    for (index, token) in tokens.iter().enumerate() {
        output.push_str(&text[cursor..token.start]);
        output.push_str(&format_marker(index));
        cursor = token.end;
    }
    output.push_str(&text[cursor..]);
    output
}

pub(crate) fn restore_line_formatting(source: &str, translation: &str) -> String {
    let source_tokens = format_tokens(source);
    if source_tokens.is_empty() {
        return translation.to_owned();
    }
    let expected = source_tokens
        .iter()
        .map(|token| token.raw)
        .collect::<Vec<_>>();
    if formatting_tokens(translation) == expected {
        return translation.to_owned();
    }

    if let Some(restored) = restore_markers(translation, &source_tokens)
        && formatting_tokens(&restored) == expected
    {
        return restored;
    }

    restore_proportionally(source, translation, &source_tokens)
}

pub(crate) fn restore_batch_formatting(source: &[SubtitleSegment], lines: &mut [TranslationLine]) {
    for segment in source {
        if let Some(line) = lines.iter_mut().find(|line| line.id == segment.id) {
            line.translation = restore_line_formatting(&segment.text, &line.translation);
        }
    }
}

pub(crate) fn formatting_tokens(text: &str) -> Vec<&str> {
    format_tokens(text)
        .into_iter()
        .map(|token| token.raw)
        .collect()
}

fn restore_markers(translation: &str, source_tokens: &[FormatToken<'_>]) -> Option<String> {
    let mut previous = 0;
    for index in 0..source_tokens.len() {
        let marker = format_marker(index);
        let relative = translation[previous..].find(&marker)?;
        let position = previous + relative;
        if translation[position + marker.len()..].contains(&marker) {
            return None;
        }
        previous = position + marker.len();
    }

    let mut restored = translation.to_owned();
    for (index, token) in source_tokens.iter().enumerate() {
        restored = restored.replace(&format_marker(index), token.raw);
    }
    Some(restored)
}

fn restore_proportionally(
    source: &str,
    translation: &str,
    source_tokens: &[FormatToken<'_>],
) -> String {
    let clean = strip_formatting_and_markers(translation, source_tokens.len());
    let target = clean.chars().collect::<Vec<_>>();
    let source_visible = source_tokens.last().map_or(0, |last| {
        last.visible_before + source[last.end..].chars().count()
    });
    let mut insertions = vec![Vec::<&str>::new(); target.len() + 1];
    for token in source_tokens {
        let position = token
            .visible_before
            .saturating_mul(target.len())
            .saturating_add(source_visible / 2)
            .checked_div(source_visible)
            .unwrap_or_default()
            .min(target.len());
        insertions[position].push(token.raw);
    }
    let mut output = String::with_capacity(clean.len() + source_tokens.len() * 7);
    for (index, ch) in target.into_iter().enumerate() {
        for token in &insertions[index] {
            output.push_str(token);
        }
        output.push(ch);
    }
    for token in &insertions[clean.chars().count()] {
        output.push_str(token);
    }
    output
}

fn strip_formatting_and_markers(text: &str, marker_count: usize) -> String {
    let tokens = format_tokens(text);
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    for token in tokens {
        output.push_str(&text[cursor..token.start]);
        cursor = token.end;
    }
    output.push_str(&text[cursor..]);
    for index in 0..marker_count {
        output = output.replace(&format_marker(index), "");
    }
    output
}

fn format_tokens(text: &str) -> Vec<FormatToken<'_>> {
    let mut tokens = Vec::new();
    let mut cursor = 0;
    let mut visible_before = 0;
    while cursor < text.len() {
        let Some(ch) = text[cursor..].chars().next() else {
            break;
        };
        let close = match ch {
            '<' => Some('>'),
            '{' => Some('}'),
            _ => None,
        };
        if let Some(close) = close
            && let Some(relative_end) = text[cursor + ch.len_utf8()..].find(close)
        {
            let end = cursor + ch.len_utf8() + relative_end + close.len_utf8();
            tokens.push(FormatToken {
                start: cursor,
                end,
                raw: &text[cursor..end],
                visible_before,
            });
            cursor = end;
            continue;
        }
        visible_before += 1;
        cursor += ch.len_utf8();
    }
    tokens
}

fn format_marker(index: usize) -> String {
    format!("{MARKER_PREFIX}{index}{MARKER_SUFFIX}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protects_and_restores_full_line_italics() {
        let source = "<i>Eye movement detected.</i>";
        assert_eq!(
            protect_formatting(source),
            "⟦SBK_FMT_0⟧Eye movement detected.⟦SBK_FMT_1⟧"
        );
        assert_eq!(
            restore_line_formatting(source, "⟦SBK_FMT_0⟧检测到眼球运动。⟦SBK_FMT_1⟧"),
            "<i>检测到眼球运动。</i>"
        );
    }

    #[test]
    fn missing_markers_are_restored_at_relative_positions() {
        let restored = restore_line_formatting("Keep <i>this</i> safe", "请妥善保管它");
        assert_eq!(formatting_tokens(&restored), vec!["<i>", "</i>"]);
    }

    #[test]
    fn preserves_nested_and_ass_formatting_tokens() {
        let source = "{\\an8}<b><i>Hello</i></b>";
        let restored = restore_line_formatting(source, "你好");
        assert_eq!(formatting_tokens(&restored), formatting_tokens(source));
        assert_eq!(restored, "{\\an8}<b><i>你好</i></b>");
    }
}
