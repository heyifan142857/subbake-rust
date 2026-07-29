use std::path::Path;

use subbake_core::formats::{RenderOptions, parse_document_text, render_document};

#[test]
fn srt_round_trips_basic_timing_and_text() {
    let text = "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n";
    let doc = parse_document_text(Path::new("sample.srt"), text, None).expect("parse");
    let rendered =
        render_document(&doc, &doc.segments, &RenderOptions::new(false, None)).expect("render");

    assert_eq!(rendered, "1\n00:00:00,000 --> 00:00:01,000\nhello\n");
}

#[test]
fn txt_preserves_line_count() {
    let doc = parse_document_text(Path::new("sample.txt"), "one\ntwo\n", None).expect("parse");
    assert_eq!(doc.segments.len(), 2);
    assert_eq!(doc.segments[1].id, "2");
}

#[test]
fn ass_preserves_script_styles_and_dialogue_fields() {
    let text = "[Script Info]\nPlayResY: 1040\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize\nStyle: Default,sans-serif,71\n\n[Events]\nFormat: Layer, Start, End, Style, Text\nDialogue: 0,0:00:00.00,0:00:01.00,Default,{\\i1}hello{\\i0}\n";
    let document =
        parse_document_text(Path::new("sample.ass"), text, None).expect("parse ASS document");
    let rendered = render_document(
        &document,
        &document.segments,
        &RenderOptions::new(false, None),
    )
    .expect("render ASS document");

    assert_eq!(rendered, text);
}
