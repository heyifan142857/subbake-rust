use std::path::Path;

use subbake_core::formats::{RenderOptions, parse_document_text, render_document};

#[test]
fn srt_round_trips_basic_timing_and_text() {
    let text = "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n";
    let doc = parse_document_text(Path::new("sample.srt"), text, None).expect("parse");
    let rendered = render_document(&doc, &doc.segments, &RenderOptions::new(false, None))
        .expect("render");

    assert_eq!(rendered, "1\n00:00:00,000 --> 00:00:01,000\nhello\n");
}

#[test]
fn txt_preserves_line_count() {
    let doc = parse_document_text(Path::new("sample.txt"), "one\ntwo\n", None).expect("parse");
    assert_eq!(doc.segments.len(), 2);
    assert_eq!(doc.segments[1].id, "2");
}
