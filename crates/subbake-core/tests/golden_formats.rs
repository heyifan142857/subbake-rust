use std::path::Path;

use subbake_core::formats::{RenderOptions, parse_document_text, render_document};

const DUCK_AND_COVER_EXCERPT: &str = include_str!("fixtures/duck_and_cover_excerpt.srt");

#[test]
fn srt_round_trips_basic_timing_and_text() {
    let text = "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n";
    let doc = parse_document_text(Path::new("sample.srt"), text, None).expect("parse");
    let rendered =
        render_document(&doc, &doc.segments, &RenderOptions::new(false, None)).expect("render");

    assert_eq!(rendered, "1\n00:00:00,000 --> 00:00:01,000\nhello\n");
}

#[test]
fn public_domain_srt_excerpt_round_trips_multiline_context_and_markup() {
    let path = Path::new("duck_and_cover_excerpt.srt");
    let document =
        parse_document_text(path, DUCK_AND_COVER_EXCERPT, None).expect("parse public-domain SRT");

    assert_eq!(document.segments.len(), 8);
    assert_eq!(
        document.segments[0].text,
        "<i>Announcer:</i> Be sure and remember\nwhat Bert the turtle just did, friends."
    );
    assert!(
        document
            .segments
            .iter()
            .filter(|segment| segment.text.contains('\n'))
            .count()
            >= 5
    );
    assert!(document.segments.iter().any(|segment| {
        segment
            .text
            .contains("Federal Civil Defense Administration")
    }));

    let rendered = render_document(
        &document,
        &document.segments,
        &RenderOptions::new(false, None),
    )
    .expect("render public-domain SRT");

    assert_eq!(rendered, DUCK_AND_COVER_EXCERPT);
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
