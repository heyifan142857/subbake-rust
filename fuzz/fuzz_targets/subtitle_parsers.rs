#![no_main]

use std::path::Path;

use libfuzzer_sys::fuzz_target;
use subbake_core::formats::{RenderOptions, parse_document_text, render_document};

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    for (format, path) in [
        ("srt", "input.srt"),
        ("vtt", "input.vtt"),
        ("ass", "input.ass"),
        ("txt", "input.txt"),
    ] {
        if let Ok(document) = parse_document_text(Path::new(path), text, Some(format)) {
            let _ = render_document(
                &document,
                &document.segments,
                &RenderOptions::new(false, None),
            );
        }
    }
});
