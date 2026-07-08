#[test]
fn cli_exposes_redesigned_commands() {
    let names = subbake_cli::command_names();

    assert!(names.contains(&"agent"));
    assert!(names.contains(&"translate"));
    assert!(names.contains(&"batch"));
    assert!(names.contains(&"pipeline"));
    assert!(names.contains(&"provider"));
    assert!(names.contains(&"runtime"));
}

#[test]
fn pipeline_reports_pending_transcription_for_media_inputs() {
    let error = subbake_cli::run(vec!["pipeline".to_owned(), "movie.mp4".to_owned()])
        .expect_err("media pipeline should be pending");

    assert!(error.to_string().contains("transcription is pending"));
}

#[test]
fn transcribe_reports_pending_backend() {
    let error = subbake_cli::run(vec!["transcribe".to_owned(), "movie.mp4".to_owned()])
        .expect_err("transcription backend should be pending");

    assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
}
