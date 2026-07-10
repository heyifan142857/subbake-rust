#[test]
fn cli_exposes_redesigned_commands() {
    let names = subbake_cli::command_names();

    assert!(names.contains(&"agent"));
    assert!(names.contains(&"resume"));
    assert!(names.contains(&"translate"));
    assert!(names.contains(&"batch"));
    assert!(names.contains(&"pipeline"));
    assert!(names.contains(&"provider"));
    assert!(names.contains(&"runtime"));
    assert!(names.contains(&"whisper"));
}

#[test]
fn pipeline_media_input_attempts_transcription() {
    let error = subbake_cli::run(vec!["pipeline".to_owned(), "movie.mp4".to_owned()])
        .expect_err("media pipeline should attempt transcription");

    let msg = error.to_string();
    // The old stub said "pending migration"; now it tries real transcription.
    assert!(
        !msg.contains("pending migration"),
        "should no longer be a stub: {msg}"
    );
}

#[test]
fn transcribe_media_attempts_transcription() {
    let error = subbake_cli::run(vec!["transcribe".to_owned(), "movie.mp4".to_owned()])
        .expect_err("transcribe should try real backend");

    let msg = error.to_string();
    assert!(
        !msg.contains("pending migration"),
        "should no longer be a stub: {msg}"
    );
}

#[test]
fn provider_check_uses_mock_backend() {
    subbake_cli::run(vec!["provider".to_owned(), "check".to_owned()])
        .expect("mock provider should check");
}

#[test]
fn agent_rejects_unknown_subcommand() {
    let error = subbake_cli::run(vec!["agent".to_owned(), "bogus".to_owned()])
        .expect_err("unknown agent command should fail");

    assert!(error.to_string().contains("start the agent"));
}

#[test]
fn runtime_clean_requires_confirmation() {
    let error = subbake_cli::run(vec![
        "runtime".to_owned(),
        "clean".to_owned(),
        "clip.srt".to_owned(),
    ])
    .expect_err("runtime clean should require confirmation");

    assert!(error.to_string().contains("--yes"));
}

#[test]
fn whisper_status_is_available_without_installation() {
    subbake_cli::run(vec!["whisper".to_owned(), "status".to_owned()])
        .expect("whisper status should not require installed backend");
}

#[test]
fn whisper_model_list_is_available_without_download() {
    subbake_cli::run(vec![
        "whisper".to_owned(),
        "model".to_owned(),
        "list".to_owned(),
    ])
    .expect("whisper model list should not require download backend");
}

#[test]
fn whisper_model_attempts_download() {
    // "model unknown-name" is rejected immediately by the CLI parser
    // as an unknown model name.
    let error = subbake_cli::run(vec![
        "whisper".to_owned(),
        "model".to_owned(),
        "nonexistentmodel12345".to_owned(),
    ])
    .expect_err("model download should attempt real download");

    let msg = error.to_string();
    assert!(
        !msg.contains("pending"),
        "should no longer be a stub: {msg}"
    );
}
