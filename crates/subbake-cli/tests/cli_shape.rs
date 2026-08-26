#[test]
fn cli_exposes_redesigned_commands() {
    let names = subbake_cli::command_names();

    #[cfg(feature = "agent")]
    {
        assert!(names.contains(&"agent"));
        assert!(names.contains(&"resume"));
    }
    #[cfg(not(feature = "agent"))]
    {
        assert!(!names.contains(&"agent"));
        assert!(!names.contains(&"resume"));
    }
    assert!(names.contains(&"translate"));
    assert!(names.contains(&"edit"));
    assert!(names.contains(&"batch"));
    assert!(names.contains(&"pipeline"));
    assert!(names.contains(&"provider"));
    assert!(names.contains(&"runtime"));
    assert!(names.contains(&"whisper"));
    assert!(names.contains(&"qa"));
    assert!(names.contains(&"project"));
    assert!(names.contains(&"memory"));
    assert!(names.contains(&"evaluate"));
    assert!(names.contains(&"overnight"));
    assert!(names.contains(&"help"));
}

#[test]
fn help_is_available_without_required_operands() {
    for args in [
        vec!["translate", "--help"],
        vec!["edit", "--help"],
        vec!["transcribe", "--help"],
        vec!["runtime", "clean", "--help"],
        vec!["provider", "check", "--help"],
        vec!["qa", "--help"],
        vec!["project", "--help"],
        vec!["memory", "--help"],
    ] {
        subbake_cli::run(args.into_iter().map(str::to_owned).collect())
            .expect("help should not execute or require operands");
    }
}

#[test]
fn project_command_writes_a_versioned_manifest() {
    let root = temp_root("project-report");
    std::fs::create_dir_all(&root).expect("create root");
    std::fs::write(
        root.join("episode.srt"),
        "1\n00:00:00,000 --> 00:00:02,000\nHello\n",
    )
    .expect("write source");
    let report_path = root.join("report.json");

    subbake_cli::run(vec![
        "project".to_owned(),
        root.to_string_lossy().into_owned(),
        "--output".to_owned(),
        report_path.to_string_lossy().into_owned(),
    ])
    .expect("inspect project");

    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report_path).expect("read project report"))
            .expect("parse project report");
    assert_eq!(report["version"], 1);
    assert_eq!(report["summary"]["pending"], 1);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn edit_dry_run_validates_and_preserves_the_target() {
    let root = temp_root("edit-dry-run");
    std::fs::create_dir_all(&root).expect("create root");
    let target = root.join("clip.translated.txt");
    std::fs::write(&target, "hello\n").expect("write target");
    let config = root.join("config.toml");
    std::fs::write(&config, "version = 3\n").expect("write config");

    subbake_cli::run(vec![
        "edit".to_owned(),
        target.to_string_lossy().into_owned(),
        "--instruction".to_owned(),
        "make it uppercase".to_owned(),
        "--dry-run".to_owned(),
        "--config".to_owned(),
        config.to_string_lossy().into_owned(),
    ])
    .expect("edit preview");

    assert_eq!(
        std::fs::read_to_string(&target).expect("read target"),
        "hello\n"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn pipeline_media_input_attempts_transcription() {
    let config = empty_config("pipeline");
    let error = subbake_cli::run(vec![
        "pipeline".to_owned(),
        "movie.mp4".to_owned(),
        "--config".to_owned(),
        config.to_string_lossy().into_owned(),
    ])
    .expect_err("media pipeline should attempt transcription");
    let _ = std::fs::remove_file(config);

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
fn translate_preserves_existing_output_without_overwrite() {
    let root = temp_root("translate-overwrite");
    std::fs::create_dir_all(&root).expect("create root");
    let input = root.join("clip.txt");
    let output = root.join("translated.txt");
    let config = root.join("config.toml");
    std::fs::write(&input, "hello\n").expect("write input");
    std::fs::write(&output, "keep me\n").expect("write existing output");
    std::fs::write(&config, "version = 3\n").expect("write config");

    let error = subbake_cli::run(vec![
        "translate".to_owned(),
        input.to_string_lossy().into_owned(),
        "--output".to_owned(),
        output.to_string_lossy().into_owned(),
        "--config".to_owned(),
        config.to_string_lossy().into_owned(),
    ])
    .expect_err("existing output must require --overwrite");

    assert!(error.to_string().contains("overwrite is false"));
    assert_eq!(
        std::fs::read_to_string(&output).expect("read preserved output"),
        "keep me\n"
    );

    subbake_cli::run(vec![
        "translate".to_owned(),
        input.to_string_lossy().into_owned(),
        "--output".to_owned(),
        output.to_string_lossy().into_owned(),
        "--config".to_owned(),
        config.to_string_lossy().into_owned(),
        "--overwrite".to_owned(),
    ])
    .expect("explicit overwrite should succeed");
    assert_eq!(
        std::fs::read_to_string(&output).expect("read translated output"),
        "[MOCK-ZH-HANS] hello\n"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn transcribe_preserves_existing_output_without_overwrite() {
    let root = temp_root("transcribe-overwrite");
    std::fs::create_dir_all(&root).expect("create root");
    let media = root.join("movie.mp4");
    let sidecar = root.join("source.srt");
    let output = root.join("movie.srt");
    std::fs::write(&sidecar, "1\n00:00:00,000 --> 00:00:01,000\nHello\n").expect("write sidecar");
    std::fs::write(&output, "keep me\n").expect("write existing output");

    let error = subbake_cli::run(vec![
        "transcribe".to_owned(),
        media.to_string_lossy().into_owned(),
        "--sidecar".to_owned(),
        sidecar.to_string_lossy().into_owned(),
        "--output".to_owned(),
        output.to_string_lossy().into_owned(),
    ])
    .expect_err("existing transcription must require --overwrite");

    assert!(error.to_string().contains("overwrite is false"));
    assert_eq!(
        std::fs::read_to_string(&output).expect("read preserved output"),
        "keep me\n"
    );

    subbake_cli::run(vec![
        "transcribe".to_owned(),
        media.to_string_lossy().into_owned(),
        "--sidecar".to_owned(),
        sidecar.to_string_lossy().into_owned(),
        "--output".to_owned(),
        output.to_string_lossy().into_owned(),
        "--overwrite".to_owned(),
    ])
    .expect("explicit transcription overwrite should succeed");
    assert!(
        std::fs::read_to_string(&output)
            .expect("read transcription")
            .contains("Hello")
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn provider_check_uses_mock_backend() {
    let config = empty_config("provider");
    subbake_cli::run(vec![
        "provider".to_owned(),
        "check".to_owned(),
        "--config".to_owned(),
        config.to_string_lossy().into_owned(),
    ])
    .expect("mock provider should check");
    let _ = std::fs::remove_file(config);
}

#[cfg(feature = "agent")]
#[test]
fn agent_rejects_unknown_subcommand() {
    let error = subbake_cli::run(vec!["agent".to_owned(), "bogus".to_owned()])
        .expect_err("unknown agent command should fail");

    assert!(error.to_string().contains("start the agent"));
}

#[cfg(not(feature = "agent"))]
#[test]
fn cli_only_build_shows_help_without_a_command() {
    subbake_cli::run(Vec::new()).expect("CLI-only build should show help");
}

#[cfg(not(feature = "agent"))]
#[test]
fn cli_only_build_rejects_agent_commands() {
    for command in ["agent", "resume"] {
        let error = subbake_cli::run(vec![command.to_owned()])
            .expect_err("Agent commands should not exist in a CLI-only build");
        assert!(error.to_string().contains("unknown command"));
    }
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
fn empty_config(label: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "subbake-cli-shape-{}-{label}.toml",
        std::process::id()
    ));
    std::fs::write(&path, "version = 3\n").expect("write empty config");
    path
}

fn temp_root(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("subbake-cli-shape-{}-{label}", std::process::id()))
}
