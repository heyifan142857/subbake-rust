#[test]
fn cli_exposes_redesigned_commands() {
    let names = subbake_cli::command_names();

    assert!(names.contains(&"agent"));
    assert!(names.contains(&"translate"));
    assert!(names.contains(&"batch"));
    assert!(names.contains(&"provider"));
    assert!(names.contains(&"runtime"));
}
