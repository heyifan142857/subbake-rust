#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn termination_signal_cancels_cli_and_terminates_whisper_process_group() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let root = std::env::temp_dir().join(format!("subbake-cli-cancel-{nonce}"));
    std::fs::create_dir_all(&root).expect("create root");
    let whisper = root.join("whisper-cli");
    let child_pid_path = root.join("whisper.pid");
    let script = format!(
        r#"#!/bin/sh
if [ "$1" = "--version" ]; then echo "whisper.cpp fake"; exit 0; fi
if [ "$1" = "--help" ]; then
  echo "--model --file --output-file --output-srt --output-vtt --threads --print-progress --no-prints --max-context" >&2
  exit 0
fi
echo $$ > "{}"
trap 'exit 143' TERM
sleep 30
"#,
        child_pid_path.display()
    );
    std::fs::write(&whisper, script).expect("write fake whisper");
    std::fs::set_permissions(&whisper, std::fs::Permissions::from_mode(0o755))
        .expect("chmod fake whisper");
    let audio = root.join("audio.wav");
    std::fs::write(&audio, b"fake wav").expect("write audio");
    std::fs::write(root.join("ggml-fake.bin"), b"fake model").expect("write model");

    // libtest itself ignores SIGINT, so exercise the same cancellation bridge
    // through SIGTERM in this isolated process. Normal CLI launches register
    // both SIGINT (Ctrl+C) and SIGTERM.
    let mut command = Command::new("env");
    let mut sbake = command
        .args(["--default-signal=INT,TERM", env!("CARGO_BIN_EXE_sbake")])
        .args([
            "transcribe",
            audio.to_str().expect("audio path"),
            "--whisper-bin",
            whisper.to_str().expect("whisper path"),
            "--whisper-models-dir",
            root.to_str().expect("models path"),
            "--model",
            "fake",
            "--output",
            root.join("result.srt").to_str().expect("output path"),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start sbake");

    let ready_deadline = Instant::now() + Duration::from_secs(5);
    while !child_pid_path.exists() {
        if let Some(status) = sbake.try_wait().expect("poll sbake") {
            panic!("sbake exited before whisper started: {status}");
        }
        assert!(
            Instant::now() < ready_deadline,
            "whisper child did not start"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let whisper_pid = std::fs::read_to_string(&child_pid_path)
        .expect("read whisper pid")
        .trim()
        .to_owned();
    let signal = Command::new("kill")
        .args(["-TERM", &sbake.id().to_string()])
        .status()
        .expect("send SIGTERM to sbake");
    assert!(signal.success(), "failed to signal sbake: {signal}");

    let exit_deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = sbake.try_wait().expect("poll cancelled sbake") {
            break status;
        }
        if Instant::now() >= exit_deadline {
            let _ = sbake.kill();
            let _ = sbake.wait();
            panic!("cancelled sbake did not finish");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(status.code(), Some(130));

    let whisper_alive = Command::new("kill")
        .args(["-0", &whisper_pid])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("probe whisper child")
        .success();
    assert!(
        !whisper_alive,
        "whisper child {whisper_pid} survived cancellation"
    );
    let _ = std::fs::remove_dir_all(root);
}
