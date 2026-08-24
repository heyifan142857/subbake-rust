#![cfg(any(unix, windows))]

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::process::Command;
#[cfg(unix)]
use std::process::Stdio;
#[cfg(unix)]
use std::time::{Duration, Instant};

#[cfg(unix)]
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
  echo "--model --file --output-file --output-srt --output-vtt --threads --print-progress --no-prints --max-context --vad --vad-model" >&2
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
    std::fs::write(root.join("ggml-silero-v6.2.0.bin"), b"fake VAD model")
        .expect("write VAD model");

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

#[cfg(windows)]
#[test]
fn ctrl_break_cancels_cli_on_windows() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let root = std::env::temp_dir().join(format!("subbake-cli-cancel-{nonce}"));
    std::fs::create_dir_all(&root).expect("create root");
    let helper_source = root.join("fake_whisper.rs");
    let whisper = root.join("whisper-cli.exe");
    let ready = root.join("whisper.ready");
    let result = root.join("result.txt");
    std::fs::write(
        &helper_source,
        r#"use std::{env, fs, thread, time::Duration};
fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--version") {
        println!("whisper.cpp fake");
        return;
    }
    if args.iter().any(|arg| arg == "--help") {
        eprintln!("--model --file --output-file --output-srt --output-vtt --threads --print-progress --no-prints --max-context --vad --vad-model");
        return;
    }
    fs::write(env::var_os("SUBBAKE_TEST_READY").expect("ready path"), b"ready")
        .expect("write ready marker");
    thread::sleep(Duration::from_secs(30));
}
"#,
    )
    .expect("write fake whisper source");
    let compiled = Command::new("rustc")
        .args(["--edition", "2024"])
        .arg(&helper_source)
        .arg("-o")
        .arg(&whisper)
        .status()
        .expect("compile fake whisper");
    assert!(compiled.success(), "rustc failed with {compiled}");

    let audio = root.join("audio.wav");
    std::fs::write(&audio, b"fake wav").expect("write audio");
    std::fs::write(root.join("ggml-fake.bin"), b"fake model").expect("write model");
    std::fs::write(root.join("ggml-silero-v6.2.0.bin"), b"fake VAD model")
        .expect("write VAD model");
    let driver = root.join("send_ctrl_break.py");
    std::fs::write(
        &driver,
        r#"import os, pathlib, signal, subprocess, sys, time
sbake, audio, whisper, root, ready, result = sys.argv[1:]
env = os.environ.copy()
env["SUBBAKE_TEST_READY"] = ready
command = [
    sbake, "transcribe", audio,
    "--whisper-bin", whisper,
    "--whisper-models-dir", root,
    "--model", "fake",
    "--output", str(pathlib.Path(root) / "cancelled.srt"),
]
process = subprocess.Popen(
    command,
    env=env,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)
deadline = time.monotonic() + 10
while not pathlib.Path(ready).exists():
    if process.poll() is not None:
        raise RuntimeError(f"sbake exited before whisper started: {process.returncode}")
    if time.monotonic() >= deadline:
        process.kill()
        raise RuntimeError("whisper did not start")
    time.sleep(0.02)
process.send_signal(signal.CTRL_BREAK_EVENT)
try:
    code = process.wait(timeout=10)
except subprocess.TimeoutExpired:
    process.kill()
    process.wait()
    raise
pathlib.Path(result).write_text(str(code), encoding="utf-8")
"#,
    )
    .expect("write Ctrl+Break driver");

    let driver_status = Command::new("python")
        .arg(&driver)
        .args([
            env!("CARGO_BIN_EXE_sbake"),
            audio.to_str().expect("audio path"),
            whisper.to_str().expect("whisper path"),
            root.to_str().expect("root path"),
            ready.to_str().expect("ready path"),
            result.to_str().expect("result path"),
        ])
        .status()
        .expect("run Ctrl+Break driver");
    assert!(
        driver_status.success(),
        "driver failed with {driver_status}"
    );
    let exit_code = std::fs::read_to_string(&result).expect("read exit code");
    assert_eq!(exit_code, "130");
    let _ = std::fs::remove_dir_all(root);
}
