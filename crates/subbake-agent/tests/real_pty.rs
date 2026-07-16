#![cfg(unix)]

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use portable_pty::{CommandBuilder, PtySize, native_pty_system};
use subbake_agent::{
    AgentError, AgentResult, CancellationGuard, CancellationToken, ProfileChoice, StartupInfo,
    SubBakeTui, TuiAction, TuiInteraction,
};

const CHILD_ENV: &str = "SUBBAKE_REAL_PTY_CHILD";
const ACTION_LOG_ENV: &str = "SUBBAKE_REAL_PTY_ACTION_LOG";
const TEST_BIN_ENV: &str = "SUBBAKE_REAL_PTY_TEST_BIN";
const TEST_TIMEOUT: Duration = Duration::from_secs(15);
const STEP_TIMEOUT: Duration = Duration::from_secs(4);
const KEYBOARD_QUERY: &[u8] = b"\x1b[?u\x1b[c";
const KEYBOARD_RESPONSE: &[u8] = b"\x1b[?1u\x1b[?1;2c";
const DSR_QUERY: &[u8] = b"\x1b[6n";
const DSR_RESPONSE: &[u8] = b"\x1b[1;1R";
const ENTER_KEY: &[u8] = b"\x1b[13u";
const ESCAPE_KEY: &[u8] = b"\x1b[27u";
const SHIFT_TAB_KEY: &[u8] = b"\x1b[9;2u";

type SharedWriter = Arc<Mutex<Box<dyn Write + Send>>>;

#[derive(Default)]
struct Transcript {
    bytes: Mutex<Vec<u8>>,
    changed: Condvar,
}

#[test]
fn real_pty_restores_terminal_and_exercises_interactions() {
    if std::env::var_os(CHILD_ENV).is_some() {
        return;
    }

    let test_binary = std::env::current_exe().expect("locate PTY test binary");
    let test_dir = unique_test_dir();
    std::fs::create_dir_all(&test_dir).expect("create PTY test directory");
    let action_log = test_dir.join("actions.log");

    let pair = native_pty_system()
        .openpty(PtySize {
            rows: 30,
            cols: 100,
            pixel_width: 0,
            pixel_height: 0,
        })
        .expect("open PTY");
    let mut command = CommandBuilder::new("/bin/sh");
    command.arg("-c");
    command.arg(
        r#"before=$(stty -g) || exit 90
printf 'PTY_STTY_BEFORE:%s\n' "$before"
"$SUBBAKE_REAL_PTY_TEST_BIN" --exact pty_child_driver --nocapture --test-threads=1
status=$?
after=$(stty -g) || exit 91
printf 'PTY_STTY_AFTER:%s\n' "$after"
printf 'PTY_SHELL_STATUS:%s\n' "$status"
exit "$status"
"#,
    );
    command.env(CHILD_ENV, "1");
    command.env(ACTION_LOG_ENV, &action_log);
    command.env(TEST_BIN_ENV, &test_binary);

    let mut child = pair
        .slave
        .spawn_command(command)
        .expect("spawn PTY test child");
    drop(pair.slave);

    let transcript = Arc::new(Transcript::default());
    let reader = pair.master.try_clone_reader().expect("clone PTY reader");
    let writer = Arc::new(Mutex::new(
        pair.master.take_writer().expect("take PTY writer"),
    ));
    let reader_thread = spawn_terminal_emulator(reader, writer.clone(), transcript.clone());

    wait_for_output(&transcript, b"\x1b[>1u", STEP_TIMEOUT);
    send(&writer, SHIFT_TAB_KEY);
    wait_for_action(&action_log, "TogglePlan", &transcript);
    wait_for_output(&transcript, b"plan toggled", STEP_TIMEOUT);

    send_text(&writer, "/profile");
    send(&writer, ENTER_KEY);
    wait_for_action(&action_log, "SubmitText:/profile", &transcript);
    wait_for_output(&transcript, b"\x1b[?1049h", STEP_TIMEOUT);

    send(&writer, b"\x1b[B");
    send(&writer, ENTER_KEY);
    send_text(&writer, "pty_profile");
    send(&writer, ENTER_KEY);
    wait_for_action(&action_log, "CreateProfile:pty_profile", &transcript);
    wait_for_output(&transcript, b"\x1b[?1049l", STEP_TIMEOUT);
    wait_for_output(&transcript, b"profile created", STEP_TIMEOUT);

    send_text(&writer, "make a plan");
    send(&writer, ENTER_KEY);
    wait_for_action(&action_log, "SubmitText:make a plan", &transcript);
    wait_for_output(&transcript, b"pending PTY plan", STEP_TIMEOUT);
    send(&writer, b"\x1b[B");
    send(&writer, ENTER_KEY);
    wait_for_action(&action_log, "RejectPlan", &transcript);
    wait_for_output(&transcript, b"plan rejected", STEP_TIMEOUT);

    send_text(&writer, "cancel me");
    send(&writer, ENTER_KEY);
    wait_for_action(&action_log, "SubmitText:cancel me", &transcript);
    send(&writer, ESCAPE_KEY);
    wait_for_action(&action_log, "CancellationObserved", &transcript);
    wait_for_output(&transcript, b"Cancelled.", STEP_TIMEOUT);

    send_text(&writer, "after cancel");
    send(&writer, ENTER_KEY);
    wait_for_action(&action_log, "SubmitText:after cancel", &transcript);
    wait_for_output(&transcript, b"worker recovered", STEP_TIMEOUT);

    send_text(&writer, "/exit");
    send(&writer, ENTER_KEY);
    let status = wait_for_child(&mut child, &transcript);
    assert!(
        status.success(),
        "PTY child failed with {status}; transcript: {}",
        escaped_transcript(&transcript)
    );

    drop(writer);
    drop(pair.master);
    reader_thread.join().expect("join PTY reader");

    let output = transcript_bytes(&transcript);
    let before = marker_value(&output, "PTY_STTY_BEFORE:").expect("stty before marker");
    let after = marker_value(&output, "PTY_STTY_AFTER:").expect("stty after marker");
    assert_eq!(before, after, "terminal attributes were not restored");
    assert_eq!(
        marker_value(&output, "PTY_SHELL_STATUS:").as_deref(),
        Some("0")
    );

    assert_eq!(
        count_subslice(&output, b"\x1b[?1049h"),
        count_subslice(&output, b"\x1b[?1049l"),
        "alternate-screen enter/leave must be paired"
    );
    assert_eq!(
        count_subslice(&output, b"\x1b[>1u"),
        count_subslice(&output, b"\x1b[<1u"),
        "keyboard enhancement push/pop must be paired"
    );
    assert!(
        count_subslice(&output, DSR_QUERY) > 0,
        "the PTY session must exercise a real DSR query"
    );

    let _ = std::fs::remove_dir_all(test_dir);
}

#[test]
fn pty_child_driver() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }

    let action_log = PathBuf::from(
        std::env::var_os(ACTION_LOG_ENV).expect("PTY child action log environment variable"),
    );
    let cancellation = CancellationToken::default();
    let mut tui = SubBakeTui::new().expect("initialize TUI inside PTY");
    tui.set_startup_info(StartupInfo {
        provider: "mock".to_owned(),
        model: "pty-model".to_owned(),
        config: "PTY test".to_owned(),
        cache_enabled: false,
        cwd: "/pty-test".to_owned(),
    });
    tui.set_cancellation_token(cancellation);

    tui.run(move |action, guard, _observer| {
        append_action(&action_log, &action_label(&action))?;
        scripted_interaction(action, &guard, &action_log)
    })
    .expect("run TUI PTY scenario");
}

fn scripted_interaction(
    action: TuiAction,
    guard: &CancellationGuard,
    action_log: &Path,
) -> AgentResult<TuiInteraction> {
    match action {
        TuiAction::TogglePlan => Ok(TuiInteraction::Message {
            message: "plan toggled".to_owned(),
        }),
        TuiAction::SubmitText(input) if input == "/profile" => Ok(TuiInteraction::ProfilePicker {
            message: String::new(),
            options: vec![
                ProfileChoice {
                    name: "active".to_owned(),
                    provider: "mock".to_owned(),
                    model: "pty-model".to_owned(),
                    active: true,
                    create: false,
                },
                ProfileChoice {
                    name: "new profile…".to_owned(),
                    provider: String::new(),
                    model: "copy active settings".to_owned(),
                    active: false,
                    create: true,
                },
            ],
        }),
        TuiAction::CreateProfile(name) => Ok(TuiInteraction::ModelChanged {
            model: "pty-created-model".to_owned(),
            message: format!("profile created: {name}"),
        }),
        TuiAction::SubmitText(input) if input == "make a plan" => {
            Ok(TuiInteraction::PlanApproval {
                message: "pending PTY plan".to_owned(),
            })
        }
        TuiAction::RejectPlan => Ok(TuiInteraction::Message {
            message: "plan rejected".to_owned(),
        }),
        TuiAction::SubmitText(input) if input == "cancel me" => {
            while !guard.is_cancelled() {
                thread::sleep(Duration::from_millis(10));
            }
            append_action(action_log, "CancellationObserved")?;
            Err(AgentError::Cancelled)
        }
        TuiAction::SubmitText(input) if input == "after cancel" => Ok(TuiInteraction::Message {
            message: "worker recovered".to_owned(),
        }),
        unexpected => Err(AgentError::invalid_input(format!(
            "unexpected PTY test action: {unexpected:?}"
        ))),
    }
}

fn action_label(action: &TuiAction) -> String {
    match action {
        TuiAction::SubmitText(input) => format!("SubmitText:{input}"),
        TuiAction::ApprovePlan => "ApprovePlan".to_owned(),
        TuiAction::RejectPlan => "RejectPlan".to_owned(),
        TuiAction::SelectProfile(name) => format!("SelectProfile:{name}"),
        TuiAction::CreateProfile(name) => format!("CreateProfile:{name}"),
        TuiAction::SelectSession(id) => format!("SelectSession:{id}"),
        TuiAction::TogglePlan => "TogglePlan".to_owned(),
    }
}

fn append_action(path: &Path, action: &str) -> std::io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{action}")?;
    file.flush()
}

fn spawn_terminal_emulator(
    mut reader: Box<dyn Read + Send>,
    writer: SharedWriter,
    transcript: Arc<Transcript>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut buffer = [0_u8; 4096];
        let mut keyboard_queries = 0;
        let mut dsr_queries = 0;
        loop {
            let read = match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => read,
                Err(error) if error.raw_os_error() == Some(5) => break,
                Err(error) => panic!("read PTY output: {error}"),
            };
            let (new_keyboard_queries, new_dsr_queries) = {
                let mut bytes = transcript.bytes.lock().expect("lock PTY transcript");
                bytes.extend_from_slice(&buffer[..read]);
                let keyboard_count = count_subslice(&bytes, KEYBOARD_QUERY);
                let dsr_count = count_subslice(&bytes, DSR_QUERY);
                transcript.changed.notify_all();
                (
                    keyboard_count.saturating_sub(keyboard_queries),
                    dsr_count.saturating_sub(dsr_queries),
                )
            };

            for _ in 0..new_keyboard_queries {
                send(&writer, KEYBOARD_RESPONSE);
            }
            keyboard_queries += new_keyboard_queries;
            for _ in 0..new_dsr_queries {
                send(&writer, DSR_RESPONSE);
            }
            dsr_queries += new_dsr_queries;
        }
        transcript.changed.notify_all();
    })
}

fn send_text(writer: &SharedWriter, text: &str) {
    send(writer, text.as_bytes());
}

fn send(writer: &SharedWriter, bytes: &[u8]) {
    let mut writer = writer.lock().expect("lock PTY writer");
    writer.write_all(bytes).expect("write PTY input");
    writer.flush().expect("flush PTY input");
}

fn wait_for_output(transcript: &Transcript, needle: &[u8], timeout: Duration) {
    let deadline = Instant::now() + timeout;
    let mut bytes = transcript.bytes.lock().expect("lock PTY transcript");
    while !contains_subslice(&bytes, needle) {
        let now = Instant::now();
        if now >= deadline {
            panic!(
                "timed out waiting for {:?}; transcript: {}",
                String::from_utf8_lossy(needle),
                escape_bytes(&bytes)
            );
        }
        let remaining = deadline.saturating_duration_since(now);
        let (next, result) = transcript
            .changed
            .wait_timeout(bytes, remaining)
            .expect("wait for PTY output");
        bytes = next;
        if result.timed_out() && !contains_subslice(&bytes, needle) {
            panic!(
                "timed out waiting for {:?}; transcript: {}",
                String::from_utf8_lossy(needle),
                escape_bytes(&bytes)
            );
        }
    }
}

fn wait_for_action(path: &Path, expected: &str, transcript: &Transcript) {
    let deadline = Instant::now() + STEP_TIMEOUT;
    loop {
        let actions = std::fs::read_to_string(path).unwrap_or_default();
        if actions.lines().any(|line| line == expected) {
            return;
        }
        if Instant::now() >= deadline {
            panic!(
                "timed out waiting for action {expected:?}; actions: {actions:?}; transcript: {}",
                escaped_transcript(transcript)
            );
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn wait_for_child(
    child: &mut Box<dyn portable_pty::Child + Send + Sync>,
    transcript: &Transcript,
) -> portable_pty::ExitStatus {
    let deadline = Instant::now() + TEST_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait().expect("poll PTY child") {
            return status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            panic!(
                "PTY child did not exit before timeout; transcript: {}",
                escaped_transcript(transcript)
            );
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn unique_test_dir() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("subbake-real-pty-{}-{nonce}", std::process::id()))
}

fn transcript_bytes(transcript: &Transcript) -> Vec<u8> {
    transcript
        .bytes
        .lock()
        .expect("lock PTY transcript")
        .clone()
}

fn escaped_transcript(transcript: &Transcript) -> String {
    escape_bytes(&transcript_bytes(transcript))
}

fn escape_bytes(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).escape_debug().to_string()
}

fn marker_value(bytes: &[u8], marker: &str) -> Option<String> {
    String::from_utf8_lossy(bytes).lines().find_map(|line| {
        line.strip_prefix(marker)
            .map(|value| value.trim_end_matches('\r').to_owned())
    })
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

fn count_subslice(haystack: &[u8], needle: &[u8]) -> usize {
    haystack
        .windows(needle.len())
        .filter(|window| *window == needle)
        .count()
}
