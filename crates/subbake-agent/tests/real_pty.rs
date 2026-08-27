#![cfg(any(unix, windows))]

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use portable_pty::{CommandBuilder, PtySize, native_pty_system};
use subbake_agent::{
    AgentError, AgentResult, ApprovalKind, ApprovalPrompt, CancellationGuard, CancellationToken,
    ConfigEditorSnapshot, ConfigFieldId, ConfigFieldView, EngineObserver, ProfileChoice,
    StartupInfo, SubBakeTui, TuiAction, TuiInteraction, TuiObserver, TurnSteering,
};

const CHILD_ENV: &str = "SUBBAKE_REAL_PTY_CHILD";
const ACTION_LOG_ENV: &str = "SUBBAKE_REAL_PTY_ACTION_LOG";
#[cfg(unix)]
const TEST_BIN_ENV: &str = "SUBBAKE_REAL_PTY_TEST_BIN";
// Hosted runners can take several seconds to drain a full-screen Ratatui
// redraw. These are failure ceilings; successful local runs remain fast.
const TEST_TIMEOUT: Duration = Duration::from_secs(45);
const STEP_TIMEOUT: Duration = Duration::from_secs(10);
const KEYBOARD_QUERY: &[u8] = b"\x1b[?u\x1b[c";
const KEYBOARD_RESPONSE: &[u8] = b"\x1b[?1u\x1b[?1;2c";
const DSR_QUERY: &[u8] = b"\x1b[6n";
const DSR_RESPONSE: &[u8] = b"\x1b[1;1R";
const CURSOR_SHOW: &[u8] = b"\x1b[?25h";
#[cfg(unix)]
const ALT_SCREEN_LEAVE: &[u8] = b"\x1b[?1049l";
const ENHANCED_ENTER_KEY: &[u8] = b"\x1b[13u";
const ENHANCED_ESCAPE_KEY: &[u8] = b"\x1b[27u";
const ENHANCED_SHIFT_TAB_KEY: &[u8] = b"\x1b[9;2u";
const LEGACY_ENTER_KEY: &[u8] = b"\r";
const LEGACY_ESCAPE_KEY: &[u8] = b"\x1b";
const LEGACY_SHIFT_TAB_KEY: &[u8] = b"\x1b[Z";

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
    let temporary = unique_test_dir();
    let test_dir = temporary.path();
    let action_log = test_dir.join("actions.log");

    let pair = native_pty_system()
        .openpty(PtySize {
            rows: 30,
            cols: 100,
            pixel_width: 0,
            pixel_height: 0,
        })
        .expect("open PTY");
    #[cfg(unix)]
    let mut command = {
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
        command
    };
    #[cfg(windows)]
    let mut command = {
        let mut command = CommandBuilder::new(&test_binary);
        command.args([
            "--exact",
            "pty_child_driver",
            "--nocapture",
            "--test-threads=1",
        ]);
        command
    };
    command.env(CHILD_ENV, "1");
    command.env(ACTION_LOG_ENV, &action_log);
    #[cfg(unix)]
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

    // The injected cwd is stable across prompt tips and rendering modes, so
    // observing it proves that the first complete main view was rendered.
    wait_for_output(&transcript, b"/pty-test", STEP_TIMEOUT);
    // Keyboard enhancement is optional. Kitty-compatible terminals use CSI-u
    // key encodings after negotiation, while ConPTY and other terminals may
    // retain the traditional encodings. Exercise whichever mode the child
    // actually selected instead of making one terminal protocol mandatory.
    let enhanced_keyboard = transcript_contains(&transcript, b"\x1b[>1u");
    let enter_key = if enhanced_keyboard {
        ENHANCED_ENTER_KEY
    } else {
        LEGACY_ENTER_KEY
    };
    let escape_key = if enhanced_keyboard {
        ENHANCED_ESCAPE_KEY
    } else {
        LEGACY_ESCAPE_KEY
    };
    let shift_tab_key = if enhanced_keyboard {
        ENHANCED_SHIFT_TAB_KEY
    } else {
        LEGACY_SHIFT_TAB_KEY
    };
    let narrow_resize_request = transcript_len(&transcript);
    pair.master
        .resize(PtySize {
            rows: 30,
            cols: 40,
            pixel_width: 0,
            pixel_height: 0,
        })
        .expect("resize PTY to narrow composer");
    let narrow_reflow_checkpoint =
        wait_for_output_after(&transcript, narrow_resize_request, b"\x1b[3J", STEP_TIMEOUT);
    let narrow_history_checkpoint = wait_for_output_after(
        &transcript,
        narrow_reflow_checkpoint,
        b"/pty-test",
        STEP_TIMEOUT,
    );
    wait_for_output_after(
        &transcript,
        narrow_history_checkpoint + b"/pty-test".len(),
        CURSOR_SHOW,
        STEP_TIMEOUT,
    );
    send_text(&writer, "after ");
    thread::sleep(Duration::from_millis(200));
    let wide_resize_request = transcript_len(&transcript);
    pair.master
        .resize(PtySize {
            rows: 30,
            cols: 120,
            pixel_width: 0,
            pixel_height: 0,
        })
        .expect("resize PTY to wide composer");
    let wide_reflow_checkpoint =
        wait_for_output_after(&transcript, wide_resize_request, b"\x1b[3J", STEP_TIMEOUT);
    let wide_history_checkpoint = wait_for_output_after(
        &transcript,
        wide_reflow_checkpoint,
        b"/pty-test",
        STEP_TIMEOUT,
    );
    wait_for_output_after(
        &transcript,
        wide_history_checkpoint + b"/pty-test".len(),
        CURSOR_SHOW,
        STEP_TIMEOUT,
    );
    let response_checkpoint = transcript_len(&transcript);
    send_text(&writer, "resize");
    send(&writer, enter_key);
    wait_for_action(&action_log, "SubmitText:after resize", &transcript);
    wait_for_output_after(
        &transcript,
        response_checkpoint,
        b"resize accepted",
        STEP_TIMEOUT,
    );
    let height_resize_request = transcript_len(&transcript);
    pair.master
        .resize(PtySize {
            rows: 20,
            cols: 120,
            pixel_width: 0,
            pixel_height: 0,
        })
        .expect("resize PTY height");
    let height_reflow_checkpoint =
        wait_for_output_after(&transcript, height_resize_request, b"\x1b[3J", STEP_TIMEOUT);
    let height_history_checkpoint = wait_for_output_after(
        &transcript,
        height_reflow_checkpoint,
        b"resize accepted",
        STEP_TIMEOUT,
    );
    let height_footer_checkpoint = wait_for_output_after(
        &transcript,
        height_history_checkpoint + b"resize accepted".len(),
        b"/pty-test",
        STEP_TIMEOUT,
    );
    wait_for_output_after(
        &transcript,
        height_footer_checkpoint + b"/pty-test".len(),
        CURSOR_SHOW,
        STEP_TIMEOUT,
    );
    #[cfg(unix)]
    {
        let stateful_checkpoint = transcript_len(&transcript);
        let resize_bytes = transcript_bytes(&transcript);
        let mut terminal_state = vt100::Parser::new(30, 100, 200);
        terminal_state.process(&resize_bytes[..narrow_reflow_checkpoint]);
        terminal_state.screen_mut().set_size(30, 40);
        terminal_state.process(&resize_bytes[narrow_reflow_checkpoint..wide_reflow_checkpoint]);
        terminal_state.screen_mut().set_size(30, 120);
        terminal_state.process(&resize_bytes[wide_reflow_checkpoint..height_reflow_checkpoint]);
        terminal_state.screen_mut().set_size(20, 120);
        terminal_state.process(&resize_bytes[height_reflow_checkpoint..stateful_checkpoint]);
        let visible = terminal_state.screen().contents();
        assert_eq!(
            visible
                .lines()
                .filter(|line| line.trim_start().starts_with("> "))
                .count(),
            1,
            "stateful terminal contains a duplicated composer after resize:\n{visible}"
        );
        assert!(
            visible.contains("resize accepted"),
            "history was not replayed: {visible}"
        );
        terminal_state.screen_mut().set_scrollback(usize::MAX);
        let scrollback_rows = terminal_state.screen().scrollback();
        for offset in (0..=scrollback_rows).step_by(20) {
            terminal_state.screen_mut().set_scrollback(offset);
            let page = terminal_state.screen().contents();
            assert!(
                page.lines()
                    .filter(|line| line.trim_start().starts_with("> "))
                    .count()
                    <= 1,
                "stateful scrollback page contains duplicate composers at offset {offset}:\n{page}"
            );
        }
        terminal_state.screen_mut().set_scrollback(0);
    }
    let restored_checkpoint = transcript_len(&transcript);
    pair.master
        .resize(PtySize {
            rows: 30,
            cols: 120,
            pixel_width: 0,
            pixel_height: 0,
        })
        .expect("restore PTY height");
    wait_for_output_after(&transcript, restored_checkpoint, b"\x1b[3J", STEP_TIMEOUT);

    send(&writer, shift_tab_key);
    wait_for_action(&action_log, "TogglePlan", &transcript);
    wait_for_output(&transcript, b"plan toggled", STEP_TIMEOUT);

    send_text(&writer, "/profile");
    send(&writer, enter_key);
    wait_for_action(&action_log, "SubmitText:/profile", &transcript);
    wait_for_output(&transcript, b"Choose a model profile", STEP_TIMEOUT);

    send(&writer, b"\x1b[B");
    send(&writer, enter_key);
    // Ratatui updates the picker header by terminal diff, so the full title is
    // not guaranteed to occur contiguously in the raw PTY stream. This form-
    // specific line is newly rendered and is therefore a stable readiness
    // marker before typing into the profile-name field.
    wait_for_output(
        &transcript,
        b"Allowed: letters, numbers, - and _",
        STEP_TIMEOUT,
    );
    send_text(&writer, "pty_profile");
    send(&writer, enter_key);
    wait_for_action(&action_log, "CreateProfile:pty_profile", &transcript);
    wait_for_output(&transcript, b"profile created", STEP_TIMEOUT);

    let config_checkpoint = transcript_len(&transcript);
    send_text(&writer, "/config");
    send(&writer, enter_key);
    wait_for_action(&action_log, "SubmitText:/config", &transcript);
    wait_for_output_after(
        &transcript,
        config_checkpoint,
        b"SubBake configuration",
        STEP_TIMEOUT,
    );
    let config_narrow_request = transcript_len(&transcript);
    pair.master
        .resize(PtySize {
            rows: 30,
            cols: 40,
            pixel_width: 0,
            pixel_height: 0,
        })
        .expect("resize PTY configuration editor");
    let config_narrow_clear =
        wait_for_output_after(&transcript, config_narrow_request, b"\x1b[2J", STEP_TIMEOUT);
    wait_for_output_after(
        &transcript,
        config_narrow_clear + b"\x1b[2J".len(),
        b"Tab focus",
        STEP_TIMEOUT,
    );
    let config_navigation_checkpoint = transcript_len(&transcript);
    send(&writer, b"\t\x1b[B");
    wait_for_output_after(
        &transcript,
        config_navigation_checkpoint,
        b"Active",
        STEP_TIMEOUT,
    );
    let config_resize_checkpoint = transcript_len(&transcript);
    pair.master
        .resize(PtySize {
            rows: 30,
            cols: 120,
            pixel_width: 0,
            pixel_height: 0,
        })
        .expect("widen PTY configuration editor");
    let config_wide_clear = wait_for_output_after(
        &transcript,
        config_resize_checkpoint,
        b"\x1b[2J",
        STEP_TIMEOUT,
    );
    wait_for_output_after(
        &transcript,
        config_wide_clear + b"\x1b[2J".len(),
        b"Tab focus",
        STEP_TIMEOUT,
    );
    let config_close_checkpoint = transcript_len(&transcript);
    send(&writer, escape_key);
    #[cfg(unix)]
    wait_for_output_after(
        &transcript,
        config_close_checkpoint,
        ALT_SCREEN_LEAVE,
        STEP_TIMEOUT,
    );
    #[cfg(windows)]
    wait_for_output_after(
        &transcript,
        config_close_checkpoint,
        b"/pty-test",
        STEP_TIMEOUT,
    );

    send_text(&writer, "make a plan");
    send(&writer, enter_key);
    wait_for_action(&action_log, "SubmitText:make a plan", &transcript);
    wait_for_output(&transcript, b"Run this plan?", STEP_TIMEOUT);
    send(&writer, b"\x1b[B");
    send(&writer, enter_key);
    wait_for_action(&action_log, "RejectApproval", &transcript);
    wait_for_output(&transcript, b"plan rejected", STEP_TIMEOUT);

    send_text(&writer, "run command");
    send(&writer, enter_key);
    wait_for_action(&action_log, "SubmitText:run command", &transcript);
    wait_for_output(&transcript, b"Run this operation?", STEP_TIMEOUT);
    send(&writer, enter_key);
    wait_for_action(&action_log, "ApproveApproval", &transcript);
    wait_for_output(&transcript, b"command continued", STEP_TIMEOUT);

    send_text(&writer, "slow work");
    send(&writer, enter_key);
    wait_for_action(&action_log, "SubmitText:slow work", &transcript);
    send_text(&writer, "steer current work");
    send(&writer, escape_key);
    wait_for_output(&transcript, b"Sent to the active turn.", STEP_TIMEOUT);
    send_text(&writer, "queued follow-up");
    send(&writer, enter_key);
    wait_for_output(&transcript, b"Queued for the next turn.", STEP_TIMEOUT);
    wait_for_action(&action_log, "SubmitText:queued follow-up", &transcript);
    wait_for_output(&transcript, b"queued follow-up complete", STEP_TIMEOUT);

    send_text(&writer, "cancel me");
    send(&writer, enter_key);
    wait_for_action(&action_log, "SubmitText:cancel me", &transcript);
    send(&writer, escape_key);
    wait_for_action(&action_log, "CancellationObserved", &transcript);
    wait_for_output(&transcript, b"Cancelled.", STEP_TIMEOUT);

    send_text(&writer, "after cancel");
    send(&writer, enter_key);
    wait_for_action(&action_log, "SubmitText:after cancel", &transcript);
    wait_for_output(&transcript, b"worker recovered", STEP_TIMEOUT);

    send_text(&writer, "inspect file");
    send(&writer, enter_key);
    wait_for_action(&action_log, "SubmitText:inspect file", &transcript);
    // Ratatui may emit the transient "Reading" label as multiple cursor-diff
    // writes, so the raw PTY byte stream is not guaranteed to contain that
    // visual text contiguously. The stable completed activity remains a
    // reliable end-to-end assertion; running labels are covered by unit tests.
    wait_for_output(&transcript, b"Read sample.srt", STEP_TIMEOUT);
    wait_for_output(&transcript, b"inspection complete", STEP_TIMEOUT);

    send_text(&writer, "cancel and exit");
    send(&writer, enter_key);
    wait_for_action(&action_log, "SubmitText:cancel and exit", &transcript);
    send(&writer, b"\x03");
    wait_for_action(&action_log, "CancellationObservedOnExit", &transcript);
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
    #[cfg(unix)]
    {
        let before = marker_value(&output, "PTY_STTY_BEFORE:").expect("stty before marker");
        let after = marker_value(&output, "PTY_STTY_AFTER:").expect("stty after marker");
        assert_eq!(before, after, "terminal attributes were not restored");
        assert_eq!(
            marker_value(&output, "PTY_SHELL_STATUS:").as_deref(),
            Some("0")
        );
    }

    // Unix crossterm uses ANSI alternate-screen commands, while Windows
    // crossterm switches ConPTY screen buffers through WinAPI. Only assert on
    // transport bytes where those bytes are the platform contract; the
    // cross-platform interaction assertions above prove the overlay opened,
    // closed, and returned control to the main TUI.
    #[cfg(unix)]
    {
        let alternate_screen_enters = count_subslice(&output, b"\x1b[?1049h");
        assert!(
            alternate_screen_enters > 0,
            "the Unix PTY session must enter the alternate screen"
        );
        assert_eq!(
            alternate_screen_enters,
            count_subslice(&output, b"\x1b[?1049l"),
            "alternate-screen enter/leave must be paired"
        );
    }
    assert_eq!(
        count_subslice(&output, b"\x1b[>1u"),
        count_subslice(&output, b"\x1b[<1u"),
        "negotiated keyboard enhancement push/pop must be paired"
    );
    assert!(
        count_subslice(&output, DSR_QUERY) > 0,
        "the PTY session must exercise a real DSR query"
    );
    assert!(
        !output
            .windows("⚡".len())
            .any(|window| window == "⚡".as_bytes()),
        "tool activity must not use the lightning icon"
    );
    assert!(
        !output
            .windows(br#"{"path":"sample.srt"}"#.len())
            .any(|window| window == br#"{"path":"sample.srt"}"#),
        "tool arguments must not be rendered as JSON"
    );
    assert!(
        !output
            .windows(b"private subtitle body".len())
            .any(|window| window == b"private subtitle body"),
        "observation contents must not leak into the activity summary"
    );
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
        version: env!("CARGO_PKG_VERSION").to_owned(),
        provider: "mock".to_owned(),
        model: "pty-model".to_owned(),
        config: "PTY test".to_owned(),
        cache_enabled: false,
        cwd: "/pty-test".to_owned(),
    });
    tui.set_cancellation_token(cancellation);
    tui.set_turn_steering(TurnSteering::default());

    tui.run(move |action, guard, observer| {
        append_action(&action_log, &action_label(&action))?;
        scripted_interaction(action, &guard, &action_log, observer)
    })
    .expect("run TUI PTY scenario");
}

fn scripted_interaction(
    action: TuiAction,
    guard: &CancellationGuard,
    action_log: &Path,
    observer: &mut TuiObserver,
) -> AgentResult<TuiInteraction> {
    match action {
        TuiAction::TogglePlan => Ok(TuiInteraction::Message {
            message: "plan toggled".to_owned(),
        }),
        TuiAction::SubmitText(input) if input == "after resize" => Ok(TuiInteraction::Message {
            message: "resize accepted".to_owned(),
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
        TuiAction::SubmitText(input) if input == "/config" => Ok(TuiInteraction::ConfigEditor {
            message: String::new(),
            snapshot: config_snapshot("defaults"),
            provider: "mock".to_owned(),
            model: "pty-model".to_owned(),
            cache_enabled: false,
        }),
        TuiAction::ApplyConfig { .. } => Ok(TuiInteraction::ConfigEditor {
            message: "Saved configuration.".to_owned(),
            snapshot: config_snapshot("saved"),
            provider: "mock".to_owned(),
            model: "pty-model".to_owned(),
            cache_enabled: true,
        }),
        TuiAction::SubmitText(input) if input == "make a plan" => Ok(TuiInteraction::Approval {
            prompt: ApprovalPrompt {
                kind: ApprovalKind::Plan,
                title: "Run this plan?".to_owned(),
                purpose: "Exercise plan approval".to_owned(),
                reason: "The plan changes a test file".to_owned(),
                operation: vec!["Update sample.srt".to_owned()],
            },
        }),
        TuiAction::RejectApproval => Ok(TuiInteraction::Message {
            message: "plan rejected".to_owned(),
        }),
        TuiAction::SubmitText(input) if input == "run command" => Ok(TuiInteraction::Approval {
            prompt: ApprovalPrompt {
                kind: ApprovalKind::Command,
                title: "Run this operation?".to_owned(),
                purpose: "Exercise command approval".to_owned(),
                reason: "The command is outside the auto-run set".to_owned(),
                operation: vec!["printf pty".to_owned()],
            },
        }),
        TuiAction::ApproveApproval => Ok(TuiInteraction::Message {
            message: "command continued".to_owned(),
        }),
        TuiAction::ReviseApproval(input) => Ok(TuiInteraction::Message {
            message: format!("approval revised: {input}"),
        }),
        TuiAction::SubmitText(input) if input == "slow work" => {
            thread::sleep(Duration::from_millis(1500));
            Ok(TuiInteraction::Message {
                message: "slow work complete".to_owned(),
            })
        }
        TuiAction::SubmitText(input) if input == "queued follow-up" => {
            Ok(TuiInteraction::Message {
                message: "queued follow-up complete".to_owned(),
            })
        }
        TuiAction::SubmitText(input) if input == "cancel me" => {
            while !guard.is_cancelled() {
                thread::sleep(Duration::from_millis(10));
            }
            append_action(action_log, "CancellationObserved")?;
            Err(AgentError::Cancelled)
        }
        TuiAction::SubmitText(input) if input == "cancel and exit" => {
            while !guard.is_cancelled() {
                thread::sleep(Duration::from_millis(10));
            }
            append_action(action_log, "CancellationObservedOnExit")?;
            Err(AgentError::Cancelled)
        }
        TuiAction::SubmitText(input) if input == "after cancel" => Ok(TuiInteraction::Message {
            message: "worker recovered".to_owned(),
        }),
        TuiAction::SubmitText(input) if input == "inspect file" => {
            let arguments = serde_json::json!({"path":"sample.srt"});
            observer.on_tool_call("pty-read", "read_file_preview", &arguments);
            thread::sleep(Duration::from_millis(200));
            let outcome =
                subbake_core::AgentToolOutcome::Observation(subbake_core::ObservationToolOutcome {
                    status: subbake_core::ToolExecutionStatus::Observed,
                    observation: "read_file_preview".to_owned(),
                    content: "private subtitle body".to_owned(),
                });
            observer.on_tool_success("pty-read", "read_file_preview", &arguments, &outcome);
            Ok(TuiInteraction::Message {
                message: "inspection complete".to_owned(),
            })
        }
        unexpected => Err(AgentError::invalid_input(format!(
            "unexpected PTY test action: {unexpected:?}"
        ))),
    }
}

fn config_snapshot(profile: &str) -> ConfigEditorSnapshot {
    ConfigEditorSnapshot {
        path: PathBuf::from("subbake.toml"),
        target: subbake_adapters::ConfigEditTarget::Defaults,
        active_profile: None,
        profiles: Vec::new(),
        fields: vec![
            ConfigFieldView {
                id: ConfigFieldId::ActiveProfile,
                value: profile.to_owned(),
                inherited: false,
                configured: true,
            },
            ConfigFieldView {
                id: ConfigFieldId::AgentMaxSteps,
                value: "64".to_owned(),
                inherited: true,
                configured: true,
            },
            ConfigFieldView {
                id: ConfigFieldId::AgentAutoApprove,
                value: "false".to_owned(),
                inherited: true,
                configured: true,
            },
        ],
    }
}

fn action_label(action: &TuiAction) -> String {
    match action {
        TuiAction::SubmitText(input) => format!("SubmitText:{input}"),
        TuiAction::ApproveApproval => "ApproveApproval".to_owned(),
        TuiAction::RejectApproval => "RejectApproval".to_owned(),
        TuiAction::ReviseApproval(input) => format!("ReviseApproval:{input}"),
        TuiAction::SelectProfile(name) => format!("SelectProfile:{name}"),
        TuiAction::CreateProfile(name) => format!("CreateProfile:{name}"),
        TuiAction::SelectConfigProfile(name) => format!("SelectConfigProfile:{name}"),
        TuiAction::CreateConfigProfile(name) => format!("CreateConfigProfile:{name}"),
        TuiAction::ApplyConfig { changes, after } => {
            format!("ApplyConfig:{}:{after:?}", changes.len())
        }
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

fn transcript_len(transcript: &Transcript) -> usize {
    transcript.bytes.lock().expect("lock PTY transcript").len()
}

fn transcript_contains(transcript: &Transcript, needle: &[u8]) -> bool {
    contains_subslice(
        &transcript.bytes.lock().expect("lock PTY transcript"),
        needle,
    )
}

fn wait_for_output_after(
    transcript: &Transcript,
    checkpoint: usize,
    needle: &[u8],
    timeout: Duration,
) -> usize {
    let deadline = Instant::now() + timeout;
    let mut bytes = transcript.bytes.lock().expect("lock PTY transcript");
    while !bytes
        .get(checkpoint..)
        .is_some_and(|tail| contains_subslice(tail, needle))
    {
        let now = Instant::now();
        if now >= deadline {
            panic!(
                "timed out waiting for new output {:?}; transcript: {}",
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
        if result.timed_out()
            && !bytes
                .get(checkpoint..)
                .is_some_and(|tail| contains_subslice(tail, needle))
        {
            panic!(
                "timed out waiting for new output {:?}; transcript: {}",
                String::from_utf8_lossy(needle),
                escape_bytes(&bytes)
            );
        }
    }
    checkpoint
        + bytes
            .get(checkpoint..)
            .and_then(|tail| {
                tail.windows(needle.len())
                    .position(|window| window == needle)
            })
            .expect("waited-for PTY output is present")
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

fn unique_test_dir() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("subbake-real-pty-")
        .tempdir()
        .expect("create PTY test directory")
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

#[cfg(unix)]
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
