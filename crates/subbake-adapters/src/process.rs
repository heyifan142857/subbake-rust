use std::collections::VecDeque;
use std::ffi::OsStr;
use std::io::{self, Read};
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

use command_group::{CommandGroup, GroupChild};
use subbake_core::CancellationGuard;

use crate::error::{AdapterError, AdapterResult};

/// Owns child-process lifecycle policy for every adapter. Callers construct
/// commands, while this boundary consistently applies cancellation, pipe
/// draining, process-group setup, and termination.
pub(crate) struct ProcessSupervisor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineStream {
    None,
    Stdout,
    Stderr,
}

const COMPLETE_OUTPUT_LIMIT: usize = 64 * 1024 * 1024;
const PROGRESS_LINE_LIMIT: usize = 64 * 1024;
const PROGRESS_QUEUE_LIMIT: usize = 64;

#[derive(Clone, Copy)]
struct CapturePolicy {
    limit: usize,
    fail_on_overflow: bool,
}

impl Default for CapturePolicy {
    fn default() -> Self {
        Self {
            limit: COMPLETE_OUTPUT_LIMIT,
            fail_on_overflow: true,
        }
    }
}

struct ProcessOptions {
    timeout: Option<Duration>,
    line_stream: LineStream,
    capture: CapturePolicy,
}

pub(crate) struct CapturedProcessOutput {
    pub output: Output,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
}

struct CapturedStream {
    bytes: Vec<u8>,
    truncated: bool,
}

/// Retains a prefix and rolling suffix without ever accumulating full output.
struct OutputBuffer {
    prefix: Vec<u8>,
    tail: VecDeque<u8>,
    policy: CapturePolicy,
    total: usize,
}

impl OutputBuffer {
    fn new(policy: CapturePolicy) -> Self {
        Self {
            prefix: Vec::new(),
            tail: VecDeque::new(),
            policy,
            total: 0,
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        self.total = self.total.saturating_add(bytes.len());
        let prefix_limit = if self.policy.fail_on_overflow {
            self.policy.limit
        } else {
            self.policy.limit / 2
        };
        let take = bytes
            .len()
            .min(prefix_limit.saturating_sub(self.prefix.len()));
        self.prefix.extend_from_slice(&bytes[..take]);
        let remaining = &bytes[take..];
        let tail_limit = self.policy.limit - prefix_limit;
        if tail_limit == 0 {
            return;
        }
        if remaining.len() >= tail_limit {
            self.tail.clear();
            self.tail.extend(&remaining[remaining.len() - tail_limit..]);
        } else {
            let discard = (self.tail.len() + remaining.len()).saturating_sub(tail_limit);
            self.tail.drain(..discard);
            self.tail.extend(remaining);
        }
    }

    fn finish(mut self) -> CapturedStream {
        self.prefix.extend(self.tail);
        CapturedStream {
            bytes: self.prefix,
            truncated: self.total > self.policy.limit,
        }
    }
}

impl ProcessSupervisor {
    /// Run a child process while continuously draining both output pipes.
    ///
    /// Waiting to read piped output until after process exit can deadlock once a
    /// verbose child fills an OS pipe. Dedicated readers keep the child moving and
    /// also preserve diagnostics for the caller.
    pub(crate) fn run(
        command: &mut Command,
        cancellation: &CancellationGuard,
        context: &str,
    ) -> AdapterResult<Output> {
        Self::run_inner(
            command,
            cancellation,
            context,
            ProcessOptions {
                timeout: None,
                line_stream: LineStream::None,
                capture: CapturePolicy::default(),
            },
            |_| {},
        )
        .map(|captured| captured.output)
    }

    #[cfg(test)]
    pub(crate) fn run_with_timeout(
        command: &mut Command,
        cancellation: &CancellationGuard,
        context: &str,
        timeout: Duration,
    ) -> AdapterResult<Output> {
        Self::run_inner(
            command,
            cancellation,
            context,
            ProcessOptions {
                timeout: Some(timeout),
                line_stream: LineStream::None,
                capture: CapturePolicy::default(),
            },
            |_| {},
        )
        .map(|captured| captured.output)
    }

    /// Run a child while delivering complete stdout lines to the caller as they
    /// arrive. Stderr is still continuously drained and retained for diagnostics.
    pub(crate) fn run_with_stdout_lines(
        command: &mut Command,
        cancellation: &CancellationGuard,
        context: &str,
        on_line: impl FnMut(&str),
    ) -> AdapterResult<Output> {
        Self::run_inner(
            command,
            cancellation,
            context,
            ProcessOptions {
                timeout: None,
                line_stream: LineStream::Stdout,
                capture: CapturePolicy::default(),
            },
            on_line,
        )
        .map(|captured| captured.output)
    }

    /// Run a child while delivering complete stderr lines to the caller as they
    /// arrive. Stdout is continuously drained and retained with the diagnostics.
    pub(crate) fn run_with_stderr_lines(
        command: &mut Command,
        cancellation: &CancellationGuard,
        context: &str,
        on_line: impl FnMut(&str),
    ) -> AdapterResult<Output> {
        Self::run_inner(
            command,
            cancellation,
            context,
            ProcessOptions {
                timeout: None,
                line_stream: LineStream::Stderr,
                capture: CapturePolicy::default(),
            },
            on_line,
        )
        .map(|captured| captured.output)
    }

    pub(crate) fn run_bounded_with_timeout(
        command: &mut Command,
        cancellation: &CancellationGuard,
        context: &str,
        timeout: Duration,
        limit: usize,
    ) -> AdapterResult<CapturedProcessOutput> {
        Self::run_inner(
            command,
            cancellation,
            context,
            ProcessOptions {
                timeout: Some(timeout),
                line_stream: LineStream::None,
                capture: CapturePolicy {
                    limit,
                    fail_on_overflow: false,
                },
            },
            |_| {},
        )
    }

    fn run_inner(
        command: &mut Command,
        cancellation: &CancellationGuard,
        context: &str,
        options: ProcessOptions,
        mut on_line: impl FnMut(&str),
    ) -> AdapterResult<CapturedProcessOutput> {
        cancellation.check().map_err(AdapterError::from)?;
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
        let program = command.get_program().to_os_string();
        let mut child = command
            .group_spawn()
            .map_err(|source| spawn_error(&program, context, source))?;
        let stdout = child
            .inner()
            .stdout
            .take()
            .ok_or_else(|| io::Error::other(format!("{context}: stdout pipe unavailable")))?;
        let stderr = child
            .inner()
            .stderr
            .take()
            .ok_or_else(|| io::Error::other(format!("{context}: stderr pipe unavailable")))?;
        let (sender, receiver) = mpsc::sync_channel(PROGRESS_QUEUE_LIMIT);
        let overflow = Arc::new(AtomicBool::new(false));
        let capture = options.capture;
        let stdout_sender = (options.line_stream == LineStream::Stdout).then(|| sender.clone());
        let stderr_sender = (options.line_stream == LineStream::Stderr).then(|| sender.clone());
        drop(sender);
        let stdout_overflow = Arc::clone(&overflow);
        let stderr_overflow = Arc::clone(&overflow);
        let stdout_reader =
            thread::spawn(move || read_stream(stdout, capture, stdout_sender, stdout_overflow));
        let stderr_reader =
            thread::spawn(move || read_stream(stderr, capture, stderr_sender, stderr_overflow));
        let started = std::time::Instant::now();

        let mut exit_status = None;
        let status = loop {
            for _ in 0..PROGRESS_QUEUE_LIMIT {
                let Ok(line) = receiver.try_recv() else {
                    break;
                };
                on_line(&line);
            }
            if cancellation.is_cancelled() {
                Self::terminate(&mut child);
                drop(receiver);
                join_readers_after_termination(stdout_reader, stderr_reader, context)?;
                return Err(AdapterError::Cancelled);
            }
            if let Some(limit) = options.timeout
                && started.elapsed() >= limit
            {
                Self::terminate(&mut child);
                drop(receiver);
                join_readers_after_termination(stdout_reader, stderr_reader, context)?;
                return Err(AdapterError::Timeout {
                    message: format!("{context} exceeded its {} second timeout", limit.as_secs()),
                });
            }
            if overflow.load(Ordering::Acquire) {
                Self::terminate(&mut child);
                drop(receiver);
                join_readers_after_termination(stdout_reader, stderr_reader, context)?;
                return Err(AdapterError::ChildProcess {
                    program: "external process",
                    status: None,
                    message: format!(
                        "{context}: output exceeded the {} byte capture limit",
                        capture.limit
                    ),
                });
            }
            if exit_status.is_none() {
                match child.try_wait() {
                    Ok(status) => exit_status = status,
                    Err(error) => {
                        Self::terminate(&mut child);
                        drop(receiver);
                        join_readers_after_termination(stdout_reader, stderr_reader, context)?;
                        return Err(error.into());
                    }
                }
            }
            // Descendants can retain pipes after the group leader exits. Keep
            // enforcing cancellation/timeouts until both readers reach EOF.
            if let Some(status) = exit_status
                && stdout_reader.is_finished()
                && stderr_reader.is_finished()
            {
                break status;
            }
            thread::sleep(Duration::from_millis(25));
        };
        while let Ok(line) = receiver.try_recv() {
            on_line(&line);
        }
        let stdout = join_reader(stdout_reader, context, "stdout");
        let stderr = join_reader(stderr_reader, context, "stderr");
        let stdout = stdout?;
        let stderr = stderr?;
        if capture.fail_on_overflow && (stdout.truncated || stderr.truncated) {
            return Err(AdapterError::ChildProcess {
                program: "external process",
                status: Some(status.code().unwrap_or(-1)),
                message: format!(
                    "{context}: output exceeded the {} byte capture limit",
                    capture.limit
                ),
            });
        }
        Ok(CapturedProcessOutput {
            stdout_truncated: stdout.truncated,
            stderr_truncated: stderr.truncated,
            output: Output {
                status,
                stdout: stdout.bytes,
                stderr: stderr.bytes,
            },
        })
    }

    pub(crate) fn terminate(child: &mut GroupChild) {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn spawn_error(program: &OsStr, context: &str, source: io::Error) -> AdapterError {
    if source.kind() == io::ErrorKind::NotFound
        && let Some(hint) = ffmpeg_install_hint(program)
    {
        return AdapterError::invalid_input(hint);
    }
    AdapterError::from(io::Error::new(
        source.kind(),
        format!("{context}: {source}"),
    ))
}

fn ffmpeg_install_hint(program: &OsStr) -> Option<String> {
    let executable = Path::new(program)
        .file_stem()?
        .to_str()?
        .to_ascii_lowercase();
    let missing = match executable.as_str() {
        "ffmpeg" => "FFmpeg",
        "ffprobe" => "ffprobe",
        _ => return None,
    };
    Some(format!(
        "Required media dependency `{missing}` is missing or not on PATH. Install the FFmpeg package, then verify that both `ffmpeg -version` and `ffprobe -version` work."
    ))
}

fn read_stream(
    mut reader: impl Read,
    policy: CapturePolicy,
    sender: Option<mpsc::SyncSender<String>>,
    overflow: Arc<AtomicBool>,
) -> io::Result<CapturedStream> {
    let mut output = OutputBuffer::new(policy);
    let mut buffer = [0_u8; 16 * 1024];
    let mut line = Vec::new();
    let mut line_overflow = false;
    loop {
        let count = match reader.read(&mut buffer) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            other => other?,
        };
        if count == 0 {
            break;
        }
        let bytes = &buffer[..count];
        output.push(bytes);
        if policy.fail_on_overflow && output.total > policy.limit {
            overflow.store(true, Ordering::Release);
        }
        if let Some(sender) = &sender {
            for byte in bytes {
                if *byte == b'\n' {
                    if !line_overflow {
                        let _ = sender.send(String::from_utf8_lossy(&line).trim().to_owned());
                    }
                    line.clear();
                    line_overflow = false;
                } else if line.len() < PROGRESS_LINE_LIMIT {
                    line.push(*byte);
                } else {
                    line_overflow = true;
                }
            }
        }
    }
    if let Some(sender) = sender
        && !line.is_empty()
        && !line_overflow
    {
        let _ = sender.send(String::from_utf8_lossy(&line).trim().to_owned());
    }
    Ok(output.finish())
}

fn join_reader(
    reader: thread::JoinHandle<io::Result<CapturedStream>>,
    context: &str,
    stream: &str,
) -> AdapterResult<CapturedStream> {
    reader
        .join()
        .map_err(|_| io::Error::other(format!("{context}: {stream} reader panicked")))?
        .map_err(AdapterError::from)
}

fn join_readers_after_termination(
    stdout_reader: thread::JoinHandle<io::Result<CapturedStream>>,
    stderr_reader: thread::JoinHandle<io::Result<CapturedStream>>,
    context: &str,
) -> AdapterResult<()> {
    let stdout = join_reader(stdout_reader, context, "stdout");
    let stderr = join_reader(stderr_reader, context, "stderr");
    let _ = stdout?;
    let _ = stderr?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;
    use std::process::Command;
    use std::thread;
    use std::time::Duration;

    use subbake_core::CancellationToken;

    use super::*;

    const HELPER_MODE: &str = "SUBBAKE_PROCESS_TEST_HELPER";
    const HELPER_SENTINEL: &str = "SUBBAKE_PROCESS_TEST_SENTINEL";

    #[test]
    fn process_test_helper() {
        let Some(mode) = std::env::var_os(HELPER_MODE) else {
            return;
        };
        match mode.to_string_lossy().as_ref() {
            "sleep" => thread::sleep(Duration::from_secs(30)),
            "grandchild" => {
                thread::sleep(Duration::from_millis(500));
                let path = std::env::var_os(HELPER_SENTINEL).expect("sentinel path");
                fs::write(path, b"survived").expect("write sentinel");
            }
            "orphan-pipe" => {
                // Deliberately orphan the helper to exercise inherited-pipe shutdown.
                #[allow(clippy::zombie_processes)]
                let _child = helper_command("sleep").spawn().expect("spawn pipe owner");
            }
            "tree" => {
                let mut grandchild = helper_command("grandchild")
                    .spawn()
                    .expect("spawn grandchild");
                let _ = grandchild.wait();
            }
            "progress-flood" => {
                for _ in 0..100_000 {
                    println!("progress=123");
                }
            }
            "output" => {
                let block = vec![b'x'; 128 * 1024];
                std::io::stdout().write_all(&block).expect("write stdout");
                std::io::stderr().write_all(&block).expect("write stderr");
            }
            other => panic!("unknown helper mode {other}"),
        }
    }

    fn helper_command(mode: &str) -> Command {
        let mut command = Command::new(std::env::current_exe().expect("test executable"));
        command
            .args([
                "--exact",
                "process::tests::process_test_helper",
                "--nocapture",
            ])
            .env(HELPER_MODE, mode);
        command
    }

    #[test]
    fn cancellation_terminates_a_running_child() {
        let token = CancellationToken::default();
        let guard = token.guard();
        let canceller = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            token.cancel();
        });
        let error = ProcessSupervisor::run(&mut helper_command("sleep"), &guard, "test child")
            .expect_err("child should be cancelled");
        canceller.join().expect("join canceller");

        assert!(error.is_cancelled());
    }

    #[test]
    fn timeout_uses_the_same_termination_path_as_cancellation() {
        let error = ProcessSupervisor::run_with_timeout(
            &mut helper_command("sleep"),
            &CancellationGuard::never(),
            "test child",
            Duration::from_millis(25),
        )
        .expect_err("child should time out");

        assert!(matches!(error, AdapterError::Timeout { .. }));
    }

    #[test]
    fn timeout_remains_active_after_leader_exits_with_inherited_pipes() {
        let started = std::time::Instant::now();
        let error = ProcessSupervisor::run_with_timeout(
            &mut helper_command("orphan-pipe"),
            &CancellationGuard::never(),
            "inherited pipes",
            Duration::from_millis(200),
        )
        .expect_err("pipe drain must time out");
        assert!(matches!(error, AdapterError::Timeout { .. }));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn cancellation_remains_active_after_leader_exits_with_inherited_pipes() {
        let token = CancellationToken::default();
        let guard = token.guard();
        let canceller = thread::spawn(move || {
            thread::sleep(Duration::from_millis(200));
            token.cancel();
        });
        let started = std::time::Instant::now();
        let error = ProcessSupervisor::run(
            &mut helper_command("orphan-pipe"),
            &guard,
            "inherited pipes",
        )
        .expect_err("pipe drain must cancel");
        canceller.join().expect("canceller");
        assert!(error.is_cancelled());
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn diagnostic_capture_is_bounded_while_draining_both_pipes() {
        let captured = ProcessSupervisor::run_bounded_with_timeout(
            &mut helper_command("output"),
            &CancellationGuard::never(),
            "large diagnostics",
            Duration::from_secs(5),
            1024,
        )
        .expect("drain bounded output");
        assert!(captured.output.status.success());
        assert_eq!(captured.output.stdout.len(), 1024);
        assert_eq!(captured.output.stderr.len(), 1024);
        assert!(captured.stdout_truncated && captured.stderr_truncated);
    }

    #[test]
    fn structured_capture_fails_instead_of_returning_truncated_payload() {
        let error = ProcessSupervisor::run_inner(
            &mut helper_command("output"),
            &CancellationGuard::never(),
            "structured data",
            ProcessOptions {
                timeout: Some(Duration::from_secs(5)),
                line_stream: LineStream::None,
                capture: CapturePolicy {
                    limit: 1024,
                    fail_on_overflow: true,
                },
            },
            |_| {},
        )
        .err()
        .expect("overflow must fail");
        assert!(
            error
                .to_string()
                .contains("output exceeded the 1024 byte capture limit")
        );
    }

    #[test]
    fn rolling_capture_preserves_prefix_suffix_and_never_grows_past_limit() {
        let mut output = OutputBuffer::new(CapturePolicy {
            limit: 1024,
            fail_on_overflow: false,
        });
        output.push(b"first marker");
        for _ in 0..1000 {
            output.push(&[b'x'; 8192]);
            assert!(output.prefix.len() + output.tail.len() <= 1024);
        }
        output.push(b"last marker");
        let result = output.finish();
        assert!(result.truncated);
        assert!(result.bytes.starts_with(b"first marker"));
        assert!(result.bytes.ends_with(b"last marker"));
    }

    #[test]
    fn cancellation_does_not_deadlock_a_full_progress_queue() {
        let token = CancellationToken::default();
        let guard = token.guard();
        let started = std::time::Instant::now();
        let error = ProcessSupervisor::run_with_stdout_lines(
            &mut helper_command("progress-flood"),
            &guard,
            "progress flood",
            |_| token.cancel(),
        )
        .expect_err("cancel flood");
        assert!(error.is_cancelled());
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn continuously_drains_both_output_streams() {
        let output = ProcessSupervisor::run(
            &mut helper_command("output"),
            &CancellationGuard::never(),
            "test output child",
        )
        .expect("run output helper");

        assert!(output.status.success());
        assert!(output.stdout.len() >= 128 * 1024);
        assert!(output.stderr.len() >= 128 * 1024);
    }

    #[test]
    fn missing_program_preserves_not_found_category() {
        let error = ProcessSupervisor::run(
            &mut Command::new("subbake-test-program-that-does-not-exist"),
            &CancellationGuard::never(),
            "test missing program",
        )
        .expect_err("missing program should fail");

        assert!(error.is_not_found());
    }

    #[test]
    fn missing_ffmpeg_names_dependency_without_hard_coding_a_package_manager() {
        let missing = std::env::temp_dir()
            .join("subbake-missing-ffmpeg-test")
            .join("ffmpeg");
        let error = ProcessSupervisor::run(
            &mut Command::new(missing),
            &CancellationGuard::never(),
            "test missing FFmpeg",
        )
        .expect_err("missing FFmpeg should fail");
        let message = error.to_string();

        assert!(message.contains("Required media dependency `FFmpeg` is missing"));
        assert!(message.contains("Install the FFmpeg package"));
        assert!(message.contains("ffprobe -version"));
        assert!(!message.contains("sudo "));
        assert!(!message.contains("brew "));
    }

    #[test]
    fn ffprobe_uses_the_same_ffmpeg_package_hint() {
        let hint = ffmpeg_install_hint(OsStr::new("/missing/ffprobe.exe")).expect("FFmpeg hint");

        assert!(hint.starts_with("Required media dependency `ffprobe` is missing"));
        assert!(hint.contains("Install the FFmpeg package"));
    }

    #[test]
    fn cancellation_terminates_descendant_processes() {
        let root = std::env::temp_dir().join(format!(
            "subbake-process-tree-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("create test root");
        let sentinel = root.join("survived.txt");
        let token = CancellationToken::default();
        let guard = token.guard();
        let canceller = thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            token.cancel();
        });
        let error = ProcessSupervisor::run(
            helper_command("tree").env(HELPER_SENTINEL, &sentinel),
            &guard,
            "test process tree",
        )
        .expect_err("process tree should be cancelled");
        canceller.join().expect("join canceller");
        thread::sleep(Duration::from_millis(650));

        assert!(error.is_cancelled());
        assert!(!sentinel.exists(), "grandchild survived process-group kill");
        let _ = fs::remove_dir_all(root);
    }
}
