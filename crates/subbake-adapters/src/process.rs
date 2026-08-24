use std::ffi::OsStr;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::sync::mpsc;
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
            None,
            LineStream::None,
            |_| {},
        )
    }

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
            Some(timeout),
            LineStream::None,
            |_| {},
        )
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
            None,
            LineStream::Stdout,
            on_line,
        )
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
            None,
            LineStream::Stderr,
            on_line,
        )
    }

    fn run_inner(
        command: &mut Command,
        cancellation: &CancellationGuard,
        context: &str,
        timeout: Option<Duration>,
        line_stream: LineStream,
        mut on_line: impl FnMut(&str),
    ) -> AdapterResult<Output> {
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
        let (sender, receiver) = mpsc::channel();
        let (stdout_reader, stderr_reader) = match line_stream {
            LineStream::None => {
                drop(sender);
                (
                    thread::spawn(move || read_all(stdout)),
                    thread::spawn(move || read_all(stderr)),
                )
            }
            LineStream::Stdout => (
                thread::spawn(move || Self::read_lines(stdout, sender)),
                thread::spawn(move || read_all(stderr)),
            ),
            LineStream::Stderr => (
                thread::spawn(move || read_all(stdout)),
                thread::spawn(move || Self::read_lines(stderr, sender)),
            ),
        };
        let started = std::time::Instant::now();

        let status = loop {
            while let Ok(line) = receiver.try_recv() {
                on_line(&line);
            }
            if cancellation.is_cancelled() {
                Self::terminate(&mut child);
                join_readers_after_termination(stdout_reader, stderr_reader, context)?;
                return Err(AdapterError::Cancelled);
            }
            if let Some(limit) = timeout
                && started.elapsed() >= limit
            {
                Self::terminate(&mut child);
                join_readers_after_termination(stdout_reader, stderr_reader, context)?;
                return Err(AdapterError::Timeout {
                    message: format!("{context} exceeded its {} second timeout", limit.as_secs()),
                });
            }
            if let Some(status) = child.try_wait()? {
                break status;
            }
            thread::sleep(Duration::from_millis(25));
        };
        while let Ok(line) = receiver.try_recv() {
            on_line(&line);
        }
        let stdout = join_reader(stdout_reader, context, "stdout");
        let stderr = join_reader(stderr_reader, context, "stderr");
        Ok(Output {
            status,
            stdout: stdout?,
            stderr: stderr?,
        })
    }

    fn read_lines(reader: impl Read, sender: mpsc::Sender<String>) -> io::Result<Vec<u8>> {
        let mut reader = BufReader::new(reader);
        let mut output = Vec::new();
        loop {
            let start = output.len();
            if reader.read_until(b'\n', &mut output)? == 0 {
                break;
            }
            let line = String::from_utf8_lossy(&output[start..]);
            let _ = sender.send(line.trim().to_owned());
        }
        Ok(output)
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

fn read_all(mut reader: impl Read) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    reader.read_to_end(&mut output)?;
    Ok(output)
}

fn join_reader(
    reader: thread::JoinHandle<io::Result<Vec<u8>>>,
    context: &str,
    stream: &str,
) -> AdapterResult<Vec<u8>> {
    reader
        .join()
        .map_err(|_| io::Error::other(format!("{context}: {stream} reader panicked")))?
        .map_err(AdapterError::from)
}

fn join_readers_after_termination(
    stdout_reader: thread::JoinHandle<io::Result<Vec<u8>>>,
    stderr_reader: thread::JoinHandle<io::Result<Vec<u8>>>,
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
            "tree" => {
                let mut grandchild = helper_command("grandchild")
                    .spawn()
                    .expect("spawn grandchild");
                let _ = grandchild.wait();
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
