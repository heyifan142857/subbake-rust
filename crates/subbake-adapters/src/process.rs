use std::io::{self, BufRead, BufReader, Read};
use std::process::{Command, Output, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

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
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .map_err(|source| io::Error::other(format!("{context}: {source}")))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other(format!("{context}: stdout pipe unavailable")))?;
        let stderr = child
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
                drop(stdout_reader);
                drop(stderr_reader);
                return Err(AdapterError::Cancelled);
            }
            if let Some(limit) = timeout
                && started.elapsed() >= limit
            {
                Self::terminate(&mut child);
                drop(stdout_reader);
                drop(stderr_reader);
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
        Ok(Output {
            status,
            stdout: join_reader(stdout_reader, context, "stdout")?,
            stderr: join_reader(stderr_reader, context, "stderr")?,
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

    pub(crate) fn terminate(child: &mut std::process::Child) {
        #[cfg(unix)]
        terminate_child_process_group(child);
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[cfg(unix)]
fn terminate_child_process_group(child: &std::process::Child) {
    use nix::sys::signal::{Signal, killpg};
    use nix::unistd::{Pid, getpgid};

    let Ok(raw_pid) = i32::try_from(child.id()) else {
        return;
    };
    let child_pid = Pid::from_raw(raw_pid);
    // Never signal a group inherited from the parent (for example a CI runner
    // job group). Commands configured with `process_group(0)` are safe group
    // targets only after the child is confirmed as that group's leader.
    if getpgid(Some(child_pid)) == Ok(child_pid) {
        let _ = killpg(child_pid, Signal::SIGTERM);
    }
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

#[cfg(all(test, unix))]
mod tests {
    use std::process::Command;
    use std::thread;
    use std::time::Duration;

    use subbake_core::CancellationToken;

    use super::*;

    #[test]
    fn cancellation_terminates_a_running_child() {
        let token = CancellationToken::default();
        let guard = token.guard();
        let canceller = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            token.cancel();
        });
        let error = ProcessSupervisor::run(
            Command::new("sh").args(["-c", "while true; do sleep 1; done"]),
            &guard,
            "test child",
        )
        .expect_err("child should be cancelled");
        canceller.join().expect("join canceller");

        assert!(error.is_cancelled());
    }

    #[test]
    fn termination_never_signals_an_inherited_process_group() {
        let mut child = Command::new("sh")
            .args(["-c", "while true; do sleep 1; done"])
            .spawn()
            .expect("spawn child in inherited process group");

        ProcessSupervisor::terminate(&mut child);

        assert!(child.try_wait().expect("poll terminated child").is_some());
    }

    #[test]
    fn timeout_uses_the_same_termination_path_as_cancellation() {
        let error = ProcessSupervisor::run_with_timeout(
            Command::new("sh").args(["-c", "while true; do sleep 1; done"]),
            &CancellationGuard::never(),
            "test child",
            Duration::from_millis(25),
        )
        .expect_err("child should time out");

        assert!(matches!(error, AdapterError::Timeout { .. }));
    }
}
