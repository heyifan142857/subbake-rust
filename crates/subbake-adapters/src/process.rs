use std::io::{self, Read};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::Duration;

use subbake_core::CancellationGuard;

use crate::error::{AdapterError, AdapterResult};

/// Run a child process while continuously draining both output pipes.
///
/// Waiting to read piped output until after process exit can deadlock once a
/// verbose child fills an OS pipe. Dedicated readers keep the child moving and
/// also preserve diagnostics for the caller.
pub(crate) fn run_command_cancellable(
    command: &mut Command,
    cancellation: &CancellationGuard,
    context: &str,
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
    let stdout_reader = thread::spawn(move || read_all(stdout));
    let stderr_reader = thread::spawn(move || read_all(stderr));

    let status = loop {
        if cancellation.is_cancelled() {
            #[cfg(unix)]
            {
                let _ = Command::new("kill")
                    .args(["-TERM", &format!("-{}", child.id())])
                    .status();
            }
            let _ = child.kill();
            let _ = child.wait();
            // Dropping JoinHandle detaches the readers. This is deliberate:
            // descendants may still hold inherited pipe handles briefly, and
            // joining here would turn cooperative cancellation into a hang.
            drop(stdout_reader);
            drop(stderr_reader);
            return Err(AdapterError::Cancelled);
        }
        if let Some(status) = child.try_wait()? {
            break status;
        }
        thread::sleep(Duration::from_millis(25));
    };

    Ok(Output {
        status,
        stdout: join_reader(stdout_reader, context, "stdout")?,
        stderr: join_reader(stderr_reader, context, "stderr")?,
    })
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
        let error = run_command_cancellable(
            Command::new("sh").args(["-c", "while true; do sleep 1; done"]),
            &guard,
            "test child",
        )
        .expect_err("child should be cancelled");
        canceller.join().expect("join canceller");

        assert!(error.is_cancelled());
    }
}
