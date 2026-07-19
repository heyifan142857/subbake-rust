//! Sandboxed execution for the interactive coding agent.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use subbake_core::CancellationGuard;

use crate::error::{AdapterError, AdapterResult};

const OUTPUT_LIMIT: usize = 64 * 1024;

#[derive(Debug, Clone)]
pub struct SandboxedCommandRequest {
    pub command: String,
    pub project_root: PathBuf,
    pub cwd: PathBuf,
    pub staging_root: PathBuf,
    pub output_aliases: Vec<String>,
    pub network: bool,
    pub timeout: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxedCommandOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub duration: Duration,
}

pub fn run_sandboxed_command(
    request: &SandboxedCommandRequest,
    cancellation: &CancellationGuard,
) -> AdapterResult<SandboxedCommandOutput> {
    cancellation.check().map_err(AdapterError::from)?;
    if !cfg!(target_os = "linux") {
        return Err(AdapterError::invalid_input(
            "run_command currently requires Linux",
        ));
    }
    if !Path::new("/usr/bin/bwrap").is_file() && !path_has_program("bwrap") {
        return Err(AdapterError::invalid_input(
            "run_command requires bubblewrap (`bwrap`) on PATH",
        ));
    }

    std::fs::create_dir_all(&request.staging_root)?;
    let mut command = Command::new("bwrap");
    command.args([
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-uts",
        "--unshare-ipc",
        "--unshare-cgroup",
    ]);
    if !request.network {
        command.arg("--unshare-net");
    }
    command.args(["--ro-bind", "/", "/"]);
    command.args(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]);
    let private_environment = configure_home_visibility(&mut command, &request.project_root);
    command.args(["--tmpfs", path_arg(&request.project_root.join(".subbake"))]);
    command.args([
        "--bind",
        path_arg(&request.staging_root),
        "/tmp/outputs",
        "--chdir",
        path_arg(&request.cwd),
        "--clearenv",
        "--setenv",
        "HOME",
        "/tmp/home",
        "--setenv",
        "TMPDIR",
        "/tmp",
        "--setenv",
        "CARGO_TARGET_DIR",
        "/tmp/subbake-target",
        "--setenv",
        "XDG_CACHE_HOME",
        "/tmp/cache",
        "--setenv",
        "PATH",
        &safe_path(),
        "--setenv",
        "LANG",
        "C.UTF-8",
    ]);
    for (name, value) in private_environment {
        command.args(["--setenv", &name, &value]);
    }
    for alias in &request.output_aliases {
        command.args([
            "--setenv",
            &format!("SUBBAKE_OUTPUT_{}", alias.to_ascii_uppercase()),
            &format!("/tmp/outputs/{alias}"),
        ]);
    }
    command.args(["--", "/bin/bash", "-c", &request.command]);

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let started = Instant::now();
    let mut child = command.spawn().map_err(|source| AdapterError::ExternalIo {
        operation: "start sandboxed command",
        path: None,
        source,
    })?;
    let stdout = child.stdout.take().ok_or_else(|| {
        AdapterError::invalid_input("sandboxed command stdout pipe was unavailable")
    })?;
    let stderr = child.stderr.take().ok_or_else(|| {
        AdapterError::invalid_input("sandboxed command stderr pipe was unavailable")
    })?;
    let stdout_reader = thread::spawn(move || read_all(stdout));
    let stderr_reader = thread::spawn(move || read_all(stderr));

    let status = loop {
        if cancellation.is_cancelled() || started.elapsed() >= request.timeout {
            terminate_child(&mut child);
            drop(stdout_reader);
            drop(stderr_reader);
            return if cancellation.is_cancelled() {
                Err(AdapterError::Cancelled)
            } else {
                Err(AdapterError::Timeout {
                    message: format!(
                        "command exceeded its {} second timeout",
                        request.timeout.as_secs()
                    ),
                })
            };
        }
        if let Some(status) = child.try_wait()? {
            break status;
        }
        thread::sleep(Duration::from_millis(25));
    };
    let stdout = join_reader(stdout_reader, "stdout")?;
    let stderr = join_reader(stderr_reader, "stderr")?;
    if !status.success() && stderr.starts_with(b"bwrap:") {
        return Err(AdapterError::invalid_input(format!(
            "bubblewrap could not start the command sandbox: {}",
            String::from_utf8_lossy(&stderr).trim()
        )));
    }
    let (stdout, stdout_truncated) = truncate_output(&stdout);
    let (stderr, stderr_truncated) = truncate_output(&stderr);
    Ok(SandboxedCommandOutput {
        exit_code: status.code().unwrap_or(-1),
        stdout,
        stderr,
        stdout_truncated,
        stderr_truncated,
        duration: started.elapsed(),
    })
}

pub fn output_environment(aliases: impl IntoIterator<Item = String>) -> BTreeMap<String, String> {
    aliases
        .into_iter()
        .map(|alias| {
            (
                format!("SUBBAKE_OUTPUT_{}", alias.to_ascii_uppercase()),
                format!("/tmp/outputs/{alias}"),
            )
        })
        .collect()
}

fn path_arg(path: &Path) -> &str {
    path.to_str().unwrap_or("")
}

fn safe_path() -> String {
    std::env::var("PATH").unwrap_or_else(|_| "/usr/local/bin:/usr/bin:/bin".to_owned())
}

fn configure_home_visibility(command: &mut Command, project_root: &Path) -> Vec<(String, String)> {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return Vec::new();
    };
    if !home.is_absolute() || home == Path::new("/") {
        return Vec::new();
    }

    command.args(["--tmpfs", path_arg(&home)]);
    let mut mounts = vec![project_root.to_path_buf()];
    if let Some(path) = std::env::var_os("PATH") {
        mounts.extend(
            std::env::split_paths(&path)
                .filter(|directory| directory.starts_with(&home) && directory.is_dir()),
        );
    }
    let cargo_home = home.join(".cargo");
    for relative in ["registry", "git"] {
        let path = cargo_home.join(relative);
        if path.is_dir() {
            mounts.push(path);
        }
    }
    let rustup_home = home.join(".rustup");
    if rustup_home.is_dir() {
        mounts.push(rustup_home.clone());
    }
    mounts.sort();
    mounts.dedup();

    let mut directories = std::collections::BTreeSet::new();
    for mount in &mounts {
        let mut parent = mount.parent();
        while let Some(path) = parent {
            if path == home {
                break;
            }
            if path.starts_with(&home) {
                directories.insert(path.to_path_buf());
            }
            parent = path.parent();
        }
    }
    let mut directories = directories.into_iter().collect::<Vec<_>>();
    directories.sort_by_key(|path| path.components().count());
    for directory in directories {
        command.args(["--dir", path_arg(&directory)]);
    }
    for mount in mounts {
        command.args(["--ro-bind", path_arg(&mount), path_arg(&mount)]);
    }

    let mut environment = Vec::new();
    if cargo_home.is_dir() {
        environment.push((
            "CARGO_HOME".to_owned(),
            cargo_home.to_string_lossy().into_owned(),
        ));
    }
    if rustup_home.is_dir() {
        environment.push((
            "RUSTUP_HOME".to_owned(),
            rustup_home.to_string_lossy().into_owned(),
        ));
    }
    environment
}

fn path_has_program(program: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|directory| directory.join(program).is_file())
    })
}

fn read_all(mut reader: impl Read) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn join_reader(
    reader: thread::JoinHandle<std::io::Result<Vec<u8>>>,
    stream: &'static str,
) -> AdapterResult<Vec<u8>> {
    reader
        .join()
        .map_err(|_| AdapterError::invalid_input(format!("{stream} reader panicked")))?
        .map_err(AdapterError::from)
}

fn truncate_output(bytes: &[u8]) -> (String, bool) {
    if bytes.len() <= OUTPUT_LIMIT {
        return (String::from_utf8_lossy(bytes).into_owned(), false);
    }
    let half = OUTPUT_LIMIT / 2;
    let mut value = String::from_utf8_lossy(&bytes[..half]).into_owned();
    value.push_str("\n… output truncated …\n");
    value.push_str(&String::from_utf8_lossy(&bytes[bytes.len() - half..]));
    (value, true)
}

fn terminate_child(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .args(["-TERM", &format!("-{}", child.id())])
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_is_bounded_with_a_visible_marker() {
        let bytes = vec![b'x'; OUTPUT_LIMIT + 10];
        let (output, truncated) = truncate_output(&bytes);
        assert!(truncated);
        assert!(output.contains("output truncated"));
    }

    #[test]
    fn output_environment_uses_stable_names() {
        assert_eq!(
            output_environment(["archive".to_owned()])["SUBBAKE_OUTPUT_ARCHIVE"],
            "/tmp/outputs/archive"
        );
    }
}
