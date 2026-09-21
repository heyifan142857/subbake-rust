//! Sandboxed execution for the interactive coding agent.

use std::collections::BTreeMap;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use subbake_core::CancellationGuard;
use subbake_core::file_policy::is_protected_component;

use crate::error::{AdapterError, AdapterResult};
use crate::platform::{CapabilitySet, PlatformPaths};
use crate::process::ProcessSupervisor;

const OUTPUT_LIMIT: usize = 64 * 1024;

#[derive(Debug, Clone)]
pub struct SandboxedCommandRequest {
    pub command: String,
    pub project_root: PathBuf,
    pub cwd: PathBuf,
    pub staging_root: PathBuf,
    pub output_aliases: Vec<String>,
    pub network: bool,
    /// Resolved configuration identities excluded in addition to protected names.
    pub protected_paths: Vec<PathBuf>,
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
    if !CapabilitySet::current().supports_command_sandbox() {
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
    mount_system_runtime(&mut command);
    command.args(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]);
    let private_environment = configure_toolchain(&mut command);
    mount_project_view(&mut command, request, cancellation)?;
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
    command.args(["--remount-ro", "/"]);

    // A large project can exceed execve ARG_MAX. Bubblewrap reads NUL-separated
    // arguments from stdin before starting the child, which then sees EOF.
    let mut arguments = tempfile::tempfile()?;
    for arg in command.get_args() {
        cancellation.check().map_err(AdapterError::from)?;
        arguments.write_all(arg.as_encoded_bytes())?;
        arguments.write_all(&[0])?;
    }
    arguments.rewind()?;
    command = Command::new("bwrap");
    command
        .args(["--args", "0", "--", "/bin/bash", "-c", &request.command])
        .stdin(Stdio::from(arguments));
    let started = Instant::now();
    let captured = match ProcessSupervisor::run_bounded_with_timeout(
        &mut command,
        cancellation,
        "sandboxed command",
        request.timeout,
        OUTPUT_LIMIT,
    ) {
        Err(AdapterError::Timeout { .. }) => {
            return Err(AdapterError::Timeout {
                message: format!(
                    "command exceeded its {} second timeout",
                    request.timeout.as_secs()
                ),
            });
        }
        outcome => outcome?,
    };
    let output = captured.output;
    if !output.status.success() && output.stderr.starts_with(b"bwrap:") {
        return Err(AdapterError::invalid_input(format!(
            "bubblewrap could not start the command sandbox: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let stdout_truncated = captured.stdout_truncated;
    let stderr_truncated = captured.stderr_truncated;
    let stdout = render_output(&output.stdout, stdout_truncated);
    let stderr = render_output(&output.stderr, stderr_truncated);
    Ok(SandboxedCommandOutput {
        exit_code: output.status.code().unwrap_or(-1),
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
    let mut paths = vec![
        PathBuf::from("/usr/local/bin"),
        "/usr/bin".into(),
        "/bin".into(),
    ];
    if let Some(home) = PlatformPaths::home_dir() {
        paths.push(home.join(".cargo/bin"));
    }
    std::env::join_paths(paths)
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "/usr/local/bin:/usr/bin:/bin".to_owned())
}

fn mount_system_runtime(command: &mut Command) {
    // Do not expose /run, other users' homes, host sockets or arbitrary host data.
    for path in [
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/etc/ld.so.cache",
        "/etc/alternatives",
        "/etc/ssl/certs",
        "/etc/pki/ca-trust/extracted",
        "/etc/resolv.conf",
        "/etc/hosts",
        "/etc/nsswitch.conf",
    ] {
        if Path::new(path).exists() {
            command.args(["--ro-bind", path, path]);
        }
    }
}

fn configure_toolchain(command: &mut Command) -> Vec<(String, String)> {
    let Some(home) = PlatformPaths::home_dir() else {
        return Vec::new();
    };
    let cargo_home = home.join(".cargo");
    let rustup_home = home.join(".rustup");
    for path in [
        cargo_home.join("bin"),
        cargo_home.join("registry"),
        rustup_home.clone(),
    ] {
        if path.is_dir() {
            command.arg("--ro-bind").arg(&path).arg(&path);
        }
    }
    vec![
        (
            "CARGO_HOME".into(),
            cargo_home.to_string_lossy().into_owned(),
        ),
        (
            "RUSTUP_HOME".into(),
            rustup_home.to_string_lossy().into_owned(),
        ),
    ]
}

fn mount_project_view(
    command: &mut Command,
    request: &SandboxedCommandRequest,
    cancellation: &CancellationGuard,
) -> AdapterResult<()> {
    let identity = PlatformPaths::identify_existing(&request.project_root)?;
    let root = identity.resolved();
    let mut pending = vec![(root.to_path_buf(), request.project_root.clone())];
    while let Some((source, destination)) = pending.pop() {
        cancellation.check().map_err(AdapterError::from)?;
        command.arg("--dir").arg(&destination);
        let mut entries = std::fs::read_dir(&source)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            cancellation.check().map_err(AdapterError::from)?;
            let name = entry.file_name();
            if name.to_str().is_some_and(is_protected_component) {
                continue;
            }
            let source = entry.path();
            if request
                .protected_paths
                .iter()
                .any(|path| source.starts_with(path))
            {
                continue;
            }
            let file_type = entry.file_type()?;
            let destination = destination.join(name);
            if file_type.is_dir() {
                if !is_disposable_cache(&source) {
                    pending.push((source, destination));
                }
            } else if file_type.is_file() {
                command.arg("--ro-bind").arg(source).arg(destination);
            }
            // Never expose symlinks, FIFOs, devices or Unix sockets. In particular,
            // aliases must not make hidden credentials visible to recursive tools.
        }
    }
    Ok(())
}

/// Cargo and other tools mark disposable caches explicitly. Omitting these
/// avoids one bind mount per build artifact; command builds use private /tmp.
fn is_disposable_cache(directory: &Path) -> bool {
    const SIGNATURE: &[u8] = b"Signature: 8a477f597d28d172789f06886806bc55";
    let path = directory.join("CACHEDIR.TAG");
    if !std::fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.is_file()) {
        return false;
    }
    let mut header = [0_u8; SIGNATURE.len()];
    std::fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut header))
        .is_ok()
        && header == SIGNATURE
}

fn path_has_program(program: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|directory| directory.join(program).is_file())
    })
}

fn render_output(bytes: &[u8], truncated: bool) -> String {
    if !truncated {
        return String::from_utf8_lossy(bytes).into_owned();
    }
    let half = bytes.len() / 2;
    let mut value = String::from_utf8_lossy(&bytes[..half]).into_owned();
    value.push_str("\n… output truncated …\n");
    value.push_str(&String::from_utf8_lossy(&bytes[half..]));
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(root: &Path, command: &str) -> SandboxedCommandRequest {
        SandboxedCommandRequest {
            command: command.into(),
            project_root: root.into(),
            cwd: root.into(),
            staging_root: root.join(".subbake/staging"),
            output_aliases: Vec::new(),
            network: false,
            protected_paths: Vec::new(),
            timeout: Duration::from_secs(5),
        }
    }

    #[test]
    fn project_mounts_exclude_nested_protected_files() {
        let root = tempfile::tempdir().expect("root");
        std::fs::create_dir_all(root.path().join("nested/.git")).expect("nested");
        for name in [
            "visible.txt",
            "nested/.env",
            "nested/.git/config",
            "nested/.env.example",
        ] {
            std::fs::write(root.path().join(name), "fixture").expect("fixture");
        }
        let mut command = Command::new("bwrap");
        mount_project_view(
            &mut command,
            &request(root.path(), "pwd"),
            &CancellationGuard::never(),
        )
        .expect("mount plan");
        let args = command.get_args().map(Path::new).collect::<Vec<_>>();
        assert!(args.iter().any(|arg| arg.ends_with("visible.txt")));
        assert!(args.iter().any(|arg| arg.ends_with(".env.example")));
        assert!(!args.iter().any(|arg| {
            arg.ends_with(".env")
                || arg
                    .components()
                    .any(|component| component.as_os_str() == ".git")
        }));
    }

    #[test]
    fn tagged_build_caches_are_omitted_but_normal_target_directories_remain_visible() {
        let root = tempfile::tempdir().expect("root");
        std::fs::create_dir_all(root.path().join("build-cache")).expect("cache");
        std::fs::create_dir_all(root.path().join("target")).expect("ordinary target");
        std::fs::write(
            root.path().join("build-cache/CACHEDIR.TAG"),
            "Signature: 8a477f597d28d172789f06886806bc55\n",
        )
        .expect("cache marker");
        std::fs::write(root.path().join("build-cache/large-artifact"), "cache").expect("artifact");
        std::fs::write(root.path().join("target/subtitle.srt"), "source").expect("subtitle");
        let mut command = Command::new("bwrap");
        mount_project_view(
            &mut command,
            &request(root.path(), "pwd"),
            &CancellationGuard::never(),
        )
        .expect("mount plan");
        let args = command.get_args().map(Path::new).collect::<Vec<_>>();
        assert!(!args.iter().any(|arg| {
            arg.components()
                .any(|component| component.as_os_str() == "build-cache")
        }));
        assert!(args.iter().any(|arg| arg.ends_with("target/subtitle.srt")));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn sandbox_recursive_reads_cannot_expose_credentials_aliases_or_sockets() {
        use std::os::unix::{fs::symlink, net::UnixListener};
        if !path_has_program("bwrap") {
            return;
        }
        let root = tempfile::tempdir().expect("root");
        std::fs::create_dir_all(root.path().join("nested")).expect("nested");
        std::fs::write(root.path().join("nested/.env"), "AUDIT_SECRET").expect("credential");
        std::fs::write(root.path().join("visible.txt"), "AUDIT_VISIBLE").expect("visible");
        symlink(".env", root.path().join("nested/alias")).expect("alias");
        let _socket = UnixListener::bind(root.path().join("nested/socket")).expect("socket");
        let output = match run_sandboxed_command(
            &request(
                root.path(),
                "grep -R -a AUDIT .; test ! -e /run; test ! -e nested/alias; test ! -e nested/socket",
            ),
            &CancellationGuard::never(),
        ) {
            Ok(output) => output,
            Err(error) if error.to_string().contains("Operation not permitted") => return,
            Err(error) => panic!("sandbox: {error}"),
        };
        assert_eq!(output.exit_code, 0, "{}", output.stderr);
        assert!(output.stdout.contains("AUDIT_VISIBLE"));
        assert!(!output.stdout.contains("AUDIT_SECRET"));
    }

    #[test]
    fn output_is_bounded_with_a_visible_marker() {
        let bytes = vec![b'x'; OUTPUT_LIMIT];
        let output = render_output(&bytes, true);
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
