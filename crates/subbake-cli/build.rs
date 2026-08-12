use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir =
        PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_else(|| ".".into()));
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .unwrap_or(&manifest_dir);

    let sha = git_output(workspace_root, &["rev-parse", "--short=8", "HEAD"])
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_owned());
    let dirty = git_output(
        workspace_root,
        &["status", "--porcelain", "--untracked-files=normal"],
    )
    .is_some_and(|value| !value.is_empty());

    println!("cargo:rustc-env=SUBBAKE_GIT_SHA={sha}");
    println!(
        "cargo:rustc-env=SUBBAKE_GIT_DIRTY={}",
        if dirty { "1" } else { "0" }
    );

    watch_git_state(workspace_root);
}

fn git_output(workspace_root: &Path, arguments: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(workspace_root)
        .args(arguments)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn watch_git_state(workspace_root: &Path) {
    for git_path in ["HEAD", "index", "packed-refs"] {
        if let Some(path) = git_output(workspace_root, &["rev-parse", "--git-path", git_path]) {
            println!(
                "cargo:rerun-if-changed={}",
                absolute_path(workspace_root, &path).display()
            );
        }
    }
    if let Some(reference) = git_output(workspace_root, &["symbolic-ref", "-q", "HEAD"])
        && let Some(path) = git_output(
            workspace_root,
            &["rev-parse", "--git-path", reference.as_str()],
        )
    {
        println!(
            "cargo:rerun-if-changed={}",
            absolute_path(workspace_root, &path).display()
        );
    }
    if let Some(files) = git_output(workspace_root, &["ls-files"]) {
        for file in files.lines().filter(|file| !file.is_empty()) {
            println!(
                "cargo:rerun-if-changed={}",
                workspace_root.join(file).display()
            );
        }
    }
}

fn absolute_path(workspace_root: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace_root.join(path)
    }
}
