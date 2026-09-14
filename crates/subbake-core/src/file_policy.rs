//! Shared project-file visibility policy for file tools and command sandboxes.

/// Path components that are never allowed in file operations.
pub const PROTECTED_PATH_PARTS: [&str; 17] = [
    ".git",
    ".hg",
    ".svn",
    ".env",
    ".ssh",
    ".venv",
    "venv",
    ".subbake",
    "subbake.toml",
    ".subbake.toml",
    "__pycache__",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "credentials.json",
    "service-account.json",
];

pub fn is_protected_component(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
    PROTECTED_PATH_PARTS.contains(&normalized.as_str())
        || (normalized.starts_with(".env.") && normalized != ".env.example")
}
