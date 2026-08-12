const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");
const GIT_SHA: &str = env!("SUBBAKE_GIT_SHA");
const GIT_DIRTY: bool = matches!(env!("SUBBAKE_GIT_DIRTY").as_bytes(), [b'1']);

pub(crate) fn build_identity() -> String {
    format_build_identity(PACKAGE_VERSION, GIT_SHA, GIT_DIRTY)
}

fn format_build_identity(version: &str, git_sha: &str, dirty: bool) -> String {
    match (git_sha, dirty) {
        ("unknown", _) => format!("{version} (git unknown)"),
        (_, true) => format!("{version} ({git_sha}, dirty)"),
        (_, false) => format!("{version} ({git_sha})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_clean_and_dirty_build_identities() {
        assert_eq!(
            format_build_identity("0.2.0-alpha.1", "1234abcd", false),
            "0.2.0-alpha.1 (1234abcd)"
        );
        assert_eq!(
            format_build_identity("0.2.0-alpha.1", "1234abcd", true),
            "0.2.0-alpha.1 (1234abcd, dirty)"
        );
        assert_eq!(
            format_build_identity("0.2.0-alpha.1", "unknown", false),
            "0.2.0-alpha.1 (git unknown)"
        );
    }
}
