use serde_json::Value;

use crate::guard::is_protected_component;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CommandApproval {
    AutoRun,
    AskUser(String),
    Deny(String),
}

pub(crate) fn classify(arguments: &Value) -> CommandApproval {
    let command = arguments
        .get("command")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    let lower = command.to_ascii_lowercase();
    if lower.contains("authorization:")
        || lower.contains("api_key=")
        || lower.contains("api-key=")
        || lower.contains("token=")
    {
        return CommandApproval::Deny(
            "inline credentials are not allowed in persisted command calls".to_owned(),
        );
    }
    if command_tokens(command).any(|token| {
        let token = token.rsplit('/').next().unwrap_or(token);
        matches!(
            token,
            "sudo" | "su" | "mount" | "umount" | "nsenter" | "unshare" | "bwrap"
        )
    }) {
        return CommandApproval::Deny(
            "commands that alter or nest privilege and sandbox boundaries are not allowed"
                .to_owned(),
        );
    }
    if command_tokens(command).any(token_references_protected_path) {
        return CommandApproval::Deny(
            "commands cannot reference protected configuration or credential paths".to_owned(),
        );
    }
    if arguments
        .get("outputs")
        .and_then(Value::as_object)
        .is_some_and(|outputs| !outputs.is_empty())
    {
        return CommandApproval::AskUser("command will commit declared output files".to_owned());
    }
    if arguments
        .get("network")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return CommandApproval::AskUser("command requests network access".to_owned());
    }
    if is_known_sandboxed_command(command) {
        CommandApproval::AutoRun
    } else {
        CommandApproval::AskUser("command is not in the strict auto-run set".to_owned())
    }
}

fn token_references_protected_path(token: &str) -> bool {
    let token = token.trim_matches(['\'', '"']);
    std::path::Path::new(token).components().any(|component| {
        component
            .as_os_str()
            .to_str()
            .is_some_and(is_protected_component)
    })
}

fn command_tokens(command: &str) -> impl Iterator<Item = &str> {
    command.split(|character: char| {
        character.is_whitespace() || matches!(character, ';' | '|' | '&' | '(' | ')')
    })
}

fn is_known_sandboxed_command(command: &str) -> bool {
    if command.is_empty()
        || command.contains([';', '|', '&', '>', '<', '\n', '\r', '`'])
        || command.contains("$(")
    {
        return false;
    }
    let parts = command.split_whitespace().collect::<Vec<_>>();
    let Some(program) = parts.first().copied() else {
        return false;
    };
    let program = program.rsplit('/').next().unwrap_or(program);
    match program {
        "pwd" | "ls" | "find" | "rg" | "grep" | "sed" | "head" | "tail" | "wc" | "stat"
        | "file" | "cat" | "which" => true,
        "git" => parts.get(1).is_some_and(|subcommand| {
            matches!(
                *subcommand,
                "status" | "diff" | "log" | "show" | "rev-parse"
            ) || (*subcommand == "branch" && parts.contains(&"--show-current"))
        }),
        "cargo" => parts.get(1).is_some_and(|subcommand| {
            matches!(*subcommand, "check" | "test" | "clippy" | "metadata")
                || (*subcommand == "fmt" && parts.contains(&"--check"))
        }),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_runs_strict_queries_and_verification() {
        assert_eq!(
            classify(&serde_json::json!({"command":"rg ToolSpec crates"})),
            CommandApproval::AutoRun
        );
        assert_eq!(
            classify(&serde_json::json!({"command":"cargo test --workspace"})),
            CommandApproval::AutoRun
        );
    }

    #[test]
    fn outputs_network_and_complex_shell_require_approval() {
        for arguments in [
            serde_json::json!({"command":"ffmpeg -i in.mkv $SUBBAKE_OUTPUT_OUT","outputs":{"out":"out.mkv"}}),
            serde_json::json!({"command":"curl example.com","network":true}),
            serde_json::json!({"command":"rg foo | head"}),
        ] {
            assert!(matches!(classify(&arguments), CommandApproval::AskUser(_)));
        }
    }

    #[test]
    fn rejects_privilege_tools_and_inline_secrets() {
        assert!(matches!(
            classify(&serde_json::json!({"command":"sudo id"})),
            CommandApproval::Deny(_)
        ));
        assert!(matches!(
            classify(&serde_json::json!({"command":"curl -H 'Authorization: Bearer x' x"})),
            CommandApproval::Deny(_)
        ));
        for command in [
            "cat .env",
            "sed -n 1p '.EnV.Local'",
            "cat ~/.ssh/id_rsa",
            "rg API_KEY keys/ID_ED25519",
        ] {
            assert!(
                matches!(
                    classify(&serde_json::json!({"command": command})),
                    CommandApproval::Deny(_)
                ),
                "{command}"
            );
        }
    }
}
