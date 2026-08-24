use std::fmt::Write as _;

use crate::{CliError, CliResult};

use super::command_specs;

pub fn run(args: &[String]) -> CliResult<()> {
    let shell = args
        .first()
        .ok_or_else(|| CliError::usage("completion requires bash, zsh, fish, or powershell"))?;
    if args.len() != 1 {
        return Err(CliError::usage(
            "completion accepts exactly one shell: bash, zsh, fish, or powershell",
        ));
    }
    let script = match shell.as_str() {
        "bash" => bash(),
        "zsh" => zsh(),
        "fish" => fish(),
        "powershell" | "pwsh" => powershell(),
        _ => {
            return Err(CliError::usage(
                "completion requires bash, zsh, fish, or powershell",
            ));
        }
    };
    print!("{script}");
    Ok(())
}

fn bash() -> String {
    let mut out = String::from(
        "_sbake() {\n  local command current\n  current=\"${COMP_WORDS[COMP_CWORD]}\"\n  if (( COMP_CWORD == 1 )); then\n    COMPREPLY=( $(compgen -W '",
    );
    out.push_str(&command_names());
    out.push_str("' -- \"$current\") )\n    return\n  fi\n  command=\"${COMP_WORDS[1]}\"\n  case \"$command\" in\n");
    for spec in command_specs() {
        let _ = writeln!(
            out,
            "    {}) COMPREPLY=( $(compgen -W '{} {}' -- \"$current\") ) ;;;;",
            spec.name,
            spec.options.join(" "),
            spec.subcommands.join(" ")
        );
    }
    out.push_str("  esac\n}\ncomplete -F _sbake sbake\n");
    out.replace(";;;;", ";;")
}

fn zsh() -> String {
    let commands = command_specs()
        .iter()
        .map(|spec| format!("'{}:{}'", spec.name, spec.summary.replace('\'', "")))
        .collect::<Vec<_>>()
        .join(" ");
    let mut out = format!(
        "#compdef sbake\n_sbake() {{\n  local -a commands\n  commands=({commands})\n  _arguments '1:command:->command' '*::argument:->args'\n  case $state in\n    command) _describe 'command' commands ;;;;\n    args)\n      case $words[2] in\n"
    );
    for spec in command_specs() {
        let values = spec
            .options
            .iter()
            .chain(spec.subcommands.iter())
            .copied()
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(
            out,
            "        {}) _values 'argument' {} ;;;;",
            spec.name, values
        );
    }
    out.push_str("      esac\n      ;;;;\n  esac\n}\n_sbake\n");
    out.replace(";;;;", ";;")
}

fn fish() -> String {
    let mut out = String::from("complete -c sbake -f\n");
    for spec in command_specs() {
        let _ = writeln!(
            out,
            "complete -c sbake -n '__fish_use_subcommand' -a '{}' -d '{}'",
            spec.name,
            spec.summary.replace('\'', "")
        );
        for option in spec.options {
            if let Some(long) = option.strip_prefix("--") {
                let _ = writeln!(
                    out,
                    "complete -c sbake -n '__fish_seen_subcommand_from {}' -l {}",
                    spec.name, long
                );
            }
        }
    }
    out
}

fn powershell() -> String {
    format!(
        "Register-ArgumentCompleter -Native -CommandName sbake -ScriptBlock {{\n  param($wordToComplete, $commandAst, $cursorPosition)\n  @('{}') | Where-Object {{ $_ -like \"$wordToComplete*\" }} | ForEach-Object {{\n    [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)\n  }}\n}}\n",
        command_names().replace(' ', "','")
    )
}

fn command_names() -> String {
    command_specs()
        .iter()
        .map(|spec| spec.name)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_completions_include_commands_and_options() {
        for script in [bash(), zsh(), fish(), powershell()] {
            assert!(script.contains("translate"));
            assert!(script.contains("project"));
        }
        assert!(bash().contains("--qa-fail-on"));
        assert!(fish().contains("qa-fail-on"));
    }
}
