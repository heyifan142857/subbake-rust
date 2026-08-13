use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use subbake_adapters::{
    ApiFormat, BackendConfig, ConfigurationResolver, ResolveRequest, build_backend,
    default_api_key_env, discover_config_path,
};
use subbake_agent::evaluation::{
    AgentEvalCase, AgentEvalResult, AgentEvalSuiteReport, run_live_case,
};

static NEXT_WORKSPACE: AtomicU64 = AtomicU64::new(0);

#[test]
fn live_agent_scenario_fixtures_are_valid_without_calling_a_provider() {
    let cases = load_cases();
    assert!(
        cases.len() >= 5,
        "expected a meaningful live red-team corpus"
    );
}

#[test]
#[ignore = "requires an explicitly configured live model and may incur provider cost"]
fn live_agent_scenarios_meet_quality_gate() {
    let local_env = load_local_env().expect("load project .env");
    let repetitions = setting("SUBBAKE_EVAL_REPETITIONS", &local_env)
        .map(|value| value.parse::<usize>().expect("valid repetition count"))
        .unwrap_or(3);
    assert!((1..=20).contains(&repetitions));

    let mut backend = live_backend(&local_env);
    let cases = load_cases();
    assert!(
        !cases.is_empty(),
        "live agent scenario corpus must not be empty"
    );

    let mut results = Vec::new();
    for case in cases {
        for repetition in 1..=repetitions {
            let workspace = EvalWorkspace::new(&case.id, repetition);
            match run_live_case(&case, workspace.path(), backend.as_mut()) {
                Ok(mut result) => {
                    result.case_id = format!("{}#{repetition}", result.case_id);
                    results.push(result);
                }
                Err(error) => results.push(AgentEvalResult {
                    case_id: format!("{}#{repetition}", case.id),
                    description: case.description.clone(),
                    passed: false,
                    failures: vec![error.to_string()],
                    trace: Default::default(),
                }),
            }
        }
    }

    let report = AgentEvalSuiteReport::from_results(results);
    let summary = serde_json::json!({
        "provider": backend.provider_name(),
        "model": backend.model_name(),
        "repetitions": repetitions,
        "cases": report.cases,
        "passed": report.passed,
        "failed": report.failed,
        "pass_rate": report.pass_rate,
        "tool_calls": report.tool_calls,
        "successful_tool_calls": report.successful_tool_calls,
        "model_steps": report.model_steps,
    });
    eprintln!(
        "{}",
        serde_json::to_string_pretty(&summary).expect("serialize live eval summary")
    );
    for result in &report.results {
        if !result.passed {
            eprintln!("{}: {}", result.case_id, result.failures.join("; "));
            eprintln!(
                "{}",
                serde_json::to_string_pretty(&result.trace)
                    .expect("serialize failed live eval trace")
            );
        }
    }

    assert!(
        report.pass_rate >= 0.95,
        "live agent pass rate {:.1}% is below 95%",
        report.pass_rate * 100.0
    );
}

fn live_backend(local_env: &BTreeMap<String, String>) -> Box<dyn subbake_core::ports::LlmBackend> {
    if setting("SUBBAKE_EVAL_PROVIDER", local_env).is_some() {
        return direct_live_backend(local_env);
    }
    let config_path = setting("SUBBAKE_EVAL_CONFIG", local_env)
        .map(PathBuf::from)
        .or_else(discover_config_path)
        .expect("no SubBake configuration found; set SUBBAKE_EVAL_CONFIG");
    let profile = setting("SUBBAKE_EVAL_PROFILE", local_env);
    let resolved = ConfigurationResolver
        .resolve(ResolveRequest {
            pinned_path: Some(config_path.clone()),
            profile,
            ..ResolveRequest::default()
        })
        .unwrap_or_else(|error| {
            panic!(
                "resolve live eval configuration `{}`: {error}",
                config_path.display()
            )
        });
    let mut config = resolved.settings.backend_config();
    apply_local_api_key(&mut config, local_env);
    build_backend(&config).expect("build configured live eval backend")
}

fn direct_live_backend(
    local_env: &BTreeMap<String, String>,
) -> Box<dyn subbake_core::ports::LlmBackend> {
    let provider = required_setting("SUBBAKE_EVAL_PROVIDER", local_env);
    let model = required_setting("SUBBAKE_EVAL_MODEL", local_env);
    let api_format = ApiFormat::parse(&required_setting("SUBBAKE_EVAL_API_FORMAT", local_env))
        .expect("valid SUBBAKE_EVAL_API_FORMAT");
    let mut config = BackendConfig::new(provider, model);
    config.api_format = Some(api_format);
    config.base_url = setting("SUBBAKE_EVAL_BASE_URL", local_env);
    config.endpoint_url = setting("SUBBAKE_EVAL_ENDPOINT_URL", local_env);
    config.api_key_env = setting("SUBBAKE_EVAL_API_KEY_ENV", local_env);
    apply_local_api_key(&mut config, local_env);
    build_backend(&config).expect("build live eval backend")
}

fn apply_local_api_key(config: &mut BackendConfig, local_env: &BTreeMap<String, String>) {
    if config.api_key.is_some() {
        return;
    }
    let name = config
        .api_key_env
        .as_deref()
        .or_else(|| default_api_key_env(&config.id));
    if let Some(value) = name.and_then(|name| local_env.get(name)) {
        config.api_key = Some(value.clone());
    }
}

fn required_setting(name: &str, local_env: &BTreeMap<String, String>) -> String {
    setting(name, local_env).unwrap_or_else(|| panic!("{name} must be set for live agent evals"))
}

fn setting(name: &str, local_env: &BTreeMap<String, String>) -> Option<String> {
    std::env::var(name)
        .ok()
        .or_else(|| local_env.get(name).cloned())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn load_local_env() -> Result<BTreeMap<String, String>, String> {
    let path = workspace_root().join(".env");
    match std::fs::read_to_string(&path) {
        Ok(content) => parse_env_file(&content),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(error) => Err(format!("read `{}`: {error}", path.display())),
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("agent crate is inside workspace/crates")
        .to_path_buf()
}

fn parse_env_file(content: &str) -> Result<BTreeMap<String, String>, String> {
    let mut values = BTreeMap::new();
    for (index, raw_line) in content.lines().enumerate() {
        let mut line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("export ") {
            line = rest.trim_start();
        }
        let (name, encoded) = line
            .split_once('=')
            .ok_or_else(|| format!(".env line {} is missing `=`", index + 1))?;
        let name = name.trim();
        if !valid_env_name(name) {
            return Err(format!(".env line {} has an invalid name", index + 1));
        }
        let encoded = encoded.trim();
        let value = if encoded.len() >= 2
            && ((encoded.starts_with('"') && encoded.ends_with('"'))
                || (encoded.starts_with('\'') && encoded.ends_with('\'')))
        {
            encoded[1..encoded.len() - 1].to_owned()
        } else {
            encoded.to_owned()
        };
        values.insert(name.to_owned(), value);
    }
    Ok(values)
}

fn valid_env_name(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|value| value == '_' || value.is_ascii_alphabetic())
        && characters.all(|value| value == '_' || value.is_ascii_alphanumeric())
}

fn load_cases() -> Vec<AgentEvalCase> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("live_scenarios");
    let mut paths = std::fs::read_dir(&directory)
        .expect("read live agent scenario directory")
        .map(|entry| entry.expect("read live agent scenario entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths
        .iter()
        .map(|path| AgentEvalCase::load(path).expect("load live agent scenario"))
        .collect()
}

struct EvalWorkspace(PathBuf);

impl EvalWorkspace {
    fn new(case_id: &str, repetition: usize) -> Self {
        let sequence = NEXT_WORKSPACE.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!(
            "subbake-live-agent-eval-{}-{}-{repetition}-{sequence}",
            std::process::id(),
            sanitize(case_id)
        )))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for EvalWorkspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' {
                character
            } else {
                '-'
            }
        })
        .collect()
}

#[test]
fn local_env_parser_supports_exports_quotes_and_equals_without_exposing_values() {
    let values = parse_env_file(
        "# local only\nexport SUBBAKE_EVAL_PROFILE=example\nPROVIDER_API_KEY='secret=part'\n",
    )
    .expect("parse .env");
    assert_eq!(
        values.get("SUBBAKE_EVAL_PROFILE").map(String::as_str),
        Some("example")
    );
    assert_eq!(
        values.get("PROVIDER_API_KEY").map(String::as_str),
        Some("secret=part")
    );
    assert!(parse_env_file("BAD-NAME=value\n").is_err());
}
