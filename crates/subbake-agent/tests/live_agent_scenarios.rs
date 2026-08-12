use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use subbake_adapters::{ApiFormat, BackendConfig, build_backend};
use subbake_agent::evaluation::{
    AgentEvalCase, AgentEvalResult, AgentEvalSuiteReport, run_live_case,
};

static NEXT_WORKSPACE: AtomicU64 = AtomicU64::new(0);

#[test]
#[ignore = "requires an explicitly configured live model and may incur provider cost"]
fn live_agent_scenarios_meet_quality_gate() {
    let repetitions = optional_env("SUBBAKE_EVAL_REPETITIONS")
        .map(|value| value.parse::<usize>().expect("valid repetition count"))
        .unwrap_or(3);
    assert!((1..=20).contains(&repetitions));

    let mut backend = live_backend();
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
        }
    }

    assert!(
        report.pass_rate >= 0.95,
        "live agent pass rate {:.1}% is below 95%",
        report.pass_rate * 100.0
    );
}

fn live_backend() -> Box<dyn subbake_core::ports::LlmBackend> {
    let provider = required_env("SUBBAKE_EVAL_PROVIDER");
    let model = required_env("SUBBAKE_EVAL_MODEL");
    let api_format = ApiFormat::parse(&required_env("SUBBAKE_EVAL_API_FORMAT"))
        .expect("valid SUBBAKE_EVAL_API_FORMAT");
    let mut config = BackendConfig::new(provider, model);
    config.api_format = Some(api_format);
    config.base_url = optional_env("SUBBAKE_EVAL_BASE_URL");
    config.endpoint_url = optional_env("SUBBAKE_EVAL_ENDPOINT_URL");
    config.api_key_env = optional_env("SUBBAKE_EVAL_API_KEY_ENV");
    build_backend(&config).expect("build live eval backend")
}

fn required_env(name: &str) -> String {
    optional_env(name).unwrap_or_else(|| panic!("{name} must be set for live agent evals"))
}

fn optional_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
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
