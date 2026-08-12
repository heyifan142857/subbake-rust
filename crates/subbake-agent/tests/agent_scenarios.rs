use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use subbake_agent::evaluation::{
    AgentEvalCase, AgentEvalResult, AgentEvalSuiteReport, run_scripted_case,
};

static NEXT_WORKSPACE: AtomicU64 = AtomicU64::new(0);

#[test]
fn scripted_agent_scenarios_meet_deterministic_contracts() {
    let cases = load_cases();
    assert!(!cases.is_empty(), "agent scenario corpus must not be empty");

    let mut results = Vec::new();
    for case in cases {
        let workspace = EvalWorkspace::new(&case.id);
        match run_scripted_case(&case, workspace.path()) {
            Ok(result) => results.push(result),
            Err(error) => results.push(AgentEvalResult {
                case_id: case.id,
                description: case.description,
                passed: false,
                failures: vec![error.to_string()],
                trace: Default::default(),
            }),
        }
    }

    let report = AgentEvalSuiteReport::from_results(results);
    let encoded = serde_json::to_string_pretty(&report).expect("serialize agent eval report");
    eprintln!("{encoded}");
    assert_eq!(
        report.failed, 0,
        "{} of {} agent scenario(s) failed",
        report.failed, report.cases
    );
}

fn load_cases() -> Vec<AgentEvalCase> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("scenarios");
    let mut paths = std::fs::read_dir(&directory)
        .expect("read agent scenario directory")
        .map(|entry| entry.expect("read agent scenario entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths
        .iter()
        .map(|path| AgentEvalCase::load(path).expect("load agent scenario"))
        .collect()
}

struct EvalWorkspace {
    path: PathBuf,
}

impl EvalWorkspace {
    fn new(case_id: &str) -> Self {
        let sequence = NEXT_WORKSPACE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "subbake-agent-eval-{}-{}-{sequence}",
            std::process::id(),
            sanitize(case_id)
        ));
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for EvalWorkspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
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
