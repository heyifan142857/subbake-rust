//! Data-driven evaluation support for the interactive agent.
//!
//! The evaluator deliberately separates deterministic runtime assertions from
//! model quality. Scripted cases exercise the real decision loop and tools at
//! zero provider cost; the recorder and assertion layer can also wrap a live
//! backend in an ignored or scheduled test.

use std::collections::VecDeque;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use subbake_core::AgentToolOutcome;
use subbake_core::entities::Usage;
use subbake_core::error::LlmCallError;
use subbake_core::ports::{
    GenerationInput, GenerationRequest, GenerationResponse, LlmBackend, NativeToolSupport,
};
use thiserror::Error;

use crate::{AgentEngine, AgentError, CommandDecision, EngineObserver, PlanDecision};

pub const AGENT_EVAL_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum AgentEvalError {
    #[error("read agent eval fixture `{path}`: {source}")]
    ReadFixture {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("decode agent eval fixture `{path}`: {source}")]
    DecodeFixture {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("agent eval case `{case_id}` uses unsupported version {actual}; expected {expected}")]
    UnsupportedVersion {
        case_id: String,
        actual: u32,
        expected: u32,
    },
    #[error("invalid project-local eval path `{path}`")]
    InvalidPath { path: PathBuf },
    #[error("agent eval I/O failed for `{path}`: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("agent eval trace lock is poisoned")]
    TracePoisoned,
    #[error("agent eval step {step} in `{case_id}` failed: {source}")]
    AgentStep {
        case_id: String,
        step: usize,
        source: AgentError,
    },
    #[error("agent eval step {step} in `{case_id}` left {remaining} scripted decision(s) unused")]
    UnusedDecisions {
        case_id: String,
        step: usize,
        remaining: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentEvalCase {
    pub version: u32,
    pub id: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub initial_files: Vec<AgentEvalInputFile>,
    pub steps: Vec<AgentEvalStep>,
    pub expectations: AgentEvalExpectations,
}

impl AgentEvalCase {
    pub fn load(path: &Path) -> Result<Self, AgentEvalError> {
        let encoded =
            std::fs::read_to_string(path).map_err(|source| AgentEvalError::ReadFixture {
                path: path.to_path_buf(),
                source,
            })?;
        let case: Self =
            serde_json::from_str(&encoded).map_err(|source| AgentEvalError::DecodeFixture {
                path: path.to_path_buf(),
                source,
            })?;
        case.validate()?;
        Ok(case)
    }

    pub fn validate(&self) -> Result<(), AgentEvalError> {
        if self.version != AGENT_EVAL_FORMAT_VERSION {
            return Err(AgentEvalError::UnsupportedVersion {
                case_id: self.id.clone(),
                actual: self.version,
                expected: AGENT_EVAL_FORMAT_VERSION,
            });
        }
        for file in &self.initial_files {
            validate_relative_path(Path::new(&file.path))?;
        }
        for file in &self.expectations.files {
            validate_relative_path(Path::new(&file.path))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentEvalInputFile {
    pub path: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum AgentEvalStep {
    User {
        input: String,
        decisions: Vec<JsonValue>,
    },
    SetPlanMode {
        enabled: bool,
    },
    PlanDecision {
        decision: AgentEvalDecision,
    },
    CommandDecision {
        decision: AgentEvalDecision,
        #[serde(default)]
        decisions: Vec<JsonValue>,
    },
    Undo,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentEvalDecision {
    Approve,
    Reject,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentEvalExpectations {
    pub final_response_contains: Vec<String>,
    pub required_calls: Vec<ExpectedToolCall>,
    pub required_any_calls: Vec<Vec<ExpectedToolCall>>,
    pub forbidden_tools: Vec<String>,
    pub tool_sequence: Vec<String>,
    pub required_events: Vec<String>,
    pub forbidden_events: Vec<String>,
    pub files: Vec<ExpectedFile>,
    pub max_tool_calls: Option<usize>,
    pub max_model_steps: Option<usize>,
    pub step_limit_reached: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Map<String, JsonValue>,
    #[serde(default)]
    pub status: Option<AgentEvalToolStatus>,
    #[serde(default = "default_min_count")]
    pub min_count: usize,
    #[serde(default)]
    pub max_count: Option<usize>,
}

const fn default_min_count() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedFile {
    pub path: String,
    #[serde(default = "default_true")]
    pub exists: bool,
    #[serde(default)]
    pub equals: Option<String>,
    #[serde(default)]
    pub contains: Vec<String>,
}

const fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentEvalToolStatus {
    Started,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvalToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: JsonValue,
    pub status: AgentEvalToolStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AgentEvalTrace {
    pub model_steps: usize,
    pub tool_calls: Vec<AgentEvalToolCall>,
    pub responses: Vec<String>,
    pub observer_errors: Vec<String>,
    pub event_kinds: Vec<String>,
    pub step_limit_reached: bool,
    pub scripted_backend_exhausted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvalResult {
    pub case_id: String,
    pub description: String,
    pub passed: bool,
    pub failures: Vec<String>,
    pub trace: AgentEvalTrace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvalSuiteReport {
    pub cases: usize,
    pub passed: usize,
    pub failed: usize,
    pub pass_rate: f64,
    pub tool_calls: usize,
    pub successful_tool_calls: usize,
    pub tool_success_rate: f64,
    pub model_steps: usize,
    pub results: Vec<AgentEvalResult>,
}

impl AgentEvalSuiteReport {
    pub fn from_results(results: Vec<AgentEvalResult>) -> Self {
        let cases = results.len();
        let passed = results.iter().filter(|result| result.passed).count();
        let tool_calls = results
            .iter()
            .map(|result| result.trace.tool_calls.len())
            .sum();
        let successful_tool_calls = results
            .iter()
            .flat_map(|result| &result.trace.tool_calls)
            .filter(|call| call.status == AgentEvalToolStatus::Succeeded)
            .count();
        let model_steps = results.iter().map(|result| result.trace.model_steps).sum();
        Self {
            cases,
            passed,
            failed: cases.saturating_sub(passed),
            pass_rate: ratio(passed, cases),
            tool_calls,
            successful_tool_calls,
            tool_success_rate: ratio(successful_tool_calls, tool_calls),
            model_steps,
            results,
        }
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[derive(Clone, Default)]
pub struct AgentEvalRecorder {
    trace: Arc<Mutex<AgentEvalTrace>>,
}

impl AgentEvalRecorder {
    pub fn snapshot(&self) -> Result<AgentEvalTrace, AgentEvalError> {
        self.trace
            .lock()
            .map(|trace| trace.clone())
            .map_err(|_| AgentEvalError::TracePoisoned)
    }

    fn update_call(
        &mut self,
        call_id: &str,
        status: AgentEvalToolStatus,
        outcome: Option<JsonValue>,
        error: Option<String>,
    ) {
        let Ok(mut trace) = self.trace.lock() else {
            return;
        };
        if let Some(call) = trace
            .tool_calls
            .iter_mut()
            .rev()
            .find(|call| call.call_id == call_id)
        {
            call.status = status;
            call.outcome = outcome;
            call.error = error;
        }
    }

    fn record_response_if_new(&self, response: &str) {
        let Ok(mut trace) = self.trace.lock() else {
            return;
        };
        if trace.responses.last().is_none_or(|last| last != response) {
            trace.responses.push(response.to_owned());
        }
    }
}

impl EngineObserver for AgentEvalRecorder {
    fn on_tool_call(&mut self, call_id: &str, name: &str, arguments: &JsonValue) {
        let Ok(mut trace) = self.trace.lock() else {
            return;
        };
        trace.tool_calls.push(AgentEvalToolCall {
            call_id: call_id.to_owned(),
            name: name.to_owned(),
            arguments: arguments.clone(),
            status: AgentEvalToolStatus::Started,
            outcome: None,
            error: None,
        });
    }

    fn on_tool_success(
        &mut self,
        call_id: &str,
        _name: &str,
        _arguments: &JsonValue,
        outcome: &AgentToolOutcome,
    ) {
        self.update_call(
            call_id,
            AgentEvalToolStatus::Succeeded,
            compact_outcome(outcome),
            None,
        );
    }

    fn on_tool_failure(&mut self, call_id: &str, _name: &str, _arguments: &JsonValue, error: &str) {
        self.update_call(
            call_id,
            AgentEvalToolStatus::Failed,
            None,
            Some(error.to_owned()),
        );
    }

    fn on_tool_cancelled(&mut self, call_id: &str, _name: &str) {
        self.update_call(call_id, AgentEvalToolStatus::Cancelled, None, None);
    }

    fn on_error(&mut self, error: &str) {
        let Ok(mut trace) = self.trace.lock() else {
            return;
        };
        trace.observer_errors.push(error.to_owned());
    }

    fn on_response(&mut self, text: &str) {
        let Ok(mut trace) = self.trace.lock() else {
            return;
        };
        trace.responses.push(text.to_owned());
    }

    fn on_step_limit(&mut self) {
        let Ok(mut trace) = self.trace.lock() else {
            return;
        };
        trace.step_limit_reached = true;
    }
}

fn compact_outcome(outcome: &AgentToolOutcome) -> Option<JsonValue> {
    let encoded = serde_json::to_value(outcome).ok()?;
    let operation = encoded.get("operation")?.clone();
    let status = encoded.pointer("/facts/status").cloned();
    Some(serde_json::json!({
        "operation": operation,
        "status": status
    }))
}

#[derive(Default)]
pub struct ScriptedEvalBackend {
    decisions: VecDeque<JsonValue>,
    model_steps: usize,
    exhausted: bool,
}

impl ScriptedEvalBackend {
    pub fn extend(&mut self, decisions: impl IntoIterator<Item = JsonValue>) {
        self.decisions.extend(decisions);
    }

    pub fn remaining(&self) -> usize {
        self.decisions.len()
    }

    pub fn model_steps(&self) -> usize {
        self.model_steps
    }

    pub fn exhausted(&self) -> bool {
        self.exhausted
    }
}

impl LlmBackend for ScriptedEvalBackend {
    fn provider_name(&self) -> &str {
        "agent-eval"
    }

    fn model_name(&self) -> &str {
        "scripted"
    }

    fn execute(
        &mut self,
        request: GenerationRequest,
        cancellation: &subbake_core::CancellationGuard,
    ) -> Result<GenerationResponse, LlmCallError> {
        cancellation.check().map_err(LlmCallError::from)?;
        if !matches!(request.input, GenerationInput::Messages(_)) {
            return Err(LlmCallError::ContinuationMismatch(
                "scripted eval backend only supports JSON decision turns".to_owned(),
            ));
        }
        self.model_steps += 1;
        let decision = self.decisions.pop_front().unwrap_or_else(|| {
            self.exhausted = true;
            serde_json::json!({
                "action": "respond",
                "text": "scripted eval backend exhausted"
            })
        });
        Ok(GenerationResponse::json(decision, Usage::default()))
    }
}

pub fn run_scripted_case(
    case: &AgentEvalCase,
    project_root: &Path,
) -> Result<AgentEvalResult, AgentEvalError> {
    case.validate()?;
    prepare_workspace(case, project_root)?;

    let recorder = AgentEvalRecorder::default();
    let observer = recorder.clone();
    let mut engine = AgentEngine::new(project_root.to_path_buf()).with_observer(Box::new(observer));
    engine
        .start_session()
        .map_err(|source| AgentEvalError::AgentStep {
            case_id: case.id.clone(),
            step: 0,
            source,
        })?;
    let mut backend = ScriptedEvalBackend::default();

    for (index, step) in case.steps.iter().enumerate() {
        let step_number = index + 1;
        let response = match step {
            AgentEvalStep::User { input, decisions } => {
                backend.extend(decisions.iter().cloned());
                engine.run_line(input, &mut backend)
            }
            AgentEvalStep::SetPlanMode { enabled } => engine.set_plan_mode(*enabled),
            AgentEvalStep::PlanDecision { decision } => {
                engine.handle_plan_decision(match decision {
                    AgentEvalDecision::Approve => PlanDecision::Approve,
                    AgentEvalDecision::Reject => PlanDecision::Reject,
                })
            }
            AgentEvalStep::CommandDecision {
                decision,
                decisions,
            } => {
                backend.extend(decisions.iter().cloned());
                engine.handle_command_decision(
                    match decision {
                        AgentEvalDecision::Approve => CommandDecision::Approve,
                        AgentEvalDecision::Reject => CommandDecision::Reject,
                    },
                    &mut backend,
                )
            }
            AgentEvalStep::Undo => engine.undo_last(),
        }
        .map_err(|source| AgentEvalError::AgentStep {
            case_id: case.id.clone(),
            step: step_number,
            source,
        })?;
        recorder.record_response_if_new(&response);

        if backend.remaining() != 0 {
            return Err(AgentEvalError::UnusedDecisions {
                case_id: case.id.clone(),
                step: step_number,
                remaining: backend.remaining(),
            });
        }
    }

    let mut trace = recorder.snapshot()?;
    trace.model_steps = backend.model_steps();
    trace.scripted_backend_exhausted = backend.exhausted();
    trace.event_kinds = engine
        .session_events()
        .into_iter()
        .map(|event| event.kind)
        .collect();
    Ok(evaluate_trace(case, project_root, trace))
}

/// Run an eval case with a real decision backend.
///
/// Scripted decisions embedded in user and command steps are intentionally
/// ignored. This keeps the scenario's workspace, interactions, and assertions
/// reusable while allowing the live model to choose its own valid trajectory.
pub fn run_live_case(
    case: &AgentEvalCase,
    project_root: &Path,
    backend: &mut dyn LlmBackend,
) -> Result<AgentEvalResult, AgentEvalError> {
    case.validate()?;
    prepare_workspace(case, project_root)?;

    let recorder = AgentEvalRecorder::default();
    let observer = recorder.clone();
    let mut engine = AgentEngine::new(project_root.to_path_buf()).with_observer(Box::new(observer));
    engine
        .start_session()
        .map_err(|source| AgentEvalError::AgentStep {
            case_id: case.id.clone(),
            step: 0,
            source,
        })?;
    let mut backend = CountingBackend::new(backend);

    for (index, step) in case.steps.iter().enumerate() {
        let step_number = index + 1;
        let response = match step {
            AgentEvalStep::User { input, .. } => engine.run_line(input, &mut backend),
            AgentEvalStep::SetPlanMode { enabled } => engine.set_plan_mode(*enabled),
            AgentEvalStep::PlanDecision { decision } => {
                engine.handle_plan_decision(match decision {
                    AgentEvalDecision::Approve => PlanDecision::Approve,
                    AgentEvalDecision::Reject => PlanDecision::Reject,
                })
            }
            AgentEvalStep::CommandDecision { decision, .. } => engine.handle_command_decision(
                match decision {
                    AgentEvalDecision::Approve => CommandDecision::Approve,
                    AgentEvalDecision::Reject => CommandDecision::Reject,
                },
                &mut backend,
            ),
            AgentEvalStep::Undo => engine.undo_last(),
        }
        .map_err(|source| AgentEvalError::AgentStep {
            case_id: case.id.clone(),
            step: step_number,
            source,
        })?;
        recorder.record_response_if_new(&response);
    }

    let mut trace = recorder.snapshot()?;
    trace.model_steps = backend.model_steps;
    trace.event_kinds = engine
        .session_events()
        .into_iter()
        .map(|event| event.kind)
        .collect();
    Ok(evaluate_trace(case, project_root, trace))
}

struct CountingBackend<'a> {
    inner: &'a mut dyn LlmBackend,
    model_steps: usize,
}

impl<'a> CountingBackend<'a> {
    fn new(inner: &'a mut dyn LlmBackend) -> Self {
        Self {
            inner,
            model_steps: 0,
        }
    }
}

impl LlmBackend for CountingBackend<'_> {
    fn native_tool_support(&self) -> NativeToolSupport {
        self.inner.native_tool_support()
    }

    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn execute(
        &mut self,
        request: GenerationRequest,
        cancellation: &subbake_core::CancellationGuard,
    ) -> Result<GenerationResponse, LlmCallError> {
        self.model_steps += 1;
        self.inner.execute(request, cancellation)
    }
}

pub fn evaluate_trace(
    case: &AgentEvalCase,
    project_root: &Path,
    trace: AgentEvalTrace,
) -> AgentEvalResult {
    let mut failures = Vec::new();
    let final_response = trace.responses.last().map(String::as_str).unwrap_or("");

    for expected in &case.expectations.final_response_contains {
        if !final_response.contains(expected) {
            failures.push(format!(
                "final response does not contain expected text `{expected}`"
            ));
        }
    }
    for expected in &case.expectations.required_calls {
        let matching = matching_call_count(&trace, expected);
        if matching < expected.min_count {
            failures.push(format!(
                "tool `{}` matched {} time(s), fewer than required {}",
                expected.name, matching, expected.min_count
            ));
        }
        if expected.max_count.is_some_and(|maximum| matching > maximum) {
            failures.push(format!(
                "tool `{}` matched {} time(s), above maximum {}",
                expected.name,
                matching,
                expected.max_count.unwrap_or_default()
            ));
        }
    }
    for alternatives in &case.expectations.required_any_calls {
        if alternatives
            .iter()
            .all(|expected| !expected_call_satisfied(&trace, expected))
        {
            failures.push(format!(
                "none of the alternative required tools matched: {:?}",
                alternatives
                    .iter()
                    .map(|expected| expected.name.as_str())
                    .collect::<Vec<_>>()
            ));
        }
    }
    for forbidden in &case.expectations.forbidden_tools {
        if trace.tool_calls.iter().any(|call| call.name == *forbidden) {
            failures.push(format!("forbidden tool `{forbidden}` was executed"));
        }
    }
    if !is_subsequence(
        &case.expectations.tool_sequence,
        trace.tool_calls.iter().map(|call| call.name.as_str()),
    ) {
        failures.push(format!(
            "tool trajectory {:?} does not contain required subsequence {:?}",
            trace
                .tool_calls
                .iter()
                .map(|call| call.name.as_str())
                .collect::<Vec<_>>(),
            case.expectations.tool_sequence
        ));
    }
    for expected in &case.expectations.required_events {
        if !trace.event_kinds.contains(expected) {
            failures.push(format!("required session event `{expected}` is missing"));
        }
    }
    for forbidden in &case.expectations.forbidden_events {
        if trace.event_kinds.contains(forbidden) {
            failures.push(format!(
                "forbidden session event `{forbidden}` was recorded"
            ));
        }
    }
    if case
        .expectations
        .max_tool_calls
        .is_some_and(|maximum| trace.tool_calls.len() > maximum)
    {
        failures.push(format!(
            "used {} tool call(s), above maximum {}",
            trace.tool_calls.len(),
            case.expectations.max_tool_calls.unwrap_or_default()
        ));
    }
    if case
        .expectations
        .max_model_steps
        .is_some_and(|maximum| trace.model_steps > maximum)
    {
        failures.push(format!(
            "used {} model step(s), above maximum {}",
            trace.model_steps,
            case.expectations.max_model_steps.unwrap_or_default()
        ));
    }
    if case
        .expectations
        .step_limit_reached
        .is_some_and(|expected| expected != trace.step_limit_reached)
    {
        failures.push(format!(
            "step-limit state was {}, expected {}",
            trace.step_limit_reached,
            case.expectations.step_limit_reached.unwrap_or_default()
        ));
    }
    if trace.scripted_backend_exhausted {
        failures.push("scripted backend ran out of decisions".to_owned());
    }
    for expected in &case.expectations.files {
        evaluate_file(project_root, expected, &mut failures);
    }

    AgentEvalResult {
        case_id: case.id.clone(),
        description: case.description.clone(),
        passed: failures.is_empty(),
        failures,
        trace,
    }
}

fn prepare_workspace(case: &AgentEvalCase, root: &Path) -> Result<(), AgentEvalError> {
    std::fs::create_dir_all(root).map_err(|source| AgentEvalError::Io {
        path: root.to_path_buf(),
        source,
    })?;
    for file in &case.initial_files {
        let path = root.join(&file.path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| AgentEvalError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        std::fs::write(&path, &file.content)
            .map_err(|source| AgentEvalError::Io { path, source })?;
    }
    Ok(())
}

fn validate_relative_path(path: &Path) -> Result<(), AgentEvalError> {
    let valid = !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir));
    if valid {
        Ok(())
    } else {
        Err(AgentEvalError::InvalidPath {
            path: path.to_path_buf(),
        })
    }
}

fn tool_call_matches(call: &AgentEvalToolCall, expected: &ExpectedToolCall) -> bool {
    call.name == expected.name
        && expected.status.is_none_or(|status| call.status == status)
        && call.arguments.as_object().is_some_and(|arguments| {
            expected
                .arguments
                .iter()
                .all(|(key, value)| arguments.get(key) == Some(value))
        })
}

fn matching_call_count(trace: &AgentEvalTrace, expected: &ExpectedToolCall) -> usize {
    trace
        .tool_calls
        .iter()
        .filter(|call| tool_call_matches(call, expected))
        .count()
}

fn expected_call_satisfied(trace: &AgentEvalTrace, expected: &ExpectedToolCall) -> bool {
    let matching = matching_call_count(trace, expected);
    matching >= expected.min_count && expected.max_count.is_none_or(|maximum| matching <= maximum)
}

fn is_subsequence<'a>(required: &[String], actual: impl IntoIterator<Item = &'a str>) -> bool {
    let mut required = required.iter();
    let mut next = required.next();
    for name in actual {
        if next.is_some_and(|expected| expected == name) {
            next = required.next();
        }
    }
    next.is_none()
}

fn evaluate_file(root: &Path, expected: &ExpectedFile, failures: &mut Vec<String>) {
    let path = root.join(&expected.path);
    if path.exists() != expected.exists {
        failures.push(format!(
            "file `{}` existence was {}, expected {}",
            expected.path,
            path.exists(),
            expected.exists
        ));
        return;
    }
    if !expected.exists {
        return;
    }
    let content = match std::fs::read_to_string(&path) {
        Ok(content) => content,
        Err(error) => {
            failures.push(format!("read expected file `{}`: {error}", expected.path));
            return;
        }
    };
    if expected
        .equals
        .as_ref()
        .is_some_and(|value| value != &content)
    {
        failures.push(format!("file `{}` did not exactly match", expected.path));
    }
    for fragment in &expected.contains {
        if !content.contains(fragment) {
            failures.push(format!(
                "file `{}` does not contain `{fragment}`",
                expected.path
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use subbake_core::{ObservationToolOutcome, ToolExecutionStatus};

    use super::*;

    #[test]
    fn trajectory_matching_allows_other_valid_steps() {
        assert!(is_subsequence(
            &["list_files".to_owned(), "read_file".to_owned()],
            [
                "candidate_subtitles",
                "list_files",
                "diagnose_text",
                "read_file"
            ]
        ));
    }

    #[test]
    fn fixture_paths_cannot_escape_the_eval_workspace() {
        assert!(validate_relative_path(Path::new("subtitles/a.srt")).is_ok());
        assert!(validate_relative_path(Path::new("../outside")).is_err());
        assert!(validate_relative_path(Path::new("/absolute")).is_err());
    }

    #[test]
    fn compact_trace_outcomes_do_not_copy_observed_file_contents() {
        let outcome = AgentToolOutcome::Observation(ObservationToolOutcome {
            status: ToolExecutionStatus::Observed,
            observation: "read_file".to_owned(),
            content: "synthetic-secret-canary".to_owned(),
        });
        let compact = compact_outcome(&outcome).expect("compact outcome");
        assert!(!compact.to_string().contains("synthetic-secret-canary"));
        assert_eq!(compact["operation"], "observation");
        assert_eq!(compact["status"], "observed");
    }
}
