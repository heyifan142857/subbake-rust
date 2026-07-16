//! Decision pipeline — quick-path matching + bounded LLM tool-call loop.
//!
//! Flow per `run_line`:
//!   1. Quick-path keywords match → immediate tool execution.
//!   2. Otherwise → bounded LLM loop (max 5 steps).
//!      - LLM returns a single structured decision (respond / tool_call / ask_user).
//!      - Discovery tools append observation + loop continues.
//!      - Mutating tools exit the loop after execution (or enter plan mode).
//!      - `respond` / `ask_user` / step-limit exit.

use std::io;

use serde_json::{Value as JsonValue, json};
use subbake_core::entities::Usage;
use subbake_core::error::LlmCallError;
use subbake_core::ports::{
    ChatMessage, GenerationContent, GenerationInput, GenerationRequest, GenerationResponse,
    LlmBackend, ModelToolResult, NativeToolSupport, ResponseContract, ToolChoice,
};
#[cfg(test)]
use subbake_core::ports::{ModelToolCall, ToolContinuation};

use crate::discovery::summarize_observation;
use crate::engine::AgentEngine;
use crate::error::{AgentError, AgentResult};
use crate::event::{EventKind, ToolCallDraft};
use crate::profile_coordinator::ProfileCoordinator;
use crate::session::{EventTag, PendingAction};
use crate::tool_runner::ToolRunner;
use crate::tools::{
    ToolAuthorization, ToolIntent, authorize_tool, tool_specs_for_intent, validate_tool_call,
};

mod intent;
mod model;
mod prompts;

use intent::{Route, parse_route};
use model::{
    Decision, DecisionAction, LoopState, NativeTurn, Observation, invalid_decision_response,
    parse_decision_value,
};
use prompts::{build_decision_messages, build_native_messages, build_route_messages};

// ---------------------------------------------------------------------------
// Echo backend for agent decision loop (no TASK_START markers needed)
// ---------------------------------------------------------------------------

/// A lightweight `LlmBackend` that echoes the user message as a JSON
/// decision.  Used when no real LLM provider is configured — the pipeline
/// always chooses "respond" so the TUI/CLI flow can be exercised end-to-end.
pub struct EchoDecisionBackend {
    model: String,
}

impl EchoDecisionBackend {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }
}

impl LlmBackend for EchoDecisionBackend {
    fn provider_name(&self) -> &str {
        "echo-decision"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn execute(
        &mut self,
        request: GenerationRequest,
        cancellation: &subbake_core::CancellationGuard,
    ) -> Result<GenerationResponse, LlmCallError> {
        cancellation.check().map_err(LlmCallError::from)?;
        if request.tools.is_some() {
            return Err(LlmCallError::UnsupportedCapability(
                "native tools".to_owned(),
            ));
        }
        let messages = match request.input {
            GenerationInput::Messages(messages) => messages,
            GenerationInput::Continue { .. } => {
                return Err(LlmCallError::ContinuationMismatch(
                    "echo backend cannot continue native tool calls".to_owned(),
                ));
            }
        };
        // Extract the user message (last user message).
        let user_text = messages
            .iter()
            .rev()
            .find(|msg| msg.role == "user")
            .map(|msg| msg.content.as_str())
            .unwrap_or("");

        let decision = json!({
            "action": "respond",
            "text": user_text,
            "confidence": 1.0
        });
        let input_tokens = user_text.chars().count().div_ceil(4).max(1);
        Ok(GenerationResponse::json(
            decision,
            Usage {
                input_tokens,
                output_tokens: 1,
                total_tokens: input_tokens + 1,
            },
        ))
    }
}

fn execute_json(
    backend: &mut dyn LlmBackend,
    messages: Vec<ChatMessage>,
    cancellation: &subbake_core::CancellationGuard,
) -> Result<(JsonValue, Usage), LlmCallError> {
    backend
        .execute(GenerationRequest::json(messages), cancellation)?
        .into_json()
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const AGENT_LOOP_MAX_STEPS: usize = 5;

// ---------------------------------------------------------------------------
// Engine entry point
// ---------------------------------------------------------------------------

impl AgentEngine {
    /// Create a profile by appending an effective-settings snapshot. It stays
    /// inactive so the current conversation never loses working credentials.
    pub fn create_profile(&mut self, name: &str) -> AgentResult<String> {
        let result = ProfileCoordinator::new(&self.project_root, self.session.as_ref())
            .create_snapshot(name)?;
        self.record_if_active(EventKind::Assistant {
            text: result.clone(),
        })?;
        Ok(result)
    }

    /// Process a single user input line.
    ///
    /// Returns the response text to show to the user.
    pub fn run_line(&mut self, input: &str, backend: &mut dyn LlmBackend) -> AgentResult<String> {
        self.check_cancelled()?;
        self.record_if_active(EventKind::User {
            text: input.to_owned(),
        })?;

        // 1. Quick-path: keyword matching without LLM.
        if let Some(result) = self.try_quick_path(input)? {
            return self.finish_response(result.output, false, result.response_text.is_some());
        }

        let (mut intent, request, inspect_project) = match self.route_request(input, backend)? {
            Route::Respond(text) => return self.finish_response(text, false, true),
            Route::AskUser(text) => return self.finish_response(text, true, true),
            Route::Act {
                intent,
                request,
                inspect_project,
            } => (intent, request, inspect_project),
        };

        // 2. Bounded, intent-scoped tool loop.
        let mut state = LoopState {
            step: 0,
            max_steps: AGENT_LOOP_MAX_STEPS,
            observations: Vec::new(),
            discovery_calls: 0,
            force_no_tools: false,
        };
        let mut native_turn: Option<NativeTurn> = None;
        let mut native_validation_failures = 0usize;
        let mut native_policy_failures = 0usize;

        if inspect_project {
            let arguments = json!({"path": "."});
            let text = self.run_tool("list_files", &arguments)?;
            let summary = summarize_observation("list_files", &text);
            state.observations.push(Observation {
                tool_name: "list_files".to_owned(),
                arguments: arguments.clone(),
                text: text.clone(),
                summary,
            });
            state.discovery_calls = 1;
            state.force_no_tools = !observation_made_progress(&text);
            if let Some(ref mut observer) = self.observer {
                observer.on_tool_call("list_files", &arguments);
                observer.on_observation(&text);
            }
        }

        loop {
            self.check_cancelled()?;
            if state.step > state.max_steps {
                return self.finish_response(
                    "I don't have enough grounded information to act safely. What would you like me to do?"
                        .to_owned(),
                    true,
                    true,
                );
            }
            if state.step >= state.max_steps {
                if let Some(ref mut obs) = self.observer {
                    obs.on_step_limit();
                }
                state.force_no_tools = true;
            }
            state.step += 1;

            // Build context + call LLM.
            let mut decision =
                self.call_llm_for_decision(backend, &request, &state, &mut native_turn, intent)?;

            if !decision.native_calls.is_empty() {
                let continuation = decision.native_continuation.take().ok_or_else(|| {
                    io::Error::other("native tool calls are missing continuation state")
                })?;
                let has_discovery = decision
                    .native_calls
                    .iter()
                    .any(|call| self.is_discovery_tool(&call.name));
                if has_discovery {
                    let mut results = Vec::new();
                    for call in &decision.native_calls {
                        if let Err(error) = authorize_and_refine(&mut intent, &call.name) {
                            results.push(ModelToolResult {
                                id: call.id.clone(),
                                name: call.name.clone(),
                                output: error.to_string(),
                                is_error: true,
                            });
                            continue;
                        }
                        if let Err(error) = validate_tool_call(&call.name, &call.arguments) {
                            results.push(ModelToolResult {
                                id: call.id.clone(),
                                name: call.name.clone(),
                                output: error.to_string(),
                                is_error: true,
                            });
                            continue;
                        }
                        if !self.is_discovery_tool(&call.name) {
                            results.push(ModelToolResult {
                                id: call.id.clone(),
                                name: call.name.clone(),
                                output: "deferred until discovery results are incorporated"
                                    .to_owned(),
                                is_error: true,
                            });
                            continue;
                        }
                        if state.force_no_tools || state.discovery_calls >= 2 {
                            results.push(ModelToolResult {
                                id: call.id.clone(),
                                name: call.name.clone(),
                                output: "discovery budget exhausted; answer from existing context"
                                    .to_owned(),
                                is_error: true,
                            });
                            state.force_no_tools = true;
                            continue;
                        }
                        let cached_output = state
                            .observations
                            .iter()
                            .find(|observation| {
                                observation.tool_name == call.name
                                    && observation.arguments == call.arguments
                            })
                            .map(|observation| observation.text.clone());
                        match cached_output
                            .clone()
                            .map(Ok)
                            .unwrap_or_else(|| self.run_tool(&call.name, &call.arguments))
                        {
                            Ok(output) => {
                                let summary = summarize_observation(&call.name, &output);
                                if cached_output.is_none() {
                                    state.discovery_calls += 1;
                                    if state.discovery_calls >= 2
                                        || !observation_made_progress(&output)
                                    {
                                        state.force_no_tools = true;
                                    }
                                    state.observations.push(Observation {
                                        tool_name: call.name.clone(),
                                        arguments: call.arguments.clone(),
                                        text: output.clone(),
                                        summary,
                                    });
                                }
                                if let Some(ref mut observer) = self.observer {
                                    observer.on_tool_call(&call.name, &call.arguments);
                                    observer.on_observation(&output);
                                }
                                results.push(ModelToolResult {
                                    id: call.id.clone(),
                                    name: call.name.clone(),
                                    output,
                                    is_error: false,
                                });
                            }
                            Err(error) if error.is_cancelled() => {
                                return Err(error);
                            }
                            Err(error) => results.push(ModelToolResult {
                                id: call.id.clone(),
                                name: call.name.clone(),
                                output: error.to_string(),
                                is_error: true,
                            }),
                        }
                    }
                    native_turn = Some(NativeTurn {
                        continuation,
                        results,
                    });
                    continue;
                }

                let mut candidate_intent = intent;
                let validation_errors = decision
                    .native_calls
                    .iter()
                    .map(
                        |call| match authorize_and_refine(&mut candidate_intent, &call.name) {
                            Ok(()) => (
                                validate_tool_call(&call.name, &call.arguments)
                                    .err()
                                    .map(|error| error.to_string()),
                                false,
                            ),
                            Err(error) => (Some(error.to_string()), true),
                        },
                    )
                    .collect::<Vec<_>>();
                if validation_errors.iter().any(|(error, _)| error.is_some()) {
                    let has_policy_error = validation_errors.iter().any(|(_, policy)| *policy);
                    native_validation_failures += 1;
                    native_policy_failures += usize::from(has_policy_error);
                    let details = validation_errors
                        .iter()
                        .filter_map(|(error, _)| error.as_deref())
                        .collect::<Vec<_>>()
                        .join("; ");
                    if native_validation_failures >= 2 {
                        let message = if native_policy_failures > 0 {
                            format!(
                                "I couldn't execute the proposed action because one or more tools are not available for the routed intent: {details}"
                            )
                        } else {
                            format!(
                                "I couldn't execute the proposed action because its arguments were invalid: {details}"
                            )
                        };
                        return self.finish_response(message, true, true);
                    }
                    let results = decision
                        .native_calls
                        .iter()
                        .zip(validation_errors)
                        .map(|(call, (error, _))| ModelToolResult {
                            id: call.id.clone(),
                            name: call.name.clone(),
                            output: error.unwrap_or_else(|| {
                                "not executed because another call in the batch was invalid"
                                    .to_owned()
                            }),
                            is_error: true,
                        })
                        .collect();
                    native_turn = Some(NativeTurn {
                        continuation,
                        results,
                    });
                    continue;
                }
                if decision.native_calls.len() == 1 {
                    let call = &decision.native_calls[0];
                    let result = self.execute_or_plan_tool(&call.name, &call.arguments)?;
                    return self.finish_response(result, false, true);
                }
                let calls = decision
                    .native_calls
                    .into_iter()
                    .map(|call| ToolCallDraft {
                        tool_name: call.name,
                        arguments: call.arguments,
                    })
                    .collect::<Vec<_>>();
                if self.is_in_plan_mode()
                    || calls
                        .iter()
                        .any(|call| self.tool_requires_approval(&call.tool_name))
                {
                    self.store_plan(&decision.text, calls)?;
                    return self.finish_response(self.pending_plan_summary(), false, true);
                }
                let mut outputs = Vec::new();
                for call in calls {
                    self.check_cancelled()?;
                    outputs.push(format!(
                        "{}: {}",
                        call.tool_name,
                        self.run_tool(&call.tool_name, &call.arguments)?
                    ));
                }
                return self.finish_response(outputs.join("\n"), false, true);
            }

            match decision.action {
                DecisionAction::Respond => {
                    return self.finish_response(decision.text, false, true);
                }

                DecisionAction::AskUser => {
                    self.set_pending_action(intent, &request)?;
                    return self.finish_response(decision.text, true, true);
                }

                DecisionAction::ToolCall => {
                    let tool_name = decision.tool_name.as_deref().unwrap_or("unknown");
                    let args = decision.arguments.unwrap_or(json!({}));

                    if let Err(error) = authorize_and_refine(&mut intent, tool_name) {
                        return self.finish_response(
                            format!("I couldn't use the proposed tool: {error}"),
                            true,
                            true,
                        );
                    }

                    if self.is_discovery_tool(tool_name) {
                        if state.force_no_tools || state.discovery_calls >= 2 {
                            state.force_no_tools = true;
                            continue;
                        }
                        // Discovery → run, append observation, continue.
                        let cached_output = state
                            .observations
                            .iter()
                            .find(|observation| {
                                observation.tool_name == tool_name && observation.arguments == args
                            })
                            .map(|observation| observation.text.clone());
                        let obs_text = if let Some(output) = cached_output.as_ref() {
                            output.clone()
                        } else {
                            self.run_tool(tool_name, &args)?
                        };
                        let summary = summarize_observation(tool_name, &obs_text);
                        if cached_output.is_none() {
                            state.discovery_calls += 1;
                            if state.discovery_calls >= 2 || !observation_made_progress(&obs_text) {
                                state.force_no_tools = true;
                            }
                            state.observations.push(Observation {
                                tool_name: tool_name.to_owned(),
                                arguments: args.clone(),
                                text: obs_text.clone(),
                                summary,
                            });
                        }
                        if let Some(ref mut obs) = self.observer {
                            obs.on_tool_call(tool_name, &args);
                            obs.on_observation(&obs_text);
                        }
                        continue;
                    }

                    // Mutating tool (execute, then exit loop).
                    // Check plan mode / approval.
                    let result_text = self.execute_or_plan_tool(tool_name, &args)?;
                    return self.finish_response(result_text, false, true);
                }

                DecisionAction::Plan => {
                    let mut candidate_intent = intent;
                    for call in &decision.tool_calls {
                        if let Err(error) =
                            authorize_and_refine(&mut candidate_intent, &call.tool_name)
                        {
                            return self.finish_response(
                                format!(
                                    "The proposed plan contains an unauthorized action: {error}"
                                ),
                                true,
                                true,
                            );
                        }
                    }
                    if self.is_in_plan_mode() {
                        self.store_plan(&decision.text, decision.tool_calls)?;
                        return self.finish_response(self.pending_plan_summary(), false, true);
                    }

                    let mut outputs = Vec::new();
                    for call in decision.tool_calls {
                        self.check_cancelled()?;
                        outputs.push(format!(
                            "{}: {}",
                            call.tool_name,
                            self.run_tool(&call.tool_name, &call.arguments)?
                        ));
                    }
                    let response = if outputs.is_empty() {
                        decision.text
                    } else {
                        outputs.join("\n")
                    };
                    return self.finish_response(response, false, true);
                }

                DecisionAction::NativeToolCalls => {
                    return self.finish_response(
                        "The model returned native tool calls without continuation state. Please try again."
                            .to_owned(),
                        true,
                        true,
                    );
                }
            }
        }
    }

    fn finish_response(
        &mut self,
        text: String,
        ask_user: bool,
        notify_observer: bool,
    ) -> AgentResult<String> {
        if !ask_user {
            self.clear_pending_action()?;
        }
        let event = if ask_user {
            EventKind::AskUser { text: text.clone() }
        } else {
            EventKind::Assistant { text: text.clone() }
        };
        self.record_if_active(event)?;
        if notify_observer && let Some(ref mut observer) = self.observer {
            observer.on_response(&text);
        }
        Ok(text)
    }

    // ------------------------------------------------------------------
    // Quick-path deterministic matching
    // ------------------------------------------------------------------

    fn try_quick_path(&mut self, input: &str) -> AgentResult<Option<QuickResult>> {
        let trimmed = input.trim();

        // Pattern: "translate @<path>" or "translate <path>"
        if let Some(path) = trimmed.strip_prefix("translate ") {
            let args = json!({"path": self.tool_path_arg(path)});
            let output = self.execute_or_plan_tool("translate_file", &args)?;
            return Ok(Some(QuickResult {
                response_text: Some(output.clone()),
                output,
            }));
        }

        // Pattern: "transcribe @<path>"
        if let Some(path) = trimmed.strip_prefix("transcribe ") {
            let args = json!({"path": self.tool_path_arg(path)});
            let output = self.execute_or_plan_tool("transcribe_audio", &args)?;
            return Ok(Some(QuickResult {
                response_text: Some(output.clone()),
                output,
            }));
        }

        // Pattern: "list files" or "ls"
        if matches!(trimmed, "list files" | "ls" | "list") {
            return Ok(Some(QuickResult {
                output: self
                    .guard
                    .list_files(std::path::Path::new("."))
                    .map(|files| {
                        files
                            .iter()
                            .map(|p| p.display().to_string())
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default(),
                response_text: None,
            }));
        }

        Ok(None)
    }

    fn tool_path_arg(&self, input: &str) -> String {
        // Strip @ prefix if present.
        input.trim().trim_start_matches('@').to_owned()
    }

    fn route_request(&mut self, input: &str, backend: &mut dyn LlmBackend) -> AgentResult<Route> {
        if let Some(route) = self.pending_action_route(input) {
            return Ok(route);
        }
        let messages = build_route_messages(
            input,
            None,
            self.pending_action(),
            self.dialogue_context_summary(8),
        );
        if let Some(ref mut observer) = self.observer {
            observer.on_thinking("Understanding your request…");
        }
        let first = execute_json(backend, messages, &self.operation_guard);
        let (value, _) = match first {
            Ok(value) => value,
            Err(LlmCallError::Cancelled) => {
                return Err(AgentError::Cancelled);
            }
            Err(error) => {
                return Ok(Route::AskUser(format!(
                    "I couldn't understand the request safely: {error}"
                )));
            }
        };
        match parse_route(&value, input) {
            Ok(route) => Ok(route),
            Err(error) => {
                let repair = build_route_messages(
                    input,
                    Some(&error.to_string()),
                    self.pending_action(),
                    self.dialogue_context_summary(8),
                );
                match execute_json(backend, repair, &self.operation_guard) {
                    Ok((value, _)) => parse_route(&value, input).or_else(|_| {
                        Ok(Route::AskUser(
                            "Could you clarify whether you want to discuss something or act on the current project?"
                                .to_owned(),
                        ))
                    }),
                    Err(LlmCallError::Cancelled) => Err(AgentError::Cancelled),
                    Err(_) => Ok(Route::AskUser(
                        "Could you clarify whether you want to discuss something or act on the current project?"
                            .to_owned(),
                    )),
                }
            }
        }
    }

    fn pending_action(&self) -> Option<&PendingAction> {
        self.session
            .as_ref()
            .and_then(|session| session.pending_action.as_ref())
    }

    fn pending_action_route(&self, input: &str) -> Option<Route> {
        let pending = self.pending_action()?;
        if !looks_like_path_value(input) {
            return None;
        }
        let intent = ToolIntent::parse(&pending.intent)?;
        Some(Route::Act {
            intent,
            request: format!(
                "{}\n\nThe user supplied this requested path: {}",
                pending.request,
                input.trim()
            ),
            inspect_project: false,
        })
    }

    fn set_pending_action(&mut self, intent: ToolIntent, request: &str) -> AgentResult<()> {
        let Some(session) = self.session.as_mut() else {
            return Ok(());
        };
        session.pending_action = Some(PendingAction {
            intent: intent.as_str().to_owned(),
            request: request.to_owned(),
        });
        self.save()
    }

    fn clear_pending_action(&mut self) -> AgentResult<()> {
        let Some(session) = self.session.as_mut() else {
            return Ok(());
        };
        if session.pending_action.take().is_some() {
            self.save()?;
        }
        Ok(())
    }

    fn dialogue_context_summary(&self, limit: usize) -> Option<String> {
        let session = self.session.as_ref()?;
        let mut lines = session
            .events
            .iter()
            .rev()
            .skip_while(|event| event.tag() == EventTag::User)
            .filter_map(|event| match event.tag() {
                EventTag::User => Some(format!("User: {}", truncate_text(&event.text, 240))),
                EventTag::Assistant | EventTag::AskUser => {
                    Some(format!("Assistant: {}", truncate_text(&event.text, 240)))
                }
                _ => None,
            })
            .take(limit)
            .collect::<Vec<_>>();
        lines.reverse();
        (!lines.is_empty()).then(|| lines.join("\n"))
    }

    // ------------------------------------------------------------------
    // LLM decision call
    // ------------------------------------------------------------------

    fn call_llm_for_decision(
        &mut self,
        backend: &mut dyn LlmBackend,
        user_input: &str,
        state: &LoopState,
        native_turn: &mut Option<NativeTurn>,
        intent: ToolIntent,
    ) -> AgentResult<Decision> {
        if backend.native_tool_support() != NativeToolSupport::Unsupported {
            let was_continuation = native_turn.is_some();
            let tools = if state.force_no_tools {
                Vec::new()
            } else {
                tool_specs_for_intent(intent)
                    .into_iter()
                    .map(|spec| spec.native_definition())
                    .collect()
            };
            if let Some(ref mut observer) = self.observer {
                observer.on_thinking("Deciding next action…");
            }
            let choice = if state.force_no_tools {
                ToolChoice::None
            } else {
                ToolChoice::Auto
            };
            let request = if let Some(turn) = native_turn.take() {
                GenerationRequest::continue_with_tools(
                    turn.continuation,
                    turn.results,
                    tools,
                    choice,
                    ResponseContract::Text,
                )
            } else {
                GenerationRequest::text(build_native_messages(
                    user_input,
                    state,
                    intent,
                    self.dialogue_context_summary(12),
                ))
                .with_tools(tools, choice)
            };
            match backend.execute(request, &self.operation_guard) {
                Ok(response) => {
                    let GenerationResponse {
                        content,
                        tool_calls,
                        continuation,
                        ..
                    } = response;
                    let text = match content {
                        GenerationContent::Empty => String::new(),
                        GenerationContent::Text(text) => text,
                        GenerationContent::Json(json) => json.to_string(),
                    };
                    if !tool_calls.is_empty() {
                        return Ok(Decision::native(text, tool_calls, continuation));
                    }
                    return Ok(Decision::response(text));
                }
                Err(LlmCallError::UnsupportedCapability(_)) if !was_continuation => {}
                Err(LlmCallError::Cancelled) => {
                    return Err(AgentError::Cancelled);
                }
                Err(error) => {
                    if let Some(ref mut observer) = self.observer {
                        observer.on_error(&error.to_string());
                    }
                    return Ok(Decision::response(format!("Error: {error}")));
                }
            }
        }
        self.call_legacy_decision(backend, user_input, state, intent)
    }

    fn call_legacy_decision(
        &mut self,
        backend: &mut dyn LlmBackend,
        user_input: &str,
        state: &LoopState,
        intent: ToolIntent,
    ) -> AgentResult<Decision> {
        let messages = build_decision_messages(
            user_input,
            state,
            None,
            intent,
            self.dialogue_context_summary(12),
        );
        if let Some(ref mut obs) = self.observer {
            obs.on_thinking("Deciding next action…");
        }
        let result = execute_json(backend, messages, &self.operation_guard);
        match result {
            Ok((decision, _usage)) => match parse_decision_value(&decision, |name| {
                self.is_discovery_tool(name)
            }) {
                Ok(decision) => Ok(decision),
                Err(first_error) => {
                    if let Some(ref mut obs) = self.observer {
                        obs.on_error(&first_error.to_string());
                    }
                    let repair_messages = build_decision_messages(
                        user_input,
                        state,
                        Some(&first_error.to_string()),
                        intent,
                        self.dialogue_context_summary(12),
                    );
                    match execute_json(backend, repair_messages, &self.operation_guard) {
                        Ok((repaired, _usage)) => match parse_decision_value(&repaired, |name| {
                            self.is_discovery_tool(name)
                        }) {
                            Ok(decision) => Ok(decision),
                            Err(second_error) => {
                                if let Some(ref mut obs) = self.observer {
                                    obs.on_error(&second_error.to_string());
                                }
                                Ok(invalid_decision_response(&second_error))
                            }
                        },
                        Err(LlmCallError::Cancelled) => Err(AgentError::Cancelled),
                        Err(error) => Ok(Decision::ask_user(format!(
                            "The proposed action was invalid ({first_error}), and the repair attempt failed: {error}"
                        ))),
                    }
                }
            },
            Err(e) => {
                if matches!(e, LlmCallError::Cancelled) {
                    return Err(AgentError::Cancelled);
                }
                if let Some(ref mut obs) = self.observer {
                    obs.on_error(&e.to_string());
                }
                Ok(Decision::response(format!("Error: {e}")))
            }
        }
    }

    // ------------------------------------------------------------------
    // Plan mode check
    // ------------------------------------------------------------------

    fn is_in_plan_mode(&self) -> bool {
        self.session
            .as_ref()
            .is_some_and(|s| s.mode == crate::session::SessionMode::Plan)
    }

    // ------------------------------------------------------------------
    // Tool runner (stub — dispatches to real adapters)
    // ------------------------------------------------------------------

    fn execute_or_plan_tool(&mut self, tool_name: &str, args: &JsonValue) -> AgentResult<String> {
        if let Some(ref mut obs) = self.observer {
            obs.on_tool_call(tool_name, args);
        }

        if self.is_in_plan_mode() || self.tool_requires_approval(tool_name) {
            let draft = crate::event::ToolCallDraft {
                tool_name: tool_name.to_owned(),
                arguments: args.clone(),
            };
            self.store_plan("", vec![draft])?;
            return Ok(self.pending_plan_summary());
        }

        self.run_tool(tool_name, args)
    }

    pub(crate) fn record_if_active(&mut self, kind: EventKind) -> AgentResult<()> {
        if self.session.is_some() {
            self.record(kind)?;
        }
        Ok(())
    }

    /// Execute a tool by name with arguments. Returns a text summary.
    pub(crate) fn run_tool(&mut self, name: &str, args: &JsonValue) -> AgentResult<String> {
        ToolRunner::run(self, name, args)
    }
}

fn truncate_text(text: &str, limit: usize) -> String {
    let value = text.chars().take(limit).collect::<String>();
    if text.chars().count() > limit {
        format!("{value}...")
    } else {
        value
    }
}

fn authorize_and_refine(intent: &mut ToolIntent, name: &str) -> AgentResult<()> {
    match authorize_tool(*intent, name).map_err(|error| AgentError::ToolPolicy {
        message: error.to_string(),
    })? {
        ToolAuthorization::Allowed => Ok(()),
        ToolAuthorization::Transition(target) => {
            *intent = target;
            Ok(())
        }
    }
}

fn looks_like_path_value(input: &str) -> bool {
    let value = input.trim();
    !value.is_empty()
        && (value.starts_with('/')
            || value.starts_with("./")
            || value.starts_with("../")
            || value.starts_with("~/")
            || value.contains('\\'))
}

fn observation_made_progress(text: &str) -> bool {
    let trimmed = text.trim();
    !trimmed.is_empty() && trimmed != "(no files found)"
}

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct QuickResult {
    output: String,
    response_text: Option<String>,
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    struct RawDecisionBackend {
        decision: JsonValue,
    }

    struct SequenceDecisionBackend {
        decisions: VecDeque<JsonValue>,
        prompts: Vec<Vec<ChatMessage>>,
    }

    enum NativeScript {
        Calls(Vec<ModelToolCall>),
        Text(String),
        Error(LlmCallError),
    }

    struct NativeSequenceBackend {
        scripts: VecDeque<NativeScript>,
        continued_results: Vec<Vec<ModelToolResult>>,
        legacy_decision: Option<JsonValue>,
        native_calls: usize,
        legacy_calls: usize,
        native_error: Option<subbake_core::CoreError>,
    }

    impl LlmBackend for NativeSequenceBackend {
        fn provider_name(&self) -> &str {
            "native-test"
        }

        fn model_name(&self) -> &str {
            "native-test"
        }

        fn native_tool_support(&self) -> NativeToolSupport {
            if matches!(
                self.native_error,
                Some(subbake_core::CoreError::UnsupportedCapability(_))
            ) {
                NativeToolSupport::Unknown
            } else {
                NativeToolSupport::Supported
            }
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &subbake_core::CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            cancellation.check().map_err(LlmCallError::from)?;
            if request.tools.is_some() {
                self.native_calls += 1;
                if let Some(error) = &self.native_error {
                    return Err(error.clone().into());
                }
                if let GenerationInput::Continue { tool_results, .. } = request.input {
                    self.continued_results.push(tool_results);
                }
                return match self
                    .scripts
                    .pop_front()
                    .unwrap_or_else(|| NativeScript::Text("native done".to_owned()))
                {
                    NativeScript::Calls(tool_calls) => Ok(GenerationResponse {
                        content: GenerationContent::Text(String::new()),
                        tool_calls,
                        continuation: Some(ToolContinuation::new(
                            "native-test",
                            "test continuation".to_owned(),
                        )),
                        usage: Usage::default(),
                    }),
                    NativeScript::Text(text) => Ok(GenerationResponse {
                        content: GenerationContent::Text(text),
                        tool_calls: Vec::new(),
                        continuation: None,
                        usage: Usage::default(),
                    }),
                    NativeScript::Error(error) => Err(error),
                };
            }
            let messages = match request.input {
                GenerationInput::Messages(messages) => messages,
                GenerationInput::Continue { .. } => {
                    return Err(LlmCallError::ContinuationMismatch(
                        "test continuation lacks tools".to_owned(),
                    ));
                }
            };
            self.legacy_calls += 1;
            let routing = messages.iter().any(|message| {
                message.role == "system" && message.content.contains("semantic router")
            });
            let inferred_route = self.scripts.front().and_then(|script| match script {
                NativeScript::Calls(calls) => calls.first().map(|call| {
                    json!({
                        "route": "act",
                        "intent": test_intent_for_tool(&call.name),
                        "restated_request": "test action",
                        "inspect_project": false
                    })
                }),
                NativeScript::Text(_) => None,
                NativeScript::Error(_) => None,
            });
            Ok(GenerationResponse::json(
                if routing {
                    self.legacy_decision
                        .clone()
                        .or(inferred_route)
                        .unwrap_or_else(|| json!({"route":"respond","text":"legacy"}))
                } else {
                    json!({"action":"respond","text":"legacy fallback"})
                },
                Usage::default(),
            ))
        }
    }

    fn test_intent_for_tool(name: &str) -> &'static str {
        match name {
            "candidate_subtitles" | "translate_file" | "translate_series" => "translate",
            "transcribe_audio" => "transcribe",
            "recent_translations" | "edit_subtitle" => "edit",
            "diagnose_path" | "diagnose_text" => "diagnose",
            "create_file" => "file_create",
            "append_file" => "file_append",
            "replace_in_file" => "file_replace",
            "rename_path" => "file_rename",
            "delete_file" => "file_delete",
            "list_profiles" | "switch_profile" => "profile",
            "manage_whisper" => "whisper",
            _ => "browse",
        }
    }

    impl LlmBackend for SequenceDecisionBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "sequence"
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &subbake_core::CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            cancellation.check().map_err(LlmCallError::from)?;
            let GenerationInput::Messages(messages) = request.input else {
                return Err(LlmCallError::ContinuationMismatch(
                    "sequence backend cannot continue".to_owned(),
                ));
            };
            self.prompts.push(messages);
            Ok(GenerationResponse::json(
                self.decisions
                    .pop_front()
                    .unwrap_or_else(|| json!({"action": "respond", "text": "done"})),
                Usage::default(),
            ))
        }
    }

    #[test]
    fn invalid_tool_call_gets_one_contextual_repair_attempt() {
        let root = temp_root("decision-repair");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = SequenceDecisionBackend {
            decisions: VecDeque::from([
                json!({"route": "act"}),
                json!({"route": "respond", "text": "repaired"}),
            ]),
            prompts: Vec::new(),
        };

        let response = engine
            .run_line("process this request", &mut backend)
            .expect("run decision loop");
        assert_eq!(response, "repaired");
        assert_eq!(backend.prompts.len(), 2);
        let repair_system = backend.prompts[1]
            .iter()
            .find(|message| message.role == "system")
            .expect("repair system prompt");
        assert!(repair_system.content.contains("missing `intent`"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn translation_prompt_distinguishes_directory_and_bilingual_requests() {
        let root = temp_root("translation-prompt");
        let engine = AgentEngine::new(root);
        let messages = build_decision_messages(
            "翻译目录下的srt文件成为中英双语字幕",
            &LoopState {
                step: 1,
                max_steps: AGENT_LOOP_MAX_STEPS,
                observations: vec![Observation {
                    tool_name: "list_files".to_owned(),
                    arguments: json!({"path": "."}),
                    text: "/project/movie.srt".to_owned(),
                    summary: "1 file".to_owned(),
                }],
                discovery_calls: 1,
                force_no_tools: false,
            },
            None,
            ToolIntent::Translate,
            engine.dialogue_context_summary(12),
        );
        let system = messages
            .iter()
            .find(|message| message.role == "system")
            .expect("system prompt");
        let user = messages
            .iter()
            .find(|message| message.role == "user")
            .expect("user prompt");
        assert!(system.content.contains("translate_series"));
        assert!(system.content.contains("bilingual=true"));
        assert!(user.content.contains("/project/movie.srt"));
    }

    #[test]
    fn vague_translation_request_discovers_and_translates_unique_candidate() {
        let root = temp_root("tool-first-unique");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join("movie.txt"), "hello\n").expect("write subtitle");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = SequenceDecisionBackend {
            decisions: VecDeque::from([
                json!({"route":"act","intent":"translate","restated_request":"translate movie.txt","inspect_project":true}),
                json!({"action":"tool_call","tool_name":"translate_file","arguments":{"path":"movie.txt"}}),
            ]),
            prompts: Vec::new(),
        };

        let response = engine.run_line("帮我翻译", &mut backend).expect("run line");

        assert!(response.contains("Translated:"), "{response}");
        assert!(root.join("movie.translated.txt").exists());
        assert!(
            engine
                .session
                .as_ref()
                .expect("session")
                .events
                .iter()
                .any(|event| event.kind == "tool_call" && event.text == "list_files")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn model_requested_project_inspection_precedes_native_discovery() {
        let root = temp_root("tool-first-native-unique");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(
            root.join("sample.srt"),
            "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
        )
        .expect("write subtitle");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = NativeSequenceBackend {
            scripts: VecDeque::from([NativeScript::Calls(vec![ModelToolCall {
                id: "wrong_discovery".to_owned(),
                name: "list_files".to_owned(),
                arguments: json!({"path": "."}),
            }])]),
            continued_results: Vec::new(),
            legacy_decision: None,
            native_calls: 0,
            legacy_calls: 0,
            native_error: None,
        };

        let response = engine.run_line("帮我翻译", &mut backend).expect("run line");

        assert_eq!(response, "native done");
        assert_eq!(backend.native_calls, 2);
        assert_eq!(backend.legacy_calls, 1);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn vague_translation_request_asks_when_discovery_finds_multiple_candidates() {
        let root = temp_root("tool-first-multiple");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join("one.srt"), "one\n").expect("write first subtitle");
        std::fs::write(root.join("two.srt"), "two\n").expect("write second subtitle");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = SequenceDecisionBackend {
            decisions: VecDeque::from([
                json!({"route":"act","intent":"translate","restated_request":"translate a subtitle","inspect_project":true}),
                json!({"action":"ask_user","text":"请选择 one.srt 或 two.srt"}),
            ]),
            prompts: Vec::new(),
        };

        let response = engine.run_line("帮我翻译", &mut backend).expect("run line");

        assert!(response.contains("请选择"), "{response}");
        assert!(response.contains("one.srt"), "{response}");
        assert!(response.contains("two.srt"), "{response}");
        assert!(!root.join("one.translated.srt").exists());
        assert!(!root.join("two.translated.srt").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn vague_translation_request_asks_for_path_only_after_empty_discovery() {
        let root = temp_root("tool-first-empty");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = SequenceDecisionBackend {
            decisions: VecDeque::from([
                json!({"route":"act","intent":"translate","restated_request":"translate a subtitle","inspect_project":true}),
                json!({"action":"ask_user","text":"path?"}),
            ]),
            prompts: Vec::new(),
        };

        let response = engine.run_line("帮我翻译", &mut backend).expect("run line");

        assert_eq!(response, "path?");
        assert_eq!(
            engine
                .session
                .as_ref()
                .expect("session")
                .events
                .last()
                .expect("event")
                .kind,
            "ask_user"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn supplied_path_resumes_the_pending_edit_intent() {
        let root = temp_root("pending-edit-path");
        std::fs::create_dir_all(&root).expect("create root");
        let source = root.join("source.srt");
        std::fs::write(&source, "1\n00:00:00,000 --> 00:00:01,000\nhello\n")
            .expect("write source subtitle");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = SequenceDecisionBackend {
            decisions: VecDeque::from([
                json!({"route":"act","intent":"edit","restated_request":"make the translated subtitle bilingual","inspect_project":false}),
                json!({"action":"ask_user","text":"Please provide the original subtitle path."}),
                json!({"action":"tool_call","tool_name":"read_file_preview","arguments":{"path":source}}),
                json!({"action":"respond","text":"The source subtitle is available."}),
            ]),
            prompts: Vec::new(),
        };

        engine
            .run_line("make this translated subtitle bilingual", &mut backend)
            .expect("ask for source");
        assert_eq!(
            engine
                .session
                .as_ref()
                .expect("session")
                .pending_action
                .as_ref()
                .expect("pending action")
                .intent,
            "edit"
        );

        let response = engine
            .run_line(&source.display().to_string(), &mut backend)
            .expect("continue edit");

        assert_eq!(response, "The source subtitle is available.");
        assert!(
            engine
                .session
                .as_ref()
                .expect("session")
                .pending_action
                .is_none()
        );
        assert!(
            engine
                .session
                .as_ref()
                .expect("session")
                .events
                .iter()
                .any(|event| event.kind == "tool_call" && event.text == "read_file_preview")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn unique_discovered_translation_waits_in_plan_mode() {
        let root = temp_root("tool-first-plan");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join("movie.srt"), "hello\n").expect("write subtitle");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        engine.set_plan_mode(true).expect("enable plan mode");
        let mut backend = SequenceDecisionBackend {
            decisions: VecDeque::from([
                json!({"route":"act","intent":"translate","restated_request":"translate movie.srt","inspect_project":true}),
                json!({"action":"tool_call","tool_name":"translate_file","arguments":{"path":"movie.srt"}}),
            ]),
            prompts: Vec::new(),
        };

        let response = engine.run_line("帮我翻译", &mut backend).expect("run line");

        assert!(response.contains("Choose an action below"), "{response}");
        assert!(!root.join("movie.translated.srt").exists());
        let pending = engine
            .session
            .as_ref()
            .expect("session")
            .pending_plan
            .as_ref()
            .expect("pending plan");
        assert_eq!(pending.tool_calls[0].tool_name, "translate_file");
        assert_eq!(
            pending.tool_calls[0].arguments["path"].as_str(),
            Some("movie.srt")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn informational_translation_question_can_receive_a_direct_response() {
        let root = temp_root("tool-first-information");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join("movie.srt"), "hello\n").expect("write subtitle");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = RawDecisionBackend {
            decision: json!({"action": "respond", "text": "Use the translate command."}),
        };

        let response = engine
            .run_line("如何翻译字幕？", &mut backend)
            .expect("run line");

        assert_eq!(response, "Use the translate command.");
        assert!(!root.join("movie.translated.srt").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn short_conversational_reply_does_not_call_tools() {
        let root = temp_root("short-chat");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = RawDecisionBackend {
            decision: json!({"route":"respond","text":"Got it."}),
        };

        let response = engine.run_line("1", &mut backend).expect("chat response");

        assert_eq!(response, "Got it.");
        assert!(
            !engine
                .session
                .as_ref()
                .expect("session")
                .events
                .iter()
                .any(|event| event.kind == "tool_call")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn model_can_route_a_contextual_short_reply_to_an_action() {
        let root = temp_root("contextual-short-action");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        engine
            .record_if_active(EventKind::AskUser {
                text: "Should I inspect the project files?".to_owned(),
            })
            .expect("record question");
        let mut backend = SequenceDecisionBackend {
            decisions: VecDeque::from([
                json!({"route":"act","intent":"browse","restated_request":"inspect project files","inspect_project":true}),
                json!({"action":"respond","text":"The project is empty."}),
            ]),
            prompts: Vec::new(),
        };

        let response = engine
            .run_line("1", &mut backend)
            .expect("contextual action");

        assert_eq!(response, "The project is empty.");
        assert!(
            engine
                .session
                .as_ref()
                .expect("session")
                .events
                .iter()
                .any(|event| event.kind == "tool_call" && event.text == "list_files")
        );
        let route_user = backend.prompts[0]
            .iter()
            .find(|message| message.role == "user")
            .expect("route user");
        assert!(route_user.content.contains("Should I inspect"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn routed_intent_rejects_an_unrelated_tool() {
        let root = temp_root("intent-allowlist");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join("keep.txt"), "keep").expect("write file");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = SequenceDecisionBackend {
            decisions: VecDeque::from([
                json!({"route":"act","intent":"translate","restated_request":"translate subtitles","inspect_project":false}),
                json!({"action":"tool_call","tool_name":"delete_file","arguments":{"path":"keep.txt"}}),
            ]),
            prompts: Vec::new(),
        };

        let response = engine
            .run_line("work on subtitles", &mut backend)
            .expect("route");

        assert!(
            response.contains(
                "tool `delete_file` requires intent `file_delete`, but this turn is using intent `translate`"
            ),
            "{response}"
        );
        assert!(root.join("keep.txt").exists());
        assert!(
            !engine
                .session
                .as_ref()
                .expect("session")
                .events
                .iter()
                .any(|event| event.kind == "file_operation")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn other_existing_resource_actions_explore_before_asking() {
        let root = temp_root("tool-first-transcribe");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join("clip.mp4"), "media").expect("write media placeholder");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = SequenceDecisionBackend {
            decisions: VecDeque::from([
                json!({
                    "route": "act",
                    "intent": "transcribe",
                    "restated_request": "transcribe the project media",
                    "inspect_project": true
                }),
                json!({"action": "respond", "text": "found the media"}),
            ]),
            prompts: Vec::new(),
        };

        let response = engine.run_line("帮我转录", &mut backend).expect("run line");

        assert_eq!(response, "found the media");
        assert_eq!(backend.prompts.len(), 2);
        assert!(
            backend.prompts[1]
                .iter()
                .find(|message| message.role == "user")
                .expect("user message")
                .content
                .contains("clip.mp4")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_discovery_result_continues_without_legacy_json() {
        let root = temp_root("native-discovery");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join("clip.srt"), "subtitle").expect("write file");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = NativeSequenceBackend {
            scripts: VecDeque::from([
                NativeScript::Calls(vec![ModelToolCall {
                    id: "call_1".to_owned(),
                    name: "list_files".to_owned(),
                    arguments: json!({"path":"."}),
                }]),
                NativeScript::Text("found it".to_owned()),
            ]),
            continued_results: Vec::new(),
            legacy_decision: None,
            native_calls: 0,
            legacy_calls: 0,
            native_error: None,
        };

        let response = engine
            .run_line("show project files", &mut backend)
            .expect("native loop");

        assert_eq!(response, "found it");
        assert_eq!(backend.native_calls, 2);
        assert_eq!(backend.legacy_calls, 1);
        assert_eq!(backend.continued_results.len(), 1);
        assert!(backend.continued_results[0][0].output.contains("clip.srt"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn repeated_native_discovery_reuses_the_existing_observation() {
        let root = temp_root("native-discovery-cache");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join("clip.srt"), "subtitle").expect("write file");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let repeated_call = || ModelToolCall {
            id: "call_1".to_owned(),
            name: "list_files".to_owned(),
            arguments: json!({"path":"."}),
        };
        let mut backend = NativeSequenceBackend {
            scripts: VecDeque::from([
                NativeScript::Calls(vec![repeated_call()]),
                NativeScript::Calls(vec![repeated_call()]),
                NativeScript::Text("done".to_owned()),
            ]),
            continued_results: Vec::new(),
            legacy_decision: None,
            native_calls: 0,
            legacy_calls: 0,
            native_error: None,
        };

        let response = engine
            .run_line("show project files", &mut backend)
            .expect("native loop");

        assert_eq!(response, "done");
        assert_eq!(backend.continued_results.len(), 2);
        assert_eq!(
            engine
                .session
                .as_ref()
                .expect("session")
                .events
                .iter()
                .filter(|event| event.kind == "tool_call" && event.text == "list_files")
                .count(),
            1
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_discovery_budget_blocks_a_third_read() {
        let root = temp_root("native-discovery-budget");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join("clip.srt"), "subtitle").expect("write file");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = NativeSequenceBackend {
            scripts: VecDeque::from([
                NativeScript::Calls(vec![ModelToolCall {
                    id: "call_1".to_owned(),
                    name: "list_files".to_owned(),
                    arguments: json!({"path":"."}),
                }]),
                NativeScript::Calls(vec![ModelToolCall {
                    id: "call_2".to_owned(),
                    name: "search_files".to_owned(),
                    arguments: json!({"path":".","pattern":"*.srt"}),
                }]),
                NativeScript::Calls(vec![ModelToolCall {
                    id: "call_3".to_owned(),
                    name: "read_file_preview".to_owned(),
                    arguments: json!({"path":"clip.srt"}),
                }]),
                NativeScript::Text("I found clip.srt.".to_owned()),
            ]),
            continued_results: Vec::new(),
            legacy_decision: None,
            native_calls: 0,
            legacy_calls: 0,
            native_error: None,
        };

        let response = engine
            .run_line("inspect this project", &mut backend)
            .expect("bounded discovery");

        assert_eq!(response, "I found clip.srt.");
        let tool_names = engine
            .session
            .as_ref()
            .expect("session")
            .events
            .iter()
            .filter(|event| event.kind == "tool_call")
            .map(|event| event.text.as_str())
            .collect::<Vec<_>>();
        assert_eq!(tool_names, vec!["list_files", "search_files"]);
        assert!(
            backend.continued_results[2][0]
                .output
                .contains("budget exhausted")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn unsupported_native_backend_falls_back_to_json_once() {
        let root = temp_root("native-fallback");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = NativeSequenceBackend {
            scripts: VecDeque::new(),
            continued_results: Vec::new(),
            legacy_decision: Some(
                json!({"route":"act","intent":"browse","restated_request":"show files","inspect_project":false}),
            ),
            native_calls: 0,
            legacy_calls: 0,
            native_error: Some(subbake_core::CoreError::UnsupportedCapability(
                "native tools".to_owned(),
            )),
        };

        let response = engine
            .run_line("hello", &mut backend)
            .expect("fallback loop");

        assert_eq!(response, "legacy fallback");
        assert_eq!(backend.native_calls, 1);
        assert_eq!(backend.legacy_calls, 2);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn unsupported_continuation_does_not_replay_the_legacy_decision() {
        let root = temp_root("native-continuation-no-fallback");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = NativeSequenceBackend {
            scripts: VecDeque::from([
                NativeScript::Calls(vec![ModelToolCall {
                    id: "list_1".to_owned(),
                    name: "list_files".to_owned(),
                    arguments: json!({"path": "."}),
                }]),
                NativeScript::Error(LlmCallError::UnsupportedCapability(
                    "continuation rejected".to_owned(),
                )),
            ]),
            continued_results: Vec::new(),
            legacy_decision: None,
            native_calls: 0,
            legacy_calls: 0,
            native_error: None,
        };

        let response = engine
            .run_line("show files", &mut backend)
            .expect("native continuation failure response");

        assert!(response.contains("continuation rejected"), "{response}");
        assert_eq!(backend.native_calls, 2);
        assert_eq!(backend.legacy_calls, 1);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_backend_failure_does_not_fall_back() {
        let root = temp_root("native-no-fallback");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = NativeSequenceBackend {
            scripts: VecDeque::new(),
            continued_results: Vec::new(),
            legacy_decision: Some(
                json!({"route":"act","intent":"browse","restated_request":"show files","inspect_project":false}),
            ),
            native_calls: 0,
            legacy_calls: 0,
            native_error: Some(subbake_core::CoreError::InvalidBackendResponse(
                "rate limited".to_owned(),
            )),
        };

        let response = engine
            .run_line("hello", &mut backend)
            .expect("native failure response");

        assert!(response.contains("rate limited"), "{response}");
        assert_eq!(backend.native_calls, 1);
        assert_eq!(backend.legacy_calls, 1);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_mutation_uses_existing_plan_approval() {
        let root = temp_root("native-plan");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        engine.set_plan_mode(true).expect("plan mode");
        let mut backend = NativeSequenceBackend {
            scripts: VecDeque::from([NativeScript::Calls(vec![ModelToolCall {
                id: "call_1".to_owned(),
                name: "create_file".to_owned(),
                arguments: json!({"path":"note.txt","content":"hello"}),
            }])]),
            continued_results: Vec::new(),
            legacy_decision: None,
            native_calls: 0,
            legacy_calls: 0,
            native_error: None,
        };

        let response = engine
            .run_line("create a note", &mut backend)
            .expect("native plan");

        assert!(response.contains("Choose an action below"));
        assert!(!root.join("note.txt").exists());
        assert_eq!(
            engine
                .session
                .as_ref()
                .expect("session")
                .pending_plan
                .as_ref()
                .expect("plan")
                .tool_calls[0]
                .tool_name,
            "create_file"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_invalid_arguments_get_one_tool_result_repair() {
        let root = temp_root("native-repair");
        std::fs::create_dir_all(&root).expect("create root");
        let invalid = || ModelToolCall {
            id: "call_1".to_owned(),
            name: "translate_file".to_owned(),
            arguments: json!({}),
        };
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = NativeSequenceBackend {
            scripts: VecDeque::from([
                NativeScript::Calls(vec![invalid()]),
                NativeScript::Calls(vec![invalid()]),
            ]),
            continued_results: Vec::new(),
            legacy_decision: None,
            native_calls: 0,
            legacy_calls: 0,
            native_error: None,
        };

        let response = engine
            .run_line("perform a translation", &mut backend)
            .expect("native repair");

        assert!(
            response.contains("requires string argument `path`"),
            "{response}"
        );
        assert_eq!(backend.continued_results.len(), 1);
        assert!(backend.continued_results[0][0].is_error);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_policy_rejection_is_not_reported_as_an_argument_error() {
        let root = temp_root("native-policy-rejection");
        std::fs::create_dir_all(&root).expect("create root");
        let rejected = || ModelToolCall {
            id: "call_1".to_owned(),
            name: "delete_file".to_owned(),
            arguments: json!({"path":"keep.txt"}),
        };
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = NativeSequenceBackend {
            scripts: VecDeque::from([
                NativeScript::Calls(vec![rejected()]),
                NativeScript::Calls(vec![rejected()]),
            ]),
            continued_results: Vec::new(),
            legacy_decision: Some(json!({
                "route":"act",
                "intent":"translate",
                "restated_request":"translate subtitles",
                "inspect_project":false
            })),
            native_calls: 0,
            legacy_calls: 0,
            native_error: None,
        };

        let response = engine
            .run_line("please translate subtitles", &mut backend)
            .expect("native policy rejection");

        assert!(
            response.contains(
                "tool `delete_file` requires intent `file_delete`, but this turn is using intent `translate`"
            ),
            "{response}"
        );
        assert!(!response.contains("arguments were invalid"), "{response}");
        assert_eq!(backend.continued_results.len(), 1);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_translation_route_refines_to_edit_after_discovery() {
        let root = temp_root("native-translate-to-edit");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(
            root.join("clip.srt"),
            "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
        )
        .expect("write source");
        std::fs::write(
            root.join("clip.translated.srt"),
            "1\n00:00:00,000 --> 00:00:01,000\n你好\n",
        )
        .expect("write translation");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = NativeSequenceBackend {
            scripts: VecDeque::from([
                NativeScript::Calls(vec![ModelToolCall {
                    id: "read_source".to_owned(),
                    name: "read_file_preview".to_owned(),
                    arguments: json!({"path":"clip.srt"}),
                }]),
                NativeScript::Calls(vec![ModelToolCall {
                    id: "edit_translation".to_owned(),
                    name: "edit_subtitle".to_owned(),
                    arguments: json!({
                        "path":"clip.translated.srt",
                        "instruction":"combine each source line with its translation"
                    }),
                }]),
            ]),
            continued_results: Vec::new(),
            legacy_decision: Some(json!({
                "route":"act",
                "intent":"translate",
                "restated_request":"make clip.translated.srt bilingual",
                "inspect_project":false
            })),
            native_calls: 0,
            legacy_calls: 0,
            native_error: None,
        };

        let response = engine
            .run_line(
                "make the existing translated subtitle bilingual",
                &mut backend,
            )
            .expect("refine translation route to edit");

        assert!(response.contains("Edited:"), "{response}");
        assert_eq!(backend.continued_results.len(), 1);
        assert!(!backend.continued_results[0][0].is_error);
        assert!(
            engine
                .session
                .as_ref()
                .expect("session")
                .events
                .iter()
                .any(|event| event.kind == "tool_call" && event.text == "edit_subtitle")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn legacy_translation_route_refines_to_edit() {
        let root = temp_root("legacy-translate-to-edit");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(
            root.join("clip.srt"),
            "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
        )
        .expect("write source");
        std::fs::write(
            root.join("clip.translated.srt"),
            "1\n00:00:00,000 --> 00:00:01,000\n你好\n",
        )
        .expect("write translation");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = SequenceDecisionBackend {
            decisions: VecDeque::from([
                json!({
                    "route":"act",
                    "intent":"translate",
                    "restated_request":"make clip.translated.srt bilingual",
                    "inspect_project":false
                }),
                json!({
                    "action":"tool_call",
                    "tool_name":"edit_subtitle",
                    "arguments":{
                        "path":"clip.translated.srt",
                        "instruction":"combine each source line with its translation"
                    }
                }),
            ]),
            prompts: Vec::new(),
        };

        let response = engine
            .run_line(
                "make the existing translated subtitle bilingual",
                &mut backend,
            )
            .expect("refine legacy route to edit");

        assert!(response.contains("Edited:"), "{response}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn approval_required_native_tool_waits_even_in_chat_mode() {
        let root = temp_root("native-required-approval");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = NativeSequenceBackend {
            scripts: VecDeque::from([NativeScript::Calls(vec![ModelToolCall {
                id: "call_1".to_owned(),
                name: "manage_whisper".to_owned(),
                arguments: json!({"action":"status"}),
            }])]),
            continued_results: Vec::new(),
            legacy_decision: None,
            native_calls: 0,
            legacy_calls: 0,
            native_error: None,
        };

        let response = engine
            .run_line("check whisper", &mut backend)
            .expect("approval plan");

        assert!(response.contains("Choose an action below"), "{response}");
        assert_eq!(
            engine
                .session
                .as_ref()
                .expect("session")
                .pending_plan
                .as_ref()
                .expect("pending plan")
                .tool_calls[0]
                .tool_name,
            "manage_whisper"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    impl LlmBackend for RawDecisionBackend {
        fn provider_name(&self) -> &str {
            "test"
        }

        fn model_name(&self) -> &str {
            "decision"
        }

        fn execute(
            &mut self,
            _request: GenerationRequest,
            cancellation: &subbake_core::CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            cancellation.check().map_err(LlmCallError::from)?;
            Ok(GenerationResponse::json(
                self.decision.clone(),
                Usage::default(),
            ))
        }
    }

    #[test]
    fn quick_path_translate_executes_tool() {
        let root = temp_root("quick-translate");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join("clip.txt"), "hello\n").expect("write subtitle");

        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = EchoDecisionBackend::new("test");

        let output = engine
            .run_line("translate @clip.txt", &mut backend)
            .expect("run line");

        assert!(output.contains("Translated:"), "{output}");
        assert!(root.join("clip.translated.txt").exists());
        assert!(
            engine
                .session
                .as_ref()
                .expect("session")
                .events
                .iter()
                .any(|event| event.kind == "assistant")
        );
        let context = engine
            .conversation_context_summary(12)
            .expect("conversation summary");
        assert!(context.contains("User:"));
        assert!(context.contains("Assistant:"));

        engine.undo_last().expect("undo translation");
        assert!(!root.join("clip.translated.txt").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn translation_tool_bilingual_argument_overrides_settings_for_one_call() {
        let root = temp_root("bilingual-override");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join("clip.txt"), "hello\n").expect("write subtitle");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        engine
            .run_tool(
                "translate_file",
                &json!({"path": "clip.txt", "bilingual": true}),
            )
            .expect("translate bilingual");

        let output = root.join("clip.bilingual.txt");
        assert!(output.exists());
        let content = std::fs::read_to_string(output).expect("read output");
        assert!(content.lines().count() >= 2, "{content}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn undo_restores_overwritten_translation_output() {
        let root = temp_root("translate-overwrite-undo");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(root.join("clip.txt"), "new\n").expect("write subtitle");
        std::fs::write(root.join("clip.translated.txt"), "old\n").expect("write old output");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = EchoDecisionBackend::new("test");

        engine
            .run_line("translate @clip.txt", &mut backend)
            .expect("translate");
        assert_ne!(
            std::fs::read_to_string(root.join("clip.translated.txt")).expect("translated"),
            "old\n"
        );

        engine.undo_last().expect("undo translation");
        assert_eq!(
            std::fs::read_to_string(root.join("clip.translated.txt")).expect("restored"),
            "old\n"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn undo_removes_series_outputs_as_one_group() {
        let root = temp_root("series-undo");
        let season = root.join("season");
        std::fs::create_dir_all(&season).expect("create season");
        std::fs::write(season.join("one.txt"), "one\n").expect("write one");
        std::fs::write(season.join("two.txt"), "two\n").expect("write two");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        engine
            .run_tool(
                "translate_series",
                &json!({"path": "season", "recursive": false}),
            )
            .expect("translate series");
        assert!(season.join("one.translated.txt").exists());
        assert!(season.join("two.translated.txt").exists());
        let recent = engine
            .run_tool("recent_translations", &json!({}))
            .expect("recent");
        assert!(recent.contains("one.translated.txt"));
        assert!(recent.contains("two.translated.txt"));

        let result = engine.undo_last().expect("undo series");
        assert!(result.contains("2 operations"));
        assert!(!season.join("one.translated.txt").exists());
        assert!(!season.join("two.translated.txt").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn llm_plan_waits_for_approval_and_then_executes() {
        let root = temp_root("llm-plan");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        engine.set_plan_mode(true).expect("enable plan mode");
        let mut backend = RawDecisionBackend {
            decision: json!({
                "action": "plan",
                "text": "Create notes",
                "tool_calls": [{
                    "tool_name": "create_file",
                    "arguments": {"path": "notes.txt", "content": "hello"}
                }],
                "confidence": 0.95
            }),
        };

        let response = engine.run_line("create notes", &mut backend).expect("plan");
        assert!(response.contains("Choose an action below"));
        assert!(!root.join("notes.txt").exists());
        assert_eq!(
            engine
                .session
                .as_ref()
                .expect("session")
                .pending_plan
                .as_ref()
                .expect("pending")
                .tool_calls
                .len(),
            1
        );

        engine.approve_plan().expect("approve");
        assert_eq!(
            std::fs::read_to_string(root.join("notes.txt")).expect("created"),
            "hello"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn failed_plan_keeps_only_unfinished_tool_calls() {
        let root = temp_root("partial-plan");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        engine
            .store_plan(
                "Create then append",
                vec![
                    ToolCallDraft {
                        tool_name: "create_file".to_owned(),
                        arguments: json!({"path": "notes.txt", "content": "hello"}),
                    },
                    ToolCallDraft {
                        tool_name: "rename_path".to_owned(),
                        arguments: json!({"from": "notes.txt"}),
                    },
                ],
            )
            .expect("store plan");

        engine.approve_plan().expect_err("rename must fail");
        assert_eq!(
            std::fs::read_to_string(root.join("notes.txt")).expect("first call completed"),
            "hello"
        );
        let remaining = &engine
            .session
            .as_ref()
            .expect("session")
            .pending_plan
            .as_ref()
            .expect("pending")
            .tool_calls;
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].tool_name, "rename_path");

        engine.approve_plan().expect_err("retry still fails");
        assert_eq!(
            std::fs::read_to_string(root.join("notes.txt")).expect("not recreated"),
            "hello"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn explicit_plan_on_and_off_are_idempotent() {
        let root = temp_root("explicit-plan-mode");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        assert_eq!(
            engine.set_plan_mode(true).expect("on"),
            "Plan mode on. Mutating tools will wait for your approval."
        );
        assert_eq!(
            engine.set_plan_mode(true).expect("on again"),
            "Plan mode on. Mutating tools will wait for your approval."
        );
        assert_eq!(engine.set_plan_mode(false).expect("off"), "Plan mode off.");
        assert_eq!(
            engine.set_plan_mode(false).expect("off again"),
            "Plan mode off."
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn revision_in_plan_mode_replaces_the_pending_plan_without_execution() {
        let root = temp_root("revise-plan");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        engine
            .store_plan(
                "Create the original note",
                vec![ToolCallDraft {
                    tool_name: "create_file".to_owned(),
                    arguments: json!({"path": "original.txt", "content": "old"}),
                }],
            )
            .expect("store original plan");
        let mut backend = RawDecisionBackend {
            decision: json!({
                "action": "plan",
                "text": "Create the revised note",
                "tool_calls": [{
                    "tool_name": "create_file",
                    "arguments": {"path": "revised.txt", "content": "new"}
                }],
                "confidence": 0.95
            }),
        };

        let response = engine
            .run_line("Instead create revised.txt", &mut backend)
            .expect("revised plan");
        assert!(response.contains("Choose an action below"));
        assert!(!root.join("original.txt").exists());
        assert!(!root.join("revised.txt").exists());
        let pending = engine
            .session
            .as_ref()
            .expect("session")
            .pending_plan
            .as_ref()
            .expect("pending plan");
        assert_eq!(pending.tool_calls.len(), 1);
        assert_eq!(
            pending.tool_calls[0].arguments["path"].as_str(),
            Some("revised.txt")
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cancellation_prevents_the_next_tool_side_effect() {
        let root = temp_root("cancel-tool");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let token = engine.cancellation_token();
        let queued_guard = token.guard();
        token.cancel();
        engine.begin_operation(queued_guard);

        let error = engine
            .run_tool(
                "create_file",
                &json!({"path": "cancelled.txt", "content": "no"}),
            )
            .expect_err("cancelled tool must not run");
        assert!(error.is_cancelled());
        assert!(!root.join("cancelled.txt").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn repeatedly_invalid_llm_tool_call_reports_specific_validation_error() {
        let root = temp_root("invalid-decision");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");
        let mut backend = RawDecisionBackend {
            decision: json!({
                "action": "tool_call",
                "tool_name": "shell",
                "arguments": {"command": "rm -rf ."},
                "confidence": 1.0
            }),
        };

        let response = engine
            .run_line("do something", &mut backend)
            .expect("clarification");

        assert!(response.contains("Could you clarify"), "{response}");
        assert!(
            !engine
                .session
                .as_ref()
                .expect("session")
                .events
                .iter()
                .any(|event| event.kind == "tool_call")
        );
        assert_eq!(
            engine
                .session
                .as_ref()
                .expect("session")
                .events
                .last()
                .expect("event")
                .kind,
            "ask_user"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn unknown_tool_never_reports_placeholder_success() {
        let root = temp_root("unknown-tool");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        let error = engine
            .run_tool("not_registered", &json!({}))
            .expect_err("unknown tool must fail");
        assert!(matches!(error, AgentError::InvalidInput { .. }));
        assert!(error.to_string().contains("unknown agent tool"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn file_tool_records_event_and_undo_removes_created_file() {
        let root = temp_root("file-undo");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        let result = engine
            .run_tool(
                "create_file",
                &json!({"path": "notes.txt", "content": "hello"}),
            )
            .expect("create file");

        assert!(result.contains("Created"));
        assert!(root.join("notes.txt").exists());
        assert!(
            engine
                .session
                .as_ref()
                .expect("session")
                .events
                .iter()
                .any(|event| event.kind == "file_operation")
        );

        let undone = engine.undo_last().expect("undo");
        assert!(undone.contains("Undone"));
        assert!(!root.join("notes.txt").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn edit_subtitle_updates_file_and_undo_restores_previous_content() {
        let root = temp_root("edit-undo");
        std::fs::create_dir_all(&root).expect("create root");
        let target = root.join("clip.translated.txt");
        std::fs::write(&target, "hello\n").expect("write target");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        let output = engine
            .run_tool(
                "edit_subtitle",
                &json!({"path": "clip.translated.txt", "instruction": "make it uppercase"}),
            )
            .expect("edit subtitle");
        assert!(output.contains("Edited:"));
        assert_eq!(
            std::fs::read_to_string(&target).expect("read edited"),
            "HELLO\n"
        );

        engine.undo_last().expect("undo edit");
        assert_eq!(
            std::fs::read_to_string(&target).expect("read restored"),
            "hello\n"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn diagnose_text_reports_structured_cause() {
        let root = temp_root("diagnose-text");
        std::fs::create_dir_all(&root).expect("create root");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        let output = engine
            .run_tool(
                "diagnose_text",
                &json!({"text": "missing api key for provider"}),
            )
            .expect("diagnose text");

        assert!(output.contains("credentials"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn switch_profile_updates_active_session_profile() {
        let root = temp_root("switch-profile");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(
            root.join("subbake.toml"),
            "version = 1\n\
             default_profile = \"fast\"\n\
             [defaults.backend]\nid = \"mock\"\n\
             [profiles.fast.backend]\nmodel = \"mock-fast\"\n\
             [profiles.strict.backend]\nmodel = \"mock-strict\"\n",
        )
        .expect("write config");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        let output = engine
            .run_tool("switch_profile", &json!({"name": "strict"}))
            .expect("switch profile");

        assert!(output.contains("strict"));
        let session = engine.session.as_ref().expect("session");
        assert_eq!(session.profile.as_deref(), Some("strict"));
        assert_eq!(
            session.config_path.as_deref(),
            Some(root.join("subbake.toml").to_string_lossy().as_ref())
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn profile_picker_rows_include_active_model_metadata_and_new_choice() {
        let root = temp_root("profile-picker-rows");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::write(
            root.join("subbake.toml"),
            "version = 1\n\
             default_profile = \"fast\"\n\
             [defaults.backend]\nid = \"mock\"\n\
             [profiles.fast.backend]\nmodel = \"mock-fast\"\n\
             [profiles.strict.backend]\nmodel = \"mock-strict\"\n",
        )
        .expect("write config");
        let mut engine = AgentEngine::new(root.clone());
        engine.start_session().expect("start session");

        let rows = engine.profile_picker_choices().expect("profile rows");
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].name, "fast");
        assert!(rows[0].active);
        assert_eq!(rows[0].model, "mock-fast");
        assert_eq!(rows[1].name, "strict");
        assert!(rows[2].create);
        assert_eq!(rows[2].name, "new profile…");
        let _ = std::fs::remove_dir_all(&root);
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-agent-decision-{label}-{nanos}"))
    }
}
