//! Stable-tool agent task loop.
//!
//! Each user message enters one bounded loop. The model sees the complete
//! model-visible registry throughout the task, and every validation or
//! execution result is fed back until the model returns a final response.

use std::collections::{HashMap, HashSet, VecDeque};

use serde_json::{Value as JsonValue, json};
use subbake_core::entities::Usage;
use subbake_core::error::LlmCallError;
use subbake_core::languages::normalize_language;
use subbake_core::ports::{
    ChatMessage, GenerationContent, GenerationInput, GenerationRequest, GenerationResponse,
    LlmBackend, ModelToolCall, ModelToolResult, NativeToolSupport, ResponseContract, ToolChoice,
};

use crate::engine::AgentEngine;
use crate::error::{AgentError, AgentResult};
use crate::event::{
    EventKind, PendingAgentTurn, PendingAggregateFailureCount, PendingFailureCount,
    PendingToolCall, PendingToolExchange, ToolCallDraft,
};
use crate::profile_coordinator::ProfileCoordinator;
use crate::tool_execution::render_tool_outcome;
use crate::tools::{
    ToolValidationError, find_tool_spec, model_visible_tool_names, model_visible_tool_specs,
    validate_tool_call,
};

mod model;
mod prompts;

use model::{
    AgentTaskLoop, Decision, DecisionAction, NativeTurn, ToolExchange, ToolFeedback,
    parse_json_decision,
};
use prompts::{build_json_messages, build_native_messages};

pub const AGENT_LOOP_MAX_STEPS: usize = 64;

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
        let GenerationInput::Messages(messages) = request.input else {
            return Err(LlmCallError::ContinuationMismatch(
                "echo backend cannot continue native tool calls".to_owned(),
            ));
        };
        let user_text = messages
            .iter()
            .rev()
            .find(|message| message.role == "user")
            .map(|message| message.content.as_str())
            .unwrap_or("");
        let user_text = user_text
            .strip_prefix("Current user request:\n")
            .and_then(|text| text.split("\n\n").next())
            .unwrap_or(user_text);
        let input_tokens = user_text.chars().count().div_ceil(4).max(1);
        Ok(GenerationResponse::json(
            json!({
                "action": "respond",
                "text": user_text,
            }),
            Usage {
                input_tokens,
                output_tokens: 1,
                total_tokens: input_tokens + 1,
                requests: 1,
                ..Usage::default()
            },
        ))
    }
}

impl AgentEngine {
    pub fn create_profile(&mut self, name: &str) -> AgentResult<String> {
        let result = ProfileCoordinator::new(&self.project_root, self.session.as_ref())
            .create_snapshot(name)?;
        self.record_if_active(EventKind::Assistant {
            text: result.clone(),
        })?;
        Ok(result)
    }

    pub fn run_line(&mut self, input: &str, backend: &mut dyn LlmBackend) -> AgentResult<String> {
        self.check_cancelled()?;
        let dialogue = self.dialogue_context_summary(12);
        let legacy_pending = self.take_legacy_pending_action()?;
        let effective_defaults = self.effective_defaults_summary()?;
        self.record_if_active(EventKind::User {
            text: input.to_owned(),
        })?;

        let state = AgentTurnState {
            input: input.to_owned(),
            dialogue,
            legacy_pending,
            effective_defaults,
            task: AgentTaskLoop::default(),
            legacy_mode: backend.native_tool_support() == NativeToolSupport::Unsupported,
            failure_counts: HashMap::new(),
            aggregate_failure_counts: HashMap::new(),
            completed_mutations: HashSet::new(),
            steps_used: 0,
        };
        self.run_turn(backend, state, None)
    }

    fn run_turn(
        &mut self,
        backend: &mut dyn LlmBackend,
        mut state: AgentTurnState,
        mut native_turn: Option<NativeTurn>,
    ) -> AgentResult<String> {
        while state.steps_used < self.runtime_policy.max_steps() {
            self.check_cancelled()?;
            let decision = self.call_model(
                backend,
                &state.input,
                &state.task,
                state.dialogue.as_deref(),
                state.legacy_pending.as_deref(),
                &mut native_turn,
                &mut state.legacy_mode,
                true,
                &state.effective_defaults,
            )?;
            state.steps_used += 1;
            match decision.action {
                DecisionAction::Respond => {
                    self.clear_pending_agent_turn()?;
                    return self.finish_response(nonempty_response(decision.text), false);
                }
                DecisionAction::AskUser => {
                    self.clear_pending_agent_turn()?;
                    return self.finish_response(nonempty_question(decision.text), true);
                }
                DecisionAction::ToolCalls => {
                    let continuation = decision.continuation;
                    let processed = self.process_tool_calls(
                        decision.calls,
                        &mut state.task,
                        &mut state.failure_counts,
                        &mut state.aggregate_failure_counts,
                        &mut state.completed_mutations,
                    )?;
                    match processed {
                        ProcessedCalls::Continue(results) => {
                            native_turn = continuation.map(|continuation| NativeTurn {
                                continuation,
                                results,
                            });
                        }
                        ProcessedCalls::Planned => {
                            self.clear_pending_agent_turn()?;
                            return self.finish_response(self.pending_plan_summary(), false);
                        }
                        ProcessedCalls::AwaitingCommandApproval {
                            call,
                            reason,
                            completed_results,
                            remaining_calls,
                        } => {
                            self.pending_native_continuation = continuation.map(|continuation| {
                                crate::engine::PendingNativeContinuation {
                                    continuation,
                                    results: completed_results,
                                }
                            });
                            self.store_command_approval(
                                ToolCallDraft {
                                    tool_name: call.name,
                                    arguments: call.arguments,
                                },
                                call.id,
                                remaining_calls
                                    .into_iter()
                                    .map(PendingToolCall::from)
                                    .collect(),
                                reason,
                                state.into_pending(),
                            )?;
                            return self
                                .finish_response(self.pending_command_approval_summary(), false);
                        }
                        ProcessedCalls::RepeatedFailure(message) => {
                            self.clear_pending_agent_turn()?;
                            return self.finish_response(message, false);
                        }
                    }
                }
            }
        }

        if let Some(observer) = self.observer.as_mut() {
            observer.on_step_limit();
        }
        let final_decision = self.call_model(
            backend,
            &state.input,
            &state.task,
            state.dialogue.as_deref(),
            state.legacy_pending.as_deref(),
            &mut native_turn,
            &mut state.legacy_mode,
            false,
            &state.effective_defaults,
        )?;
        self.clear_pending_agent_turn()?;
        match final_decision.action {
            DecisionAction::Respond => {
                self.finish_response(nonempty_response(final_decision.text), false)
            }
            DecisionAction::AskUser => {
                self.finish_response(nonempty_question(final_decision.text), true)
            }
            DecisionAction::ToolCalls => self.finish_response(
                format!(
                    "Agent reached its configured model-step limit ({}/{}). No further tools were executed; increase `agent.max_steps` only if this task is making valid progress.",
                    state.steps_used,
                    self.runtime_policy.max_steps()
                ),
                true,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn call_model(
        &mut self,
        backend: &mut dyn LlmBackend,
        input: &str,
        task: &AgentTaskLoop,
        dialogue: Option<&str>,
        legacy_pending: Option<&str>,
        native_turn: &mut Option<NativeTurn>,
        legacy_mode: &mut bool,
        tools_enabled: bool,
        effective_defaults: &str,
    ) -> AgentResult<Decision> {
        if !*legacy_mode {
            let was_continuation = native_turn.is_some();
            let definitions = if tools_enabled {
                model_visible_tool_specs()
                    .into_iter()
                    .map(|spec| spec.native_definition())
                    .collect()
            } else {
                Vec::new()
            };
            let choice = if tools_enabled {
                ToolChoice::Auto
            } else {
                ToolChoice::None
            };
            let request = if let Some(turn) = native_turn.take() {
                GenerationRequest::continue_with_tools(
                    turn.continuation,
                    turn.results,
                    definitions,
                    choice,
                    ResponseContract::Text,
                )
            } else {
                GenerationRequest::text(build_native_messages(
                    input,
                    task,
                    dialogue,
                    legacy_pending,
                    effective_defaults,
                ))
                .with_tools(definitions, choice)
            };
            if let Some(observer) = self.observer.as_mut() {
                observer.on_thinking("Deciding next action…");
            }
            match backend.execute(request, &self.operation_guard) {
                Ok(response) => return native_decision(response),
                Err(LlmCallError::UnsupportedCapability(_)) if !was_continuation => {
                    *legacy_mode = true;
                }
                Err(LlmCallError::Cancelled) => return Err(AgentError::Cancelled),
                Err(error) => {
                    if let Some(observer) = self.observer.as_mut() {
                        observer.on_error(&error.to_string());
                    }
                    return Ok(Decision::response(format!("Provider error: {error}")));
                }
            }
        }

        if let Some(observer) = self.observer.as_mut() {
            observer.on_thinking("Deciding next action…");
        }
        let messages = build_json_messages(
            input,
            task,
            dialogue,
            legacy_pending,
            tools_enabled,
            None,
            effective_defaults,
        );
        match execute_json(backend, messages, &self.operation_guard) {
            Ok((value, _)) => match parse_json_decision(&value) {
                Ok(decision) => Ok(decision),
                Err(error) => {
                    if let Some(observer) = self.observer.as_mut() {
                        observer.on_error(&error.to_string());
                    }
                    Ok(Decision::ask_user(format!(
                        "I couldn't obtain a valid model decision: {error}"
                    )))
                }
            },
            Err(LlmCallError::Cancelled) => Err(AgentError::Cancelled),
            Err(error) => {
                if let Some(observer) = self.observer.as_mut() {
                    observer.on_error(&error.to_string());
                }
                Ok(Decision::response(format!("Provider error: {error}")))
            }
        }
    }

    fn process_tool_calls(
        &mut self,
        calls: Vec<ModelToolCall>,
        task: &mut AgentTaskLoop,
        failure_counts: &mut HashMap<FailureKey, usize>,
        aggregate_failure_counts: &mut HashMap<AggregateFailureKey, usize>,
        completed_mutations: &mut HashSet<CallKey>,
    ) -> AgentResult<ProcessedCalls> {
        if calls.is_empty() {
            return Ok(ProcessedCalls::RepeatedFailure(
                "The model returned an empty tool-call turn. Please retry the request.".to_owned(),
            ));
        }

        let planned = calls
            .iter()
            .filter_map(|call| {
                find_tool_spec(&call.name)
                    .filter(|spec| spec.model_visible)
                    .filter(|spec| {
                        spec.mutates_with(&call.arguments)
                            && !(call.name == "run_command" && !self.is_in_plan_mode())
                            && (self.is_in_plan_mode()
                                || (spec.requires_approval_with(&call.arguments)
                                    && !(call.name == "run_command"
                                        && self.runtime_policy.auto_approve_commands())))
                    })
                    .and_then(|_| {
                        validate_tool_call(&call.name, &call.arguments)
                            .is_ok()
                            .then(|| ToolCallDraft {
                                tool_name: call.name.clone(),
                                arguments: call.arguments.clone(),
                            })
                    })
            })
            .collect::<Vec<_>>();
        if !planned.is_empty() {
            for call in &planned {
                if let Some(observer) = self.observer.as_mut() {
                    observer.on_tool_call(&call.tool_name, &call.arguments);
                }
            }
            self.store_plan("", planned)?;
            return Ok(ProcessedCalls::Planned);
        }

        let available_tools = model_visible_tool_names()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let mut native_results = Vec::new();
        let mut calls = VecDeque::from(calls);
        while let Some(call) = calls.pop_front() {
            self.check_cancelled()?;
            if call.name == "run_command"
                && validate_tool_call(&call.name, &call.arguments).is_ok()
                && !self.runtime_policy.auto_approve_commands()
                && let crate::command_policy::CommandApproval::AskUser(reason) =
                    crate::command_policy::classify(&call.arguments)
            {
                if let Some(observer) = self.observer.as_mut() {
                    observer.on_tool_call(&call.name, &call.arguments);
                }
                return Ok(ProcessedCalls::AwaitingCommandApproval {
                    call,
                    reason,
                    completed_results: native_results,
                    remaining_calls: calls.into(),
                });
            }
            if let Some(observer) = self.observer.as_mut() {
                observer.on_tool_call(&call.name, &call.arguments);
            }
            let call_key = CallKey::new(&call.name, &call.arguments);
            let feedback = match find_tool_spec(&call.name) {
                None => ToolFeedback::failure(
                    &call.name,
                    format!("unknown tool `{}`", call.name),
                    "unknown_tool",
                    available_tools.clone(),
                ),
                Some(spec) if !spec.model_visible => ToolFeedback::failure(
                    &call.name,
                    format!("tool `{}` is compatibility-only and unavailable", call.name),
                    "unknown_tool",
                    available_tools.clone(),
                ),
                Some(spec) => match validate_tool_call(&call.name, &call.arguments) {
                    Err(error) => ToolFeedback::failure(
                        &call.name,
                        error.to_string(),
                        validation_category(&error),
                        available_tools.clone(),
                    ),
                    Ok(())
                        if spec.mutates_with(&call.arguments)
                            && completed_mutations.contains(&call_key) =>
                    {
                        ToolFeedback::failure(
                            &call.name,
                            "this successful mutating call was already executed in this task"
                                .to_owned(),
                            "duplicate_mutation",
                            available_tools.clone(),
                        )
                    }
                    Ok(()) => match self.run_tool(&call.name, &call.arguments) {
                        Ok(outcome) => {
                            if spec.mutates_with(&call.arguments) {
                                completed_mutations.insert(call_key.clone());
                            }
                            let observation = render_tool_outcome(&outcome);
                            if let Some(observer) = self.observer.as_mut() {
                                observer.on_observation(&observation);
                            }
                            ToolFeedback::success(&call.name, outcome)
                        }
                        Err(error) if error.is_cancelled() => return Err(error),
                        Err(error) => {
                            let category = error.tool_failure_category();
                            ToolFeedback::failure(
                                &call.name,
                                error.to_string(),
                                &category,
                                available_tools.clone(),
                            )
                        }
                    },
                },
            };
            let recorded = self.record_tool_feedback(
                &call,
                &call_key,
                feedback,
                task,
                failure_counts,
                aggregate_failure_counts,
            )?;
            native_results.push(recorded.result);
            if let Some(message) = recorded.stop_message {
                return Ok(ProcessedCalls::RepeatedFailure(message));
            }
        }
        Ok(ProcessedCalls::Continue(native_results))
    }

    fn record_tool_feedback(
        &mut self,
        call: &ModelToolCall,
        call_key: &CallKey,
        feedback: ToolFeedback,
        task: &mut AgentTaskLoop,
        failure_counts: &mut HashMap<FailureKey, usize>,
        aggregate_failure_counts: &mut HashMap<AggregateFailureKey, usize>,
    ) -> AgentResult<RecordedToolFeedback> {
        let is_error = !feedback.success;
        let output = feedback.json();
        task.exchanges.push(ToolExchange {
            call_id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
            feedback: output.clone(),
            is_error,
        });
        let result = ModelToolResult {
            id: call.id.clone(),
            name: call.name.clone(),
            output,
            is_error,
        };
        if !is_error {
            return Ok(RecordedToolFeedback {
                result,
                stop_message: None,
            });
        }

        let category = feedback
            .error_category
            .as_deref()
            .unwrap_or("execution")
            .to_owned();
        let error = feedback
            .error
            .as_deref()
            .unwrap_or("the tool call failed")
            .to_owned();
        if let Some(observer) = self.observer.as_mut() {
            observer.on_error(&format!("{}: {error}", call.name));
        }
        self.record_if_active(EventKind::ToolFailure {
            tool_name: call.name.clone(),
            category: category.clone(),
            error: error.clone(),
        })?;

        let exact = FailureKey {
            call: call_key.clone(),
            category: category.clone(),
        };
        let exact_count = failure_counts.entry(exact).or_default();
        *exact_count += 1;
        if *exact_count >= 2 {
            return Ok(RecordedToolFeedback {
                result,
                stop_message: Some(format!(
                    "The model repeated the same failed `{}` call twice: {error}. Please retry with a more specific path or instruction.",
                    call.name
                )),
            });
        }

        if is_aggregate_failure_category(&category) {
            let aggregate = AggregateFailureKey {
                tool_name: call.name.clone(),
                category: category.clone(),
            };
            let count = aggregate_failure_counts.entry(aggregate).or_default();
            *count += 1;
            if *count >= 3 {
                return Ok(RecordedToolFeedback {
                    result,
                    stop_message: Some(format!(
                        "Stopped after 3 `{}` failures in category `{category}` across this task. Last error: {error}",
                        call.name
                    )),
                });
            }
        }

        Ok(RecordedToolFeedback {
            result,
            stop_message: None,
        })
    }

    fn finish_response(&mut self, text: String, ask_user: bool) -> AgentResult<String> {
        self.record_if_active(if ask_user {
            EventKind::AskUser { text: text.clone() }
        } else {
            EventKind::Assistant { text: text.clone() }
        })?;
        if let Some(observer) = self.observer.as_mut() {
            observer.on_response(&text);
        }
        Ok(text)
    }

    pub fn handle_command_decision(
        &mut self,
        decision: crate::engine::CommandDecision,
        backend: &mut dyn LlmBackend,
    ) -> AgentResult<String> {
        let pending = self
            .session
            .as_ref()
            .and_then(|session| session.pending_command_approval.clone())
            .ok_or_else(|| AgentError::invalid_state("no command awaiting approval"))?;
        let Some(persisted_turn) = self
            .session
            .as_ref()
            .and_then(|session| session.pending_agent_turn.clone())
        else {
            return self.handle_legacy_command_decision(decision, &pending.tool_call);
        };

        let mut state = AgentTurnState::from_pending(persisted_turn);
        let available_tools = model_visible_tool_names()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let feedback = match decision {
            crate::engine::CommandDecision::Approve => {
                match self.run_tool(&pending.tool_call.tool_name, &pending.tool_call.arguments) {
                    Ok(outcome) => ToolFeedback::success(&pending.tool_call.tool_name, outcome),
                    Err(error) if error.is_cancelled() => return Err(error),
                    Err(error) => {
                        let category = error.tool_failure_category();
                        ToolFeedback::failure(
                            &pending.tool_call.tool_name,
                            error.to_string(),
                            &category,
                            available_tools,
                        )
                    }
                }
            }
            crate::engine::CommandDecision::Reject => ToolFeedback::failure(
                &pending.tool_call.tool_name,
                "the user declined this command".to_owned(),
                "user_denied",
                available_tools,
            ),
        };
        let call_id = if pending.call_id.is_empty() {
            "json-tool-call".to_owned()
        } else {
            pending.call_id.clone()
        };
        let call = ModelToolCall {
            id: call_id,
            name: pending.tool_call.tool_name.clone(),
            arguments: pending.tool_call.arguments.clone(),
        };
        let successful_mutation = feedback.success
            && find_tool_spec(&pending.tool_call.tool_name)
                .is_some_and(|spec| spec.mutates_with(&pending.tool_call.arguments));
        if successful_mutation {
            state.completed_mutations.insert(CallKey::new(
                &pending.tool_call.tool_name,
                &pending.tool_call.arguments,
            ));
        }
        let call_key = CallKey::new(&call.name, &call.arguments);
        let recorded = self.record_tool_feedback(
            &call,
            &call_key,
            feedback,
            &mut state.task,
            &mut state.failure_counts,
            &mut state.aggregate_failure_counts,
        )?;
        let pending_native = self.pending_native_continuation.take();
        let (continuation, mut accumulated_results) = pending_native.map_or_else(
            || (None, Vec::new()),
            |pending| (Some(pending.continuation), pending.results),
        );
        accumulated_results.push(recorded.result);
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| AgentError::invalid_state("no active session"))?;
        session.pending_command_approval = None;
        session.pending_agent_turn = Some(state.to_pending());
        self.record(match decision {
            crate::engine::CommandDecision::Approve => EventKind::CommandApproved,
            crate::engine::CommandDecision::Reject => EventKind::CommandRejected,
        })?;
        if let Some(message) = recorded.stop_message {
            self.clear_pending_agent_turn()?;
            return self.finish_response(message, false);
        }

        let remaining_calls = pending
            .remaining_tool_calls
            .into_iter()
            .map(ModelToolCall::from)
            .collect::<Vec<_>>();
        let processed = if remaining_calls.is_empty() {
            ProcessedCalls::Continue(Vec::new())
        } else {
            self.process_tool_calls(
                remaining_calls,
                &mut state.task,
                &mut state.failure_counts,
                &mut state.aggregate_failure_counts,
                &mut state.completed_mutations,
            )?
        };
        match processed {
            ProcessedCalls::Continue(results) => {
                accumulated_results.extend(results);
                let native_turn = continuation.map(|continuation| NativeTurn {
                    continuation,
                    results: accumulated_results,
                });
                self.run_turn(backend, state, native_turn)
            }
            ProcessedCalls::AwaitingCommandApproval {
                call,
                reason,
                completed_results,
                remaining_calls,
            } => {
                accumulated_results.extend(completed_results);
                self.pending_native_continuation =
                    continuation.map(|continuation| crate::engine::PendingNativeContinuation {
                        continuation,
                        results: accumulated_results,
                    });
                self.store_command_approval(
                    ToolCallDraft {
                        tool_name: call.name,
                        arguments: call.arguments,
                    },
                    call.id,
                    remaining_calls
                        .into_iter()
                        .map(PendingToolCall::from)
                        .collect(),
                    reason,
                    state.into_pending(),
                )?;
                self.finish_response(self.pending_command_approval_summary(), false)
            }
            ProcessedCalls::Planned => {
                self.clear_pending_agent_turn()?;
                self.finish_response(self.pending_plan_summary(), false)
            }
            ProcessedCalls::RepeatedFailure(message) => {
                self.clear_pending_agent_turn()?;
                self.finish_response(message, false)
            }
        }
    }

    fn handle_legacy_command_decision(
        &mut self,
        decision: crate::engine::CommandDecision,
        call: &ToolCallDraft,
    ) -> AgentResult<String> {
        let result = match decision {
            crate::engine::CommandDecision::Approve => {
                render_tool_outcome(&self.run_tool(&call.tool_name, &call.arguments)?)
            }
            crate::engine::CommandDecision::Reject => "Rejected pending command.".to_owned(),
        };
        self.session
            .as_mut()
            .ok_or_else(|| AgentError::invalid_state("no active session"))?
            .pending_command_approval = None;
        self.record(match decision {
            crate::engine::CommandDecision::Approve => EventKind::CommandApproved,
            crate::engine::CommandDecision::Reject => EventKind::CommandRejected,
        })?;
        self.record_if_active(EventKind::Assistant {
            text: result.clone(),
        })?;
        Ok(result)
    }

    fn clear_pending_agent_turn(&mut self) -> AgentResult<()> {
        let Some(session) = self.session.as_mut() else {
            return Ok(());
        };
        if session.pending_agent_turn.take().is_some() {
            self.save()?;
        }
        Ok(())
    }

    fn take_legacy_pending_action(&mut self) -> AgentResult<Option<String>> {
        let Some(session) = self.session.as_mut() else {
            return Ok(None);
        };
        let pending = session.pending_action.take().map(|pending| {
            format!(
                "intent: {}\nrequest: {}\nTreat the current message as a continuation only when that preserves the older request's meaning.",
                pending.intent, pending.request
            )
        });
        if pending.is_some() {
            self.save()?;
        }
        Ok(pending)
    }

    fn dialogue_context_summary(&self, limit: usize) -> Option<String> {
        let session = self.session.as_ref()?;
        let mut lines = session
            .events
            .iter()
            .rev()
            .filter_map(|event| match event.tag() {
                crate::session::EventTag::User => {
                    Some(format!("User: {}", truncate_text(&event.text, 240)))
                }
                crate::session::EventTag::Assistant | crate::session::EventTag::AskUser => {
                    Some(format!("Assistant: {}", truncate_text(&event.text, 240)))
                }
                _ => None,
            })
            .take(limit)
            .collect::<Vec<_>>();
        lines.reverse();
        (!lines.is_empty()).then(|| lines.join("\n"))
    }

    fn effective_defaults_summary(&self) -> AgentResult<String> {
        let settings =
            ProfileCoordinator::new(&self.project_root, self.session.as_ref()).active_settings()?;
        let source_language = normalize_language(&settings.translation.source_language, true)
            .map_err(|error| AgentError::InvalidInput {
                message: error.to_string(),
            })?;
        let target_language = normalize_language(&settings.translation.target_language, false)
            .map_err(|error| AgentError::InvalidInput {
                message: error.to_string(),
            })?;
        let output_format = settings.output.format.as_deref().unwrap_or("source");
        let transcription_model = settings
            .transcription
            .model
            .as_deref()
            .unwrap_or("auto-select-installed");
        Ok(format!(
            "translation: source={}, target={}, provider={}, model={}, format={}, bilingual={}, bilingual_order={}, preserve_names={}, preserve_source_container={}, dry_run={}\ntranscription: provider=whisper_cpp, model={}, language=Auto, format=srt",
            source_language,
            target_language,
            settings.backend.id,
            settings.backend.model,
            output_format,
            settings.output.bilingual,
            settings.output.bilingual_order.as_str(),
            settings.translation.preserve_names,
            settings.output.preserve_source_container,
            settings.translation.dry_run,
            transcription_model,
        ))
    }

    fn is_in_plan_mode(&self) -> bool {
        self.session
            .as_ref()
            .is_some_and(|session| session.mode == crate::session::SessionMode::Plan)
    }

    pub(crate) fn record_if_active(&mut self, kind: EventKind) -> AgentResult<()> {
        if self.session.is_some() {
            self.record(kind)?;
        }
        Ok(())
    }

    pub(crate) fn run_tool(
        &mut self,
        name: &str,
        args: &JsonValue,
    ) -> AgentResult<subbake_core::AgentToolOutcome> {
        crate::tool_runner::ToolRunner::run(self, name, args)
    }
}

struct AgentTurnState {
    input: String,
    dialogue: Option<String>,
    legacy_pending: Option<String>,
    effective_defaults: String,
    task: AgentTaskLoop,
    legacy_mode: bool,
    failure_counts: HashMap<FailureKey, usize>,
    aggregate_failure_counts: HashMap<AggregateFailureKey, usize>,
    completed_mutations: HashSet<CallKey>,
    steps_used: usize,
}

impl AgentTurnState {
    fn into_pending(self) -> PendingAgentTurn {
        self.to_pending()
    }

    fn to_pending(&self) -> PendingAgentTurn {
        PendingAgentTurn {
            input: self.input.clone(),
            dialogue: self.dialogue.clone(),
            legacy_pending: self.legacy_pending.clone(),
            effective_defaults: self.effective_defaults.clone(),
            exchanges: self
                .task
                .exchanges
                .iter()
                .map(|exchange| PendingToolExchange {
                    call_id: exchange.call_id.clone(),
                    tool_name: exchange.name.clone(),
                    arguments: exchange.arguments.clone(),
                    feedback: exchange.feedback.clone(),
                    is_error: exchange.is_error,
                })
                .collect(),
            completed_mutations: self
                .completed_mutations
                .iter()
                .map(|call| ToolCallDraft {
                    tool_name: call.name.clone(),
                    arguments: serde_json::from_str(&call.arguments)
                        .unwrap_or_else(|_| JsonValue::Object(Default::default())),
                })
                .collect(),
            failure_counts: self
                .failure_counts
                .iter()
                .map(|(failure, count)| PendingFailureCount {
                    tool_call: ToolCallDraft {
                        tool_name: failure.call.name.clone(),
                        arguments: serde_json::from_str(&failure.call.arguments)
                            .unwrap_or_else(|_| JsonValue::Object(Default::default())),
                    },
                    category: failure.category.clone(),
                    count: *count,
                })
                .collect(),
            aggregate_failure_counts: self
                .aggregate_failure_counts
                .iter()
                .map(|(failure, count)| PendingAggregateFailureCount {
                    tool_name: failure.tool_name.clone(),
                    category: failure.category.clone(),
                    count: *count,
                })
                .collect(),
            steps_used: self.steps_used,
            legacy_mode: self.legacy_mode,
        }
    }

    fn from_pending(pending: PendingAgentTurn) -> Self {
        Self {
            input: pending.input,
            dialogue: pending.dialogue,
            legacy_pending: pending.legacy_pending,
            effective_defaults: pending.effective_defaults,
            task: AgentTaskLoop {
                exchanges: pending
                    .exchanges
                    .into_iter()
                    .map(|exchange| ToolExchange {
                        call_id: exchange.call_id,
                        name: exchange.tool_name,
                        arguments: exchange.arguments,
                        feedback: exchange.feedback,
                        is_error: exchange.is_error,
                    })
                    .collect(),
            },
            legacy_mode: pending.legacy_mode,
            failure_counts: pending
                .failure_counts
                .into_iter()
                .map(|failure| {
                    (
                        FailureKey {
                            call: CallKey::new(
                                &failure.tool_call.tool_name,
                                &failure.tool_call.arguments,
                            ),
                            category: failure.category,
                        },
                        failure.count,
                    )
                })
                .collect(),
            aggregate_failure_counts: pending
                .aggregate_failure_counts
                .into_iter()
                .map(|failure| {
                    (
                        AggregateFailureKey {
                            tool_name: failure.tool_name,
                            category: failure.category,
                        },
                        failure.count,
                    )
                })
                .collect(),
            completed_mutations: pending
                .completed_mutations
                .into_iter()
                .map(|call| CallKey::new(&call.tool_name, &call.arguments))
                .collect(),
            steps_used: pending.steps_used,
        }
    }
}

fn native_decision(response: GenerationResponse) -> AgentResult<Decision> {
    let GenerationResponse {
        content,
        tool_calls,
        continuation,
        ..
    } = response;
    let text = match content {
        GenerationContent::Empty => String::new(),
        GenerationContent::Text(text) => text,
        GenerationContent::Json(value) => value.to_string(),
    };
    if tool_calls.is_empty() {
        Ok(Decision::response(text))
    } else if continuation.is_none() {
        Err(AgentError::InvalidState {
            message: "native tool calls are missing provider continuation state".to_owned(),
        })
    } else {
        Ok(Decision::native_calls(text, tool_calls, continuation))
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

fn validation_category(error: &ToolValidationError) -> &'static str {
    match error {
        ToolValidationError::UnknownTool { .. } => "unknown_tool",
        ToolValidationError::ArgumentsNotObject { .. }
        | ToolValidationError::UnexpectedArgument { .. }
        | ToolValidationError::MissingArgument { .. }
        | ToolValidationError::WrongArgumentType { .. }
        | ToolValidationError::InvalidArgument { .. } => "invalid_arguments",
    }
}

fn nonempty_response(text: String) -> String {
    if text.trim().is_empty() {
        "The model returned no final response.".to_owned()
    } else {
        text
    }
}

fn nonempty_question(text: String) -> String {
    if text.trim().is_empty() {
        "What path or subtitle should I use?".to_owned()
    } else {
        text
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CallKey {
    name: String,
    arguments: String,
}

impl CallKey {
    fn new(name: &str, arguments: &JsonValue) -> Self {
        Self {
            name: name.to_owned(),
            arguments: canonical_json(arguments).to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FailureKey {
    call: CallKey,
    category: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AggregateFailureKey {
    tool_name: String,
    category: String,
}

struct RecordedToolFeedback {
    result: ModelToolResult,
    stop_message: Option<String>,
}

enum ProcessedCalls {
    Continue(Vec<ModelToolResult>),
    Planned,
    AwaitingCommandApproval {
        call: ModelToolCall,
        reason: String,
        completed_results: Vec<ModelToolResult>,
        remaining_calls: Vec<ModelToolCall>,
    },
    RepeatedFailure(String),
}

impl From<ModelToolCall> for PendingToolCall {
    fn from(call: ModelToolCall) -> Self {
        Self {
            call_id: call.id,
            tool_name: call.name,
            arguments: call.arguments,
        }
    }
}

impl From<PendingToolCall> for ModelToolCall {
    fn from(call: PendingToolCall) -> Self {
        Self {
            id: call.call_id,
            name: call.tool_name,
            arguments: call.arguments,
        }
    }
}

fn is_aggregate_failure_category(category: &str) -> bool {
    matches!(
        category.split(':').next().unwrap_or(category),
        "authentication"
            | "rate_limited"
            | "timeout"
            | "transport"
            | "service_rejected"
            | "child_process"
            | "external_io"
    )
}

fn canonical_json(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Object(object) => {
            let mut entries = object.iter().collect::<Vec<_>>();
            entries.sort_by_key(|(key, _)| *key);
            JsonValue::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key.clone(), canonical_json(value)))
                    .collect(),
            )
        }
        JsonValue::Array(values) => JsonValue::Array(values.iter().map(canonical_json).collect()),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use subbake_core::ports::{ModelToolCall, ToolContinuation};

    use super::*;

    struct JsonSequenceBackend {
        decisions: VecDeque<JsonValue>,
        prompts: Vec<Vec<ChatMessage>>,
    }

    impl LlmBackend for JsonSequenceBackend {
        fn provider_name(&self) -> &str {
            "json-test"
        }

        fn model_name(&self) -> &str {
            "json-test"
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &subbake_core::CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            cancellation.check().map_err(LlmCallError::from)?;
            let GenerationInput::Messages(messages) = request.input else {
                return Err(LlmCallError::ContinuationMismatch(
                    "json test backend cannot continue".to_owned(),
                ));
            };
            self.prompts.push(messages);
            Ok(GenerationResponse::json(
                self.decisions
                    .pop_front()
                    .unwrap_or_else(|| json!({"action":"respond","text":"done"})),
                Usage::default(),
            ))
        }
    }

    enum NativeStep {
        Calls(Vec<ModelToolCall>),
        Text(String),
    }

    struct NativeSequenceBackend {
        steps: VecDeque<NativeStep>,
        definitions: Vec<Vec<String>>,
        continued_results: Vec<Vec<ModelToolResult>>,
    }

    impl LlmBackend for NativeSequenceBackend {
        fn provider_name(&self) -> &str {
            "native-test"
        }

        fn model_name(&self) -> &str {
            "native-test"
        }

        fn native_tool_support(&self) -> NativeToolSupport {
            NativeToolSupport::Supported
        }

        fn execute(
            &mut self,
            request: GenerationRequest,
            cancellation: &subbake_core::CancellationGuard,
        ) -> Result<GenerationResponse, LlmCallError> {
            cancellation.check().map_err(LlmCallError::from)?;
            self.definitions.push(
                request
                    .tools
                    .as_ref()
                    .map(|tools| {
                        tools
                            .definitions
                            .iter()
                            .map(|definition| definition.name.clone())
                            .collect()
                    })
                    .unwrap_or_default(),
            );
            if let GenerationInput::Continue { tool_results, .. } = request.input {
                self.continued_results.push(tool_results);
            }
            match self
                .steps
                .pop_front()
                .unwrap_or_else(|| NativeStep::Text("done".to_owned()))
            {
                NativeStep::Calls(tool_calls) => Ok(GenerationResponse {
                    content: GenerationContent::Empty,
                    tool_calls,
                    continuation: Some(ToolContinuation::new("native-test", ())),
                    usage: Usage::default(),
                }),
                NativeStep::Text(text) => Ok(GenerationResponse {
                    content: GenerationContent::Text(text),
                    tool_calls: Vec::new(),
                    continuation: None,
                    usage: Usage::default(),
                }),
            }
        }
    }

    #[test]
    fn discovery_never_closes_translation_tools_in_json_loop() {
        let root = temp_root("stable-json-tools");
        std::fs::create_dir_all(&root).expect("root");
        std::fs::write(
            root.join("sample.srt"),
            "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
        )
        .expect("subtitle");
        let mut engine = active_engine(root.clone());
        let mut backend = JsonSequenceBackend {
            decisions: VecDeque::from([
                json!({"action":"tool_call","tool_name":"list_files","arguments":{"path":"."}}),
                json!({"action":"tool_call","tool_name":"candidate_subtitles","arguments":{"path":"."}}),
                json!({"action":"tool_call","tool_name":"translate_series","arguments":{"path":"."}}),
                json!({"action":"respond","text":"目录字幕已翻译。"}),
            ]),
            prompts: Vec::new(),
        };

        let response = engine
            .run_line("翻译目录下的 srt 文件", &mut backend)
            .expect("run");
        assert_eq!(response, "目录字幕已翻译。");
        assert!(root.join("sample.translated.srt").exists());
        assert!(backend.prompts.iter().all(|messages| {
            messages[0].content.contains("- translate_series:")
                && messages[0].content.contains("- candidate_subtitles:")
        }));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn whisper_status_is_an_intermediate_observation_not_a_pending_plan() {
        let root = temp_root("whisper-status-continuation");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let mut backend = JsonSequenceBackend {
            decisions: VecDeque::from([
                json!({"action":"tool_call","tool_name":"manage_whisper","arguments":{"action":"status"}}),
                json!({"action":"tool_call","tool_name":"list_files","arguments":{"path":"."}}),
                json!({"action":"respond","text":"Whisper checked; continued the task."}),
            ]),
            prompts: Vec::new(),
        };

        let response = engine
            .run_line("check Whisper, then inspect the directory", &mut backend)
            .expect("run");

        assert_eq!(response, "Whisper checked; continued the task.");
        assert!(!engine.has_pending_plan());
        assert_eq!(backend.prompts.len(), 3);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn approved_json_command_continues_the_same_task_without_new_user_input() {
        let root = temp_root("json-command-approval-continuation");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let mut backend = JsonSequenceBackend {
            decisions: VecDeque::from([
                json!({
                    "action":"tool_call",
                    "tool_name":"run_command",
                    "arguments":{"command":"printf approved-output","network":true}
                }),
                json!({"action":"respond","text":"continued after approval"}),
            ]),
            prompts: Vec::new(),
        };

        let pending = engine
            .run_line("inspect, then continue", &mut backend)
            .expect("request approval");
        assert!(pending.contains("Command awaiting approval"));
        assert!(engine.has_pending_command_approval());

        let response = engine
            .handle_command_decision(crate::engine::CommandDecision::Approve, &mut backend)
            .expect("approve and continue");

        assert_eq!(response, "continued after approval");
        assert!(!engine.has_pending_command_approval());
        assert!(
            engine
                .session
                .as_ref()
                .expect("session")
                .pending_agent_turn
                .is_none()
        );
        assert!(backend.prompts[1][1].content.contains("approved-output"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn automatic_command_approval_continues_without_opening_a_picker() {
        let root = temp_root("automatic-command-approval");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        engine.set_runtime_policy(
            crate::engine::AgentRuntimePolicy::new(24, true).expect("runtime policy"),
        );
        let mut backend = JsonSequenceBackend {
            decisions: VecDeque::from([
                json!({
                    "action":"tool_call",
                    "tool_name":"run_command",
                    "arguments":{"command":"printf automatic-output","network":true}
                }),
                json!({"action":"respond","text":"continued automatically"}),
            ]),
            prompts: Vec::new(),
        };

        let response = engine
            .run_line("inspect automatically", &mut backend)
            .expect("auto approve and continue");

        assert_eq!(response, "continued automatically");
        assert!(!engine.has_pending_command_approval());
        assert!(backend.prompts[1][1].content.contains("automatic-output"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn automatic_command_approval_keeps_hard_denials_and_plan_approval() {
        let root = temp_root("automatic-command-policy-boundaries");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        engine.set_runtime_policy(
            crate::engine::AgentRuntimePolicy::new(24, true).expect("runtime policy"),
        );
        let mut denied_backend = JsonSequenceBackend {
            decisions: VecDeque::from([
                json!({
                    "action":"tool_call",
                    "tool_name":"run_command",
                    "arguments":{"command":"sudo id"}
                }),
                json!({"action":"respond","text":"hard denial observed"}),
            ]),
            prompts: Vec::new(),
        };

        let response = engine
            .run_line("try a forbidden command", &mut denied_backend)
            .expect("hard denial is model feedback");
        assert_eq!(response, "hard denial observed");
        assert!(
            denied_backend.prompts[1][1]
                .content
                .contains("sandbox boundaries")
        );

        engine.set_plan_mode(true).expect("plan mode");
        let mut planned_backend = JsonSequenceBackend {
            decisions: VecDeque::from([json!({
                "action":"tool_call",
                "tool_name":"run_command",
                "arguments":{
                    "command":"printf artifact > $SUBBAKE_OUTPUT_ARTIFACT",
                    "outputs":{"artifact":"artifact.txt"}
                }
            })]),
            prompts: Vec::new(),
        };
        engine
            .run_line("create an artifact", &mut planned_backend)
            .expect("plan approval");
        assert!(engine.has_pending_plan());
        assert!(!root.join("artifact.txt").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn approved_native_command_returns_the_original_call_result_to_the_model() {
        let root = temp_root("native-command-approval-continuation");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let mut backend = NativeSequenceBackend {
            steps: VecDeque::from([
                NativeStep::Calls(vec![ModelToolCall {
                    id: "call-approved".to_owned(),
                    name: "run_command".to_owned(),
                    arguments: json!({"command":"printf native-output","network":true}),
                }]),
                NativeStep::Text("native continued".to_owned()),
            ]),
            definitions: Vec::new(),
            continued_results: Vec::new(),
        };

        engine
            .run_line("inspect natively", &mut backend)
            .expect("request approval");
        let response = engine
            .handle_command_decision(crate::engine::CommandDecision::Approve, &mut backend)
            .expect("approve and continue");

        assert_eq!(response, "native continued");
        assert_eq!(backend.continued_results.len(), 1);
        assert_eq!(backend.continued_results[0][0].id, "call-approved");
        assert!(
            !backend.continued_results[0][0].is_error,
            "{}",
            backend.continued_results[0][0].output
        );
        assert!(
            backend.continued_results[0][0]
                .output
                .contains("native-output")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_batch_approval_returns_every_tool_result_exactly_once() {
        let root = temp_root("native-batch-command-approval");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let mut backend = NativeSequenceBackend {
            steps: VecDeque::from([
                NativeStep::Calls(vec![
                    ModelToolCall {
                        id: "before".to_owned(),
                        name: "list_files".to_owned(),
                        arguments: json!({"path":"."}),
                    },
                    ModelToolCall {
                        id: "approved".to_owned(),
                        name: "run_command".to_owned(),
                        arguments: json!({"command":"printf approved","network":true}),
                    },
                    ModelToolCall {
                        id: "after".to_owned(),
                        name: "search_files".to_owned(),
                        arguments: json!({"pattern":"*.srt"}),
                    },
                ]),
                NativeStep::Text("batch continued".to_owned()),
            ]),
            definitions: Vec::new(),
            continued_results: Vec::new(),
        };

        engine
            .run_line("inspect with approval", &mut backend)
            .expect("request approval");
        let pending = engine
            .session
            .as_ref()
            .and_then(|session| session.pending_command_approval.as_ref())
            .expect("pending command");
        assert_eq!(pending.remaining_tool_calls.len(), 1);
        assert_eq!(pending.remaining_tool_calls[0].call_id, "after");

        let response = engine
            .handle_command_decision(crate::engine::CommandDecision::Approve, &mut backend)
            .expect("approve batch");

        assert_eq!(response, "batch continued");
        let ids = backend.continued_results[0]
            .iter()
            .map(|result| result.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, ["before", "approved", "after"]);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_batch_handles_multiple_command_approvals_sequentially() {
        let root = temp_root("native-batch-multiple-approvals");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let mut backend = NativeSequenceBackend {
            steps: VecDeque::from([
                NativeStep::Calls(vec![
                    ModelToolCall {
                        id: "first".to_owned(),
                        name: "run_command".to_owned(),
                        arguments: json!({"command":"printf first","network":true}),
                    },
                    ModelToolCall {
                        id: "second".to_owned(),
                        name: "run_command".to_owned(),
                        arguments: json!({"command":"printf second","network":true}),
                    },
                    ModelToolCall {
                        id: "files".to_owned(),
                        name: "list_files".to_owned(),
                        arguments: json!({"path":"."}),
                    },
                ]),
                NativeStep::Text("all handled".to_owned()),
            ]),
            definitions: Vec::new(),
            continued_results: Vec::new(),
        };

        engine
            .run_line("inspect", &mut backend)
            .expect("first approval");
        let second_pending = engine
            .handle_command_decision(crate::engine::CommandDecision::Approve, &mut backend)
            .expect("approve first");
        assert!(second_pending.contains("Command awaiting approval"));
        assert_eq!(backend.continued_results.len(), 0);

        let response = engine
            .handle_command_decision(crate::engine::CommandDecision::Reject, &mut backend)
            .expect("reject second");
        assert_eq!(response, "all handled");
        assert_eq!(backend.continued_results[0].len(), 3);
        assert!(!backend.continued_results[0][0].is_error);
        assert!(backend.continued_results[0][1].is_error);
        assert!(!backend.continued_results[0][2].is_error);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn resumed_native_batch_executes_persisted_remaining_calls_without_continuation() {
        let root = temp_root("resumed-native-batch-approval");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let mut first_backend = NativeSequenceBackend {
            steps: VecDeque::from([NativeStep::Calls(vec![
                ModelToolCall {
                    id: "command".to_owned(),
                    name: "run_command".to_owned(),
                    arguments: json!({"command":"printf resumed-batch","network":true}),
                },
                ModelToolCall {
                    id: "files".to_owned(),
                    name: "list_files".to_owned(),
                    arguments: json!({"path":"."}),
                },
            ])]),
            definitions: Vec::new(),
            continued_results: Vec::new(),
        };
        engine
            .run_line("inspect across restart", &mut first_backend)
            .expect("request approval");
        let session_id = engine.session.as_ref().expect("session").id.clone();

        let mut resumed = AgentEngine::new(root.clone());
        resumed
            .resume_session(Some(&session_id))
            .expect("resume session");
        let mut resumed_backend = NativeSequenceBackend {
            steps: VecDeque::from([NativeStep::Text("resumed batch completed".to_owned())]),
            definitions: Vec::new(),
            continued_results: Vec::new(),
        };
        let response = resumed
            .handle_command_decision(
                crate::engine::CommandDecision::Approve,
                &mut resumed_backend,
            )
            .expect("approve resumed batch");

        assert_eq!(response, "resumed batch completed");
        assert!(resumed_backend.continued_results.is_empty());
        assert!(
            resumed
                .session_events()
                .iter()
                .any(|event| { event.kind == "tool_call" && event.text == "list_files" })
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn aggregate_external_failures_stop_distinct_paths_after_three_attempts() {
        let root = temp_root("aggregate-tool-failure-breaker");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let mut task = AgentTaskLoop::default();
        let mut exact = HashMap::new();
        let mut aggregate = HashMap::new();
        let mut last_stop = None;

        for index in 1..=3 {
            let call = ModelToolCall {
                id: format!("call-{index}"),
                name: "translate_file".to_owned(),
                arguments: json!({"path":format!("episode-{index}.srt")}),
            };
            let recorded = engine
                .record_tool_feedback(
                    &call,
                    &CallKey::new(&call.name, &call.arguments),
                    ToolFeedback::failure(
                        &call.name,
                        "provider timed out".to_owned(),
                        "timeout",
                        Vec::new(),
                    ),
                    &mut task,
                    &mut exact,
                    &mut aggregate,
                )
                .expect("record failure");
            last_stop = recorded.stop_message;
        }

        assert!(last_stop.expect("breaker message").contains("after 3"));
        assert_eq!(
            engine
                .session_events()
                .iter()
                .filter(|event| event.kind == "tool_failure")
                .count(),
            3
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn rejected_command_is_feedback_and_the_model_can_recover() {
        let root = temp_root("rejected-command-continuation");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let mut backend = NativeSequenceBackend {
            steps: VecDeque::from([
                NativeStep::Calls(vec![ModelToolCall {
                    id: "call-rejected".to_owned(),
                    name: "run_command".to_owned(),
                    arguments: json!({"command":"printf must-not-run","network":true}),
                }]),
                NativeStep::Text("used a safer alternative".to_owned()),
            ]),
            definitions: Vec::new(),
            continued_results: Vec::new(),
        };

        engine
            .run_line("inspect safely", &mut backend)
            .expect("request approval");
        let response = engine
            .handle_command_decision(crate::engine::CommandDecision::Reject, &mut backend)
            .expect("reject and continue");

        assert_eq!(response, "used a safer alternative");
        assert!(backend.continued_results[0][0].is_error);
        assert!(
            backend.continued_results[0][0]
                .output
                .contains("user_denied")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn resumed_session_approves_and_replays_the_persisted_turn() {
        let root = temp_root("resumed-command-approval");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let mut first_backend = JsonSequenceBackend {
            decisions: VecDeque::from([json!({
                "action":"tool_call",
                "tool_name":"run_command",
                "arguments":{"command":"printf resumed-output","network":true}
            })]),
            prompts: Vec::new(),
        };
        engine
            .run_line("inspect across restart", &mut first_backend)
            .expect("request approval");
        let session_id = engine.session.as_ref().expect("session").id.clone();

        let mut resumed = AgentEngine::new(root.clone());
        resumed
            .resume_session(Some(&session_id))
            .expect("resume session");
        let mut resumed_backend = JsonSequenceBackend {
            decisions: VecDeque::from([json!({
                "action":"respond",
                "text":"resumed and continued"
            })]),
            prompts: Vec::new(),
        };
        let response = resumed
            .handle_command_decision(
                crate::engine::CommandDecision::Approve,
                &mut resumed_backend,
            )
            .expect("approve resumed command");

        assert_eq!(response, "resumed and continued");
        assert!(
            resumed_backend.prompts[0][1]
                .content
                .contains("resumed-output")
        );
        assert!(
            resumed_backend.prompts[0][1]
                .content
                .contains("inspect across restart")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn japanese_request_is_executed_with_normalized_override_and_structured_facts() {
        let root = temp_root("japanese-override");
        std::fs::create_dir_all(&root).expect("root");
        std::fs::write(
            root.join("sample.srt"),
            "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
        )
        .expect("subtitle");
        let mut engine = active_engine(root.clone());
        let mut backend = JsonSequenceBackend {
            decisions: VecDeque::from([
                json!({
                    "action":"tool_call",
                    "tool_name":"translate_file",
                    "arguments":{"path":"sample.srt","target_language":"Japanese"}
                }),
                json!({"action":"respond","text":"Translated to Japanese."}),
            ]),
            prompts: Vec::new(),
        };

        let response = engine
            .run_line("translate sample.srt to Japanese", &mut backend)
            .expect("run");

        assert_eq!(response, "Translated to Japanese.");
        assert!(root.join("sample.ja.translated.srt").exists());
        let result_context = &backend.prompts[1][1].content;
        assert!(result_context.contains(r#""target_language":"ja""#));
        assert!(result_context.contains("sample.ja.translated.srt"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn omitted_override_reports_actual_profile_default_not_user_intent() {
        let root = temp_root("omitted-override");
        std::fs::create_dir_all(&root).expect("root");
        std::fs::write(root.join("sample.txt"), "hello\n").expect("subtitle");
        let mut engine = active_engine(root.clone());
        let mut backend = JsonSequenceBackend {
            decisions: VecDeque::from([
                json!({
                    "action":"tool_call",
                    "tool_name":"translate_file",
                    "arguments":{"path":"sample.txt"}
                }),
                json!({"action":"respond","text":"The tool used the profile default."}),
            ]),
            prompts: Vec::new(),
        };

        engine
            .run_line("translate sample.txt to Japanese", &mut backend)
            .expect("run");

        let result_context = &backend.prompts[1][1].content;
        assert!(result_context.contains(r#""target_language":"zh-Hans""#));
        assert!(!result_context.contains(r#""target_language":"ja""#));
        assert!(root.join("sample.translated.txt").exists());
        assert!(!root.join("sample.ja.translated.txt").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn unknown_tool_result_lists_real_tools_and_loop_recovers() {
        let root = temp_root("unknown-recovery");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let mut backend = JsonSequenceBackend {
            decisions: VecDeque::from([
                json!({"action":"tool_call","tool_name":"list_tools","arguments":{}}),
                json!({"action":"tool_call","tool_name":"list_files","arguments":{"path":"."}}),
                json!({"action":"respond","text":"No files found."}),
            ]),
            prompts: Vec::new(),
        };

        let response = engine.run_line("inspect", &mut backend).expect("run");
        assert_eq!(response, "No files found.");
        let second_user = &backend.prompts[1][1].content;
        assert!(second_user.contains(r#""success":false"#));
        assert!(second_user.contains(r#""available_tools""#));
        assert!(second_user.contains("list_files"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn same_failed_call_twice_stops_without_a_third_model_turn() {
        let root = temp_root("repeat-failure");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let repeated = json!({"action":"tool_call","tool_name":"read_file","arguments":{"path":"missing.txt"}});
        let mut backend = JsonSequenceBackend {
            decisions: VecDeque::from([repeated.clone(), repeated]),
            prompts: Vec::new(),
        };

        let response = engine.run_line("read missing", &mut backend).expect("run");
        assert!(response.contains("repeated"));
        assert_eq!(backend.prompts.len(), 2);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn successful_mutation_is_not_executed_twice_in_one_task() {
        let root = temp_root("duplicate-mutation");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let patch = "*** Begin Patch\n*** Add File: note.txt\n+hello\n*** End Patch";
        let call =
            json!({"action":"tool_call","tool_name":"apply_patch","arguments":{"patch":patch}});
        let mut backend = JsonSequenceBackend {
            decisions: VecDeque::from([
                call.clone(),
                call,
                json!({"action":"respond","text":"Created note.txt once."}),
            ]),
            prompts: Vec::new(),
        };

        let response = engine.run_line("create note", &mut backend).expect("run");
        assert_eq!(response, "Created note.txt once.");
        assert_eq!(
            std::fs::read_to_string(root.join("note.txt")).expect("note"),
            "hello\n"
        );
        let file_events = engine
            .session_events()
            .into_iter()
            .filter(|event| event.kind == "file_operation")
            .count();
        assert_eq!(file_events, 1);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_loop_keeps_definitions_stable_and_summarizes_mutation() {
        let root = temp_root("stable-native-tools");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let mut backend = NativeSequenceBackend {
            steps: VecDeque::from([
                NativeStep::Calls(vec![ModelToolCall {
                    id: "patch".to_owned(),
                    name: "apply_patch".to_owned(),
                    arguments: json!({
                        "patch":"*** Begin Patch\n*** Add File: native.txt\n+ok\n*** End Patch"
                    }),
                }]),
                NativeStep::Text("Created native.txt.".to_owned()),
            ]),
            definitions: Vec::new(),
            continued_results: Vec::new(),
        };

        let response = engine.run_line("create native", &mut backend).expect("run");
        assert_eq!(response, "Created native.txt.");
        assert_eq!(backend.definitions.len(), 2);
        assert_eq!(backend.definitions[0], backend.definitions[1]);
        assert!(!backend.definitions[0].contains(&"create_file".to_owned()));
        assert!(
            backend.continued_results[0][0]
                .output
                .contains(r#""success":true"#)
        );
        assert!(
            backend.continued_results[0][0]
                .output
                .contains(r#""operation":"file""#)
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn native_unknown_tool_is_structured_and_can_recover() {
        let root = temp_root("native-unknown");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let mut backend = NativeSequenceBackend {
            steps: VecDeque::from([
                NativeStep::Calls(vec![ModelToolCall {
                    id: "unknown".to_owned(),
                    name: "list_tools".to_owned(),
                    arguments: json!({}),
                }]),
                NativeStep::Calls(vec![ModelToolCall {
                    id: "files".to_owned(),
                    name: "list_files".to_owned(),
                    arguments: json!({"path":"."}),
                }]),
                NativeStep::Text("Recovered with the registered tools.".to_owned()),
            ]),
            definitions: Vec::new(),
            continued_results: Vec::new(),
        };

        let response = engine.run_line("inspect", &mut backend).expect("run");
        assert_eq!(response, "Recovered with the registered tools.");
        assert!(backend.continued_results[0][0].is_error);
        assert!(
            backend.continued_results[0][0]
                .output
                .contains(r#""available_tools""#)
        );
        assert!(!backend.continued_results[1][0].is_error);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn configured_tool_step_limit_gets_one_tool_disabled_final_turn() {
        let root = temp_root("step-limit");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let max_steps = 3;
        engine.set_runtime_policy(
            crate::engine::AgentRuntimePolicy::new(max_steps, false).expect("runtime policy"),
        );
        let mut decisions = (0..max_steps)
            .map(
                |_| json!({"action":"tool_call","tool_name":"list_files","arguments":{"path":"."}}),
            )
            .collect::<VecDeque<_>>();
        decisions.push_back(json!({"action":"respond","text":"Reached a safe conclusion."}));
        let mut backend = JsonSequenceBackend {
            decisions,
            prompts: Vec::new(),
        };

        let response = engine.run_line("keep checking", &mut backend).expect("run");
        assert_eq!(response, "Reached a safe conclusion.");
        assert_eq!(backend.prompts.len(), max_steps + 1);
        let final_system = &backend.prompts.last().expect("final prompt")[0].content;
        assert!(final_system.contains("No tools are available now"));
        assert!(!final_system.contains("- list_files:"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn plan_mode_intercepts_apply_patch_and_grouped_undo_restores_all_files() {
        let root = temp_root("patch-plan-undo");
        std::fs::create_dir_all(&root).expect("root");
        std::fs::write(root.join("old.txt"), "old\n").expect("old");
        let mut engine = active_engine(root.clone());
        engine.set_plan_mode(true).expect("plan");
        let mut backend = JsonSequenceBackend {
            decisions: VecDeque::from([json!({
                "action":"tool_call",
                "tool_name":"apply_patch",
                "arguments":{"patch":"*** Begin Patch\n*** Add File: new.txt\n+new\n*** Update File: old.txt\n-old\n+changed\n*** End Patch"}
            })]),
            prompts: Vec::new(),
        };

        let response = engine.run_line("change files", &mut backend).expect("run");
        assert!(response.contains("Choose an action below"));
        assert!(!root.join("new.txt").exists());
        engine.approve_plan().expect("approve");
        assert!(root.join("new.txt").exists());
        assert_eq!(
            std::fs::read_to_string(root.join("old.txt")).expect("changed"),
            "changed\n"
        );
        let undo = engine.undo_last().expect("undo");
        assert!(undo.contains("2 operations"));
        assert!(!root.join("new.txt").exists());
        assert_eq!(
            std::fs::read_to_string(root.join("old.txt")).expect("restored"),
            "old\n"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn legacy_pending_action_is_used_once_and_cleared() {
        let root = temp_root("legacy-pending");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        engine.session.as_mut().expect("session").pending_action =
            Some(crate::session::PendingAction {
                intent: "translate".to_owned(),
                request: "translate the selected subtitle".to_owned(),
            });
        let mut backend = JsonSequenceBackend {
            decisions: VecDeque::from([json!({"action":"respond","text":"continued"})]),
            prompts: Vec::new(),
        };

        engine.run_line("movie.srt", &mut backend).expect("run");
        assert!(
            backend.prompts[0][1]
                .content
                .contains("translate the selected subtitle")
        );
        assert!(
            engine
                .session
                .as_ref()
                .expect("session")
                .pending_action
                .is_none()
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn resumed_legacy_plan_executes_hidden_tools_and_persists_each_success() {
        let root = temp_root("legacy-plan");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        engine
            .store_plan(
                "legacy pending plan",
                vec![
                    ToolCallDraft {
                        tool_name: "create_file".to_owned(),
                        arguments: json!({"path":"legacy.txt","content":"created once"}),
                    },
                    ToolCallDraft {
                        tool_name: "rename_path".to_owned(),
                        arguments: json!({"from":"legacy.txt"}),
                    },
                ],
            )
            .expect("store plan");
        let session_id = engine.session.as_ref().expect("session").id.clone();

        let mut resumed = AgentEngine::new(root.clone());
        resumed
            .resume_session(Some(&session_id))
            .expect("resume session");
        resumed
            .approve_plan()
            .expect_err("second legacy call is intentionally invalid");
        assert_eq!(
            std::fs::read_to_string(root.join("legacy.txt")).expect("created"),
            "created once"
        );
        let remaining = &resumed
            .session
            .as_ref()
            .expect("session")
            .pending_plan
            .as_ref()
            .expect("pending")
            .tool_calls;
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].tool_name, "rename_path");

        resumed
            .approve_plan()
            .expect_err("retry must not repeat the completed create");
        assert_eq!(
            std::fs::read_to_string(root.join("legacy.txt")).expect("still once"),
            "created once"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    fn active_engine(root: PathBuf) -> AgentEngine {
        let mut engine = AgentEngine::new(root);
        engine.start_session().expect("session");
        engine
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-{label}-{nanos}"))
    }
}
