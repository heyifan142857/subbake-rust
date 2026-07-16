//! Stable-tool agent task loop.
//!
//! Each user message enters one bounded loop. The model sees the complete
//! model-visible registry throughout the task, and every validation or
//! execution result is fed back until the model returns a final response.

use std::collections::{HashMap, HashSet};

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
use crate::event::{EventKind, ToolCallDraft};
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

pub const AGENT_LOOP_MAX_STEPS: usize = 8;

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

        let mut task = AgentTaskLoop::default();
        let mut native_turn = None;
        let mut legacy_mode = backend.native_tool_support() == NativeToolSupport::Unsupported;
        let mut failure_counts = HashMap::new();
        let mut completed_mutations = HashSet::new();

        for _ in 0..AGENT_LOOP_MAX_STEPS {
            self.check_cancelled()?;
            let decision = self.call_model(
                backend,
                input,
                &task,
                dialogue.as_deref(),
                legacy_pending.as_deref(),
                &mut native_turn,
                &mut legacy_mode,
                true,
                &effective_defaults,
            )?;
            match decision.action {
                DecisionAction::Respond => {
                    return self.finish_response(nonempty_response(decision.text), false);
                }
                DecisionAction::AskUser => {
                    return self.finish_response(nonempty_question(decision.text), true);
                }
                DecisionAction::ToolCalls => {
                    let continuation = decision.continuation;
                    let processed = self.process_tool_calls(
                        decision.calls,
                        &mut task,
                        &mut failure_counts,
                        &mut completed_mutations,
                    )?;
                    match processed {
                        ProcessedCalls::Continue(results) => {
                            native_turn = continuation.map(|continuation| NativeTurn {
                                continuation,
                                results,
                            });
                        }
                        ProcessedCalls::Planned => {
                            return self.finish_response(self.pending_plan_summary(), false);
                        }
                        ProcessedCalls::RepeatedFailure(message) => {
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
            input,
            &task,
            dialogue.as_deref(),
            legacy_pending.as_deref(),
            &mut native_turn,
            &mut legacy_mode,
            false,
            &effective_defaults,
        )?;
        match final_decision.action {
            DecisionAction::Respond => {
                self.finish_response(nonempty_response(final_decision.text), false)
            }
            DecisionAction::AskUser => {
                self.finish_response(nonempty_question(final_decision.text), true)
            }
            DecisionAction::ToolCalls => self.finish_response(
                "I reached the task step limit before completion. Please narrow the request or provide the missing path."
                    .to_owned(),
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
                        spec.mutating && (self.is_in_plan_mode() || spec.requires_approval)
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
        for call in calls {
            self.check_cancelled()?;
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
                    Ok(()) if spec.mutating && completed_mutations.contains(&call_key) => {
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
                            if spec.mutating {
                                completed_mutations.insert(call_key.clone());
                            }
                            let observation = render_tool_outcome(&outcome);
                            if let Some(observer) = self.observer.as_mut() {
                                observer.on_observation(&observation);
                            }
                            ToolFeedback::success(&call.name, outcome)
                        }
                        Err(error) if error.is_cancelled() => return Err(error),
                        Err(error) => ToolFeedback::failure(
                            &call.name,
                            error.to_string(),
                            "execution",
                            available_tools.clone(),
                        ),
                    },
                },
            };
            let is_error = !feedback.success;
            let result_json = feedback.json();
            task.exchanges.push(ToolExchange {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                feedback: feedback.clone(),
            });
            native_results.push(ModelToolResult {
                id: call.id,
                name: call.name.clone(),
                output: result_json,
                is_error,
            });
            if is_error {
                let category = feedback
                    .error_category
                    .as_deref()
                    .unwrap_or("execution")
                    .to_owned();
                let failure = FailureKey {
                    call: call_key,
                    category,
                };
                let count = failure_counts.entry(failure).or_default();
                *count += 1;
                if *count >= 2 {
                    let error = feedback.error.as_deref().unwrap_or("the tool call failed");
                    return Ok(ProcessedCalls::RepeatedFailure(format!(
                        "The model repeated the same failed `{}` call twice: {error}. Please retry with a more specific path or instruction.",
                        call.name
                    )));
                }
            }
        }
        Ok(ProcessedCalls::Continue(native_results))
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
        Ok(format!(
            "translation: source={}, target={}, provider={}, model={}, format={}, bilingual={}, bilingual_order={}, dry_run={}\ntranscription: provider=whisper_api, model=whisper-1, language=Auto, format=srt",
            source_language,
            target_language,
            settings.backend.id,
            settings.backend.model,
            output_format,
            settings.output.bilingual,
            settings.output.bilingual_order.as_str(),
            settings.translation.dry_run,
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
        | ToolValidationError::WrongArgumentType { .. } => "invalid_arguments",
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

enum ProcessedCalls {
    Continue(Vec<ModelToolResult>),
    Planned,
    RepeatedFailure(String),
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
    fn eighth_tool_step_gets_one_tool_disabled_final_turn() {
        let root = temp_root("step-limit");
        std::fs::create_dir_all(&root).expect("root");
        let mut engine = active_engine(root.clone());
        let mut decisions = (0..AGENT_LOOP_MAX_STEPS)
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
        assert_eq!(backend.prompts.len(), AGENT_LOOP_MAX_STEPS + 1);
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
