use subbake_core::ports::ChatMessage;

use super::model::LoopState;
use crate::session::PendingAction;
use crate::tools::{ToolIntent, tool_specs_for_intent};

pub(super) fn build_route_messages(
    input: &str,
    repair_error: Option<&str>,
    pending: Option<&PendingAction>,
    dialogue: Option<String>,
) -> Vec<ChatMessage> {
    let mut system = "You are the semantic router for SubBake. No tools are available in this stage. Understand the current message using recent dialogue. Return one JSON object only. Use {\"route\":\"respond\",\"text\":\"...\"} for conversation or informational questions; {\"route\":\"ask_user\",\"text\":\"...\"} when clarification is essential; or {\"route\":\"act\",\"intent\":\"browse|translate|transcribe|edit|diagnose|file_create|file_append|file_replace|file_rename|file_delete|profile|whisper\",\"restated_request\":\"...\",\"inspect_project\":true|false} for an actionable request. Set inspect_project=true only when shallow project files are needed to ground the action. Route creating a translated or bilingual output from a source subtitle as translate. Route changing an existing .translated or .bilingual subtitle, including combining it with a source subtitle, as edit. A short reply such as 1/yes/好 may continue an action only when recent dialogue clearly establishes that action; otherwise respond conversationally. Never invent a path or action from old tool activity."
        .to_owned();
    if let Some(error) = repair_error {
        system.push_str("\nThe previous route was invalid: ");
        system.push_str(error);
        system.push_str(". Return a corrected route.");
    }
    let mut user = format!("Current message: {input}");
    if let Some(pending) = pending {
        user.push_str("\n\nPending action awaiting a user-supplied value:\n");
        user.push_str(&format!(
            "intent: {}\nrequest: {}\n",
            pending.intent, pending.request
        ));
        user.push_str(
            "If the current message supplies the requested value rather than starting a new task, continue this action with the same intent.",
        );
    }
    if let Some(context) = dialogue {
        user.push_str("\n\nRecent dialogue:\n");
        user.push_str(&context);
    }
    vec![ChatMessage::system(system), ChatMessage::user(user)]
}

pub(super) fn build_native_messages(
    user_input: &str,
    state: &LoopState,
    intent: ToolIntent,
    dialogue: Option<String>,
) -> Vec<ChatMessage> {
    let system = if state.force_no_tools {
        "You are SubBake. Tool exploration is finished. Answer from the supplied dialogue and observations, or ask one specific clarification question. Do not request another tool.".to_owned()
    } else {
        format!(
            "You are SubBake, a subtitle translation assistant. The preliminary routed intent is `{}`; provided tools may safely refine translate into edit or edit into translate when discovered file state requires it. Use only the provided tools and only when they advance the routed action. For an actionable request, do not reply with instructions that the available tools can perform. Preserve subtitle id order, never merge or drop entries. Use edit_subtitle when changing an existing .translated or .bilingual subtitle, including combining it with its source subtitle. Use translate_file for one source subtitle file and translate_series for a directory, batch, or all-files request. Reuse exact paths from tool results. When creating bilingual output through a translation tool, pass bilingual=true. Call one tool at a time unless independent calls are necessary.",
            intent.as_str()
        )
    };
    vec![
        ChatMessage::system(system),
        ChatMessage::user(build_user_context(user_input, state, dialogue)),
    ]
}

pub(super) fn build_decision_messages(
    user_input: &str,
    state: &LoopState,
    repair_error: Option<&str>,
    intent: ToolIntent,
    dialogue: Option<String>,
) -> Vec<ChatMessage> {
    let mut system = String::new();
    system.push_str("You are SubBake, a subtitle translation assistant.\n\n");
    system.push_str(&format!(
        "The preliminary routed intent is `{}`; adjacent subtitle tools may safely refine translate into edit or edit into translate.\n\n",
        intent.as_str()
    ));
    system.push_str("Relevant available tools:\n");
    for spec in if state.force_no_tools {
        Vec::new()
    } else {
        tool_specs_for_intent(intent)
    } {
        system.push_str(&spec.prompt_line());
        if spec.mutating {
            system.push_str(" (mutating)");
        }
        system.push('\n');
    }
    system.push_str("\nDecide the next action. Return JSON with keys:\n");
    system.push_str(r#"{"action": "respond" | "tool_call" | "plan" | "ask_user", "text": "...", "tool_name": "...", "arguments": {...}, "tool_calls": [{"tool_name": "...", "arguments": {...}}]}"#);
    system.push_str("\n- `respond`: reply to the user directly.\n");
    system.push_str("- `tool_call`: invoke a tool. Discovery tools feed observations back; mutating tools execute immediately.\n");
    system.push_str("- `plan`: propose mutating tool calls that must wait for approval.\n");
    system.push_str("- `ask_user`: ask the user for clarification.\n");
    system.push_str("Use tools before asking for project facts a read-only tool can discover.\n");
    system.push_str(
        "For an actionable request, do not respond with instructions the tools can perform.\n",
    );
    system.push_str("If a translation target is omitted, call candidate_subtitles with path `.` before asking for a path.\n");
    system.push_str("Preserve subtitle id order, never merge or drop entries.\n");
    system.push_str("Use translate_file for one source file and translate_series for a directory or all-files request.\n");
    system.push_str(
        "Use edit_subtitle when changing an existing .translated or .bilingual subtitle.\n",
    );
    system
        .push_str("Reuse exact paths from observations. Use path `.` for the current directory.\n");
    system.push_str(
        "When creating bilingual output through a translation tool, pass bilingual=true.\n",
    );
    if state.force_no_tools {
        system.push_str(
            "Tool exploration is finished. Return respond or ask_user now; do not return tool_call or plan.\n",
        );
    }
    if let Some(error) = repair_error {
        system.push_str("\nThe previous decision failed local validation. Correct it without repeating this error:\n");
        system.push_str(error);
        system.push('\n');
    }
    vec![
        ChatMessage::system(system),
        ChatMessage::user(build_user_context(user_input, state, dialogue)),
    ]
}

fn build_user_context(user_input: &str, state: &LoopState, dialogue: Option<String>) -> String {
    let mut user = format!("User: {user_input}\n");
    if let Some(summary) = dialogue {
        user.push_str("\nRecent session context:\n");
        user.push_str(&summary);
        user.push('\n');
    }
    if !state.observations.is_empty() {
        user.push_str("\nObservations from earlier steps:\n");
        for (index, observation) in state.observations.iter().enumerate() {
            user.push_str(&format!(
                "  [{index}] {}: {}\n",
                observation.tool_name, observation.summary
            ));
            for line in observation.text.lines().take(3) {
                user.push_str(&format!("      {}\n", truncate(line, 240)));
            }
        }
    }
    user
}

fn truncate(text: &str, limit: usize) -> String {
    let value = text.chars().take(limit).collect::<String>();
    if text.chars().count() > limit {
        format!("{value}...")
    } else {
        value
    }
}
