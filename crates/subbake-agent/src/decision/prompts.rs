use subbake_core::ports::ChatMessage;

use super::model::AgentTaskLoop;
use crate::tools::model_visible_tool_specs;

pub(super) fn build_native_messages(
    input: &str,
    dialogue: Option<&str>,
    legacy_pending: Option<&str>,
    effective_defaults: &str,
) -> Vec<ChatMessage> {
    vec![
        ChatMessage::system(system_contract(false, false, effective_defaults)),
        ChatMessage::user(user_context(input, dialogue, legacy_pending, None)),
    ]
}

pub(super) fn build_json_messages(
    input: &str,
    loop_state: &AgentTaskLoop,
    dialogue: Option<&str>,
    legacy_pending: Option<&str>,
    tools_enabled: bool,
    repair_error: Option<&str>,
    effective_defaults: &str,
) -> Vec<ChatMessage> {
    vec![
        ChatMessage::system(system_contract(true, !tools_enabled, effective_defaults)),
        ChatMessage::user(user_context(
            input,
            dialogue,
            legacy_pending,
            Some((loop_state, repair_error)),
        )),
    ]
}

fn system_contract(json_fallback: bool, tools_disabled: bool, effective_defaults: &str) -> String {
    let mut system = String::from(
        "You are SubBake, a subtitle workflow assistant. The registered tool list supplied in this turn is the complete list: never invent tools such as `list_tools`, shell, or unregistered aliases. Use project-reading tools to ground uncertain paths, then continue with the appropriate execution tool. Descend into directories returned by list_files until you find the actual requested media; a status or list-models check is only an intermediate observation and never completes a transcription request. For a media-to-bilingual-subtitle request, call transcribe_audio on the media, then call translate_file with bilingual=true on the exact subtitle path returned by transcription. For a request to translate all subtitles in a directory, prefer `translate_series` with `{\"path\":\".\"}` immediately; use `candidate_subtitles` only when the target is genuinely ambiguous. Keep `translate_file` subtitle-only and use `transcribe_audio` explicitly for media. Preserve subtitle IDs and ordering. Use `edit_subtitle` for an existing translated or bilingual subtitle. Reuse exact paths returned by tools. Use `apply_patch` for project text-file edits; its patch format is `*** Begin Patch`, Add/Update/Delete File sections, and `*** End Patch`. Do not produce a plan action: call a mutating tool normally and the runtime will handle any approval. After every successful mutation, use its result to produce a concise natural-language final response instead of echoing raw tool output. Responses are rendered in a terminal: use plain text only, without Markdown headings, tables, bold, code fences, or decorative status icons. For successful translation or transcription, normally respond in one to three short lines with the completed action, output path when available, and processed/skipped counts for a batch. Do not reproduce or summarize individual subtitle entries unless the user explicitly asks for their contents.",
    );
    system.push_str(
        "\nA user request expresses intent, not proof that anything happened. When the user explicitly specifies a supported language, provider, model, format, bilingual mode/order, output path/directory, recursion, or overwrite behavior, pass that value in the tool arguments. Optional call arguments override only that call and never change the session profile. If a requested modifier is unsupported by the registered tool schema, say that it cannot be applied or suggest configuring a profile/using the CLI; never silently ignore it.",
    );
    system.push_str(
        "\nTool outcomes are the only execution evidence. Every completion, language, provider, model, format, path, count, saved-file, cache, resume, skip, unchanged, or dry-run statement in the final response must come directly from a successful structured tool outcome. Never infer execution facts from the user request or from a file-read observation. If requested and actual values differ, correct the call or report the difference. A dry run wrote no output and created no undo event; skipped or unchanged work must not be described as newly generated.",
    );
    system.push_str("\nCurrent effective defaults (secrets omitted):\n");
    system.push_str(effective_defaults);
    if tools_disabled {
        system.push_str(
            "\nThe task step limit has been reached. No tools are available now. Give the best final answer from existing results, or ask one specific question if completion is impossible.",
        );
    } else {
        system.push_str("\nThe stable registered tools for this entire task are:\n");
        for spec in model_visible_tool_specs() {
            system.push_str(&spec.prompt_line());
            if spec.mutating {
                system.push_str(" (mutating)");
            }
            system.push('\n');
        }
    }
    if json_fallback {
        system.push_str(
            "\nReturn exactly one JSON object. Allowed shapes are {\"action\":\"respond\",\"text\":\"...\"}, {\"action\":\"ask_user\",\"text\":\"...\"}, or {\"action\":\"tool_call\",\"tool_name\":\"...\",\"arguments\":{...}}. Never return `plan` or multiple calls.",
        );
        if tools_disabled {
            system.push_str(" Return only `respond` or `ask_user`.");
        }
    }
    system
}

fn user_context(
    input: &str,
    dialogue: Option<&str>,
    legacy_pending: Option<&str>,
    loop_data: Option<(&AgentTaskLoop, Option<&str>)>,
) -> String {
    let mut user = format!("Current user request:\n{input}");
    if let Some(pending) = legacy_pending {
        user.push_str("\n\nOne-time context restored from an older session:\n");
        user.push_str(pending);
    }
    if let Some(dialogue) = dialogue {
        user.push_str("\n\nRecent dialogue:\n");
        user.push_str(dialogue);
    }
    if let Some((loop_state, repair_error)) = loop_data {
        if !loop_state.exchanges.is_empty() {
            user.push_str("\n\nTool calls and structured results from this task:\n");
            for exchange in &loop_state.exchanges {
                user.push_str(&format!(
                    "{} {} => {}\n",
                    exchange.name,
                    exchange.arguments,
                    exchange.feedback.json()
                ));
            }
        }
        if let Some(error) = repair_error {
            user.push_str("\n\nYour previous JSON decision was invalid. Correct it:\n");
            user.push_str(error);
        }
    }
    user
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_declares_complete_tools_and_directory_translation_preference() {
        let messages = build_json_messages(
            "翻译目录下所有字幕",
            &AgentTaskLoop::default(),
            None,
            None,
            true,
            None,
            "translation: source=Auto, target=zh-Hans, provider=mock, model=mock-zh, format=source, bilingual=false, bilingual_order=target_first, dry_run=false\ntranscription: provider=whisper_cpp, model=small, language=Auto, format=srt",
        );
        let system = &messages[0].content;
        assert!(system.contains("complete list"));
        assert!(system.contains("translate_series"));
        assert!(system.contains(r#"{"path":"."}"#));
        assert!(system.contains("never invent tools"));
        assert!(system.contains("plain text only"));
        assert!(system.contains("Do not reproduce or summarize individual subtitle entries"));
        assert!(system.contains("only execution evidence"));
        assert!(system.contains("secrets omitted"));
        assert!(!system.contains("- create_file:"));
    }
}
