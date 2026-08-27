use subbake_core::ports::ChatMessage;

use super::model::AgentTaskLoop;
use crate::tools::ToolRegistry;

pub(super) fn build_native_messages(
    input: &str,
    loop_state: &AgentTaskLoop,
    dialogue: Option<&str>,
    effective_defaults: &str,
    registry: ToolRegistry,
) -> Vec<ChatMessage> {
    vec![
        ChatMessage::system(system_contract(false, false, effective_defaults, registry)),
        ChatMessage::user(user_context(input, dialogue, Some((loop_state, None)))),
    ]
}

pub(super) fn build_json_messages(
    input: &str,
    loop_state: &AgentTaskLoop,
    dialogue: Option<&str>,
    tools_enabled: bool,
    repair_error: Option<&str>,
    effective_defaults: &str,
    registry: ToolRegistry,
) -> Vec<ChatMessage> {
    vec![
        ChatMessage::system(system_contract(
            true,
            !tools_enabled,
            effective_defaults,
            registry,
        )),
        ChatMessage::user(user_context(
            input,
            dialogue,
            Some((loop_state, repair_error)),
        )),
    ]
}

fn system_contract(
    json_fallback: bool,
    tools_disabled: bool,
    effective_defaults: &str,
    registry: ToolRegistry,
) -> String {
    let mut system = String::from(
        "You are SubBake, a subtitle workflow assistant. The registered tool list supplied in this turn is the complete list: never invent tools such as `list_tools`, shell, or unregistered aliases. Before every meaningful tool phase, write one or two concise sentences in the user's language explaining the current goal and the next action; this text is commentary, not a final answer. Use project-reading tools to ground uncertain paths, then continue with the appropriate execution tool. Descend into directories returned by list_files until you find the actual requested media; a status or list-models check is only an intermediate observation and never completes a transcription request. For a media-to-bilingual-subtitle request that explicitly requires speech recognition, call transcribe_audio on the media, then call translate_file with bilingual=true on the exact subtitle path returned by transcription. When the user asks to translate existing subtitles in a specific MKV, MP4, M4V, MOV, or WebM, call translate_file directly: it extracts text streams directly and OCRs PGS bitmap streams with Tesseract, then adds the translated text track without transcribing audio and replaces the source container by default. If that call reports `no_translatable_text_subtitle`, the container has neither a supported text stream nor a supported PGS stream: use candidate_subtitles once with the media basename to look for a matching text sidecar. Do not inspect Whisper, extract audio, call transcribe_audio, or use run_command as a workaround unless the runtime presents and the user approves a source-substitution prompt. PGS OCR preserves the original subtitle timing and source but is fallible; report OCR language-model or empty-cue errors instead of silently switching to audio. Other bitmap codecs are not currently supported. For a request to translate all subtitles in a directory, prefer `translate_series` with `{\"path\":\".\"}` immediately; use `candidate_subtitles` only when the target is genuinely ambiguous or embedded text is unavailable. Preserve subtitle IDs and ordering. Use `edit_subtitle` for an existing translated or bilingual subtitle. Reuse exact paths returned by tools. Use `apply_patch` for project text-file edits; its patch format is `*** Begin Patch`, Add/Update/Delete File sections, and `*** End Patch`. Do not produce a plan action: call a mutating tool normally and the runtime will handle any approval. Before any approval-triggering call, commentary must explain why the operation is needed and what it will change. After every successful mutation, use its result to produce a concise natural-language final response instead of echoing raw tool output. Responses are rendered in a terminal: use plain text only, without Markdown headings, tables, bold, code fences, or decorative status icons. For successful translation or transcription, normally respond in one to three short lines with the completed action, output path when available, and processed/skipped counts for a batch. Do not reproduce or summarize individual subtitle entries unless the user explicitly asks for their contents.",
    );
    system = system
        .replace(
            "a status or list-models check is only an intermediate observation and never completes a transcription request. For a media-to-bilingual-subtitle request",
            "a status or list-models check is only an intermediate observation and never completes a transcription request. Before translating or transcribing a specific MKV, MP4, M4V, MOV, or WebM container, call inspect_media on that exact path first and use its embedded subtitle metadata to choose the source. For a media-to-bilingual-subtitle request",
        )
        .replace(
            "When the user asks to translate existing subtitles in a specific MKV, MP4, M4V, MOV, or WebM, call translate_file directly:",
            "When the user asks to translate existing subtitles in a specific MKV, MP4, M4V, MOV, or WebM, call inspect_media first and then call translate_file:",
        )
        .replace(
            "OCRs PGS bitmap streams with Tesseract",
            "OCRs PGS, VobSub, and DVB bitmap streams with Tesseract",
        )
        .replace(
            "neither a supported text stream nor a supported PGS stream",
            "neither a supported text stream nor a supported PGS, VobSub, or DVB stream",
        )
        .replace(
            "PGS OCR preserves the original subtitle timing and source",
            "Bitmap OCR preserves the original subtitle timing and source",
        )
        .replace(" Other bitmap codecs are not currently supported.", "");
    system.push_str(
        " If a tool reports that FFmpeg, ffprobe, Tesseract, or source-language data is missing, state the missing dependency and verification command clearly without inventing platform-specific installation commands. For missing bitmap OCR dependencies, explain that audio transcription is available only as an explicitly approved substitute source and does not translate the existing bitmap subtitle.",
    );
    if registry.model_visible_names().contains(&"run_command") {
        system.push_str(
            "\nYou are also a small project-local coding agent. Use `run_command` for inspection, builds, tests, and general non-text artifact generation. The command sandbox cannot write project files directly: use `apply_patch` for source edits. To retain an artifact, declare an alias and final path in `outputs`, then make the command write to `$SUBBAKE_OUTPUT_<ALIAS>`. Never place credentials in a command string.",
        );
    }
    system.push_str(
        "\nUse `delete_file` for project-local paths. Use `delete_external_path` only when the user explicitly asks to delete a path outside the active project; always supply an absolute path and an explicit recursive boolean, and never claim that external deletion can be undone.",
    );
    system.push_str(
        "\nA user request expresses intent, not proof that anything happened. When the user explicitly specifies a supported language, provider, model, format, bilingual mode/order, output path/directory, recursion, overwrite behavior, or asks not to reuse existing runtime/cache state, pass that value in the tool arguments (`fresh_runtime=true` for an isolated translation run). Optional call arguments override only that call and never change the session profile. If a requested modifier is unsupported by the registered tool schema, say that it cannot be applied or suggest configuring a profile/using the CLI; never silently ignore it.",
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
        for spec in registry.model_visible_specs() {
            system.push_str(&spec.prompt_line());
            if spec.mutating {
                system.push_str(" (mutating)");
            }
            system.push('\n');
        }
    }
    if json_fallback {
        system.push_str(
            "\nReturn exactly one JSON object. Allowed shapes are {\"action\":\"respond\",\"text\":\"...\"}, {\"action\":\"ask_user\",\"text\":\"...\"}, or {\"action\":\"tool_call\",\"commentary\":\"one or two concise sentences\",\"tool_name\":\"...\",\"arguments\":{...}}. Never return `plan` or multiple calls.",
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
    loop_data: Option<(&AgentTaskLoop, Option<&str>)>,
) -> String {
    let mut user = format!("Current user request:\n{input}");
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
                    exchange.name, exchange.arguments, exchange.feedback
                ));
            }
        }
        if !loop_state.steering.is_empty() {
            user.push_str(
                "\n\nNew user instructions sent while this task was running (later instructions take priority):\n",
            );
            for instruction in &loop_state.steering {
                user.push_str("- ");
                user.push_str(instruction);
                user.push('\n');
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
            true,
            None,
            "translation: source=Auto, target=zh-Hans, provider=mock, model=mock-zh, format=source, bilingual=false, bilingual_order=target_first, dry_run=false\ntranscription: provider=whisper_cpp, model=small, language=Auto, format=srt",
            ToolRegistry::new(subbake_adapters::CapabilitySet::from_target(
                "linux", "x86_64",
            )),
        );
        let system = &messages[0].content;
        assert!(system.contains("complete list"));
        assert!(system.contains("translate_series"));
        assert!(system.contains(r#"{"path":"."}"#));
        assert!(system.contains("never invent tools"));
        assert!(system.contains("plain text only"));
        assert!(system.contains("Do not reproduce or summarize individual subtitle entries"));
        assert!(system.contains("translate existing subtitles in a specific MKV"));
        assert!(system.contains("call inspect_media on that exact path first"));
        assert!(system.contains("without transcribing audio"));
        assert!(system.contains("OCRs PGS, VobSub, and DVB bitmap streams with Tesseract"));
        assert!(system.contains("FFmpeg, ffprobe, Tesseract"));
        assert!(system.contains("without inventing platform-specific installation commands"));
        assert!(system.contains("explicitly approved substitute source"));
        assert!(system.contains("no_translatable_text_subtitle"));
        assert!(system.contains("user approves a source-substitution prompt"));
        assert!(system.contains("only execution evidence"));
        assert!(system.contains("secrets omitted"));
        assert!(!system.contains("- create_file:"));
    }
}
