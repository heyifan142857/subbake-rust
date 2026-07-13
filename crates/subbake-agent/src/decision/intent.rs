use serde_json::{Value as JsonValue, json};

pub(super) fn translation_action_requested(input: &str) -> bool {
    let normalized = input.trim().to_lowercase();
    if normalized.contains("如何翻译")
        || normalized.contains("怎么翻译")
        || normalized.starts_with("how ")
    {
        return false;
    }
    normalized.starts_with("translate")
        || normalized.contains("please translate")
        || normalized.contains("帮我翻译")
        || normalized.contains("请翻译")
        || normalized.contains("翻译一下")
        || normalized.starts_with("翻译")
}

pub(super) fn translation_target_omitted(input: &str) -> bool {
    let normalized = input.trim().to_lowercase();
    matches!(
        normalized.as_str(),
        "translate" | "please translate" | "帮我翻译" | "请翻译" | "翻译一下" | "翻译"
    )
}

pub(super) fn preferred_discovery(input: &str) -> Option<(&'static str, JsonValue)> {
    if translation_action_requested(input) {
        return Some(("candidate_subtitles", json!({"path": ".", "query": input})));
    }
    let normalized = input.trim().to_lowercase();
    if normalized.starts_with("how ") || normalized.contains("如何") || normalized.contains("怎么")
    {
        return None;
    }
    if normalized.starts_with("transcribe")
        || normalized.contains("帮我转录")
        || normalized.contains("请转录")
        || normalized.starts_with("转录")
    {
        return Some(("search_files", json!({"path": ".", "pattern": ""})));
    }
    if normalized.starts_with("edit")
        || normalized.contains("帮我修改")
        || normalized.contains("帮我编辑")
        || normalized.contains("请修改")
        || normalized.starts_with("修改")
        || normalized.starts_with("编辑")
    {
        return Some(("recent_translations", json!({})));
    }
    if normalized.starts_with("diagnose")
        || normalized.starts_with("debug")
        || normalized.contains("帮我诊断")
        || normalized.starts_with("诊断")
    {
        return Some(("search_files", json!({"path": ".", "pattern": ""})));
    }
    if [
        "delete ",
        "rename ",
        "replace ",
        "append ",
        "删除",
        "重命名",
        "替换",
        "追加",
    ]
    .iter()
    .any(|prefix| normalized.starts_with(prefix))
    {
        return Some(("search_files", json!({"path": ".", "pattern": ""})));
    }
    None
}

pub(super) fn bilingual_requested(input: &str) -> bool {
    let normalized = input.to_lowercase();
    normalized.contains("bilingual") || normalized.contains("双语") || normalized.contains("中英")
}

pub(super) fn localize(input: &str, english: &str, chinese: &str) -> String {
    if input
        .chars()
        .any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch))
    {
        chinese.to_owned()
    } else {
        english.to_owned()
    }
}
