use std::path::{Path, PathBuf};

use crate::entities::{PipelineOptions, SubtitleSegment, Usage};
use crate::languages::language_pair_slug;

pub const RUN_STATE_VERSION: u64 = 3;
pub const TRANSLATION_FINGERPRINT_VERSION: u64 = 5;
pub const RENDER_FINGERPRINT_VERSION: u64 = 2;
pub const CACHE_VERSION: u64 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimePaths {
    pub root_dir: PathBuf,
    pub run_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub state_path: PathBuf,
    pub glossary_path: PathBuf,
    pub failures_dir: PathBuf,
    pub translated_batches_dir: PathBuf,
    pub reviewed_batches_dir: PathBuf,
    pub translation_memory_path: PathBuf,
    pub agent_logs_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputSignature {
    pub sha1: String,
    pub size: u64,
    pub mtime_ns: Option<u128>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeSnapshot {
    pub translated_segments: Vec<SubtitleSegment>,
    pub reviewed_segments: Vec<SubtitleSegment>,
    pub usage: Usage,
    pub translation_batches_completed: usize,
    pub review_batches_completed: usize,
    pub validation_completed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl From<&str> for JsonValue {
    fn from(value: &str) -> Self {
        JsonValue::String(value.to_owned())
    }
}

impl From<String> for JsonValue {
    fn from(value: String) -> Self {
        JsonValue::String(value)
    }
}

impl From<bool> for JsonValue {
    fn from(value: bool) -> Self {
        JsonValue::Bool(value)
    }
}

impl From<usize> for JsonValue {
    fn from(value: usize) -> Self {
        JsonValue::Number(value.to_string())
    }
}

impl From<u64> for JsonValue {
    fn from(value: u64) -> Self {
        JsonValue::Number(value.to_string())
    }
}

pub fn build_runtime_paths(
    input_path: &Path,
    runtime_dir: Option<&Path>,
    glossary_path: Option<&Path>,
    source_language: &str,
    target_language: &str,
    fast_mode: bool,
) -> RuntimePaths {
    let root_dir = runtime_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| input_path.parent().unwrap_or_else(|| Path::new(".")).join(".subbake"));
    let safe_stem = slugify(
        input_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("input"),
    );
    let input_key = &stable_hash(&JsonValue::Object(vec![(
        "path".to_owned(),
        JsonValue::String(input_path.to_string_lossy().to_string()),
    )]))[..12];
    let run_dir = root_dir.join("runs").join(format!("{safe_stem}-{input_key}"));
    let language_pair = language_pair_slug(source_language, target_language);
    let translation_memory_mode = if fast_mode { "fast" } else { "standard" };

    RuntimePaths {
        root_dir: root_dir.clone(),
        run_dir: run_dir.clone(),
        cache_dir: root_dir.join("cache"),
        state_path: run_dir.join("run_state.json"),
        glossary_path: glossary_path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| root_dir.join(format!("glossary.{language_pair}.json"))),
        failures_dir: run_dir.join("failures"),
        translated_batches_dir: run_dir.join("translated_batches"),
        reviewed_batches_dir: run_dir.join("reviewed_batches"),
        translation_memory_path: root_dir.join(format!(
            "translation_memory.v2.{language_pair}.{translation_memory_mode}.json"
        )),
        agent_logs_dir: run_dir.join("agent_logs"),
    }
}

pub fn input_signature_from_bytes(bytes: &[u8], mtime_ns: Option<u128>) -> InputSignature {
    InputSignature {
        sha1: sha1_hex(bytes),
        size: bytes.len() as u64,
        mtime_ns,
    }
}

pub fn build_translation_fingerprint(
    options: &PipelineOptions,
    input_signature: &InputSignature,
) -> String {
    stable_hash(&JsonValue::Object(vec![
        ("version".to_owned(), JsonValue::Number(TRANSLATION_FINGERPRINT_VERSION.to_string())),
        (
            "input_signature".to_owned(),
            JsonValue::Object(vec![
                ("sha1".to_owned(), JsonValue::String(input_signature.sha1.clone())),
                ("size".to_owned(), JsonValue::Number(input_signature.size.to_string())),
                (
                    "mtime_ns".to_owned(),
                    input_signature
                        .mtime_ns
                        .map(|value| JsonValue::Number(value.to_string()))
                        .unwrap_or(JsonValue::Null),
                ),
            ]),
        ),
        (
            "input_format".to_owned(),
            JsonValue::String(
                options
                    .input_path
                    .extension()
                    .and_then(|value| value.to_str())
                    .map(|value| format!(".{}", value.to_lowercase()))
                    .unwrap_or_default(),
            ),
        ),
        ("provider".to_owned(), JsonValue::String(options.provider.clone())),
        ("model".to_owned(), JsonValue::String(options.model.clone())),
        ("batch_size".to_owned(), JsonValue::Number(options.batch_size.to_string())),
        ("fast_mode".to_owned(), JsonValue::Bool(options.fast_mode)),
        (
            "source_language".to_owned(),
            JsonValue::String(options.source_language.clone()),
        ),
        (
            "target_language".to_owned(),
            JsonValue::String(options.target_language.clone()),
        ),
    ]))
}

pub fn build_render_fingerprint(options: &PipelineOptions) -> String {
    stable_hash(&JsonValue::Object(vec![
        ("version".to_owned(), JsonValue::Number(RENDER_FINGERPRINT_VERSION.to_string())),
        ("bilingual".to_owned(), JsonValue::Bool(options.bilingual)),
        ("final_review".to_owned(), JsonValue::Bool(options.final_review)),
        (
            "output_format".to_owned(),
            options
                .output_format
                .clone()
                .map(JsonValue::String)
                .unwrap_or(JsonValue::Null),
        ),
        (
            "output_path".to_owned(),
            options
                .output_path
                .as_ref()
                .map(|path| JsonValue::String(path.to_string_lossy().to_string()))
                .unwrap_or(JsonValue::Null),
        ),
    ]))
}

pub fn build_request_hash(
    provider: &str,
    model: &str,
    stage: &str,
    messages: JsonValue,
) -> String {
    stable_hash(&JsonValue::Object(vec![
        ("version".to_owned(), JsonValue::Number(CACHE_VERSION.to_string())),
        ("provider".to_owned(), JsonValue::String(provider.to_lowercase())),
        ("model".to_owned(), JsonValue::String(model.to_owned())),
        ("stage".to_owned(), JsonValue::String(stage.to_owned())),
        ("messages".to_owned(), messages),
    ]))
}

pub fn stable_hash(payload: &JsonValue) -> String {
    sha1_hex(canonical_json(payload).as_bytes())
}

pub fn canonical_json(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".to_owned(),
        JsonValue::Bool(value) => value.to_string(),
        JsonValue::Number(value) => value.clone(),
        JsonValue::String(value) => quote_json_string(value),
        JsonValue::Array(items) => {
            let items = items.iter().map(canonical_json).collect::<Vec<_>>();
            format!("[{}]", items.join(","))
        }
        JsonValue::Object(entries) => {
            let mut entries = entries.iter().collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            let entries = entries
                .into_iter()
                .map(|(key, value)| format!("{}:{}", quote_json_string(key), canonical_json(value)))
                .collect::<Vec<_>>();
            format!("{{{}}}", entries.join(","))
        }
    }
}

fn quote_json_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for ch in value.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            ch if ch < '\u{20}' => output.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => output.push(ch),
        }
    }
    output.push('"');
    output
}

fn slugify(value: &str) -> String {
    let mut output = String::new();
    let mut previous_dash = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
            output.push(ch);
            previous_dash = false;
        } else if !previous_dash {
            output.push('-');
            previous_dash = true;
        }
    }
    let output = output.trim_matches('-');
    if output.is_empty() {
        "input".to_owned()
    } else {
        output.to_owned()
    }
}

pub fn sha1_hex(bytes: &[u8]) -> String {
    let digest = sha1_digest(bytes);
    digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

fn sha1_digest(bytes: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x6745_2301;
    let mut h1: u32 = 0xefcd_ab89;
    let mut h2: u32 = 0x98ba_dcfe;
    let mut h3: u32 = 0x1032_5476;
    let mut h4: u32 = 0xc3d2_e1f0;

    let bit_len = (bytes.len() as u64) * 8;
    let mut message = bytes.to_vec();
    message.push(0x80);
    while (message.len() % 64) != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in message.chunks(64) {
        let mut words = [0u32; 80];
        for (index, word) in words.iter_mut().take(16).enumerate() {
            let start = index * 4;
            *word = u32::from_be_bytes([
                chunk[start],
                chunk[start + 1],
                chunk[start + 2],
                chunk[start + 3],
            ]);
        }
        for index in 16..80 {
            words[index] =
                (words[index - 3] ^ words[index - 8] ^ words[index - 14] ^ words[index - 16])
                    .rotate_left(1);
        }

        let mut a = h0;
        let mut b = h1;
        let mut c = h2;
        let mut d = h3;
        let mut e = h4;

        for (index, word) in words.iter().enumerate() {
            let (f, k) = match index {
                0..=19 => ((b & c) | ((!b) & d), 0x5a82_7999),
                20..=39 => (b ^ c ^ d, 0x6ed9_eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
                _ => (b ^ c ^ d, 0xca62_c1d6),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut digest = [0u8; 20];
    for (index, value) in [h0, h1, h2, h3, h4].iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&value.to_be_bytes());
    }
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_sorts_keys_without_spaces() {
        let value = JsonValue::Object(vec![
            ("b".to_owned(), JsonValue::Number("2".to_owned())),
            (
                "a".to_owned(),
                JsonValue::Object(vec![
                    ("d".to_owned(), JsonValue::Number("4".to_owned())),
                    ("c".to_owned(), JsonValue::Number("3".to_owned())),
                ]),
            ),
        ]);
        assert_eq!(canonical_json(&value), r#"{"a":{"c":3,"d":4},"b":2}"#);
    }

    #[test]
    fn stable_hash_is_key_order_independent() {
        let left = JsonValue::Object(vec![
            ("b".to_owned(), JsonValue::Number("2".to_owned())),
            ("a".to_owned(), JsonValue::Number("1".to_owned())),
        ]);
        let right = JsonValue::Object(vec![
            ("a".to_owned(), JsonValue::Number("1".to_owned())),
            ("b".to_owned(), JsonValue::Number("2".to_owned())),
        ]);
        assert_eq!(stable_hash(&left), stable_hash(&right));
    }

    #[test]
    fn sha1_matches_known_vector() {
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    }

    #[test]
    fn runtime_paths_match_expected_shape() {
        let paths = build_runtime_paths(
            Path::new("/tmp/show.srt"),
            None,
            None,
            "Auto",
            "Chinese",
            false,
        );
        assert!(paths.run_dir.to_string_lossy().contains("show-"));
        assert!(
            paths
                .translation_memory_path
                .to_string_lossy()
                .contains("translation_memory.v2.auto-chinese.standard.json")
        );
    }
}
