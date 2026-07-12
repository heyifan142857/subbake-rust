use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub const DEFAULT_BATCH_SIZE: usize = 80;
pub const DEFAULT_BATCH_TOKEN_BUDGET: usize = 1_800;
pub const DEFAULT_TRANSLATION_CONCURRENCY: usize = 3;
pub const DEFAULT_REVIEW_CONCURRENCY: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReviewPolicy {
    Off,
    #[default]
    Targeted,
    Full,
}

impl ReviewPolicy {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "false" | "none" => Ok(Self::Off),
            "targeted" | "true" => Ok(Self::Targeted),
            "full" => Ok(Self::Full),
            _ => Err("review policy must be one of: off, targeted, full".to_owned()),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Targeted => "targeted",
            Self::Full => "full",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubtitleSegment {
    pub id: String,
    pub text: String,
    pub start: Option<String>,
    pub end: Option<String>,
    pub identifier: Option<String>,
    pub settings: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassthroughBlock {
    pub insert_before: usize,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubtitleDocument {
    pub path: PathBuf,
    pub format: String,
    pub segments: Vec<SubtitleSegment>,
    pub header: Option<String>,
    pub passthrough_blocks: Vec<PassthroughBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlossaryEntry {
    pub source: String,
    pub target: String,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn add(&mut self, other: Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranslationLine {
    pub id: String,
    pub translation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchTranslationResult {
    pub lines: Vec<TranslationLine>,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub glossary_updates: Vec<GlossaryEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewResult {
    pub lines: Vec<TranslationLine>,
    #[serde(default)]
    pub review_notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchPlanEntry {
    pub index: usize,
    pub size: usize,
    pub first_id: String,
    pub last_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentRepairRecord {
    pub stage: String,
    pub batch_index: usize,
    pub attempts: usize,
    pub success: bool,
    pub log_path: PathBuf,
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptLog {
    pub attempt: usize,
    pub cached: bool,
    pub error: Option<String>,
    #[serde(default)]
    pub payload: Option<serde_json::Value>,
    #[serde(default)]
    pub messages: Vec<crate::ports::ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split_retry: Option<SplitRetryLog>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitRetryLog {
    pub triggered: bool,
    pub sizes: Vec<usize>,
    #[serde(default)]
    pub resolved: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureLog {
    pub stage: String,
    pub batch_index: usize,
    pub request_hash: String,
    pub batch_segments: Vec<SubtitleSegment>,
    pub messages: Vec<crate::ports::ChatMessage>,
    #[serde(default)]
    pub translated_segments: Vec<SubtitleSegment>,
    pub attempts: Vec<AttemptLog>,
    #[serde(default)]
    pub agent_attempts: Vec<AttemptLog>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentLog {
    pub stage: String,
    pub batch_index: usize,
    pub success: bool,
    pub attempts: Vec<AttemptLog>,
    pub final_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PipelineOptions {
    pub input_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub output_format: Option<String>,
    pub provider: String,
    pub model: String,
    pub batch_size: usize,
    pub batch_token_budget: usize,
    pub translation_concurrency: usize,
    pub review_concurrency: usize,
    pub fast_mode: bool,
    pub bilingual: bool,
    pub target_language: String,
    pub source_language: String,
    pub retries: usize,
    pub review_policy: ReviewPolicy,
    pub terminology_preflight: bool,
    pub timeout_seconds: f64,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    /// Non-secret identity of the configured API route, used to isolate v2
    /// cache entries across protocols and relay endpoints.
    pub provider_fingerprint: Option<String>,
    pub dry_run: bool,
    pub resume: bool,
    pub use_cache: bool,
    pub agent: bool,
    pub agent_repair_attempts: usize,
    pub runtime_dir: Option<PathBuf>,
    pub glossary_path: Option<PathBuf>,
}

impl PipelineOptions {
    pub fn new(input_path: PathBuf) -> Self {
        Self {
            input_path,
            output_path: None,
            output_format: None,
            provider: default_provider(),
            model: default_model(),
            batch_size: DEFAULT_BATCH_SIZE,
            batch_token_budget: DEFAULT_BATCH_TOKEN_BUDGET,
            translation_concurrency: DEFAULT_TRANSLATION_CONCURRENCY,
            review_concurrency: DEFAULT_REVIEW_CONCURRENCY,
            fast_mode: false,
            bilingual: false,
            target_language: default_target_language(),
            source_language: default_source_language(),
            retries: default_retries(),
            review_policy: ReviewPolicy::Targeted,
            terminology_preflight: true,
            timeout_seconds: default_timeout_seconds(),
            api_key: None,
            base_url: None,
            provider_fingerprint: None,
            dry_run: false,
            resume: true,
            use_cache: true,
            agent: true,
            agent_repair_attempts: default_agent_repair_attempts(),
            runtime_dir: None,
            glossary_path: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineResult {
    pub output_path: Option<PathBuf>,
    pub batches_translated: usize,
    pub review_batches: usize,
    pub usage: Usage,
    pub dry_run: bool,
    pub planned_batches: Vec<BatchPlanEntry>,
    pub cache_hits: usize,
    pub resumed_translation_batches: usize,
    pub resumed_review_batches: usize,
    pub translation_memory_hits: usize,
    pub state_path: Option<PathBuf>,
    pub glossary_path: Option<PathBuf>,
    pub agent_repairs: Vec<AgentRepairRecord>,
}

fn default_provider() -> String {
    "mock".to_owned()
}

fn default_model() -> String {
    "mock-zh".to_owned()
}

fn default_target_language() -> String {
    "Chinese".to_owned()
}

fn default_source_language() -> String {
    "Auto".to_owned()
}

fn default_retries() -> usize {
    2
}

fn default_timeout_seconds() -> f64 {
    120.0
}

fn default_agent_repair_attempts() -> usize {
    2
}
