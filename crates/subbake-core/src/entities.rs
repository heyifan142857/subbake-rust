use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub const DEFAULT_BATCH_SIZE: usize = 80;
pub const DEFAULT_BATCH_TOKEN_BUDGET: usize = 1_800;
pub const DEFAULT_TRANSLATION_CONCURRENCY: usize = 3;
pub const DEFAULT_REVIEW_CONCURRENCY: usize = 3;
pub const DEFAULT_PROVIDER: &str = "mock";
pub const DEFAULT_MODEL: &str = "mock-zh";
pub const DEFAULT_TARGET_LANGUAGE: &str = "zh-Hans";
pub const DEFAULT_SOURCE_LANGUAGE: &str = "Auto";
pub const DEFAULT_RETRIES: usize = 2;
pub const DEFAULT_AGENT_REPAIR_ATTEMPTS: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TranslationMode {
    Economy,
    #[default]
    Turbo,
    Cinema,
}

impl TranslationMode {
    pub fn parse(value: &str) -> Result<Self, SettingParseError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "economy" | "eco" => Ok(Self::Economy),
            "turbo" | "fast" => Ok(Self::Turbo),
            "cinema" | "quality" => Ok(Self::Cinema),
            _ => Err(SettingParseError {
                setting: "translation mode",
                value: value.to_owned(),
                expected: "economy, turbo, cinema",
            }),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Economy => "economy",
            Self::Turbo => "turbo",
            Self::Cinema => "cinema",
        }
    }
}

/// Fully expanded behavior for a translation mode. Adapters may override the
/// numeric settings, while the domain keeps the semantic differences here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranslationPolicy {
    pub document_preflight: bool,
    pub include_context: bool,
    pub compact_wire: bool,
    pub deduplicate: bool,
    pub review_policy: ReviewPolicy,
}

impl TranslationPolicy {
    pub const fn for_mode(mode: TranslationMode) -> Self {
        match mode {
            TranslationMode::Economy => Self {
                document_preflight: false,
                include_context: false,
                compact_wire: true,
                deduplicate: true,
                review_policy: ReviewPolicy::Off,
            },
            TranslationMode::Turbo => Self {
                document_preflight: false,
                include_context: true,
                compact_wire: true,
                deduplicate: true,
                review_policy: ReviewPolicy::Off,
            },
            TranslationMode::Cinema => Self {
                document_preflight: true,
                include_context: true,
                compact_wire: true,
                deduplicate: true,
                review_policy: ReviewPolicy::Full,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingParseError {
    pub setting: &'static str,
    pub value: String,
    pub expected: &'static str,
}

impl Display for SettingParseError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} must be one of: {} (received `{}`)",
            self.setting, self.expected, self.value
        )
    }
}

impl Error for SettingParseError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BilingualOrder {
    SourceFirst,
    #[default]
    TargetFirst,
}

impl BilingualOrder {
    pub fn parse(value: &str) -> Result<Self, SettingParseError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "source_first" => Ok(Self::SourceFirst),
            "target_first" => Ok(Self::TargetFirst),
            _ => Err(SettingParseError {
                setting: "bilingual order",
                value: value.to_owned(),
                expected: "source_first, target_first",
            }),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SourceFirst => "source_first",
            Self::TargetFirst => "target_first",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReviewPolicy {
    #[default]
    Off,
    Targeted,
    Full,
}

impl ReviewPolicy {
    pub fn parse(value: &str) -> Result<Self, SettingParseError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "false" | "none" => Ok(Self::Off),
            "targeted" | "true" => Ok(Self::Targeted),
            "full" => Ok(Self::Full),
            _ => Err(SettingParseError {
                setting: "review policy",
                value: value.to_owned(),
                expected: "off, targeted, full",
            }),
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

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TerminologyPreflightResult {
    pub entries: Vec<GlossaryEntry>,
    /// Advisory document-level context. Older cache entries omit it.
    #[serde(default)]
    pub document_brief: String,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TerminologyStats {
    pub candidates: usize,
    pub entries_added: usize,
    pub conflicts_omitted: usize,
    pub cache_hits: usize,
    pub degraded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degraded_reason: Option<String>,
    pub usage: Usage,
    pub duration_ms: u64,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ReviewStats {
    pub candidate_lines: usize,
    pub reviewed_lines: usize,
    pub changed_lines: usize,
    pub batches: usize,
    pub cache_hits: usize,
    pub usage: Usage,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewChange {
    pub batch: usize,
    pub id: String,
    pub reasons: Vec<String>,
    pub before: String,
    pub after: String,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ReviewReport {
    pub terminology: TerminologyStats,
    pub review: ReviewStats,
    pub changes: Vec<ReviewChange>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
    #[serde(default)]
    pub cached_input_tokens: usize,
    #[serde(default)]
    pub requests: usize,
    #[serde(default)]
    pub retries: usize,
}

impl Usage {
    pub fn add(&mut self, other: Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.requests += other.requests;
        self.retries += other.retries;
    }

    pub fn billable_input_tokens(self) -> usize {
        self.input_tokens.saturating_sub(self.cached_input_tokens)
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
    /// Present only when a runtime store is configured for the pipeline.
    pub log_path: Option<PathBuf>,
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
    pub mode: TranslationMode,
    pub bilingual: bool,
    pub bilingual_order: BilingualOrder,
    pub target_language: String,
    pub source_language: String,
    pub retries: usize,
    pub review_policy: ReviewPolicy,
    pub terminology_preflight: bool,
    /// Keep personal names in their source spelling instead of translating or
    /// transliterating them into the target language.
    pub preserve_names: bool,
    pub timeout_seconds: f64,
    /// Non-secret identity of the configured API route, used to isolate v2
    /// cache entries across protocols and relay endpoints.
    pub provider_fingerprint: Option<String>,
    pub reviewer_fingerprint: Option<String>,
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
            mode: TranslationMode::Turbo,
            bilingual: false,
            bilingual_order: BilingualOrder::default(),
            target_language: default_target_language(),
            source_language: default_source_language(),
            retries: DEFAULT_RETRIES,
            review_policy: ReviewPolicy::Off,
            terminology_preflight: true,
            preserve_names: false,
            timeout_seconds: default_timeout_seconds(),
            provider_fingerprint: None,
            reviewer_fingerprint: None,
            dry_run: false,
            resume: true,
            use_cache: true,
            agent: true,
            agent_repair_attempts: DEFAULT_AGENT_REPAIR_ATTEMPTS,
            runtime_dir: None,
            glossary_path: None,
        }
    }

    pub const fn policy(&self) -> TranslationPolicy {
        TranslationPolicy::for_mode(self.mode)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineResult {
    pub output_path: Option<PathBuf>,
    pub batches_translated: usize,
    pub review_batches: usize,
    pub usage: Usage,
    pub mode: TranslationMode,
    pub deduplicated_segments: usize,
    pub reviewer_fallback: bool,
    pub dry_run: bool,
    pub planned_batches: Vec<BatchPlanEntry>,
    pub cache_hits: usize,
    pub resumed_translation_batches: usize,
    pub resumed_review_batches: usize,
    pub translation_memory_hits: usize,
    pub state_path: Option<PathBuf>,
    pub glossary_path: Option<PathBuf>,
    pub agent_repairs: Vec<AgentRepairRecord>,
    pub terminology: TerminologyStats,
    pub review: ReviewStats,
}

fn default_provider() -> String {
    DEFAULT_PROVIDER.to_owned()
}

fn default_model() -> String {
    DEFAULT_MODEL.to_owned()
}

fn default_target_language() -> String {
    DEFAULT_TARGET_LANGUAGE.to_owned()
}

fn default_source_language() -> String {
    DEFAULT_SOURCE_LANGUAGE.to_owned()
}

fn default_timeout_seconds() -> f64 {
    120.0
}
