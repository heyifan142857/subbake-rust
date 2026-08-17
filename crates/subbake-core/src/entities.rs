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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextStrategy {
    SelfContained,
    FixedNeighbors { lines: usize },
    SceneAware,
}

impl ContextStrategy {
    pub const fn includes_context(self) -> bool {
        !matches!(self, Self::SelfContained)
    }

    pub const fn uses_scene_boundaries(self) -> bool {
        matches!(self, Self::SceneAware)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrencyStrategy {
    Fixed,
    AdaptiveQueued { window_multiplier: usize },
    SceneAware,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminologyStrategy {
    Disabled,
    LightweightNames,
    Document,
}

impl TerminologyStrategy {
    pub const fn preflight_default(self) -> bool {
        matches!(self, Self::Document)
    }

    pub const fn online_default(self) -> bool {
        matches!(self, Self::Document)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralRecoveryStrategy {
    SplitImmediately,
    CorrectBeforeSplit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightFailurePolicy {
    ContinueDegraded,
    Fail,
}

impl PreflightFailurePolicy {
    pub const fn allows_degraded(self) -> bool {
        matches!(self, Self::ContinueDegraded)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewStrategy {
    Standard,
    Adjudicated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCacheStrategy {
    Standard,
    CacheableSystem,
}

/// Fully expanded behavior for a translation mode. Adapters may override the
/// numeric settings, while the domain keeps the semantic differences here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranslationPolicy {
    pub batch_size: usize,
    pub batch_token_budget: usize,
    /// Hard estimate for the complete prompt plus anticipated JSON response.
    pub request_token_budget: usize,
    pub confirmed_context_lines: usize,
    pub confirmed_context_token_budget: usize,
    pub translation_concurrency: usize,
    pub review_concurrency: usize,
    pub context_strategy: ContextStrategy,
    pub concurrency_strategy: ConcurrencyStrategy,
    pub terminology_strategy: TerminologyStrategy,
    pub structural_recovery_strategy: StructuralRecoveryStrategy,
    pub preflight_failure_policy: PreflightFailurePolicy,
    pub review_strategy: ReviewStrategy,
    pub prompt_cache_strategy: PromptCacheStrategy,
    pub compact_wire: bool,
    pub deduplicate: bool,
    pub review_policy: ReviewPolicy,
}

impl TranslationPolicy {
    pub const fn for_mode(mode: TranslationMode) -> Self {
        match mode {
            TranslationMode::Economy => Self {
                batch_size: 160,
                batch_token_budget: 6_000,
                request_token_budget: 14_000,
                confirmed_context_lines: 0,
                confirmed_context_token_budget: 0,
                translation_concurrency: 3,
                review_concurrency: 1,
                context_strategy: ContextStrategy::SelfContained,
                concurrency_strategy: ConcurrencyStrategy::Fixed,
                terminology_strategy: TerminologyStrategy::Disabled,
                structural_recovery_strategy: StructuralRecoveryStrategy::CorrectBeforeSplit,
                preflight_failure_policy: PreflightFailurePolicy::ContinueDegraded,
                review_strategy: ReviewStrategy::Standard,
                prompt_cache_strategy: PromptCacheStrategy::Standard,
                compact_wire: true,
                deduplicate: true,
                review_policy: ReviewPolicy::Off,
            },
            TranslationMode::Turbo => Self {
                batch_size: 96,
                batch_token_budget: 2_400,
                request_token_budget: 10_000,
                confirmed_context_lines: 12,
                confirmed_context_token_budget: 800,
                translation_concurrency: 8,
                review_concurrency: 4,
                context_strategy: ContextStrategy::FixedNeighbors { lines: 3 },
                concurrency_strategy: ConcurrencyStrategy::AdaptiveQueued {
                    window_multiplier: 2,
                },
                terminology_strategy: TerminologyStrategy::LightweightNames,
                structural_recovery_strategy: StructuralRecoveryStrategy::SplitImmediately,
                preflight_failure_policy: PreflightFailurePolicy::ContinueDegraded,
                review_strategy: ReviewStrategy::Standard,
                prompt_cache_strategy: PromptCacheStrategy::Standard,
                compact_wire: true,
                deduplicate: true,
                review_policy: ReviewPolicy::Off,
            },
            TranslationMode::Cinema => Self {
                batch_size: 48,
                batch_token_budget: 1_600,
                request_token_budget: 10_000,
                confirmed_context_lines: 16,
                confirmed_context_token_budget: 1_000,
                translation_concurrency: 4,
                review_concurrency: 3,
                context_strategy: ContextStrategy::SceneAware,
                concurrency_strategy: ConcurrencyStrategy::SceneAware,
                terminology_strategy: TerminologyStrategy::Document,
                structural_recovery_strategy: StructuralRecoveryStrategy::SplitImmediately,
                preflight_failure_policy: PreflightFailurePolicy::Fail,
                review_strategy: ReviewStrategy::Adjudicated,
                prompt_cache_strategy: PromptCacheStrategy::CacheableSystem,
                compact_wire: true,
                deduplicate: true,
                review_policy: ReviewPolicy::Full,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TranslationReadabilityDefaults {
    pub max_characters_per_second: Option<f64>,
    pub max_characters_per_line: Option<usize>,
    pub max_lines: Option<usize>,
}

pub fn translation_readability_defaults(
    mode: TranslationMode,
    target_language: &str,
) -> TranslationReadabilityDefaults {
    if mode != TranslationMode::Cinema {
        return TranslationReadabilityDefaults {
            max_characters_per_second: None,
            max_characters_per_line: None,
            max_lines: None,
        };
    }
    let language = target_language
        .split(['-', '_'])
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let cjk = matches!(language.as_str(), "zh" | "ja" | "ko");
    TranslationReadabilityDefaults {
        // CJK subtitle references are much denser than prose, especially on
        // sub-second cues. These hard safety rails sit just above the observed
        // maxima of the local five-episode reference corpus; stricter editorial
        // targets belong in QA because translation cannot retime source cues.
        max_characters_per_second: Some(if cjk { 23.0 } else { 17.0 }),
        max_characters_per_line: Some(if cjk { 32 } else { 42 }),
        max_lines: Some(2),
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SubtitleSemanticContext {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

impl SubtitleSemanticContext {
    pub fn is_empty(&self) -> bool {
        self.speaker.is_none()
            && self.style.is_none()
            && self.layer.is_none()
            && self.kind.is_none()
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
    #[serde(default, skip_serializing_if = "SubtitleSemanticContext::is_empty")]
    pub semantic: SubtitleSemanticContext,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassthroughBlock {
    pub insert_before: usize,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SubtitleDocumentMetadata {
    #[default]
    None,
    Ass(AssDocumentMetadata),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssDocumentMetadata {
    pub had_bom: bool,
    pub records: Vec<AssRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssRecord {
    Raw(String),
    Dialogue(AssDialogueRecord),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssDialogueRecord {
    pub segment_id: String,
    pub event_kind: String,
    pub fields: Vec<String>,
    pub start_index: usize,
    pub end_index: usize,
    pub text_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubtitleDocument {
    pub path: PathBuf,
    pub format: String,
    pub segments: Vec<SubtitleSegment>,
    pub header: Option<String>,
    pub passthrough_blocks: Vec<PassthroughBlock>,
    pub metadata: SubtitleDocumentMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlossaryEntry {
    pub source: String,
    pub target: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminologyKind {
    Person,
    Organization,
    Place,
    ProperName,
    DomainTerm,
}

impl TerminologyKind {
    pub const fn is_enforced(self) -> bool {
        !matches!(self, Self::DomainTerm)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminologyEntity {
    pub canonical_source: String,
    pub kind: TerminologyKind,
    #[serde(default)]
    pub variants: Vec<GlossaryEntry>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TerminologyPreflightResult {
    pub entries: Vec<GlossaryEntry>,
    /// Entity-aware terminology. Older caches only contain flat `entries`.
    #[serde(default)]
    pub entities: Vec<TerminologyEntity>,
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
pub struct ConfirmedTranslationContext {
    pub id: String,
    pub source: String,
    pub translation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchTranslationResult {
    pub lines: Vec<TranslationLine>,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub glossary_updates: Vec<GlossaryEntry>,
    /// Entity-aware incremental terminology emitted alongside a translation.
    #[serde(default)]
    pub terminology_updates: Vec<TerminologyEntity>,
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
    pub request_token_budget: usize,
    pub confirmed_context_lines: usize,
    pub confirmed_context_token_budget: usize,
    pub translation_concurrency: usize,
    pub review_concurrency: usize,
    pub mode: TranslationMode,
    pub bilingual: bool,
    pub bilingual_order: BilingualOrder,
    pub bilingual_font_scale: f64,
    pub target_language: String,
    pub source_language: String,
    pub retries: usize,
    pub review_policy: ReviewPolicy,
    pub terminology_preflight: bool,
    /// Ask translation batches to emit and consume incremental terminology.
    pub online_terminology: bool,
    pub allow_degraded_preflight: bool,
    /// Keep personal names in their source spelling instead of translating or
    /// transliterating them into the target language.
    pub preserve_names: bool,
    /// Optional hard subtitle readability limits. `None` disables the
    /// corresponding final-output check.
    pub max_characters_per_second: Option<f64>,
    pub max_characters_per_line: Option<usize>,
    pub max_lines: Option<usize>,
    pub timeout_seconds: f64,
    /// Non-secret identity of the configured API route, used to isolate v2
    /// cache entries across protocols and relay endpoints.
    pub provider_fingerprint: Option<String>,
    pub reviewer_fingerprint: Option<String>,
    /// Optional execution contract that isolates Resume state for composed
    /// workflows such as incremental media pipelines. Standalone subtitle
    /// translation leaves this unset so its historical fingerprints remain
    /// stable.
    pub execution_fingerprint: Option<String>,
    /// Hash of the glossary snapshot frozen after terminology preflight. This
    /// is populated by the pipeline and keeps Resume state tied to the exact
    /// terminology context without persisting glossary contents in run state.
    pub glossary_fingerprint: Option<String>,
    /// Confirmed translations immediately preceding a composed translation
    /// shard. Standalone document translation leaves this empty.
    pub initial_confirmed_context: Vec<ConfirmedTranslationContext>,
    pub dry_run: bool,
    pub resume: bool,
    pub use_cache: bool,
    pub agent: bool,
    pub agent_repair_attempts: usize,
    pub max_requests: Option<usize>,
    pub max_tokens: Option<usize>,
    pub runtime_dir: Option<PathBuf>,
    pub glossary_path: Option<PathBuf>,
}

impl PipelineOptions {
    pub fn new(input_path: PathBuf) -> Self {
        let policy = TranslationPolicy::for_mode(TranslationMode::Turbo);
        Self {
            input_path,
            output_path: None,
            output_format: None,
            provider: default_provider(),
            model: default_model(),
            batch_size: policy.batch_size,
            batch_token_budget: policy.batch_token_budget,
            request_token_budget: policy.request_token_budget,
            confirmed_context_lines: policy.confirmed_context_lines,
            confirmed_context_token_budget: policy.confirmed_context_token_budget,
            translation_concurrency: policy.translation_concurrency,
            review_concurrency: policy.review_concurrency,
            mode: TranslationMode::Turbo,
            bilingual: false,
            bilingual_order: BilingualOrder::default(),
            bilingual_font_scale: 1.0,
            target_language: default_target_language(),
            source_language: default_source_language(),
            retries: DEFAULT_RETRIES,
            review_policy: policy.review_policy,
            terminology_preflight: policy.terminology_strategy.preflight_default(),
            online_terminology: policy.terminology_strategy.online_default(),
            allow_degraded_preflight: policy.preflight_failure_policy.allows_degraded(),
            preserve_names: false,
            max_characters_per_second: None,
            max_characters_per_line: None,
            max_lines: None,
            timeout_seconds: default_timeout_seconds(),
            provider_fingerprint: None,
            reviewer_fingerprint: None,
            execution_fingerprint: None,
            glossary_fingerprint: None,
            initial_confirmed_context: Vec::new(),
            dry_run: false,
            resume: true,
            use_cache: true,
            agent: true,
            agent_repair_attempts: DEFAULT_AGENT_REPAIR_ATTEMPTS,
            max_requests: None,
            max_tokens: None,
            runtime_dir: None,
            glossary_path: None,
        }
    }

    pub const fn policy(&self) -> TranslationPolicy {
        TranslationPolicy::for_mode(self.mode)
    }

    pub const fn preflight_failure_policy(&self) -> PreflightFailurePolicy {
        if self.allow_degraded_preflight {
            PreflightFailurePolicy::ContinueDegraded
        } else {
            PreflightFailurePolicy::Fail
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_cinema_uses_scene_aware_batching() {
        assert!(
            !TranslationPolicy::for_mode(TranslationMode::Economy)
                .context_strategy
                .uses_scene_boundaries()
        );
        assert!(
            !TranslationPolicy::for_mode(TranslationMode::Turbo)
                .context_strategy
                .uses_scene_boundaries()
        );
        assert!(
            TranslationPolicy::for_mode(TranslationMode::Cinema)
                .context_strategy
                .uses_scene_boundaries()
        );
    }

    #[test]
    fn mode_strategies_are_fully_expanded_in_one_policy() {
        let economy = TranslationPolicy::for_mode(TranslationMode::Economy);
        assert_eq!(economy.context_strategy, ContextStrategy::SelfContained);
        assert_eq!(economy.concurrency_strategy, ConcurrencyStrategy::Fixed);
        assert_eq!(economy.terminology_strategy, TerminologyStrategy::Disabled);
        assert_eq!(
            economy.structural_recovery_strategy,
            StructuralRecoveryStrategy::CorrectBeforeSplit
        );
        assert_eq!(
            economy.preflight_failure_policy,
            PreflightFailurePolicy::ContinueDegraded
        );

        let turbo = TranslationPolicy::for_mode(TranslationMode::Turbo);
        assert_eq!(
            turbo.context_strategy,
            ContextStrategy::FixedNeighbors { lines: 3 }
        );
        assert_eq!(
            turbo.concurrency_strategy,
            ConcurrencyStrategy::AdaptiveQueued {
                window_multiplier: 2
            }
        );
        assert_eq!(
            turbo.terminology_strategy,
            TerminologyStrategy::LightweightNames
        );

        let cinema = TranslationPolicy::for_mode(TranslationMode::Cinema);
        assert_eq!(cinema.context_strategy, ContextStrategy::SceneAware);
        assert_eq!(cinema.concurrency_strategy, ConcurrencyStrategy::SceneAware);
        assert_eq!(cinema.terminology_strategy, TerminologyStrategy::Document);
        assert_eq!(
            cinema.preflight_failure_policy,
            PreflightFailurePolicy::Fail
        );
        assert_eq!(cinema.review_strategy, ReviewStrategy::Adjudicated);
        assert_eq!(
            cinema.prompt_cache_strategy,
            PromptCacheStrategy::CacheableSystem
        );
    }

    #[test]
    fn explicit_preflight_override_becomes_the_effective_typed_policy() {
        let mut options = PipelineOptions::new("sample.srt".into());

        options.allow_degraded_preflight = false;
        assert_eq!(
            options.preflight_failure_policy(),
            PreflightFailurePolicy::Fail
        );
    }

    #[test]
    fn cinema_uses_cjk_readability_safety_rails_above_reference_maxima() {
        let defaults = translation_readability_defaults(TranslationMode::Cinema, "zh-Hans");

        assert_eq!(defaults.max_characters_per_second, Some(23.0));
        assert_eq!(defaults.max_characters_per_line, Some(32));
        assert_eq!(defaults.max_lines, Some(2));
    }

    #[test]
    fn pipeline_defaults_match_the_turbo_policy() {
        let options = PipelineOptions::new("sample.srt".into());
        let policy = TranslationPolicy::for_mode(TranslationMode::Turbo);

        assert_eq!(options.mode, TranslationMode::Turbo);
        assert_eq!(options.batch_size, policy.batch_size);
        assert_eq!(options.batch_token_budget, policy.batch_token_budget);
        assert_eq!(options.request_token_budget, policy.request_token_budget);
        assert_eq!(
            options.confirmed_context_lines,
            policy.confirmed_context_lines
        );
        assert_eq!(
            options.confirmed_context_token_budget,
            policy.confirmed_context_token_budget
        );
        assert_eq!(
            options.translation_concurrency,
            policy.translation_concurrency
        );
        assert_eq!(options.review_concurrency, policy.review_concurrency);
        assert_eq!(options.review_policy, policy.review_policy);
        assert_eq!(
            options.terminology_preflight,
            policy.terminology_strategy.preflight_default()
        );
        assert_eq!(
            options.online_terminology,
            policy.terminology_strategy.online_default()
        );
        assert_eq!(
            options.allow_degraded_preflight,
            policy.preflight_failure_policy.allows_degraded()
        );
    }
}
