use std::path::PathBuf;

pub const DEFAULT_BATCH_SIZE: usize = 30;

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlossaryEntry {
    pub source: String,
    pub target: String,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslationLine {
    pub id: String,
    pub translation: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchTranslationResult {
    pub lines: Vec<TranslationLine>,
    pub summary: String,
    pub glossary_updates: Vec<GlossaryEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewResult {
    pub lines: Vec<TranslationLine>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct PipelineOptions {
    pub input_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub output_format: Option<String>,
    pub provider: String,
    pub model: String,
    pub batch_size: usize,
    pub fast_mode: bool,
    pub bilingual: bool,
    pub target_language: String,
    pub source_language: String,
    pub retries: usize,
    pub final_review: bool,
    pub timeout_seconds: f64,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
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
            fast_mode: false,
            bilingual: false,
            target_language: default_target_language(),
            source_language: default_source_language(),
            retries: default_retries(),
            final_review: true,
            timeout_seconds: default_timeout_seconds(),
            api_key: None,
            base_url: None,
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
