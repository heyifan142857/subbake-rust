pub mod diagnostics;
pub mod editing;
pub mod entities;
pub mod error;
pub mod formats;
pub mod languages;
pub mod memory;
pub mod pipeline;
pub mod ports;
mod recovery;
mod review;
pub mod storage;
pub mod validation;

pub use diagnostics::DiagnosticReport;
pub use editing::SubtitleEditPayload;
pub use entities::{
    AgentLog, AgentRepairRecord, AttemptLog, BatchPlanEntry, BatchTranslationResult, FailureLog,
    GlossaryEntry, PassthroughBlock, PipelineOptions, PipelineResult, ReviewResult, SplitRetryLog,
    SubtitleDocument, SubtitleSegment, TranslationLine, Usage,
};
pub use error::{CoreError, CoreResult};
pub use memory::ContextMemory;
