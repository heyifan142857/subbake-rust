pub mod diagnostics;
pub mod editing;
pub mod entities;
pub mod error;
pub mod formats;
pub mod languages;
pub mod memory;
pub mod pipeline;
pub mod ports;
pub mod progress;
mod recovery;
mod review;
pub mod storage;
pub mod tool_outcome;
pub mod validation;

pub use cancellation::{CancellationGuard, CancellationToken};
pub use diagnostics::DiagnosticReport;
pub use editing::SubtitleEditPayload;
pub use entities::{
    AgentLog, AgentRepairRecord, AttemptLog, BatchPlanEntry, BatchTranslationResult,
    BilingualOrder, FailureLog, GlossaryEntry, PassthroughBlock, PipelineOptions, PipelineResult,
    ReviewChange, ReviewPolicy, ReviewReport, ReviewStats, SplitRetryLog, SubtitleDocument,
    SubtitleSegment, TerminologyPreflightResult, TerminologyStats, TranslationLine, Usage,
};
pub use error::{CoreError, CoreResult, LlmCallError, StorageError, StorageIoKind};
pub use memory::ContextMemory;
pub use ports::{
    BatchExecutionOptions, GenerationContent, GenerationInput, GenerationRequest,
    GenerationResponse, ModelToolCall, ModelToolResult, NativeToolSupport, ResponseContract,
    ToolChoice, ToolConfiguration, ToolContinuation, ToolDefinition,
};
pub use progress::{
    NoopProgress, ProgressEvent, ProgressSink, ProgressUnit, SharedProgress, TaskKind, TaskState,
    TranslationProgress,
};
pub use tool_outcome::{
    AgentToolOutcome, FileToolOutcome, ObservationToolOutcome, ProfileToolOutcome, SkippedPath,
    SubtitleEditToolOutcome, ToolExecutionStatus, TranscriptionToolOutcome, TranslationToolOutcome,
    WhisperModelFact, WhisperToolOutcome,
};
pub mod cancellation;
