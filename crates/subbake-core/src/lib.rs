pub mod diagnostics;
pub mod editing;
pub mod entities;
pub mod error;
pub mod evaluation;
pub mod formats;
mod formatting;
mod language_rules;
pub mod languages;
pub mod memory;
mod number_facts;
pub mod overnight;
pub mod pipeline;
pub mod ports;
pub mod progress;
pub mod quality;
mod recovery;
mod review;
pub mod storage;
pub mod term_matcher;
pub mod tool_outcome;
pub mod transcription_evaluation;
pub mod validation;

pub use cancellation::{CancellationGuard, CancellationToken};
pub use diagnostics::DiagnosticReport;
pub use editing::SubtitleEditPayload;
pub use entities::{
    AgentLog, AgentRepairRecord, AssDialogueRecord, AssDocumentMetadata, AssRecord, AttemptLog,
    BatchPlanEntry, BatchTranslationResult, BilingualOrder, ConcurrencyStrategy,
    ConfirmedTranslationContext, ContextStrategy, DocumentCharacter, DocumentGuide, FailureLog,
    GlossaryEntry, PassthroughBlock, PipelineOptions, PipelineResult, PreflightFailurePolicy,
    PromptCacheStrategy, ReviewAnnotation, ReviewChange, ReviewIssueKind, ReviewPolicy,
    ReviewReport, ReviewRoute, ReviewRouteKind, ReviewStats, ReviewStrategy, SplitRetryLog,
    StructuralRecoveryStrategy, SubtitleDocument, SubtitleDocumentMetadata, SubtitleSegment,
    SubtitleSemanticContext, TerminologyEntity, TerminologyKind, TerminologyPreflightResult,
    TerminologyStats, TerminologyStrategy, TranslationLine, TranslationMode, TranslationPolicy,
    TranslationReadabilityDefaults, Usage, translation_readability_defaults,
};
pub use error::{CoreError, CoreResult, LlmCallError, StorageError, StorageIoKind};
pub use evaluation::{
    ConsistencyKind, ConsistencyRule, DocumentConsistencyReport, DocumentConsistencyViolation,
    DocumentConsistencyViolationKind, EvaluationReport, HardConstraintKind, HardConstraintReport,
    HardConstraintViolation, MqmCounts, TranslationQualityReport, evaluate,
    evaluate_translation_quality,
};
pub use memory::ContextMemory;
pub use ports::{
    BatchExecutionOptions, GenerationContent, GenerationInput, GenerationRequest,
    GenerationResponse, ModelToolCall, ModelToolResult, NativeToolSupport, ReasoningPolicy,
    ResponseContract, ToolChoice, ToolConfiguration, ToolContinuation, ToolDefinition,
    TranscriberBackend, TranscriptionFormat,
};
pub use progress::{
    NoopProgress, ProgressEvent, ProgressSink, ProgressUnit, SharedProgress, TaskKind, TaskState,
    TranslationProgress,
};
pub use quality::{
    QualityIssue, QualityIssueKind, QualityPolicy, QualityReport, QualitySeverity, inspect_quality,
};
pub use term_matcher::{TermMatch, TermMatcher};
pub use tool_outcome::{
    AgentToolOutcome, CommandToolOutcome, FileToolOutcome, ObservationToolOutcome,
    ProfileToolOutcome, SkippedPath, SubtitleEditToolOutcome, ToolExecutionStatus,
    TranscriptionToolOutcome, TranslationToolOutcome, WhisperModelFact, WhisperToolOutcome,
};
pub use transcription_evaluation::{TranscriptionEvaluationReport, evaluate_transcription};
pub use validation::{FinalValidationPolicy, validate_final_output};
pub mod cancellation;
