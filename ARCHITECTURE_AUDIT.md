# SubBake Architecture Audit

This audit records the architecture and maintainability issues identified after repeated revisions. The workspace dependency direction is currently correct and the full formatting, Clippy, and test suite passed before remediation began. The main risks are weakened boundaries, duplicated sources of truth, and oversized orchestration modules.

## Priority 1

### 1. `subbake-core` performs filesystem and process-environment reads — Completed

`subbake-core::storage::build_runtime_paths` canonicalizes input paths and reads the process current directory while calculating a run key. This conflicts with the requirement that core remain side-effect-free and makes a domain calculation depend on ambient filesystem state.

Remediation: resolve a stable input path in `subbake-adapters`, pass it explicitly into core, and keep hashing and path-layout construction pure. Preserve the existing run-key algorithm and storage locations.

Completed in commit `f829e12`: stable path resolution now occurs in `subbake-adapters`, while core receives the resolved path explicitly and performs only deterministic runtime layout calculation.

### 2. Resuming a session can overwrite its pinned configuration path — Completed

The CLI discovers configuration again when starting the TUI and immediately writes that path into the loaded session. A resumed session can therefore use a different configuration, profile list, or backend from the one it originally pinned.

Remediation: prefer the stored session path on resume and only perform discovery for sessions without a pinned path. Backend construction, profile listing, and model reporting must use that same path.

Completed: resumed sessions now retain their pinned configuration path, unpinned sessions alone use discovery, session switching rebuilds the backend from the selected session's configuration, and internal profile/model resolution no longer falls back when a pinned path is present.

### 3. Translation CLI paths silently ignore configuration errors — Completed

The `translate` and `batch` argument parsers accept only the successful `Some` result from configuration loading. Parse errors and I/O failures are discarded and execution falls back to defaults, potentially selecting the mock backend without explaining why.

Remediation: propagate configuration errors with path context. Fall back to defaults only when configuration is genuinely absent.

Completed: translation, pipeline, and batch argument resolution now propagate configuration read and parse failures with the offending path, while a genuinely missing configuration file still uses defaults.

### 4. `RuntimeStore` permits false persistence success — Completed

Several persistence methods have defaults that return success without writing anything. The default failure-log and agent-log implementations return plausible paths without creating files. New implementations can accidentally omit required persistence while the pipeline reports success.

Remediation: make required writes mandatory trait methods. Keep defaults only for explicitly optional capabilities, with names and contracts that make no-op behavior clear.

Completed: every `RuntimeStore` write operation is now a required trait method, including review reports, run state, response caches, failure logs, and agent logs. Read defaults retain their explicit empty/not-found semantics without claiming that data was persisted.

## Priority 2

### 5. Core pipeline, agent decision logic, and TUI are oversized orchestrators — In progress

- `subbake-core/src/pipeline.rs` combines batching, terminology, translation, review, retries, splitting, caching, resume, translation memory, logging, and progress.
- `subbake-agent/src/decision.rs` combines the decision loop, quick paths, tool execution, profile handling, diagnostics, translation/transcription orchestration, and presentation text.
- `subbake-agent/src/tui.rs` combines terminal ownership, interaction state, key routing, rendering, pickers, and worker communication.

Remediation: extract cohesive stage services and typed state reducers while retaining a small orchestration entry point. Split by responsibility rather than file length alone.

Progress: batch sizing and dry-run descriptions now belong to a typed core `BatchPlanner`; translation progress, resume restoration, window selection, translation-memory lookup, and ordered result assembly belong to a typed core `TranslationStage`; review planning, resume restoration, window selection, result application, and change/statistics calculation belong to a typed core `ReviewStage`; deterministic agent intent/discovery classification is isolated from the decision loop; and TUI progress rendering is separated from terminal ownership and event routing. The remaining tool execution branches and interaction reducer still need extraction before this item is complete.

### 6. Agent tools have multiple parallel registries — Completed

Tool metadata, argument schemas, discovery membership, approval membership, and executor dispatch are maintained separately. Although tests cover current entries, adding or changing a tool requires synchronized edits across several lists and match expressions.

Remediation: make one registered tool definition own its schema, policy, category, and executor. Derive prompt/native schemas and filtered views from that registry.

Completed: every tool now has one registered definition owning its argument schema, category, mutation/discovery/approval policy, and typed executor identity. Prompt and native schemas, validation, filtered views, policy checks, and execution dispatch all resolve through that registry; duplicate-name/executor regression tests protect the invariant.

### 7. TUI interaction state is not structurally mutually exclusive

`InputMode` is combined with independent flags for processing, plan mode, pending toggles, cancellation, startup, and picker exit behavior. Invalid combinations are representable and correctness relies on event ordering.

Remediation: introduce a top-level interaction-state enum whose variants carry only the data valid for that phase, then route keys and worker responses through typed transitions.

### 8. Translation configuration is repeated mechanically

The same fields are enumerated in patch merging, patch application, configuration parsing/writing, CLI parsing, defaults, and conversion into pipeline options. Compatibility fields such as `final_review` also rely on assignment order for precedence.

Remediation: separate backend, translation-domain, runtime-storage, and CLI-output settings. Convert compatibility aliases once at the configuration boundary and centralize overlay semantics.

## Boundary and Legacy Issues

### 9. Core contains provider secrets and human presentation

`PipelineOptions` carries `api_key` and `base_url` even though the core pipeline does not use them. Diagnostic formatting also emits English headings and CLI-style lists from core.

Remediation: keep provider construction and secrets in adapters, pass only non-secret backend identity required for caching into core, and render structured diagnostics at CLI/TUI edges.

### 10. Python-authoritative comments are stale — Completed

Several comments describe Rust registries and session structures as mirrors of Python. One comment claims there are 19 tools while the current registry contains 20. These comments conflict with the repository's independent Rust architecture and can misdirect future changes.

Remediation: describe current Rust contracts and mention Python only where a persisted compatibility shape is intentionally supported.

Completed: the stale Python-authoritative tool-registry comments and incorrect fixed tool count were replaced with descriptions of the current Rust registry contract.

## Recommended Order

1. Remove core filesystem/environment reads without changing runtime storage identity.
2. Correct configuration pinning and configuration error propagation.
3. Consolidate the agent tool registry.
4. Replace parallel progress/state sources and split oversized orchestrators.
5. Normalize configuration ownership and remove remaining boundary leaks.
