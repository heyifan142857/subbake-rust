# AGENTS.md

## Project

SubBake is a Rust subtitle translation and transcription CLI with an interactive terminal agent. It originated as a migration from the Python implementation, but this repository is now an independent Rust codebase. Preserve compatible runtime data where useful; do not treat Python module structure or CLI behavior as authoritative.

The workspace uses Rust 2024 and forbids unsafe code. Workspace Clippy policy denies `dbg!`, `todo!`, and `unwrap()`.

## Workspace Boundaries

- `crates/subbake-core`: side-effect-free domain code. Subtitle formats, validation, prompts and response contracts, translation pipeline orchestration, memory, storage schemas, cancellation primitives, and ports belong here.
- `crates/subbake-adapters`: integrations and side effects. Configuration, HTTP LLM backends, filesystem runtime storage, translation/editing services, transcription, ffmpeg/whisper.cpp, downloads, and runtime cleanup belong here.
- `crates/subbake-agent`: interactive-agent behavior. Decision loop, sessions, plan approval, tools, project-local file guard, undo, cancellation, history, profile/session pickers, and Ratatui/Crossterm UI belong here.
- `crates/subbake-cli`: composition root and `sbake` binary. Argument parsing, config/profile resolution, dependency wiring, output selection, and command dispatch belong here.

Dependencies must point inward: CLI may wire every crate; agent may use core and adapters; adapters implement core ports; core must not depend on terminal, network, environment variables, filesystem/process adapters, or CLI concerns.

## Current Command Surface

- `sbake`: start the interactive agent.
- `sbake resume [SESSION_ID]`: resume the latest or a specific agent session.
- `sbake translate <SUBTITLE>`: translate a subtitle or text file.
- `sbake batch <DIR>`: translate subtitle files in a directory.
- `sbake transcribe <MEDIA>`: transcribe media.
- `sbake pipeline <MEDIA_OR_SUBTITLE>`: explicitly transcribe when needed and then translate.
- `sbake provider check`: validate a provider configuration.
- `sbake runtime inspect|clean`: inspect or remove runtime artifacts.
- `sbake whisper ...`: report status, install, update, uninstall, list models, or download a whisper.cpp model.

Keep `translate` subtitle-only. Media transcription must remain explicit through `transcribe` or `pipeline`.

The interactive agent supports slash completion, `/model` and `/profile` profile pickers, `/sessions`, `/history`, `/clear`, `/plan`, `/undo`, and `/exit`; Shift+Tab toggles plan mode. Plan approval is a typed approve/reject/revise picker, not synthetic slash-command UI. Up/Down navigates the active picker or persisted input history. Esc cancels an active operation cooperatively and cancels the current picker/form when idle.

## Translation Modes

Translation modes are semantic product presets, not branding aliases. Keep `TranslationMode` and `TranslationPolicy::for_mode` in `subbake-core` as the source of truth; adapters apply the selected mode's configurable defaults before explicit profile or CLI overrides.

- `economy` is cost/throughput-first. It uses large self-contained batches and deduplication, with neighboring context, document terminology preflight, online terminology, and model review disabled by default. Do not silently add extra model stages or context that defeats its cost contract.
- `turbo` is the default latency/quality balance. It uses high adaptive concurrency, fixed neighboring source context, previously confirmed translations, and lightweight name/terminology reconciliation. Document preflight, online terminology, and model review remain off by default unless explicitly enabled.
- `cinema` is quality/consistency-first. It uses smaller scene-aware batches and context, document terminology preflight, online terminology, and full review by default. Use the configured reviewer when present; otherwise the translator backend performs the review.

All modes must retain deterministic correctness guarantees: exact ID/count alignment, formatting-marker preservation, user-required glossary enforcement, final output validation, cancellation, and safe cache/Resume isolation. A cheaper mode may reduce model work, but must not weaken these guarantees.

When changing a mode or adding one, update the enum parsing/serialization, `TranslationPolicy`, adapter mode defaults, CLI/config/help text, Resume semantic fingerprints, and focused tests together. Keep mode matches exhaustive and preserve the rule that explicit user overrides win over preset defaults.

## Architecture Rules

- Put domain rules and serializable contracts in `subbake-core`; keep human presentation at adapter/CLI/TUI edges.
- Implement model providers through `LlmBackend` and explicit `GenerationRequest`/`ResponseContract` values. Provider profile names and wire protocols are separate concepts.
- Treat neighboring source lines and previously confirmed translations as read-only prompt context. Translation responses may contain only the current editable IDs, and prompt instructions must agree with the structured response contract.
- Use the shared `TermMatcher` for glossary selection, protected spans, required-term validation, and review candidate selection. Do not reintroduce ad hoc lowercase-plus-`contains` matching.
- Run every finalized output path, including translation-memory hits and reviewer output, through the same deterministic final validator.
- Route review work through the configured reviewer backend when present and the translator backend otherwise. Resume and request-cache fingerprints must identify the backend that actually serves each stage.
- Carry `CancellationGuard` through blocking or async provider, pipeline, transcription, download, and child-process boundaries. Check cancellation before starting a new side effect.
- Register agent tools with a `ToolSpec` and implement the executor in the same change. Unknown tools must fail explicitly; never return placeholder success.
- Keep TUI/worker communication typed with `TuiAction` and `TuiInteraction`. Do not encode picker choices as generated command strings.
- Treat input modes as mutually exclusive states. When adding a key binding, define its behavior for editing, history, picker/form, pending-plan, and processing states.
- Render completed structured output immediately. Reserve character streaming for short conversational responses.
- Build and validate a replacement backend before persisting a profile switch. A failed switch must leave the session and active backend unchanged.
- Mutating plan execution must persist progress after each successful tool so retrying cannot repeat completed mutations.
- Keep terminal raw/alternate-screen ownership RAII-protected and join worker threads during shutdown.
- Keep project-local file operations inside `FileGuard`; preserve protected-path and symlink-escape checks. Mutations must participate in backup/undo bookkeeping.
- Do not put business logic in CLI handlers. Handlers assemble settings and invoke services/use cases.

## Configuration and Secrets

Configuration supports defaults plus named profiles. CLI flags override resolved configuration. Keep configuration discovery and the path pinned in an agent session consistent with backend construction and profile listing.

Never log, persist in sessions, or copy inline provider secrets unnecessarily. New profiles may reuse environment-variable names but must not duplicate inline API keys or authorization-header values. Preserve existing config comments and file permissions when updating configuration.

## Compatibility

Storage compatibility remains more important than historical CLI compatibility. Preserve readable Python-compatible shapes where implemented, including `.subbake` run state, request/review caches, failure logs, glossary and translation-memory data, batch shards, request hashes, and session metadata.

If a persisted shape changes, add an explicit version or a backwards-compatible reader and test it. Do not silently reinterpret existing runtime data.

Treat the Resume fingerprint as a versioned semantic contract. It must cover loaded glossary contents, translator/reviewer route fingerprints, and behavior-affecting policies. Bump the dedicated prompt-contract, translation-memory-policy, or final-validation-policy version whenever the corresponding semantics change.

## Change Discipline

- Extend existing traits, registries, state types, and composition boundaries instead of adding parallel sources of truth.
- Prefer typed errors at crate boundaries. Convert them to concise user-facing messages at the outer edge.
- Add focused regression tests for state transitions, storage shapes, cancellation, and failure ordering.
- Preserve user changes in a dirty worktree and avoid unrelated rewrites.
- Use conventional commit messages when committing changes.

## Verification

Run all of the following before handing off a change:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Use `cargo run -p subbake-cli -- ...` for manual command checks. Interactive changes should also receive a real PTY smoke test when they affect raw-mode lifecycle, rendering, key routing, cancellation, or picker/form behavior.
