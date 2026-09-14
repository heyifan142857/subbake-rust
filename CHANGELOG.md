# Changelog

All notable changes to SubBake are documented in this file. SubBake follows
Semantic Versioning, with the understanding that `0.x` releases may still make
breaking changes when they are called out in advance.

## [Unreleased]

### Changed

- Use whisper.cpp full JSON as the recognition contract and share stable source
  cues across Economy, Turbo, and Cinema. Raw recognition and segmented-source
  caches are mode-independent; older media pipeline shards are not reused.
- Replace `transcription.filter_hallucinations` and its CLI switches with
  `filter_silence_markers`; repeated dialogue is now retained and diagnosed.

- Remove Silero voice-activity detection and all related configuration, CLI,
  model-management, agent-tool, status, and Resume fingerprint surfaces.
  Transcription now always processes the complete audio waveform; configuration
  files containing the removed fields are intentionally rejected.
- Redesign the interactive agent transcript with distinct commentary and final
  response markers, structured connected tool activity groups, live in-place
  status transitions, and width-aware wrapping for long paths and CJK text.
- Rename translation failure repair from `agent` to `model_repair` throughout
  configuration, CLI flags, runtime diagnostics, and JSON output. Configuration
  version 3 intentionally rejects the former names instead of aliasing them.

### Added

- Add profile-configurable spoken language (default Auto), per-chunk actual
  language reports, and deterministic `smart`/`segment` source segmentation using
  token timestamps with lossless whole-segment fallback on uncertain alignment.

- Add constrained bitmap-subtitle OCR correction before terminology and
  translation, with deterministic Economy behavior, targeted model correction
  in Turbo/Cinema, CLI/config overrides, cache isolation, corrected bilingual
  source rendering, and a versioned runtime audit report.

### Fixed

- Enforce shared sensitive-path protection in the agent command sandbox with a
  filtered project view; hide project symlinks, sockets, credentials, VCS/runtime
  data, and unrelated host files even from recursive shell commands.
- Protect default and session-selected configuration files from generic agent
  reads and edits, including resolved aliases and resumed sessions.
- Restore both original files when undoing an overwrite rename, retain relative
  backup references, and reject self-renames before any mutation.
- Publish immutable, SHA-256-verified checkpoint shards with version 6 run-state
  manifests so concurrent runs and interrupted publications cannot mix batches.
  Existing run-state and legacy batch readers remain supported.
- Keep process cancellation and timeouts active until inherited output pipes
  close, and bound command output and progress queues during collection.
  Structured process output exceeding 64 MiB per stream now fails explicitly.

- Keep Turbo personal-name marker contracts stable when retrying a translation
  through smaller split batches, revalidate merged results, reject recurring
  dialogue starters as name candidates, and diagnose marker failures without
  incorrectly attributing them to runtime cache state.
- Inspect embedded subtitle streams before the interactive agent translates or
  transcribes a media container, so it chooses between existing text, bitmap
  OCR, and audio sources with current stream metadata.
- Keep the interactive composer editable while the agent is working: Enter
  queues a follow-up for the next turn, while Esc sends non-empty input into
  the active turn and interrupts only its in-flight model request.
- Preserve non-zero embedded subtitle timestamps while extracting text or PGS
  tracks, embedding translated tracks, and remuxing container undo output, so a
  subtitle-only container no longer rebases its first cue to `00:00:00`.

## [0.2.0-alpha.1]

### Added

- Rust subtitle translation and transcription CLI with an interactive terminal agent.
- SRT, VTT, ASS/SSA, TTML/DFXP, and TXT translation support.
- Text and bitmap subtitle handling for supported media containers.
- Local whisper.cpp installation, model management, and transcription.
- Economy, Turbo, and Cinema translation policies.
- Translation memory, glossary, resumable runtime state, quality checks, and safe editing.
- GNU and musl Linux x64 release archives with SHA-256 checksums and build provenance.

### Changed

- Configuration version 2 is the first public configuration contract.
- Translation modes are selected only through `--mode economy|turbo|cinema`;
  the pre-release `--fast` alias has been removed.
- Plan approval uses the typed terminal picker rather than hidden slash-command aliases.

### Compatibility

- This is an alpha preview. CLI and configuration changes may occur before 1.0.
- Existing versioned runtime state, caches, glossary data, translation memory,
  and session metadata remain readable where a compatibility reader exists.

[Unreleased]: https://github.com/heyifan142857/subbake-rust/compare/v0.2.0-alpha.1...HEAD
[0.2.0-alpha.1]: https://github.com/heyifan142857/subbake-rust/releases/tag/v0.2.0-alpha.1
