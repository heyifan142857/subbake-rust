# Changelog

All notable changes to SubBake are documented in this file. SubBake follows
Semantic Versioning, with the understanding that `0.x` releases may still make
breaking changes when they are called out in advance.

## [Unreleased]

### Changed

- Redesign the interactive agent transcript with distinct commentary and final
  response markers, structured connected tool activity groups, live in-place
  status transitions, and width-aware wrapping for long paths and CJK text.
- Rename translation failure repair from `agent` to `model_repair` throughout
  configuration, CLI flags, runtime diagnostics, and JSON output. Configuration
  version 3 intentionally rejects the former names instead of aliasing them.

### Added

- Add constrained bitmap-subtitle OCR correction before terminology and
  translation, with deterministic Economy behavior, targeted model correction
  in Turbo/Cinema, CLI/config overrides, cache isolation, corrected bilingual
  source rendering, and a versioned runtime audit report.

### Fixed

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
