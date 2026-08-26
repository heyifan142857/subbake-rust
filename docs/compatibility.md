# Compatibility policy

SubBake `0.2.0-alpha.1` is the first public preview. Its compatibility boundary
distinguishes user data from command-line conveniences.

## Public preview baseline

- Configuration files use `version = 3`. Older pre-release configuration
  versions are rejected with an explicit error instead of being reinterpreted.
- Translation model-output repair uses `translation.model_repair` and
  `translation.model_repair_attempts`; the former `agent` names are not accepted.
- Translation modes are `economy`, `turbo`, and `cinema` and are selected with
  `--mode` or the profile's `translation.mode` setting.
- Interactive plan approval is a typed approve/reject/revise picker.
- Precompiled releases include x86-64 GNU/Linux for glibc 2.35 or newer and a
  statically linked x86-64 musl/Linux build without a glibc runtime dependency.
- Windows x64 and macOS arm64/Intel remain experimental source-build and CI
  targets; no precompiled assets are published for them in this preview.

## Persisted data

Persisted data is more expensive for users to replace than a CLI spelling.
Readers therefore retain the already implemented versioned compatibility for:

- `.subbake` run state and batch shards;
- request, review, and terminology caches;
- glossary and translation-memory data;
- failure and model-repair logs;
- overnight manifests and session metadata.

When a persisted shape changes, SubBake must either retain a backwards-compatible
reader or provide an explicit migration. It must never silently reinterpret an
older shape as a different current value.

## What is not legacy compatibility

The following behavior is part of normal runtime resilience and must not be
removed merely because a symbol or test uses the word `legacy` or `fallback`:

- traditional terminal key encodings used when CSI-u is unavailable;
- JSON tool-call fallback for model providers without native tool support;
- translator fallback when no separate reviewer is configured;
- deterministic provider-response aliases that repair common structured output;
- FFmpeg, whisper.cpp, and source-build fallbacks that support the pinned runtime.

## Removal policy during 0.x

Breaking CLI or configuration changes require a changelog entry and release-note
warning. Persisted-data readers should only be removed after a migration exists
and every public release that could have written the older shape has reached the
documented end of its support window.
