# SubBake usage guide

This guide describes the current Rust CLI. SubBake is an independent codebase;
older Python command shapes are not authoritative.

## What SubBake does

SubBake has four primary workflows:

1. Translate subtitle or text files.
2. Translate an existing text subtitle stream inside a media container.
3. Transcribe audio or video with local whisper.cpp.
4. Run an interactive terminal agent that can inspect a project and invoke the
   same translation and transcription services under approval and file-safety
   rules.

`translate` never turns media audio into subtitles. Use `transcribe` for
speech-to-text only, or `pipeline` for transcription followed by translation.

## Installation

Install a recent Rust toolchain. FFmpeg and ffprobe are required for media
inspection, transcription preparation, and embedded subtitle operations.

```bash
git clone https://github.com/heyifan142857/subbake-rust.git
cd subbake-rust
cargo install --path crates/subbake-cli
```

Install without the interactive agent when only non-interactive commands are
needed:

```bash
cargo install --path crates/subbake-cli --no-default-features
```

Confirm the installation:

```bash
sbake --version
sbake --help
```

## Configuration

### Discovery and precedence

SubBake resolves configuration in this order:

1. Built-in defaults.
2. The `[defaults]` section of the selected configuration file.
3. The selected `[profiles.<name>]` section.
4. Explicit CLI flags.

Use `--config <PATH>` to select a file directly. Without that flag, SubBake
checks the user configuration location and then the current directory:

- `$XDG_CONFIG_HOME/subbake/config.toml`, when `XDG_CONFIG_HOME` is set;
- otherwise `$HOME/.config/subbake/config.toml`;
- `./subbake.toml`;
- `./.subbake.toml`.

### Generic version 2 configuration

Backend names and profile names are local identifiers. They do not select a
wire protocol; set `api_format` explicitly for every non-mock backend.

```toml
version = 2
default_profile = "standard"

[defaults.translation]
source_language = "Auto"
target_language = "English"
mode = "turbo"

[defaults.output]
bilingual = false

[defaults.agent]
max_steps = 64
auto_approve_commands = false

[backends.primary]
id = "provider-id"
model = "translation-model"
api_format = "openai_chat"
base_url = "https://api.example.com/v1"
api_key_env = "SUBBAKE_PRIMARY_API_KEY"
timeout_seconds = 120

[backends.quality-review]
id = "review-provider-id"
model = "review-model"
api_format = "anthropic_messages"
base_url = "https://review.example.com/api"
api_key_env = "SUBBAKE_REVIEW_API_KEY"
timeout_seconds = 180

[profiles.standard]
translator = "primary"

[profiles.cinema]
translator = "primary"
reviewer = "quality-review"

[profiles.cinema.translation]
mode = "cinema"
target_language = "English"

[profiles.local-transcription.transcription]
model = "base"

[profiles.local-transcription.storage]
whisper_binary_path = "/path/to/whisper-cli"
whisper_models_dir = "/path/to/whisper-models"
```

Supported `api_format` values are:

- `openai_chat`
- `openai_responses`
- `anthropic_messages`
- `gemini_generate_content`

`endpoint_url` may be used instead of `base_url` when a provider requires a
complete endpoint. Custom authorization can be configured with `auth_header`
and `auth_prefix`.

### Secrets

Prefer `api_key_env` over inline `api_key` values:

```bash
export SUBBAKE_PRIMARY_API_KEY="replace-with-a-secret"
sbake provider check --profile standard
```

The normal `sbake` CLI does not automatically load a project `.env` file. Set
variables through the shell, operating-system service configuration, or a
secret manager. The ignored live-evaluation test has a separate, narrowly
scoped `.env` loader documented in [Agent evaluation](agent-evaluation.md).

Never commit real keys, authorization headers, private-key material, or a
configuration containing inline credentials.

### Validate a backend

Validate a profile before starting a long run:

```bash
sbake provider check --config /path/to/config.toml --profile standard
```

A backend can also be checked without a stored profile:

```bash
export SUBBAKE_PROVIDER_API_KEY="replace-with-a-secret"
sbake provider check \
  --provider provider-id \
  --model model-id \
  --api-format openai_chat \
  --base-url https://api.example.com/v1 \
  --api-key-env SUBBAKE_PROVIDER_API_KEY
```

## Interactive agent

Start a new session:

```bash
sbake
# Equivalent:
sbake agent
```

Resume the latest session or a known session ID:

```bash
sbake resume
sbake resume SESSION_ID
```

Important interactive controls include:

- `/profile` and `/model` for backend/profile selection;
- `/sessions` and `/history` for persisted session navigation;
- `/plan` or Shift+Tab to toggle plan mode;
- `/undo` to restore the latest supported mutation;
- `/clear`, `/help`, and `/exit`;
- Esc to cancel an active operation cooperatively.

Mutating plans require an explicit approve/reject/revise decision. Commands
outside the strict read-only auto-run set require command approval. Project file
operations reject path traversal, project escape, protected runtime/VCS paths,
and common credential paths.

## Translate subtitle files

Supported subtitle/text formats include SRT, VTT, ASS, and TXT.

```bash
sbake translate episode.srt \
  --profile standard \
  --target-lang English \
  --output episode.en.srt
```

Create bilingual output or select another subtitle format:

```bash
sbake translate episode.ass \
  --target-lang English \
  --bilingual \
  --bilingual-font-scale 0.85 \
  --output-format ass \
  --output episode.bilingual.ass
```

Use `--dry-run` to prepare work without provider calls. Provider request and
token budgets can stop a run before its next request:

```bash
sbake translate episode.srt \
  --dry-run \
  --max-requests 20 \
  --max-tokens 50000
```

Resume and request caching are enabled by default. Disable them for an isolated
run:

```bash
sbake translate episode.srt --no-resume --no-cache
```

Useful tuning flags include `--batch-size`, `--batch-token-budget`,
`--translation-concurrency`, `--retries`, `--glossary`, and
`--timeout-seconds`. Run `sbake translate --help` for the current flag surface.

### Translation modes

Modes are semantic presets rather than branding aliases:

| Mode | Primary goal | Default behavior |
| --- | --- | --- |
| `economy` | Cost and throughput | Large self-contained batches, deduplication, fewer model stages |
| `turbo` | Latency/quality balance | High concurrency, neighboring source context, confirmed prior translations, lightweight name alignment |
| `cinema` | Quality and consistency | Smaller scene-aware batches, terminology preflight, online terminology, full review |

Select a mode explicitly when reproducibility matters:

```bash
sbake translate episode.srt --mode economy
sbake translate episode.srt --mode turbo
sbake translate episode.srt --mode cinema
```

Explicit profile and CLI settings override preset defaults. All modes retain
deterministic ID/count alignment, formatting-marker preservation, required
glossary enforcement, final validation, cancellation, and cache/resume
isolation.

### Review and readability limits

Enable targeted or full model review:

```bash
sbake translate episode.srt --review targeted
sbake translate episode.srt --review full
```

When a profile specifies a reviewer backend, that backend performs review.
Otherwise the translator backend is reused.

Reject output that exceeds configured subtitle constraints:

```bash
sbake translate episode.srt \
  --max-characters-per-second 20 \
  --max-characters-per-line 42 \
  --max-lines 2
```

## Translate an embedded subtitle stream

MKV, MP4/M4V/MOV, and WebM inputs may be used when they already contain a
supported text subtitle stream:

```bash
sbake translate movie.mkv \
  --subtitle-stream 3 \
  --target-lang English
```

SubBake selects a compatible subtitle codec, copies existing streams, verifies
the output, and atomically replaces the source container by default. Preserve
the source and write a separate translated container with:

```bash
sbake translate movie.mkv \
  --subtitle-stream 3 \
  --preserve-source-container \
  --output movie.translated.mkv
```

If the media has no usable text subtitle stream, use `pipeline`; `translate`
does not silently transcribe audio.

## Batch translation

Translate a directory:

```bash
sbake batch ./subtitles --profile standard --target-lang English
```

Common controls:

```bash
sbake batch ./subtitles --recursive
sbake batch ./subtitles --overwrite
sbake batch ./subtitles --fail-fast
sbake batch ./subtitles --retry-failed /path/to/batch-manifest.json
```

Batch runs keep a manifest and can continue after individual file failures
unless `--fail-fast` is used.

## whisper.cpp management

SubBake can manage whisper.cpp and its models, or use an existing installation.

```bash
sbake whisper status
sbake whisper versions
sbake whisper install --variant cpu
sbake whisper model list
sbake whisper model base
sbake whisper update
```

Supported build variants are `cpu`, `cuda`, `metal`, `vulkan`, and `openblas`.
Use `--bin`, `--models-dir`, and `--runtime-dir` to override managed locations.

Remove the managed installation with `uninstall`; use `--keep-models` when
model files should remain available:

```bash
sbake whisper uninstall --keep-models
```

## Transcription

Transcribe media without translation:

```bash
sbake transcribe interview.mp4 \
  --language Auto \
  --model base \
  --format srt \
  --output interview.srt
```

Use an existing timed sidecar instead of running whisper.cpp:

```bash
sbake transcribe interview.mp4 \
  --sidecar transcript.vtt \
  --format srt \
  --output interview.srt
```

Long media is processed in overlapping chunks. SubBake merges boundary text,
checks coverage, and does not publish an obviously incomplete final subtitle.

## Transcription and translation pipeline

Use `pipeline` when the input may require transcription before translation:

```bash
sbake pipeline lecture.mp4 \
  --transcribe-language Auto \
  --transcribe-model base \
  --target-lang English \
  --output lecture.en.srt
```

`pipeline` accepts translation options plus transcription-specific flags such
as `--transcribe-format`, `--sidecar`, `--whisper-bin`, and
`--whisper-models-dir`. Transcription chunks and translation groups have
separate resume state.

## Subtitle quality commands

Run deterministic reference-free timing and readability checks:

```bash
sbake qa candidate.srt
sbake qa candidate.srt --json --fail-on warning
```

Current CLI checks include empty text, invalid/overlapping timing, reading
speed, line length/count, and repeated segments.

Compare a candidate with a reference:

```bash
sbake evaluate candidate.srt reference.srt
sbake evaluate candidate.srt reference.srt --json
```

The current CLI reports deterministic chrF and mechanical MQM-style structural
findings. The richer Rust evaluation APIs for translation hard constraints,
document consistency, and transcription WER/CER/timing metrics are documented
in [Subtitle and transcription evaluation](subtitle-evaluation.md); they are not
yet exposed as the complete `sbake evaluate` CLI report.

## Glossary and translation memory

Inspect, export, import, or prune runtime memory associated with a target:

```bash
sbake memory inspect episode.srt
sbake memory export episode.srt memory-bundle.json
sbake memory import episode.srt memory-bundle.json
sbake memory prune episode.srt --yes
```

Bundles have a versioned JSON shape. Import merges entries without replacing
existing local values; prune removes blank mappings.

## Runtime inspection and cleanup

Inspect the exact runtime locations associated with an input before cleaning:

```bash
sbake runtime inspect episode.srt
```

Cleanup is explicit and requires `--yes`:

```bash
sbake runtime clean episode.srt --runs --yes
sbake runtime clean episode.srt --cache --yes
sbake runtime clean episode.srt --glossary --yes
sbake runtime clean episode.srt --all --yes
```

## Provider-managed overnight batches

Provider-managed asynchronous economy batches are a separate workflow:

```bash
sbake overnight submit episode.srt --mode economy --profile batch-profile
sbake overnight status /path/to/manifest.json --profile batch-profile
sbake overnight collect /path/to/manifest.json --profile batch-profile
```

The current implementation supports OpenAI Batch through `openai_chat` or
`openai_responses`. The saved manifest contains no API secret. Collection
verifies that the source subtitle has not changed before publishing output.

## Structured output and automation

Translation, QA, and evaluation workflows that support `--json` should be used
for scripts and CI. Treat documented JSON versions as contracts, preserve the
complete report, and record the SubBake build identity with benchmark results.

For commands not covered here or after upgrading, use the installed binary as
the source of truth:

```bash
sbake --help
sbake <COMMAND> --help
```

## Troubleshooting

### Provider validation fails

Run `sbake provider check` with the exact config/profile used by the workload.
Confirm the wire protocol, endpoint, model ID, API-key environment-variable
name, and timeout. Avoid placing a secret directly in a command because command
arguments may be retained by shell history or process inspection.

### Media translation reports no subtitle stream

`translate` only accepts supported text subtitle streams. Select the correct
stream with `--subtitle-stream`, or use `pipeline` when speech transcription is
required.

### Transcription cannot find whisper.cpp or a model

Run `sbake whisper status`. Install managed assets, configure
`whisper_binary_path` and `whisper_models_dir`, or pass `--whisper-bin` and
`--whisper-models-dir` explicitly.

### A run resumes unexpected work

Inspect its runtime state with `sbake runtime inspect <TARGET>`. Use
`--no-resume --no-cache` for a fully isolated translation, or clean only the
specific runtime categories that should be discarded.

### Output validation fails

Validation failures are intentional safety boundaries. Check missing/reordered
IDs, formatting markers, changed factual values, required glossary terms,
reading-speed limits, and line limits before retrying with weaker model stages.

## Development and evaluation

Run the required project checks:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Evaluation documentation:

- [Agent evaluation](agent-evaluation.md)
- [Subtitle and transcription evaluation](subtitle-evaluation.md)
