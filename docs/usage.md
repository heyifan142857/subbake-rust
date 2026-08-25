# SubBake usage guide

This guide describes the current Rust CLI. SubBake is an independent codebase;
older Python command shapes are not authoritative.

## What SubBake does

SubBake has five primary workflows:

1. Translate subtitle or text files.
2. Translate an existing text or bitmap subtitle stream inside a media container.
3. Transcribe audio or video with local whisper.cpp.
4. Safely refine a translated subtitle with deterministic validation and a
   previewable line diff.
5. Run an interactive terminal agent that can inspect a project and invoke the
   same translation and transcription services under approval and file-safety
   rules.

`translate` never turns media audio into subtitles. Use `transcribe` for
speech-to-text only, or `pipeline` for transcription followed by translation.

## Installation

### Precompiled Linux x64 archive

Preview releases provide an x86-64 GNU/Linux archive for glibc 2.35 or newer.
Download both release assets, verify the checksum, and then install the binary:

```bash
release_version="0.2.0-alpha.1"
archive="subbake-v${release_version}-x86_64-unknown-linux-gnu.tar.gz"
curl -LO "https://github.com/heyifan142857/subbake-rust/releases/download/v${release_version}/${archive}"
curl -LO "https://github.com/heyifan142857/subbake-rust/releases/download/v${release_version}/SHA256SUMS"
sha256sum --check --ignore-missing SHA256SUMS
tar -xzf "$archive"
install -Dm755 "subbake-v${release_version}-x86_64-unknown-linux-gnu/sbake" "$HOME/.local/bin/sbake"
```

The archive does not bundle FFmpeg, bubblewrap, Tesseract, whisper.cpp, or
Whisper models.

### Install from source

Install Rust 1.88 or newer. Official CI and release builds pin Rust 1.97.0.
FFmpeg and ffprobe are required for media inspection, transcription preparation,
and embedded subtitle operations.

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

### FFmpeg for media operations

FFmpeg supplies both the `ffmpeg` and `ffprobe` commands used by SubBake.
Install the package for your platform:

```bash
# Debian / Ubuntu
sudo apt install ffmpeg

# Arch Linux
sudo pacman -S ffmpeg

# Fedora (official repository build)
sudo dnf install ffmpeg-free

# macOS with Homebrew
brew install ffmpeg
```

Fedora's official `ffmpeg-free` package provides both commands but intentionally
supports fewer codecs than some third-party FFmpeg builds. If SubBake reports a
missing decoder after installation, use an FFmpeg build permitted by your
system's repository and codec policies.

On Windows, install an FFmpeg distribution and add its `bin` directory to
`PATH`. Verify that both commands are visible before processing media:

```bash
ffmpeg -version
ffprobe -version
```

If either command is missing, SubBake and the interactive Agent name the
missing dependency and verification commands instead of returning a raw
process-spawn error. Platform-specific installation commands remain in this
guide rather than being hard-coded into runtime errors.

### Tesseract for bitmap subtitles

Translating PGS, VobSub, or DVB bitmap subtitles requires the external
Tesseract executable and trained language data for the language shown in the
source subtitle. The source language data is required, not merely the target
translation language.

Install the engine plus the languages you need. For example, for English and
Simplified Chinese:

```bash
# Debian / Ubuntu
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim

# Arch Linux
sudo pacman -S tesseract tesseract-data-eng tesseract-data-chi_sim

# Fedora
sudo dnf install tesseract tesseract-langpack-eng tesseract-langpack-chi_sim

# macOS with Homebrew; tesseract-lang supplies additional languages
brew install tesseract tesseract-lang
```

On Windows, install a Tesseract 5 distribution, add `tesseract.exe` to `PATH`,
and install the required `.traineddata` files in its `tessdata` directory.
Verify both the executable and languages before starting a long translation:

```bash
tesseract --version
tesseract --list-langs
```

Common Tesseract language identifiers include `eng`, `chi_sim`, `chi_tra`,
`jpn`, and `kor`. If the language list does not contain the identifier needed
by the selected subtitle stream, install that language package and retry. See
the [official Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)
for other operating systems and language packages.

### Platform support

SubBake is developed and maintained primarily on Linux. Windows and macOS are
tested on real GitHub-hosted runners, but remain experimental support targets.

| Platform | Status | Tested behavior and limitations |
| --- | --- | --- |
| Linux x64 | Primary | Full CLI, TUI, transcription, and the bubblewrap-backed agent `run_command` tool |
| Windows x64 | Experimental | Translation, transcription, managed prebuilt whisper.cpp, and TUI; no agent command sandbox |
| macOS arm64 and Intel | Experimental | Translation, transcription, source-built whisper.cpp, and TUI; no agent command sandbox |

Windows ARM, every GPU/CUDA/Metal combination, every terminal emulator, and
arbitrary system FFmpeg codec builds are outside the current compatibility
guarantee. Platform bug reports should include the OS version, CPU architecture,
terminal, FFmpeg and whisper.cpp versions, and the smallest reproducing command.

## Configuration

### Discovery and precedence

SubBake resolves configuration in this order:

1. Built-in defaults.
2. The `[defaults]` section of the selected configuration file.
3. The selected `[profiles.<name>]` section.
4. Explicit CLI flags.

Use `--config <PATH>` to select a file directly. Without that flag, SubBake
checks the user configuration location and then the current directory:

- `%APPDATA%\subbake\config.toml` on Windows;
- `$XDG_CONFIG_HOME/subbake/config.toml` on non-Windows systems when set;
- otherwise `$HOME/.config/subbake/config.toml` on non-Windows systems;
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
vad_enabled = true
vad_model = "silero-v6.2.0"
vad_threshold = 0.5
vad_min_speech_duration_ms = 250
vad_min_silence_duration_ms = 100
vad_speech_pad_ms = 30

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

Supported subtitle/text formats include SRT, VTT, ASS/SSA, TTML/DFXP, and TXT.

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
`--request-token-budget`, `--confirmed-context-lines`,
`--confirmed-context-token-budget`, `--translation-concurrency`, `--retries`,
`--glossary`, and `--timeout-seconds`. Run `sbake translate --help` for the
current flag surface.

### Translation modes

Modes are semantic presets rather than branding aliases:

| Mode | Primary goal | Default behavior |
| --- | --- | --- |
| `economy` | Cost and throughput | Large self-contained batches, strict semantic deduplication, one corrected retry before structural splitting, fewer model stages |
| `turbo` | Latency/quality balance | Adaptive concurrency, strict semantic deduplication, neighboring source context, bounded confirmed prior translations, lightweight name alignment |
| `cinema` | Quality and consistency | Scene-aware scheduling, strict semantic deduplication, cross-scene repeated-source consistency review, strict terminology preflight, online terminology, timing-aware full review, language-aware readability defaults |

Semantic deduplication requires normalized source text, neighboring source text, and available
speaker, ASS style/layer, and cue-setting metadata to match. Cinema additionally exposes other
occurrences of the same source text to the reviewer as read-only consistency context. Equivalent
occurrences should remain identical, while speaker-, purpose-, register-, or scene-dependent
differences remain allowed.

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

`request_token_budget` caps the estimated complete prompt plus JSON response and
splits an oversized batch before a provider side effect. `confirmed_context_lines`
and `confirmed_context_token_budget` bound rolling translated context. Cinema
fails when requested terminology preflight is unavailable or fails; set
`allow_degraded_preflight = true` or pass `--allow-degraded-preflight` to opt in
to the previous best-effort behavior.

### Review and readability limits

Enable targeted or full model review:

```bash
sbake translate episode.srt --review targeted
sbake translate episode.srt --review full
```

Existing output files are preserved by default. Replace one explicitly with
`--overwrite`:

```bash
sbake translate episode.srt --overwrite
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

## Safely edit translated subtitles

Refine a generated translation without changing subtitle IDs, order, timing,
formatting markers, factual numbers, required glossary terms, or configured
readability limits:

```bash
sbake edit episode.translated.srt \
  --instruction "make the dialogue more conversational"
```

Preview and validate the complete proposed edit without writing the file:

```bash
sbake edit episode.translated.srt \
  --instruction "shorten lines that read too quickly" \
  --dry-run
```

The command prints a per-entry before/after diff. Generated
`.translated.*`/`.bilingual.*` names are required by default; use
`--allow-non-generated` only when intentionally editing another subtitle.
When `--glossary` is supplied, its terms are hard validation requirements.

## Translate an embedded subtitle stream

MKV, MP4/M4V/MOV, and WebM inputs may be used when they already contain a
supported text subtitle stream or a PGS, VobSub, or DVB bitmap subtitle stream:

```bash
sbake translate movie.mkv \
  --subtitle-stream 3 \
  --target-lang English
```

Text subtitles are extracted directly. Bitmap subtitles are rendered and OCRed
with Tesseract while retaining their timing, then pass through the normal
translation and validation pipeline. SubBake copies existing streams, verifies
the output, and atomically replaces the source container by default. Preserve
the source and write a separate translated container with:

```bash
sbake translate movie.mkv \
  --subtitle-stream 3 \
  --preserve-source-container \
  --output movie.translated.mkv
```

If the media has neither a supported text track nor a usable bitmap track,
`translate` reports that boundary instead of silently transcribing audio. Use
`pipeline` only when speech recognition is actually intended.

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
sbake whisper vad-model list
sbake whisper vad-model
sbake whisper update
```

Supported build variants are `cpu`, `cuda`, `metal`, `vulkan`, and `openblas`.
Use `--bin`, `--models-dir`, and `--runtime-dir` to override managed locations.
Silero VAD is enabled by default. `sbake whisper vad-model` downloads the
default `silero-v6.2.0` model into the same managed model directory. VAD files
are tracked separately and are never considered transcription models.

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

Transcription also preserves an existing output unless `--overwrite` is
passed explicitly.

Disable VAD for a comparison run with `--no-vad`, or tune it with
`--vad-threshold`, `--vad-min-speech-duration-ms`,
`--vad-min-silence-duration-ms`, and `--vad-speech-pad-ms`.

Use an existing timed sidecar instead of running whisper.cpp:

```bash
sbake transcribe interview.mp4 \
  --sidecar transcript.vtt \
  --format srt \
  --output interview.srt
```

Long media is processed in overlapping chunks. SubBake merges boundary text,
checks coverage, and does not publish an obviously incomplete final subtitle.
The same VAD contract is applied to short audio, every long-audio chunk, and
resumed streaming pipelines. For container media, the selected audio stream's
`start_time` is added back after FFmpeg normalization so a delayed audio track
does not produce a globally early subtitle.

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

Output-producing commands can apply the same threshold before publication:

```bash
sbake translate episode.srt --qa-fail-on warning
sbake batch season/ --recursive --qa-fail-on error
sbake transcribe interview.mp4 --qa-fail-on warning
sbake pipeline movie.mkv --qa-fail-on error
```

`never` is the default. `error` blocks structural timing failures; `warning`
also blocks readability findings. A failed gate leaves the requested output
unpublished, while JSON results include the QA report for successful runs.

Transcription post-processing normalizes repeated whitespace and spaces before
punctuation by default. It also recognizes Whisper-style prefixes such as
`[SPEAKER_01]`, preserves the visible label, and records it as structured
speaker metadata for translation context. Use `--no-normalize-transcript` or
`--no-speaker-labels` to disable either behavior; profile configuration accepts
`transcription.normalize_text` and `transcription.speaker_labels` as well.

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

## Project and season preflight

Before translating a season or a multi-file project, build an inventory and
consistency report:

```bash
sbake project season/ --recursive --output season-report.json
sbake project season/ --recursive --json
sbake project season/ --recursive --fail-on warning
```

The versioned manifest classifies source files as pending, translated, or
bilingual; records segment counts and QA findings; verifies source/output ID
alignment; and detects identical source lines with divergent translations
across episodes. `--fail-on` makes QA and consistency findings suitable for CI.

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

Command names, summaries, primary options, and subcommands come from one
declarative command specification. Generate completion from that same source:

```bash
sbake completion bash
sbake completion zsh
sbake completion fish
sbake completion powershell
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
