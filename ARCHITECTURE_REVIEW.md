# Interactive Agent Architecture Review

This document tracks issues found while reviewing the interactive-agent changes. Items are ordered by recommended implementation priority.

## P0 — Correctness

### 1. Profile picker submission conflicts with generic suggestions — fixed

Picker submission and slash completion are now mutually exclusive. A regression test verifies the typed profile-selection action, so picker behavior no longer depends on synthesizing `/profile <name>` text.

### 2. Configuration discovery has two sources of truth — fixed

The CLI now discovers configuration once at the composition edge, pins that path in the active session, and uses the same path for initial and replacement backends. Engine profile listing and model reporting prioritize the pinned session path. A backend factory remains desirable when implementing items 3 and 4.

### 3. Backend construction silently falls back to mock — fixed

Backend construction now returns an explicit error. Mock is used only when the resolved provider is deliberately `mock`; invalid provider configuration no longer changes the active backend or masquerades as a successful switch.

### 4. Profile switching is not atomic — fixed

Profile selection now follows this order:

1. Resolve and validate the target profile.
2. Construct its backend successfully.
3. Persist the session profile and profile event.
4. Swap the active backend.

Missing profiles do not build a candidate, and construction failures occur before the engine persists the profile.

## P1 — Interaction contracts

### 5. TUI actions are encoded as synthetic slash-command strings — fixed

The TUI/worker boundary now uses `TuiAction` variants for submitted text, plan approval/rejection, and profile selection. User-authored slash commands remain supported as text for headless compatibility, but picker and approval UI no longer synthesize them.

### 6. Input behavior is an implicit state machine — fixed

`InputMode` now makes editing, history browsing, profile selection, and pending-plan decisions mutually exclusive. Typing exits picker/decision/history modes explicitly, while Up/Down behavior is selected from the active mode.

### 7. TUI result data mixes domain state and presentation state — fixed

`TuiInteraction` now represents ordinary messages, plan approval requests, and profile pickers as mutually exclusive variants. The TUI derives its input mode directly from the interaction rather than combining a pending flag with optional picker data.

### 8. Input history is process-local — fixed

The TUI now seeds Up/Down history from persisted session `user` events, removes consecutive duplicates, and continues appending inputs during the current process. Engine-routed slash commands are recorded as user events; internal typed picker and approval actions are not treated as textual history.

### 9. Not every response should use character-by-character streaming — fixed

Short conversational answers benefit from animated streaming, but already-complete structured output should render immediately. Examples include file listings, profile/session pickers, help, diagnostics, tables, and tool results such as `ls`:

```text
["2026-07-10T07:06:44Z"] ls
⎿ Deciding next action…
➔ /home/azote/Codes/subbake-rust/.agents
/home/azote/Codes/subbake-rust/.git
...
```

`TuiInteraction::Message` now carries an explicit `RenderPolicy`. Approval and picker interactions render immediately, as do slash-command and multiline structured results; single-line conversational responses retain animated streaming. The composition layer owns the policy and the TUI only executes it.

## P2 — Reliability

### 10. Worker thread lifecycle is detached — fixed

The TUI now owns a named worker `JoinHandle`, disconnects its channels during shutdown, restores the visible terminal, and joins the worker. Esc uses a generation-based cancellation guard that reaches the agent loop, providers, translation pipeline, transcription, and cancellable child processes.

### 11. Terminal restoration is not RAII-protected — fixed

`TerminalSessionGuard` now owns raw-mode and alternate-screen state. Normal shutdown restores explicitly before waiting for the worker; initialization failures and early error/panic paths are covered by its `Drop` fallback.

### 12. Multi-tool plan approval can partially execute — fixed

After each successful tool call, the engine removes that call from the pending plan and persists the remaining calls using the existing storage shape. A later failure leaves only the failed and subsequent calls pending, so approval retries cannot repeat already-completed mutations. A regression test covers create-success/append-failure/retry behavior.

## Interaction validation — complete

The interaction state machine has regression coverage for profile selection and creation, pending-plan approve/reject/revise outcomes, revision-plan replacement, history navigation with draft restoration, and failed profile-backend construction before session mutation.

A real PTY smoke test additionally verified `/profile`, arrow-key navigation, entering the new-profile form, profile-name input, Esc cancellation back to chat, `/exit`, and terminal restoration. The remaining `crossterm` `poll`/`read` calls are library event-source plumbing rather than an application state boundary, so duplicating them with an in-process mock is not required.

## Deliberately deferred wiki items

- **Provider-call interruption:** completed. Esc uses a generation-based guard; provider HTTP futures, translation/editing calls, transcription, ffmpeg, and whisper.cpp receive it. The remaining limitation is external processes or remote servers that do not respond to termination promptly.
- **Profile picker `new`:** completed. The picker exposes a typed `new profile…` choice. Creation appends a validated effective-settings snapshot using an adjacent temporary file and rename, preserves the existing file and comments, omits inline API-key/auth-header credentials, and deliberately leaves the current profile active until the user reviews and selects the new profile.
- **Top-level `sbake resume`:** the Rust CLI intentionally uses `sbake agent resume [SESSION_ID]` per the repository CLI direction; the Python/wiki alias is not reintroduced.
