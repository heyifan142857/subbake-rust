# Agent evaluation

SubBake evaluates the interactive agent separately from subtitle translation
quality. Agent tests answer whether the decision loop selected safe tools,
respected approval and state boundaries, and produced the expected artifact.
Translation evaluation answers whether the artifact is linguistically good.
Do not combine these into one score.

## Deterministic scenario suite

The normal workspace test run includes the versioned JSON cases under
`crates/subbake-agent/tests/scenarios`. Each case runs the real `AgentEngine`,
tool registry, file guard, session log, plan coordinator, and undo service with
a scripted decision backend. It makes no network requests.

Run it directly when editing prompts, tools, approval behavior, or the loop:

```bash
cargo test -p subbake-agent --test agent_scenarios -- --nocapture
```

The report includes case pass rate, model steps, tool calls, tool statuses, and
assertion failures. Successful outcomes are deliberately reduced to operation
and status; file contents read by a tool are not copied into the report.

A case may assert:

- required tool calls with a subset of their arguments;
- groups where any one of several valid tool calls may satisfy the task;
- minimum and maximum call counts and completion status;
- forbidden tools;
- a required tool subsequence, allowing other valid trajectories;
- required or forbidden session event kinds;
- final-response fragments and project-local file contents;
- model/tool step budgets and whether the step limit was reached.

Use exact full trajectories only when the order itself is a product invariant.
Usually a subsequence plus forbidden tools and final artifact assertions gives
a less brittle correctness definition.

The fixture `version` is a contract. Increment
`AGENT_EVAL_FORMAT_VERSION` and provide an explicit migration when the persisted
shape changes. Fixture paths must be project-relative and cannot contain `..`.

## Live decision-model suite

`live_agent_scenarios` reuses the same runner and assertion format but lets a
real model choose the trajectory. It is ignored by default because it requires
credentials and incurs model usage. Translation tool execution still uses the
configured SubBake translator; the included cases work with the default mock
translator so the score isolates agent decision quality.

Example for an OpenAI-compatible Responses endpoint:

```bash
export SUBBAKE_EVAL_PROVIDER=openai
export SUBBAKE_EVAL_MODEL=<decision-model>
export SUBBAKE_EVAL_API_FORMAT=openai_responses
export SUBBAKE_EVAL_API_KEY_ENV=OPENAI_API_KEY
export OPENAI_API_KEY=<secret>
export SUBBAKE_EVAL_REPETITIONS=3
cargo test -p subbake-agent --test live_agent_scenarios -- --ignored --nocapture
```

Optional endpoint variables are `SUBBAKE_EVAL_BASE_URL` and
`SUBBAKE_EVAL_ENDPOINT_URL`. Supported API format values are the same as the
normal configuration: `openai_chat`, `openai_responses`,
`anthropic_messages`, and `gemini_generate_content`.

The initial live quality gate is 95% across all case repetitions. Safety,
approval, path containment, output validation, and mutation idempotency should
remain deterministic 100% gates in the scripted suite rather than probabilistic
LLM-judge metrics.

For model comparisons, run the same corpus and repetition count for every
candidate. Record at least pass rate, all-runs-pass rate per case, average model
steps, tool-call success, latency, tokens, and cost. Do not update a baseline
from a single run.

## Property and fuzz testing

`proptest` generates tool arguments, output aliases, and traversal paths during
the standard Rust tests. The separate `cargo-fuzz` package targets subtitle
parsers/renderers and the model-visible tool-schema boundary:

```bash
cargo install cargo-fuzz
cargo fuzz run subtitle_parsers
cargo fuzz run agent_tool_validation
```

Keep minimized regressions as ordinary focused unit tests or reviewed corpus
inputs. Fuzz crashes alone are build artifacts and are ignored by Git.

## Adding cases

Every production failure should become the narrowest reproducible case:

1. Add a deterministic scripted case for the violated runtime invariant.
2. If the failure depended on model choice, add or extend a live case too.
3. Prefer canaries, file hashes, event ordering, and tool evidence to an LLM
   judge.
4. Use a rubric or semantic judge only for properties that code cannot decide,
   such as whether a final explanation is useful or a translation is natural.

High-priority corpus categories are discovery, explicit translation versus
transcription, ambiguity and `ask_user`, plan and command approval, cancellation,
resume after partial success, duplicate mutation, profile-switch rollback,
prompt injection in subtitles and command output, path/symlink escape, and
secret handling.

When trace volume becomes difficult to inspect locally, export the recorder's
structured trace to an OpenTelemetry-compatible system such as Phoenix. Ragas,
DeepEval, or Promptfoo can consume an exported trace as an additional scoring or
red-team layer; they should not replace the deterministic Rust gates.
