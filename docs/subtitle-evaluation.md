# Subtitle and transcription evaluation

Subtitle quality is a separate gate from agent task completion. A release can
pass agent tool/approval scenarios and still fail translation or transcription
quality.

## Current implementation status

- The deterministic Rust APIs and regression tests described below are
  implemented and run in the normal workspace test suite.
- `sbake qa` exposes reference-free timing/readability checks, and
  `sbake evaluate` exposes the older reference chrF and mechanical MQM subset.
  The complete translation-hard-constraint and transcription reports are Rust
  APIs and are not yet wired into one complete CLI command.
- `scripts/mt_quality.py`, its optional dependency list, and aligned sample
  input are implemented. The repository does not contain downloaded COMET or
  XCOMET weights, a completed model run, or a committed learned-metric score.
- The human MQM section is an annotation protocol only. No human annotations,
  adjudication results, or inter-annotator agreement measurements have been
  produced as part of this implementation.

## Deterministic translation gate

`subbake_core::evaluate_translation_quality` returns four independent sections:

- `hard_constraints`: exact segment count and ID order, no duplicate IDs,
  unchanged timestamps and formatting markers, preserved numbers/dates/amounts/
  percentages, and required glossary targets. `passed` must be `true`; this is a
  100% gate, not an average score.
- `reference`: the existing exact-match, chrF (0–1), and mechanical MQM counts
  when a reference document is supplied.
- `document_consistency`: explicit person-name, terminology, pronoun, and
  honorific rules, including missing targets and variant drift.
- `readability`: reference-free empty/timing/overlap/repetition, characters per
  second, characters per line, and line-count checks.

The data-driven regression cases are in
`crates/subbake-core/tests/subtitle_quality_scenarios.rs`. Run them with:

```bash
cargo test -p subbake-core --test subtitle_quality_scenarios
```

## Transcription gate

`subbake_core::evaluate_transcription` reports WER, whitespace-free CER,
ID-aligned mean/max timestamp boundary offsets, reference-speech coverage,
candidate overlap count/duration, maximum characters per second, and maximum
characters per line. Keep language-specific normalization and thresholds in the
evaluation manifest used by the experiment; do not silently change a baseline.

## Optional COMET, XCOMET, and SacreBLEU

Learned metrics are intentionally excluded from normal CI because model
downloads are large and scores can depend on package/checkpoint versions. Create
an isolated Python environment, install the optional dependencies, and run the
aligned JSONL evaluator:

```bash
python -m venv .eval-venv
.eval-venv/bin/pip install -r requirements/evaluation.txt
.eval-venv/bin/python scripts/mt_quality.py \
  crates/subbake-core/tests/fixtures/mt_quality.sample.jsonl \
  --output target/evaluation/mt-quality.json
```

Rows require `id`, `src`, and `mt`; either every row has `ref` or none do. In
`auto` mode the script uses `Unbabel/wmt22-comet-da` with references and
`Unbabel/wmt22-cometkiwi-da` without references. Pass
`--xcomet-model Unbabel/XCOMET-XL` to collect explainable error spans where the
selected checkpoint exposes them. Model licenses and language coverage must be
checked before adopting a checkpoint.

When references are present, the output saves SacreBLEU's chrF signature and
uses its native 0–100 scale. SubBake's Rust chrF remains on 0–1, so compare
after dividing the SacreBLEU score by 100. Store the full JSON report with the
dataset revision, package lock, checkpoint name, language pair, and hardware
metadata.

## Human MQM sampling

COMET/QE is a prioritization signal, not a replacement for human review. Sample
at least the lowest-scoring segments, every hard-constraint failure, all
XCOMET critical/major spans, and a random control slice. For each annotation
record evaluator, blinded system ID, document/segment ID, MQM category,
severity (`critical`, `major`, `minor`), source span, target span, and note.
Report error points per 1,000 target words and inter-annotator agreement; retain
raw annotations so rubric changes can be recomputed.
