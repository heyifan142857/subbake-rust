#!/usr/bin/env python3
"""Run optional learned/reference MT metrics over aligned JSONL segments.

Each input line must contain ``id``, ``src`` and ``mt``. ``ref`` is optional,
but must be present on every row when using a reference-based COMET model.
This script is deliberately outside the default Rust test path: COMET downloads
large model checkpoints and may require a GPU for practical runtimes.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="aligned JSONL: id/src/mt[/ref]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--comet-model",
        default="auto",
        help=(
            "COMET checkpoint, 'auto' (wmt22-comet-da with refs, "
            "wmt22-cometkiwi-da without), or 'none'"
        ),
    )
    parser.add_argument(
        "--xcomet-model",
        help="optional explainable COMET checkpoint, for example Unbabel/XCOMET-XL",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=0)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("id", "src", "mt"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"line {line_number}: {field!r} must be a non-empty string")
            if "ref" in row and not isinstance(row["ref"], str):
                raise ValueError(f"line {line_number}: 'ref' must be a string")
            rows.append(row)
    if not rows:
        raise ValueError("input contains no evaluation rows")
    reference_count = sum(bool(row.get("ref")) for row in rows)
    if reference_count not in (0, len(rows)):
        raise ValueError("'ref' must be present on every row or omitted from every row")
    return rows


def sacrebleu_report(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    if not rows[0].get("ref"):
        return None
    try:
        from sacrebleu.metrics import CHRF
    except ImportError as error:
        raise RuntimeError("install requirements/evaluation.txt to compute chrF") from error
    metric = CHRF(char_order=6, word_order=0, beta=2)
    score = metric.corpus_score(
        [row["mt"] for row in rows],
        [[row["ref"] for row in rows]],
    )
    return {
        "score": score.score,
        "scale": "0-100",
        "signature": str(metric.get_signature()),
    }


def comet_report(
    rows: list[dict[str, str]],
    model_name: str,
    batch_size: int,
    gpus: int,
) -> dict[str, Any]:
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as error:
        raise RuntimeError("install requirements/evaluation.txt to compute COMET") from error
    data = [
        {key: row[key] for key in ("src", "mt", "ref") if key in row and row[key]}
        for row in rows
    ]
    checkpoint = download_model(model_name)
    model = load_from_checkpoint(checkpoint)
    prediction = model.predict(data, batch_size=batch_size, gpus=gpus)
    scores = [float(score) for score in prediction.scores]
    metadata = getattr(prediction, "metadata", None)
    error_spans = getattr(metadata, "error_spans", None)
    return {
        "model": model_name,
        "system_score": float(prediction.system_score),
        "mean_segment_score": statistics.fmean(scores),
        "segments": [
            {
                "id": row["id"],
                "score": score,
                **(
                    {"error_spans": json_safe(error_spans[index])}
                    if error_spans is not None
                    else {}
                ),
            }
            for index, (row, score) in enumerate(zip(rows, scores, strict=True))
        ],
    }


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return str(value)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.gpus < 0:
        raise ValueError("--batch-size must be positive and --gpus cannot be negative")
    rows = load_rows(args.input)
    has_reference = bool(rows[0].get("ref"))
    model_name = args.comet_model
    if model_name == "auto":
        model_name = (
            "Unbabel/wmt22-comet-da"
            if has_reference
            else "Unbabel/wmt22-cometkiwi-da"
        )
    report: dict[str, Any] = {
        "version": 1,
        "input": str(args.input),
        "segments": len(rows),
        "has_reference": has_reference,
        "sacrebleu_chrf": sacrebleu_report(rows),
    }
    if model_name != "none":
        report["comet"] = comet_report(rows, model_name, args.batch_size, args.gpus)
    if args.xcomet_model:
        report["xcomet"] = comet_report(
            rows, args.xcomet_model, args.batch_size, args.gpus
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
