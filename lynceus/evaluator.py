"""Batch eval runner: loads samples, scores with detectors + writing
metrics, outputs JSONL."""
from __future__ import annotations

from typing import Any
import sys

from .cli import _load_detectors, _score_text
from .eval_io import (
    DuplicateIdError,
    load_eval_samples,
    load_preference_pairs,
    parse_eval_samples_jsonl,
    json_dumps_record,
)
from .eval_scoring import (
    aggregate_samples,
    component_scores,
    composite_score,
    score_preference_pairs,
)
from .output import render_eval_summary
from .writing_metrics import compute_writing_metrics


def run_eval(
    input_path: str,
    preferences_path: str | None,
    *,
    model: str,
    model_id: str | None,
    batch_size: int,
    device: str,
    output_stream,
    error_stream,
) -> int:
    """Run batch evaluation.

    Returns 0 on success, 1 on detector/scoring error, 2 on input validation error.
    """
    try:
        if input_path == "-":
            stdin_text = sys.stdin.read()
            samples = parse_eval_samples_jsonl(stdin_text, "<stdin>")
        else:
            try:
                samples = load_eval_samples(input_path)
            except FileNotFoundError as exc:
                error_stream.write(f"Error: {exc}\n")
                return 2
    except (ValueError, DuplicateIdError) as exc:
        error_stream.write(f"Error: {exc}\n")
        return 2

    # Step 2: Load optional preferences
    pairs = None
    if preferences_path is not None:
        try:
            pairs = load_preference_pairs(preferences_path)
        except (ValueError, FileNotFoundError, OSError) as exc:
            error_stream.write(f"Error: {exc}\n")
            return 2

    # Step 3: Load detectors
    model_names = [m.strip() for m in model.split(",")]
    try:
        loaded_detectors = _load_detectors(
            model_names, model_id=model_id, device_name=device, suppress_stdout=True
        )
    except ValueError as exc:
        error_stream.write(f"Error: {exc}\n")
        return 1

    # Step 4: Score each sample
    sample_records: list[dict[str, Any]] = []
    for sample in samples:
        try:
            result = _score_text(
                sample.candidate,
                loaded_detectors=loaded_detectors,
                batch_size=batch_size,
            )
        except Exception as exc:
            error_stream.write(f"Error scoring sample {sample.id}: {exc}\n")
            return 1

        metrics = compute_writing_metrics(
            sample.prompt,
            sample.candidate,
            sample.reference,
        )
        components = component_scores(metrics, detector_stats=result["stats"])
        composite = composite_score(components)

        record: dict[str, Any] = {
            "schema_version": 1,
            "type": "sample",
            "id": sample.id,
            "metrics": {
                "raw": metrics["raw"],
                "normalized": metrics["normalized"],
            },
            "component_scores": components,
            "composite_score": composite["composite"],
            "weights_applied": composite["weights_applied"],
            "detectors": [
                {
                    "name": d["name"],
                    "model_id": d["model_id"],
                    "stats": d_result["stats"],
                }
                for d, d_result in zip(loaded_detectors, result["detector_results"])
            ],
            "errors": [],
        }
        sample_records.append(record)
        output_stream.write(json_dumps_record(record) + "\n")

    # Step 5: Aggregate and optional preference scoring
    agg = aggregate_samples(sample_records)

    correlation_result = None
    if pairs is not None:
        correlation_result = score_preference_pairs(
            pairs,
            lambda p, c, r: {
                "composite_score": composite_score(
                    component_scores(compute_writing_metrics(p, c, r))
                )["composite"]
            },
        )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "type": "summary",
        "sample_count": agg["sample_count"],
        "valid_count": agg["valid_count"],
        "invalid_count": agg["invalid_count"],
        "duplicate_id_count": agg.get("duplicate_id_count", 0),
        "metrics": agg["metrics"],
        "weights": {
            "preference_rubric": 0.40,
            "readability_style": 0.25,
            "repetition_variety": 0.20,
            "lynceus_diagnostic": 0.15,
        },
        "preference_correlation": correlation_result
        or {
            "pair_count": 0,
            "non_tie_count": 0,
            "tie_count": 0,
            "agreement": None,
            "pearson": None,
            "spearman": None,
        },
    }
    output_stream.write(json_dumps_record(summary) + "\n")

    render_eval_summary(
        sample_records, summary, model_names, correlation_result
    )

    return 0
