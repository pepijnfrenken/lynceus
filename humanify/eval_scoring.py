"""Deterministic composite scoring, aggregation, and preference correlation."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from .eval_io import PreferencePair, safe_float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def safe_mean(values: list[float | None]) -> float | None:
    """Mean ignoring None values; return None if all are None or list empty."""
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def _rank(values: list[float]) -> list[float]:
    """Assign ranks (average for ties)."""
    n = len(values)
    if n == 0:
        return []

    indexed = sorted(enumerate(values), key=lambda x: x[1])

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-based average rank
        for k in range(i, j):
            original_idx = indexed[k][0]
            ranks[original_idx] = avg_rank
        i = j

    return ranks


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Standard Pearson correlation coefficient.

    Returns None when fewer than 2 points or either variable has zero variance.
    """
    n = len(xs)
    if n < 2 or len(ys) < 2:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    if var_x == 0 or var_y == 0:
        return None

    r = cov / (math.sqrt(var_x) * math.sqrt(var_y))
    return max(-1.0, min(1.0, r))


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation using local rank assignment.

    Returns None when fewer than 2 points or either variable has zero variance.
    """
    n = len(xs)
    if n < 2 or len(ys) < 2:
        return None

    rx = _rank(xs)
    ry = _rank(ys)

    return pearson(rx, ry)


# ---------------------------------------------------------------------------
# Component scoring
# ---------------------------------------------------------------------------


def component_scores(
    metrics: dict,
    detector_stats: dict | None = None,
) -> dict:
    """Compute component scores from writing metrics and optional detector stats.

    Parameters
    ----------
    metrics:
        The dict returned by ``compute_writing_metrics()``, containing
        ``"raw"`` and ``"normalized"`` keys.
    detector_stats:
        Optional dict with a ``"mean"`` key (the mean detector score
        across models). When provided, ``humanify_diagnostic`` is
        ``1.0 - detector_stats["mean"]``.

    Returns
    -------
    dict with keys ``preference_rubric``, ``readability_style``,
    ``repetition_variety``, and ``humanify_diagnostic``.
    """
    norm = metrics.get("normalized", {})

    # --- Preference rubric (weighted composite of all normalized metrics) ---
    _preference_weights: dict[str, float] = {
        "coherence_rubric": 0.45,
        "style_consistency": 0.25,
        "readability": 0.15,
        "lexical_diversity": 0.10,
        "repetition": 0.05,
    }

    weighted_sum = 0.0
    weight_sum = 0.0
    for key, weight in _preference_weights.items():
        val = norm.get(key)
        if val is not None:
            weighted_sum += weight * val
            weight_sum += weight

    preference_rubric: float | None = (
        weighted_sum / weight_sum if weight_sum > 0 else None
    )

    # --- Readability-Style (simple mean) ---
    readability_style = safe_mean(
        [
            norm.get("readability"),
            norm.get("lexical_diversity"),
            norm.get("style_consistency"),
        ]
    )

    # --- Repetition-Variety ---
    repetition_variety = safe_mean(
        [
            norm.get("repetition"),
            norm.get("sentence_variety"),
        ]
    )

    # --- Humanify diagnostic ---
    humanify_diagnostic: float | None = None
    if detector_stats is not None:
        mean_det = detector_stats.get("mean")
        if mean_det is not None:
            humanify_diagnostic = 1.0 - mean_det

    return {
        "preference_rubric": preference_rubric,
        "readability_style": readability_style,
        "repetition_variety": repetition_variety,
        "humanify_diagnostic": humanify_diagnostic,
    }


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


def composite_score(components: dict) -> dict:
    """Compute final composite score from component scores.

    Formula: ``0.40*preference_rubric + 0.25*readability_style +
    0.20*repetition_variety + 0.15*humanify_diagnostic``.

    When ``humanify_diagnostic`` is None the remaining weights are
    renormalized (divide by 0.85).  Returns a dict with ``"composite"``
    (``float | None``) and ``"weights_applied"`` (``list[str]``).
    """
    _weights: dict[str, float] = {
        "preference_rubric": 0.40,
        "readability_style": 0.25,
        "repetition_variety": 0.20,
        "humanify_diagnostic": 0.15,
    }

    composite = 0.0
    total_weight = 0.0
    weights_applied: list[str] = []

    for key, weight in _weights.items():
        val = components.get(key)
        if val is not None:
            composite += weight * val
            total_weight += weight
            weights_applied.append(key)

    if total_weight == 0:
        return {"composite": None, "weights_applied": []}

    return {
        "composite": composite / total_weight,
        "weights_applied": weights_applied,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _flatten_metrics(
    metrics: dict,
    prefix: str = "",
) -> dict[str, float | None]:
    """Flatten nested metric dict to dotted-key scalars.

    Dict values are flattened recursively.  List values are summarised as
    their mean.  Other values are passed through ``safe_float``.
    """
    result: dict[str, float | None] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_metrics(value, full_key))
        elif isinstance(value, list):
            safe_vals = [safe_float(v) for v in value]
            non_none = [v for v in safe_vals if v is not None]
            result[full_key] = (
                sum(non_none) / len(non_none) if non_none else None
            )
        else:
            result[full_key] = safe_float(value)
    return result


def aggregate_samples(sample_records: list[dict]) -> dict:
    """Compute aggregate statistics across sample records.

    For each metric key in the first valid record's metrics (both
    ``"raw"`` and ``"normalized"`` levels), computes mean, population
    standard deviation, minimum, and maximum across all valid records.
    """
    sample_count = len(sample_records)
    valid = [r for r in sample_records if not r.get("errors")]
    valid_count = len(valid)
    invalid_count = sample_count - valid_count

    result: dict[str, Any] = {
        "sample_count": sample_count,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "duplicate_id_count": 0,
        "metrics": {},
        "weights": {
            "preference_rubric": 0.40,
            "readability_style": 0.25,
            "repetition_variety": 0.20,
            "humanify_diagnostic": 0.15,
        },
        "preference_correlation": {},
    }

    if not valid:
        return result

    # Flatten each valid record's metrics into dotted-key scalars
    flat_records: list[dict[str, float | None]] = []
    for rec in valid:
        metrics = rec.get("metrics", {})
        flat: dict[str, float | None] = {}
        for level in ("raw", "normalized"):
            level_data = metrics.get(level, {})
            if isinstance(level_data, dict):
                flat.update(_flatten_metrics(level_data, prefix=level))
        flat_records.append(flat)

    if not flat_records:
        return result

    # Compute per-path stats
    paths = sorted(flat_records[0].keys())
    stats: dict[str, dict[str, float | None]] = {}
    for path in paths:
        values = [rec.get(path) for rec in flat_records]
        cleaned = [v for v in values if v is not None]
        if cleaned:
            n = len(cleaned)
            mn = sum(cleaned) / n
            variance = sum((x - mn) ** 2 for x in cleaned) / n  # population
            sd = math.sqrt(variance)
            stats[path] = {
                "mean": mn,
                "std": sd,
                "min": min(cleaned),
                "max": max(cleaned),
            }
        else:
            stats[path] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }

    result["metrics"] = stats
    return result


# ---------------------------------------------------------------------------
# Preference pair scoring
# ---------------------------------------------------------------------------


def score_preference_pairs(
    pairs: list[PreferencePair],
    score_candidate: Callable[[str, str, str | None], dict],
) -> dict:
    """Score preference pairs and compute prediction agreement / correlation.

    Parameters
    ----------
    pairs:
        List of ``PreferencePair`` dataclass instances.
    score_candidate:
        Callable ``(prompt, candidate, reference) -> dict`` that returns
        a result dict containing ``"composite_score"``.

    Returns
    -------
    dict with keys ``pair_count``, ``non_tie_count``, ``tie_count``,
    ``agreement``, ``pearson``, and ``spearman``.
    """
    pair_count = len(pairs)

    score_deltas: list[float] = []
    human_deltas: list[float] = []

    correct_non_tie = 0
    non_tie_count = 0
    tie_count = 0
    for pair in pairs:
        score_a_result = score_candidate(
            pair.prompt, pair.candidate_a, None
        )
        score_b_result = score_candidate(
            pair.prompt, pair.candidate_b, None
        )

        composite_a = score_a_result["composite_score"]
        composite_b = score_b_result["composite_score"]

        # Round to 6 decimals for tie-breaking
        if composite_a is not None and composite_b is not None:
            ra = round(composite_a, 6)
            rb = round(composite_b, 6)
            if ra > rb:
                pred = "a"
            elif rb > ra:
                pred = "b"
            else:
                pred = "tie"
        else:
            pred = "tie"

        is_tie_ground = pair.preferred == "tie"

        if is_tie_ground:
            tie_count += 1
        else:
            non_tie_count += 1
            if pred == pair.preferred:
                correct_non_tie += 1

            score_deltas.append(composite_a - composite_b)
            human_deltas.append(1.0 if pair.preferred == "a" else -1.0)

    agreement: float | None = (
        correct_non_tie / non_tie_count if non_tie_count > 0 else None
    )

    pears: float | None = None
    spear: float | None = None
    if len(score_deltas) >= 2:
        pears = pearson(score_deltas, human_deltas)
        spear = spearman(score_deltas, human_deltas)

    return {
        "pair_count": pair_count,
        "non_tie_count": non_tie_count,
        "tie_count": tie_count,
        "agreement": agreement,
        "pearson": pears,
        "spearman": spear,
    }
