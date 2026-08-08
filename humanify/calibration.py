"""Domain calibration (Phase 6.2) — KB 01 §8 false-positive tuning.

Formal prose (academic/legal) is legitimately uniform (low burstiness), so it
false-positives on the burstiness / lexical layers. Short casual social text
needs a lower sentence-flag threshold. ``calibrate`` never removes existing
keys -- it only ADDS ``domain``, ``domain_warnings`` and ``domain_settings``.
"""

from __future__ import annotations

from typing import Any

# Per-domain calibration specs. ``burstiness_weight`` down-weights the
# lexical/burstiness signal for formal prose; ``lexical_weight_cap`` is the max
# weight a lexical axis score may contribute; ``sentence_threshold`` lowers the
# per-sentence flag bar for short casual text. Values are documented in
# ``domain_settings`` on the result so consumers can inspect them.
DOMAINS: dict[str, dict] = {
    "creative": {
        "label": "Creative / general prose (default)",
        "burstiness_weight": 1.0,
        "lexical_weight_cap": 1.0,
        "sentence_threshold": 0.5,
        "warnings": [],
    },
    "academic": {
        "label": "Academic / formal prose",
        "burstiness_weight": 0.5,
        "lexical_weight_cap": 0.65,
        "sentence_threshold": 0.5,
        # "formal prose may false-positive" is attached conditionally at runtime
        # when the text is suspiciously uniform (low burstiness).
        "warnings": [
            "Formal prose is legitimately uniform; low burstiness is expected "
            "and not necessarily AI - treat the burstiness signal as weak here.",
        ],
    },
    "legal": {
        "label": "Legal / regulatory text",
        "burstiness_weight": 0.5,
        "lexical_weight_cap": 0.65,
        "sentence_threshold": 0.5,
        "warnings": [
            "Legal text is deliberately uniform and formulaic; lexical and "
            "burstiness signals may false-positive - prefer structural/classifier "
            "evidence.",
        ],
    },
    "social": {
        "label": "Social media / short casual text",
        "burstiness_weight": 1.0,
        "lexical_weight_cap": 1.0,
        "sentence_threshold": 0.4,
        "warnings": [
            "Short casual text strips down per-sentence flag threshold so "
            "informal terse writing is not over-flagged.",
        ],
    },
}

# Burstiness (stats["std"]) below this for formal domains triggers the
# "formal prose may false-positive" warning (KB 02 S5: human 0.65-0.85, AI <0.30,
# but formal prose is legitimately uniform).
_FORMAL_LOW_BURSTINESS = 0.30


def calibrate(result: dict, domain: str = "creative") -> dict:
    """Adjust thresholds/tips for ``domain``; return a NEW enriched dict.

    Adds the (always-present) keys ``domain``, ``domain_warnings`` and
    ``domain_settings``. Input keys are never removed and the input dict is not
    mutated. Raises ``ValueError`` for an unknown domain.
    """
    if domain not in DOMAINS:
        raise ValueError(
            f"Unknown domain '{domain}'. Choose one of: "
            f"{', '.join(DOMAINS)}"
        )

    spec = DOMAINS[domain]
    adjusted = dict(result)  # copy; never mutate the caller's dict
    adjusted["domain"] = domain
    adjusted["domain_settings"] = _settings_for(spec)

    domain_warnings = list(spec.get("warnings", []))

    # Formal prose that is suspiciously uniform => explicit false-positive warning.
    stats = result.get("stats") or {}
    burstiness = stats.get("std")
    if (
        domain in ("academic", "legal")
        and isinstance(burstiness, (int, float))
        and burstiness < _FORMAL_LOW_BURSTINESS
    ):
        domain_warnings.append(
            "Low burstiness detected in formal prose (std "
            f"{burstiness:.2f} < {_FORMAL_LOW_BURSTINESS:.2f}): uniformity is "
            "expected in this domain and may false-positive - interpret the "
            "AI verdict with caution."
        )
    elif domain == "social" and isinstance(burstiness, (int, float)):
        # Short casual text is often uneven; don't penalize for being too
        # variable either, but no direct warning here.
        pass

    adjusted["domain_warnings"] = domain_warnings
    return adjusted


def _settings_for(spec: dict) -> dict:
    """Expose the tunable sliders so downstream output can cite them."""
    return {
        "burstiness_weight": spec["burstiness_weight"],
        "lexical_weight_cap": spec["lexical_weight_cap"],
        "sentence_threshold": spec["sentence_threshold"],
    }
