"""Eval data I/O: JSONL schema, validation, and loading."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal

SCHEMA_VERSION = 1

__all__ = [
    "SCHEMA_VERSION",
    "DuplicateIdError",
    "EvalSample",
    "PreferencePair",
    "parse_eval_samples_jsonl",
    "load_eval_samples",
    "parse_preference_pairs_jsonl",
    "load_preference_pairs",
    "safe_float",
    "json_dumps_record",
]


class DuplicateIdError(ValueError):
    """Raised when duplicate eval sample IDs are found in parsed data."""

    def __init__(self, duplicate_id: str) -> None:
        self.duplicate_id = duplicate_id
        super().__init__(f"Duplicate eval sample ID: {duplicate_id!r}")


@dataclass
class EvalSample:
    """A single eval sample: prompt, candidate text, optional reference."""

    id: str
    prompt: str
    candidate: str
    reference: str | None = None


@dataclass
class PreferencePair:
    """A pair of candidates with a preference judgment."""

    id: str
    prompt: str
    candidate_a: str
    candidate_b: str
    preferred: Literal["a", "b", "tie"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _require_nonempty_string(
    obj: dict, field: str, line: int, source: str
) -> str:
    """Return *obj[field]* or raise ``ValueError``."""
    val = obj.get(field)
    if not isinstance(val, str) or val == "":
        if not isinstance(val, str):
            msg = (
                f"{field} on line {line} of {source} is not a string"
                f" (got {type(val).__name__})"
            )
        else:
            msg = f"empty {field} on line {line} of {source}"
        raise ValueError(msg)
    return val


def _parse_jsonl_lines(
    text: str,
    source: str,
    *,
    required_fields: list[str],
    optional_fields: list[str] = [],
    extra_validator: None
    | (callable) = None,
) -> list[dict]:
    """Parse *text* as JSONL and return a list of validated dicts.

    Blank lines are silently ignored.  On the first malformed/missing/empty
    row a ``ValueError`` is raised immediately (no partial output).
    """
    records: list[dict] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue

        try:
            obj: dict = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {lineno} of {source}: {exc}"
            ) from exc

        if not isinstance(obj, dict):
            raise ValueError(
                f"Line {lineno} of {source} is not a JSON object"
            )

        # Validate required fields
        for field in required_fields:
            _require_nonempty_string(obj, field, lineno, source)

        # Validate optional fields (must be string when present)
        for field in optional_fields:
            val = obj.get(field)
            if val is not None and not isinstance(val, str):
                raise ValueError(
                    f"{field} on line {lineno} of {source} must be a string"
                    f" or null, got {type(val).__name__}"
                )

        if extra_validator is not None:
            extra_validator(obj, lineno, source)

        records.append(obj)

    return records


# ---------------------------------------------------------------------------
# Eval samples
# ---------------------------------------------------------------------------


def _validate_eval_record(obj: dict, lineno: int, source: str) -> None:
    """Validate the ``preferred`` field for a preference record."""
    _ = obj, lineno, source  # no extra validation for eval samples


def parse_eval_samples_jsonl(text: str, source: str) -> list[EvalSample]:
    """Parse JSONL *text* into a list of :class:`EvalSample`.

    Blank lines are silently ignored.  Raises ``ValueError`` (or
    :class:`DuplicateIdError`) on the first problem.
    """
    records = _parse_jsonl_lines(
        text,
        source,
        required_fields=["id", "prompt", "candidate"],
        optional_fields=["reference"],
    )

    # Duplicate detection (after all basic validation passes)
    seen: set[str] = set()
    for rec in records:
        sid = rec["id"]
        if sid in seen:
            raise DuplicateIdError(sid)
        seen.add(sid)

    return [
        EvalSample(
            id=rec["id"],
            prompt=rec["prompt"],
            candidate=rec["candidate"],
            reference=rec.get("reference"),
        )
        for rec in records
    ]


def load_eval_samples(path: str) -> list[EvalSample]:
    """Read a JSONL file and return parsed :class:`EvalSample` list."""
    import pathlib

    text = pathlib.Path(path).read_text(encoding="utf-8")
    return parse_eval_samples_jsonl(text, source=path)


# ---------------------------------------------------------------------------
# Preference pairs
# ---------------------------------------------------------------------------


def _validate_preference_record(obj: dict, lineno: int, source: str) -> None:
    """Validate the ``preferred`` field for a preference record."""
    pref = obj.get("preferred")
    if pref not in ("a", "b", "tie"):
        raise ValueError(
            f"invalid preferred value {pref!r} on line {lineno} of {source}"
            f" (expected 'a', 'b', or 'tie')"
        )


def parse_preference_pairs_jsonl(text: str, source: str) -> list[PreferencePair]:
    """Parse JSONL *text* into a list of :class:`PreferencePair`.

    Blank lines are silently ignored.  Raises ``ValueError`` on
    the first malformed/missing/empty row.
    """
    records = _parse_jsonl_lines(
        text,
        source,
        required_fields=["id", "prompt", "candidate_a", "candidate_b", "preferred"],
        extra_validator=_validate_preference_record,
    )

    return [
        PreferencePair(
            id=rec["id"],
            prompt=rec["prompt"],
            candidate_a=rec["candidate_a"],
            candidate_b=rec["candidate_b"],
            preferred=rec["preferred"],  # type: ignore[arg-type]
        )
        for rec in records
    ]


def load_preference_pairs(path: str) -> list[PreferencePair]:
    """Read a JSONL file and return parsed :class:`PreferencePair` list."""
    import pathlib

    text = pathlib.Path(path).read_text(encoding="utf-8")
    return parse_preference_pairs_jsonl(text, source=path)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def safe_float(value: float | None) -> float | None:
    """Return *value* coerced to float, or ``None`` for NaN/Inf/None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (ValueError, TypeError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _sanitize_nan(obj: object) -> object:
    """Recursively replace NaN/Inf with ``None`` for JSON-safe serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def json_dumps_record(record: dict) -> str:
    """Serialize *record* to a stable, deterministic JSON string
    with NaN/Inf as null."""
    sanitized = _sanitize_nan(record)
    return json.dumps(sanitized, sort_keys=True, ensure_ascii=False)
