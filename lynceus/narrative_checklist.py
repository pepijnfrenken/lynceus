"""Narrative / discourse checklist (Phase 5.1) — KB 02 N1–N11.

Opt-in, *document-level* narrative analysis (KB 01 §7.2 / KB 02 §4). Asks an
injected causal-LLM callable to return a per-dimension verdict over the 11
STORYSCOPE narrative dimensions (theme over-explanation, causal linearity,
moral ambiguity, ...). Gated at a minimum word count because KB says
STORYSCOPE narrative features need *long* text — per-sentence narrative scores
are noise. The LLM call is **dependency-injected** (a plain ``Callable[[str],
str]``) so tests never touch a live model.
"""

import json
import re
from typing import Callable

# Default minimum word count before the narrative checklist is meaningful.
DEFAULT_MIN_WORDS = 500

# ---------------------------------------------------------------------------
# The 11 STORYSCOPE dimensions (KB 02 N1–N11, arXiv:2604.03136).
#
# ``direction`` says which side of the dimension reads as AI-authored:
#   * "ai_high"    -> the AI-prior attribute being PRESENT is the tell
#   * "human_high" -> AI text tends to LACK it, so its ABSENCE is the tell
# ``ai_prior`` / ``human_prior`` are the measured baselines from KB 02 (0-100
# percentages where measured, else the raw STORYSCOPE 1-5 means).
# ---------------------------------------------------------------------------
NARRATIVE_DIMENSIONS: list[dict] = [
    {
        "key": "theme_over_explanation",
        "label": "Theme over-explanation & moralizing",
        "ai_prior": 77.0,
        "human_prior": 52.0,
        "direction": "ai_high",
    },
    {
        "key": "causal_linearity",
        "label": "Tidy single-track causal chain (no subplots)",
        "ai_prior": 79.0,
        "human_prior": 57.0,
        "direction": "ai_high",
    },
    {
        "key": "moral_ambiguity",
        "label": "Morally ambiguous protagonist",
        "ai_prior": 38.0,
        "human_prior": 59.0,
        "direction": "human_high",
    },
    {
        "key": "temporal_complexity",
        "label": "Temporal complexity (flashbacks, nonlinearity)",
        "ai_prior": 2.12,
        "human_prior": 2.40,
        "direction": "human_high",
    },
    {
        "key": "subplot_independence",
        "label": "Independent / contrasting subplots",
        "ai_prior": 21.0,
        "human_prior": 42.0,
        "direction": "human_high",
    },
    {
        "key": "embodied_emotion",
        "label": "Emotion described through the body (over-performed)",
        "ai_prior": 81.0,
        "human_prior": 38.0,
        "direction": "ai_high",
    },
    {
        "key": "intertextual_explicitness",
        "label": "Explicit, named intertextuality",
        "ai_prior": 24.0,
        "human_prior": 47.0,
        "direction": "human_high",
    },
    {
        "key": "reader_address",
        "label": "Narrator directly addresses the reader",
        "ai_prior": 7.0,
        "human_prior": 28.0,
        "direction": "human_high",
    },
    {
        "key": "sensory_density",
        "label": "Sensory over-description / psychological-mirror setting",
        "ai_prior": 82.0,
        "human_prior": 57.0,
        "direction": "ai_high",
    },
    {
        "key": "protagonist_agency",
        "label": "Resolution driven by protagonist choice",
        "ai_prior": 69.0,
        "human_prior": 46.0,
        "direction": "ai_high",
    },
    {
        "key": "social_network",
        "label": "Rich social network / secondary characters / gossip",
        "ai_prior": 30.0,
        "human_prior": 50.0,
        "direction": "human_high",
    },
]

_DIMENSIONS_BY_KEY: dict[str, dict] = {d["key"]: d for d in NARRATIVE_DIMENSIONS}


def build_narrative_prompt(text: str) -> str:
    """Build a concise instruction asking for a JSON verdict per dimension.

    The expected response is a JSON object mapping each dimension key to
    ``{"present": bool, "evidence": "one-sentence snippet"}``.
    """
    dim_lines = "\n".join(
        f'- "{d["key"]}" ({d["label"]}). Human baseline {d["human_prior"]} vs '
        f'AI baseline {d["ai_prior"]}.'
        for d in NARRATIVE_DIMENSIONS
    )
    return (
        "You are a narrative analyst. Read the story below and judge the "
        f"presence of {len(NARRATIVE_DIMENSIONS)} narrative features grounded in "
        "STORYSCOPE human-vs-AI baselines (a higher baseline means the feature "
        "is more pronounced; several are *human* behaviors that AI stories tend "
        "to lack). For each feature decide 'present' or 'absent' and give ONE "
        "short sentence of evidence. Respond with ONLY a JSON object of the "
        'form {"key": {"present": true-or-false, "evidence": "one-sentence"}} '
        "--- no prose, no markdown fences.\n\n"
        f"STORY:\n{text}\n\n"
        f"FEATURES TO JUDGE ({len(NARRATIVE_DIMENSIONS)}):\n{dim_lines}"
    )


# ---------------------------------------------------------------------------
# Tolerant response parsing (never crashes on malformed output)
# ---------------------------------------------------------------------------
def _extract_json_object(raw: str):
    """Extract the first parseable ``{...}`` object from arbitrary LLM text.

    Handles stray prose, ```json fences, labels, etc. Returns the loaded object
    or ``None`` if nothing parseable is found.
    """
    if not raw:
        return None
    cleaned = re.sub(r"```(?:json)?", "", raw)
    # Fast path: whole string is valid JSON.
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass
    # Fall back to scanning for the first balanced {...} block.
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, TypeError):
                        break
        start = cleaned.find("{", start + 1)
    return None


def _verdict_present(value) -> bool | None:
    """Tolerantly interpret a per-dimension value as True/False/Unknown."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("present", "true", "yes", "1"):
            return True
        if v in ("absent", "false", "no", "0"):
            return False
        return None
    if isinstance(value, dict):
        candidates = ("present", "verdict", "detected", "value")
        for cand in candidates:
            if cand in value:
                return _verdict_present(value[cand])
        return None
    return None


def _evidence_for(value) -> str:
    if isinstance(value, dict):
        for cand in ("evidence", "explanation", "quote", "example"):
            e = value.get(cand)
            if isinstance(e, str):
                return e.strip()
    return ""


def parse_narrative_response(raw: str) -> dict:
    """Tolerantly parse an LLM narrative response into clean verdicts.

    Returns ``{"dimensions": {key: {"present": bool|None, "evidence": str}},
    "error": str|None}``. Never raises; on malformed output it returns whatever
    it could salvage plus an ``"error"`` note.
    """
    dims: dict[str, dict] = {}
    error = None

    obj = _extract_json_object(raw)
    if obj is None:
        return {"dimensions": dims, "error": "no valid JSON object found in response"}

    if not isinstance(obj, dict):
        return {"dimensions": dims, "error": "response JSON is not a mapping"}

    payload = obj
    # Accept a wrapper object like {"dimensions": {...}} as well as a bare map.
    if set(obj) == {"dimensions"} and isinstance(obj.get("dimensions"), dict):
        payload = obj["dimensions"]

    for key, value in payload.items():
        base_key = key
        if base_key not in _DIMENSIONS_BY_KEY:
            # tolerate keys that wrap/prefix the known key (e.g. "n1_theme_...")
            for known in _DIMENSIONS_BY_KEY:
                if known in base_key:
                    base_key = known
                    break
        if base_key not in _DIMENSIONS_BY_KEY:
            continue
        dims[base_key] = {
            "present": _verdict_present(value),
            "evidence": _evidence_for(value) if isinstance(value, dict) else "",
        }

    if not dims:
        error = "response JSON contained no recognized dimension keys"

    return {"dimensions": dims, "error": error}


# ---------------------------------------------------------------------------
# Flagging + scoring
# ---------------------------------------------------------------------------
def _is_flagged(dimension: dict, present: bool | None) -> bool:
    """A dimension is flagged when the AI-typical side is present.

    * ai_high    -> flagged when the feature is PRESENT
    * human_high -> flagged when the feature is ABSENT (AI tends to lack it)
    Unknown verdicts (``None``) are never flagged.
    """
    if present is None:
        return False
    if dimension["direction"] == "ai_high":
        return present is True
    return present is False


def run_narrative_checklist(
    text: str,
    llm_call: Callable[[str], str],
    min_words: int = DEFAULT_MIN_WORDS,
) -> dict:
    """Run the document-level narrative checklist via an injected callable.

    ``llm_call(prompt) -> raw_response``. The caller owns tokenize/generate/
    decode. Gated at ``min_words``: short input is disabled with NO
    ``llm_call`` made (narrative features need long text).

    Returns ``{"enabled", "reason", "dimensions", "flagged", "score"}``.
    ``score`` = fraction of dimensions flagged (0.0 least AI-like narrative,
    1.0 all AI-typical tells present).
    """
    if len(text.split()) < min_words:
        return {
            "enabled": False,
            "reason": "too_short",
            "dimensions": [],
            "flagged": [],
            "score": 0.0,
        }

    raw = llm_call(build_narrative_prompt(text))
    parsed = parse_narrative_response(raw)
    verdicts = parsed.get("dimensions", {})
    error = parsed.get("error")

    dimensions: list[dict] = []
    flagged: list[dict] = []
    for dimension in NARRATIVE_DIMENSIONS:
        key = dimension["key"]
        entry = dict(dimension)
        verdict = verdicts.get(key, {})
        present = verdict.get("present")
        entry["present"] = present
        entry["evidence"] = verdict.get("evidence", "")
        entry["flagged"] = _is_flagged(dimension, present)
        dimensions.append(entry)
        if entry["flagged"]:
            flagged.append(entry)

    if error:
        dimensions.append({
            "key": "_parse_error",
            "label": "Parser note",
            "ai_prior": None,
            "human_prior": None,
            "direction": None,
            "present": None,
            "evidence": "",
            "flagged": False,
            "note": error,
        })

    n_evaluated = sum(
        1 for d in dimensions if d["present"] is not None and "note" not in d
    )
    score = (len(flagged) / n_evaluated) if n_evaluated else 0.0

    return {
        "enabled": True,
        "reason": None,
        "dimensions": dimensions,
        "flagged": flagged,
        "score": round(min(max(score, 0.0), 1.0), 4),
    }
