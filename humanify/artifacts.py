"""Detector-of-humanizers / post-processing artifact layer (Phase 4.3).

Pure-Python, no models. Looks for *mechanical* fingerprints of a text having
been run through a "humanizer" / stealth-rewriter pipeline (KB 04 Tier 3 #11):

* **zero-width characters** (U+200B zwnb-zs space, U+200C non-joiner,
  U+200D joiner, U+FEFF BOM, direction marks, word joiners): splice blobs that
  a post-processing tool can inject to evade detectors -> a *post-processing*
  artifact (KB 02 T10, markdown/markup-bug structural tell = machine tooling).
* **Unicode homoglyphs / confusable letters** (e.g. Cyrillic 'а' for Latin 'a'):
  a way to dodge n-gram/watermark tools while keeping the text legible; flag
  if more than 2 confusable chars appear (KB 02 T10).
* **mechanical synonym cycling** (KB 02 S4, "elegant variation"): a thesaurus
  rewriter replaces nouns ad nauseam so lexical diversity collapses. Soft,
  severity-LOW heuristic (documented below).
* **high edit-magnitude risk**: composite recommendation to run EditLens (a
  separate heavyweight 0-1 edit-magnitude detector) when the text *looks*
  post-processed (zero-width or homoglyphs fire).

Return::

    {"findings": [{"name","severity","evidence","kb_ref"}],
     "score": 0.0-1.0,
     "editlens_recommended": bool}

``score`` is a severity-weighted sum capped at 1.0 (same scheme as
``humanify.structure``); higher means more post-processing artifacts.
"""

import re
from collections import Counter

# ---------------------------------------------------------------------------
# Zero-width / format characters
# ---------------------------------------------------------------------------
_ZERO_WIDTH_CHARS = (
    "​",  # zero-width space
    "‌",  # zero-width non-joiner
    "‍",  # zero-width joiner
    "﻿",  # zero-width no-break space / BOM
    "‎",  # left-to-right mark
    "‏",  # right-to-left mark
    "⁠",  # word joiner
    "⁦",  # LRI
    "⁧",  # RLI
    "⁩",  # PDI
)

# ---------------------------------------------------------------------------
# Unicode homoglyphs -> confusable letters (non-ASCII chars that render like
# common Latin letters). Hardcoded small set for speed; keep it small per the
# spec (e.g. Cyrillic 'а', 'е', 'і').
# ---------------------------------------------------------------------------
_CONFUSABLE_CHARS = frozenset(
    # Cyrillic lookalikes
    "аеорсухіІвјє"
    # Greek lookalikes
    "οειόαρς"
    # Latin Extended / misc lookalikes
    "ǫạṣệⁱ"
)

# ---------------------------------------------------------------------------
# Mechanical synonym cycling (KB 02 S4)
# ---------------------------------------------------------------------------
_SYNC_MIN_WORDS = 40          # need enough content words to judge
_SYNC_MAX_UNIQUE_RATIO = 0.35  # unique/total BELOW this => narrowed vocabulary
_SYNC_MIN_MID_RATIO = 0.30    # fraction of tokens that repeat (freq >= 2)
_SYNC_MIN_FREQ = 2

# Common English function words are excluded so synonym-cycling measures
# *content* words (nouns/verbs/adjectives), not grammar.
_STOPWORDS = frozenset(
    ("the and of to in a is was were are be been has have had do does did it "
     "its that this those these for on with as at by from or an not but if "
     "about into than them they he she we you your our could would should can "
     "may will all any then so such there here when which while who whom each "
     "one two three no yes also more most very just get got go going up out "
     "off him her his their ours like over under again further".split())
)

_ARTIFACT_SEVERITY = {"low": 0.2, "medium": 0.4, "high": 0.6}


def _count_zero_width(text: str) -> int:
    return sum(text.count(c) for c in _ZERO_WIDTH_CHARS)


def _count_confusables(text: str) -> int:
    return sum(1 for ch in text if ch in _CONFUSABLE_CHARS)


def _mechanical_synonym_cycling(text: str) -> dict | None:
    """S4 elegant-variation / synonym-cycling heuristic.

    Returns a stats dict when the text looks like narrowed, repeating content
    vocabulary; otherwise ``None``. Documented heuristic:

    * tokenize to lowercase alphabetic words, drop stopwords;
    * require ``>= _SYNC_MIN_WORDS`` content tokens (short text carries no
      signal);
    * ``unique_ratio = n_unique / n_tokens``. Natural prose keeps producing new
      words (unique_ratio usually >> 0.4); a thesaurus rewriter recycles a small
      pool (unique_ratio < 0.35);
    * require the fraction of *repeated* (frequency >= 2) tokens to be high
      (>= 0.30) so that a merely short document doesn't fire.

    Both conditions must hold (logical AND). Severity LOW because it's heuristic.
    """
    tokens = [
        w for w in re.findall(r"[a-zA-Z']+", text.lower())
        if w not in _STOPWORDS
    ]
    if len(tokens) < _SYNC_MIN_WORDS:
        return None
    counts = Counter(tokens)
    unique_ratio = len(counts) / len(tokens)
    if unique_ratio >= _SYNC_MAX_UNIQUE_RATIO:
        return None
    repeated = sum(c for c in counts.values() if c >= _SYNC_MIN_FREQ)
    mid_ratio = repeated / len(tokens)
    if mid_ratio < _SYNC_MIN_MID_RATIO:
        return None
    return {
        "unique_ratio": round(unique_ratio, 3),
        "mid_repeat_ratio": round(mid_ratio, 3),
        "n_tokens": len(tokens),
        "n_unique": len(counts),
    }


def detect_artifacts(text: str) -> dict:
    """Detect post-processing artifacts in *text*.

    Returns ``{"findings": [...], "score": 0.0-1.0,
    "editlens_recommended": bool}``.
    """
    findings: list[dict] = []

    zero_width = _count_zero_width(text)
    if zero_width > 0:
        findings.append({
            "name": "zero_width_chars",
            "severity": "medium",
            "evidence": f"{zero_width} zero-width/format characters found",
            "kb_ref": "T10",
        })

    homoglyphs = _count_confusables(text)
    if homoglyphs > 2:
        findings.append({
            "name": "unicode_homoglyphs",
            "severity": "low",
            "evidence": f"{homoglyphs} confusable (non-Latin) glyphs found",
            "kb_ref": "T10",
        })

    syn = _mechanical_synonym_cycling(text)
    if syn is not None:
        findings.append({
            "name": "mechanical_synonym_cycling",
            "severity": "low",
            "evidence": (
                f"unique_ratio={syn['unique_ratio']}, "
                f"mid_repeat_ratio={syn['mid_repeat_ratio']}, "
                f"n_tokens={syn['n_tokens']}, n_unique={syn['n_unique']}"
            ),
            "kb_ref": "S4",
        })

    # Composite: text looks post-processed -> recommend EditLens (a separate
    # heavyweight 0-1 edit-magnitude scorer). Here we only *recommend* it.
    editlens_recommended = bool((zero_width > 0) or (homoglyphs > 2))
    if editlens_recommended:
        findings.append({
            "name": "high_edit_magnitude_risk",
            "severity": "low",
            "evidence": "post-processing artifacts present; EditLens recommended",
            "kb_ref": "edit-magnitude",
        })

    score = min(
        sum(_ARTIFACT_SEVERITY.get(f["severity"], 0.0) for f in findings),
        1.0,
    )
    return {
        "findings": findings,
        "score": round(score, 3),
        "editlens_recommended": editlens_recommended,
    }
