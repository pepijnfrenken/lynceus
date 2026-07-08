"""Explain WHY each sentence scores the way it does.

Pure-Python, model-free style reasons: generic polished phrases, a formal-word
heatmap, actionable specificity hints, repeated sentence-pattern detection, and
a generic fallback so a high-scoring sentence never reads ``markers: []``.

This module mirrors the decoupling of :mod:`humanify.writing_metrics` — it
imports nothing from torch/transformers and can be exercised anywhere.
"""

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Reason:
    """A human-readable explanation for why a sentence triggered a signal.

    Attributes
    ----------
    type:
        Stable category identifier, e.g. ``"generic_phrase"``, ``"formal_word"``,
        ``"specificity"``, ``"repeated_pattern"``, ``"ai_filler"``, or
        ``"high_score"`` (the catch-all fallback).
    detail:
        What was detected, phrased as a clause
        (e.g. "uses the generic polished phrase 'structured methodology'").
    suggestion:
        A concrete, actionable fix (e.g. "try 'a simple plan' instead").
    """

    type: str
    detail: str
    suggestion: str

    def to_dict(self) -> dict:
        return {"type": self.type, "detail": self.detail, "suggestion": self.suggestion}


# --- Generic polished reflection / corporate-academic phrases (#3) ----------
# Multi-word phrases common in AI-generated academic reflection. Each maps to a
# plainer, more specific alternative. Matched case-insensitively as substrings.
GENERIC_PHRASES: dict[str, str] = {
    "structured methodology": "a simple plan, a rough plan, or describe what the team actually did",
    "aligned with the project goals": "name the specific goal it served",
    "aligns with the project goals": "name the specific goal it served",
    "ongoing professional development": "a concrete skill you built or a specific moment you learned something",
    "professional development": "a concrete skill you built or a specific moment you learned something",
    "professional growth": "a specific thing you got better at",
    "cohesive story": "what actually connected, or drop the phrase",
    "clear internal deadlines": "the actual dates or milestones you set",
    "internal deadlines": "the actual dates or milestones you set",
    "actionable insights": "the actual takeaway you used",
    "best practices": "the specific habit or rule you followed",
    "stakeholders": "who, specifically",
    "deliverables": "the actual thing you produced",
    "holistic approach": "the specific parts you combined",
    "synergy": "how the parts actually worked together",
    "in conclusion": "drop it and just state the point",
    "it is worth noting": "just say it directly",
    "it's worth noting": "just say it directly",
    "plays a vital role": "name what it actually did",
    "navigate the complexities": "the actual problem you handled",
    "navigating the complexities": "the actual problem you handled",
    "at the end of the day": "drop it or replace with the specific point",
    "moving forward": "drop it, or say specifically what happens next",
    "in today's world": "drop it \u2014 it adds nothing specific",
}
_GENERIC_PHRASES_LOWER = {p.lower(): alt for p, alt in GENERIC_PHRASES.items()}


def find_generic_phrases(sentence: str) -> list[Reason]:
    """Return a reason listing any generic polished phrases found.

    Multiple matches are combined into a single reason, preferring the longest
    phrase when one contains another (so "ongoing professional development" is
    not double-counted with "professional development").
    """
    lower = sentence.lower()
    raw = [(p, alt) for p, alt in _GENERIC_PHRASES_LOWER.items() if p in lower]
    if not raw:
        return []
    raw.sort(key=lambda x: len(x[0]), reverse=True)
    kept: list[tuple[str, str]] = []
    for p, alt in raw:
        if any(p in kp for kp, _ in kept):
            continue
        kept.append((p, alt))
    phrases = ", ".join(f"'{p}'" for p, _ in kept)
    alts = "; ".join(dict.fromkeys(alt for _, alt in kept))
    return [Reason(
        type="generic_phrase",
        detail=f"generic polished language: {phrases}",
        suggestion=f"sound more specific \u2014 try: {alts}",
    )]


# --- Formality heatmap (#11) ------------------------------------------------
# Single words that raise formality above a natural student voice. Each maps to
# a simpler everyday alternative. Matched as whole words (case-insensitive).
FORMAL_WORDS: dict[str, str] = {
    "concerning": "about",
    "ensuring": "making sure",
    "ensure": "make sure",
    "incorporate": "include",
    "incorporated": "included",
    "sustained": "steady",
    "utilize": "use",
    "utilized": "used",
    "commence": "start",
    "commenced": "started",
    "subsequently": "then",
    "facilitate": "help",
    "facilitated": "helped",
    "endeavor": "try",
    "demonstrate": "show",
    "demonstrated": "showed",
    "approximately": "about",
    "sufficient": "enough",
    "obtain": "get",
    "require": "need",
    "requires": "needs",
    "regarding": "about",
    "numerous": "many",
    "ascertain": "find out",
    "aforementioned": "this",
    "methodology": "method",
    "individuals": "people",
    "assistance": "help",
    "modification": "change",
    "additionally": "also",
    "furthermore": "also",
    "moreover": "also",
    "nevertheless": "still",
    "notwithstanding": "even so",
    "prioritize": "focus on",
    "implementation": "doing it",
}


def find_formal_words(sentence: str) -> list[Reason]:
    """Return a reason listing formal words and simpler alternatives."""
    found: dict[str, str] = {}  # word -> simpler alt, preserves first-seen order
    for w in sentence.split():
        clean = w.lower().strip(".,;:!?\"'()")
        if clean in FORMAL_WORDS and clean not in found:
            found[clean] = FORMAL_WORDS[clean]
    if not found:
        return []
    pairs = ", ".join(f"'{w}' (try '{alt}')" for w, alt in found.items())
    return [Reason(
        type="formal_word",
        detail=f"raises the formality above a natural voice: {pairs}",
        suggestion="swap these for the simpler everyday words",
    )]


# --- Actionable specificity (#15) -------------------------------------------
_NUMBER_RE = re.compile(r"\d")
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_PERCENT_RE = re.compile(r"\d+\s*%|\d+\s*percent", re.IGNORECASE)
_EXAMPLE_HINTS = ("for example", "for instance", "such as", "e.g.", "namely")


def _has_concrete_detail(sentence: str) -> bool:
    """Mirror of writing_metrics._sentence_has_concrete_details (kept local so
    this module stays decoupled from the metrics internals)."""
    words = sentence.split()
    for i, w in enumerate(words):
        if i == 0:
            continue
        clean = w.strip("(),;:.!?\"'")
        if len(clean) > 1 and clean[0].isupper() and not clean.islower():
            return True
    return bool(
        _NUMBER_RE.search(sentence)
        or _DATE_RE.search(sentence)
        or _PERCENT_RE.search(sentence)
    )


def specificity_reason(sentence: str) -> Reason | None:
    """If a sentence lacks concrete detail, say what kind of detail to add."""
    if _has_concrete_detail(sentence):
        return None
    return Reason(
        type="specificity",
        detail="no concrete details \u2014 this could describe almost any project",
        suggestion=(
            "add a specific time/date or quantity, a name/place/tool, "
            "or a concrete example (e.g. 'for instance\u2026')"
        ),
    )


# --- Repeated sentence-pattern detection (#9) -------------------------------
_REPEATED_PATTERN_MIN = 3

# (label, compiled regex, human-readable form, alternative openers)
_OPENING_PATTERNS = (
    ("this-opener", re.compile(r"^This\b", re.IGNORECASE), "'This \u2026'",
     ("Looking back, \u2026", "One thing I noticed was \u2026", "For me, \u2026")),
    ("i-opener", re.compile(r"^I\b\s+\w"), "'I \u2026'",
     ("Looking back, \u2026", "One thing I noticed was \u2026", "Because of this, \u2026")),
    ("next-time", re.compile(r"^Next time\b", re.IGNORECASE), "'Next time, \u2026'",
     ("Looking back, \u2026", "If I did this again, \u2026")),
)


def repeated_pattern_reasons(sentences: list[str]) -> dict[int, list[Reason]]:
    """Map sentence index -> reasons for opener patterns repeated across the doc.

    A pattern only generates reasons once it appears at least
    ``_REPEATED_PATTERN_MIN`` times, mirroring the existing opener advice.
    """
    alts_by_label = {label: alts for label, _rx, _human, alts in _OPENING_PATTERNS}
    human_by_label = {label: human for label, _rx, human, _alts in _OPENING_PATTERNS}

    classes: list[str | None] = []
    for s in sentences:
        matched = None
        for label, rx, _human, _alts in _OPENING_PATTERNS:
            if rx.search(s):
                matched = label
                break
        classes.append(matched)

    counts = Counter(c for c in classes if c)
    reasons: dict[int, list[Reason]] = {}
    for idx, label in enumerate(classes):
        if label is None or counts[label] < _REPEATED_PATTERN_MIN:
            continue
        alts = alts_by_label[label]
        reasons.setdefault(idx, []).append(Reason(
            type="repeated_pattern",
            detail=(
                f"repeated {human_by_label[label]} sentence opener "
                f"({counts[label]}\u00d7 in this document)"
            ),
            suggestion=f"mix in other openers like {', '.join(alts)}",
        ))
    return reasons


# --- Generic high-score fallback (#1) ---------------------------------------
_PERSONAL_RE = re.compile(r"\b(i|my|we|our|me|us)\b", re.IGNORECASE)


def generic_high_score_reason(
    sentence: str, prob: float, threshold: float
) -> Reason | None:
    """Last-resort explanation when a high-scoring sentence has no other reason."""
    if prob < threshold:
        return None
    parts = ["smooth, generic formal structure"]
    if not _NUMBER_RE.search(sentence):
        parts.append("no specific numbers or dates")
    if not _PERSONAL_RE.search(sentence):
        parts.append("no personal detail")
    parts.append("could fit almost any student project")
    return Reason(
        type="high_score",
        detail="sounds generic: " + ", ".join(parts),
        suggestion="add a specific moment, name, date, or what you personally did",
    )


# --- Top-level assembly -----------------------------------------------------
_MAX_REASONS_PER_SENTENCE = 5


def explain_sentences(
    sentences: list[str],
    probs: list[float],
    markers: list[list[str]],
    threshold: float,
) -> list[list[Reason]]:
    """Build a de-duplicated, ordered list of reasons for each sentence.

    Existing ``markers`` (AI filler phrases from
    :func:`humanify.analysis.find_markers`) are folded into reasons so nothing
    is lost. ``sentences`` and ``markers`` must be index-aligned (both derive
    from :func:`humanify.analysis.split_sentences`).
    """
    pattern_reasons = repeated_pattern_reasons(sentences)
    all_reasons: list[list[Reason]] = []
    for idx, sentence in enumerate(sentences):
        prob = probs[idx] if idx < len(probs) else 0.0
        reasons: list[Reason] = []
        # Most specific first.
        reasons += find_generic_phrases(sentence)
        reasons += find_formal_words(sentence)
        marks = markers[idx] if idx < len(markers) else []
        if marks:
            reasons.append(Reason(
                type="ai_filler",
                detail=f"AI-typical word(s): {', '.join(marks)}",
                suggestion="these are strong AI signals \u2014 replace with simpler wording or drop",
            ))
        spec = specificity_reason(sentence)
        if spec is not None:
            reasons.append(spec)
        reasons += pattern_reasons.get(idx, [])

        # De-duplicate by (type, detail).
        seen: set[tuple[str, str]] = set()
        uniq: list[Reason] = []
        for r in reasons:
            key = (r.type, r.detail)
            if key not in seen:
                seen.add(key)
                uniq.append(r)

        # Fallback: a high score that still has no explanation.
        if not uniq:
            fb = generic_high_score_reason(sentence, prob, threshold)
            if fb is not None:
                uniq.append(fb)

        all_reasons.append(uniq[:_MAX_REASONS_PER_SENTENCE])
    return all_reasons
