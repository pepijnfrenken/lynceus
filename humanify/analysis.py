"""Model-agnostic analysis: markers, advice, sentence splitting, input reading."""

import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_SCORING_WORDS = 320


@dataclass(frozen=True, slots=True)
class ScoringUnit:
    """Normalized detector input mapped back to a contiguous sentence range."""

    text: str
    start: int
    stop: int

# Era-aware AI vocabulary, loaded from a data file (KB 02 L1b).
# Each entry: {"term", "era", "weight", "category", "reliability"}.
_AI_VOCAB_PATH = Path(__file__).parent / "data" / "ai_vocab.json"
with open(_AI_VOCAB_PATH, encoding="utf-8") as _vocab_f:
    _AI_VOCAB = json.load(_vocab_f)

# AI filler phrases (case-insensitive substring match)
AI_FILLERS: set[str] = {entry["term"] for entry in _AI_VOCAB}

# Term -> per-term signal weight (0.0 weak .. 1.0 strong, era-aware).
_AI_WEIGHTS: dict[str, float] = {
    entry["term"].lower(): float(entry["weight"]) for entry in _AI_VOCAB
}
# Term -> KB reliability (1 weak .. 3 strong).
_AI_RELIABILITY: dict[str, int] = {
    entry["term"].lower(): int(entry["reliability"]) for entry in _AI_VOCAB
}
# Term -> era tag, for explainable findings.
_AI_ERAS: dict[str, str] = {
    entry["term"].lower(): entry["era"] for entry in _AI_VOCAB
}
_AI_FILLERS_LOWER = {p.lower() for p in AI_FILLERS}


def find_markers(sentence: str) -> list[str]:
    """Find AI filler phrases in a sentence.

    Returns the sorted set of matched terms (backward compatible signature).
    """
    lower = sentence.lower()
    return sorted({p for p in _AI_FILLERS_LOWER if p in lower})


def find_markers_weighted(sentence: str) -> list[tuple[str, float]]:
    """Find AI filler phrases with their era-aware signal weight.

    Returns a list of ``(term, weight)`` for every matched term. Current-era
    terms (e.g. ``showcase`` in 2026) carry a higher weight than stale
    2023-era words (e.g. ``delve``), and common connectives that humans also
    use are intentionally down-weighted (KB 02 L10/S10).
    """
    matched: dict[str, float] = {}
    lower = sentence.lower()
    for term in _AI_FILLERS_LOWER:
        if term in lower:
            matched[term] = _AI_WEIGHTS.get(term, 0.3)
    return sorted(matched.items())


# ---------------------------------------------------------------------------
# Pattern tells (KB 02 S1/S2/S7/S10, L3) -- explainable, never bare numbers
# ---------------------------------------------------------------------------

_CONNECTIVE_OPENERS = (
    "moreover", "furthermore", "additionally", "in conclusion", "in summary",
    "however", "nevertheless", "therefore",
)
_RULE_OF_THREE = re.compile(
    r"\b[a-z][a-z0-9-]*\s*,\s*[a-z][a-z0-9-]*\s*,?\s+(?:and|&)\s+[a-z][a-z0-9-]*",
    re.IGNORECASE,
)
_NOT_X_BUT_Y = re.compile(
    r"\bnot\s+(?:only\s+)?(?!but\b)(.*?)\s+but\s+(?:also\s+)?(.*?)[\s.!?]",
    re.IGNORECASE,
)
# L3 -- two or more consecutive bare "-ing" verb phrases chained by a
# coordinating/listing connector (", 'and', 'plus', 'but', 'or').
# "leveraging X and utilizing Y" fires; "she is reading a book about cooking"
# (separate gerunds with no connector) stays silent.
_ING_ANALYSIS = re.compile(
    r"\b(\w+ing)\b[^.!?]*?(?:,| and | plus | but | or )[^.!?]*?\b(\w+ing)\b",
    re.IGNORECASE,
)


def _sentence_opener_word(sentence: str) -> str | None:
    """Return a matching connective-word opener (S10) or ``None``."""
    words = sentence.lower().split()
    if not words:
        return None
    cands = [words[0].strip(".,;:!?()\"'")]
    if len(words) >= 2:
        cands.append(
            " ".join(w.strip(".,;:!?()\"'") for w in words[:2])
        )
    for c in cands:
        if c in _CONNECTIVE_OPENERS:
            return c
    return None


def find_pattern_tells(sentence: str) -> list[tuple[str, float, str]]:
    """Detect KB 02 structural/pattern tells in a single sentence.

    Returns a list of ``(pattern_name, weight, matched_text)``. Each pattern is
    a *named, cheap, explainable* finding -- never a bare probability. Single
    weak tells are never proof on their own (KB 03 8.5).
    """
    tells: list[tuple[str, float, str]] = []

    # S10 -- sentence-initial formulaic connective
    opener = _sentence_opener_word(sentence)
    if opener is not None:
        tells.append(("connective_opener", 0.3, opener))

    # S1 -- rule of three (tricolon)
    m = _RULE_OF_THREE.search(sentence)
    if m:
        tells.append(("rule_of_three", 0.4, m.group(0)))

    # S2 -- not-X-but-Y / not only...but
    m = _NOT_X_BUT_Y.search(sentence)
    if m:
        tells.append(("not_x_but_y", 0.5, m.group(0)))

    # L3 -- consecutive 2+ "-ing" verb phrases
    m = _ING_ANALYSIS.search(sentence)
    if m:
        tells.append(("ing_analysis", 0.4, m.group(0)))

    # S7 -- em-dash usage (density handled at document level)
    n_dashes = sentence.count("\u2014")
    if n_dashes > 0:
        tells.append(("em_dash", 0.3, "\u2014" * n_dashes))

    return tells

_PATTERN_LABELS: dict[str, str] = {
    "connective_opener": "connective sentence-starter (S10)",
    "rule_of_three": "rule-of-three list (S1)",
    "not_x_but_y": "not-X-but-Y negative parallelism (S4/S2)",
    "ing_analysis": "consecutive -ing analysis phrases (L3)",
    "em_dash": "em-dash usage (S7)",
}


def normalize_for_detection(text: str) -> str:
    """Collapse presentation whitespace before model scoring."""
    return re.sub(r"\s+", " ", text.strip())


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, skipping empty/whitespace-only and junk fragments."""
    text = normalize_for_detection(text)
    # Protect abbreviations from triggering splits
    abbrevs = {"vs.": "vs\x00", "e.g.": "eg\x00", "i.e.": "ie\x00", "etc.": "etc\x00"}
    for orig, safe in abbrevs.items():
        text = text.replace(orig, safe)
    raw = re.split(r"(?<=[.!?])\s+", text)
    for orig, safe in abbrevs.items():
        raw = [s.replace(safe, orig) for s in raw]
    return [
        s
        for s in (s.strip() for s in raw)
        if s and len(s) >= 15 and len(s.split()) >= 4
    ]



def _append_scoring_units(
    units: list[ScoringUnit],
    sentences: list[str],
    start: int,
    max_words: int,
) -> None:
    """Append sentence chunks that keep document context without
    exceeding model limits."""
    chunk_start = 0
    chunk_words = 0
    chunk: list[str] = []

    for rel_index, sentence in enumerate(sentences):
        sentence_words = len(sentence.split())
        if chunk and chunk_words + sentence_words > max_words:
            units.append(
                ScoringUnit(" ".join(chunk), start + chunk_start, start + rel_index)
            )
            chunk = []
            chunk_start = rel_index
            chunk_words = 0
        chunk.append(sentence)
        chunk_words += sentence_words

    if chunk:
        units.append(
            ScoringUnit(" ".join(chunk), start + chunk_start, start + len(sentences))
        )


def prepare_scoring_units(
    text: str,
    max_words: int = DEFAULT_MAX_SCORING_WORDS,
) -> tuple[list[str], list[ScoringUnit]]:
    """Return display sentences plus normalized document-like detector chunks.

    Classifier detectors are trained on passages, not isolated one-sentence
    fragments. Scoring whitespace-normalized chunks keeps document context and
    avoids treating blank-line formatting as model evidence.
    """
    if max_words < 1:
        raise ValueError("max_words must be positive")

    sentences = split_sentences(text)
    units: list[ScoringUnit] = []
    if sentences:
        _append_scoring_units(units, sentences, 0, max_words)

    return sentences, units


def read_input(file_path: str | None) -> str:
    """Read from file or stdin."""
    if file_path:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def _sentence_openers(sentences: list[str]) -> list[tuple[str, int]]:
    """Find repeated sentence openers (first 1-2 significant words)."""
    openers = []
    stop = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "but", "or",
        "it", "its",
    }
    for s in sentences:
        words = s.split()
        for w in words[:3]:
            clean = w.strip("\"\u201c\u201d'(").lower()
            if clean and clean not in stop:
                openers.append(clean)
                break
    return Counter(openers).most_common(5)


def _burstiness_band(sentences: list[str]) -> str:
    """Classify std/mean sentence-length CV into a KB 02 S5 burstiness band."""
    word_counts = [len(s.split()) for s in sentences] if sentences else [0]
    raw_std = statistics.pstdev(word_counts) if len(word_counts) >= 2 else 0.0
    mean_wc = statistics.mean(word_counts)
    ratio = raw_std / mean_wc if mean_wc > 0 else 0.0
    if ratio < 0.30:
        return "ai_like"
    if ratio < 0.65:
        return "mixed"
    if ratio <= 0.85:
        return "human_like"
    return "over_varied"


def _em_dash_density_per_1k(sentences: list[str]) -> float:
    """Em-dashes (— and en-dashes –) per 1,000 words (S7)."""
    text = " ".join(sentences)
    n_words = len(text.split())
    if n_words <= 0:
        return 0.0
    n_dashes = text.count("\u2014") + text.count("\u2013")
    return (n_dashes * 1000.0) / n_words


def generate_advice(
    sentences: list[str],
    probs: list[float],
    markers: list[list[str]],
    stats: dict[str, float],
    threshold: float,
) -> tuple[list[str], list[tuple[str, float, list[str]]], list[str]]:
    """Generate structural writing advice from analysis results.

    Returns (tips, focus_sentences, all_markers).
    """
    tips = []
    # 1. Burstiness
    if stats["std"] < 0.2 and stats["mean"] >= 0.3:
        tips.append(
            "\u26a0 Burstiness is low \u2014 your sentences all score similarly. "
            "Mix in more varied phrasing and personal voice."
        )
    # 1a. Burstiness band + em-dash density (KB 02 S5 humzakt bands / S7)
    # The band tip needs a few sentences to be meaningful, so we only surface it
    # with 4+ sentences to avoid noisy advice on tiny inputs.
    if len(sentences) >= 4 and _burstiness_band(sentences) == "ai_like":
        tips.append(
            "\u26a0 Sentence lengths are unnaturally uniform (AI-like "
            "burstiness band). Mix short and long sentences to sound more human."
        )
    emdash = _em_dash_density_per_1k(sentences)
    if emdash > 10:
        tips.append(
            f"\u26a0 Heavy em-dash usage ({emdash:.1f} per 1,000 words). "
            "Em-dashes are a strong AI/casual signature \u2014 prefer commas "
            "or restructure instead."
        )
    # 2. Long sentences
    long_sents = [(i, s) for i, s in enumerate(sentences) if len(s.split()) > 30]
    if long_sents:
        tips.append(
            f"\u26a0 {len(long_sents)} sentence(s) are 30+ words. "
            "Consider breaking them at natural pause points (commas, semicolons)."
        )
    # 3. Repetitive openers
    openers = _sentence_openers(sentences)
    if openers and openers[0][1] >= 3:
        words = ", ".join(f'"{w}" ({n}x)' for w, n in openers[:3] if n >= 2)
        if words:
            tips.append(
                f"\u26a0 Repetitive sentence starts: {words}. "
                "Vary how you begin sentences \u2014 "
                "try subordinate clauses or questions."
            )
    # 4. Marker frequency (era-weighted: surface higher-weight matches first)
    all_marks = [m for marks in markers for m in marks]
    if all_marks:
        weight_map: dict[str, float] = {}
        for s in sentences:
            for term, w in find_markers_weighted(s):
                if w > weight_map.get(term, -1.0):
                    weight_map[term] = w
        top = sorted(weight_map.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        words = ", ".join(f'"{w}" ({weight:.2f})' for w, weight in top)
        tips.append(
            f"\u26a0 AI-typical words detected: {words}. "
            "Current-era high-weight signs in particular \u2014 "
            "use sparingly or replace with simpler alternatives."
        )

    # 4a. Pattern tells (KB 02 S1/S2/S7/S10/L3) -- named, explainable
    pattern_seen: dict[str, list[int]] = {}
    for i, s in enumerate(sentences):
        for name, _w, _txt in find_pattern_tells(s):
            pattern_seen.setdefault(name, []).append(i + 1)
    if pattern_seen:
        bits = []
        for name in sorted(pattern_seen):
            lbl = _PATTERN_LABELS.get(name, name)
            idxs = ", ".join(f"s{n}" for n in pattern_seen[name])
            bits.append(f"{lbl} in {idxs}")
        tips.append(
            f"\u26a0 AI-typical patterns: {'; '.join(bits)}. "
            "These constructions read as formulaic \u2014 vary them."
        )

    # 5. Overall assessment
    high_ai = sum(1 for p in probs if p >= threshold)
    if high_ai > len(probs) * 0.7:
        tips.append(
            "\u26a0 Most sentences score above threshold. "
            "Focus on adding personal anecdotes, "
            "specific examples, or informal phrasing."
        )
    if not tips:
        tips.append(
            "\u2713 Text reads naturally \u2014 "
            "no major structural patterns to address."
        )
    # Per-sentence focus (top 3 highest AI + markers)
    focus = sorted(
        [
            (s, p, m)
            for s, p, m in zip(sentences, probs, markers)
            if p >= threshold and m
        ],
        key=lambda x: -x[1],
    )[:3]
    return tips, focus, all_marks


def generate_quality_advice(metrics: dict, style: str = "general") -> list[str]:
    """Generate concrete writing-quality advice from metric results.

    Parameters
    ----------
    metrics:
        The dict returned by ``compute_writing_metrics()``.
    style:
        ``"general"`` (default) or ``"academic"``. Shifts thresholds to
        account for formal academic prose conventions.

    Returns
    -------
    List of balanced tips: green checkmarks for strengths and yellow
    warnings for weaknesses.
    """
    is_academic = style == "academic"
    tips = []
    raw = metrics.get("raw", {})
    norm = metrics.get("normalized", {})
    pa = metrics.get("paragraph_analysis", {})

    # Thresholds depend on style
    passive_thresh = 0.40 if is_academic else 0.20
    weak_thresh = 0.10 if is_academic else 0.05
    runon_thresh = 0.15 if is_academic else 0.20
    vague_thresh = 0.08 if is_academic else 0.05
    abstract_thresh = 0.25 if is_academic else 0.15
    concrete_thresh = 0.15 if is_academic else 0.30
    citation_thresh = 0.05 if is_academic else 0.10
    example_thresh = 0.05 if is_academic else 0.10
    data_thresh = 0.05 if is_academic else 0.10
    experience_thresh = 0.05 if is_academic else 0.10

    # --- Grammar heuristics ---
    grammar_norm = norm.get("grammar_heuristics", 0.5)
    grammar_raw = raw.get("grammar_heuristics", {})
    if grammar_norm < 0.5:
        issues = []
        if grammar_raw.get("passive_ratio", 0) > passive_thresh:
            issues.append("passive voice")
        if grammar_raw.get("weak_modifier_ratio", 0) > weak_thresh:
            issues.append("weak modifiers (very, really, quite)")
        if grammar_raw.get("runon_ratio", 0) > runon_thresh:
            issues.append("run-on sentences")
        if issues:
            tips.append(
                f"\u26a0 Grammar: {', '.join(issues)} detected. "
                "Use active voice, cut filler words, and break long sentences."
            )
        else:
            tips.append(
                "\u26a0 Grammar heuristics flag room for improvement. "
                "Review sentence structure and word choice."
            )
    elif grammar_norm >= 0.8:
        tips.append(
            "\u2713 Grammar: strong active voice and clean sentence structure."
        )

    # --- Specificity ---
    specificity_norm = norm.get("specificity", 0.5)
    specificity_raw = raw.get("specificity", {})
    if specificity_norm < 0.5:
        issues = []
        if specificity_raw.get("vague_word_ratio", 0) > vague_thresh:
            issues.append("vague words (thing, stuff, good, bad)")
        if specificity_raw.get("abstract_noun_ratio", 0) > abstract_thresh:
            issues.append("abstract nouns (-tion, -ness, -ity)")
        if specificity_raw.get("concrete_detail_ratio", 0) < concrete_thresh:
            issues.append("lack of concrete details")
        if issues:
            tips.append(
                f"\u26a0 Specificity: {', '.join(issues)}. "
                "Replace abstractions with numbers, names, dates, and vivid examples."
            )
        else:
            tips.append(
                "\u26a0 Specificity could be stronger. Add concrete details and precise language."
            )
    elif specificity_norm >= 0.7:
        tips.append(
            "\u2713 Specificity: concrete details and precise language make this vivid."
        )

    # --- Evidence ---
    evidence_norm = norm.get("evidence", 0.0)
    evidence_raw = raw.get("evidence", {})
    if evidence_norm < 0.3:
        issues = []
        if evidence_raw.get("citation_density", 0) < citation_thresh:
            issues.append("citations")
        if evidence_raw.get("example_density", 0) < example_thresh:
            issues.append("examples")
        if evidence_raw.get("data_density", 0) < data_thresh:
            issues.append("data/statistics")
        if evidence_raw.get("experience_density", 0) < experience_thresh:
            issues.append("personal experience")
        if issues:
            tips.append(
                f"\u26a0 Evidence: missing {', '.join(issues)}. "
                "Strengthen claims with research, examples, or firsthand observations."
            )
        else:
            tips.append(
                "\u26a0 Evidence is thin. Add citations, examples, or data to support claims."
            )
    elif evidence_norm >= 0.6:
        tips.append(
            "\u2713 Evidence: well-supported with citations, examples, or data."
        )

    # --- Paragraph structure ---
    if pa:
        n_para = pa.get("paragraph_count", 0)
        if n_para > 0:
            if pa.get("topic_sentence_coverage", 0) < 0.5:
                tips.append(
                    "\u26a0 Paragraphs: many paragraphs lack a clear topic sentence. "
                    "Start each paragraph with a focused claim."
                )
            elif pa.get("topic_sentence_coverage", 0) >= 0.8:
                tips.append(
                    "\u2713 Paragraphs: most paragraphs open with clear topic sentences."
                )

            if pa.get("paragraph_transition_coverage", 0) < 0.2:
                tips.append(
                    "\u26a0 Flow: few paragraph transitions detected. "
                    "Use signposting words to guide readers between ideas."
                )
            elif pa.get("paragraph_transition_coverage", 0) >= 0.5:
                tips.append(
                    "\u2713 Flow: good paragraph-level transitions keep the argument coherent."
                )

            if pa.get("thesis_alignment", 0) < 0.1 and n_para >= 2:
                tips.append(
                    "\u26a0 Structure: introduction and conclusion share few keywords. "
                    "Ensure the conclusion directly addresses the claims made in the introduction."
                )
            elif pa.get("thesis_alignment", 0) >= 0.3:
                tips.append(
                    "\u2713 Structure: introduction and conclusion are well-aligned."
                )

    if not tips:
        tips.append(
            "\u2713 Writing quality is balanced \u2014 no major issues or standout weaknesses."
        )

    return tips
