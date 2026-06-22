"""Model-agnostic analysis: markers, advice, sentence splitting, input reading."""

import re
import sys
from collections import Counter
from dataclasses import dataclass

DEFAULT_MAX_SCORING_WORDS = 320


@dataclass(frozen=True, slots=True)
class ScoringUnit:
    """Normalized detector input mapped back to a contiguous sentence range."""

    text: str
    start: int
    stop: int

# AI filler phrases (case-insensitive substring match)
AI_FILLERS: set[str] = {
    "furthermore", "moreover", "crucial", "leverage", "robust",
    "comprehensive", "significant", "notably", "essentially",
    "delve", "tapestry", "intricate", "pivotal", "testament",
    "underscore", "multifaceted", "holistic", "paradigm", "synergy",
    "innovative", "cutting-edge", "groundbreaking", "transformative",
    "streamline", "optimize", "foster", "facilitate", "endeavor",
    "ascertain", "subsequently", "consequently", "nevertheless",
    "notwithstanding", "albeit", "whereby", "therein", "henceforth",
    "aforementioned", "juxtapose", "myriad", "ubiquitous", "ephemeral",
    "poignant", "nuanced", "meticulous", "unparalleled", "spearhead",
    "in conclusion", "it's worth noting", "it is worth noting",
    "it is important to note", "in today's world", "as we navigate",
    "plays a vital role", "shaping the future",
    "in this context", "at the end of the day",
}
_AI_FILLERS_LOWER = {p.lower() for p in AI_FILLERS}


def find_markers(sentence: str) -> list[str]:
    """Find AI filler phrases in a sentence."""
    lower = sentence.lower()
    return sorted({p for p in _AI_FILLERS_LOWER if p in lower})


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
    # 4. Marker frequency
    all_marks = [m for marks in markers for m in marks]
    if all_marks:
        top = Counter(all_marks).most_common(3)
        words = ", ".join(f'"{w}" ({n}x)' for w, n in top)
        tips.append(
            f"\u26a0 AI-typical words detected: {words}. "
            "These are strong signals \u2014 "
            "use sparingly or replace with simpler alternatives."
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
