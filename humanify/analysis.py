"""Model-agnostic analysis: markers, advice, sentence splitting, input reading."""

import re
import sys
from collections import Counter

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


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, skipping empty/whitespace-only and junk fragments."""
    text = re.sub(r"\s+", " ", text.strip())
    # Protect abbreviations from triggering splits
    abbrevs = {"vs.": "vs\x00", "e.g.": "eg\x00", "i.e.": "ie\x00", "etc.": "etc\x00"}
    for orig, safe in abbrevs.items():
        text = text.replace(orig, safe)
    raw = re.split(r"(?<=[.!?])\s+", text)
    for orig, safe in abbrevs.items():
        raw = [s.replace(safe, orig) for s in raw]
    return [s for s in (s.strip() for s in raw) if s and len(s) >= 15 and len(s.split()) >= 4]


def read_input(file_path: str | None) -> str:
    """Read from file or stdin."""
    if file_path:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def _sentence_openers(sentences: list[str]) -> list[tuple[str, int]]:
    """Find repeated sentence openers (first 1-2 significant words)."""
    openers = []
    stop = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "but", "or", "it", "its"}
    for s in sentences:
        words = s.split()
        for w in words[:3]:
            clean = w.strip("\"\u201c\u201d'(").lower()
            if clean and clean not in stop:
                openers.append(clean)
                break
    return Counter(openers).most_common(5)


def generate_advice(sentences, probs, markers, stats, threshold):
    """Generate structural writing advice from analysis results.

    Returns (tips, focus_sentences, all_markers).
    """
    tips = []
    # 1. Burstiness
    if stats["std"] < 0.2:
        tips.append("\u26a0 Burstiness is low \u2014 your sentences all score similarly. "
                    "Mix in more varied phrasing and personal voice.")
    # 2. Long sentences
    long_sents = [(i, s) for i, s in enumerate(sentences) if len(s.split()) > 30]
    if long_sents:
        tips.append(f"\u26a0 {len(long_sents)} sentence(s) are 30+ words. "
                    "Consider breaking them at natural pause points (commas, semicolons).")
    # 3. Repetitive openers
    openers = _sentence_openers(sentences)
    if openers and openers[0][1] >= 3:
        words = ", ".join(f'"{w}" ({n}x)' for w, n in openers[:3] if n >= 2)
        if words:
            tips.append(f"\u26a0 Repetitive sentence starts: {words}. "
                        "Vary how you begin sentences \u2014 try subordinate clauses or questions.")
    # 4. Marker frequency
    all_marks = [m for marks in markers for m in marks]
    if all_marks:
        top = Counter(all_marks).most_common(3)
        words = ", ".join(f'"{w}" ({n}x)' for w, n in top)
        tips.append(f"\u26a0 AI-typical words detected: {words}. "
                    "These are strong signals \u2014 use sparingly or replace with simpler alternatives.")
    # 5. Overall assessment
    high_ai = sum(1 for p in probs if p >= threshold)
    if high_ai > len(probs) * 0.7:
        tips.append("\u26a0 Most sentences score above threshold. "
                    "Focus on adding personal anecdotes, specific examples, or informal phrasing.")
    if not tips:
        tips.append("\u2713 Text reads naturally \u2014 no major structural patterns to address.")
    # Per-sentence focus (top 3 highest AI + markers)
    focus = sorted(
        [(s, p, m) for s, p, m in zip(sentences, probs, markers) if p >= threshold and m],
        key=lambda x: -x[1],
    )[:3]
    return tips, focus, all_marks
