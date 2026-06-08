"""Deterministic writing quality metrics — no model, no advice."""

import re
import statistics
from collections import Counter

from .analysis import normalize_for_detection, split_sentences


# Common English stopwords for prompt/reference keyword filtering
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "on", "by", "at", "with", "without", "into", "across",
    "over", "between", "out", "against", "within", "under", "around", "among",
    "it", "its", "he", "she", "they", "them", "their", "his", "her", "my",
    "your", "our", "we", "you", "i", "me", "us", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "shall", "should", "may", "might", "must", "need", "dare", "ought",
    "used", "not", "no", "nor", "none", "nothing", "never", "neither",
    "very", "too", "quite", "rather", "some", "any", "no", "every",
    "each", "all", "both", "few", "more", "most", "other", "another", "much",
    "many", "several", "own", "same", "enough", "such", "so", "here", "there",
    "when", "where", "why", "how", "which", "who", "whom", "what", "whose",
    "am", "are", "was", "were",
})

_TRANSITIONS: frozenset[str] = frozenset({
    "however", "therefore", "furthermore", "moreover", "consequently",
    "additionally", "meanwhile", "nevertheless", "nonetheless", "thus",
    "hence", "firstly", "secondly", "finally", "then", "next", "also",
    "in addition", "on the other hand", "in contrast", "as a result",
    "for example", "for instance", "specifically",
})


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the inclusive interval [*lo*, *hi*]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


# ---------------------------------------------------------------------------
# Readability – Flesch Reading Ease
# ---------------------------------------------------------------------------

def _count_syllables(word: str) -> int:
    """Heuristic syllable count via vowel groups (``aeiouy``), minimum 1."""
    word = word.lower().strip()
    if not word:
        return 0
    # Count runs of vowel characters; at least one per word
    count = len(re.findall(r"[aeiouy]+", word))
    return max(count, 1)


def _compute_flesch_reading_ease(
    words: list[str], num_sentences: int
) -> tuple[float, float]:
    """Return (raw, normalized) Flesch Reading Ease.

    ``raw`` is clamped to [-20, 120] before normalising into [0, 1].
    """
    total_words = len(words)
    if total_words == 0 or num_sentences == 0:
        return -20.0, 0.0

    total_syllables = sum(_count_syllables(w) for w in words)
    raw = (
        206.835
        - 1.015 * (total_words / num_sentences)
        - 84.6 * (total_syllables / total_words)
    )
    raw = clamp(raw, -20.0, 120.0)
    normalized = clamp((raw + 20.0) / 120.0, 0.0, 1.0)
    return raw, normalized


# ---------------------------------------------------------------------------
# Lexical diversity
# ---------------------------------------------------------------------------

def _compute_lexical_diversity(words: list[str]) -> tuple[float, float]:
    """Return (raw, normalized) lexical diversity."""
    total = len(words)
    if total == 0:
        return 0.0, 0.0

    unique = len({w.lower() for w in words})
    raw = unique / total
    normalized = clamp(raw / 0.72, 0.0, 1.0)
    return raw, normalized


# ---------------------------------------------------------------------------
# Sentence variety
# ---------------------------------------------------------------------------

def _compute_sentence_variety(
    sentences: list[str],
) -> tuple[dict[str, float], float]:
    """Return (raw-dict, normalized) sentence variety.

    Normalized is 0.5 when there are fewer than 2 sentences.
    """
    word_counts = [len(s.split()) for s in sentences] if sentences else [0]

    if len(word_counts) < 2:
        return {"raw_std": 0.0, "raw_range": 0.0}, 0.5

    raw_std = statistics.pstdev(word_counts)
    raw_range = max(word_counts) - min(word_counts)
    mean_words = statistics.mean(word_counts)
    normalized = clamp(
        (raw_std / max(mean_words, 1.0)) / 0.8, 0.0, 1.0
    )
    return {"raw_std": raw_std, "raw_range": raw_range}, normalized


# ---------------------------------------------------------------------------
# Repetition
# ---------------------------------------------------------------------------

# Matches the stop-set used by analysis._sentence_openers
_OPENER_STOP: frozenset[str] = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "but",
    "or", "it", "its",
})


def _sentence_opener(sentence: str) -> str:
    """Return the first significant word, or ``""`` if none found."""
    for w in sentence.split()[:3]:
        clean = w.strip("\"\u201c\u201d'(").lower()
        if clean and clean not in _OPENER_STOP:
            return clean
    return ""


def _compute_repetition(
    sentences: list[str], words: list[str]
) -> tuple[list[float], float]:
    """Return (raw-list, normalized) repetition.

    Raw = [bigram_ratio, trigram_ratio, sentence_ratio, opener_ratio].
    """
    # Bigram / trigram repetition (position-based)
    lowered = [w.lower() for w in words]

    bigram_counts = Counter(
        tuple(lowered[i : i + 2]) for i in range(len(lowered) - 1)
    )
    trigram_counts = Counter(
        tuple(lowered[i : i + 3]) for i in range(len(lowered) - 2)
    )

    total_bigrams = len(lowered) - 1 if len(lowered) >= 2 else 0
    total_trigrams = len(lowered) - 2 if len(lowered) >= 3 else 0

    bigram_ratio = (
        sum(c for c in bigram_counts.values() if c > 1) / total_bigrams
        if total_bigrams > 0
        else 0.0
    )
    trigram_ratio = (
        sum(c for c in trigram_counts.values() if c > 1) / total_trigrams
        if total_trigrams > 0
        else 0.0
    )

    # Repeated-sentence ratio
    sent_counts: Counter[str] = Counter(s.lower() for s in sentences)
    total_sentences = len(sentences)
    sentence_ratio = (
        sum(c for c in sent_counts.values() if c > 1) / total_sentences
        if total_sentences > 0
        else 0.0
    )

    # Repeated-opener ratio
    openers = [_sentence_opener(s) for s in sentences]
    opener_counts: Counter[str] = Counter(openers)
    total_openers = len(openers)
    opener_ratio = (
        sum(c for c in opener_counts.values() if c > 1) / total_openers
        if total_openers > 0
        else 0.0
    )

    raw = [bigram_ratio, trigram_ratio, sentence_ratio, opener_ratio]
    mean_raw = statistics.mean(raw) if raw else 0.0
    normalized = 1.0 - clamp(mean_raw, 0.0, 1.0)
    return raw, normalized


# ---------------------------------------------------------------------------
# Coherence rubric
# ---------------------------------------------------------------------------

def _prompt_keywords(text: str) -> set[str]:
    """Return lowercase tokens of *text* that are not stopwords."""
    return {w.lower() for w in text.split() if w.lower() not in _STOPWORDS}


def _adjacent_overlap(sentences: list[str]) -> float:
    """Mean fraction of bigrams in sentence *i* that also appear in sentence *i+1*."""
    if len(sentences) < 2:
        return 0.0

    total = 0.0
    for i in range(len(sentences) - 1):
        words_a = sentences[i].split()
        words_b = sentences[i + 1].split()
        if not words_a:
            total += 0.0
            continue
        bigrams_a = {
            tuple(words_a[j : j + 2]) for j in range(len(words_a) - 1)
        }
        bigrams_b = {
            tuple(words_b[j : j + 2]) for j in range(len(words_b) - 1)
        }
        matches = len(bigrams_a & bigrams_b)
        total += matches / max(len(words_a), 1)

    return total / (len(sentences) - 1)


def _transition_balance(sentences: list[str]) -> float:
    """Return 1 - |p - 0.5| * 2 where *p* is the fraction of transition sentences."""
    if not sentences:
        return 0.0

    n_tr = 0
    for s in sentences:
        toks = s.lower().split()
        if not toks:
            continue
        # first word
        if toks[0] in _TRANSITIONS:
            n_tr += 1
            continue
        # first bigram
        if len(toks) >= 2 and " ".join(toks[:2]) in _TRANSITIONS:
            n_tr += 1

    proportion = n_tr / len(sentences)
    return 1.0 - abs(proportion - 0.5) * 2.0


def _reference_overlap(
    candidate_words: list[str], reference_text: str | None
) -> float | None:
    """Fraction of non-stopword reference unigrams present in candidate.

    Returns ``None`` when there is no reference.
    """
    if reference_text is None:
        return None

    ref_kw = _prompt_keywords(reference_text)
    if not ref_kw:
        return 0.0

    cand_lower = {w.lower() for w in candidate_words}
    matches = sum(1 for kw in ref_kw if kw in cand_lower)
    return matches / len(ref_kw)


# ---------------------------------------------------------------------------
# Style consistency
# ---------------------------------------------------------------------------

def _terminal_punctuation_consistency(sentences: list[str]) -> float:
    """Fraction: most common terminal-punctuation type / total sentences."""
    if not sentences:
        return 0.0

    punct_re = re.compile(r"[.!?…]+$")
    types: list[str] = []
    for s in sentences:
        m = punct_re.search(s.rstrip())
        if m:
            ch = m.group()[0]
            if ch == ".":
                types.append("period")
            elif ch == "!":
                types.append("exclamation")
            elif ch == "?":
                types.append("question")
            else:
                types.append("ellipsis")
        else:
            types.append("other")

    counts = Counter(types)
    return max(counts.values()) / len(sentences)


def _casing_consistency(sentences: list[str]) -> float:
    """Fraction of sentences starting with an uppercase letter."""
    if not sentences:
        return 0.0
    return sum(1 for s in sentences if s and s[0].isupper()) / len(sentences)


def _compute_style_consistency(
    sentences: list[str],
) -> tuple[dict[str, float], float]:
    """Return (raw-dict, normalized) style consistency."""
    word_counts = [len(s.split()) for s in sentences] if sentences else [0]
    n_sentences = len(sentences)

    # CV of sentence lengths
    if n_sentences >= 2:
        mean_wc = statistics.mean(word_counts)
        sentence_length_cv = (
            statistics.pstdev(word_counts) / mean_wc if mean_wc > 0 else 0.0
        )
    else:
        sentence_length_cv = 0.0

    punct = _terminal_punctuation_consistency(sentences)
    casing = _casing_consistency(sentences)

    length_norm = 1.0 - min(sentence_length_cv / 1.0, 1.0)
    punct_norm = punct
    casing_norm = casing
    normalized = statistics.mean([length_norm, punct_norm, casing_norm])

    return {
        "sentence_length_cv": sentence_length_cv,
        "terminal_punctuation_consistency": punct,
        "casing_consistency": casing,
    }, normalized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_writing_metrics(
    prompt: str,
    candidate: str,
    reference: str | None = None,
) -> dict:
    """Compute deterministic writing quality metrics for *candidate*.

    Parameters
    ----------
    prompt:
        The original writing prompt (used for keyword-coverage).
    candidate:
        The text to evaluate.
    reference:
        Optional reference/human-written text for overlap comparison.

    Returns
    -------
    dict with ``"raw"`` and ``"normalized"`` keys, each containing sub-keys
    for ``readability``, ``lexical_diversity``, ``sentence_variety``,
    ``repetition``, ``coherence_rubric``, and ``style_consistency``.
    """
    # --- Preprocessing ---------------------------------------------------
    sentences = split_sentences(candidate)
    normalized_candidate = normalize_for_detection(candidate)

    # If the candidate is non-empty but split_sentences rejects it as too
    # short (≤15 chars or <4 words), treat it as a single short sentence.
    if not sentences and candidate.strip():
        sentences = [normalized_candidate]

    words = normalized_candidate.split() if normalized_candidate else []

    # --- 1. Readability --------------------------------------------------
    readability_raw, readability_norm = _compute_flesch_reading_ease(
        words, len(sentences)
    )

    # --- 2. Lexical diversity -------------------------------------------
    lexdiv_raw, lexdiv_norm = _compute_lexical_diversity(words)

    # --- 3. Sentence variety ---------------------------------------------
    sentvar_raw, sentvar_norm = _compute_sentence_variety(sentences)

    # --- 4. Repetition ---------------------------------------------------
    rep_raw, rep_norm = _compute_repetition(sentences, words)

    # --- 5. Coherence rubric ---------------------------------------------
    prompt_kw = _prompt_keywords(prompt)
    cand_lower_words = [w.lower() for w in words]
    cand_lower_set = set(cand_lower_words)

    if prompt_kw:
        prompt_keyword_coverage = (
            sum(1 for kw in prompt_kw if kw in cand_lower_set) / len(prompt_kw)
        )
    else:
        prompt_keyword_coverage = 1.0

    adjacent_overlap = _adjacent_overlap(sentences)
    transition_balance = _transition_balance(sentences)
    ref_overlap = _reference_overlap(words, reference)

    coherence_raw: dict = {
        "prompt_keyword_coverage": prompt_keyword_coverage,
        "adjacent_overlap": adjacent_overlap,
        "transition_balance": transition_balance,
        "reference_overlap": ref_overlap,
    }

    if reference is not None:
        w = {
            "prompt_keyword": 0.30,
            "adjacent_overlap": 0.25,
            "transition_balance": 0.20,
            "reference_overlap": 0.25,
        }
    else:
        w = {
            "prompt_keyword": 0.40,
            "adjacent_overlap": 0.35,
            "transition_balance": 0.25,
            "reference_overlap": 0.0,
        }

    wsum = sum(w.values())
    coherence_norm = (
        w["prompt_keyword"] * prompt_keyword_coverage
        + w["adjacent_overlap"] * adjacent_overlap
        + w["transition_balance"] * transition_balance
        + (w["reference_overlap"] * ref_overlap if ref_overlap is not None else 0.0)
    ) / wsum

    # --- 6. Style consistency --------------------------------------------
    style_raw, style_norm = _compute_style_consistency(sentences)

    return {
        "raw": {
            "readability": readability_raw,
            "lexical_diversity": lexdiv_raw,
            "sentence_variety": sentvar_raw,
            "repetition": rep_raw,
            "coherence_rubric": coherence_raw,
            "style_consistency": style_raw,
        },
        "normalized": {
            "readability": readability_norm,
            "lexical_diversity": lexdiv_norm,
            "sentence_variety": sentvar_norm,
            "repetition": rep_norm,
            "coherence_rubric": coherence_norm,
            "style_consistency": style_norm,
        },
    }
