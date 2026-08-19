"""Deterministic writing quality metrics — no model, no advice."""

import re
import statistics
from collections import Counter

import numpy as np

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


# ---------------------------------------------------------------------------
# Grammar / specificity / evidence constants
# ---------------------------------------------------------------------------

_PASSIVE_RE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\s+[a-z]{3,}(?:ed|en)\b",
    re.IGNORECASE,
)

_WEAK_MODIFIERS: frozenset[str] = frozenset({
    "very", "really", "quite", "rather", "somewhat", "fairly", "pretty",
    "kind", "sort", "extremely", "incredibly", "totally", "completely",
    "absolutely", "highly", "deeply", "greatly", "strongly", "especially",
    "particularly", "mainly", "mostly", "basically", "literally", "actually",
    "definitely", "certainly", "probably", "maybe", "perhaps", "possibly",
    "just", "only", "simply", "merely", "almost", "nearly", "roughly",
    "around", "about", "approximately",
})

_VAGUE_WORDS: frozenset[str] = frozenset({
    "thing", "things", "stuff", "aspect", "aspects", "factor", "factors",
    "nature", "something", "someone", "somebody", "anything", "everything",
    "nothing", "everyone", "anyone", "good", "bad", "important", "interesting",
    "nice", "great", "hard", "easy", "big", "small", "lot", "lots", "many",
    "much", "various", "several", "certain", "general", "overall", "basically",
    "essentially", "ultimately", "obviously", "clearly", "definitely",
    "absolutely", "completely", "totally", "really", "very", "quite",
    "rather", "pretty", "fairly", "somewhat", "somehow", "somewhere",
    "sometime", "always", "never", "often", "sometimes", "usually",
    "probably", "maybe", "perhaps", "possibly", "likely", "unlikely",
})

_ABSTRACT_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ance", "ence", "ism", "cy")

_CITATION_RE = re.compile(
    r"\([A-Z][A-Za-z\s&]+,?\s*\d{4}[a-z]?\)|\[\d+\]|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\(\d{4}[a-z]?\)",
)

_EXAMPLE_SIGNALS: frozenset[str] = frozenset({
    "for example", "for instance", "such as", "e.g.", "i.e.", "namely",
    "to illustrate", "as an example", "take for example", "consider the",
    "illustrated by", "demonstrated by",
})

_DATA_SIGNALS: frozenset[str] = frozenset({
    "percent", "percentage", "study", "studies", "research", "survey",
    "surveys", "data", "statistics", "statistic", "finding", "findings",
    "result", "results", "found that", "showed that", "demonstrated that",
    "reported that", "according to", "evidence suggests", "studies show",
    "research indicates", "data shows", "figures show", "survey found",
})

_EXPERIENCE_SIGNALS: frozenset[str] = frozenset({
    "i observed", "i noticed", "i saw", "i experienced", "i witnessed",
    "i found", "i discovered", "i learned", "in my experience",
    "from my experience", "from my perspective", "in my view",
    "personally", "i believe", "i feel", "i think", "my experience",
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
# Burstiness / signal flags (KB 02 S5 humzakt bands; PhD S7 em-dash; lag-1
# autocorrelation "fake burstiness" heuristic)
# ---------------------------------------------------------------------------

def _classify_burstiness_band(ratio: float) -> str:
    """Classify std/mean sentence-length CV into a KB 02 S5 burstiness band."""
    if ratio < 0.30:
        return "ai_like"
    if ratio < 0.65:
        return "mixed"
    if ratio <= 0.85:
        return "human_like"
    return "over_varied"


def _em_dash_density(text: str, n_words: int) -> float:
    """Em-dashes (— and en-dashes –) per 1,000 words (S7)."""
    if n_words <= 0:
        return 0.0
    n_dashes = text.count("\u2014") + text.count("\u2013")
    return (n_dashes * 1000.0) / n_words


def _lag1_autocorrelation(word_counts: list[int]) -> float:
    """Normalized lag-1 autocorrelation of sentence-lengths (numpy)."""
    if len(word_counts) < 2:
        return 0.0
    arr = np.asarray(word_counts, dtype=float)
    arr = arr - arr.mean()
    denom = float(np.sum(arr * arr))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(arr[:-1] * arr[1:]) / denom)


def _compute_signal_flags(
    sentences: list[str],
    text: str,
    words: list[str],
    raw_sentence_variety: dict,
) -> dict:
    """Burstiness band, em-dash density, and fake-burstiness flags."""
    word_counts = [len(s.split()) for s in sentences] if sentences else [0]
    mean_wc = statistics.mean(word_counts) if word_counts else 0.0
    ratio = (
        raw_sentence_variety.get("raw_std", 0.0) / mean_wc
        if mean_wc > 0
        else 0.0
    )
    band = _classify_burstiness_band(ratio)

    n_words = len(words)
    emdash_density_raw = _em_dash_density(text, n_words)

    r1 = _lag1_autocorrelation(word_counts)
    if abs(r1) > 0.5 and len(word_counts) >= 6:
        fake_burstiness = "suspicious_alternation"
    else:
        fake_burstiness = "normal"

    return {
        "burstiness_band": band,
        "em_dash_density": emdash_density_raw,
        "fake_burstiness": fake_burstiness,
    }


# Grammar heuristics
# ---------------------------------------------------------------------------

def _compute_grammar_heuristics(
    sentences: list[str], words: list[str]
) -> tuple[dict[str, float], float]:
    """Return (raw-dict, normalized) grammar heuristics."""
    n_sentences = len(sentences) if sentences else 0
    n_words = len(words) if words else 0

    if n_sentences == 0 or n_words == 0:
        return {
            "passive_ratio": 0.0,
            "weak_modifier_ratio": 0.0,
            "runon_ratio": 0.0,
        }, 1.0

    passive_count = sum(1 for s in sentences if _PASSIVE_RE.search(s))
    passive_ratio = passive_count / n_sentences

    weak_count = sum(
        1 for w in words if w.lower().rstrip(".,;:!?") in _WEAK_MODIFIERS
    )
    weak_modifier_ratio = weak_count / n_words

    runon_count = 0
    for s in sentences:
        if s.count(",") > 2 and len(s.split()) > 25:
            runon_count += 1
    runon_ratio = runon_count / n_sentences

    raw = {
        "passive_ratio": passive_ratio,
        "weak_modifier_ratio": weak_modifier_ratio,
        "runon_ratio": runon_ratio,
    }

    passive_norm = 1.0 - clamp(passive_ratio / 0.5, 0.0, 1.0)
    weak_norm = 1.0 - clamp(weak_modifier_ratio / 0.15, 0.0, 1.0)
    runon_norm = 1.0 - clamp(runon_ratio / 0.5, 0.0, 1.0)
    normalized = statistics.mean([passive_norm, weak_norm, runon_norm])

    return raw, normalized


# ---------------------------------------------------------------------------
# Specificity
# ---------------------------------------------------------------------------

def _is_abstract_noun(word: str) -> bool:
    w = word.lower().rstrip(".,;:!?")
    return w.endswith(_ABSTRACT_SUFFIXES)


def _compute_specificity(
    sentences: list[str], words: list[str]
) -> tuple[dict[str, float], float]:
    """Return (raw-dict, normalized) specificity metrics."""
    n_sentences = len(sentences) if sentences else 0
    n_words = len(words) if words else 0

    if n_sentences == 0 or n_words == 0:
        return {
            "vague_word_ratio": 0.0,
            "abstract_noun_ratio": 0.0,
            "concrete_detail_ratio": 0.0,
        }, 0.5

    vague_count = sum(
        1 for w in words if w.lower().rstrip(".,;:!?") in _VAGUE_WORDS
    )
    vague_word_ratio = vague_count / n_words

    abstract_count = sum(1 for w in words if _is_abstract_noun(w))
    abstract_noun_ratio = abstract_count / n_words

    concrete_count = 0
    for s in sentences:
        has_number = bool(re.search(r"\d", s))
        has_date = bool(re.search(r"\b(?:19|20)\d{2}\b", s))
        has_percent = bool(re.search(r"\d+\s*%|\d+\s*percent", s, re.IGNORECASE))

        words_in_sent = s.split()
        has_proper = False
        for i, w in enumerate(words_in_sent):
            if i == 0:
                continue
            clean = w.strip("(),;:.!?\"'")
            if len(clean) > 1 and clean[0].isupper() and not clean.islower():
                has_proper = True
                break

        if has_number or has_date or has_percent or has_proper:
            concrete_count += 1

    concrete_detail_ratio = concrete_count / n_sentences

    raw = {
        "vague_word_ratio": vague_word_ratio,
        "abstract_noun_ratio": abstract_noun_ratio,
        "concrete_detail_ratio": concrete_detail_ratio,
    }

    vague_norm = 1.0 - clamp(vague_word_ratio / 0.10, 0.0, 1.0)
    abstract_norm = 1.0 - clamp(abstract_noun_ratio / 0.20, 0.0, 1.0)
    concrete_norm = clamp(concrete_detail_ratio / 0.50, 0.0, 1.0)
    normalized = statistics.mean([vague_norm, abstract_norm, concrete_norm])

    return raw, normalized


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

def _compute_evidence(sentences: list[str]) -> tuple[dict[str, float], float]:
    """Return (raw-dict, normalized) evidence metrics."""
    n_sentences = len(sentences) if sentences else 0

    if n_sentences == 0:
        return {
            "citation_density": 0.0,
            "example_density": 0.0,
            "data_density": 0.0,
            "experience_density": 0.0,
        }, 0.0

    citation_count = 0
    example_count = 0
    data_count = 0
    experience_count = 0

    for s in sentences:
        lower = s.lower()

        if _CITATION_RE.search(s):
            citation_count += 1

        for signal in _EXAMPLE_SIGNALS:
            if signal in lower:
                example_count += 1
                break

        for signal in _DATA_SIGNALS:
            if signal in lower:
                data_count += 1
                break

        for signal in _EXPERIENCE_SIGNALS:
            if signal in lower:
                experience_count += 1
                break

    raw = {
        "citation_density": citation_count / n_sentences,
        "example_density": example_count / n_sentences,
        "data_density": data_count / n_sentences,
        "experience_density": experience_count / n_sentences,
    }

    citation_norm = clamp(raw["citation_density"] / 0.30, 0.0, 1.0)
    example_norm = clamp(raw["example_density"] / 0.30, 0.0, 1.0)
    data_norm = clamp(raw["data_density"] / 0.30, 0.0, 1.0)
    experience_norm = clamp(raw["experience_density"] / 0.30, 0.0, 1.0)
    normalized = statistics.mean([citation_norm, example_norm, data_norm, experience_norm])

    return raw, normalized


# ---------------------------------------------------------------------------
# Sentence-level analysis
# ---------------------------------------------------------------------------

def _sentence_has_passive(sentence: str) -> bool:
    return bool(_PASSIVE_RE.search(sentence))

def _sentence_has_weak_modifiers(sentence: str) -> int:
    return sum(
        1 for w in sentence.split()
        if w.lower().rstrip(".,;:!?") in _WEAK_MODIFIERS
    )

def _sentence_is_runon(sentence: str) -> bool:
    return sentence.count(",") > 2 and len(sentence.split()) > 25

def _sentence_has_vague_words(sentence: str) -> int:
    return sum(
        1 for w in sentence.split()
        if w.lower().rstrip(".,;:!?") in _VAGUE_WORDS
    )

def _sentence_has_concrete_details(sentence: str) -> bool:
    has_number = bool(re.search(r"\d", sentence))
    has_date = bool(re.search(r"\b(?:19|20)\d{2}\b", sentence))
    has_percent = bool(re.search(r"\d+\s*%|\d+\s*percent", sentence, re.IGNORECASE))
    words_in_sent = sentence.split()
    has_proper = False
    for i, w in enumerate(words_in_sent):
        if i == 0:
            continue
        clean = w.strip("(),;:.!?\"'")
        if len(clean) > 1 and clean[0].isupper() and not clean.islower():
            has_proper = True
            break
    return has_number or has_date or has_percent or has_proper

def _sentence_has_evidence(sentence: str) -> list[str]:
    lower = sentence.lower()
    flags = []
    if _CITATION_RE.search(sentence):
        flags.append("citation")
    for signal in _EXAMPLE_SIGNALS:
        if signal in lower:
            flags.append("example")
            break
    for signal in _DATA_SIGNALS:
        if signal in lower:
            flags.append("data")
            break
    for signal in _EXPERIENCE_SIGNALS:
        if signal in lower:
            flags.append("experience")
            break
    return flags


def _sentence_has_synthesis(sentence: str) -> bool:
    """Detect literature-review synthesis patterns as evidence."""
    lower = sentence.lower()
    synthesis_patterns = (
        "several studies",
        "multiple studies",
        "research generally",
        "research suggests",
        "research indicates",
        "research has found",
        "research has shown",
        "literature suggests",
        "literature indicates",
        "prior research",
        "previous research",
        "existing literature",
        "the literature",
        "in contrast to",
        "in contrast,",
        "similarly,",
        "conversely,",
        "building on",
        "extending",
        "consistent with",
        "in line with",
        "aligns with",
        "contradicts",
        "however,",
        "nevertheless,",
        "although",
        "whereas",
        "while",
        "despite",
        "compared to",
    )
    return any(p in lower for p in synthesis_patterns)


# Severity weights for sentence-level quality flags (higher = more severe).
_FLAG_SEVERITY: dict[str, float] = {
    "runon": 3.0,
    "passive_voice": 2.0,
    "no_evidence": 2.0,
    "no_concrete_details": 1.5,
    "weak_modifiers": 1.0,
    "vague_words": 1.0,
    "synthesis": 0.0,
    "citation": 0.0,
    "example": 0.0,
    "data": 0.0,
    "experience": 0.0,
}


def _compute_sentence_quality_score(flags: list[str]) -> float:
    """Return a quality score from 0 (worst) to 10 (best) based on flags."""
    if not flags:
        return 10.0

    total_penalty = 0.0
    for flag in flags:
        base = flag.split("(")[0]  # strip counts like "weak_modifiers(2)"
        severity = _FLAG_SEVERITY.get(base, 1.0)
        # Extract count if present (e.g., "weak_modifiers(2)")
        count_match = re.search(r"\((\d+)\)", flag)
        count = int(count_match.group(1)) if count_match else 1
        total_penalty += severity * count

    score = 10.0 - total_penalty
    return clamp(score, 0.0, 10.0)


def analyze_sentences(sentences: list[str]) -> list[dict]:
    """Return per-sentence quality flags and a weighted quality score."""
    results = []
    for s in sentences:
        flags = []
        if _sentence_has_passive(s):
            flags.append("passive_voice")
        weak_count = _sentence_has_weak_modifiers(s)
        if weak_count > 0:
            flags.append(f"weak_modifiers({weak_count})")
        if _sentence_is_runon(s):
            flags.append("runon")
        vague_count = _sentence_has_vague_words(s)
        if vague_count > 0:
            flags.append(f"vague_words({vague_count})")
        if not _sentence_has_concrete_details(s):
            flags.append("no_concrete_details")
        ev_flags = _sentence_has_evidence(s)
        if not ev_flags and not _sentence_has_synthesis(s):
            flags.append("no_evidence")
        else:
            flags.extend(ev_flags)
            if _sentence_has_synthesis(s) and "synthesis" not in flags:
                flags.append("synthesis")
        score = _compute_sentence_quality_score(flags)
        results.append({
            "text": s,
            "flags": flags,
            "score": round(score, 1),
        })
    return results


# ---------------------------------------------------------------------------
# Top-worst sentences
# ---------------------------------------------------------------------------

def _score_sentence_quality(sentence: str) -> dict[str, float]:
    """Return per-dimension quality scores for a single sentence."""
    words = sentence.split()
    n_words = len(words)

    # Grammar: lower is worse
    passive = 1.0 if _sentence_has_passive(sentence) else 0.0
    weak = _sentence_has_weak_modifiers(sentence) / max(n_words, 1)
    runon = 1.0 if _sentence_is_runon(sentence) else 0.0
    grammar_score = 1.0 - (passive + weak + runon) / 3.0

    # Specificity: lower is worse
    vague = _sentence_has_vague_words(sentence) / max(n_words, 1)
    abstract = sum(1 for w in words if _is_abstract_noun(w)) / max(n_words, 1)
    concrete = 1.0 if _sentence_has_concrete_details(sentence) else 0.0
    specificity_score = (1.0 - vague + 1.0 - abstract + concrete) / 3.0

    # Evidence: higher is better (synthesis counts as evidence)
    ev_flags = _sentence_has_evidence(sentence)
    has_synthesis = _sentence_has_synthesis(sentence)
    evidence_score = 1.0 if (ev_flags or has_synthesis) else 0.0

    # Readability: sentence length as proxy
    readability_score = 1.0 - min(n_words / 50.0, 1.0)

    return {
        "grammar": clamp(grammar_score, 0.0, 1.0),
        "specificity": clamp(specificity_score, 0.0, 1.0),
        "evidence": evidence_score,
        "readability": clamp(readability_score, 0.0, 1.0),
    }

def get_top_worst_sentences(sentences: list[str], dimension: str, top_n: int = 3) -> list[tuple[str, float]]:
    """Return the top_n worst sentences for a given dimension."""
    scored = [(s, _score_sentence_quality(s)[dimension]) for s in sentences]
    scored.sort(key=lambda x: x[1])
    return scored[:top_n]


# ---------------------------------------------------------------------------
# Paragraph analysis
# ---------------------------------------------------------------------------

# Regex for cleaning LaTeX/PDF artifacts before paragraph splitting.
_PAGE_NUMBER_RE = re.compile(r"\n\s*\d+\s*\n")
_FORM_FEED_RE = re.compile(r"\f")
_HEADER_FOOTER_RE = re.compile(r"\n\s*Figure\s+\d+:\s*[^\n]*\n", re.IGNORECASE)


def _clean_text_for_paragraphs(text: str) -> str:
    """Strip page numbers, form feeds, and figure captions before splitting."""
    text = _FORM_FEED_RE.sub("\n", text)
    text = _PAGE_NUMBER_RE.sub("\n", text)
    text = _HEADER_FOOTER_RE.sub("\n", text)
    # Collapse multiple newlines to double-newline (paragraph boundary)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[list[str]]:
    """Split text into paragraphs, each containing its sentences."""
    text = _clean_text_for_paragraphs(text)
    paragraphs = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        # Skip short blocks that are likely headers or page numbers
        if len(block) < 20 and not block.endswith((".", "!", "?")):
            continue
        sents = split_sentences(block)
        if sents:
            paragraphs.append(sents)
    return paragraphs

def analyze_paragraphs(text: str) -> dict:
    """Analyze paragraph structure and flow."""
    paragraphs = split_paragraphs(text)
    n_paragraphs = len(paragraphs)

    if n_paragraphs == 0:
        return {
            "paragraph_count": 0,
            "avg_sentences_per_paragraph": 0.0,
            "paragraph_length_variety": 0.0,
            "topic_sentence_coverage": 0.0,
            "paragraph_transition_coverage": 0.0,
            "thesis_alignment": 0.0,
        }

    sentence_counts = [len(p) for p in paragraphs]
    avg_sentences = statistics.mean(sentence_counts)

    if n_paragraphs >= 2:
        length_variety = clamp(
            statistics.pstdev(sentence_counts) / max(avg_sentences, 1.0) / 1.0,
            0.0, 1.0,
        )
    else:
        length_variety = 0.0

    # Topic sentences: first sentence of paragraph should contain a claim
    # Heuristic: first sentence has a verb and is not a transition/question
    topic_sentences = 0
    for p in paragraphs:
        if not p:
            continue
        first = p[0].lower()
        if first.endswith("?"):
            continue
        # Check if it starts with a transition (likely not a topic sentence)
        first_words_raw = first.split()[:2]
        first_words = [w.strip(".,;:!?") for w in first_words_raw]
        first_bigram = " ".join(first_words)
        if first_bigram in _TRANSITIONS or first_words[0] in _TRANSITIONS:
            continue
        # Check if it contains a verb-like claim
        if re.search(r"\b(?:is|are|was|were|shows|demonstrates|argues|suggests|indicates|found|found that|can|should|will|may|might|could|would)\b", first):
            topic_sentences += 1

    topic_coverage = topic_sentences / n_paragraphs

    # Paragraph transitions: first sentence of paragraph starts with transition
    transitions = 0
    for p in paragraphs:
        if not p:
            continue
        first_raw = p[0].lower().split()[:2]
        first = [w.strip(".,;:!?") for w in first_raw]
        first_bigram = " ".join(first)
        if first_bigram in _TRANSITIONS or first[0] in _TRANSITIONS:
            transitions += 1

    transition_coverage = transitions / n_paragraphs

    # Thesis alignment: check intro/conclusion keyword overlap
    intro_words = set()
    conclusion_words = set()
    if n_paragraphs >= 2:
        intro = paragraphs[0]
        conclusion = paragraphs[-1]
        intro_words = _prompt_keywords(" ".join(intro))
        conclusion_words = _prompt_keywords(" ".join(conclusion))

    if intro_words:
        thesis_alignment = len(intro_words & conclusion_words) / len(intro_words)
    else:
        thesis_alignment = 0.0

    return {
        "paragraph_count": n_paragraphs,
        "avg_sentences_per_paragraph": avg_sentences,
        "paragraph_length_variety": length_variety,
        "topic_sentence_coverage": topic_coverage,
        "paragraph_transition_coverage": transition_coverage,
        "thesis_alignment": thesis_alignment,
    }


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
    ``repetition``, ``coherence_rubric``, ``style_consistency``,
    ``grammar_heuristics``, ``specificity``, and ``evidence``.
    Additionally includes ``"sentence_analysis"`` and ``"paragraph_analysis"``.
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

    # --- 7. Signal flags (burstiness band, em-dash density, fake burstiness) --
    sig = _compute_signal_flags(sentences, candidate, words, sentvar_raw)
    w_norm = clamp(sig["em_dash_density"] / 20.0, 0.0, 1.0)

    # --- 8. Grammar heuristics -------------------------------------------
    grammar_raw, grammar_norm = _compute_grammar_heuristics(sentences, words)

    # --- 9. Specificity --------------------------------------------------
    specificity_raw, specificity_norm = _compute_specificity(sentences, words)

    # --- 10. Evidence -----------------------------------------------------
    evidence_raw, evidence_norm = _compute_evidence(sentences)

    # --- 11. Sentence-level analysis -------------------------------------
    sentence_analysis = analyze_sentences(sentences)

    # --- 12. Paragraph analysis ------------------------------------------
    paragraph_analysis = analyze_paragraphs(candidate)

    return {
        "raw": {
            "readability": readability_raw,
            "lexical_diversity": lexdiv_raw,
            "sentence_variety": sentvar_raw,
            "repetition": rep_raw,
            "coherence_rubric": coherence_raw,
            "style_consistency": style_raw,
            "signal_flags": {
                "burstiness_band": sig["burstiness_band"],
                "em_dash_density": sig["em_dash_density"],
                "fake_burstiness": sig["fake_burstiness"],
            },
            "grammar_heuristics": grammar_raw,
            "specificity": specificity_raw,
            "evidence": evidence_raw,
        },
        "normalized": {
            "readability": readability_norm,
            "lexical_diversity": lexdiv_norm,
            "sentence_variety": sentvar_norm,
            "repetition": rep_norm,
            "coherence_rubric": coherence_norm,
            "style_consistency": style_norm,
            "signal_flags": {
                "burstiness_band": sig["burstiness_band"],
                "em_dash_density": round(w_norm, 4),
                "fake_burstiness": sig["fake_burstiness"],
            },
            "grammar_heuristics": grammar_norm,
            "specificity": specificity_norm,
            "evidence": evidence_norm,
        },
        "sentence_analysis": sentence_analysis,
        "paragraph_analysis": paragraph_analysis,
    }
