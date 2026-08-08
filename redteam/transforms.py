"""
Transformation pipeline for AI-generated text.

Each transform takes a string and returns a transformed string.
Transforms are designed to mimic human editing patterns.
"""

import random
import re
from typing import Callable

# ─── Synonym banks ───────────────────────────────────────────────
# Common AI filler words → more natural alternatives
AI_FILLER_REPLACEMENTS: dict[str, list[str]] = {
    "crucial": ["important", "key", "vital", "essential", "necessary"],
    "leverage": ["use", "apply", "draw on", "tap into", "put to work"],
    "foster": ["encourage", "promote", "nurture", "build", "grow"],
    "comprehensive": ["thorough", "complete", "full", "detailed", "wide-ranging"],
    "innovative": ["new", "creative", "novel", "fresh", "original"],
    "facilitate": ["help", "enable", "support", "make easier", "speed up"],
    "utilize": ["use", "employ", "apply", "make use of"],
    "implement": ["set up", "roll out", "put in place", "deploy", "start"],
    "subsequently": ["then", "after that", "next", "later"],
    "furthermore": ["also", "plus", "on top of that", "and"],
    "moreover": ["also", "besides", "in addition", "plus"],
    "consequently": ["so", "as a result", "therefore", "because of that"],
    "nevertheless": ["still", "yet", "even so", "all the same"],
    "notably": ["especially", "particularly", "in particular", "above all"],
    "significant": ["big", "major", "important", "meaningful", "notable"],
    "demonstrate": ["show", "prove", "illustrate", "make clear"],
    "establish": ["set up", "create", "build", "found"],
    "approach": ["method", "way", "strategy", "technique", "tactic"],
    "framework": ["structure", "system", "model", "scaffold"],
    "landscape": ["field", "scene", "world", "terrain", "picture"],
    "paradigm": ["model", "approach", "way of thinking", "pattern"],
    "synergy": ["collaboration", "teamwork", "combined effort", "partnership"],
    "robust": ["strong", "solid", "reliable", "sturdy", "tough"],
    "streamline": ["simplify", "speed up", "optimize", "make smoother"],
    "holistic": ["complete", "full", "well-rounded", "broad", "overall"],
    "dynamic": ["active", "changing", "lively", "flexible", "adaptable"],
    "multifaceted": ["complex", "varied", "layered", "many-sided"],
    "pivotal": ["key", "central", "critical", "important"],
    "seamless": ["smooth", "easy", "effortless", "fluid"],
    "unprecedented": ["new", "unlike anything before", "extraordinary", "remarkable"],
    "navigate": ["work through", "deal with", "handle", "manage"],
    "landscape": ["field", "area", "domain", "space"],
    "delve": ["dig into", "explore", "look into", "go deep on"],
    "embark": ["start", "begin", "set out", "kick off"],
    "realm": ["area", "field", "domain", "world"],
    "tapestry": ["mix", "blend", "collection", "patchwork"],
    "myriad": ["many", "lots of", "countless", "a range of"],
    "underscores": ["highlights", "shows", "makes clear", "emphasizes"],
    "amidst": ["in", "during", "while", "among"],
    "prioritize": ["focus on", "put first", "rank higher", "emphasize"],
    "endeavor": ["attempt", "try", "effort", "project"],
    "elucidate": ["explain", "clarify", "shed light on", "spell out"],
    "contribute": ["add", "help", "play a part", "matter"],
}

# Sentence openers that AI loves
AI_OPENERS = [
    "It is worth noting that",
    "It is important to note that",
    "It should be noted that",
    "It goes without saying that",
    "As mentioned earlier",
    "In today's world",
    "In the modern era",
    "In this day and age",
    "With the advent of",
    "When it comes to",
    "In terms of",
    "From a broader perspective",
    "That said",
    "Having said that",
    "In this context",
    "In light of this",
    "On a fundamental level",
    "At its core",
]

# Natural human openers
HUMAN_OPENERS = [
    "Honestly,",
    "Look,",
    "The thing is,",
    "Here's what matters:",
    "The way I see it,",
    "So basically,",
    "What this really means is",
    "Think about it this way:",
    "You know how",
    "The short version:",
    "Quick take:",
    "Bottom line:",
    "Real talk:",
    "Let me put it this way:",
]


def replace_ai_filler(text: str) -> str:
    """Replace AI-typical vocabulary with more natural alternatives."""
    result = text
    for word, alternatives in AI_FILLER_REPLACEMENTS.items():
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        matches = list(pattern.finditer(result))
        # Replace ~70% of occurrences randomly
        for match in reversed(matches):
            if random.random() < 0.7:
                replacement = random.choice(alternatives)
                # Preserve capitalization
                if match.group()[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                result = result[:match.start()] + replacement + result[match.end():]
    return result


def replace_ai_openers(text: str) -> str:
    """Replace stiff AI sentence openers with more natural ones."""
    result = text
    for opener in AI_OPENERS:
        if random.random() < 0.6:
            pattern = re.compile(r'^' + re.escape(opener) + r'\s+', re.MULTILINE | re.IGNORECASE)
            replacement = random.choice(HUMAN_OPENERS) + " "
            result = pattern.sub(replacement, result, count=1)
    return result


def remove_hedging(text: str) -> str:
    """Remove excessive hedging and filler phrases."""
    hedging_patterns = [
        (r'It is important to note that\s*', ''),
        (r'It is worth noting that\s*', ''),
        (r'It should be noted that\s*', ''),
        (r'It goes without saying that\s*', ''),
        (r'In other words,\s*', ''),
        (r'To put it simply,\s*', ''),
        (r'In essence,\s*', ''),
        (r'At its core,\s*', ''),
        (r'In conclusion,\s*', ''),
        (r'In summary,\s*', ''),
        (r'To summarize,\s*', ''),
        (r'Overall,\s*', ''),
        (r'All in all,\s*', ''),
        (r'On the other hand,\s*', ''),
        (r'On the contrary,\s*', ''),
        (r'However,\s*', ''),
        (r'Nevertheless,\s*', 'Still, '),
        (r'Furthermore,\s*', ''),
        (r'Moreover,\s*', ''),
        (r'Additionally,\s*', ''),
        (r'Consequently,\s*', 'So, '),
        (r'Subsequently,\s*', 'Then, '),
        (r'Notwithstanding\s*', ''),
    ]
    result = text
    for pattern, replacement in hedging_patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def split_long_sentences(text: str) -> str:
    """Break up overly long sentences (common in AI text)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for sent in sentences:
        if len(sent) > 120 and ', ' in sent:
            # Try to split at a comma
            parts = sent.split(', ', 1)
            if len(parts) == 2 and len(parts[0]) > 20 and len(parts[1]) > 20:
                # Capitalize the second part
                parts[1] = parts[1][0].upper() + parts[1][1:]
                result.append(parts[0] + '.')
                result.append(parts[1])
                continue
        result.append(sent)
    return ' '.join(result)


def add_transitions(text: str) -> str:
    """Add more varied transitions between sentences."""
    transition_words = [
        "Meanwhile, ", "Still, ", "Then again, ", "On top of that, ",
        "The flip side is ", "Here's the catch: ", "What's interesting is ",
        "The problem with that is ", "But here's where it gets tricky: ",
        "That brings us to ", "Here's why this matters: ",
    ]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) < 3:
        return text
    # Add a transition to ~30% of sentences (not the first)
    result = [sentences[0]]
    for i, sent in enumerate(sentences[1:], 1):
        if random.random() < 0.3 and sent and not any(sent.startswith(t.strip()) for t in transition_words):
            result.append(random.choice(transition_words) + sent[0].lower() + sent[1:])
        else:
            result.append(sent)
    return ' '.join(result)


def vary_sentence_length(text: str) -> str:
    """Break monotony in sentence length."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) < 4:
        return text
    
    result = []
    for i, sent in enumerate(sentences):
        words = sent.split()
        # If 3+ consecutive sentences are similar length, try to break one
        if i >= 2:
            prev_lens = [len(s.split()) for s in sentences[max(0,i-2):i]]
            curr_len = len(words)
            if all(abs(curr_len - pl) < 5 for pl in prev_lens) and curr_len > 15:
                # Split this sentence
                mid = len(words) // 2
                part1 = ' '.join(words[:mid]) + '.'
                part2 = ' '.join(words[mid:mid+3]) + ' ' + ' '.join(words[mid+3:])
                result.append(part1)
                result.append(part2)
                continue
        result.append(sent)
    return ' '.join(result)


def inject_personal_touch(text: str) -> str:
    """Add occasional personal phrasing to break AI monotony."""
    # Add a personal touch to the beginning ~20% of the time
    if random.random() < 0.2:
        intros = [
            "I've been thinking about this, and ",
            "Here's what I keep coming back to: ",
            "The more I look at this, the more ",
            "Something that doesn't get enough attention: ",
            "What strikes me about this is ",
        ]
        sentences = re.split(r'(?<=[.!?])\s+', text, maxsplit=1)
        if len(sentences) >= 2:
            return random.choice(intros) + sentences[0][0].lower() + sentences[0][1:] + ' ' + sentences[1]
    return text


def lowercase_after_connector(text: str) -> str:
    """Make sure words after connectors aren't always capitalized (AI habit)."""
    connectors = ['However,', 'Furthermore,', 'Moreover,', 'Additionally,', 
                  'Consequently,', 'Nevertheless,', 'Therefore,']
    result = text
    for conn in connectors:
        # After "However, The" → "However, the" (50% of the time)
        pattern = re.compile(r'(' + re.escape(conn) + r')\s+([A-Z])')
        if random.random() < 0.5:
            result = pattern.sub(lambda m: m.group(1) + ' ' + m.group(2).lower(), result)
    return result


def reduce_bullet_points(text: str) -> str:
    """Convert obvious list-like text into flowing prose."""
    # Match lines starting with - or • or * or numbered
    bullet_pattern = re.compile(r'^[-•*]\s+(.+)$|^\d+[.)]\s+(.+)$', re.MULTILINE)
    matches = list(bullet_pattern.finditer(text))
    if not matches:
        return text
    # Only convert ~40% of bullets
    result = text
    for match in reversed(matches):
        if random.random() < 0.4:
            content = match.group(1) or match.group(2)
            connector = random.choice(['Also, ', 'Plus, ', 'And ', 'On top of that, ', ''])
            replacement = connector + content[0].lower() + content[1:]
            if not replacement.rstrip().endswith('.'):
                replacement = replacement.rstrip() + '.'
            result = result[:match.start()] + replacement + result[match.end():]
    return result


# ─── Pipeline ─────────────────────────────────────────────────────

ALL_TRANSFORMS: list[tuple[str, Callable[[str], str]]] = [
    ("replace_ai_filler", replace_ai_filler),
    ("replace_ai_openers", replace_ai_openers),
    ("remove_hedging", remove_hedging),
    ("split_long_sentences", split_long_sentences),
    ("vary_sentence_length", vary_sentence_length),
    ("inject_personal_touch", inject_personal_touch),
    ("lowercase_after_connector", lowercase_after_connector),
    ("reduce_bullet_points", reduce_bullet_points),

]

def apply_random_transforms(text: str, n: int = 3) -> tuple[str, list[str]]:
    """Apply n random transforms and return (result, list of applied transform names)."""
    chosen = random.sample(ALL_TRANSFORMS, min(n, len(ALL_TRANSFORMS)))
    result = text
    applied = []
    for name, fn in chosen:
        new_text = fn(result)
        if new_text != result:
            applied.append(name)
            result = new_text
    return result, applied


def apply_all_transforms(text: str) -> tuple[str, list[str]]:
    """Apply every transform once. Returns (result, list of applied names)."""
    result = text
    applied = []
    for name, fn in ALL_TRANSFORMS:
        new_text = fn(result)
        if new_text != result:
            applied.append(name)
            result = new_text
    return result, applied


def apply_specific_transforms(text: str, names: list[str]) -> tuple[str, list[str]]:
    """Apply a specific ordered list of transforms."""
    transform_map = {name: fn for name, fn in ALL_TRANSFORMS}
    result = text
    applied = []
    for name in names:
        if name in transform_map:
            new_text = transform_map[name](result)
            if new_text != result:
                applied.append(name)
                result = new_text
    return result, applied


# ─── LLM-based rewrite ────────────────────────────────────────────
# Uses Ollama API to rewrite text in a more human style.
# This is the heavy hitter — changes the statistical fingerprint.
# Uses stream=false + think=false for ~2s per call instead of ~30s.

OLLAMA_URL = "http://localhost:11434/api/generate"


def _ollama_generate(
    prompt: str,
    model: str = "qwen3:1.7b",
    timeout: int = 60,
    options: dict | None = None,
) -> str:
    """Call Ollama HTTP API with thinking disabled for fast inference.

    *options* is an optional dict of Ollama sampling options (``temperature``,
    ``top_p``, ``repeat_penalty``, ...) merged over a sane default of
    ``{"temperature": 0.7}``. Sampling-level attack transforms pass their own
    options through this dict (KB 03 §3).
    """
    import urllib.request
    import json as _json

    opts = {"temperature": 0.7}
    if options:
        opts.update(options)

    payload = _json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": opts,
    }).encode("utf-8")
    
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
    
    output = data.get("response", "").strip()
    # Clean up any <think> tags
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
    return output


def _chunk_text(text: str, max_words: int = 150) -> list[str]:
    """Split text into chunks that fit in the model's context window."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    for sent in sentences:
        words = len(sent.split())
        if current_len + words > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sent]
            current_len = words
        else:
            current_chunk.append(sent)
            current_len += words
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def llm_rewrite(text: str, model: str = "qwen3:1.7b") -> str:
    """Rewrite text using a local LLM to sound more human.
    
    This changes the token probability distribution that detectors look at,
    unlike simple word replacement which only changes surface vocabulary.
    """
    chunks = _chunk_text(text)
    rewritten = []
    
    for chunk in chunks:
        prompt = (
            "Rewrite the following text to sound like a natural human wrote it. "
            "Keep the same meaning and information. Make it sound like a smart "
            "person writing casually for a professional audience. Vary sentence "
            "length. Use contractions. Avoid corporate/AI buzzwords. "
            "Return ONLY the rewritten text, nothing else.\n\n"
            f"Text:\n{chunk}"
        )
        
        try:
            output = _ollama_generate(prompt, model=model, timeout=60)
            if output:
                rewritten.append(output)
            else:
                rewritten.append(chunk)
        except Exception:
            rewritten.append(chunk)
    
    return ' '.join(rewritten)


def llm_rewrite_aggressive(text: str, model: str = "qwen3:1.7b") -> str:
    """More aggressive rewrite — inject personality and break AI patterns harder."""
    prompt = (
        "Rewrite this text completely. Make it sound like an experienced "
        "professional wrote it in a blog post or email. Use: "
        "contractions, short sentences mixed with long ones, "
        "concrete examples instead of abstractions, occasional informal phrases. "
        "Remove all corporate buzzwords (leverage, foster, paradigm, tapestry, etc). "
        "Add specific details where the original was vague. "
        "Return ONLY the rewritten text, nothing else.\n\n"
        f"Text:\n{text}"
    )
    
    try:
        output = _ollama_generate(prompt, model=model, timeout=120)
        return output if output else text
    except Exception:
        return text


def llm_rewrite_paraphrase(text: str, model: str = "qwen3:1.7b") -> str:
    """Paraphrase each sentence independently — breaks burstiness patterns."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    rewritten = []
    
    for sent in sentences:
        if not sent.strip():
            continue
        prompt = (
            f"Paraphrase this sentence to sound more natural and human-written. "
            f"Keep the meaning. Use different words and structure. "
            f"Return ONLY the paraphrased sentence, nothing else.\n\n"
            f"Sentence: {sent}"
        )
        try:
            output = _ollama_generate(prompt, model=model, timeout=30)
            if output:
                rewritten.append(output)
            else:
                rewritten.append(sent)
        except Exception:
            rewritten.append(sent)
    
    return ' '.join(rewritten)


# ─── Sampling-level attack transforms ────────────────────────────
# KB 03 §3: temperature / repetition-penalty / top-p sweeps flatten the
# too-smooth statistical signature detectors rely on. Ollama exposes them as
# per-call ``options``; we keep the existing ``_ollama_generate`` contract and
# pass the attacker-chosen options through the ``options`` dict.
# NOTE: these regenerated the WHOLE text via Ollama, so they are kept OUT of
# the pure-regex ``ALL_TRANSFORMS`` list (which ``apply_*`` helpers run with a
# single ``Callable[[str], str]``). They live in ``SAMPLING_TRANSFORMS`` and are
# dispatched by ``loop.py`` via ``apply_sampling_transform`` (Task 7.1).

_SAMPLING_REGEN_PROMPT = (
    "Regenerate the following text, preserving its meaning, information and "
    "structure exactly. Make no commentary. Return ONLY the regenerated text, "
    "nothing else.\n\nText:\n"
)


def sample_temperature(text: str, model: str = "qwen3:1.7b", temperature: float = 1.4) -> str:
    """Regenerate *text* with elevated sampling temperature (flattens the
    next-token distribution; ``T = 1.4`` breaks the too-smooth signature,
    KB 03 §3.2)."""
    return _ollama_generate(
        _SAMPLING_REGEN_PROMPT + text,
        model=model,
        timeout=120,
        options={"temperature": temperature},
    )


def sample_rep_penalty(text: str, model: str = "qwen3:1.7b", rep_penalty: float = 1.08) -> str:
    """Regenerate *text* with a mild repetition/frequency penalty.

    Ollama exposes this as the ``repeat_penalty`` option (its official option
    name; the API rejects ``frequency_penalty``). 1.05-1.15 is the sweet spot;
    stronger values degrade coherence (KB 03 §3.2)."""
    return _ollama_generate(
        _SAMPLING_REGEN_PROMPT + text,
        model=model,
        timeout=120,
        options={"repeat_penalty": rep_penalty},
    )


def sample_top_p(text: str, model: str = "qwen3:1.7b", top_p: float = 0.9) -> str:
    """Regenerate *text* with nucleus sampling ``top_p = 0.9`` (changes tail
    behavior of the next-token pool; weak standalone, pairs with T≈1, KB 03)."""
    return _ollama_generate(
        _SAMPLING_REGEN_PROMPT + text,
        model=model,
        timeout=120,
        options={"top_p": top_p},
    )


# Sampling transforms are (name, callable) pairs whose signature is
# ``fn(text, model=..., **opt)`` — NOT the plain ``Callable[[str], str]`` used
# by ``ALL_TRANSFORMS``.
SAMPLING_TRANSFORMS: list[tuple[str, Callable]] = [
    ("sample_temperature", sample_temperature),
    ("sample_rep_penalty", sample_rep_penalty),
    ("sample_top_p", sample_top_p),
]
SAMPLING_TRANSFORM_MAP: dict[str, Callable] = dict(SAMPLING_TRANSFORMS)


def apply_sampling_transform(
    text: str,
    name: str,
    model: str = "qwen3:1.7b",
) -> tuple[str, list[str]]:
    """Apply a single named sampling transform (Ollama-backed).

    Returns ``(result, [name])`` for symmetry with the regex ``apply_*``
    helpers, or ``(text, [])`` when the name is unknown.
    """
    fn = SAMPLING_TRANSFORM_MAP.get(name)
    if fn is None:
        return text, []
    return fn(text, model=model), [name]


def combo_sample_all(text: str, model: str = "qwen3:1.7b") -> str:
    """Chain temperature + repetition penalty + top-p regeneration (Task 7.1
    combo). Each step regenerates the whole text with its sampling option."""
    result = text
    for _name, fn in SAMPLING_TRANSFORMS:
        result = fn(result, model=model)
    return result


# ─── Iterative paraphrase loop with verify (Task 7.2) ────────────
# KB 03 §5: generate → detect → rewrite ONLY flagged chunks → max ~3 rounds →
# keep best. `detect_fn: Callable[[str], float]` (lower = more human-like AI
# probability) is INJECTED by the caller so the loop stays decoupled from any
# particular detector.

_PARAPHRASE_FLAG_THRESHOLD = 0.5  # detect() > threshold → chunk is "flagged"


def _rewrite_flagged_chunks(
    text: str,
    model: str,
    detect_fn,
    flag_threshold: float = _PARAPHRASE_FLAG_THRESHOLD,
) -> str:
    """Rewrite only the sentence-chunks that still score above *flag_threshold*.

    Low-scoring chunks are kept byte-identical. This is the "re-iterate only
    flagged paragraphs" step of the KB pipeline (KB 03 §5, layer 7)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    out = []
    for sent in sentences:
        if not sent.strip():
            continue
        score = detect_fn(sent)
        if score > flag_threshold:
            prompt = (
                "Rewrite this sentence to sound like a natural human wrote it. "
                "Keep the meaning. Vary the structure, avoid machine-favored "
                "words. Return ONLY the rewritten sentence, nothing else.\n\n"
                f"Sentence: {sent}"
            )
            try:
                rewritten = _ollama_generate(prompt, model=model, timeout=60)
                out.append(rewritten if rewritten else sent)
            except Exception:
                out.append(sent)
        else:
            out.append(sent)
    return ' '.join(out)


def paraphrase_with_verify(
    text: str,
    model: str = "qwen3:1.7b",
    detect_fn: Callable[[str], float] | None = None,
    max_rounds: int = 3,
) -> tuple[str, float, int]:
    """Iterative paraphrase with detector verification (KB 03 §5 pipeline).

    * Round 1 rewrites the WHOLE text (reuses the ``llm_rewrite`` logic).
    * Rounds 2..max_rounds rewrite ONLY the chunks a per-chunk detection still
      flags.
    * Best-scoring candidate is kept across rounds; never worse than the input.

    Returns ``(best_text, best_score, rounds_used)``.

    ``detect_fn`` is injected; ``None`` disables the verify signal (keeps the
    input unchanged, best_score = 1.0)."""
    if detect_fn is None:
        detect_fn = lambda _t: 1.0  # never improves, keep original

    best_text = text
    best_score = detect_fn(text)
    current = text
    rounds_used = 0

    for r in range(1, int(max_rounds) + 1):
        rounds_used = r
        if r == 1:
            candidate = llm_rewrite(current, model=model)
        else:
            candidate = _rewrite_flagged_chunks(
                current, model, detect_fn,
                flag_threshold=_PARAPHRASE_FLAG_THRESHOLD,
            )
        score = detect_fn(candidate)
        if score < best_score:
            best_score = score
            best_text = candidate
        current = candidate

    return best_text, best_score, rounds_used


# ─── Guided detector-based optimization (Task 7.3, SICO-lite) ────
# KB 03 §1.1: SICO treats the detector score as a loss to minimize against
# prompt variants. This is the "basic version" the plan asks for: a small
# hardcoded set of style instructions, generate → score → keep best → mutate
# (re-apply the best variant with a rotated prompt seed). Deterministic-ish.

GUIDED_STYLE_PROMPTS: list[str] = [
    "Rewrite this text to sound like a natural human blog writer: contractions, "
    "mixed sentence lengths, concrete examples instead of abstractions.",
    "Rewrite this text in an informal, conversational register with varied "
    "rhythm and occasional rhetorical questions.",
    "Rewrite this text with deliberate imperfection: vary sentence length "
    "wildly, avoid formulaic transitions, drop a placeholder detail.",
]


def guided_optimize(
    text: str,
    model: str = "qwen3:1.7b",
    score_fn: Callable[[str], float] | None = None,
    iterations: int = 5,
) -> tuple[str, float, int]:
    """SICO-lite detector-guided prompt optimization over style instructions.

    For *iterations* rounds, generate a variant from the current best text under
    a (cycling) style prompt, score it with ``score_fn`` (injected; lower =
    more human-like), keep the best, and mutate by re-applying on the newest
    best with a per-round seed suffix. Returns ``(best_text, best_score, used)``.

    ``score_fn`` is injected so this never reaches for an external detector."""
    if score_fn is None:
        score_fn = lambda _t: 1.0
    if iterations is None or iterations < 1:
        iterations = 1

    best_text = text
    best_score = score_fn(text)
    used = 0
    for i in range(int(iterations)):
        used = i + 1
        variant = GUIDED_STYLE_PROMPTS[i % len(GUIDED_STYLE_PROMPTS)]
        prompt = (
            f"{variant} Variant #{i}; produce a genuinely different rewrite "
            "than the previous one.\n\nText:\n" + best_text
        )
        try:
            candidate = _ollama_generate(prompt, model=model, timeout=120)
        except Exception:
            continue
        if not candidate.strip():
            continue
        score = score_fn(candidate)
        if score < best_score:
            best_score = score
            best_text = candidate
    return best_text, best_score, used
