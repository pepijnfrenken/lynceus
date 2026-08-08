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
    ("add_transitions", add_transitions),
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


def _ollama_generate(prompt: str, model: str = "qwen3:1.7b", timeout: int = 60) -> str:
    """Call Ollama API with thinking disabled for fast inference."""
    import urllib.request
    import json as _json
    
    payload = _json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.7},
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
