"""LLM-based writing advice generator.

Uses a causal LM to generate conversational, actionable writing advice
based on the statistical analysis results from the detection pipeline.
"""

from collections import Counter
import re
from typing import Any

import torch


def _build_prompt(
    sentences: list[str],
    probs: list[float],
    markers: list[list[str]],
    stats: dict[str, float],
    threshold: float,
) -> str:
    """Format analysis results into a structured prompt for the LLM."""
    from .analysis import _sentence_openers

    overall = stats["mean"] * 100
    burstiness = stats["std"]
    n = len(sentences)
    high_ai_count = sum(1 for p in probs if p >= threshold)

    # Collect markers with counts
    all_marks = [m for marks in markers for m in marks]
    mark_str = (
        ", ".join(f"{w} ({c}x)" for w, c in Counter(all_marks).most_common(5))
        or "none"
    )

    # Repetitive openers
    openers = _sentence_openers(sentences)
    opener_str = ", ".join(f'"{w}" ({c}x)' for w, c in openers[:3] if c >= 2) or "none"

    # Long sentences
    long_count = sum(1 for s in sentences if len(s.split()) > 30)

    # Top AI-scored sentences (up to 3)
    ranked = sorted(enumerate(sentences), key=lambda x: -probs[x[0]])
    top_lines = []
    for idx, sent in ranked[:3]:
        if probs[idx] >= threshold:
            short = sent if len(sent) <= 80 else sent[:77] + "..."
            top_lines.append(f'  "{short}" ({probs[idx] * 100:.0f}% AI)')
    top_str = "\n".join(top_lines) or "none above threshold"

    return (
        f"""You are a writing coach. Based on this statistical analysis of text,"""
        f""" give 3-5 specific, actionable tips to make the writing sound more"""
        f""" natural. Focus on structural changes, not word-by-word swaps.

Analysis results:
- Overall AI probability: {overall:.0f}%
- Burstiness (sentence variety): {burstiness:.2f} (low = uniform, high = varied)
- Total sentences: {n}
- Sentences above threshold ({threshold * 100:.0f}%): {high_ai_count}
- AI-typical words found: {mark_str}
- Repetitive sentence starters: {opener_str}
- Long sentences (>30 words): {long_count}

Highest AI-scored sentences:
{top_str}

Give concise, direct advice. Be specific to this text."""
    )


def _format_generation_prompt(tokenizer: Any, prompt: str) -> str:
    """Apply a chat template when available, disabling model reasoning if supported."""
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return prompt

    messages = [{"role": "user", "content": prompt}]
    try:
        return apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
    except TypeError:
        return apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def _pad_token_id(tokenizer: Any) -> int:
    """Return pad token id without treating token id 0 as missing."""
    return (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )


def _strip_think_blocks(text: str) -> str:
    """Remove closed or truncated model reasoning blocks."""
    return re.sub(
        r"<think>.*?(?:</think>|$)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )


def generate_llm_advice(
    model: Any,
    tokenizer: Any,
    device: str,
    sentences: list[str],
    probs: list[float],
    markers: list[list[str]],
    stats: dict[str, float],
    threshold: float,
) -> str:
    """Generate writing advice using a causal LM.

    Args:
        model: Loaded causal LM (AutoModelForCausalLM).
        tokenizer: Corresponding tokenizer.
        device: torch device string.
        sentences, probs, markers, stats, threshold: Analysis results.

    Returns:
        Generated advice as a plain string.
    """
    prompt = _build_prompt(sentences, probs, markers, stats, threshold)

    formatted_prompt = _format_generation_prompt(tokenizer, prompt)

    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True,
                       max_length=1024, return_token_type_ids=False).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=_pad_token_id(tokenizer),
        )

    # Decode only the generated tokens (not the prompt)
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Safety net: strip closed or truncated <think> blocks if the model emits them.
    text = _strip_think_blocks(text)

    return text


def _build_quality_prompt(metrics: dict) -> str:
    """Format quality metrics into a structured prompt for the LLM."""
    norm = metrics.get("normalized", {})
    raw = metrics.get("raw", {})

    def _pct(key: str) -> str:
        val = norm.get(key)
        if val is None:
            return "N/A"
        return f"{val * 100:.0f}%"

    def _raw_val(path: str) -> str:
        keys = path.split(".")
        d = raw
        for k in keys:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        if isinstance(d, float):
            return f"{d * 100:.0f}%" if d <= 1 else f"{d:.2f}"
        return str(d) if d else "N/A"

    lines = [
        "You are a writing coach. Based on these essay quality metrics,",
        "identify 2\u20133 specific strengths and 2\u20133 concrete weaknesses.",
        "For each weakness, explain exactly what to change and why.",
        "Be concise, direct, and specific to the numbers below.\n",
        "Quality Metrics:",
        f"- Grammar (active voice, weak modifiers, run-ons): {_pct('grammar_heuristics')}",
        f"  - Passive voice: {_raw_val('grammar_heuristics.passive_ratio')}",
        f"  - Weak modifiers: {_raw_val('grammar_heuristics.weak_modifier_ratio')}",
        f"  - Run-on risk: {_raw_val('grammar_heuristics.runon_ratio')}",
        f"- Specificity (concrete details vs vague/abstract): {_pct('specificity')}",
        f"  - Vague words: {_raw_val('specificity.vague_word_ratio')}",
        f"  - Abstract nouns: {_raw_val('specificity.abstract_noun_ratio')}",
        f"  - Concrete details: {_raw_val('specificity.concrete_detail_ratio')}",
        f"- Evidence (citations, examples, data, experience): {_pct('evidence')}",
        f"  - Citations: {_raw_val('evidence.citation_density')}",
        f"  - Examples: {_raw_val('evidence.example_density')}",
        f"  - Data signals: {_raw_val('evidence.data_density')}",
        f"  - Experience: {_raw_val('evidence.experience_density')}",
        f"- Readability: {_pct('readability')}",
        f"- Lexical diversity: {_pct('lexical_diversity')}",
        f"- Sentence variety: {_pct('sentence_variety')}",
        f"- Repetition: {_pct('repetition')}",
        f"- Coherence: {_pct('coherence_rubric')}",
        f"- Style consistency: {_pct('style_consistency')}",
        "\nGive concise, direct advice. Be specific to this text.",
    ]
    return "\n".join(lines)


def generate_quality_llm_advice(
    model: Any,
    tokenizer: Any,
    device: str,
    metrics: dict,
) -> str:
    """Generate quality-coach advice using a causal LM.

    Args:
        model: Loaded causal LM (AutoModelForCausalLM).
        tokenizer: Corresponding tokenizer.
        device: torch device string.
        metrics: Dict returned by ``compute_writing_metrics()``.

    Returns:
        Generated advice as a plain string.
    """
    prompt = _build_quality_prompt(metrics)
    formatted_prompt = _format_generation_prompt(tokenizer, prompt)

    inputs = tokenizer(
        formatted_prompt, return_tensors="pt", truncation=True,
        max_length=1024, return_token_type_ids=False,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=False,
            pad_token_id=_pad_token_id(tokenizer),
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    text = _strip_think_blocks(text)
    return text
