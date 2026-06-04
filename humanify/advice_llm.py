"""LLM-based writing advice generator.

Uses a causal LM to generate conversational, actionable writing advice
based on the statistical analysis results from the detection pipeline.
"""

from collections import Counter
import re

import torch


def _build_prompt(sentences, probs, markers, stats, threshold) -> str:
    """Format analysis results into a structured prompt for the LLM."""
    from .analysis import _sentence_openers

    overall = stats["mean"] * 100
    burstiness = stats["std"]
    n = len(sentences)
    high_ai_count = sum(1 for p in probs if p >= threshold)

    # Collect markers with counts
    all_marks = [m for marks in markers for m in marks]
    mark_str = ", ".join(f"{w} ({c}x)" for w, c in Counter(all_marks).most_common(5)) or "none"

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

    return f"""You are a writing coach. Based on this statistical analysis of text, give 3-5 specific, actionable tips to make the writing sound more natural. Focus on structural changes, not word-by-word swaps.

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


def _format_generation_prompt(tokenizer, prompt: str) -> str:
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


def _pad_token_id(tokenizer) -> int:
    """Return pad token id without treating token id 0 as missing."""
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def _strip_think_blocks(text: str) -> str:
    """Remove closed or truncated model reasoning blocks."""
    return re.sub(
        r"<think>.*?(?:</think>|$)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )


def generate_llm_advice(model, tokenizer, device, sentences, probs, markers, stats, threshold) -> str:
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