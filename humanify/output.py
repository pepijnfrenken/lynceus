"""Output formatters: rich terminal and JSON."""

import json
from collections import Counter

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .analysis import generate_advice


def _comparison_table(detector_results: list[dict], ensemble_stats: dict) -> Text:
    """Render a model comparison table."""
    t = Text()
    t.append(" Model Comparison\n", style="bold")
    t.append("\n")
    for r in detector_results:
        s = r["stats"]
        t.append(f'  {r["name"]:<15s}', style="bold")
        t.append(f' {s["mean"] * 100:.0f}% AI  │  burstiness: {s["std"]:.2f}\n')
    t.append("  " + "─" * 42 + "\n", style="dim")
    t.append(f'  {"ensemble":<15s}', style="bold cyan")
    t.append(f' {ensemble_stats["mean"] * 100:.0f}% AI  │  burstiness: {ensemble_stats["std"]:.2f}\n')
    t.append("\n")
    # Agreement analysis
    if len(detector_results) >= 2:
        agree_high = 0
        agree_low = 0
        disagree = 0
        n = len(detector_results[0]["probs"])
        for i in range(n):
            probs_i = [r["probs"][i] for r in detector_results]
            all_high = all(p >= 0.5 for p in probs_i)
            all_low = all(p < 0.5 for p in probs_i)
            if all_high:
                agree_high += 1
            elif all_low:
                agree_low += 1
            else:
                disagree += 1
        t.append(f"  Agreement: {agree_high} flagged by all, {agree_low} cleared by all, "
                 f"{disagree} disagree\n", style="dim")
    return t


def output_json(sentences: list[str], probs: list[float], markers: list[list[str]],
                stats: dict, threshold: float, model_name: str,
                detector_results: list[dict] | None = None,
                llm_advice: str | None = None) -> None:
    """Print machine-readable JSON."""
    tips, focus, all_marks = generate_advice(sentences, probs, markers, stats, threshold)
    top = [{"word": w, "count": c} for w, c in Counter(all_marks).most_common(5)]
    focus_data = [{"text": s, "score": round(p, 4), "markers": m} for s, p, m in focus]

    result = {
        "model": model_name,
        "sentences": [
            {"text": s, "ai_probability": round(p, 4), "markers": m}
            for s, p, m in zip(sentences, probs, markers)
        ],
        "overall_score": round(stats["mean"], 4),
        "burstiness": round(stats["std"], 4),
        "stats": {k: round(v, 4) for k, v in stats.items()},
        "sentence_count": len(sentences),
        "advice": {
            "mode": "llm" if llm_advice else "stats",
            "tips": tips,
            "top_markers": top,
            "focus": focus_data,
            **({"text": llm_advice} if llm_advice else {}),
        },
        "disclaimer": "This tool detects statistical similarity to modern AI-generated text. "
                       "It is NOT definitive. Known false positives: formal prose, academic writing, "
                       "legal text, non-native English, and pre-1900 literature. Use as one signal among many.",
    }

    if detector_results:
        result["detectors"] = {}
        for r in detector_results:
            result["detectors"][r["name"]] = {
                "model_id": r["model_id"],
                "overall_score": round(r["stats"]["mean"], 4),
                "burstiness": round(r["stats"]["std"], 4),
                "sentences": [
                    {"ai_probability": round(p, 4)}
                    for p in r["probs"]
                ],
            }

    print(json.dumps(result, indent=2))


def output_rich(sentences: list[str], probs: list[float], markers: list[list[str]],
                stats: dict, threshold: float, model_name: str,
                detector_results: list[dict] | None = None,
                llm_advice: str | None = None) -> None:
    """Print color-coded rich output with advice."""
    console = Console()
    body = Text()

    # Multi-detector comparison table
    if detector_results:
        body.append_text(_comparison_table(detector_results, stats))
        body.append(" " + "\u2500" * 44 + "\n\n", style="dim")

    # Sentence-by-sentence analysis
    for idx, (sent, prob, marks) in enumerate(zip(sentences, probs, markers)):
        color = "green" if prob < 0.3 else ("yellow" if prob < 0.7 else "red")
        icon = "\u2713" if prob < 0.3 else ("\u26a0" if prob < 0.7 else "\u2717")
        display = sent if len(sent) <= 80 else sent[:77] + "..."
        body.append(f" {icon} ", style=f"bold {color}")
        body.append(display)
        body.append(f"  \u2502 {prob * 100:.0f}% AI", style=f"bold {color}")
        # Show per-detector scores for ensemble
        if detector_results:
            body.append("  \u2502 ", style="dim")
            for r in detector_results:
                p = r["probs"][idx]
                body.append(f'{r["name"]}: {p * 100:.0f}% ', style="dim")
        body.append("\n")
        if marks and prob >= threshold:
            body.append(f'   \u2514\u2500 markers: {", ".join(f"{m!r}" for m in marks)}\n', style="dim red")

    # Stats summary
    body.append("\n")
    body.append(" " + "\u2500" * 44 + "\n", style="dim")
    overall_color = "green" if stats["mean"] < 0.3 else ("yellow" if stats["mean"] < 0.7 else "red")
    body.append("  Overall:    ", style="bold")
    body.append(f'{stats["mean"] * 100:.0f}% AI probability\n', style=f"bold {overall_color}")
    body.append("  Burstiness: ", style="bold")
    body.append(f'{stats["std"]:.2f} (std dev)\n')
    body.append("  Sentences:  ", style="bold")
    body.append(f"{len(sentences)} analyzed\n")
    body.append("  Score range:", style="bold")
    body.append(f' {stats["min"] * 100:.0f}% \u2013 {stats["max"] * 100:.0f}%\n')
    body.append("  Model:      ", style="bold")
    body.append(f"{model_name}\n")

    # Writing advice section
    tips, focus, _ = generate_advice(sentences, probs, markers, stats, threshold)
    body.append("\n")
    body.append(" " + "\u2500" * 44 + "\n", style="dim")
    if llm_advice:
        body.append(" Writing Advice (AI-generated)\n", style="bold cyan")
    else:
        body.append(" Writing Advice\n", style="bold cyan")
    body.append("\n")

    if llm_advice:
        # Render LLM advice as wrapped text
        for line in llm_advice.split("\n"):
            body.append(f" {line}\n")
    else:
        for tip in tips:
            body.append(f" {tip}\n", style="yellow" if tip.startswith("\u26a0") else "green")

    if focus:
        body.append("\n")
        body.append(" Focus on these sentences:\n", style="bold")
        for sent, prob, marks in focus:
            display = sent if len(sent) <= 70 else sent[:67] + "..."
            body.append(f' \u2717 "{display}"\n', style="red")
            body.append(f'   markers: {", ".join(marks)}\n', style="dim red")

    # Disclaimer
    body.append("\n")
    body.append(" " + "\u2500" * 44 + "\n", style="dim")
    body.append(" \u26a0  Disclaimer\n", style="bold yellow")
    body.append(" Detects similarity to modern AI text, NOT authorship.\n", style="dim")
    body.append(" Known false positives: formal prose, academic writing,\n", style="dim")
    body.append(" legal text, non-native English, pre-1900 literature.\n", style="dim")
    body.append(" Use as one signal among many.\n", style="dim")
    console.print(Panel(body, title="[bold]humanify[/bold]", border_style="blue", expand=False))
