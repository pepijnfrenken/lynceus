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
    t.append(
        f' {ensemble_stats["mean"] * 100:.0f}% AI  │  '
        f'burstiness: {ensemble_stats["std"]:.2f}\n'
    )
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
        t.append(
            f"  Agreement: {agree_high} flagged by all, "
            f"{agree_low} cleared by all, {disagree} disagree\n",
            style="dim",
        )
    return t


def output_json(sentences: list[str], probs: list[float], markers: list[list[str]],
                stats: dict, threshold: float, model_name: str,
                detector_results: list[dict] | None = None,
                llm_advice: str | None = None,
                structure_audit: dict | None = None,
                artifact_audit: dict | None = None,
                narrative: dict | None = None,
                axes: dict | None = None,
                confidence: float | None = None,
                domain: str | None = None,
                domain_settings: dict | None = None,
                domain_warnings: list | None = None,
                quality_metrics: dict | None = None,
                quality_advice: list[str] | None = None,
                quality_llm_advice: str | None = None,
                sentence_analysis: list[dict] | None = None,
                paragraph_analysis: dict | None = None,
                top_worst: dict | None = None) -> None:
    """Print machine-readable JSON."""
    tips, focus, all_marks = generate_advice(
        sentences, probs, markers, stats, threshold
    )
    top = [{"word": w, "count": c} for w, c in Counter(all_marks).most_common(5)]
    focus_data = [
        {"text": s, "score": round(p, 4), "markers": m}
        for s, p, m in focus
    ]

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
        **({"structure_audit": structure_audit} if structure_audit else {}),
        **({"artifact_audit": artifact_audit} if artifact_audit else {}),
        **({"narrative": narrative} if narrative else {}),
        **({"axes": axes} if axes else {}),
        **({"confidence": confidence} if confidence is not None else {}),
        **({"domain": domain} if domain else {}),
        **({"domain_settings": domain_settings} if domain_settings else {}),
        **({"domain_warnings": domain_warnings} if domain_warnings else {}),
        "disclaimer": (
            "This tool detects statistical similarity to modern AI-generated text. "
            "It is NOT definitive. Known false positives: formal prose, "
            "academic writing, legal text, non-native English, and pre-1900 "
            "literature. Use as one signal among many."
        ),
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

    if quality_metrics is not None:
        result["writing_quality"] = {
            "metrics": quality_metrics,
            "advice": quality_advice or [],
            **({"llm_text": quality_llm_advice} if quality_llm_advice else {}),
        }
        if sentence_analysis is not None:
            result["writing_quality"]["sentence_analysis"] = sentence_analysis
        if paragraph_analysis is not None:
            result["writing_quality"]["paragraph_analysis"] = paragraph_analysis
        if top_worst is not None:
            result["writing_quality"]["top_worst"] = {
                dim: [
                    {"text": s, "score": round(score, 4)}
                    for s, score in items
                ]
                for dim, items in top_worst.items()
            }

    print(json.dumps(result, indent=2))


def output_rich(sentences: list[str], probs: list[float], markers: list[list[str]],
                stats: dict, threshold: float, model_name: str,
                detector_results: list[dict] | None = None,
                llm_advice: str | None = None,
                structure_audit: dict | None = None,
                artifact_audit: dict | None = None,
                narrative: dict | None = None,
                axes: dict | None = None,
                confidence: float | None = None,
                domain: str | None = None,
                domain_settings: dict | None = None,
                domain_warnings: list | None = None,
                quality_metrics: dict | None = None,
                quality_advice: list[str] | None = None,
                quality_llm_advice: str | None = None,
                sentence_analysis: list[dict] | None = None,
                paragraph_analysis: dict | None = None,
                top_worst: dict | None = None) -> None:
    """Print color-coded rich output with advice."""
    console = Console()
    body = Text()

    # Multi-detector comparison table
    if detector_results:
        body.append_text(_comparison_table(detector_results, stats))
        body.append(" " + "\u2500" * 44 + "\n\n", style="dim")

    # Sentence-by-sentence analysis with quality flags
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
            body.append(
                f'   \u2514\u2500 markers: {", ".join(f"{m!r}" for m in marks)}\n',
                style="dim red",
            )
        # Sentence-level quality flags
        if sentence_analysis and idx < len(sentence_analysis):
            qflags = sentence_analysis[idx].get("flags", [])
            qscore = sentence_analysis[idx].get("score", 10.0)
            if qflags or qscore < 10.0:
                flag_str = ", ".join(qflags) if qflags else "clean"
                score_color = "green" if qscore >= 8 else ("yellow" if qscore >= 5 else "red")
                body.append(
                    f"   \u2514\u2500 quality: [{score_color}]{qscore:.1f}/10[/{score_color}] | {flag_str}\n",
                    style="dim cyan",
                )

    # Stats summary
    body.append("\n")
    body.append(" " + "\u2500" * 44 + "\n", style="dim")
    overall_color = (
        "green" if stats["mean"] < 0.3
        else ("yellow" if stats["mean"] < 0.7 else "red")
    )
    body.append("  Overall:    ", style="bold")
    body.append(
        f'{stats["mean"] * 100:.0f}% AI probability\n',
        style=f"bold {overall_color}",
    )
    body.append("  Burstiness: ", style="bold")
    body.append(f'{stats["std"]:.2f} (std dev)\n')
    body.append("  Sentences:  ", style="bold")
    body.append(f"{len(sentences)} analyzed\n")
    body.append("  Score range:", style="bold")
    body.append(f' {stats["min"] * 100:.0f}% \u2013 {stats["max"] * 100:.0f}%\n')
    body.append("  Model:      ", style="bold")
    body.append(f"{model_name}\n")

    # Writing quality section
    if quality_metrics is not None:
        body.append("\n")
        body.append(" " + "\u2500" * 44 + "\n", style="dim")
        body.append(" Writing Quality\n", style="bold cyan")
        norm = quality_metrics.get("normalized", {})
        body.append(f"  Grammar:        {norm.get('grammar_heuristics', 0) * 100:.0f}%\n")
        body.append(f"  Specificity:    {norm.get('specificity', 0) * 100:.0f}%\n")
        body.append(f"  Evidence:       {norm.get('evidence', 0) * 100:.0f}%\n")
        body.append(f"  Readability:    {norm.get('readability', 0) * 100:.0f}%\n")
        body.append(f"  Lexical diversity: {norm.get('lexical_diversity', 0) * 100:.0f}%\n")
        body.append(f"  Coherence:      {norm.get('coherence_rubric', 0) * 100:.0f}%\n")

        # Paragraph analysis
        pa = paragraph_analysis or {}
        if pa:
            body.append("\n")
            body.append(" Paragraph Structure\n", style="bold cyan")
            body.append(f"  Paragraphs:           {pa.get('paragraph_count', 0)}\n")
            body.append(f"  Avg sentences/para:   {pa.get('avg_sentences_per_paragraph', 0):.1f}\n")
            body.append(f"  Topic sentences:        {pa.get('topic_sentence_coverage', 0) * 100:.0f}%\n")
            body.append(f"  Paragraph transitions:  {pa.get('paragraph_transition_coverage', 0) * 100:.0f}%\n")
            body.append(f"  Thesis alignment:       {pa.get('thesis_alignment', 0) * 100:.0f}%\n")

        body.append("\n")
        if quality_llm_advice:
            body.append(" Quality Feedback (AI-generated)\n", style="bold")
            for line in quality_llm_advice.split("\n"):
                body.append(f" {line}\n")
        elif quality_advice:
            for tip in quality_advice:
                body.append(
                    f" {tip}\n",
                    style="yellow" if tip.startswith("\u26a0") else "green",
                )

        # Top-worst sentences
        if top_worst:
            body.append("\n")
            body.append(" Top Sentences to Improve\n", style="bold cyan")
            dim_labels = {
                "grammar": "Grammar",
                "specificity": "Specificity",
                "evidence": "Evidence",
                "readability": "Readability",
            }
            for dim, items in top_worst.items():
                if not items:
                    continue
                label = dim_labels.get(dim, dim)
                body.append(f"  {label}:\n", style="bold")
                for s, score in items[:3]:
                    display = s if len(s) <= 70 else s[:67] + "..."
                    color = "red" if score < 0.3 else ("yellow" if score < 0.6 else "green")
                    body.append(f'    \u2022 "{display}"\n', style=color)

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
            body.append(
                f" {tip}\n",
                style="yellow" if tip.startswith("\u26a0") else "green",
            )

    if focus:
        body.append("\n")
        body.append(" Focus on these sentences:\n", style="bold")
        for sent, prob, marks in focus:
            display = sent if len(sent) <= 70 else sent[:67] + "..."
            body.append(f' \u2717 "{display}"\n', style="red")
            body.append(f'   markers: {", ".join(marks)}\n', style="dim red")

    # Structural audit findings (Task F)
    if structure_audit and structure_audit.get("findings"):
        body.append("\n")
        body.append(" " + "\u2500" * 44 + "\n", style="dim")
        body.append(" Structural Audit\n", style="bold cyan")
        body.append(f'  score: {structure_audit["score"]:.2f}\n', style="dim")
        for f in structure_audit["findings"]:
            sev_style = {
                "low": "green", "medium": "yellow", "high": "red",
            }.get(f["severity"], "yellow")
            body.append(f'  [{f["kb_ref"]}] {f["name"]} '
                        f'({f["severity"]})\n', style=f"bold {sev_style}")
            body.append(f"      {f['evidence']}\n", style="dim")

    # Post-processing artifact audit (Task C, Phase 4.3)
    if artifact_audit and artifact_audit.get("findings"):
        body.append("\n")
        body.append(" " + "\u2500" * 44 + "\n", style="dim")
        body.append(" Post-Processing Artifacts\n", style="bold cyan")
        body.append(
            f'  score: {artifact_audit["score"]:.2f}'
            + (
                f'\n  EditLens recommended: yes'
                if artifact_audit.get("editlens_recommended")
                else ""
            ),
            style="dim",
        )
        body.append("\n")
        for f in artifact_audit["findings"]:
            sev_style = {
                "low": "green", "medium": "yellow", "high": "red",
            }.get(f["severity"], "yellow")
            body.append(
                f'  [{f["kb_ref"]}] {f["name"]} ({f["severity"]})\n',
                style=f"bold {sev_style}",
            )
            body.append(f"      {f['evidence']}\n", style="dim")

    # Per-layer ensemble axes + confidence (Task C / Phase 6.1)
    if axes:
        body.append("\n")
        body.append(" " + "\u2500" * 44 + "\n", style="dim")
        body.append(" Per-Layer Evidence\n", style="bold cyan")
        names = {
            "classifier_axis": "classifier",
            "zero_shot_axis": "zero-shot",
            "lexical_axis": "lexical",
            "structural_axis": "structural",
            "artifact_axis": "artifacts",
        }
        for k, v in axes.items():
            if k in names and v.get("score") is not None:
                body.append(f"  {names[k]:<12s} {v['score'] * 100:5.0f}% AI\n",
                            style="dim")
        body.append(f"  {'overall':<12s} {axes['overall'] * 100:5.0f}% AI\n",
                    style="bold cyan")
        if confidence is not None:
            body.append(f"  Confidence:     {confidence * 100:5.0f}% "
                        f"(agreement spread {axes.get('axis_agreement', 0.0):.3f})\n",
                        style="dim")

    # Narrative checklist (Task B / Phase 5.2)
    if narrative is not None:
        body.append("\n")
        body.append(" " + "\u2500" * 44 + "\n", style="dim")
        body.append(" Narrative Checklist\n", style="bold cyan")
        if not narrative.get("enabled"):
            reason = narrative.get("reason")
            body.append(f"  disabled ({reason}) \u2014 needs >=500 words\n",
                        style="dim")
        else:
            body.append(f"  score: {narrative['score']:.2f}  "
                        f"({len(narrative['flagged'])}/{len(narrative['dimensions'])} "
                        f"flagged)\n", style="dim")
            for d in narrative["flagged"]:
                body.append(f"  - {d['key']}  ({d['direction']})\n", style="yellow")
                if d.get("evidence"):
                    body.append(f"      {d['evidence']}\n", style="dim")

    # Domain calibration (Task D / Phase 6.2)
    if domain:
        body.append("\n")
        body.append(" " + "\u2500" * 44 + "\n", style="dim")
        body.append(f" Domain: {domain}\n", style="bold cyan")
        for w in domain_warnings or []:
            body.append(f"  \u26a0 {w}\n", style="yellow")
        if domain_settings:
            body.append(
                f"  setup: burstiness_w={domain_settings.get('burstiness_weight')}, "
                f"lexical_cap={domain_settings.get('lexical_weight_cap')}, "
                f"sentence_th={domain_settings.get('sentence_threshold')}\n",
                style="dim",
            )

    # Disclaimer
    body.append("\n")
    body.append(" " + "\u2500" * 44 + "\n", style="dim")
    body.append(" \u26a0  Disclaimer\n", style="bold yellow")
    body.append(" Detects similarity to modern AI text, NOT authorship.\n", style="dim")
    body.append(
        " Known false positives: formal prose, academic writing,\n",
        style="dim",
    )
    body.append(" legal text, non-native English, pre-1900 literature.\n", style="dim")
    body.append(" Use as one signal among many.\n", style="dim")
    console.print(
        Panel(body, title="[bold]humanify[/bold]", border_style="blue", expand=False)
    )


def _format_val(val: float | None, precision: int = 2, na: str = "\u2014") -> str:
    """Format a float or None as a short string."""
    if val is None:
        return na
    return f"{val:.{precision}f}"


def render_eval_summary(
    sample_records: list[dict],
    summary: dict,
    model_names: list[str],
    preference_correlation: dict | None = None,
) -> None:
    """Render a rich terminal summary of batch eval results to stderr."""
    console = Console(stderr=True)
    body = Text()

    # --- Header ---
    body.append(" Samples: ", style="bold")
    body.append(f'{summary["sample_count"]} total, ')
    valid = summary.get("valid_count", 0)
    invalid = summary.get("invalid_count", 0)
    body.append(f"{valid} valid", style="green")
    if invalid:
        body.append(f", {invalid} invalid", style="red")
    dup = summary.get("duplicate_id_count", 0)
    if dup:
        body.append(f", {dup} duplicate IDs", style="yellow")
    body.append("\n")

    body.append(" Models:  ", style="bold")
    body.append(", ".join(model_names))
    body.append("\n\n")

    # --- Per-sample table ---
    if sample_records:
        # Manual fixed-width table
        header = f"   {'ID':<20s} {'Comp':>6s} {'Pref':>6s} {'Read':>6s} {'Rep':>6s} {'HuDiag':>6s}"
        sep = f"   {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}"
        body.append(" Per-Sample Scores\n", style="bold")
        body.append(f" {header}\n", style="bold cyan")
        body.append(f" {sep}\n", style="dim")

        for rec in sample_records:
            cs = rec.get("component_scores", {})
            cid = rec["id"][:20]
            comp = rec.get("composite_score")
            comp_s = _format_val(comp)
            pref_s = _format_val(cs.get("preference_rubric"))
            read_s = _format_val(cs.get("readability_style"))
            rep_s  = _format_val(cs.get("repetition_variety"))
            hu_s   = _format_val(cs.get("humanify_diagnostic"))
            row = f"   {cid:<20s} {comp_s:>6s} {pref_s:>6s} {read_s:>6s} {rep_s:>6s} {hu_s:>6s}"
            if comp is not None:
                color = "green" if comp >= 0.70 else ("yellow" if comp >= 0.40 else "red")
                body.append(row + "\n", style=color)
            else:
                body.append(row + "\n")
        body.append("\n")
    # --- Aggregate stats ---
    comps = [r.get("composite_score") for r in sample_records
             if r.get("composite_score") is not None]
    if comps:
        mn = sum(comps) / len(comps)
        variance = sum((x - mn) ** 2 for x in comps) / len(comps)
        sd = variance ** 0.5
        body.append(" Aggregates:\n", style="bold")
        body.append(f"   Composite: {min(comps):.2f}\u2013{max(comps):.2f} ")
        body.append(f"(\u03bc={mn:.2f}, \u03c3={sd:.2f})\n")

    # --- Quality aggregate stats ---
    quality_keys = [
        ("grammar_heuristics", "Grammar"),
        ("specificity", "Specificity"),
        ("evidence", "Evidence"),
        ("readability", "Readability"),
        ("lexical_diversity", "Lexical diversity"),
        ("coherence_rubric", "Coherence"),
    ]
    quality_avgs: dict[str, list[float]] = {}
    for rec in sample_records:
        norm = rec.get("metrics", {}).get("normalized", {})
        for key, _ in quality_keys:
            val = norm.get(key)
            if val is not None:
                quality_avgs.setdefault(key, []).append(val)
    if quality_avgs:
        body.append("\n Quality Averages:\n", style="bold")
        for key, label in quality_keys:
            vals = quality_avgs.get(key, [])
            if vals:
                avg = sum(vals) / len(vals)
                color = "green" if avg >= 0.7 else ("yellow" if avg >= 0.4 else "red")
                body.append(f"   {label:<18s} ", style="bold")
                body.append(f"[{color}]{avg * 100:.0f}%[/{color}]\n")

    # --- Preference correlation ---
    corr = preference_correlation or summary.get("preference_correlation", {})
    pair_count = corr.get("pair_count", 0)
    if pair_count > 0:
        body.append("\n Preference Correlation", style="bold")
        body.append(f" ({pair_count} pairs):\n")
        non_tie = corr.get("non_tie_count", 0)
        tie = corr.get("tie_count", 0)
        body.append(f"   Non-tie: {non_tie}, Ties: {tie}\n")
        agreement = corr.get("agreement")
        if agreement is not None:
            body.append(f"   Agreement: {agreement * 100:.1f}%    ")
        pearson = corr.get("pearson")
        if pearson is not None:
            p_color = "green" if pearson > 0.7 else ("yellow" if pearson > 0.4 else "red")
            body.append(f"Pearson: [{p_color}]{pearson:.3f}[/{p_color}]    ")
        spearman = corr.get("spearman")
        if spearman is not None:
            s_color = "green" if spearman > 0.7 else ("yellow" if spearman > 0.4 else "red")
            body.append(f"Spearman: [{s_color}]{spearman:.3f}[/{s_color}]")
        body.append("\n")

    console.print(
        Panel(body, title="[bold]humanify eval[/bold]", border_style="blue", expand=False)
    )
