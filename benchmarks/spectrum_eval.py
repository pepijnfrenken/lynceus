#!/usr/bin/env python3
"""Cross-detector benchmark for humanify — full-stack comparison on a curated
spectrum: human → light AI edit → heavy AI edit → fully AI-generated.

Uses the in-repo benchmarks._common metrics (AUROC, TPR@5%FPR, macro-F1) and
the humanify detector registry. EditLens is treated as its own axis (edit
magnitude 0-1, 4 buckets) in addition to the binary AI-probability framing.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks import _common as C  # noqa: E402
from humanify.analysis import split_sentences  # noqa: E402

# ---------------------------------------------------------------------------
# Curated spectrum: 4 classes x N samples. Hand-written, real human voice.
# ---------------------------------------------------------------------------

HUMAN = [
    "I've run the kitchen in a small cafe for eleven years now, and I still feel a small thrill every morning when the first batch of scones comes out of the oven. The regulars wander in before the street is even awake. My hands know the dough machine better than they know my own knuckles.",
    "We bought the old house on a Monday because the porch was sagging and no one else wanted it. Three winters later the porch is level, the rafters smell of fresh oak, and my son claims the attic as his fort. It took longer than we promised and cost more, too.",
    "The bus to the docks runs every thirty minutes on weekdays, and the neighbours have the timetable wired into their heads. You can set your watch by the morning driver, Marta, who never raises her voice yet knows everybody waiting at the stop by name.",
    "I notice the gap between how people talk about saving money and what they actually do with it. My grandmother kept a jar of buttons by the sink for exactly this reason, she would say. It shaded everything else.",
    "Rain came down hard on Tuesday, the kind that turns the gutter into a small river. I stood at the window with my coffee and watched the street empty out. The dog refused to go out until it stopped.",
    "The last ferry leaves at ten, and the ticket man knows me by now. He waves me through without checking, which is either trust or indifference. Either way it beats the city commute.",
]

# Light AI edit: grammar + fluency polish of the same ideas (contraction
# expansion, smoother connectors, no new content).
LIGHT_EDIT = [
    "I have run the kitchen in a small cafe for eleven years now, and I still feel a small thrill every morning when the first batch of scones comes out of the oven. The regulars wander in before the street is even awake, and my hands know the dough machine better than they know my own knuckles.",
    "We bought the old house on a Monday because the porch was sagging and no one else wanted it. Three winters later, the porch is level, the rafters smell of fresh oak, and my son claims the attic as his fort. It took longer than we promised and cost more, too.",
    "The bus to the docks runs every thirty minutes on weekdays, and the neighbours have the timetable wired into their heads. You can set your watch by the morning driver, Marta, who never raises her voice yet knows everybody waiting at the stop by name.",
    "I notice the gap between how people talk about saving money and what they actually do with it. My grandmother kept a jar of buttons by the sink for exactly this reason, she would say, and it shaded everything else.",
    "Rain came down hard on Tuesday, the kind that turns the gutter into a small river. I stood at the window with my coffee and watched the street empty out, and the dog refused to go out until it had stopped.",
    "The last ferry leaves at ten, and the ticket man knows me by now. He waves me through without checking, which is either trust or indifference, but either way it beats the city commute.",
]

# Heavy AI edit: the ideas are kept but the text is fully rewritten into
# polished, more descriptive prose (the "make it more descriptive" prompt).
HEAVY_EDIT = [
    "For over a decade, I have been the quiet heartbeat of a small cafe kitchen, and every morning still carries a spark of joy as the first batch of scones emerges, golden and fragrant, from the oven. The regulars drift in before the city fully awakens, and my hands navigate the dough machine with an intimacy that surpasses even my own self-knowledge.",
    "Our story with the old house began on a Monday, drawn to its sagging porch and the fact that nobody else wanted it. Three winters later, the porch stands level, the rafters carry the scent of fresh oak, and my son has claimed the attic as his cherished fortress. The journey took longer and cost more than we ever promised.",
    "The bus to the docks adheres to a precise thirty-minute schedule on weekdays, a rhythm the neighbours have internalised completely. One could set a watch by the morning driver, Marta, a woman of quiet authority who knows every waiting passenger by name.",
    "There exists a profound disconnect between the way people speak about financial prudence and the choices they actually make. My grandmother kept a jar of buttons beside the sink, a silent testament to this very principle, and it shaped my perspective on everything.",
    "Tuesday brought a torrential downpour, transforming the gutter into a small, rushing river. Coffee in hand, I watched from the window as the streets gradually emptied, while the dog steadfastly refused to venture outside until the rain had fully ceased.",
    "The final ferry departs at ten o'clock, and the ticket agent, who knows me well by now, waves me through without a second glance. Whether this reflects genuine trust or simple indifference remains unclear, but either way, it is preferable to the city commute.",
]

# Fully AI-generated: coherent, generic prose in the same register, no human source.
AI_GEN = [
    "In today's fast-paced world, the importance of maintaining a healthy work-life balance cannot be overstated. Research has consistently shown that individuals who prioritise personal well-being alongside professional ambitions tend to experience greater long-term satisfaction and productivity. It is therefore essential to cultivate habits that support both dimensions of life.",
    "The rapid advancement of artificial intelligence has fundamentally transformed the landscape of modern industry. From automated customer service to sophisticated data analysis, AI technologies are reshaping how businesses operate and compete. As these systems continue to evolve, organisations must adapt their strategies to remain relevant in an increasingly digital economy.",
    "In this paper, we propose a novel framework that leverages state-of-the-art techniques to achieve unprecedented results in the field of natural language processing. Our approach introduces an innovative architecture that combines attention mechanisms with hierarchical representations, enabling superior performance across multiple benchmarks. Extensive experiments demonstrate the effectiveness of our method.",
    "As we delve into the intricate tapestry of human experience, it becomes evident that the journey of self-discovery is a nuanced odyssey, weaving together threads of resilience, growth, and transformation. Each chapter of this voyage presents unique challenges that ultimately contribute to the rich narrative of our lives. Embracing this complexity allows us to appreciate the profound beauty inherent in our shared humanity.",
    "The intersection of technology and ethics presents both unprecedented opportunities and significant challenges for society. While innovation promises to solve pressing global issues, it also raises critical questions about privacy, equity, and accountability. Policymakers, researchers, and industry leaders must collaborate to ensure that technological progress serves the broader interests of humanity.",
    "Effective communication is the cornerstone of successful collaboration in any professional environment. By fostering an atmosphere of openness and mutual respect, teams can unlock their full creative potential and navigate complex challenges with confidence. It is through this collective effort that meaningful innovation truly flourishes.",
]

SAMPLES = []
for t in HUMAN:
    SAMPLES.append((t, 0.0, "human"))
for t in LIGHT_EDIT:
    SAMPLES.append((t, 1.0, "light_edit"))
for t in HEAVY_EDIT:
    SAMPLES.append((t, 1.0, "heavy_edit"))
for t in AI_GEN:
    SAMPLES.append((t, 1.0, "ai_gen"))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detectors",
                        default="tmr,desklib,fast-detect,binoculars,editlens,radar",
                        help="comma-separated detector registry names")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default="benchmarks/spectrum_bench.json")
    args = parser.parse_args()

    texts = [s[0] for s in SAMPLES]
    labels = [s[1] for s in SAMPLES]
    classes = [s[2] for s in SAMPLES]

    det_names = [n.strip() for n in args.detectors.split(",") if n.strip()]
    print(f"[spectrum] loading {det_names} on {args.device}...", file=sys.stderr)
    specs = C.load_detector_specs(det_names, device=args.device)

    # Score everything
    results = {}
    per_class = {}
    for name, det in specs:
        t0 = time.time()
        scores = det.score(texts, batch_size=args.batch_size)
        dt = time.time() - t0
        results[name] = [float(s) for s in scores]
        per_class[name] = {}
        for cls in ("human", "light_edit", "heavy_edit", "ai_gen"):
            idxs = [i for i, c in enumerate(classes) if c == cls]
            per_class[name][cls] = {
                "mean": round(sum(results[name][i] for i in idxs) / len(idxs), 4),
                "scores": [round(results[name][i], 3) for i in idxs],
            }
        print(f"[spectrum] {name}: {dt:.1f}s for {len(texts)} texts", file=sys.stderr)

    # Metrics: binary framing (human=0, everything else=1)
    metrics = {name: C.sample_metrics(results[name], labels)
               for name in results}
    # EditLens special: also report the 4-bucket edit-magnitude readout
    editlens_extra = {}
    if "editlens" in results:
        editlens_extra["editlens"] = {
            "buckets": ["human", "light_edit", "heavy_edit", "ai_gen"],
            "per_class_mean": per_class["editlens"],
        }

    summary = {
        "name": "spectrum_cross_detector",
        "meta": {
            "n": len(texts),
            "classes": ["human", "light_edit", "heavy_edit", "ai_gen"],
            "n_per_class": {c: classes.count(c) for c in
                            ("human", "light_edit", "heavy_edit", "ai_gen")},
            "note": ("Curated spectrum: human voice originals, light AI polish, "
                     "heavy AI rewrite, and fully AI-generated. EditLens scores "
                     "edit magnitude 0-1 (4 buckets); others score AI probability."),
        },
        "detectors": metrics,
        "per_class_mean": per_class,
        "editlens_extra": editlens_extra,
        "ensemble": C.sample_metrics(C.ensemble_scores(results), labels),
    }
    C.write_json(args.out, summary)

    # Pretty table
    print("\n=== SPECTRUM BENCHMARK (AUROC / TPR@5%FPR / macro-F1) ===")
    print(f"{'detector':<16}{'AUROC':>9}{'TPR@5%':>9}{'mF1':>8}   per-class mean")
    for name in results:
        m = metrics[name]
        pc = per_class[name]
        row = (f"{name:<16}{m['auroc']:>9.4f}{m['tpr_at_5_%_fpr']:>9.4f}"
               f"{m['macro_f1']:>8.4f}   "
               f"h={pc['human']['mean']:.2f} l={pc['light_edit']['mean']:.2f} "
               f"hv={pc['heavy_edit']['mean']:.2f} ai={pc['ai_gen']['mean']:.2f}")
        print(row)
    em = summary["ensemble"]
    print(f"{'ensemble':<16}{em['auroc']:>9.4f}{em['tpr_at_5_%_fpr']:>9.4f}"
          f"{em['macro_f1']:>8.4f}")
    print(f"\n[wrote {args.out}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
