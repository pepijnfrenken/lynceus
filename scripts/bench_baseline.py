#!/usr/bin/env python3
"""Pin a baseline before changing any detector/tell code (Plan Phase 0 Task 0.2, Task A).

Runs the 4 detectors (tmr, desklib, fast-detect, binoculars -- the latter kept
but labelled ``binoculars(degen)`` because its same-model pair is plain
perplexity, KB 06) over a fixed, embedded short-text sample (--n-each AI and
--n-each human texts) and writes a JSON summary to
``benchmarks/baseline_<date>.json`` with per-detector mean/std plus the overall
ensemble mean/std.

Detectors download their weights on first load, so a cold run pulls a few GB.
Usage:
    cd /home/pino/humanify && .venv/bin/python scripts/bench_baseline.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks import _common as C

# ---------------------------------------------------------------------------
# Fixed embedded short-text sample (20 human, 20 AI)
# ---------------------------------------------------------------------------
_HUMAN = [
    "The creek behind the barn runs fast after the spring thaw, and the kids "
    "have worn a path down to the water by the time the lilacs bloom. My "
    "father taught me to read the water, where the eddies hide the big ones. "
    "We still do not need a sign to tell visitors which way is open.",
    "I keep a box of my grandmother's letters in the hall closet. Reading "
    "them is like hearing her talk again, the way she settled a whole room "
    "of worries into one calm paragraph. I save most of the grocery lists "
    "she left tucked inside.",
    "The blue-route bus drops me two blocks from the market, and I have "
    "learned to carry my own bags because the paper ones never survive the "
    "walk up the hill. The vendor with the good tomatoes knows me by name, "
    "which feels like a small win that is coded nowhere.",
    "We painted the garden shed a terrible green two summers ago and nobody "
    "admits to having chosen it. Every so often someone says another coat "
    "might improve it, but no one can be bothered to settle it. The shed "
    "stays green and the joke gets longer.",
    "The platform clock has been wrong for years, and the regulars subtract "
    "four minutes from it so they are never late. It broke the night the "
    "service was cancelled, nobody fixed it, and nobody now minds. It is "
    "simply furniture.",
    "My neighbour waters both rows of chilli plants while I am away and "
    "returns the hose with a note about the forecast in the corner. The "
    "whole street keeps an unspoken calendar for whose turn the fence is, "
    "and we hold to it.",
    "I folded the map years ago and it sits in the glovebox bent at the same "
    "creases, though I know the way better from memory than from paper. The "
    "edges have frazzled into the shape of the coast. Driving it always "
    "brings a tour out of nothing.",
    "My aunt sends holiday cards with the year always wrong, and we have "
    "decided it is on purpose. The postmark is the only true date and we "
    "judge each card by which belated holiday it announces. Nobody has ever "
    "complained.",
    "The bakery at the end of the street sells yesterday's loaves at half "
    "price at the same cautious hour, and the line keeps its comfortable "
    "order. People compare notes about the rain. The baker knows most faces "
    "and the rest by their orders.",
    "The runners gather at the foot of the hill every good morning and are "
    "gone by six-forty, holding the hill against the fog. The neighbour "
    "keeps their time by the taillights. There is a cold regularity that "
    "needs no explanation.",
    "I taught myself to mend buttonholes from a booklet my mother never "
    "finished. The stack of half-done mending is honest, and the finished "
    "ones are the rarer ones. It fills a whole evening to do one small "
    "sleeve.",
    "The hardware shop smells of dust and oil, and the good and the worn "
    "end touch the same counter. Everyone asks for the same three bolts by "
    "name. The owner leaves a note when a part is in and walks me out to "
    "talk about the roof.",
    "The neighbour's dog has learnt to sit on the mat whenever the biscuits "
    "are likely, which is whenever I am. The two of them have a whole "
    "conversation in a glance across the yards. I think the dog reads the "
    "clock better than I do.",
    "We rearranged the kitchen so the cupboards open the wrong way, and two "
    "months on we still bump into the new shape. Neither of us will admit "
    "to remembering the old one. The change has made the room a daily "
    "surprise.",
    "Sunday lunch is the one meal where nobody mentions the phone, and the "
    "rule has held for years. The kids set the table before I finish, and "
    "we all talk over one another and that somehow is the calm of the week.",
    "I keep a chipped bowl of sea glass on the sill, and every winter a "
    "peculiar compact has to be made with the morning light. The colours "
    "are only good early. It is the only thing in the house that keeps its "
    "own season.",
    "The joiner who fixed our door said to give the hinge a season to "
    "forget the winter before it sets. A year on he was right, though we "
    "had paid him to say so. The door now swings without complaint.",
    "Her knitting is slow but exact, and the corners of her sweaters still "
    "hold after forty winters of use. She gives almost everything away and "
    "keeps only repeats of old favours. Nobody has the heart to throw out "
    "the good ones.",
    "The little bridge at the end of the lane is local because one plank is "
    "always loose and everyone crosses it with a small hop. It has been "
    "that way since memory, and the hop is part of the walk to the bus. You "
    "only notice it when the hop is missing.",
    "We keep a running joke that the cuckoo clock is the only punctual thing "
    "in the house and it is never right. It rings at the wrong hours and "
    "the cat flees when it does. Over the years it has become a second "
    "voice of the place.",
]

_AI = [
    "In conclusion, the implementation of a robust reminder system "
    "significantly enhances daily productivity. It is important to note that "
    "consistency and discipline are pivotal in achieving lasting gains, "
    "notably through streamlined task management and effective delegation. "
    "Furthermore, integrating modern tools fosters a holistic approach.",
    "It goes without saying that effective communication is crucial in any "
    "collaborative endeavour. Moreover, a multidisciplinary perspective "
    "enables teams to navigate complex challenges with greater agility. "
    "Consequently, leveraging diverse expertise will effectively improve "
    "outcomes.",
    "In today's world, the transformative impact of technology cannot be "
    "overstated. As we navigate an increasingly digital landscape, robust "
    "analytics and strategic frameworks become essential. Furthermore, "
    "adopting cutting-edge solutions empowers stakeholders and streamlines "
    "workflows.",
    "First and foremost, achieving optimal results requires a comprehensive "
    "understanding of underlying dynamics. Not only this, but the swift "
    "adoption of innovative practices facilitates seamless transitions. In "
    "conclusion, this represents a significant milestone.",
    "A comprehensive framework is essential to foster innovation and ensure "
    "scalable growth. Leveraging state-of-the-art techniques enables "
    "organisations to navigate multifaceted challenges with resilience. "
    "Moreover, this paradigm shift is a testament to the potential of "
    "strategic thinking.",
    "In summary, unwavering commitment to excellence substantially drives "
    "measureable outcomes. As we navigate the dynamic terrain, "
    "cutting-edge methodologies present a synergistic approach. It is worth "
    "noting that empowered teams embody this groundbreaking endeavour.",
    "The implementation of a robust strategy is paramount. Notably, "
    "collaboration across diverse domains and the prudent deployment of "
    "resources establish a workable framework. Consequently, this holistic "
    "approach facilitates seamless integration and paves the way for "
    "future gains.",
    "Moreover, it is crucial to acknowledge that modern paradigms require a "
    "forward-looking mindset. Leveraging innovative tools and fostering "
    "synergy enable us to navigate uncertainty with confidence. This "
    "comprehensive vision underscores the importance of sustainable "
    "planning.",
    "In today's world, data-driven decisions are essential for success. By "
    "utilizing comprehensive analytics and leveraging robust frameworks, "
    "organisations can ascertain meaningful insights. Furthermore, these "
    "systems improve adaptability and drive unprecedented growth.",
    "A holistic perspective is indispensable for addressing complex "
    "challenges. Employing innovative strategies and adopting a new "
    "paradigm helps optimise resources and streamline operations. "
    "Additionally, this multifaceted approach reveals myriad opportunities.",
    "It is worth noting that fostering a culture of adaptability is pivotal "
    "for sustained gains. Leveraging continuous feedback facilitates "
    "constructive dialogue and builds a robust ecosystem. In conclusion, "
    "this synergy generates lasting impact.",
    "the pursuit of excellence cannot be overstated. Innovative and "
    "inclusive frameworks help address the complexity of modern times; "
    "additionally, agile and comprehensive coordination is vital. Finally, "
    "and importantly, robust governance underpins all of it.",
    "One key recommendation is to adopt a best-practice framework from the "
    "outset. This enables scalability, reduces operational friction, and "
    "ensures that every stakeholder is aligned. Consequently, future "
    "iterations become more resilient and measurable.",
    "his exploratory analysis highlights several dimensions of the issue. "
    "the findings suggest that a multifaceted, evidence-based response is "
    "necessary. It is important to note that meaningful change requires "
    "sustained, collaborative effort across all levels.",
    "In conclusion, this initiative underscores the value of committing to "
    "a comprehensive roadmap. Moreover, it reserves careful consideration "
    "of trade-offs and enables iterative improvement. Moving forward, "
    "stakeholders should prioritise transparent and scalable execution.",
    "It is worth noting that resilience is built over time through "
    "deliberate practice. By leveraging a broad range of perspectives and "
    "refining the approach, teams can navigate unforeseen challenges. "
    "In turn, this fosters a virtuous and continuous cycle of learning.",
    "Moreover, the evidence highlights several key takeaways. First, a "
    "comprehensive framework should be adopted. Second, innovation is "
    "central. Lastly, continuous improvement must be championed at every "
    "level.",
    "It is important to mention the role of structured communication in "
    "driving outcomes. This multifaceted approach enables clarity, "
    "aligns expectations, and facilitates efficient resolution. "
    "Consequently, it underscores the importance of robust ongoing "
    "governance.",
    "In summary, leveraging proven methodologies and championing "
    "adaptability yields robust results. The findings demonstrate that "
    "collaborative, incremental planning continues to be the most effective "
    "road map. Therefore, stakeholder engagement remains a fundamental "
    "pillar of success.",
    "Finally, it is crucial to recognise that modern tools only add value "
    "when paired with clear objectives. By embedding comprehensive review "
    "cycles, teams make consistent, and importantly, measurable progress. "
    "This, in turn, paves the way for sustained and lasting improvement.",
]


def _tile(items: list[str], n: int) -> list[str]:
    out = []
    while len(out) < n:
        out.extend(items[: n - len(out)])
    return out


def build_sample(n_each: int = 20) -> list[str]:
    """Fixed baseline sample: *n_each* human texts then *n_each* AI texts."""
    return _tile(_HUMAN, n_each) + _tile(_AI, n_each)


def run_baseline(detector_specs, n_each: int = 20, batch_size: int = 16) -> dict:
    """Pure baseline body: score the fixed sample and return per-detector stats."""
    texts = build_sample(n_each)
    det_results = C.run_detectors(detector_specs, texts, batch_size=batch_size)

    per_detector = {}
    for name, scores in det_results.items():
        arr = np.asarray(scores)
        per_detector[name] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
        }
    ens = np.asarray(C.ensemble_scores(det_results))
    return {
        "name": "baseline",
        "n_human": n_each,
        "n_ai": n_each,
        "per_detector": per_detector,
        "ensemble": {
            "mean": round(float(ens.mean()), 4),
            "std": round(float(ens.std()), 4),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="scripts/bench_baseline.py",
        description="Pin a pre-change detector baseline (Plan 0.2).",
    )
    parser.add_argument("--detector", default="tmr,desklib,fast-detect,binoculars",
                        help="comma-separated registry detector names")
    parser.add_argument("--n-each", type=int, default=20,
                        help="texts per class (default: 20)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default=C.default_out_path("baseline"))
    args = parser.parse_args(argv)

    det_names = [d.strip() for d in args.detector.split(",") if d.strip()]
    print(f"[baseline] loading detectors {det_names} (weights download on first "
          f"load)...", file=sys.stderr)
    try:
        specs = C.load_detector_specs(det_names, device=args.device)
    except Exception as e:  # noqa: BLE001
        print(f"Error loading detectors: {e}", file=sys.stderr)
        return 2

    summary = run_baseline(specs, n_each=args.n_each, batch_size=args.batch_size)
    summary["detectors"] = [n for n, _ in specs]
    summary["generated"] = C.today()
    C.write_json(args.out, summary)
    print(json.dumps(summary["ensemble"], indent=2))
    print(f"[baseline] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
