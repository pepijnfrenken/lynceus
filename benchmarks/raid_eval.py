#!/usr/bin/env python3
"""RAID robustness eval for lynceus detectors.

Loads the RAID test split from ``benchmarks/data/raid/`` (``*.jsonl`` files
with human_text/ai_text rows), runs each detector plus the mean-prob ensemble,
and reports per-detector AUROC / TPR@5%FPR / macro-F1 plus the per-layer metric
report (Task E).

WARNING: TMR and Desklib were TRAINED on RAID, so their RAID numbers are a
reference / wiring sanity check, NOT a holdout (see KB doc 25). Use M4 for the
honest OOD generalization number.
RAID is multi-GB and this harness NEVER auto-downloads it: ``--fetch`` only
prints the manual download steps; absent data exits fast with code 2.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks import _common as C

NAME = "raid1"
PAT = "*.jsonl"
REPO = "https://huggingface.co/datasets/liamdugan/raid"


def run_harness(texts, labels, detector_specs, batch_size=16):
    """Pure harness body: score detectors and return the full report.

    *detector_specs* is a list of ``(display_name, loaded_detector)``. Passing
    in already-loaded detectors keeps model download out of the smoke-test path.
    """
    det_results = C.run_detectors(detector_specs, texts, batch_size=batch_size)
    layers = C.compute_layer_metrics(texts, labels, det_results)
    return {
        "name": NAME,
        "meta": {
            "mode": "raid_test_split",
            "source": REPO,
            "n": layers["n"],
            "n_human": layers["n_human"],
            "n_ai": layers["n_ai"],
            "note": "TMR & Desklib trained on RAID: reference numbers, not "
                    "holdout (KB doc 25).",
        },
        "layers": layers,
        "detectors": layers["classifier"],
    }


def main(argv=None):
    parser = C.add_eval_args(
        "benchmarks/raid_eval.py",
        "RAID (reference) robustness eval for lynceus detectors.",
        default_data_dir="benchmarks/data/raid",
        out_name="raid1",
    )
    args = parser.parse_args(argv)
    C.handle_fetch(args, NAME, REPO)

    try:
        samples = C.load_labeled_texts(args.data_dir, pattern=PAT, limit=args.limit)
    except C.DataError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    det_names = [n.strip() for n in args.detectors.split(",") if n.strip()]
    print(f"[raid_eval] loading detectors {det_names} on {args.device} "
          f"(first load downloads weights)...", file=sys.stderr)
    try:
        specs = C.load_detector_specs(det_names, device=args.device)
    except Exception as e:  # noqa: BLE001 - surface any load failure clearly
        print(f"Error loading detectors: {e}", file=sys.stderr)
        return 2

    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]
    summary = run_harness(texts, labels, specs, batch_size=args.batch_size)
    summary["args"] = vars(args)

    C.write_json(args.out, summary)
    C.print_tabular("RAID1 per-layer AUROC", summary["layers"])
    print(f"[raid1] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
