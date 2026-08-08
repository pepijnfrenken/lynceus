#!/usr/bin/env python3
"""M4 OOD robustness eval for humanify detectors.

Loads the M4 test split from ``benchmarks/data/m4/`` (HF mirror
``d0rj/SemEval2024-task8``; text + label jsonl rows, 0=human / 1=machine),
runs each detector + the mean-prob ensemble, and reports per-detector
AUROC / TPR@5%FPR / macro-F1 plus the per-layer report (Task E).

M4 is the honest OOD generalization number (unlike RAID, neither TMR nor
Desklib trained on M4). The dataset (hundreds of thousands of rows) is
multi-GB and this harness NEVER auto-downloads it: ``--fetch`` only prints the
manual steps; absent data exits fast with code 2.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks import _common as C

NAME = "m4"
PAT = "*.jsonl"
REPO = "https://huggingface.co/datasets/d0rj/SemEval2024-task8"


def main(argv=None):
    parser = C.add_eval_args(
        "benchmarks/m4_eval.py",
        "M4 (OOD) robustness eval for humanify detectors.",
        default_data_dir="benchmarks/data/m4",
        out_name="m4",
    )
    args = parser.parse_args(argv)
    C.handle_fetch(args, NAME, REPO)

    try:
        samples = C.load_labeled_texts(args.data_dir, pattern=PAT, limit=args.limit)
    except C.DataError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    det_names = [n.strip() for n in args.detectors.split(",") if n.strip()]
    print(f"[m4_eval] loading detectors {det_names} on {args.device} "
          f"(first load downloads weights)...", file=sys.stderr)
    try:
        specs = C.load_detector_specs(det_names, device=args.device)
    except Exception as e:  # noqa: BLE001
        print(f"Error loading detectors: {e}", file=sys.stderr)
        return 2

    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]
    summary = C.run_harness(NAME, texts, labels, specs, args.batch_size)
    summary["meta"]["source"] = REPO
    summary["meta"]["mode"] = "m4_ood"
    summary["meta"]["note"] = ("M4 is out-of-domain for these detectors; this "
                               "is the honest generalization number.")
    summary["args"] = vars(args)
    del summary["args"]["fetch"]

    C.write_json(args.out, summary)
    C.print_tabular("M4 per-layer AUROC", summary["layers"])
    print(f"[m4] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
