#!/usr/bin/env python3
"""H2L (human-polished-by-LLM) robustness eval -- the real threat model.

Per KB doc 07, H2L (human text rewritten/polished by an LLM) is the dominant
real-world failure mode; detection drops sharply there (e.g. FastDetectGPT
91.2% -> 30.8%, Binoculars 93.5% -> 15.1% on ARB, arXiv 2607.29539).

This script produces an H2L number in one of two modes:

1. ARB mode  if ``--arb-dir`` points at real ARB data  -- human text paired
   with an LLM-polished rewrite.
2. Surrogate mode (default) -- synthesises H2L locally: human samples are
   "polished" with a documented stand-in (sentence re-ordering + contraction
   expansion + a formulaic connector). If a local Ollama server at
   http://localhost:11434 is reachable, ``redteam.transforms.llm_rewrite`` is
   used instead; otherwise a deterministic surrogate. The caveat is always
   written verbatim into the output JSON.

The harness NEVER downloads the multi-GB datasets: ``--fetch`` only prints the
manual steps and ARB-absent means surrogate mode, never a hang.
"""

import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks import _common as C
from lynceus.analysis import split_sentences

NAME = "h2l"
OLLAMA_URL = "http://localhost:11434"

_DEFAULT_HUMAN = (
    "I've run the kitchen in a small cafe for eleven years now, and I still "
    "feel a small thrill every morning when the first batch of scones comes "
    "out of the oven. The regulars wander in before the street is even awake. "
    "My hands know the dough machine better than they know my own knuckles.",
    "We bought the old house on a Monday because the porch was sagging and no "
    "one else wanted it. Three winters later the porch is level, the rafters "
    "smell of fresh oak, and my son claims the attic as his fort. It took "
    "longer than we promised and cost more, too.",
    "The bus to the docks runs every thirty minutes on weekdays, and the "
    "neighbours have the timetable wired into their heads. You can set your "
    "watch by the morning driver, Marta, who never raises her voice yet knows "
    "everybody waiting at the stop by name.",
    "I notice the gap between how people talk about saving money and what "
    "they actually do with it. My grandmother kept a jar of buttons by the "
    "sink for exactly this reason, she would say. It shaded everything else.",
)


def _tidy(s: str) -> str:
    return " ".join(s.split())


def _formalize(sentence: str) -> str:
    """Light deterministic 'AI-polish' stand-in: expand contractions."""
    pairs = {
        "don't": "do not", "won't": "will not", "can't": "cannot",
        "i'm": "i am", "it's": "it is", "that's": "that is",
        "there's": "there is", "i've": "i have", "you're": "you are",
    }
    out = sentence
    for a, b in pairs.items():
        out = out.replace(a, b).replace(a.capitalize(), b.capitalize())
    return out


def surrogate_polish(text: str, seed: int = 42) -> str:
    """Deterministic, documented 'LLM-polish' stand-in.

    Re-orders a pair of sentences, expands contractions (formal drift) and
    inserts one formulaic connective. That shift in fluency/register is the
    signature of a light LLM polish. No model is involved.
    """
    rnd = random.Random(seed)
    sentences = [s for s in split_sentences(text) if s.strip()]
    if len(sentences) < 2:
        return _tidy(_formalize(text))
    order = list(range(len(sentences)))
    if len(order) >= 3:
        i, j = rnd.sample(order, 2)
        order[i], order[j] = order[j], order[i]
    polished = [_formalize(sentences[k]) for k in order]
    if len(polished) >= 2:
        k = len(polished) // 2
        s = polished[k].strip()
        s = s[0].upper() + s[1:] if s else s
        polished[k] = "Moreover, " + s[0].lower() + s[1:]
    return " ".join(polished)


def ollama_reachable(timeout: float = 1.2) -> bool:
    """Cheap probe of a local Ollama (no model download, no inference)."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def default_human_samples() -> list[str]:
    return [_tidy(t) for t in _DEFAULT_HUMAN]


def load_human_samples(human_dir: str, limit: int) -> list[str]:
    """Human-side (label 0) samples from *human_dir*, else embedded defaults."""
    if human_dir:
        try:
            samples = C.load_labeled_texts(human_dir, limit=limit)
        except C.DataError:
            samples = []
        humans = [s["text"] for s in samples if s["label"] < 0.5]
        if humans:
            return humans[:limit]
    return default_human_samples()[:limit]


def build_surrogate_pairs(
    human_dir: str, limit: int, use_ollama: bool, force_deterministic: bool
) -> dict:
    """Build (texts, labels, note) for surrogate H2L mode.

    Each human sample yields two rows: the original (label 0) and its polished
    rewrite (label 1). Returns mechanism + the note to embed in JSON.
    """
    humans = load_human_samples(human_dir, limit)
    use_real_llm = (use_ollama and not force_deterministic
                    and ollama_reachable())
    texts: list[str] = []
    labels: list[float] = []
    if use_real_llm:
        from redteam.transforms import llm_rewrite  # noqa: PLC0415
        mechanism = "ollama_llm_rewrite"
        for h in humans:
            texts.append(h)
            texts.append(llm_rewrite(h) or h)
    else:
        mechanism = "deterministic"
        for h in humans:
            texts.append(h)
            texts.append(surrogate_polish(h))
    labels = []
    for _ in range(len(texts)):
        labels.append(0.0 if len(labels) % 2 == 0 else 1.0)
    note = ("surrogate H2L (synthesized via " + mechanism +
            "), NOT ARB data.")
    return {"texts": texts, "labels": labels, "mechanism": mechanism, "note": note}


def _encode_rows(texts, labels):
    return [{"text": t, "label": l} for t, l in zip(texts, labels)]


def main(argv=None):
    parser = C.add_eval_args("benchmarks/h2l_eval.py",
                             "H2L (human-polished-by-LLM) eval.",
                             default_data_dir="benchmarks/data/arb",
                             out_name="h2l")
    parser.add_argument("--arb-dir", default=None,
                        help="ARB dataset dir (real H2 data); else surrogate.")
    parser.add_argument("--human-dir", default=None,
                        help="human-only texts to polish (RAID etc.), else embedded.")
    parser.add_argument("--ollama", action="store_true",
                        help="prefer local Ollama rewrite when reachable.")
    parser.add_argument("--deterministic", action="store_true",
                        help="force the deterministic surrogate (no Ollama).")
    args = parser.parse_args(argv)
    C.handle_fetch(args, NAME, None)

    if args.arb_dir:
        try:
            arb = C.load_labeled_texts(args.arb_dir, limit=args.limit)
            arb_rows = _encode_rows([s["text"] for s in arb],
                                    [s["label"] for s in arb])
            h2l_mode = "arb"
        except C.DataError:
            print(f"[h2l] no ARB data under {args.arb_dir!r}; using surrogate.",
                  file=sys.stderr)
            arb_rows = None
            h2l_mode = "surrogate"
    else:
        arb_rows = None
        h2l_mode = "surrogate"

    if arb_rows is not None:
        texts = [r["text"] for r in arb_rows]
        labels = [r["label"] for r in arb_rows]
        note = "ARB H2L data."
    else:
        pairs = build_surrogate_pairs(args.human_dir, args.limit,
                                      use_ollama=args.ollama,
                                      force_deterministic=args.deterministic)
        texts = pairs["texts"]
        labels = pairs["labels"]
        note = pairs["note"]

    det_names = [n.strip() for n in args.detectors.split(",") if n.strip()]
    print(f"[h2l_eval] loading detectors {det_names} on {args.device} "
          f"(first load downloads weights)...", file=sys.stderr)
    try:
        specs = C.load_detector_specs(det_names, device=args.device)
    except Exception as e:  # noqa: BLE001
        print(f"Error loading detectors: {e}", file=sys.stderr)
        return 2

    summary = C.run_harness(NAME, texts, labels, specs, args.batch_size)
    summary["meta"]["mode"] = h2l_mode
    summary["meta"]["note"] = note
    if arb_rows is None:
        summary["meta"]["polish_mechanism"] = pairs["mechanism"]
    summary["meta"]["threat"] = ("H2L: human text rewritten by an LLM -- the "
                                 "dominant real-world failure mode (KB 07).")
    summary["args"] = vars(args)
    del summary["args"]["fetch"]

    C.write_json(args.out, summary)
    C.print_tabular(f"H2L per-layer AUROC ({h2l_mode})", summary["layers"])
    print(f"[h2l] mode={h2l_mode} wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
