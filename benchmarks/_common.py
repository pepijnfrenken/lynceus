"""Shared evaluation harness for the lynceus robustness suite (Phase 1).

This module implements the metric + detector-loop + per-layer logic shared by
the RAID, M4 and H2L eval scripts. It depends only on :mod:`numpy` plus the
in-repo :mod:`lynceus` package, so it runs under the project's existing
dependencies (transformers, torch, rich, numpy). It deliberately does NOT
require sklearn -- all ranking metrics are implemented in numpy.

Per KB doc 04 section 6 and KB doc 07 we never report a bare accuracy number:
every eval reports a per-layer breakdown, plus AUROC / TPR@5%FPR / macro-F1.
The H2L (human-polished-by-LLM) regime is the dominant real-world failure mode
and is treated specially by ``benchmarks/h2l_eval.py``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import statistics
from pathlib import Path

import numpy as np

from lynceus.analysis import find_markers, normalize_for_detection, split_sentences


# ---------------------------------------------------------------------------
# Data presence / loading helpers
# ---------------------------------------------------------------------------

class DataError(Exception):
    """Raised when a required (multi-GB, manually downloaded) dataset is missing."""


def check_data_present(data_dir: str, pattern: str = "*.jsonl") -> Path:
    """Return the first file matching *pattern*, or raise :class:`DataError`.

    Used to FAIL FAST (exit code 2 in the CLI) instead of hanging on a dataset
    that was never downloaded.
    """
    d = Path(data_dir)
    files = sorted(d.glob(pattern)) if d.exists() else []
    if not files:
        raise DataError(
            f"No {pattern!r} files found under {data_dir!r}. The dataset has "
            "not been downloaded (it is multi-GB). Download it manually, then "
            "point --data-dir at the directory containing the split."
        )
    return files[0]


def _to_ai_label(value) -> float:
    """Normalise a raw label from a dataset row to 1 (AI) / 0 (human)."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "ai", "machine", "machine-generated", "generated",
                 "ai-generated", "text"}:
            return 1.0
        if v in {"0", "false", "human", "human-written", "human_written"}:
            return 0.0
        try:
            return 1.0 if float(v) >= 1 else 0.0
        except ValueError:
            return 1.0
    return 1.0 if float(value) >= 1 else 0.0


def load_labeled_texts(data_dir, pattern="*.jsonl", limit=None):
    """Load labelled text samples from jsonl files.

    Handles two common row shapes:

    * ``{"human_text": ..., "ai_text": ...}`` (RAID-style pair) -> a human
      (label 0) and an AI (label 1) sample per row.
    * ``{"text": ..., "label": ...}`` (M4-style) -> one sample.
    """
    check_data_present(data_dir, pattern)
    samples = []
    for f in sorted(Path(data_dir).glob(pattern)):
        if limit is not None and len(samples) >= limit:
            break
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if limit is not None and len(samples) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "human_text" in row and "ai_text" in row:
                    if row.get("human_text"):
                        samples.append({"text": row["human_text"], "label": 0.0,
                                        "sid": row.get("id")})
                    if row.get("ai_text"):
                        samples.append({"text": row["ai_text"], "label": 1.0,
                                        "sid": row.get("id")})
                elif "text" in row and "label" in row:
                    samples.append({"text": row["text"],
                                    "label": _to_ai_label(row["label"]),
                                    "sid": row.get("id")})
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise DataError(f"No labelled text found in {data_dir!r}.")
    return samples


# ---------------------------------------------------------------------------
# Ranking metrics (numpy-only AUROC / TPR@FPR / macro-F1)
# ---------------------------------------------------------------------------

def _f1_class(tp: int, fp: int, fn: int) -> float:
    """F1 for a single class from its tp/fp/fn counts (0 when undefined)."""
    denom = 2 * tp + fp + fn
    return 2.0 * tp / denom if denom > 0 else 0.0


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    """Average (mean-of-tied-group) ranks for *scores* (higher = better)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(len(scores))
    sorted_sc = scores[order]
    start = 0
    for i in range(1, len(sorted_sc) + 1):
        if i == len(sorted_sc) or sorted_sc[i] != sorted_sc[start]:
            ranks[order[start:i]] = float(start + i - 1) / 2.0
            start = i
    return ranks


def auroc(scores, labels) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    Returns NaN when either class is absent.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    # 0-based average ranks (0 = smallest); Mann-Whitney U = R1 - n1(n1-1)/2
    return float((ranks[labels == 1].sum() - n_pos * (n_pos - 1) / 2.0)
                 / (n_pos * n_neg))


def tpr_at_fpr(scores, labels, target_fpr: float = 0.05) -> float:
    """Highest TPR achieved with FPR <= *target_fpr* (sorted by score desc)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores)
    tp = fp = 0
    best = 0.0
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        if fp / n_neg <= target_fpr:
            best = max(best, tp / n_pos)
        else:
            break
    return float(best)


def macro_f1(scores, labels, threshold: float = 0.5) -> float:
    """Macro-averaged per-class F1 at a fixed decision threshold."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    f1_pos = _f1_class(tp, fp, fn)
    f1_neg = _f1_class(tn, fn, fp)  # neg-class "tps" = tn, fp=fn, fn=fp
    return round((f1_pos + f1_neg) / 2.0, 4)


def sample_metrics(scores, labels, threshold: float = 0.5) -> dict:
    """Per-detector metric dict (AUROC, TPR@5%FPR, macro-F1)."""
    return {
        "auroc": round(auroc(scores, labels), 4),
        "tpr_at_5_%_fpr": round(tpr_at_fpr(scores, labels, 0.05), 4),
        "macro_f1": macro_f1(scores, labels, threshold),
    }


def best_threshold_f1_metrics(scores, labels) -> dict:
    """Precision/recall/F1 at the threshold that maximises macro-F1."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": 0.0}
    for t in np.unique(scores):
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = _f1_class(tp, fp, fn)
        if f1 > best["f1"]:
            best = {"precision": prec, "recall": rec, "f1": f1,
                    "threshold": float(t)}
    return {k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in best.items()}


# ---------------------------------------------------------------------------
# Detector loop
# ---------------------------------------------------------------------------

def run_detectors(detector_specs, texts: list, batch_size: int = 16) -> dict:
    """Run each (already-loaded) detector over *texts*.

    *detector_specs* is a list of ``(display_name, detector_instance)``.
    Returns ``{display_name: [0-1 scores]}`` with one score per text.
    """
    results = {}
    for name, det in detector_specs:
        scores = det.score(texts, batch_size=batch_size)
        if len(scores) != len(texts):
            raise RuntimeError(
                f"{name}: detector returned {len(scores)} scores for "
                f"{len(texts)} texts"
            )
        results[name] = [float(s) for s in scores]
    return results


def ensemble_scores(detector_results, detector_names=None) -> list:
    """Per-text mean AI probability across detectors (simple ensemble)."""
    names = detector_names or list(detector_results.keys())
    stack = np.stack([np.asarray(detector_results[n], dtype=float) for n in names])
    return [float(x) for x in stack.mean(axis=0)]


# ---------------------------------------------------------------------------
# Per-layer metrics  (Task E)
# ---------------------------------------------------------------------------

def burstiness_feature(text: str) -> float:
    """Sentence-length coefficient of variation (std/mean) for *text*."""
    counts = [len(s.split()) for s in split_sentences(text)]
    if len(counts) < 2:
        return 0.0
    mean = statistics.mean(counts)
    if mean <= 0:
        return 0.0
    return statistics.pstdev(counts) / mean


def lexical_density(text: str) -> float:
    """Number of distinct AI filler markers per word."""
    norm = normalize_for_detection(text)
    tokens = norm.split()
    if not tokens:
        return 0.0
    return len(find_markers(norm)) / len(tokens)


def compute_layer_metrics(texts, labels, detector_results) -> dict:
    """Per-layer robustness report reused by every eval (Task E).

    *detector_results* maps display name -> list of 0-1 AI probabilities
    aligned with *texts*.
    """
    labels_f = [float(l) for l in labels]
    n = len(texts)

    densities = [lexical_density(t) for t in texts]
    lex = best_threshold_f1_metrics(densities, labels_f)
    lexical = {
        "feature": "find_markers marker density (markers/word)",
        "auroc": round(auroc(densities, labels_f), 4),
        "precision": lex["precision"],
        "recall": lex["recall"],
        "f1": lex["f1"],
        "best_threshold": lex["threshold"],
    }

    burst = [burstiness_feature(t) for t in texts]
    burstiness = {
        "feature": "sentence-length coeff-of-variation (std/mean)",
        "auroc": round(auroc(burst, labels_f), 4),
    }

    if detector_results:
        classifier = {
            name: sample_metrics(scores, labels_f)
            for name, scores in detector_results.items()
        }
        ensemble = sample_metrics(ensemble_scores(detector_results), labels_f)
    else:
        classifier = {}
        ensemble = {"auroc": float("nan"), "tpr_at_5_%_fpr": float("nan"),
                    "macro_f1": float("nan")}

    return {
        "n": n,
        "n_human": int(sum(1 for l in labels_f if l < 0.5)),
        "n_ai": int(sum(1 for l in labels_f if l >= 0.5)),
        "lexical": lexical,
        "burstiness": burstiness,
        "classifier": classifier,
        "ensemble": ensemble,
    }


# ---------------------------------------------------------------------------
# Real model loading (only from main(); never from smoke tests, which must not
# download weights)
# ---------------------------------------------------------------------------

def load_detector_specs(det_names, device="cpu", fp16=False):
    """Instantiate + load real detectors from the lynceus registry.

    The same-model Binoculars implementation is degenerate (cross-perplexity
    with identical models == plain perplexity, KB 06), so it is labelled
    ``binoculars(degen)`` in every output.
    """
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "critical")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from lynceus.detectors import get_detector

    specs = []
    for name in det_names:
        det = get_detector(name)
        label = "binoculars(degen)" if name == "binoculars" else name
        det.load(device, fp16=fp16)
        specs.append((label, det))
    return specs


# ---------------------------------------------------------------------------
# Output + arg-parsing helpers
# ---------------------------------------------------------------------------

def today() -> str:
    return _dt.date.today().isoformat()


def default_out_path(script: str) -> str:
    d = Path("benchmarks")
    d.mkdir(exist_ok=True)
    return str(d / f"{script}_{today()}.json")


def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def print_tabular(title: str, layers: dict) -> None:
    """Print a per-layer AUROC table (rich if available, plain otherwise)."""

    def _fmt(x):
        if isinstance(x, float) and x != x:  # NaN
            return "n/a"
        return f"{x:.4f}"

    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title=title)
        table.add_column("layer", style="cyan")
        table.add_column("auroc", justify="right")
        table.add_row("lexical", _fmt(layers["lexical"]["auroc"]))
        table.add_row("burstiness", _fmt(layers["burstiness"]["auroc"]))
        for name, m in sorted(layers["classifier"].items()):
            table.add_row(name, _fmt(m["auroc"]))
        table.add_row("ensemble", _fmt(layers["ensemble"]["auroc"]))
        Console().print(table)
    except Exception:
        print(f"  lexical    : {_fmt(layers['lexical']['auroc'])}")
        print(f"  burstiness : {_fmt(layers['burstiness']['auroc'])}")
        for name, m in sorted(layers["classifier"].items()):
            print(f"  {name:<12}: {_fmt(m['auroc'])}")
        print(f"  ensemble   : {_fmt(layers['ensemble']['auroc'])}")


# ---------------------------------------------------------------------------
# CLI plumbing shared by raid_eval.py / m4_eval.py / h2l_eval.py
# ---------------------------------------------------------------------------

def add_eval_args(prog, description, default_data_dir, out_name):
    """Build an argparse parser common to the benchmark scripts."""
    parser = argparse.ArgumentParser(prog=prog, description=description,
                                     add_help=True)
    parser.add_argument("--data-dir", default=default_data_dir,
                        help="Directory holding the dataset .jsonl split "
                             "(default: %(default)s)")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max samples to score (default: %(default)s). "
                             "Keep small; the full --fetch run is manual.")
    parser.add_argument("--fetch", action="store_true",
                        help="Print the manual multi-GB download steps and do "
                             "NOT auto-download.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="detector inference batch size (default: %(default)s)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="compute device (default: %(default)s)")
    parser.add_argument("--detectors", default="tmr,desklib,fast-detect,binoculars",
                        help="comma-separated detector registry names "
                             "(default: %(default)s)")
    parser.add_argument("--out", default=default_out_path(out_name),
                        help="output JSON path (default: %(default)s)")
    return parser


def handle_fetch(args, dataset_label: str, dataset_repo: str) -> None:
    """Print the multi-GB download warning; never download inside the harness."""
    if not args.fetch:
        return
    print(f"[fetch:{dataset_label}] --fetch requested but this harness never "
          f"auto-downloads (datasets are multi-GB, KB Phase 1 rule 2).", flush=True)
    dest = getattr(args, "data_dir", "benchmarks/data")
    print(f"[fetch:{dataset_label}] manual step for you to run, then re-run "
          f"this eval with --data-dir {dest}.", flush=True)
    if "/datasets/" in dataset_repo:
        repo = dataset_repo.split("/datasets/")[-1]
    else:
        repo = dataset_repo.strip("/")
    print(f"    huggingface-cli download {repo} --repo-type dataset "
          f"--local-dir {dest}", flush=True)




def run_harness(name, texts, labels, detector_specs, batch_size=16, note=None):
    """Run the detector loop + per-layer report and return a summary dict.

    *detector_specs* is a list of ``(display_name, loaded_detector)``. Passing
    already-loaded detectors keeps model download out of the smoke-test path.
    """
    det_results = run_detectors(detector_specs, texts, batch_size=batch_size)
    layers = compute_layer_metrics(texts, labels, det_results)
    meta = {"n": layers["n"], "n_human": layers["n_human"], "n_ai": layers["n_ai"]}
    if note:
        meta["note"] = note
    return {
        "name": name,
        "meta": meta,
        "layers": layers,
        "detectors": layers["classifier"],
    }
