"""CLI entry point for humanify."""

import argparse
import logging
import sys

import numpy as np
import torch

from .analysis import ScoringUnit, find_markers, prepare_scoring_units, read_input
from .detectors import DETECTORS, get_detector
from .output import output_json, output_rich
import os


def _suppress_transformers():
    """Suppress all transformers output including stdout warnings."""
    import os
    import warnings
    os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    warnings.filterwarnings("ignore", module="transformers")
    import transformers
    transformers.logging.set_verbosity(logging.CRITICAL)
    logging.getLogger("transformers").setLevel(logging.CRITICAL)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.CRITICAL)


def _resolve_device(name: str) -> tuple[str, bool]:
    """Map CLI device selection to detector load arguments.

    Do not auto-probe CUDA here. Broken CUDA runtimes can hang inside the first
    driver call and cannot always be interrupted safely.
    """
    return name, name == "cuda"

def _expand_unit_scores(
    units: list[ScoringUnit],
    scores: list[float],
    sentence_count: int,
) -> list[float]:
    """Expand paragraph/chunk scores back to one score per display sentence."""
    if len(scores) != len(units):
        raise RuntimeError(
            f"detector returned {len(scores)} score(s) for {len(units)} scoring unit(s)"
        )

    probs: list[float] = []
    expected_start = 0
    for unit, score in zip(units, scores):
        if unit.start != expected_start or unit.stop <= unit.start:
            raise RuntimeError("internal scoring units are not contiguous")
        probs.extend([float(score)] * (unit.stop - unit.start))
        expected_start = unit.stop

    if expected_start != sentence_count:
        raise RuntimeError("internal scoring units do not cover all sentences")
    return probs

def _load_detectors(
    model_names: list[str],
    *,
    model_id: str | None,
    device_name: str,
    suppress_stdout: bool = False,
) -> list[dict]:
    """Validate detector names, load each detector, return list of result dicts.

    Each dict contains ``name``, ``detector`` instance, and ``model_id``.
    Detectors are loaded onto the resolved device with appropriate dtype.
    """
    for m in model_names:
        if m not in DETECTORS:
            raise ValueError(
                f"Unknown detector '{m}'. Available: {', '.join(DETECTORS)}"
            )

    device, use_cuda = _resolve_device(device_name)
    _suppress_transformers()

    loaded = []
    for name in model_names:
        det = get_detector(name)
        if model_id:
            det.set_model_id(model_id)
        _old_stdout = None
        try:
            if suppress_stdout:
                _old_stdout = sys.stdout
                sys.stdout = sys.stderr
            det.load(device, fp16=use_cuda)
        finally:
            if _old_stdout is not None:
                sys.stdout = _old_stdout
        loaded.append({
            "name": name,
            "detector": det,
            "model_id": det.model_id,
        })
    return loaded


def _score_text(
    raw: str,
    *,
    loaded_detectors: list[dict],
    batch_size: int,
) -> dict:
    """Score raw text using pre-loaded detectors.

    Returns a dict with ``sentences``, ``scoring_units``, ``probs``,
    ``stats``, ``markers``, ``model_label``, and ``detector_results``.
    Does not generate advice, rewrite suggestions, or use LLMs.
    """
    sentences, scoring_units = prepare_scoring_units(raw)
    score_texts = [unit.text for unit in scoring_units]

    detector_results = []
    for entry in loaded_detectors:
        det = entry["detector"]
        name = entry["name"]
        unit_probs = det.score(score_texts, batch_size=batch_size)
        probs = _expand_unit_scores(scoring_units, unit_probs, len(sentences))
        arr = np.array(probs)
        detector_results.append({
            "name": name,
            "model_id": det.model_id,
            "probs": probs,
            "stats": {
                "mean": float(arr.mean()), "std": float(arr.std()),
                "min": float(arr.min()), "max": float(arr.max()),
            },
            "detector": det,
        })

    # Ensemble: average probabilities across detectors
    if len(detector_results) == 1:
        probs = detector_results[0]["probs"]
        stats = detector_results[0]["stats"]
        model_label = detector_results[0]["model_id"]
    else:
        all_probs = np.array([r["probs"] for r in detector_results])
        probs = all_probs.mean(axis=0).tolist()
        ens_arr = np.array(probs)
        stats = {
            "mean": float(ens_arr.mean()), "std": float(ens_arr.std()),
            "min": float(ens_arr.min()), "max": float(ens_arr.max()),
        }
        model_label = "ensemble(" + ",".join(r["name"] for r in detector_results) + ")"

    markers = [find_markers(s) for s in sentences]

    return {
        "sentences": sentences,
        "scoring_units": scoring_units,
        "probs": probs,
        "stats": stats,
        "markers": markers,
        "model_label": model_label,
        "detector_results": detector_results,
    }


def _main_single(argv: list[str] | None = None) -> int:
    """Legacy single-text analysis. Returns exit code.

    On success this calls ``os._exit(0)`` to work around PyTorch/Transformer
    deadlocks during interpreter cleanup. Error paths return a normal int.
    """
    parser = argparse.ArgumentParser(
        prog="humanify",
        description="Analyze text for AI-like patterns using transformer models.",
    )
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="Input file path (default: stdin)")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output machine-readable JSON")
    parser.add_argument("--model", "-m", type=str, default="tmr",
                        help=(
                            f"Detection method (default: tmr). "
                            f"Comma-separate for ensemble: tmr,fast-detect. "
                            f"Available: {', '.join(DETECTORS)}"
                        ))
    parser.add_argument("--model-id", type=str, default=None,
                        help=(
                            "Override default model ID. For binoculars: "
                            "'observer,performer'"
                        ))
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="AI probability threshold for advice focus (default: 0.5)")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="Inference batch size (default: 32)")
    parser.add_argument("--device", "-d", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help=(
                            "Compute device (default: cpu; use cuda only when "
                            "your CUDA runtime is healthy)"
                        ))
    parser.add_argument("--advice", "-a", type=str, default="stats",
                        choices=["stats", "llm"],
                        help=(
                            "Advice mode: stats (rule-based) or llm "
                            "(AI-generated via causal LM)"
                        ))
    args = parser.parse_args(argv)

    # Read and prepare input. Detectors score normalized document-like chunks;
    # scores are expanded back to sentences for highlighting.
    raw = read_input(args.file)
    if not raw.strip():
        print("Error: no input text provided.", file=sys.stderr)
        return 1
    sentences = prepare_scoring_units(raw)[0]
    if not sentences:
        print("Error: no sentences found.", file=sys.stderr)
        return 1

    device, use_cuda = _resolve_device(args.device)

    # Load detectors using the shared helper
    try:
        loaded_detectors = _load_detectors(
            [m.strip() for m in args.model.split(",")],
            model_id=args.model_id,
            device_name=args.device,
            suppress_stdout=args.json,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Score text using the shared helper (advice-free by design)
    result = _score_text(
        raw,
        loaded_detectors=loaded_detectors,
        batch_size=args.batch_size,
    )

    sentences = result["sentences"]
    probs = result["probs"]
    stats = result["stats"]
    markers = result["markers"]
    model_label = result["model_label"]
    detector_results = result["detector_results"]

    # LLM advice generation (reuse causal LM from detector if available)
    llm_advice = None
    if args.advice == "llm":
        from .advice_llm import generate_llm_advice
        causal_lm = None
        for r in detector_results:
            causal_lm = r["detector"].get_causal_lm()
            if causal_lm:
                break
        if causal_lm is None:
            # No causal LM loaded — load Qwen3.5-0.8B-Base
            print("Loading Qwen3.5-0.8B-Base for LLM advice...", file=sys.stderr)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from .detectors.fast_detect import _login_hf
            _login_hf()
            _adv_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
            if _adv_tok.pad_token_id is None:
                _adv_tok.pad_token_id = _adv_tok.eos_token_id
            dtype = torch.float16 if use_cuda else None
            _adv_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3.5-0.8B-Base",
                dtype=dtype,
            )
            _adv_model.to(device).eval()
            causal_lm = (_adv_model, _adv_tok, device)
        llm_advice = generate_llm_advice(
            *causal_lm,
            sentences,
            probs,
            markers,
            stats,
            args.threshold,
        )

    # Output
    if args.json:
        output_json(
            sentences,
            probs,
            markers,
            stats,
            args.threshold,
            model_label,
            detector_results=detector_results if len(detector_results) > 1 else None,
            llm_advice=llm_advice,
        )
    else:
        output_rich(
            sentences,
            probs,
            markers,
            stats,
            args.threshold,
            model_label,
            detector_results=detector_results if len(detector_results) > 1 else None,
            llm_advice=llm_advice,
        )

    # Some transformer/PyTorch builds can deadlock during interpreter cleanup
    # after successful CPU inference. Flush the rendered result and skip object
    # finalization; this is a CLI process and all requested work is complete.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
    return 0  # unreachable


def _main_eval(argv: list[str]) -> None:
    """Parse eval subcommand args and run batch evaluator."""
    parser = argparse.ArgumentParser(
        prog="humanify eval",
        description="Batch evaluate text quality with deterministic metrics. "
                    "Reads JSONL input, emits metrics-only sample and summary records. "
                    "No rewrite advice or detection-evasion suggestions.",
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input JSONL file (use '-' for stdin)")
    parser.add_argument("--preferences", type=str, default=None,
                        help="Optional path to preference label JSONL")
    parser.add_argument("--model", "-m", type=str, default="tmr",
                        help=(
                            f"Detection method (default: tmr). "
                            f"Comma-separate for ensemble. "
                            f"Available: {', '.join(DETECTORS)}"
                        ))
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override default model ID")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="Inference batch size (default: 32)")
    parser.add_argument("--device", "-d", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Compute device (default: cpu)")
    args = parser.parse_args(argv)

    from .evaluator import run_eval

    exit_code = run_eval(
        args.input,
        args.preferences,
        model=args.model,
        model_id=args.model_id,
        batch_size=args.batch_size,
        device=args.device,
        output_stream=sys.stdout,
        error_stream=sys.stderr,
    )
    sys.exit(exit_code)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint. Dispatches to eval subcommand or legacy single-text mode."""
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "eval":
        _main_eval(argv[1:])
        return
    ret = _main_single(argv)
    sys.exit(ret)
