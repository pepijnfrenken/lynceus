"""CLI entry point for humanify."""

import argparse
import logging
import sys

import numpy as np
import torch

from .analysis import find_markers, generate_advice, read_input, split_sentences
from .detectors import DETECTORS, get_detector
from .output import output_json, output_rich


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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="humanify",
        description="Analyze text for AI-like patterns using transformer models.",
    )
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="Input file path (default: stdin)")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output machine-readable JSON")
    parser.add_argument("--model", "-m", type=str, default="tmr",
                        help=f"Detection method (default: tmr). Comma-separate for ensemble: tmr,fast-detect. "
                             f"Available: {', '.join(DETECTORS)}")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override default model ID. For binoculars: 'observer,performer'")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="AI probability threshold for advice focus (default: 0.5)")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="Inference batch size (default: 32)")
    parser.add_argument("--advice", "-a", type=str, default="stats",
                        choices=["stats", "llm"],
                        help="Advice mode: stats (rule-based) or llm (AI-generated via causal LM)")
    args = parser.parse_args()

    # Read and split input
    raw = read_input(args.file)
    if not raw.strip():
        print("Error: no input text provided.", file=sys.stderr)
        sys.exit(1)
    sentences = split_sentences(raw)
    if not sentences:
        print("Error: no sentences found.", file=sys.stderr)
        sys.exit(1)

    # Device setup
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    # Always suppress noisy transformers output
    _suppress_transformers()

    # Parse model names (comma-separated for ensemble)
    model_names = [m.strip() for m in args.model.split(",")]
    for m in model_names:
        if m not in DETECTORS:
            print(f"Error: unknown detector '{m}'. Available: {', '.join(DETECTORS)}", file=sys.stderr)
            sys.exit(1)

    # Load and score with each detector
    # In JSON mode, redirect stdout to stderr during model loading to suppress
    # transformers' stdout error messages (they use print() not logging)
    detector_results = []
    for name in model_names:
        det = get_detector(name)
        if args.model_id:
            det.set_model_id(args.model_id)
        if args.json:
            _old_stdout = sys.stdout
            sys.stdout = sys.stderr
        det.load(device, fp16=use_cuda)
        probs = det.score(sentences, batch_size=args.batch_size)
        if args.json:
            sys.stdout = _old_stdout
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
            _adv_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base", dtype=dtype)
            _adv_model.to(device).eval()
            causal_lm = (_adv_model, _adv_tok, device)
        llm_advice = generate_llm_advice(*causal_lm, sentences, probs, markers, stats, args.threshold)

    # Output
    if args.json:
        output_json(sentences, probs, markers, stats, args.threshold, model_label,
                    detector_results=detector_results if len(detector_results) > 1 else None,
                    llm_advice=llm_advice)
    else:
        output_rich(sentences, probs, markers, stats, args.threshold, model_label,
                    detector_results=detector_results if len(detector_results) > 1 else None,
                    llm_advice=llm_advice)
