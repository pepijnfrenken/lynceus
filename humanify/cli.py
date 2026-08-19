"""CLI entry point for humanify."""

import argparse
import logging
import sys

import numpy as np
import torch

from .analysis import (
    ScoringUnit,
    find_markers,
    find_markers_weighted,
    generate_quality_advice,
    prepare_scoring_units,
    read_input,
    _AI_WEIGHTS,
)
from .artifacts import detect_artifacts
from .structure import audit_structure
from .detectors import DETECTORS, get_detector
from .detectors.base import _ensure_pad_token
from .output import output_json, output_rich
from .writing_metrics import compute_writing_metrics
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


# Narrative checklist tunables (Task B, Phase 5.2).
NARRATIVE_MIN_WORDS = 500      # KB: narrative features need long text
NARRATIVE_MAX_PROMPT_TOKENS = 4096  # tokenizer truncation for the checklist prompt


def _axis_name(name: str) -> str | None:
    """Map a detector name to its per-layer ensemble axis (Task C, Phase 6.1).

    * ``classifier_axis`` -- TMR / Desklib / Radar / EditLens (supervised
      classifiers; EditLens is edit-magnitude, included here but documented).
    * ``zero_shot_axis``  -- Fast-DetectGPT (+ Binoculars if non-degenerate).
    Returns ``None`` for detectors outside the ensemble axes.
    """
    if name in ("tmr", "desklib", "radar", "editlens"):
        return "classifier_axis"
    if name in ("fast-detect", "binoculars"):
        return "zero_shot_axis"
    return None


def _lexical_axis_score(sentences: list[str]) -> float:
    """Normalized era-weighted marker density (KB 02 L1b), in [0, 1].

    Normalization (documented): lexical density = (sum of matched term weights
    across all sentences) / (num_sentences * max_possible_per_sentence), where
    max_possible_per_sentence = the sum of ALL era-aware vocabulary weights (the
    highest density a single sentence could theoretically carry). Clamped to
    [0, 1]. Higher = more AI-typical vocabulary density.
    """
    if not sentences:
        return 0.0
    max_possible_per_sentence = sum(_AI_WEIGHTS.values())
    if max_possible_per_sentence <= 0:
        return 0.0
    matched_total = sum(
        weight for s in sentences for _term, weight in find_markers_weighted(s)
    )
    raw = matched_total / (len(sentences) * max_possible_per_sentence)
    return float(min(max(raw, 0.0), 1.0))


# Default (equal) per-axis weights for the weighted summary; documented here.
_AXIS_WEIGHTS: dict[str, float] = {
    "classifier_axis": 1.0,
    "zero_shot_axis": 1.0,
    "lexical_axis": 1.0,
    "structural_axis": 1.0,
    "artifact_axis": 1.0,
}


def _ensemble_axes(
    detector_results: list[dict],
    sentences: list[str],
    structure_audit: dict,
    artifact_audit: dict,
    scoring_failures: list[tuple[str, str]] | None = None,
) -> dict:
    """Compute per-AXIS evidence + overall score + agreement/confidence.

    Returns ``{"axes": {...}, "confidence": float}``. Axis scores live in
    [0, 1] (higher = more AI-like). ``axis_agreement`` is the std of available
    axis scores (higher std = lower confidence); ``confidence`` is derived as
    ``1 - min(std / 0.5, 1.0)`` (0.5 is the max possible std of a [0,1] value).
    ``axes["overall"]`` is a weight-mean of the non-missing axis scores.
    """
    axis_scores: dict[str, float | None] = {
        "classifier_axis": None,
        "zero_shot_axis": None,
        "lexical_axis": None,
        "structural_axis": None,
        "artifact_axis": None,
    }
    axis_exclusions: dict[str, list[str]] = {
        k: [] for k in axis_scores
    }
    for name, reason in scoring_failures or []:
        axis = _axis_name(name)
        if axis is not None:
            axis_exclusions[axis].append(f"{name} excluded ({reason})")

    # Detector backbones -> classifier / zero-shot axes.
    members: dict[str, list[np.ndarray]] = {
        "classifier_axis": [], "zero_shot_axis": [],
    }
    for r in detector_results:
        axis = _axis_name(r["name"])
        if axis is None:
            continue
        if r["name"] == "binoculars" and getattr(r["detector"], "is_degenerate", False):
            axis_exclusions[axis].append(
                f"binoculars excluded (degenerate observer==performer pair)"
            )
            continue
        members[axis].append(np.array(r["probs"]))

    for axis, arrs in members.items():
        if arrs:
            axis_scores[axis] = float(np.mean(np.array(arrs).mean(axis=0)))

    # Lexical / structural / artifact axes are always present.
    axis_scores["lexical_axis"] = _lexical_axis_score(sentences)
    axis_scores["structural_axis"] = float(structure_audit.get("score", 0.0))
    axis_scores["artifact_axis"] = float(artifact_audit.get("score", 0.0))

    present = [(k, v) for k, v in axis_scores.items() if v is not None]
    weights = _AXIS_WEIGHTS

    if present:
        overall = sum(v * weights[k] for k, v in present) / sum(
            weights[k] for k, _ in present
        )
        vals = np.array([v for _, v in present])
        axis_agreement = float(vals.std())
    else:
        overall = 0.5
        axis_agreement = 0.0

    confidence = max(0.0, min(1.0, 1.0 - (axis_agreement / 0.5)))

    axes = {
        "overall": round(overall, 4),
        "weights": dict(weights),
        "axis_agreement": round(axis_agreement, 4),
        "exclusions": axis_exclusions,
        "axis_definitions": {
            "classifier_axis": "TMR/Desklib/RADAR/EditLens mean (EditLens = edit-magnitude, included here but orthogonal to authorship)",
            "zero_shot_axis": "Fast-DetectGPT (+ Binoculars if non-degenerate)",
            "lexical_axis": "normalized era-aware marker density (mean matched weight / max possible)",
            "structural_axis": "structure_audit score (T2/T3/T5)",
            "artifact_axis": "artifact_audit score (post-processing / EditLens risk)",
        },
    }
    axes.update(
        {k: {"score": (round(v, 4) if v is not None else None),
             "weight": weights[k],
             "excluded": axis_exclusions[k]}
         for k, v in axis_scores.items()}
    )

    return {
        "axes": axes,
        "confidence": round(confidence, 4),
    }


def _score_text(
    raw: str,
    *,
    loaded_detectors: list[dict],
    batch_size: int,
) -> dict:
    """Score raw text using pre-loaded detectors.

    Returns a dict with ``sentences``, ``scoring_units``, ``probs``,
    ``stats``, ``markers``, ``model_label``, ``detector_results`` (all
    backward-compatible) plus ``axes`` and ``confidence`` (per-layer ensemble,
    Task C / Phase 6.1). Also ``structure_audit`` and ``artifact_audit``.
    """
    sentences, scoring_units = prepare_scoring_units(raw)
    score_texts = [unit.text for unit in scoring_units]

    detector_results = []
    scoring_failures: list[tuple[str, str]] = []  # (name, error)
    for entry in loaded_detectors:
        det = entry["detector"]
        name = entry["name"]
        try:
            unit_probs = det.score(score_texts, batch_size=batch_size)
            probs = _expand_unit_scores(scoring_units, unit_probs, len(sentences))
        except Exception as exc:  # noqa: BLE001 - a detector must never kill a run
            # Exclude this detector from its axis and note the failure.
            scoring_failures.append((name, f"score failed: {type(exc).__name__}"))
            continue
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

    # Ensemble: average probabilities across detectors (backward compatible).
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
    structure_audit = audit_structure(raw)
    artifact_audit = detect_artifacts(raw)
    per_layer = _ensemble_axes(
        detector_results, sentences, structure_audit, artifact_audit,
        scoring_failures=scoring_failures,
    )

    return {
        "sentences": sentences,
        "scoring_units": scoring_units,
        "probs": probs,
        "stats": stats,
        "markers": markers,
        "model_label": model_label,
        "detector_results": detector_results,
        "structure_audit": structure_audit,
        "artifact_audit": artifact_audit,
        "axes": per_layer["axes"],
        "confidence": per_layer["confidence"],
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
                        choices=["stats", "llm", "narrative"],
                        help=(
                            "Advice mode: stats (rule-based), llm (AI-generated "
                            "via causal LM), or narrative (document-level "
                            "narrative checklist; needs 500+ words)"
                        ))
    parser.add_argument("--domain", type=str, default="creative",
                        choices=["creative", "academic", "social", "legal"],
                        help=(
                            "Domain for calibration: creative (default), "
                            "academic, social, legal. Adjusts thresholds and "
                            "adds false-positive warnings."
                        ))
    parser.add_argument("--prompt", type=str, default=None,
                        help=(
                            "Original writing prompt (improves keyword-coverage "
                            "and coherence scoring)"
                        ))
    parser.add_argument("--style", type=str, default="general",
                        choices=["general", "academic"],
                        help=(
                            "Writing style calibration: general (default) or "
                            "academic (shifts thresholds for formal prose)"
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
    except (ValueError, RuntimeError) as e:
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

# Writing quality metrics (deterministic)
    quality_metrics = compute_writing_metrics(
        args.prompt or raw, raw
    )
    quality_advice = generate_quality_advice(quality_metrics, style=args.style)

    # Extract sentence analysis and top-worst sentences
    sentence_analysis = quality_metrics.get("sentence_analysis", [])
    paragraph_analysis = quality_metrics.get("paragraph_analysis", {})
    from .writing_metrics import get_top_worst_sentences
    top_worst = {
        dim: get_top_worst_sentences(sentences, dim, top_n=3)
        for dim in ("grammar", "specificity", "evidence", "readability")
    }

    # LLM-gated layers: "llm" advice and "narrative" checklist. Both reuse the
    # detector's causal LM when available, else load Qwen3.5-0.8B-Base. The
    # narrative checklist is gated at 500+ words and NEVER loads a model for
    # short input (KB: narrative features need long text; short scores are
    # noise).
    llm_advice = None
    quality_llm_advice = None
    narrative = None
    if args.advice in ("llm", "narrative"):
        short_narrative = args.advice == "narrative" and (
            len(raw.split()) < NARRATIVE_MIN_WORDS
        )
        if short_narrative:
            # Gate fires first: no model load, no LLM docs.
            from .narrative_checklist import run_narrative_checklist

            narrative = run_narrative_checklist(
                raw, lambda _prompt: "", min_words=NARRATIVE_MIN_WORDS
            )
        else:
            from .advice_llm import (
                _format_generation_prompt,
                _pad_token_id,
                _strip_think_blocks,
            )
            causal_lm = None
            for r in detector_results:
                causal_lm = r["detector"].get_causal_lm()
                if causal_lm:
                    break
            if causal_lm is None:
                # No causal LM loaded — load Qwen3.5-0.8B-Base
                print("Loading Qwen3.5-0.8B-Base for LLM-assisted analysis...",
                      file=sys.stderr)
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from .detectors.fast_detect import _login_hf

                _login_hf()
                _adv_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
                _ensure_pad_token(_adv_tok)
                dtype = torch.float16 if use_cuda else None
                _adv_model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen3.5-0.8B-Base",
                    dtype=dtype,
                )
                _adv_model.to(device).eval()
                causal_lm = (_adv_model, _adv_tok, device)

            _model, _tokenizer, _device = causal_lm

            def llm_call(prompt: str) -> str:
                """Tokenize -> generate -> decode, reusing detector/loaded LM."""
                formatted = _format_generation_prompt(_tokenizer, prompt)
                inputs = _tokenizer(
                    formatted, return_tensors="pt", truncation=True,
                    max_length=NARRATIVE_MAX_PROMPT_TOKENS,
                    return_token_type_ids=False,
                ).to(_device)
                with torch.no_grad():
                    output_ids = _model.generate(
                        **inputs,
                        max_new_tokens=300,
                        do_sample=False,
                        pad_token_id=_pad_token_id(_tokenizer),
                    )
                generated = output_ids[0][inputs["input_ids"].shape[1]:]
                text = _tokenizer.decode(generated,
                                         skip_special_tokens=True).strip()
                return _strip_think_blocks(text)

            if args.advice == "llm":
                from .advice_llm import generate_llm_advice, generate_quality_llm_advice

                llm_advice = generate_llm_advice(
                    *causal_lm, sentences, probs, markers, stats, args.threshold
                )
                quality_llm_advice = generate_quality_llm_advice(
                    *causal_lm,
                    quality_metrics,
                )
            else:  # narrative
                from .narrative_checklist import run_narrative_checklist

                narrative = run_narrative_checklist(
                    raw, llm_call, min_words=NARRATIVE_MIN_WORDS
                )

    # Output
    structure_audit = result["structure_audit"]
    axes = result["axes"]
    confidence = result["confidence"]

    # Domain calibration (Task D / Phase 6.2): adds "domain", "domain_warnings"
    # and "domain_settings" to the result dict; never removes existing keys.
    from .calibration import calibrate

    calibrated = calibrate(result, args.domain)
    domain = calibrated["domain"]
    domain_settings = calibrated.get("domain_settings", {})
    domain_warnings = calibrated.get("domain_warnings", [])

    _shared = dict(
        detector_results=detector_results if len(detector_results) > 1 else None,
        llm_advice=llm_advice,
        structure_audit=structure_audit,
        artifact_audit=result["artifact_audit"],
        narrative=narrative,
        axes=axes,
        confidence=confidence,
        domain=domain,
        domain_settings=domain_settings,
        domain_warnings=domain_warnings,
    )
    if args.json:
        output_json(
            sentences,
            probs,
            markers,
            stats,
            args.threshold,
            model_label,
            quality_metrics=quality_metrics,
            quality_advice=quality_advice,
            quality_llm_advice=quality_llm_advice,
            sentence_analysis=sentence_analysis,
            paragraph_analysis=paragraph_analysis,
            top_worst=top_worst,
            **_shared,
        )
    else:
        output_rich(
            sentences,
            probs,
            markers,
            stats,
            args.threshold,
            model_label,
            quality_metrics=quality_metrics,
            quality_advice=quality_advice,
            quality_llm_advice=quality_llm_advice,
            sentence_analysis=sentence_analysis,
            paragraph_analysis=paragraph_analysis,
            top_worst=top_worst,
            **_shared,
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
