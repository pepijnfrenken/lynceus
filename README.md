# humanify

<p align="center">
  <img src="assets/humanify-mascot.svg" alt="A friendly robot inspecting highlighted writing in a terminal window" width="820">
</p>

`humanify` is a local-first command-line analyzer for spotting text that is statistically similar to modern AI-generated writing. It scores each sentence, highlights passages worth reviewing, reports repeated AI-like markers, measures sentence-score variation, and can return either rich terminal output or JSON for automation.

Use it as a writing-quality and risk-review signal, not as proof of authorship.

![CI](https://github.com/pepijnfrenken/humanify/actions/workflows/ci.yml/badge.svg)

## Why use it

- **Sentence-level evidence**: see which passages contributed to the score instead of only a document-level verdict.
- **Multiple detector backends**: run classifier-based, zero-shot, perplexity-ratio, or ensemble scoring.
- **Reviewer-friendly output**: rich terminal diagnostics for humans; JSON for CI, dashboards, and pipelines.
- **Writing advice**: deterministic statistics-based tips by default, with optional local LLM-generated coaching.
- **Local inference**: input text is read from a file or standard input and scored with local model execution.

## Contents

- [Important limitations](#important-limitations)
- [Installation](#installation)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [Detectors](#detectors)
- [Output](#output)
- [Evaluation mode](#evaluation-mode)
- [Feature analysis layers](#feature-analysis-layers)
- [Robustness](#robustness)
- [Red team](#red-team)
- [Writing advice](#writing-advice)
- [Model overrides](#model-overrides)
- [Thresholds](#thresholds)
- [Performance](#performance)
- [Hugging Face authentication](#hugging-face-authentication)
- [Privacy and data handling](#privacy-and-data-handling)
- [Exit behavior](#exit-behavior)
- [Development](#development)
- [License](#license)

## Important limitations

AI-text detection is probabilistic. `humanify` detects statistical similarity to text produced by modern language models; it does **not** prove that a person or an AI wrote a document.

Known sources of false positives include:

- Formal business writing
- Academic prose
- Legal and policy text
- Non-native English
- Highly edited or templated writing
- Older public-domain writing styles
- Text that is too short to provide enough signal

Do not use the output as the sole basis for academic, employment, compliance, or enforcement decisions. Treat high scores as prompts for careful review of the actual sentences, context, model choice, and reason for the score.

## Installation

### From source

```bash
git clone <repo-url>
cd humanify
uv sync
uv run humanify --help
```

### As a local tool

```bash
uv tool install .
humanify --help
```

### Requirements

- Python 3.11 or newer
- PyTorch 2.0 or newer
- `transformers` 4.30 or newer
- `rich` 13.0 or newer
- `numpy` 1.24 or newer
- Internet access on first run to download Hugging Face models
- Optional: CUDA-capable GPU for faster inference

Model files are cached by Hugging Face after the first download.

## Quick start

Analyze a file:

```bash
uv run humanify --file examples/mixed-writing-sample.txt
```

Analyze text from standard input:

```bash
printf '%s\n' "It is crucial to leverage robust paradigms." | uv run humanify
```

Emit machine-readable JSON:

```bash
uv run humanify --json --file examples/mixed-writing-sample.txt
```

Run a stronger classifier:

```bash
uv run humanify --model desklib --file examples/mixed-writing-sample.txt
```

Run an ensemble:

```bash
uv run humanify --model tmr,fast-detect --file examples/mixed-writing-sample.txt
```

Focus advice on sentences above a custom threshold:

```bash
uv run humanify --threshold 0.3 --file examples/mixed-writing-sample.txt
```

Generate local LLM-based writing advice:

```bash
uv run humanify --advice llm --file examples/mixed-writing-sample.txt
```

## CLI reference

```text
usage: humanify [-h] [--file FILE] [--json] [--model MODEL]
                [--model-id MODEL_ID] [--threshold THRESHOLD]
                [--batch-size BATCH_SIZE] [--device {cpu,cuda}]
                [--advice {stats,llm,narrative}]
                [--domain {creative,academic,social,legal}] [--prompt PROMPT]
                [--style {general,academic}]

Analyze text for AI-like patterns using transformer models.

options:
  -h, --help            show this help message and exit
  --file, -f FILE       Input file path (default: stdin)
  --json, -j            Output machine-readable JSON
  --model, -m MODEL     Detection method (default: tmr). Comma-separate for
                        ensemble: tmr,fast-detect. Available: tmr, desklib,
                        fast-detect, binoculars, editlens, radar, meld
  --model-id MODEL_ID   Override default model ID. For binoculars:
                        'observer,performer'
  --threshold, -t THRESHOLD
                        AI probability threshold for advice focus (default:
                        0.5)
  --batch-size, -b BATCH_SIZE
                        Inference batch size (default: 32)
  --device, -d {cpu,cuda}
                        Compute device (default: cpu; use cuda only when your
                        CUDA runtime is healthy)
  --advice, -a {stats,llm,narrative}
                        Advice mode: stats (rule-based), llm (AI-generated via
                        causal LM), or narrative (document-level narrative
                        checklist; needs 500+ words)
  --domain {creative,academic,social,legal}
                        Domain for calibration: creative (default), academic,
                        social, legal. Adjusts thresholds and adds
                        false-positive warnings.
  --prompt PROMPT       Original writing prompt (improves keyword-coverage and
                        coherence scoring)
  --style {general,academic}
                        Writing style calibration: general (default) or
                        academic (shifts thresholds for formal prose)
```
### Evaluation subcommand

```bash
uv run humanify eval --help
```

The `eval` subcommand provides batch metrics-only evaluation. It accepts a JSONL input file, optional preference labels for correlation analysis, and outputs per-sample and aggregate JSONL records. See the [Evaluation mode](#evaluation-mode) section for details.

## Detectors

| Detector | Default model | Method | Best for | Notes |
| --- | --- | --- | --- | --- |
| `tmr` | `Oxidane/tmr-ai-text-detector` | Text-classification pipeline | Fast default reviews | 125M RoBERTa-base model. |
| `desklib` | `desklib/ai-text-detector-v1.01` | DeBERTa-v3-large classifier | Stronger classifier scoring | Heavier model with stronger resource requirements. |
| `fast-detect` | `Qwen/Qwen3.5-0.8B-Base` | Fast-DetectGPT-style log-probability curvature | Zero-shot likelihood behavior | Uses a causal LM. |
| `binoculars` | `Qwen/Qwen3.5-0.8B-Base,Qwen/Qwen3.5-0.8B-Base` | Binoculars-style perplexity ratio | Perplexity-ratio review | Loads two causal LMs; highest memory use among defaults. **Default pair is degenerate** (see note). |
| `editlens` | `pangram/editlens_roberta-large` | AI-edit-magnitude classifier | Text that looks post-processed / paraphrased | CC BY-NC-SA, contact-gated on HF. A separate *edit* axis (0-1) shown apart from authorship probability, never blended into the mean ensemble. |
| `radar` | `TrustSafeAI/RADAR-Vicuna-7B` | Paraphrase-robust classifier | Text that has already been reworded by a humanizer | Non-commercial license. The one classifier family that stays strong after a paraphrase attack (AUROC ≈ 0.92 post-attack). |
| `meld` | `anon-review-meld-2026/meld` | Custom anchor/prototype classifier (Ettin-400m BiLM + style-space metric head) | Strongest open-source detector on RAID (arXiv 2605.06903) | 395M params, MIT license. Needs ≥ 100 words (see note). The scoring head is custom — `AutoModel`/`pipeline()` silently discard its weights, so the checkpoint loads with `strict=True`. |

> **Binoculars default caveat.** The shipped `binoculars` default uses the *same* model (`Qwen/Qwen3.5-0.8B-Base`) for both observer and performer roles — cross-perplexity between identical models collapses to plain perplexity. Pass `--model-id observer-model,performer-model` with two different close models for a real Binoculars-style ratio; otherwise read the default as a plain-perplexity signal.

> **MELD usage notes.** MELD needs **≥ 100 words** to be reliable — below ~50 words it flags most genuine human writing, so `humanify` blends short readings toward neutral 0.5 (same policy as RADAR). It reads the first **2048 tokens** (`max_length`) of a document; humanify's 320-word scoring units stay well under that cap, and longer documents are scored unit-by-unit and aggregated. **Threshold guidance:** for a 1%-FPR flag, compare MELD's *raw* score against the checkpoint's `score_offsets.overall.fpr_0.01` (per-stratum offsets in `score_offsets.strata`) — do not use P(AI) ≈ 0.5, since MELD's P(AI) is not calibrated to 0.5 (genuine human prose averages ≈ 0.2). The detector keeps the raw per-unit scores and threshold available in Python (`detector.last_raw_scores`, `detector.threshold`, `detector.flag_units()`); the CLI JSON output reports the calibrated P(AI) like every other detector. Keep original line breaks — re-flowing paragraphs shifts the score. Weights go to the Hugging Face cache; set `HUMANIFY_MELD_DIR` to a local checkpoint directory (containing `meld_config.json` + `model.safetensors`) as a fallback.

### Choosing a detector

- Start with `tmr` for the fastest default experience.
- Use `desklib` when you can afford a larger classifier and want a stronger single-model signal.
- Use `fast-detect` when you want a zero-shot detector based on causal-LM likelihood behavior.
- Use `binoculars` when you specifically want a perplexity-ratio method and have enough memory for two LMs.
- Use `meld` when you want the strongest open-source detector on RAID (395M params; needs ≥ 100 words and the first download is ~1.6 GB).
- Use an ensemble when disagreement between methods is useful for review:

```bash
uv run humanify --model tmr,desklib,fast-detect --file examples/mixed-writing-sample.txt
```

## Output

### Terminal output

The default rich output shows every sentence with its context-aware score and visual status:

- Green check: below 30% AI probability
- Yellow warning: 30-70% AI probability
- Red cross: above 70% AI probability

For sentences above `--threshold`, detected markers are shown below the sentence. Model inference runs on normalized document-like chunks, then maps each chunk score back to its sentences so short fragments and blank-line formatting are not over-weighted. The summary includes:

- Overall AI probability across context-aware sentence scores
- Burstiness, reported as standard deviation across context-aware sentence scores
- Sentence count
- Minimum and maximum context-aware sentence scores
- Model or ensemble label
- **Writing quality metrics** (grammar, specificity, evidence, readability, lexical diversity, coherence, style consistency)
- **Writing quality advice** — concrete strengths and weaknesses based on deterministic metrics
- Writing advice (AI-detection focused)
- False-positive disclaimer

### JSON output

`--json` writes structured output suitable for automation:

```bash
uv run humanify --json --file examples/mixed-writing-sample.txt
```

Top-level fields include:

- `model`: model ID or ensemble label
- `sentences`: per-sentence text, context-aware probability, and markers
- `overall_score`: mean context-aware sentence probability
- `burstiness`: standard deviation of context-aware sentence probabilities
- `stats`: mean, standard deviation, minimum, and maximum
- `sentence_count`: number of analyzed sentences
- `advice`: deterministic or LLM-generated AI advice payload
- `writing_quality`: deterministic quality metrics and advice
  - `metrics`: raw and normalized scores for grammar, specificity, evidence, readability, lexical diversity, sentence variety, repetition, coherence, and style consistency
  - `advice`: concrete tips on what works and what doesn't
  - `llm_text`: optional LLM-generated quality coaching (when `--advice llm` is used)
- `detectors`: per-detector scores when using an ensemble
- `disclaimer`: usage warning

Example shape:

```json
{
  "model": "Oxidane/tmr-ai-text-detector",
  "sentences": [
    {
      "text": "It is crucial to leverage robust paradigms.",
      "ai_probability": 0.8712,
      "markers": ["crucial", "leverage"]
    }
  ],
  "overall_score": 0.8712,
  "burstiness": 0.0,
  "stats": {
    "mean": 0.8712,
    "std": 0.0,
    "min": 0.8712,
    "max": 0.8712
  },
  "sentence_count": 1,
  "advice": {
    "mode": "stats",
    "tips": [],
    "top_markers": [],
    "focus": []
  },
  "disclaimer": "This tool detects statistical similarity to modern AI-generated text. It is NOT definitive. Known false positives: formal prose, academic writing, legal text, non-native English, and pre-1900 literature. Use as one signal among many."
}
```

Scores in the example are illustrative; actual values depend on the selected detector and model version.

## Evaluation mode

The `eval` subcommand performs batch, metrics-only evaluation of candidate writing against prompts, with optional human preference correlation. It reads JSONL input, writes one JSONL record per sample, and finishes with an aggregate summary record.

**Key differences from single-text analysis:**

- No rewrite advice or detection-avoidance suggestions — output is metrics-only
- No LLM judge dependency — all metrics are deterministic
- Batch input via JSONL instead of stdin/file single-text mode
- Output is always JSONL (no rich terminal output)

### Usage

Basic evaluation:

```bash
uv run humanify eval --input tests/fixtures/eval_input.jsonl
```

Evaluation with preference labels for correlation:

```bash
uv run humanify eval --input tests/fixtures/eval_input.jsonl \\
    --preferences tests/fixtures/preference_labels.jsonl
```

Read input from stdin:

```bash
uv run humanify eval --input -
```

### Input schema

Each line is a JSON object with the following fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `id` | string | yes | Unique sample identifier |
| `prompt` | string | yes | Writing prompt or instruction |
| `candidate` | string | yes | Generated text to evaluate |
| `reference` | string | no | Optional reference or ground-truth text |

### Preference label schema

Optional `--preferences` file, one JSON object per line:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `id` | string | yes | Pair identifier |
| `prompt` | string | yes | Shared writing prompt |
| `candidate_a` | string | yes | First candidate text |
| `candidate_b` | string | yes | Second candidate text |
| `preferred` | string | yes | `"a"`, `"b"`, or `"tie"` |

### Per-sample output

Each output line is a JSON object with these fields:

- `schema_version`: always `1`
- `type`: `"sample"`
- `id`: the sample identifier from input
- `metrics`: object containing `raw` and `normalized` writing metrics (readability, lexical diversity, sentence variety, repetition, coherence rubric, style consistency, grammar heuristics, specificity, evidence)
- `component_scores`: four component scores used in the composite
  - `preference_rubric` — weighted blend of coherence, style, readability, lexical diversity, repetition, specificity, and evidence
  - `readability_style` — mean of readability, lexical diversity, style consistency, and grammar heuristics
  - `repetition_variety` — mean of repetition and sentence variety
  - `humanify_diagnostic` — `1.0 - mean_detector_probability`
- `composite_score`: final weighted score combining all components
- `weights_applied`: list of component names that contributed to the composite
- `detectors`: per-detector name, model ID, and probability stats
- `errors`: list of error strings (empty on success)

### Summary record

The final output line has `type: "summary"` and contains:

- `sample_count`, `valid_count`, `invalid_count`, `duplicate_id_count`
- `metrics`: aggregate mean, standard deviation, minimum, and maximum for each metric across all valid samples
- `weights`: composite weight values used
- `preference_correlation`: correlation between predicted and human preferences
  - `pair_count`, `non_tie_count`, `tie_count`
  - `agreement`: fraction of non-tie pairs where the composite-score prediction matches the human label (`null` when absent)
  - `pearson`, `spearman`: correlation coefficients for score deltas vs human deltas (`null` when undefined)

When `--preferences` is not provided, `preference_correlation` defaults to:
```json
{"pair_count": 0, "non_tie_count": 0, "tie_count": 0, "agreement": null, "pearson": null, "spearman": null}
```

### Composite formula

```
composite_score = 0.40 × preference_rubric
                + 0.25 × readability_style
                + 0.20 × repetition_variety
                + 0.15 × humanify_diagnostic
```

When `humanify_diagnostic` is unavailable (no detector), the remaining weights are renormalized.

### Metrics safety

All writing metrics are computed deterministically using regular expressions and standard Python libraries. No additional model downloads are required. The metrics module does not import `torch`, `transformers`, or any detector classes. Evaluation output never contains rewrite suggestions, detection-avoidance tactics, or LLM-generated advice.

## Feature analysis layers

Beyond a single score, `humanify` reports several explainable analysis layers, so you can see *why* a passage was flagged:

- **Era-aware vocabulary** — filter-based markers load from `humanify/data/ai_vocab.json` with per-term era tags and weights, so a `delve`/`tapestry` (2023-24 era) flag counts for less than a current `showcase`-type signal.
- **Pattern tells** — repeatable named findings: sentence-initial connectives (`Moreover`/`Furthermore`/`Additionally`/`In conclusion`), rule-of-three lists, `not … but …` constructions, `-ing`-suffix runs, and em-dash density per 1,000 words (plus the spaced-em-dash variant).
- **Burstiness bands** — sentence-length coefficient of variation is compared against reference bands (AI-like < 0.30, typical human 0.65–0.85) and given a named verdict, plus a soft "mechanical alternation" autocorrelation warning.
- **Structure audit** — document-level findings from `humanify/structure.py`: paragraph-length variance, hourglass (generic positive closing paragraph), and formulaic "Challenges and Future Prospects" closings.
- **Artifact layer** — `humanify/artifacts.py` flags text that looks post-processed: zero-width characters, Unicode homoglyphs, mechanical synonym cycling, or a high edit-magnitude + low lexical-variance combination.
- **Narrative checklist (opt-in)** — `--advice narrative` runs an LLM-assisted document-level checklist over the Story scope discourse dimensions; gated at **500+ words** and never run on short text.
- **Per-layer ensemble + confidence** — the ensemble reports each axis (classifier / zero-shot / lexical / structure / edit) separately with a confidence spread instead of a single opaque mean; `--domain academic|creative|social|legal` calibrates weights and emits domain-specific warnings.

## Robustness

`humanify` ships an evaluation harness under `benchmarks/` that measures each
layer separately instead of a single accuracy figure. Every script reports a
per-layer breakdown: lexical precision/recall, burstiness AUROC, per-classifier
AUROC / TPR@5%FPR, and a mean ensemble AUROC.

```bash
cd /home/pino/humanify
.venv/bin/python benchmarks/raid_eval.py --fetch --limit 500     # RAID (reference)
.venv/bin/python benchmarks/m4_eval.py   --fetch --limit 100     # M4 (OOD)
.venv/bin/python benchmarks/h2l_eval.py  --fetch --limit 100     # H2L (human + LLM rewrite)
```

**Numbers pending dataset download.** The RAID / M4 / H2L corpora are multi-GB and this harness deliberately never auto-downloads them — `--fetch` prints the manual `huggingface-cli download …` step, and the scripts exit early (code 2) until the split is present. Until then, no real AUROC / TPR figures are claimed in this README.

**The H2L caveat (the honest framing).** The dominant real-world failure mode
for this kind of text is a human author rewriting text with an LLM ("human-
polished"), not a raw LLM output. Benchmarks that only compare
human-vs-direct-LLM text consistently overstate robustness. Treat any robustness
claim as valid only against the exact distribution it was measured on; humanify
recommends framing claims around H2L-style text (KB doc 07,
`~/.omp/knowledge/ai-writing/07-frontier-papers-sweep.md`). Style-level claims
overstate robustness; structural layers are the durable signal.

## Red team

`redteam/` is the adversarial harness that pressure-tests detection from the
attacker side (for measurement and regression, not as an end-user feature). It
It is **not deprecated**: it is actively tested (four test files:
`test_redteam_loop.py`, `test_redteam_guided.py`, `test_redteam_verify.py`,
`test_redteam_transforms.py`) and wired into the repo. Only a
single internal component — the **StealthWriter detector wrapper** — is
deprecated; see the note below. The rest of the harness (sampling attacks,
paraphrase verification, guided optimization, detectors) is current. It
now includes:

- **Sampling attacks** — temperature, repetition-penalty, and top-p regeneration
  sweeps (`sample_temperature`, `sample_rep_penalty`, `sample_top_p`) plus a
  chained `combo_sample_all`. These target the too-smooth statistical signature
  that zero-shot detectors depend on (KB 03 §3).
- **Iterative paraphrase with verification** — `paraphrase_verify` generates,
  scores with the local detector, and in later rounds rewrites *only* the chunks
  that still score high, capped at 3 rounds, keeping the best candidate (KB 03 §5).
- **Guided detector optimization** — `guided_optimize` is a SICO-style basic
  version that uses 3 style prompts, keeps the best-scoring variant per round and
  re-applies the best prompt with a rotated seed (KB 03 §1.1), rather than blind
  single-shot rewrites.

The **StealthWriter detector wrapper is deprecated**: its API endpoint is
guessed/unverified, and KB 03 finds GPTZero flags StealthWriter output at 100% AI,
so it is not an informative detection signal. The wrapper is kept only for
backward compatibility (constructing it emits a `DeprecationWarning`); leave it out
of new detector sets.

The loop keeps its original CLI (`python redteam/loop.py --input … --detectors
humanify,gptzero …`). No test depends on Ollama being live — the Ollama call layer
and the detector are injectable/mocked in the test suite.

## Writing advice

### Writing quality metrics

`humanify` separately computes deterministic writing-quality signals
(`humanify/writing_metrics.py` → `compute_writing_metrics`): grammar,
specificity, evidence, readability, lexical diversity, sentence variety,
repetition, coherence, and style consistency. These appear in the terminal
summary and under `writing_quality` in JSON output, alongside concrete
strengths and weaknesses generated by `generate_quality_advice`. The metrics
are rule-based — no additional model downloads.

Two flags tune this writing-quality analysis:

```bash
uv run humanify --prompt "Write a policy memo" --file examples/mixed-writing-sample.txt
uv run humanify --style academic --file examples/mixed-writing-sample.txt
```

- `--prompt` supplies the original writing prompt; it improves keyword-coverage and coherence scoring.
- `--style {general,academic}` shifts the advice thresholds to account for formal academic prose (default: `general`).

### Statistics-based advice

The default advice mode is deterministic and generated from sentence scores, markers, burstiness, and repeated sentence openers:

```bash
uv run humanify --advice stats --file examples/mixed-writing-sample.txt
```

This mode is fast, does not load an additional language model, and is appropriate for repeatable automation.

### LLM-generated advice

LLM advice uses a causal language model to produce concise writing-coach feedback:

```bash
uv run humanify --advice llm --file examples/mixed-writing-sample.txt
```

If the selected detector already loaded a compatible causal LM, `humanify` reuses it. Otherwise it loads `Qwen/Qwen3.5-0.8B-Base`. The prompt requests structural advice rather than word-by-word substitutions. Reasoning tags such as `<think>` are suppressed when the tokenizer supports it and stripped from output as a safety net.

## Model overrides

Use `--model-id` to replace a detector's default Hugging Face model.

Single-model detectors:

```bash
uv run humanify --model tmr --model-id Oxidane/tmr-ai-text-detector --file examples/mixed-writing-sample.txt
uv run humanify --model fast-detect --model-id Qwen/Qwen3.5-0.8B-Base --file examples/mixed-writing-sample.txt
```

`binoculars` accepts either one model ID for both roles or an `observer,performer` pair:

```bash
uv run humanify --model binoculars --model-id Qwen/Qwen3.5-0.8B-Base --file examples/mixed-writing-sample.txt
uv run humanify --model binoculars --model-id observer-model-id,performer-model-id --file examples/mixed-writing-sample.txt
```

## Thresholds

`--threshold` controls which sentences are treated as high-priority for advice and marker display. It does not change the raw model probabilities.

```bash
uv run humanify --threshold 0.7 --file examples/mixed-writing-sample.txt
```

Use a lower threshold for a broader review set. Use a higher threshold when you only want the strongest signals.

## Performance

`--batch-size` controls how many sentences are scored per inference batch:

```bash
uv run humanify --batch-size 16 --file examples/mixed-writing-sample.txt
```

Guidance:

- CPU is the default because some systems report CUDA as available while the
  first real CUDA driver call hangs.
- Use `--device cuda` only when your CUDA runtime is known to be healthy.
- Increase batch size on GPUs with available memory.
- Decrease batch size if inference runs out of memory.
- `binoculars` and LLM advice require more memory than classifier-only detection.

## Hugging Face authentication

Some model downloads may require authentication or acceptance of model terms on Hugging Face. Set `HF_TOKEN` before running `humanify` when needed:

```bash
export HF_TOKEN=hf_...
uv run humanify --model fast-detect --file examples/mixed-writing-sample.txt
```

The token is read from the environment and passed to Hugging Face Hub login by detectors that need causal language models.

## Privacy and data handling

`humanify` reads text from a local file or standard input and runs local model inference. The text being analyzed is not intentionally sent to an external API by this package.

Operational caveats:

- Model files are downloaded from Hugging Face on first use.
- Hugging Face authentication, if configured, contacts Hugging Face to authenticate and download models.
- The selected model's own license and terms apply.
- Do not paste secrets, regulated data, or private documents into shell history. Prefer `--file` over shell history for sensitive text.

## Exit behavior

`humanify` exits with an error when:

- No input text is provided
- The input does not contain any detectable sentences
- An unknown detector name is requested
- A model cannot be downloaded or loaded

## Development

Install dependencies:

```bash
uv sync
```

Run the test suite (requires `uv sync` to have installed the `dev` group):

```bash
uv run python -m pytest tests/
```

Compile-check the package:

```bash
uv run python -m compileall humanify tests
```

### Cross-detector benchmark

`benchmarks/spectrum_eval.py` is a reproducible cross-detector benchmark: it scores a curated spectrum of writing (human voice → light AI polish → heavy AI rewrite → fully AI-generated) with every registered detector and reports AUROC, TPR@5%FPR, and macro-F1 per detector, plus a simple ensemble. Results are checked into `benchmarks/spectrum_bench.json`. Run it with:

```bash
uv run python benchmarks/spectrum_eval.py
```

Pass `--detectors` to limit the set (e.g. `--detectors tmr,fast-detect`) and `--out` to write results elsewhere; the full default set downloads the large detector weights on first run.

Run the CLI locally:

```bash
uv run humanify --file examples/mixed-writing-sample.txt
```

### Adding a detector

Detectors live in `humanify/detectors/` and implement `Detector` from `humanify/detectors/base.py`.

A detector must provide:

- `load(device: str, fp16: bool = False) -> None`
- `score(sentences: list[str], batch_size: int = 32) -> list[float]`
- `name`
- `model_id`

Register the detector in `humanify/detectors/__init__.py` so it becomes available through `--model`.

## License

`humanify` is released under the MIT License. See [`LICENSE`](LICENSE).
