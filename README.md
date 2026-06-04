# humanify

`humanify` is a command-line text analysis tool that scores writing for statistical similarity to modern AI-generated text. It splits input into sentences, runs one or more transformer-based detectors, highlights high-scoring passages, reports repeated AI-like markers, computes burstiness, and returns either rich terminal output or machine-readable JSON.

Use it as a writing-quality and risk-review signal, not as proof of authorship.

## Features

- Per-sentence AI-likeness probabilities.
- Overall document score and score distribution statistics.
- Burstiness measurement for sentence-level variation.
- Marker detection for common AI-style filler phrases.
- Rich terminal output with color-coded sentence diagnostics.
- JSON output for automation, CI, dashboards, and pipelines.
- Multiple detector backends: classifier-based and zero-shot methods.
- Detector ensembles by comma-separating model names.
- Optional writing advice: deterministic statistics-based tips or LLM-generated coaching.
- CPU and CUDA support via PyTorch.

## Important limitations

AI-text detection is probabilistic. `humanify` detects statistical similarity to text produced by modern language models; it does not prove that a person or an AI wrote a document.

Known sources of false positives include:

- Formal business writing.
- Academic prose.
- Legal and policy text.
- Non-native English.
- Highly edited or templated writing.
- Older public-domain writing styles.
- Text that is too short to provide enough signal.

Use the output to guide review and rewriting. Do not use it as the sole basis for academic, employment, compliance, or enforcement decisions.

## Installation

### From source

```bash
git clone <repo-url>
cd humanify
uv sync
```

Run the CLI through `uv`:

```bash
uv run humanify --help
```

### As an installed local tool

```bash
uv tool install .
humanify --help
```

### Requirements

- Python 3.11 or newer.
- PyTorch 2.0 or newer.
- `transformers` 4.30 or newer.
- `rich` 13.0 or newer.
- `numpy` 1.24 or newer.
- Internet access on first run to download Hugging Face models.
- Optional: CUDA-capable GPU for faster inference.

Model downloads are cached by Hugging Face after the first run.

## Quick start

Analyze a file:

```bash
uv run humanify --file essay.txt
```

Analyze text from standard input:

```bash
echo "It is crucial to leverage robust paradigms." | uv run humanify
```

Emit JSON:

```bash
uv run humanify --json --file essay.txt
```

Use a heavier detector:

```bash
uv run humanify --model desklib --file essay.txt
```

Run an ensemble:

```bash
uv run humanify --model tmr,fast-detect --file essay.txt
```

Focus advice on sentences above a custom threshold:

```bash
uv run humanify --threshold 0.3 --file essay.txt
```

Generate LLM-based writing advice:

```bash
uv run humanify --advice llm --file essay.txt
```

## CLI reference

```text
usage: humanify [-h] [--file FILE] [--json] [--model MODEL]
                [--model-id MODEL_ID] [--threshold THRESHOLD]
                [--batch-size BATCH_SIZE] [--advice {stats,llm}]

Analyze text for AI-like patterns using transformer models.

options:
  -h, --help            show this help message and exit
  --file, -f FILE       Input file path (default: stdin)
  --json, -j            Output machine-readable JSON
  --model, -m MODEL     Detection method (default: tmr). Comma-separate for
                        ensemble: tmr,fast-detect. Available: tmr, desklib,
                        fast-detect, binoculars
  --model-id MODEL_ID   Override default model ID. For binoculars:
                        'observer,performer'
  --threshold, -t THRESHOLD
                        AI probability threshold for advice focus (default:
                        0.5)
  --batch-size, -b BATCH_SIZE
                        Inference batch size (default: 32)
  --advice, -a {stats,llm}
                        Advice mode: stats (rule-based) or llm (AI-generated
                        via causal LM)
```

## Detectors

| Detector | Default model | Method | Notes |
| --- | --- | --- | --- |
| `tmr` | `Oxidane/tmr-ai-text-detector` | Text-classification pipeline | Default detector; 125M RoBERTa-base model. |
| `desklib` | `desklib/ai-text-detector-v1.01` | DeBERTa-v3-large classifier | Heavier detector with strong RAID leaderboard performance. |
| `fast-detect` | `Qwen/Qwen3.5-0.8B-Base` | Fast-DetectGPT-style log-probability curvature | Zero-shot causal-LM method. |
| `binoculars` | `Qwen/Qwen3.5-0.8B-Base,Qwen/Qwen3.5-0.8B-Base` | Binoculars-style perplexity ratio | Loads two causal LMs; highest memory use among default options. |

### Choosing a detector

- Use `tmr` for the fastest default experience and good accuracy-per-parameter.
- Use `desklib` when you want a stronger classifier and can afford the larger model.
- Use `fast-detect` when you want a zero-shot detector based on causal-LM likelihood behavior.
- Use `binoculars` when you want a perplexity-ratio detector and have enough memory for two LMs.
- Use an ensemble when disagreement between methods is useful for review:

```bash
uv run humanify --model tmr,desklib,fast-detect --file essay.txt
```

## Model overrides

Use `--model-id` to replace a detector's default Hugging Face model.

Single-model detectors:

```bash
uv run humanify --model tmr --model-id Oxidane/tmr-ai-text-detector --file essay.txt
uv run humanify --model fast-detect --model-id Qwen/Qwen3.5-0.8B-Base --file essay.txt
```

`binoculars` accepts either one model ID for both roles or `observer,performer`:

```bash
uv run humanify --model binoculars --model-id Qwen/Qwen3.5-0.8B-Base --file essay.txt
uv run humanify --model binoculars --model-id observer-model-id,performer-model-id --file essay.txt
```

## Output

### Terminal output

The default rich output shows every sentence with a score and a visual status:

- Green check: below 30% AI probability.
- Yellow warning: 30-70% AI probability.
- Red cross: above 70% AI probability.

For sentences above `--threshold`, detected markers are shown below the sentence. The summary includes:

- Overall AI probability.
- Burstiness, reported as standard deviation across sentence scores.
- Sentence count.
- Minimum and maximum sentence scores.
- Model or ensemble label.
- Writing advice.
- False-positive disclaimer.

### JSON output

`--json` writes structured output suitable for automation:

```bash
uv run humanify --json --file essay.txt
```

Top-level fields include:

- `model`: model ID or ensemble label.
- `sentences`: per-sentence text, probability, and markers.
- `overall_score`: mean sentence probability.
- `burstiness`: standard deviation of sentence probabilities.
- `stats`: mean, standard deviation, minimum, and maximum.
- `sentence_count`: number of analyzed sentences.
- `advice`: deterministic or LLM-generated advice payload.
- `detectors`: per-detector scores when using an ensemble.
- `disclaimer`: usage warning.

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

## Writing advice modes

### Statistics-based advice

The default advice mode is deterministic and generated from sentence scores, markers, burstiness, and repeated sentence openers:

```bash
uv run humanify --advice stats --file essay.txt
```

This mode is fast, does not load an additional language model, and is appropriate for repeatable automation.

### LLM-generated advice

LLM advice uses a causal language model to produce concise writing-coach feedback:

```bash
uv run humanify --advice llm --file essay.txt
```

If the selected detector already loaded a compatible causal LM, `humanify` reuses it. Otherwise it loads `Qwen/Qwen3.5-0.8B-Base`. The prompt requests structural advice rather than word-by-word substitutions. Reasoning tags such as `<think>` are suppressed when the tokenizer supports it and stripped from output as a safety net.

## Thresholds

`--threshold` controls which sentences are treated as high-priority for advice and marker display. It does not change the raw model probabilities.

```bash
uv run humanify --threshold 0.7 --file essay.txt
```

Use a lower threshold when you want a broader review set. Use a higher threshold when you only want the strongest signals.

## Batch size and performance

`--batch-size` controls how many sentences are scored per inference batch:

```bash
uv run humanify --batch-size 16 --file essay.txt
```

Guidance:

- Increase batch size on GPUs with available memory.
- Decrease batch size if inference runs out of memory.
- `binoculars` and LLM advice require more memory than classifier-only detection.
- CPU inference works but can be slow for larger models.

## Hugging Face authentication

Some model downloads may require authentication or acceptance of model terms on Hugging Face. Set `HF_TOKEN` before running `humanify` when needed:

```bash
export HF_TOKEN=hf_...
uv run humanify --model fast-detect --file essay.txt
```

The token is read from the environment and passed to Hugging Face Hub login by detectors that need causal language models.

## Privacy and data handling

`humanify` reads text from a local file or standard input and runs local model inference. The text being analyzed is not intentionally sent to an external API by this package.

Operational caveats:

- Model files are downloaded from Hugging Face on first use.
- Hugging Face authentication, if configured, contacts Hugging Face to authenticate and download models.
- The selected model's own license and terms apply.
- Do not paste secrets, regulated data, or private documents into shell history. Prefer `--file` over `echo` for sensitive text.

## Exit behavior

`humanify` exits with an error when:

- No input text is provided.
- The input does not contain any detectable sentences.
- An unknown detector name is requested.
- A model cannot be downloaded or loaded.

## Development

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run python -m unittest discover tests
```

Compile-check the package:

```bash
uv run python -m compileall humanify tests
```

Run the CLI locally:

```bash
uv run humanify --file test_input.txt
```

## Adding a detector

Detectors live in `humanify/detectors/` and implement `Detector` from `humanify/detectors/base.py`.

A detector must provide:

- `load(device: str, fp16: bool = False) -> None`
- `score(sentences: list[str], batch_size: int = 32) -> list[float]`
- `name`
- `model_id`

Register the detector in `humanify/detectors/__init__.py` so it becomes available through `--model`.

## Responsible use

Before acting on a high score, review the actual sentences, context, model choice, and reason for the score. A high score can mean the text is formulaic, polished, templated, domain-specific, or similar to AI-generated writing. It is not a reliable standalone authorship claim.
