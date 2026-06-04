# humanify

CLI tool that scores text per-sentence for AI-likeness using transformer models. Computes burstiness, identifies AI markers, and renders color-coded output.

## Install

```bash
uv sync
```

## Usage

```bash
# Analyze a file
uv run humanify --file essay.txt

# Pipe from stdin
echo "It is crucial to leverage robust paradigms." | uv run humanify

# Machine-readable JSON output
uv run humanify --json --file essay.txt

# Use the more accurate (but heavier) desklib model
uv run humanify --model desklib --file essay.txt

# Adjust AI marker threshold
uv run humanify --threshold 0.3 --file essay.txt
```

## Models

| Model | Params | RAID AUROC | Notes |
|---|---|---|---|
| `tmr` (default) | 125M | 99.28% | Best accuracy-per-param |
| `desklib` | 434M | ~99.5% | #1 on RAID, heavier |

## Output

### Terminal (rich)

Color-coded sentences with inline AI marker annotations:
- ✓ Green (<30% AI probability)
- ⚠ Yellow (30–70%)
- ✗ Red (>70%) — markers listed below

### JSON (`--json`)

Per-sentence probabilities, markers, overall score, burstiness, and stats.

## How It Works

1. Split input into sentences (regex-based)
2. Score each sentence with the selected transformer model
3. Compute burstiness (std dev of per-sentence scores)
4. Scan for ~50 hardcoded AI filler phrases
5. Render results with color-coded rich output or JSON
