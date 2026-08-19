# Task: Resolve the humanify merge conflict (master ← origin/master) correctly, keeping BOTH feature sets

You are in `/home/pino/humanify`. A `git merge origin/master --no-edit` is in progress and has conflicts. Your job is to resolve them so the result **keeps both** the local branch's features AND the remote's new feature, then pass verification.

## Background — the two sides

**Local branch (HEAD = your 11 commits):** added detector stack (EditLens, RADAR, Binoculars, fast-detect), narrative checklist + StoryScope, artifacts layer, calibration, redteam harness, robustness evals (RAID/M4/H2L), spectrum benchmark, CI workflow, .gitignore hygiene, README updates. It uses these imports/symbols that MUST keep working:
- `humanify/cli.py`: imports `find_markers_weighted`, `_AI_WEIGHTS` from `.analysis`; `detect_artifacts` from `.artifacts`; `audit_structure` from `.structure`; narrative-checklist gating via `NARRATIVE_MIN_WORDS`, `run_narrative_checklist`, `NARRATIVE_MAX_PROMPT_TOKENS`, advice-llm helpers; calls `output_json(sentences, probs, markers, stats, args.threshold, model_label, **_shared)` and `output_rich(...)`.
- `humanify/output.py`: local side added artifact/structure/narrative output fields.
- `humanify/writing_metrics.py`: local side added/extended metrics (sentence-level variation, signal flags, etc.) — check what local-specific functions exist.
- `tests/test_cli.py`, `tests/test_writing_metrics.py`: local tests for the above.

**Remote (origin/master = commit 8cb1726 "feat: integrate writing quality detection alongside AI detection"):** adds a deterministic writing-quality pipeline:
- `humanify/cli.py`: new import `generate_quality_advice` from `.analysis`; new call `compute_writing_metrics(args.prompt or raw, raw, ...)` returning `quality_metrics`; passes `quality_metrics`, `paragraph_analysis`, `top_worst` into `output_json`/`output_rich`; CLI args may include `--prompt`.
- `humanify/analysis.py`: added `generate_quality_advice` (154 lines changed)
- `humanify/writing_metrics.py`: big additions (582 lines) — `compute_writing_metrics` and related
- `humanify/advice_llm.py` (new), `humanify/eval_scoring.py`, `humanify/evaluator.py` updated
- `humanify/output.py`: added `paragraph_analysis`, `top_worst` output handling
- New test file `tests/test_quality_advice.py` (305 lines)
- README updated for writing-quality detection

## The merge conflict files (5)
`humanify/cli.py`, `humanify/output.py`, `humanify/writing_metrics.py`, `tests/test_cli.py`, `tests/test_writing_metrics.py`

## Resolution rules — CRITICAL
1. **Keep BOTH sides' functionality.** The merged result must support:
   - The remote's `compute_writing_metrics` / `generate_quality_advice` / `quality_metrics` / `paragraph_analysis` / `top_worst` pipeline, AND
   - The local `find_markers_weighted` / `_AI_WEIGHTS` / `detect_artifacts` / `audit_structure` / narrative-checklist / StoryScope features.
   Do NOT drop either side to make it compile. If both sides added different things to the same function, combine them in the natural order (e.g., compute quality metrics AND run narrative checklist).
2. **Imports:** merge the import blocks. Keep local's `from .artifacts import detect_artifacts`, `from .structure import audit_structure`, `find_markers_weighted`, `_AI_WEIGHTS`, AND remote's `generate_quality_advice`. Watch for duplicate imports (e.g., `os`, `compute_writing_metrics`) — dedupe cleanly.
3. **Output:** both sides changed `output_json`/`output_rich` signatures. The merged version must accept ALL parameters: local's artifact/structure/narrative fields AND remote's `paragraph_analysis`/`top_worst`/`quality_metrics`. Check the actual definitions in `humanify/output.py` (both `output_json` and `output_rich`) and reconcile — prefer keyword args with defaults so both call sites work.
4. **writing_metrics.py:** merge the two bodies — remote's `compute_writing_metrics` additions + local's metric extensions. Keep every function both sides define; if a function name collides, check whether it's the same function evolved (then merge bodies) or two different functions (then rename or keep both with distinct names). Do NOT delete a function the tests reference.
5. **Tests:** merge `tests/test_cli.py` and `tests/test_writing_metrics.py` so BOTH sides' tests exist and pass. The remote added `tests/test_quality_advice.py` — it should already be staged/added by the merge; verify it's present.
6. After resolving all 5 files:
   - `git add` the resolved files
   - `git commit` (message: `merge: integrate writing-quality detection (origin/master) with local detector stack`)
   - DO NOT push. DO NOT rebase. DO NOT `git reset`. Just complete the merge commit.
7. **Verify** (run all, record exact output):
   - `source .venv/bin/activate && python -m pytest tests/ -q` → must be **260 + new tests passing** (the remote added ~1,300 lines of tests; expect total > 260, all green). If the remote's tests fail for a real reason (not a merge artifact), fix minimally.
   - `python -m compileall humanify tests` → clean
   - `uv run humanify --help` → exits 0, shows both old and new args (e.g., `--prompt` if remote added it, `--advice`, `--model`, `--threshold`)
   - `uv run python benchmarks/spectrum_eval.py --help` → exits 0
   - `git status` → clean tree, merge commit present

## Environment
- venv: `/home/pino/humanify/.venv` (activate before pytest)
- Use `uv` / `uv run` for CLI/build commands. `uv sync --frozen` may be needed if deps changed.
- You have full shell access. Work autonomously, no questions. If something is ambiguous, prefer the resolution that keeps both features working and document your choice in the final report.

## Deliverables / proof
- Merge commit on `master` (not pushed) with all 5 files resolved
- Full test suite green (count + any failures)
- compileall + CLI + spectrum --help green
- Final report: what each conflict was, how you resolved it, the verification outputs, and any compromises made
