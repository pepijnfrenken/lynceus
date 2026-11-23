# Lynceus Evaluation Fit

## TL;DR
> **Summary**: Add a deterministic `lynceus eval` subsystem for batch JSONL evaluation, writing-quality metrics, human preference correlation, and a composite score while preserving the existing single-text detector CLI.
> **Deliverables**:
> - `lynceus eval` subcommand for JSONL input records `{id, prompt, candidate, reference?}`.
> - Metrics-only per-sample JSONL output plus final aggregate summary line.
> - Deterministic writing metrics, composite scoring, and preference-correlation support.
> - Tiny repo fixtures, external preference-label support, unittest coverage, and README docs.
> **Effort**: Large
> **Parallel**: YES - 6 waves, with Task 2 and Task 3 parallelized
> **Critical Path**: Task 1 + Task 3 → Task 4 → Task 5 → Task 6 → Task 7

## Context
### Original Request
Add batch JSONL evaluator, non-detector writing metrics, human preference correlation, evaluation-only mode, and composite score so `lynceus` fits better as an evaluation tool.

### Interview Summary
- Evaluator interface: new `lynceus eval` subcommand.
- Output contract: one JSON object per sample, then a final aggregate summary object as the last JSONL line.
- Test strategy: tests-after using the existing `unittest` suite.
- Composite judge/rubric: deterministic only; no required LLM dependency.
- Preference labels: include tiny repo fixture for tests and support external user-supplied JSONL.

### Metis Review (gaps addressed)
- Exact schemas, score directions, null handling, invalid input behavior, duplicate IDs, and preference label semantics are specified below.
- Guardrails prohibit `generate_advice()`, `generate_llm_advice()`, rewrite advice, or detection-evasion text from eval output.
- All composite components are normalized to `0.0-1.0`, higher-is-better, before weighting.
- Undefined correlations and zero-denominator metrics serialize as JSON `null`, never `NaN`/`Infinity`.

## Work Objectives
### Core Objective
Implement a deterministic, metrics-only evaluation mode that scores batches of candidate writing and correlates metrics with human pairwise preferences without providing rewrite advice or detection-avoidance suggestions.

### Deliverables
- New modules for JSONL eval I/O, deterministic writing metrics, composite scoring, aggregation, and preference correlation.
- Refactored CLI with `lynceus eval --input INPUT.jsonl [--preferences PREFS.jsonl]` while preserving existing `lynceus --file ...` behavior.
- Test fixtures under `tests/fixtures/` for valid eval input, invalid eval input, duplicate IDs, and preference labels.
- README section documenting eval mode, schemas, examples, and safety boundaries.

### Definition of Done (verifiable conditions with commands)
- `uv run python -m unittest discover tests` exits `0`.
- `uv run python -m compileall lynceus tests` exits `0`.
- `uv run lynceus --help` still documents the existing single-text options.
- `uv run lynceus eval --help` documents JSONL eval input, preferences, model, device, and batch-size options.
- Eval output contains no keys or text containing `advice`, `rewrite`, `avoid detection`, `bypass`, or `evade`.

### Must Have
- Input sample schema: `{"id": string, "prompt": string, "candidate": string, "reference"?: string}`.
- Preference schema: `{"id": string, "prompt": string, "candidate_a": string, "candidate_b": string, "preferred": "a"|"b"|"tie"}`.
- Per-sample output schema:
  - `schema_version: 1`
  - `type: "sample"`
  - `id`
  - `metrics` with raw and normalized writing metrics
  - `component_scores` with `preference_rubric`, `readability_style`, `repetition_variety`, `lynceus_diagnostic`
  - `composite_score`
  - `detectors` with detector/model/probability diagnostics only
  - `errors: []`
- Final summary schema:
  - `schema_version: 1`
  - `type: "summary"`
  - `sample_count`, `valid_count`, `invalid_count`, `duplicate_id_count`
  - aggregate metric `mean`, `std`, `min`, `max` where applicable
  - `weights`
  - `preference_correlation` with `pair_count`, `non_tie_count`, `tie_count`, `agreement`, `pearson`, `spearman`
- Composite formula: `0.40 * preference_rubric + 0.25 * readability_style + 0.20 * repetition_variety + 0.15 * lynceus_diagnostic`.
- Diagnostic conversion: `lynceus_diagnostic = 1.0 - normalized_ai_probability`.
- Undefined numeric values: JSON `null`.
- Blank JSONL lines: ignored.
- Malformed JSON, missing required fields, empty required string fields, and duplicate sample IDs: nonzero exit with stderr error; do not emit partial success summary.
- Missing optional `reference`: reference-dependent raw metrics are `null`; non-reference metrics and composite still run.
- When `--preferences` is absent, summary still includes `preference_correlation` as `{"pair_count": 0, "non_tie_count": 0, "tie_count": 0, "agreement": null, "pearson": null, "spearman": null}`.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- Do not add required LLM dependencies.
- Do not call `analysis.generate_advice()` from `lynceus/analysis.py:132-173` in eval mode.
- Do not call `generate_llm_advice()` from `lynceus/advice_llm.py:96-130` in eval mode.
- Do not reuse `output_json()` from `lynceus/output.py:47-91` for eval output because it always includes `advice` at `lynceus/output.py:66-72`.
- Do not change detector names or detector registration in `lynceus/detectors/__init__.py:9-21`.
- Do not change existing single-run CLI behavior documented in `README.md:124-154`.
- Do not add CI, coverage, dashboards, or heavy NLP packages in this plan.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-after with Python `unittest`.
- QA policy: Every task has agent-executed scenarios.
- Evidence: `.omo/evidence/task-{N}-{slug}.{ext}`.
- Existing test command reference: `README.md:356-365`.

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: Task 1 foundation CLI/scoring refactor.
Wave 2: Tasks 2 and 3 in parallel: schemas/fixtures and writing metrics.
Wave 3: Task 4 scoring/correlation built against Task 3 metric contract.
Wave 4: Task 5 batch evaluator engine.
Wave 5: Task 6 CLI command wiring.
Wave 6: Task 7 docs and final user-facing validation.

### Dependency Matrix (full, all tasks)
| Task | Blocks | Blocked By |
| --- | --- | --- |
| 1 | 5, 6 | None |
| 2 | 5, 6 | None |
| 3 | 4, 5 | None |
| 4 | 5 | 3 |
| 5 | 6, 7 | 1, 2, 3, 4 |
| 6 | 7 | 1, 2, 5 |
| 7 | Final Verification | 5, 6 |

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 1 task → `unspecified-high`
- Wave 2 → 2 tasks → `unspecified-high`, `deep`
- Wave 3 → 1 task → `deep`
- Wave 4 → 1 task → `unspecified-high`
- Wave 5 → 1 task → `unspecified-high`
- Wave 6 → 1 task → `writing`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Refactor CLI scoring into reusable single-text analysis without changing legacy behavior

  **What to do**:
  - In `lynceus/cli.py`, split current `main()` into:
    - `_main_single(argv: list[str] | None) -> int` containing the existing single-input behavior from `lynceus/cli.py:58-188`.
    - `_load_detectors(model_names: list[str], *, model_id: str | None, device_name: str, suppress_stdout: bool) -> list[dict]` containing detector validation/loading currently represented by `lynceus/cli.py:101-123`; each dict must include `name`, `detector`, and `model_id`.
    - `_score_text(raw: str, *, loaded_detectors: list[dict], batch_size: int) -> dict` containing sentence preparation, detector scoring, score expansion, marker extraction, and ensemble stats currently represented by `lynceus/cli.py:84-154`.
    - `main(argv: list[str] | None = None) -> None` dispatching to eval only when the first argument is exactly `eval`; otherwise preserve legacy parsing.
  - Keep `_resolve_device()` from `lynceus/cli.py:28-34` and `_expand_unit_scores()` from `lynceus/cli.py:36-55` as shared helpers.
  - Preserve the existing single-run `os._exit(0)` workaround from `lynceus/cli.py:190-196` only after successful legacy single-text output; eval mode must return through normal exit-code handling so tests can call it safely.
  - `_load_detectors()` must be called once by batch eval and once per legacy single-text CLI invocation; `_score_text()` must never load models.
  - `_score_text()` must return sentences, scoring units, per-sentence probabilities, markers, stats, model label, and per-detector summaries. It must not generate advice.
  - Extend `tests/test_cli.py` with fake detector coverage for `_score_text()` and dispatch behavior without loading transformer models.
  **Must NOT do**:
  - Do not change root options documented in `README.md:124-154`.
  - Do not call `output_json()` or `output_rich()` from `_score_text()`.
  - Do not make `lynceus eval` share the legacy `os._exit(0)` path.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: shared CLI refactor with regression risk.
  - Skills: [] - No specialized skill needed.
  - Omitted: [`security-research`] - No vulnerability audit requested.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [5, 6] | Blocked By: []

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `lynceus/cli.py:58-82` - current argparse setup and legacy flags.
  - Pattern: `lynceus/cli.py:84-154` - input preparation, detector loading/scoring, ensemble stats.
  - Pattern: `lynceus/cli.py:156-178` - advice generation path that eval must not enter.
  - Pattern: `lynceus/cli.py:180-196` - output and `os._exit(0)` behavior to preserve for legacy path only.
  - Test: `tests/test_cli.py:7-29` - current unittest style and helper tests.
  - API/Type: `lynceus/detectors/base.py:6-19` - detector interface contract.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `uv run python -m unittest tests.test_cli` exits `0`.
  - [ ] `uv run python -m unittest discover tests` exits `0`.
  - [ ] `uv run python -m compileall lynceus tests` exits `0`.
  - [ ] A unit test proves `_score_text()` does not call `analysis.generate_advice()` or `advice_llm.generate_llm_advice()`.
  - [ ] A unit test proves legacy `_main_single()` still rejects empty input with exit code `1` and stderr text matching `no input text provided`.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Reusable scoring helper returns detector metrics with fake detector
    Tool: Bash
    Steps: Run `uv run python -m unittest tests.test_cli`
    Expected: Command exits 0; fake detector test asserts probabilities expand to all display sentences and returned stats include mean/std/min/max.
    Evidence: .omo/evidence/task-1-cli-refactor.txt

  Scenario: Eval path remains advice-free during scoring
    Tool: Bash
    Steps: Run `uv run python -m unittest tests.test_cli`
    Expected: Command exits 0; test patches advice functions to raise and `_score_text()` still succeeds.
    Evidence: .omo/evidence/task-1-cli-refactor-no-advice.txt
  ```

  **Commit**: YES | Message: `refactor(cli): expose reusable scoring path` | Files: [`lynceus/cli.py`, `tests/test_cli.py`]

- [ ] 2. Add JSONL evaluation schemas, validation, and fixtures

  **What to do**:
  - Add `lynceus/eval_io.py` with deterministic schema helpers:
    - `SCHEMA_VERSION = 1`.
    - `EvalSample` and `PreferencePair` dataclasses or typed dictionaries.
    - `parse_eval_samples_jsonl(text: str, source: str) -> list[EvalSample]` and `load_eval_samples(path: str) -> list[EvalSample]`.
    - `parse_preference_pairs_jsonl(text: str, source: str) -> list[PreferencePair]` and `load_preference_pairs(path: str) -> list[PreferencePair]`.
    - `safe_float(value: float | None) -> float | None` converting `NaN`/`Infinity` to `None`.
    - `json_dumps_record(record: dict) -> str` using `json.dumps(..., sort_keys=True, ensure_ascii=False)` for stable output.
  - Validation rules:
    - Eval sample requires non-empty string `id`, `prompt`, `candidate`; optional `reference` must be string when present.
    - Preference pair requires non-empty string `id`, `prompt`, `candidate_a`, `candidate_b`; `preferred` must be `"a"`, `"b"`, or `"tie"`.
    - Blank JSONL lines are ignored.
    - Malformed JSON, missing fields, wrong types, empty required strings, and duplicate eval sample IDs raise a deterministic validation error with line number when applicable.
  - Add fixtures:
    - `tests/fixtures/eval_input.jsonl` with at least 3 valid samples: one with `reference`, one without `reference`, one Unicode/non-ASCII sample.
    - `tests/fixtures/eval_invalid.jsonl` with malformed JSON or missing field for negative tests.
    - `tests/fixtures/eval_duplicate_ids.jsonl` with duplicate `id`.
    - `tests/fixtures/preference_labels.jsonl` with at least 3 pairs: one `a`, one `b`, one `tie`.
  - Add `tests/test_eval_io.py` using `unittest`.
  **Must NOT do**:
  - Do not silently repair malformed rows.
  - Do not allow duplicate sample IDs.
  - Do not serialize `NaN`, `Infinity`, or `-Infinity`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: schema contract is foundational for all eval behavior.
  - Skills: [] - No specialized skill needed.
  - Omitted: [`writing`] - This is implementation/schema work, not prose.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [5, 6] | Blocked By: []

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `lynceus/output.py:47-91` - existing JSON assembly style, but do not reuse because it injects advice.
  - Pattern: `tests/test_analysis.py:1-15` - fixture reading with `Path` and unittest style.
  - Pattern: `tests/test_advice_llm.py:19-21` - `subTest` style for multiple cases.
  - External: `https://jsonlines.org/` - JSON Lines convention: one valid JSON value per line.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `uv run python -m unittest tests.test_eval_io` exits `0`.
  - [ ] `uv run python -m unittest discover tests` exits `0`.
  - [ ] Tests assert blank lines are ignored and output order matches input order.
  - [ ] Tests assert duplicate IDs and invalid rows raise deterministic validation errors.
  - [ ] Tests assert `safe_float(float("nan")) is None` and JSON output contains `null`, not `NaN`.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Valid eval and preference fixtures load deterministically
    Tool: Bash
    Steps: Run `uv run python -m unittest tests.test_eval_io`
    Expected: Command exits 0; tests assert 3 eval samples and 3 preference labels load in fixture order.
    Evidence: .omo/evidence/task-2-eval-io.txt

  Scenario: Invalid JSONL fails closed
    Tool: Bash
    Steps: Run `uv run python -m unittest tests.test_eval_io`
    Expected: Command exits 0; negative tests assert malformed row and duplicate ID raise validation errors with line/id context.
    Evidence: .omo/evidence/task-2-eval-io-errors.txt
  ```

  **Commit**: YES | Message: `feat(eval): add jsonl schema validation` | Files: [`lynceus/eval_io.py`, `tests/test_eval_io.py`, `tests/fixtures/eval_input.jsonl`, `tests/fixtures/eval_invalid.jsonl`, `tests/fixtures/eval_duplicate_ids.jsonl`, `tests/fixtures/preference_labels.jsonl`]

- [ ] 3. Add deterministic non-detector writing metrics

  **What to do**:
  - Add `lynceus/writing_metrics.py` with `compute_writing_metrics(prompt: str, candidate: str, reference: str | None = None) -> dict`.
  - Reuse `split_sentences()` and `normalize_for_detection()` from `lynceus/analysis.py:45-60`, but add fallback behavior inside `writing_metrics.py`: if `split_sentences(candidate)` returns empty for non-empty candidate, treat the normalized candidate as one short sentence for metric denominators.
  - Use lightweight regex tokenization only; do not add dependencies beyond existing `pyproject.toml:9-14`.
  - Implement raw and normalized metrics:
    - `readability`: Flesch Reading Ease using heuristic syllable count; normalized as `clamp((raw + 20) / 120, 0, 1)`.
    - `lexical_diversity`: unique lowercase token ratio; normalized as `clamp(raw / 0.72, 0, 1)`.
    - `sentence_variety`: raw sentence length standard deviation and range; normalized as `0.5` for fewer than 2 sentences, otherwise `clamp((std_words / max(mean_words, 1)) / 0.8, 0, 1)`.
    - `repetition`: raw repeated bigram ratio, trigram ratio, repeated-sentence ratio, repeated-opener ratio; normalized quality score `1 - clamp(mean(ratios), 0, 1)`.
    - `coherence_rubric`: deterministic proxy combining prompt keyword coverage, adjacent sentence token overlap, transition balance, and reference overlap when reference exists; reweight available components to sum to 1 when reference is absent.
    - `style_consistency`: mean of sentence length consistency, terminal punctuation consistency, and casing consistency; each normalized 0-1 where higher is better.
  - Return shape must include `raw` and `normalized` objects; normalized values must be JSON-safe floats or `None`.
  - Add `tests/test_writing_metrics.py` for normal text, repeated text, short one-sentence text, missing reference, Unicode punctuation, and zero-denominator safety.
  **Must NOT do**:
  - Do not import `torch`, `transformers`, detector classes, `generate_advice()`, or `generate_llm_advice()`.
  - Do not emit prose advice, suggestions, or replacement wording.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: multiple deterministic formulas and edge-case normalization require careful reasoning.
  - Skills: [] - No specialized skill needed.
  - Omitted: [`librarian`] - No external API/library research needed; formulas are defined here.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [4, 5] | Blocked By: []

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `lynceus/analysis.py:45-60` - normalization and sentence splitting to reuse.
  - Pattern: `lynceus/analysis.py:118-129` - repeated opener logic that can inspire repetition metrics without advice text.
  - Guardrail: `lynceus/analysis.py:132-173` - advice generation not allowed in eval metrics.
  - Test: `tests/test_analysis.py:7-51` - unittest assertions for deterministic text-analysis helpers.
  - Dependency boundary: `pyproject.toml:9-14` - do not add heavy dependencies.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `uv run python -m unittest tests.test_writing_metrics` exits `0`.
  - [ ] `uv run python -m unittest discover tests` exits `0`.
  - [ ] Tests assert all normalized metrics are either `None` or between `0.0` and `1.0`.
  - [ ] Tests assert repeated sentences lower `repetition.normalized` than varied sentences.
  - [ ] Tests assert missing `reference` leaves reference-specific raw metrics `None` while coherence still has a normalized score.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Writing metrics are stable and normalized
    Tool: Bash
    Steps: Run `uv run python -m unittest tests.test_writing_metrics`
    Expected: Command exits 0; tests assert deterministic exact/rounded values for representative fixture strings.
    Evidence: .omo/evidence/task-3-writing-metrics.txt

  Scenario: Short and repeated text do not divide by zero
    Tool: Bash
    Steps: Run `uv run python -m unittest tests.test_writing_metrics`
    Expected: Command exits 0; tests cover one-sentence candidate and repeated-sentence candidate with JSON-safe outputs.
    Evidence: .omo/evidence/task-3-writing-metrics-edge.txt
  ```

  **Commit**: YES | Message: `feat(eval): add deterministic writing metrics` | Files: [`lynceus/writing_metrics.py`, `tests/test_writing_metrics.py`]

- [ ] 4. Add composite scoring, aggregation, and human preference correlation

  **What to do**:
  - Add `lynceus/eval_scoring.py` with pure deterministic functions:
    - `component_scores(metrics: dict, detector_stats: dict | None) -> dict`.
    - `composite_score(components: dict) -> float | None`.
    - `aggregate_samples(sample_records: list[dict]) -> dict`.
    - `score_preference_pairs(pairs: list[PreferencePair], score_candidate: Callable[[prompt, candidate, reference], dict]) -> dict`.
    - `pearson(xs: list[float], ys: list[float]) -> float | None`.
    - `spearman(xs: list[float], ys: list[float]) -> float | None` using local rank helper; do not add SciPy.
  - Component definitions:
    - `preference_rubric = 0.45*coherence_rubric + 0.25*style_consistency + 0.15*readability + 0.10*lexical_diversity + 0.05*repetition`.
    - `readability_style = mean(readability, lexical_diversity, style_consistency)` ignoring `None` values; `None` only if all inputs are `None`.
    - `repetition_variety = mean(repetition, sentence_variety)` ignoring `None` values.
    - `lynceus_diagnostic = 1.0 - detector_stats["mean"]` when detector stats exist; otherwise `None`.
    - `composite_score = 0.40*preference_rubric + 0.25*readability_style + 0.20*repetition_variety + 0.15*lynceus_diagnostic`; if `lynceus_diagnostic` is `None`, re-normalize over available component weights and include `weights_applied` in output.
  - Preference correlation behavior:
    - For each non-tie pair, compute candidate A and B composite scores with the same metrics pipeline.
    - Predicted preference: `"a"` if `score_a > score_b`, `"b"` if `score_b > score_a`, `"tie"` if equal after rounding to 6 decimals.
    - `agreement = correct_non_tie_predictions / non_tie_count`; `null` if no non-tie pairs.
    - `pearson` and `spearman` use `score_delta = score_a - score_b` and `human_delta = 1` for preferred `a`, `-1` for preferred `b`; ties excluded; return `null` for fewer than 2 non-tie pairs or zero variance.
    - Count ties separately in `tie_count`; do not treat tie labels as agreement wins or losses.
  - Add `tests/test_eval_scoring.py` covering formula weights, missing diagnostic reweighting, aggregate summary stats, agreement, Pearson/Spearman, ties, and undefined correlations as `None`.
  **Must NOT do**:
  - Do not use random sampling, model calls, or time-dependent values.
  - Do not serialize `NaN`; use `None` and let JSON emit `null`.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: scoring math and correlation edge cases need precise implementation.
  - Skills: [] - No specialized skill needed.
  - Omitted: [`visual-engineering`] - No UI work.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: [5] | Blocked By: [3]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `lynceus/cli.py:127-135` - existing stats dictionary shape: mean/std/min/max.
  - Pattern: `lynceus/cli.py:145-152` - existing ensemble mean/std/min/max aggregation style.
  - Pattern: `lynceus/output.py:62-64` - current rounded score serialization style to mirror for stable display values.
  - Input: Task 3 `lynceus/writing_metrics.py` normalized metric contract.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `uv run python -m unittest tests.test_eval_scoring` exits `0`.
  - [ ] `uv run python -m unittest discover tests` exits `0`.
  - [ ] Tests assert confirmed weights exactly: 40/25/20/15.
  - [ ] Tests assert all output scores are `None` or in `0.0-1.0`.
  - [ ] Tests assert undefined Pearson/Spearman return `None`, not `0` or `NaN`.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Composite score uses confirmed weighted formula
    Tool: Bash
    Steps: Run `uv run python -m unittest tests.test_eval_scoring`
    Expected: Command exits 0; tests assert component and final composite scores against fixed numeric fixtures.
    Evidence: .omo/evidence/task-4-eval-scoring.txt

  Scenario: Preference correlation handles ties and tiny samples
    Tool: Bash
    Steps: Run `uv run python -m unittest tests.test_eval_scoring`
    Expected: Command exits 0; tests assert tie labels are counted but excluded from binary agreement/correlation and one-pair correlation is null.
    Evidence: .omo/evidence/task-4-preference-correlation.txt
  ```

  **Commit**: YES | Message: `feat(eval): add composite scoring and preference correlation` | Files: [`lynceus/eval_scoring.py`, `tests/test_eval_scoring.py`]

- [ ] 5. Add batch evaluator engine that emits metrics-only sample and summary records

  **What to do**:
  - Add `lynceus/evaluator.py` with `run_eval(input_path: str, preferences_path: str | None, *, model: str, model_id: str | None, batch_size: int, device: str, output_stream, error_stream) -> int`.
  - Load eval samples with `eval_io.load_eval_samples()` for paths and `eval_io.parse_eval_samples_jsonl(stdin_text, "<stdin>")` when `input_path == "-"`; load optional preferences with `eval_io.load_preference_pairs()`.
  - Load detectors once per evaluator invocation using `_load_detectors()` from Task 1; pass the resulting `loaded_detectors` into `_score_text()` for each sample. Preserve detector choices from `lynceus/detectors/__init__.py:9-14` and detector contract from `lynceus/detectors/base.py:6-19`.
  - For each sample in input order:
    - compute detector diagnostics via `_score_text()`;
    - compute writing metrics via `compute_writing_metrics()`;
    - compute component scores and composite score;
    - write one JSONL sample record with `schema_version: 1`, `type: "sample"`, `id`, `metrics`, `component_scores`, `composite_score`, `weights_applied`, `detectors`, and `errors: []`.
  - After all samples, compute aggregate summary via `aggregate_samples()` and optional `score_preference_pairs()`; write final JSONL summary record as the last line.
  - If `preferences_path is None`, summary must still include `preference_correlation` exactly as `{"pair_count": 0, "non_tie_count": 0, "tie_count": 0, "agreement": None, "pearson": None, "spearman": None}` before JSON serialization.
  - Error behavior:
    - Validation errors before scoring: print deterministic message to stderr and return `2`.
    - Unknown detector/model errors: match existing unknown detector style from `lynceus/cli.py:103-106` where possible and return `1`.
    - Detector failure during scoring: print deterministic stderr message with sample id and return `1`; do not emit misleading summary.
  - Add `tests/test_evaluator.py` using fake detector injection/patching so tests never download Hugging Face models.
  **Must NOT do**:
  - Do not call `output_json()` from `lynceus/output.py:47-91` because it includes `advice`.
  - Do not call `output_rich()` from `lynceus/output.py:94-174`.
  - Do not print progress bars or rich output in eval mode.
  - Do not include advice/rewrite/evasion keys or strings anywhere in records.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: central integration of schema, metrics, scoring, and errors.
  - Skills: [] - No specialized skill needed.
  - Omitted: [`writing`] - Docs are separate.

  **Parallelization**: Can Parallel: NO | Wave 4 | Blocks: [6, 7] | Blocked By: [1, 2, 3, 4]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `lynceus/cli.py:101-137` - detector validation/load/score loop.
  - Pattern: `lynceus/cli.py:139-154` - ensemble score aggregation.
  - Guardrail: `lynceus/output.py:66-72` - existing advice key forbidden in eval output.
  - Test: `tests/test_cli.py:17-29` - fake scoring helper tests can be extended for evaluator patching.
  - Fixture: `tests/fixtures/eval_input.jsonl` from Task 2.
  - Fixture: `tests/fixtures/preference_labels.jsonl` from Task 2.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `uv run python -m unittest tests.test_evaluator` exits `0`.
  - [ ] `uv run python -m unittest discover tests` exits `0`.
  - [ ] Tests parse every emitted line as JSON and assert final line has `type == "summary"`.
  - [ ] Tests assert sample record count equals valid input sample count and order matches input order.
  - [ ] Tests assert output string does not contain `advice`, `rewrite`, `avoid detection`, `bypass`, or `evade` case-insensitively.
  - [ ] Tests assert invalid input returns code `2` and writes error details to stderr.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Batch evaluator emits sample records plus final summary
    Tool: Bash
    Steps: Run `uv run python -m unittest tests.test_evaluator`
    Expected: Command exits 0; tests parse JSONL output, verify 3 sample records followed by one summary record.
    Evidence: .omo/evidence/task-5-evaluator-jsonl.txt

  Scenario: Evaluation mode emits metrics only
    Tool: Bash
    Steps: Run `uv run python -m unittest tests.test_evaluator`
    Expected: Command exits 0; tests scan all output for forbidden advice/evasion words and find none.
    Evidence: .omo/evidence/task-5-evaluator-no-advice.txt
  ```

  **Commit**: YES | Message: `feat(eval): add batch evaluator engine` | Files: [`lynceus/evaluator.py`, `tests/test_evaluator.py`]

- [ ] 6. Wire `lynceus eval` CLI subcommand and help text

  **What to do**:
  - In `lynceus/cli.py`, implement dispatch so:
    - `lynceus eval --help` shows eval-specific help.
    - `lynceus eval --input tests/fixtures/eval_input.jsonl` runs `evaluator.run_eval(...)`.
    - `lynceus eval -i tests/fixtures/eval_input.jsonl --preferences tests/fixtures/preference_labels.jsonl` is equivalent long/short input syntax.
    - `lynceus eval --input -` reads eval JSONL from stdin.
  - Eval flags:
    - `--input`, `-i`: required path or `-` for stdin.
    - `--preferences`: optional preference-label JSONL path.
    - `--model`, `-m`: same default `tmr` and comma-separated ensemble behavior as `lynceus/cli.py:67-69`.
    - `--model-id`: same override behavior as `lynceus/cli.py:70-71`.
    - `--batch-size`, `-b`: same default `32` as `lynceus/cli.py:74-75`.
    - `--device`, `-d`: same choices/default as `lynceus/cli.py:76-78`.
  - Eval mode must not expose `--advice`; advice is forbidden by design.
  - Preserve legacy root help from `README.md:124-154`; do not force users to write `lynceus analyze` or similar.
  - Add `tests/test_eval_cli.py` with patched `evaluator.run_eval()` and patched `sys.argv`/streams to verify dispatch, help, required input, stdin path, and absence of `--advice` in eval help.
  **Must NOT do**:
  - Do not break `python -m lynceus`; `lynceus/__main__.py` must continue to call the CLI entrypoint.
  - Do not add `--json` to eval mode; JSONL is always the eval output format.
  - Do not make eval output rich/colorized.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: public CLI contract and backward compatibility.
  - Skills: [] - No specialized skill needed.
  - Omitted: [`visual-engineering`] - No UI styling beyond argparse help.

  **Parallelization**: Can Parallel: NO | Wave 5 | Blocks: [7] | Blocked By: [1, 2, 5]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `lynceus/cli.py:58-82` - current argparse options to preserve for legacy path.
  - Pattern: `pyproject.toml:21-22` - installed script entrypoint remains `lynceus.cli:main`.
  - Pattern: `lynceus/__main__.py:1-5` - module execution entrypoint; verify unchanged unless needed.
  - Docs: `README.md:124-154` - current CLI reference must remain true.
  - Test: `tests/test_cli.py:1-32` - CLI unittest style.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `uv run python -m unittest tests.test_eval_cli` exits `0`.
  - [ ] `uv run python -m unittest discover tests` exits `0`.
  - [ ] `uv run lynceus --help` exits `0` and includes `--advice`.
  - [ ] `uv run lynceus eval --help` exits `0`, includes `--input`, `--preferences`, `--model`, and does not include `--advice`.
  - [ ] `uv run python -m compileall lynceus tests` exits `0`.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Eval help exposes metrics-only interface
    Tool: Bash
    Steps: Run `uv run lynceus eval --help`
    Expected: Command exits 0; stdout includes `--input` and `--preferences`; stdout does not include `--advice`.
    Evidence: .omo/evidence/task-6-eval-help.txt

  Scenario: Legacy help remains backward compatible
    Tool: Bash
    Steps: Run `uv run lynceus --help`
    Expected: Command exits 0; stdout still includes `--file`, `--json`, `--model`, `--advice`.
    Evidence: .omo/evidence/task-6-legacy-help.txt
  ```

  **Commit**: YES | Message: `feat(cli): add eval subcommand` | Files: [`lynceus/cli.py`, `tests/test_eval_cli.py`]

- [ ] 7. Document evaluation mode, schemas, examples, and safety boundaries

  **What to do**:
  - Update `README.md` with a new `Evaluation mode` section after the current JSON output section at `README.md:197-248`.
  - Document command examples:
    - `uv run lynceus eval --input tests/fixtures/eval_input.jsonl`
    - `uv run lynceus eval --input tests/fixtures/eval_input.jsonl --preferences tests/fixtures/preference_labels.jsonl`
  - Document input sample schema, preference schema, per-sample output fields, final summary fields, and composite weights.
  - Explicitly state eval mode is metrics-only: no rewrite advice, no suggestions for avoiding detection, no LLM judge dependency.
  - Update CLI reference section around `README.md:124-154` to mention `lynceus eval --help` without replacing the legacy root usage block.
  - Update development/testing section around `README.md:348-365` only if new test files/fixtures need mention; keep existing commands unchanged.
  - Add or extend a test that reads README text and asserts the eval examples mention `--input`, `--preferences`, metrics-only, and do not contain detection-evasion phrasing.
  **Must NOT do**:
  - Do not market scores as proof of authorship; preserve limitation language from README intro and `README.md:328-346`.
  - Do not include instructions for rewriting text to lower detector scores.
  - Do not remove existing detector/advice documentation for the legacy single-text analyzer.

  **Recommended Agent Profile**:
  - Category: `writing` - Reason: user-facing docs and schema explanations.
  - Skills: [] - No specialized skill needed.
  - Omitted: [`deep`] - Implementation decisions already fixed in prior tasks.

  **Parallelization**: Can Parallel: NO | Wave 6 | Blocks: [Final Verification] | Blocked By: [5, 6]

  **References** (executor has NO interview context - be exhaustive):
  - Docs: `README.md:197-248` - existing JSON output section; add eval docs after it.
  - Docs: `README.md:250-270` - existing advice docs; distinguish eval mode from advice mode.
  - Docs: `README.md:328-346` - privacy and exit behavior caveats to preserve.
  - Docs: `README.md:356-365` - existing test and compile commands.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `uv run python -m unittest discover tests` exits `0`.
  - [ ] `uv run python -m compileall lynceus tests` exits `0`.
  - [ ] README includes `lynceus eval --input` and `--preferences` examples.
  - [ ] README describes composite weights exactly as 40/25/20/15.
  - [ ] README states eval mode is metrics-only and does not provide rewrite advice or detection-avoidance suggestions.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: README documents eval schema and commands
    Tool: Bash
    Steps: Run `uv run python -m unittest discover tests`
    Expected: Command exits 0; README-focused test asserts eval examples and schema names are present.
    Evidence: .omo/evidence/task-7-readme-docs.txt

  Scenario: README avoids evasion framing
    Tool: Bash
    Steps: Run `uv run python -m unittest discover tests`
    Expected: Command exits 0; README-focused test asserts forbidden evasion phrases are absent from the eval section.
    Evidence: .omo/evidence/task-7-readme-safety.txt
  ```

  **Commit**: YES | Message: `docs(eval): document metrics-only evaluation mode` | Files: [`README.md`, `tests/test_readme_eval_docs.py`]

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit after each implementation task if repo policy permits. Use conventional messages listed per task.
- Never commit `.omo/evidence/` unless the repository already tracks evidence artifacts.

## Success Criteria
- Existing single-input command behavior remains intact.
- Batch eval emits deterministic metrics-only JSONL records and final summary.
- Preference correlation works with bundled fixture and external labels.
- Composite score uses the confirmed weights and normalized higher-is-better components.
- Eval output contains no advice, rewrite, or detection-evasion suggestions.
