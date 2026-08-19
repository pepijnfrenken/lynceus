"""Tests for humanify.evaluator — batch eval runner."""
from __future__ import annotations

import io
import json
import os
import unittest
from unittest.mock import patch

from humanify.evaluator import run_eval

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
EVAL_INPUT = os.path.join(FIXTURES_DIR, "eval_input.jsonl")
PREF_INPUT = os.path.join(FIXTURES_DIR, "preference_labels.jsonl")

# ---------------------------------------------------------------------------
# Deterministic fixtures
# ---------------------------------------------------------------------------

FAKE_NORMALIZED = {
    "readability": 0.6,
    "lexical_diversity": 0.5,
    "sentence_variety": 0.7,
    "repetition": 0.8,
    "coherence_rubric": 0.65,
    "style_consistency": 0.9,
    "grammar_heuristics": 0.75,
    "specificity": 0.55,
    "evidence": 0.40,
}

FAKE_RAW = {
    "readability": 50.0,
    "lexical_diversity": 0.5,
    "sentence_variety": {"raw_std": 2.0, "raw_range": 5.0},
    "repetition": [0.1, 0.2],
    "coherence_rubric": {
        "prompt_keyword_coverage": 0.8,
        "adjacent_overlap": 0.3,
        "transition_balance": 0.6,
        "reference_overlap": None,
    },
    "style_consistency": {"terminal_punctuation": 0.9, "casing": 1.0},
    "grammar_heuristics": {
        "passive_ratio": 0.1,
        "weak_modifier_ratio": 0.05,
        "runon_ratio": 0.0,
    },
    "specificity": {
        "vague_word_ratio": 0.04,
        "abstract_noun_ratio": 0.12,
        "concrete_detail_ratio": 0.4,
    },
    "evidence": {
        "citation_density": 0.1,
        "example_density": 0.1,
        "data_density": 0.1,
        "experience_density": 0.0,
    },
}

FAKE_METRICS = {"raw": FAKE_RAW, "normalized": FAKE_NORMALIZED}


class FakeDetector:
    """Minimal detector stub that returns fixed scores without loading models."""
    name = "fake"
    model_id = "fake-model"

    def load(self, device: str, fp16: bool = False) -> None:
        pass

    def score(self, texts: list[str], batch_size: int = 32) -> list[float]:
        return [0.3] * len(texts)

    def set_model_id(self, model_id: str) -> None:
        pass

    def get_causal_lm(self):
        return None


def _fake_load_detectors(model_names, **kwargs):
    return [{"name": "fake", "detector": FakeDetector(), "model_id": "fake-model"}]


def _fake_metrics(prompt, candidate, reference=None):
    return FAKE_METRICS


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class EvaluatorTest(unittest.TestCase):
    maxDiff = None

    def _run(self, input_path=EVAL_INPUT, preferences_path=None, **kwargs):
        """Run evaluator and return (exit_code, stdout_text, stderr_text)."""
        stdout = io.StringIO()
        stderr = io.StringIO()
        ret = run_eval(
            input_path,
            preferences_path,
            model="tmr",
            model_id=None,
            batch_size=32,
            device="cpu",
            output_stream=stdout,
            error_stream=stderr,
            **kwargs,
        )
        return ret, stdout.getvalue(), stderr.getvalue()

    def _run_patched(self, **kwargs):
        """Run with writing metrics and detector patched."""
        with (
            patch("humanify.evaluator._load_detectors", side_effect=_fake_load_detectors),
            patch("humanify.evaluator.compute_writing_metrics", side_effect=_fake_metrics),
        ):
            return self._run(**kwargs)

    def test_eval_basic_no_preferences(self):
        """3 samples + 1 summary record with no preferences."""
        ret, out, err = self._run_patched()
        self.assertEqual(ret, 0, msg=err)
        lines = out.strip().split("\n")
        self.assertEqual(len(lines), 4)  # 3 samples + 1 summary

        # All lines parse as JSON
        records = [json.loads(line) for line in lines]

        # Last line is summary
        self.assertEqual(records[-1]["type"], "summary")

        # Sample count matches
        sample_records = [r for r in records if r["type"] == "sample"]
        self.assertEqual(len(sample_records), 3)

        # IDs in input order
        for r, expected_id in zip(sample_records, ["s1", "s2", "s3"]):
            self.assertEqual(r["id"], expected_id)

        # Summary fields
        summary = records[-1]
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["valid_count"], 3)
        self.assertEqual(summary["invalid_count"], 0)

        # No preference correlation data (no pairs loaded)
        pref = summary["preference_correlation"]
        self.assertEqual(pref["pair_count"], 0)
        self.assertEqual(pref["agreement"], None)

    def test_eval_with_preferences(self):
        """Preferences fixture yields preference_correlation with pair_count=3."""
        ret, out, err = self._run_patched(preferences_path=PREF_INPUT)
        self.assertEqual(ret, 0, msg=err)
        records = [json.loads(line) for line in out.strip().split("\n")]
        summary = records[-1]
        self.assertEqual(summary["type"], "summary")

        pref = summary["preference_correlation"]
        self.assertEqual(pref["pair_count"], 3)
        self.assertIn("non_tie_count", pref)
        self.assertIn("tie_count", pref)
        # 2 non-ties (p1=a, p2=b) + 1 tie (p3=tie)
        self.assertEqual(pref["non_tie_count"], 2)
        self.assertEqual(pref["tie_count"], 1)

    def test_eval_invalid_input_path_returns_2(self):
        """Nonexistent input path returns 2 and writes to stderr."""
        # Only patch writing metrics — detector loading is irrelevant here
        # since we fail before reaching it
        with patch("humanify.evaluator.compute_writing_metrics", side_effect=_fake_metrics):
            ret, out, err = self._run(input_path="/nonexistent/eval_input.jsonl")
        self.assertEqual(ret, 2)
        self.assertIn("Error:", err)

    def test_eval_stdin_path(self):
        """Input path '-' reads from sys.stdin."""
        with open(EVAL_INPUT) as f:
            stdin_data = f.read()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("humanify.evaluator._load_detectors", side_effect=_fake_load_detectors),
            patch("humanify.evaluator.compute_writing_metrics", side_effect=_fake_metrics),
            patch("sys.stdin", new=io.StringIO(stdin_data)),
        ):
            ret = run_eval(
                "-", None,
                model="tmr", model_id=None, batch_size=32, device="cpu",
                output_stream=stdout, error_stream=stderr,
            )
        self.assertEqual(ret, 0, msg=stderr.getvalue())
        lines = stdout.getvalue().strip().split("\n")
        self.assertEqual(len(lines), 4)

    def test_eval_composite_score_in_range(self):
        """Each sample record has composite_score in [0, 1]."""
        ret, out, err = self._run_patched()
        self.assertEqual(ret, 0, msg=err)
        records = [json.loads(line) for line in out.strip().split("\n")]
        for rec in records:
            if rec["type"] == "sample":
                cs = rec["composite_score"]
                self.assertIsNotNone(cs, f"sample {rec['id']} missing composite_score")
                self.assertGreaterEqual(cs, 0.0)
                self.assertLessEqual(cs, 1.0)

    def test_eval_new_metric_keys_present(self):
        """Sample records include grammar, specificity, and evidence metrics."""
        ret, out, err = self._run_patched()
        self.assertEqual(ret, 0, msg=err)
        records = [json.loads(line) for line in out.strip().split("\n")]
        for rec in records:
            if rec["type"] == "sample":
                norm = rec["metrics"]["normalized"]
                raw = rec["metrics"]["raw"]
                self.assertIn("grammar_heuristics", norm)
                self.assertIn("grammar_heuristics", raw)
                self.assertIn("specificity", norm)
                self.assertIn("specificity", raw)
                self.assertIn("evidence", norm)
                self.assertIn("evidence", raw)

    def test_eval_output_contains_no_advice_or_evasion_phrasing(self):
        """Output must not contain forbidden advice/evasion words."""
        ret, out, err = self._run_patched()
        self.assertEqual(ret, 0, msg=err)
        text = out.lower()
        for word in ["advice", "rewrite", "avoid detection", "bypass", "evade"]:
            self.assertNotIn(word, text, msg=f"found forbidden word: {word!r}")

    def test_eval_detector_error_returns_1(self):
        """If _score_text raises, we get return code 1."""
        def broken_score(*a, **kw):
            raise RuntimeError("simulated scoring failure")

        with (
            patch("humanify.evaluator._load_detectors", side_effect=_fake_load_detectors),
            patch("humanify.evaluator.compute_writing_metrics", side_effect=_fake_metrics),
            patch("humanify.evaluator._score_text", side_effect=broken_score),
        ):
            stdout = io.StringIO()
            stderr = io.StringIO()
            ret = run_eval(
                EVAL_INPUT, None,
                model="tmr", model_id=None, batch_size=32, device="cpu",
                output_stream=stdout, error_stream=stderr,
            )
        self.assertEqual(ret, 1)
        self.assertIn("Error scoring sample", stderr.getvalue())

    def test_eval_unknown_detector_returns_1(self):
        """Unknown detector name returns 1 with descriptive error."""
        # Do NOT patch _load_detectors — let the real cli function raise
        with patch("humanify.evaluator.compute_writing_metrics", side_effect=_fake_metrics):
            stdout = io.StringIO()
            stderr = io.StringIO()
            ret = run_eval(
                EVAL_INPUT, None,
                model="bogus", model_id=None, batch_size=32, device="cpu",
                output_stream=stdout, error_stream=stderr,
            )
        self.assertEqual(ret, 1)
        self.assertIn("unknown detector", stderr.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
