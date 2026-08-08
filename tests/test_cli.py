import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from humanify.analysis import ScoringUnit
from humanify.cli import (
    _expand_unit_scores,
    _main_single,
    _resolve_device,
    _score_text,
)


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


_FAKE_DETECTORS = [{"name": "fake", "detector": FakeDetector(), "model_id": "fake-model"}]


class CliDeviceTest(unittest.TestCase):
    def test_cpu_device_does_not_enable_cuda(self):
        self.assertEqual(_resolve_device("cpu"), ("cpu", False))

    def test_cuda_device_is_explicit_opt_in(self):
        self.assertEqual(_resolve_device("cuda"), ("cuda", True))


class CliScoringUnitTest(unittest.TestCase):
    def test_expand_unit_scores_maps_context_scores_to_sentences(self):
        units = [
            ScoringUnit("first passage", 0, 2),
            ScoringUnit("second passage", 2, 3),
        ]

        self.assertEqual(_expand_unit_scores(units, [0.1, 0.8], 3), [0.1, 0.1, 0.8])

    def test_expand_unit_scores_rejects_missing_detector_scores(self):
        units = [ScoringUnit("first passage", 0, 1)]

        with self.assertRaisesRegex(RuntimeError, "detector returned"):
            _expand_unit_scores(units, [], 1)


class CliScoreTextTest(unittest.TestCase):
    def test_score_text_returns_probs_and_stats_with_fake_detector(self):
        raw = "This is the first sentence here. This is the second sentence there. This is the third sentence everywhere."
        result = _score_text(raw, loaded_detectors=_FAKE_DETECTORS, batch_size=32)

        self.assertIn("sentences", result)
        self.assertIn("probs", result)
        self.assertIn("stats", result)
        self.assertIn("model_label", result)
        self.assertIn("markers", result)
        self.assertEqual(len(result["sentences"]), 3)
        self.assertEqual(len(result["probs"]), 3)
        self.assertEqual(result["stats"]["mean"], 0.3)
        self.assertEqual(result["stats"]["std"], 0.0)
        self.assertEqual(result["model_label"], "fake-model")

    def test_score_text_does_not_call_advice_functions(self):
        raw = "This is test sentence number one. This is test sentence number two."
        with (patch("humanify.analysis.generate_advice", side_effect=Exception("should not be called")),
              patch("humanify.advice_llm.generate_llm_advice", side_effect=Exception("should not be called"))):
            result = _score_text(raw, loaded_detectors=_FAKE_DETECTORS, batch_size=32)
        self.assertEqual(len(result["sentences"]), 2)

    def test_main_single_rejects_empty_input(self):
        with patch("humanify.cli.read_input", return_value=""):
            with patch("sys.stderr", new=io.StringIO()) as stderr:
                ret = _main_single(["--file", "/dev/null"])
        self.assertEqual(ret, 1)
        self.assertIn("no input text provided", stderr.getvalue())


class CliEnsembleAxesTest(unittest.TestCase):
    """Task C (Phase 6.1): per-layer axes + agreement/confidence math."""

    RAW = ("The old ferry carries the heavy crates to the day pier each morning. "
           "Children watch the thick rope and the gulls wheel round. "
           "By noon the whole quay hums with heavy cargo.")

    class AxisDetector:
        def __init__(self, p=0.5):
            self._p = p
            self._deg = False

        def set_degenerate(self, val):
            self._deg = val
            return self

        @property
        def is_degenerate(self):
            return self._deg

        def load(self, *a, **k):
            return None

        def set_model_id(self, *a, **k):
            return None

        def get_causal_lm(self):
            return None

        def score(self, texts, batch_size=32):
            return [self._p] * len(texts)

    def _det(self, name, p, degenerate=False):
        d = self.AxisDetector(p)
        d.name = name
        d.model_id = f"m/{name}"
        entry = {"name": name, "detector": d, "model_id": f"m/{name}"}
        d.set_degenerate(degenerate)
        return entry

    def _score(self, loaded):
        return _score_text(self.RAW, loaded_detectors=loaded, batch_size=32)

    def test_two_axes_computed_plus_lexical_structural_artifact(self):
        loaded = [self._det("tmr", 0.9), self._det("fast-detect", 0.1)]
        axes = self._score(loaded)["axes"]
        self.assertEqual(round(axes["classifier_axis"]["score"], 4), 0.9)
        self.assertEqual(round(axes["zero_shot_axis"]["score"], 4), 0.1)
        for axis in ("lexical_axis", "structural_axis", "artifact_axis"):
            self.assertIn(axis, axes)
            self.assertIsNotNone(axes[axis]["score"])

    def test_overall_is_equal_weighted_mean_of_axes(self):
        loaded = [self._det("tmr", 0.9), self._det("fast-detect", 0.1)]
        axes = self._score(loaded)["axes"]
        present = [axes[k]["score"] for k in
                   ("classifier_axis", "zero_shot_axis", "lexical_axis",
                    "structural_axis", "artifact_axis")
                   if axes[k]["score"] is not None]
        expected = sum(present) / len(present)
        self.assertAlmostEqual(axes["overall"], expected, places=4)

    def test_agreement_and_confidence_math(self):
        loaded = [self._det("tmr", 0.9), self._det("fast-detect", 0.1)]
        result = self._score(loaded)
        axes = result["axes"]
        present = [axes[k]["score"] for k in
                   ("classifier_axis", "zero_shot_axis", "lexical_axis",
                    "structural_axis", "artifact_axis")
                   if axes[k]["score"] is not None]
        std = float(np.std(present))
        self.assertAlmostEqual(axes["axis_agreement"], std, places=4)
        self.assertAlmostEqual(result["confidence"],
                               round(1 - min(std / 0.5, 1.0), 4), places=3)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_agreement_raises_confidence_relative_to_disagreement(self):
        agree = self._score([self._det("tmr", 0.5), self._det("fast-detect", 0.5)])
        disagree = self._score([self._det("tmr", 0.9), self._det("fast-detect", 0.1)])
        self.assertGreater(agree["confidence"], disagree["confidence"])

    def test_single_detector_path_keeps_backcompat(self):
        result = self._score([self._det("tmr", 0.3)])
        self.assertEqual(result["stats"]["mean"], 0.3)
        self.assertEqual(result["model_label"], "m/tmr")
        self.assertIn("axes", result)
        self.assertIn("confidence", result)

    def test_degenerate_binoculars_excluded_from_axis(self):
        loaded = [
            self._det("tmr", 0.9),
            self._det("binoculars", 0.8, degenerate=True),
            self._det("fast-detect", 0.1),
        ]
        axes = self._score(loaded)["axes"]
        self.assertEqual(round(axes["zero_shot_axis"]["score"], 4), 0.1)
        self.assertTrue(
            any("degenerate" in e for e in axes["exclusions"]["zero_shot_axis"])
        )

    def test_failed_scoring_detector_excluded_from_axis_with_note(self):
        class Bad:
            name = "tmr"
            model_id = "m/tmr"

            @property
            def is_degenerate(self):
                return False

            def load(self, *a, **k):
                return None

            def set_model_id(self, *a, **k):
                return None

            def score(self, texts, batch_size=32):
                raise RuntimeError("down")

        passed = self._det("fast-detect", 0.1)
        bad = {"name": "tmr", "detector": Bad(), "model_id": "m/tmr"}
        axes = self._score([passed, bad])["axes"]
        self.assertIsNone(axes["classifier_axis"]["score"])
        self.assertEqual(round(axes["zero_shot_axis"]["score"], 4), 0.1)
        self.assertTrue(
            any("score failed" in e for e in axes["exclusions"]["classifier_axis"])
        )


class CliAxesOutputJsonTest(unittest.TestCase):
    def test_output_json_emits_axes_and_confidence(self):
        from humanify.output import output_json

        axes = {
            "overall": 0.2,
            "axis_agreement": 0.35,
            "classifier_axis": {"score": 0.9, "weight": 1.0, "excluded": []},
            "zero_shot_axis": {"score": 0.1, "weight": 1.0, "excluded": []},
            "lexical_axis": {"score": 0.0, "weight": 1.0, "excluded": []},
            "structural_axis": {"score": 0.0, "weight": 1.0, "excluded": []},
            "artifact_axis": {"score": 0.0, "weight": 1.0, "excluded": []},
        }
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            output_json(["sentence one here."], [0.5], [[]],
                        {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
                        0.5, "fake", axes=axes, confidence=0.3)
        data = json.loads(buf.getvalue())
        self.assertEqual(data["axes"]["overall"], 0.2)
        self.assertEqual(data["confidence"], 0.3)
        self.assertIn("classifier_axis", data["axes"])


if __name__ == "__main__":
    unittest.main()
