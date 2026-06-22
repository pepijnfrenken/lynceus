import io
import unittest
from unittest.mock import patch

from humanify.analysis import ScoringUnit
from humanify.cli import _expand_unit_scores, _main_single, _resolve_device, _score_text


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

if __name__ == "__main__":
    unittest.main()


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

    def test_main_single_json_includes_writing_quality(self):
        raw = "The cat sat on the mat. It was very happy."
        with patch("humanify.cli.read_input", return_value=raw):
            with patch("humanify.cli._load_detectors", return_value=_FAKE_DETECTORS):
                with patch("sys.stdout", new=io.StringIO()) as stdout:
                    with patch("os._exit"):
                        _main_single(["--json"])
        output = stdout.getvalue()
        import json
        result = json.loads(output)
        self.assertIn("writing_quality", result)
        self.assertIn("metrics", result["writing_quality"])
        self.assertIn("advice", result["writing_quality"])

    def test_main_single_academic_style_includes_paragraph_analysis(self):
        raw = "Gamification can improve learning. It adds game elements.\n\nHowever, poorly designed gamification can cause anxiety.\n\nIn conclusion, gamification helps when designed well."
        with patch("humanify.cli.read_input", return_value=raw):
            with patch("humanify.cli._load_detectors", return_value=_FAKE_DETECTORS):
                with patch("sys.stdout", new=io.StringIO()) as stdout:
                    with patch("os._exit"):
                        _main_single(["--json", "--style", "academic"])
        output = stdout.getvalue()
        import json
        result = json.loads(output)
        self.assertIn("writing_quality", result)
        wq = result["writing_quality"]
        self.assertIn("paragraph_analysis", wq)
        self.assertIn("sentence_analysis", wq)
        self.assertIn("top_worst", wq)
        # Academic style should include paragraph analysis
        pa = wq["paragraph_analysis"]
        self.assertEqual(pa["paragraph_count"], 3)
