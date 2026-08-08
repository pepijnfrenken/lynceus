"""Tests for the Fast-DetectGPT detector (Task B, Phase 2.2).

No model weights are downloaded here: ``load`` is not invoked, and the
"after load" get_causal_lm() check drives the loaded fields directly.
"""

import unittest
from unittest.mock import patch, MagicMock

from humanify.detectors.fast_detect import FastDetectDetector


class FastDetectModelIdTest(unittest.TestCase):
    def test_default_model_id(self):
        det = FastDetectDetector()
        self.assertEqual(det.model_id, "Qwen/Qwen3.5-0.8B-Base")

    def test_set_model_id_overrides(self):
        det = FastDetectDetector()
        det.set_model_id("EleutherAI/gpt2-xl")
        self.assertEqual(det.model_id, "EleutherAI/gpt2-xl")

    def test_override_reflects_in_description(self):
        det = FastDetectDetector()
        det.set_model_id("EleutherAI/gpt2-xl")
        self.assertIn("gpt2-xl", det.model_id)


class TestCausalLm(unittest.TestCase):
    def test_get_causal_lm_none_before_load(self):
        det = FastDetectDetector()
        self.assertIsNone(det.get_causal_lm())

    def test_get_causal_lm_tuple_after_load(self):
        det = FastDetectDetector()
        # Simulate a successful load (no weights are downloaded).
        det._tokenizer = MagicMock()
        det._model = MagicMock()
        det._device = "cpu"
        result = det.get_causal_lm()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertIs(result[0], det._model)

    def test_load_populates_causal_lm(self):
        det = FastDetectDetector()
        fake_tok = MagicMock()
        fake_tok.pad_token_id = None
        fake_tok.eos_token_id = 0
        fake_model = MagicMock()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok), \
             patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_model):
            det.load("cpu", fp16=False)
        self.assertIsNotNone(det.get_causal_lm())


if __name__ == "__main__":
    unittest.main()
