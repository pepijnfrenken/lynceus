"""Tests for the EditLens detector (Task A, Phase 4.1).

No weights are downloaded here: ``load`` is driven with monkeypatched fakes
(``FakeSeqClassifier`` / ``FakeTokenizer``) and gated access is simulated by
mocking ``from_pretrained`` to raise.
"""

import unittest
from unittest.mock import patch

from humanify.detectors import DETECTORS
from humanify.detectors.editlens import EditLensDetector
from tests._fakes import FakeSeqClassifier, FakeTokenizer


class EditlensRegistryTest(unittest.TestCase):
    def test_registry_contains_editlens(self):
        self.assertIn("editlens", DETECTORS)
        self.assertTrue(issubclass(DETECTORS["editlens"], EditLensDetector))

    def test_description_notes_license_and_gating(self):
        det = EditLensDetector()
        desc = det.description.lower()
        self.assertIn("cc by-nc-sa", desc)
        self.assertIn("gate", desc)


class EditlensModelIdTest(unittest.TestCase):
    def test_default_model_id(self):
        det = EditLensDetector()
        self.assertEqual(det.model_id, "pangram/editlens_roberta-large")

    def test_set_model_id_overrides(self):
        det = EditLensDetector()
        det.set_model_id("pangram/editlens_roberta-base")
        self.assertEqual(det.model_id, "pangram/editlens_roberta-base")


class EditlensLoadAndScoreTest(unittest.TestCase):
    def test_score_returns_bounded_float_per_text(self):
        det = EditLensDetector()
        fake_model = FakeSeqClassifier(num_labels=3)
        with patch("humanify.detectors.editlens.AutoTokenizer.from_pretrained",
                   return_value=FakeTokenizer()), \
             patch("humanify.detectors.editlens.AutoModelForSequenceClassification.from_pretrained",
                   return_value=fake_model):
            det.load("cpu", fp16=False)
        texts = ["Human-like polished sentence with a few changes.",
                 "A sentence that looks freshly edited and tightened up right here.",
                 "Third sample sentence to check batching across multiple units."]
        scores = det.score(texts, batch_size=2)
        self.assertEqual(len(scores), len(texts))
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_score_unloaded_raises_runtimeerror(self):
        det = EditLensDetector()
        with self.assertRaises(RuntimeError):
            det.score(["some text"])


class EditlensGatedLoadTest(unittest.TestCase):
    def test_load_raises_runtimeerror_on_hub_error(self):
        det = EditLensDetector()

        def boom(*a, **k):
            raise RuntimeError("401 Unauthorized / gated repository")

        with patch("humanify.detectors.editlens.AutoTokenizer.from_pretrained",
                   return_value=FakeTokenizer()), \
             patch("humanify.detectors.editlens.AutoModelForSequenceClassification.from_pretrained",
                   side_effect=boom):
            with self.assertRaises(RuntimeError) as ctx:
                det.load("cpu", fp16=False)
        self.assertIn("gated", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
