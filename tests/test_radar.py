"""Tests for the RADAR detector (Task B, Phase 4.2).

No model downloads — the pipeline is driven through monkeypatched fakes.
"""

import unittest
from unittest.mock import patch

from humanify.detectors import DETECTORS
from humanify.detectors.radar import RadarDetector


def _fake_pipeline_row(labels_scores):
    """Build a text-classification result list (list of {label, score})."""
    return [{"label": lab, "score": sc} for lab, sc in labels_scores]


class RadarRegistryTest(unittest.TestCase):
    def test_registry_contains_radar(self):
        self.assertIn("radar", DETECTORS)
        self.assertTrue(issubclass(DETECTORS["radar"], RadarDetector))

    def test_description_captures_noncommercial_and_paraphrase(self):
        desc = RadarDetector().description.lower()
        self.assertIn("non-commercial", desc)
        self.assertIn("paraphrase", desc)


class RadarModelIdTest(unittest.TestCase):
    def test_default_model_id(self):
        det = RadarDetector()
        self.assertEqual(det.model_id, "TrustSafeAI/RADAR-Vicuna-7B")

    def test_set_model_id_overrides(self):
        det = RadarDetector()
        det.set_model_id("bobae/radar_finetuned_detector")
        self.assertEqual(det.model_id, "bobae/radar_finetuned_detector")


class RadarScoreTest(unittest.TestCase):
    def test_score_is_bounded_float_per_text(self):
        det = RadarDetector()
        det._pipe = FakePipeline([
            _fake_pipeline_row([("LABEL_0", 0.9), ("LABEL_1", 0.1)]),
            _fake_pipeline_row([("LABEL_0", 0.2), ("LABEL_1", 0.8)]),
        ])
        scores = det.score(["first segment.", "second segment."], batch_size=2)
        self.assertEqual(len(scores), 2)
        self.assertTrue(0.0 <= scores[0] <= 1.0)
        self.assertTrue(0.0 <= scores[1] <= 1.0)
        # RADAR convention: AI-probability is class index 0 (log_softmax[0])
        self.assertAlmostEqual(scores[0], 0.9)
        self.assertAlmostEqual(scores[1], 0.2)

    def test_named_ai_human_labels_map_correctly(self):
        det = RadarDetector()
        det._pipe = FakePipeline([
            _fake_pipeline_row([("AI", 0.81), ("Human", 0.19)]),
            _fake_pipeline_row([("AI", 0.05), ("Human", 0.95)]),
        ])
        scores = det.score(["a", "b"])
        self.assertAlmostEqual(scores[0], 0.81)
        self.assertAlmostEqual(scores[1], 0.05)

    def test_human_only_labels_invert(self):
        det = RadarDetector()
        det._pipe = FakePipeline([
            _fake_pipeline_row([("human", 0.9), ("human", 0.1)] + []),
        ])
        # duplicate key collapses; use distinct keys instead
        det2 = RadarDetector()
        det2._pipe = FakePipeline([
            _fake_pipeline_row([("human-written", 0.2), ("LABEL_1", 0.8)]),
        ])
        scores = det2.score(["sample text"], batch_size=1)
        self.assertAlmostEqual(scores[0], 0.8)


class FakePipeline:
    """Stub mapping a list of texts to a list of pre-built result rows."""

    def __init__(self, rows):
        self.rows = rows

    def __call__(self, texts, **kw):
        rows = []
        for t in texts:
            rows.append(self.rows.pop(0))
        return rows


if __name__ == "__main__":
    unittest.main()
