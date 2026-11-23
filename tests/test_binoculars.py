"""Tests for the degenerate Binoculars detector (Task A, Phase 2.1)."""

import unittest
import warnings

from lynceus.detectors.binoculars import BinocularsDetector


class BinocularsDegenerationTest(unittest.TestCase):
    def test_same_model_default_is_degenerate_and_warns(self):
        """Constructing with observer==performer (the default) sets the
        degenerate flag and emits a user warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            det = BinocularsDetector()
        self.assertTrue(det.is_degenerate)
        self.assertTrue(
            any(issubclass(w.category, UserWarning)
                for w in caught),
            "expected a UserWarning about the degenerate same-model pair",
        )
        # description must reflect the degeneration
        self.assertIn("degenerate", det.description.lower())

    def test_two_different_ids_clears_flag(self):
        det = BinocularsDetector()
        det.set_model_id("anthropics/claude-2,openai/gpt-4")
        self.assertFalse(det.is_degenerate)
        self.assertEqual(det.model_id, "anthropics/claude-2,openai/gpt-4")

    def test_single_id_makes_pair_degenerate_again(self):
        det = BinocularsDetector()
        det.set_model_id("obs,perf")
        self.assertFalse(det.is_degenerate)
        det.set_model_id("same-model")
        self.assertTrue(det.is_degenerate)


class BinocularsGracefulScoreTest(unittest.TestCase):
    def test_score_empty_or_invalid_gracefully(self):
        det = BinocularsDetector()
        # unloaded detector returns neutral 0.5 for every text
        self.assertEqual(det.score(["some text here"], batch_size=4), [0.5])
        # empty / whitespace texts are neutral
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(det.score(["", "   "]), [0.5, 0.5])
            self.assertEqual(det.score([]), [])


if __name__ == "__main__":
    unittest.main()
