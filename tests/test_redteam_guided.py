"""Tests for the SICO-lite guided detector optimization (Task C / Phase 7.3).

The detector (``score_fn``) and the generator (``_ollama_generate``) are both
injected/mocked, so nothing external is touched.
"""

import unittest
from unittest.mock import patch

from redteam import transforms as T


class GuidedOptimizeTest(unittest.TestCase):
    def test_keeps_best_after_n_iterations(self):
        calls = []

        def fake_gen(prompt, **kwargs):
            calls.append(prompt)
            return f"candidate_{len(calls)}"

        # Monotonically-improving detector: candidate_n -> 0.9 - 0.15*n.
        def score_fn(text):
            if text.startswith("candidate_"):
                idx = int(text.rsplit("_", 1)[1])
                return 0.9 - 0.15 * idx
            return 1.0

        with patch("redteam.transforms._ollama_generate", side_effect=fake_gen):
            best_text, best_score, used = T.guided_optimize(
                "base", model="m", score_fn=score_fn, iterations=5,
            )

        self.assertEqual(used, 5)
        self.assertEqual(len(calls), 5)
        self.assertEqual(best_text, "candidate_5")
        self.assertAlmostEqual(best_score, 0.9 - 0.15 * 5, places=6)

    def test_non_improving_iteration_keeps_previous_best(self):
        def fake_gen(prompt, **kw):
            return "candidate_identical"

        # score_fn returns a constant => nothing improves; original kept.
        with patch("redteam.transforms._ollama_generate",
                   side_effect=fake_gen):
            best_text, best_score, used = T.guided_optimize(
                "base", model="m", score_fn=lambda t: 0.7, iterations=4,
            )
        self.assertEqual(used, 4)
        self.assertEqual(best_text, "base")
        self.assertEqual(best_score, 0.7)

    def test_iterations_default_is_five(self):
        calls = []

        def fake_gen(prompt, **kw):
            calls.append(1)
            return "candidate_ok"

        def score_fn(text):
            # always improves (arbitrary low score) so best tracks last
            return 0.1

        with patch("redteam.transforms._ollama_generate", side_effect=fake_gen):
            _b, _s, used = T.guided_optimize("base", model="m", score_fn=score_fn)
        self.assertEqual(used, 5)


if __name__ == "__main__":
    unittest.main()
