"""Tests for the iterative paraphrase-with-verify loop (Task B / Phase 7.2).

``_ollama_generate`` and ``llm_rewrite`` are monkeypatched so no network is
touched; ``detect_fn`` is a deterministic fake that improves after round 1 and
worsens after round 2.
"""

import unittest
from unittest.mock import patch

from redteam import transforms as T

ORIGINAL = (
    "Alpha sentence first good. "
    "Beta marked BETAMARK high. "
    "Gamma sentence third low."
)

# detect_fn: whole-text + per-chunk scorer. ROUND1 => improv, ROUNDBAD => worse.
def _detect_fn(text):
    if "ROUNDBAD" in text:
        return 0.80
    if "ROUND1" in text:
        return 0.20
    if "BETAMARK" in text:
        return 0.95
    return 0.50


class ParaphraseVerifyTest(unittest.TestCase):
    def test_keeps_best_caps_rounds_only_rewrites_flagged_chunks(self):
        # Round 1: whole-text rewrite returns ORIGINAL + " ROUND1" (improves).
        def fake_llm_rewrite(text, model="qwen3:1.7b"):
            return text + " ROUND1"

        # Later rounds: per-chunk generation appends ROUNDBAD (worsens).
        generated_prompts = []

        def fake_ollama(prompt, model="qwen3:1.7b", timeout=60, options=None):
            generated_prompts.append(prompt)
            return "Beta rewritten chunk ROUNDBAD."

        with patch("redteam.transforms.llm_rewrite", side_effect=fake_llm_rewrite), \
             patch("redteam.transforms._ollama_generate",
                   side_effect=fake_ollama):
            best_text, best_score, rounds_used = T.paraphrase_with_verify(
                ORIGINAL, model="mt", detect_fn=_detect_fn, max_rounds=3,
            )

        # The loop caps at exactly max_rounds=3.
        self.assertEqual(rounds_used, 3)

        # It never got worse than the round-1 best: best is the "ROUND1" text.
        self.assertEqual(best_text, ORIGINAL + " ROUND1")
        self.assertEqual(best_score, 0.20)

        # Only the flagged "Beta" chunk was re-paraphrased in later rounds.
        self.assertGreaterEqual(len(generated_prompts), 1)
        for prompt in generated_prompts:
            self.assertIn("Beta", prompt)
            self.assertNotIn("Gamma", prompt)
            self.assertNotIn("first good", prompt)  # unflagged "Alpha" chunk

    def test_max_rounds_one_means_whole_text_only(self):
        def fake_llm_rewrite(text, model="qwen3:1.7b"):
            return text + " ROUND1"

        with patch("redteam.transforms.llm_rewrite",
                   side_effect=fake_llm_rewrite), \
             patch("redteam.transforms._ollama_generate",
                   return_value="SHOULD_NOT_HAPPEN") as gen:
            best_text, best_score, rounds_used = T.paraphrase_with_verify(
                ORIGINAL, model="m", detect_fn=_detect_fn, max_rounds=1,
            )
        self.assertEqual(rounds_used, 1)
        self.assertEqual(best_text, ORIGINAL + " ROUND1")
        # With a single round there is no chunk-level pass at all.
        gen.assert_not_called()

    def test_injected_detect_fn_governs_verdict(self):
        # Without detect_fn (None) nothing improves and input is returned.
        def fake_llm_rewrite(text, model="qwen3:1.7b"):
            return text + " ROUND1"
        with patch("redteam.transforms.llm_rewrite",
                   side_effect=fake_llm_rewrite), \
             patch("redteam.transforms._ollama_generate", return_value="x"):
            best_text, _s, _r = T.paraphrase_with_verify(
                ORIGINAL, model="m", detect_fn=None, max_rounds=2,
            )
        self.assertEqual(best_text, ORIGINAL)


if __name__ == "__main__":
    unittest.main()
