"""Tests for humanify.writing_metrics — deterministic writing quality metrics."""

import unittest

from humanify.writing_metrics import compute_writing_metrics, clamp


def _is_none_or_in_range(value, lo=0.0, hi=1.0):
    """Return True when *value* is None or lo <= value <= hi."""
    return value is None or lo <= value <= hi


def _check_all_normalized_in_range(result: dict, msg: str = ""):
    """Assert every normalized metric is either None or in [0, 1]."""
    norm = result["normalized"]
    for key, val in norm.items():
        if key == "coherence_rubric" and isinstance(val, dict):
            for sub_key, sub_val in val.items():
                assert _is_none_or_in_range(sub_val), (
                    f"{msg}: normalized.{key}.{sub_key}={sub_val!r} not in [0,1] or None"
                )
        else:
            assert _is_none_or_in_range(val), (
                f"{msg}: normalized.{key}={val!r} not in [0,1] or None"
            )


class ClampTest(unittest.TestCase):
    def test_clamp_lower(self):
        self.assertEqual(clamp(-5, 0, 1), 0)

    def test_clamp_upper(self):
        self.assertEqual(clamp(5, 0, 1), 1)

    def test_clamp_middle(self):
        self.assertEqual(clamp(0.5, 0, 1), 0.5)

    def test_clamp_at_edge(self):
        self.assertEqual(clamp(0, 0, 1), 0)
        self.assertEqual(clamp(1, 0, 1), 1)


class WritingMetricsTest(unittest.TestCase):

    def test_normal_varied_text_all_normalized_in_range(self):
        """A normally varied piece of writing should have all metrics in [0,1]."""
        candidate = (
            "The morning sun cast long shadows across the empty school yard. "
            "Children usually arrived early, but today the gates remained closed. "
            "A single paper cup rolled slowly across the pavement, pushed by "
            "the wind. Meanwhile, a teacher stood by the entrance, checking her "
            "phone. She had received no explanation for the sudden closure. "
            "Consequently, she decided to wait for further instructions before "
            "informing the anxious parents. The uncertainty created a strange "
            "tension in the usually calm morning routine."
        )
        prompt = "Describe a scene at a school on an unexpected day off."

        result = compute_writing_metrics(prompt, candidate)
        _check_all_normalized_in_range(result)

    def test_repeated_sentences_lower_repetition_score(self):
        """Repeating the same text should produce a lower repetition.normalized."""
        prompt = "Write about anything."

        varied = (
            "Dogs bark at passing cars every morning. "
            "Cats prefer to sleep through the afternoon heat. "
            "Birds gather near the feeder when the weather cools. "
            "Fish swim in circles around the ornamental pond. "
            "Horses graze quietly in the distant pasture."
        )
        repeated = (
            "Dogs bark at passing cars every morning. "
            "Dogs bark at passing cars every morning. "
            "Dogs bark at passing cars every morning. "
            "Dogs bark at passing cars every morning. "
            "Dogs bark at passing cars every morning."
        )

        varied_result = compute_writing_metrics(prompt, varied)
        repeated_result = compute_writing_metrics(prompt, repeated)

        varied_rep = varied_result["normalized"]["repetition"]
        repeated_rep = repeated_result["normalized"]["repetition"]

        self.assertGreater(
            varied_rep, repeated_rep,
            f"Varied repetition ({varied_rep}) should be > repeated repetition ({repeated_rep})",
        )

    def test_short_one_sentence_candidate(self):
        """A single short sentence should get sentence_variety.normalized = 0.5."""
        prompt = "Say something."
        candidate = "This is a very short sentence."

        result = compute_writing_metrics(prompt, candidate)
        self.assertEqual(
            result["normalized"]["sentence_variety"], 0.5,
        )

    def test_missing_reference(self):
        """Without a reference, reference_overlap should be None and coherence
        should still have a normalized score."""
        prompt = "Write a description."
        candidate = (
            "The river wound through the valley. "
            "Tall pines lined both banks. "
            "Water rushed over smooth stones."
        )

        result = compute_writing_metrics(prompt, candidate, reference=None)

        # reference_overlap in raw should be None
        self.assertIsNone(
            result["raw"]["coherence_rubric"]["reference_overlap"]
        )
        # coherence rubric should still have a normalized value
        coherence_norm = result["normalized"]["coherence_rubric"]
        self.assertIsNotNone(coherence_norm)
        self.assertGreaterEqual(coherence_norm, 0.0)
        self.assertLessEqual(coherence_norm, 1.0)

    def test_with_reference_overlap(self):
        """With a reference, reference_overlap should be a float in [0,1]."""
        prompt = "Describe a forest."
        candidate = (
            "The forest floor was soft with moss. "
            "Sunlight filtered through the leaves above. "
            "A gentle breeze rustled through the branches."
        )
        reference = (
            "Forests are filled with layers of plant life. "
            "Light passes through the canopy in dappled patterns. "
            "Wind moves through the upper branches creating sound."
        )

        result = compute_writing_metrics(prompt, candidate, reference=reference)
        ref_overlap = result["raw"]["coherence_rubric"]["reference_overlap"]
        self.assertIsNotNone(ref_overlap)
        self.assertGreaterEqual(ref_overlap, 0.0)
        self.assertLessEqual(ref_overlap, 1.0)

    def test_unicode_non_ascii_punctuation(self):
        """Unicode non-ASCII punctuation does not cause errors."""
        prompt = "Write something creative."
        candidate = (
            "She stood there—silent and still. "
            "The world felt… different somehow. "
            "What could it mean? "
            "He wasn\u2019t sure anymore! "
            "\u201cPerhaps,\u201d she whispered, \u201cit\u2019s time.\u201d"
        )

        result = compute_writing_metrics(prompt, candidate)
        _check_all_normalized_in_range(result)

    def test_empty_candidate(self):
        """An empty candidate should be handled without division-by-zero."""
        prompt = "Write something."

        result = compute_writing_metrics(prompt, "")

        # All normalized metrics should be in range or None
        _check_all_normalized_in_range(result)

        # Specific degenerate-case sanity checks
        norm = result["normalized"]
        self.assertEqual(norm["readability"], 0.0)
        self.assertEqual(norm["lexical_diversity"], 0.0)
        self.assertEqual(norm["sentence_variety"], 0.5)

    def test_very_short_candidate_falls_back_to_one_sentence(self):
        """A short candidate rejected by split_sentences becomes one sentence."""
        prompt = "Hi."
        candidate = "Hello world."

        result = compute_writing_metrics(prompt, candidate)
        _check_all_normalized_in_range(result)
        self.assertEqual(result["normalized"]["sentence_variety"], 0.5)

    def test_repetition_detects_ngram_overlap(self):
        """Text with repeated bigrams should have lower repetition than varied."""
        prompt = "Write a sentence."

        repetitive = (
            "The cat sat on the mat. "
            "The cat sat on the mat. "
            "The dog sat on the mat."
        )
        varied = (
            "The cat stretched on the sunny mat. "
            "A brown dog wandered through the garden. "
            "Birds chirped loudly from the old oak tree."
        )

        rep_result = compute_writing_metrics(prompt, repetitive)
        var_result = compute_writing_metrics(prompt, varied)

        self.assertLess(
            rep_result["normalized"]["repetition"],
            var_result["normalized"]["repetition"],
        )

    def test_all_normalized_values_in_range_or_none(self):
        """Guard: every normalized metric across several texts is in [0,1]."""
        prompt = "Test."
        candidates = [
            (
                "Technology has transformed how we communicate with one another. "
                "Consider the impact of smartphones on daily interactions. "
                "Moreover, social media platforms have reshaped public discourse. "
                "For example, news spreads faster now than ever before. "
                "However, this speed comes at a cost to accuracy."
            ),
            "Short text.",
            (
                "Repeat. Repeat. Repeat. Repeat. Repeat. Repeat. Repeat. Repeat. "
            ),
            "",
            "\u00bfC\u00f3mo est\u00e1s? \u00a1Muy bien!",
        ]

        for i, cand in enumerate(candidates):
            with self.subTest(candidate_idx=i):
                result = compute_writing_metrics(prompt, cand)
                _check_all_normalized_in_range(result, msg=f"candidate[{i}]")


if __name__ == "__main__":
    unittest.main()
