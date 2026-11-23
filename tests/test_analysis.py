from pathlib import Path
import unittest

from lynceus.analysis import generate_advice, normalize_for_detection, prepare_scoring_units


class AnalysisScoringUnitTest(unittest.TestCase):
    def test_normalize_for_detection_collapses_formatting_whitespace(self):
        self.assertEqual(
            normalize_for_detection("First line.\n\n  Second\tline."),
            "First line. Second line.",
        )

    def test_prepare_scoring_units_scores_dataset_in_contextual_chunks(self):
        text = Path("dataset_test.txt").read_text(encoding="utf-8")

        sentences, units = prepare_scoring_units(text)

        self.assertEqual(len(sentences), 32)
        self.assertEqual(len(units), 2)
        self.assertEqual((units[0].start, units[0].stop), (0, 25))
        self.assertEqual((units[1].start, units[1].stop), (25, 32))
        self.assertNotIn("\n", units[0].text)
        self.assertIn("Conclude if students get classes from home", units[0].text)

    def test_prepare_scoring_units_chunks_long_paragraphs_without_losing_sentences(self):
        text = " ".join(
            f"Sentence {i} has enough words to survive filtering."
            for i in range(6)
        )

        sentences, units = prepare_scoring_units(text, max_words=12)

        self.assertEqual(len(sentences), 6)
        self.assertGreater(len(units), 1)
        self.assertEqual(units[0].start, 0)
        self.assertEqual(units[-1].stop, len(sentences))

    def test_generate_advice_does_not_warn_on_uniformly_low_ai_scores(self):
        sentences = [
            "Clouds moved across the empty school yard.",
            "Morning buses arrived after the heavy rain.",
            "Families waited near the front office.",
        ]
        stats = {"mean": 0.02, "std": 0.0}

        tips, focus, markers = generate_advice(sentences, [0.02, 0.02, 0.02], [[], [], []], stats, 0.5)

        self.assertEqual(tips, ["✓ Text reads naturally — no major structural patterns to address."])
        self.assertEqual(focus, [])
        self.assertEqual(markers, [])


class SignalFlagsAdviceTest(unittest.TestCase):
    """Task E (Phase 3.3): burstiness-band and em-dash density advice tips."""

    @staticmethod
    def _advice(sentences):
        probs = [0.05] * len(sentences)
        markers = [[] for _ in sentences]
        stats = {"mean": 0.05, "std": 0.1}
        return generate_advice(sentences, probs, markers, stats, 0.5)[0]

    def test_uniform_short_sentences_tips_ai_like_band(self):
        tips = self._advice([
            "The core idea here.",
            "The dog runs fast.",
            "The cat sleeps now.",
            "The bird flies high.",
        ])
        joined = " ".join(tips)
        self.assertIn("AI-like", joined)

    def test_varied_sentences_no_ai_like_band_tip(self):
        tips = self._advice([
            "A very long sentence with quite a lot of words to vary things up a bit.",
            "Short one.",
            "A medium length sentence here for the mix.",
            "An even longer sentence so the lengths clearly vary a great deal more.",
        ])
        joined = " ".join(tips)
        self.assertNotIn("ai_like", joined)

    def test_em_dash_density_over_threshold_adds_tip(self):
        tips = self._advice([
            "One thing \u2014 a big thing \u2014 matters most.",
            "Another \u2013 dash-heavy sentence emerges later here.",
        ])
        joined = " ".join(tips)
        self.assertIn("em-dash", joined)

    def test_no_em_dash_no_em_dash_tip(self):
        tips = self._advice(["Plain text with no dashes any way at all."])
        joined = " ".join(tips)
        self.assertNotIn("em-dash", joined)

if __name__ == "__main__":
    unittest.main()



import unittest as _u
from lynceus.analysis import (
    AI_FILLERS, find_markers, find_markers_weighted, find_pattern_tells,
)


class EraAwareVocabTest(unittest.TestCase):
    def test_ai_fillers_still_a_set_of_all_terms(self):
        self.assertIsInstance(AI_FILLERS, set)
        self.assertIn("delve", AI_FILLERS)
        self.assertIn("in conclusion", AI_FILLERS)

    def test_find_markers_still_returns_plain_terms(self):
        self.assertEqual(
            find_markers("The team leveraged a robust framework."),
            ["leverage", "robust"],
        )

    def test_stale_2023_word_weighs_less_than_2026_word(self):
        # "delve" is a stale 2023-era signal; "showcase" a current 2026 one.
        def w(term):
            return dict(find_markers_weighted(f"this text {term} here"))[term]
        self.assertLess(w("delve"), w("showcase"))

    def test_weighted_markers_include_connective_terms(self):
        matched = dict(find_markers_weighted(
            "Moreover, increasing productivity remains crucial for many reasons."
        ))
        self.assertIn("moreover", matched)
        self.assertIn("crucial", matched)

    def test_high_weight_match_surfaces_first_in_advice(self):
        from lynceus.analysis import generate_advice as _ga
        sentence = "We showcase a delve into the topic."
        tips, _, _ = _ga(
            [sentence], [0.9], [find_markers(sentence)],
            {"std": 0.0, "mean": 0.9}, 0.5,
        )
        # the heavier "showcase" should appear before the light "delve"
        joined = " ".join(tips)
        self.assertLess(joined.index("showcase"), joined.index("delve"))


class PatternTellTest(unittest.TestCase):
    def test_connective_opener_fires(self):
        hits = find_pattern_tells("Moreover, the results support this view.")
        self.assertIn("connective_opener", [h[0] for h in hits])

    def test_connective_opener_silent_on_control(self):
        hits = find_pattern_tells("Results support this view quite well.")
        self.assertNotIn("connective_opener", [h[0] for h in hits])

    def test_rule_of_three_fires(self):
        hits = find_pattern_tells(
            "The plan is fast, simple, and effective."
        )
        self.assertIn("rule_of_three", [h[0] for h in hits])

    def test_rule_of_three_silent_on_control(self):
        hits = find_pattern_tells("The plan is fast and simple.")
        self.assertNotIn("rule_of_three", [h[0] for h in hits])

    def test_not_x_but_y_fires(self):
        hits = find_pattern_tells("It is not a bug but a feature.")
        self.assertIn("not_x_but_y", [h[0] for h in hits])

    def test_not_only_fires(self):
        hits = find_pattern_tells("Not only does it scale, but also it is cheap.")
        self.assertIn("not_x_but_y", [h[0] for h in hits])

    def test_not_x_but_y_silent_on_control(self):
        hits = find_pattern_tells("It works but not always quickly enough.")
        self.assertNotIn("not_x_but_y", [h[0] for h in hits])

    def test_ing_analysis_fires(self):
        hits = find_pattern_tells(
            "By leveraging modern tools and utilizing fresh data, it works."
        )
        self.assertIn("ing_analysis", [h[0] for h in hits])

    def test_ing_analysis_silent_on_control(self):
        hits = find_pattern_tells("She is reading a book about cooking.")
        self.assertNotIn("ing_analysis", [h[0] for h in hits])

    def test_em_dash_fires(self):
        hits = find_pattern_tells(
            "One thing — and it is key — matters most."
        )
        self.assertIn("em_dash", [h[0] for h in hits])

    def test_em_dash_silent_on_control(self):
        hits = find_pattern_tells("One thing matters most over the rest.")
        self.assertNotIn("em_dash", [h[0] for h in hits])


if __name__ == "__main__":
    unittest.main()