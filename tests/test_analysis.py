from pathlib import Path
import unittest

from humanify.analysis import generate_advice, normalize_for_detection, prepare_scoring_units


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


if __name__ == "__main__":
    unittest.main()
