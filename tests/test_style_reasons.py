import unittest

from humanify.style_reasons import (
    Reason,
    explain_sentences,
    find_formal_words,
    find_generic_phrases,
    generic_high_score_reason,
    repeated_pattern_reasons,
    specificity_reason,
)


class ReasonDataclassTest(unittest.TestCase):
    def test_to_dict_round_trips_all_fields(self):
        r = Reason(type="formal_word", detail="d", suggestion="s")
        self.assertEqual(r.to_dict(), {"type": "formal_word", "detail": "d", "suggestion": "s"})


class GenericPhrasesTest(unittest.TestCase):
    def test_detects_polished_phrase_with_suggestion(self):
        reasons = find_generic_phrases("We used a structured methodology for the project.")
        self.assertEqual(len(reasons), 1)
        self.assertEqual(reasons[0].type, "generic_phrase")
        self.assertIn("'structured methodology'", reasons[0].detail)
        self.assertIn("simple plan", reasons[0].suggestion)

    def test_case_insensitive_match(self):
        reasons = find_generic_phrases("Our deliverables were on time.")
        self.assertTrue(reasons)
        self.assertIn("deliverables", reasons[0].detail.lower())

    def test_dedupes_overlapping_phrases(self):
        # "ongoing professional development" contains "professional development"
        reasons = find_generic_phrases("I value ongoing professional development highly.")
        self.assertEqual(len(reasons), 1)
        self.assertEqual(reasons[0].detail.count("professional development"), 1)

    def test_clean_sentence_returns_nothing(self):
        self.assertEqual(find_generic_phrases("We split the work by who knew what best."), [])


class FormalWordsTest(unittest.TestCase):
    def test_maps_formal_word_to_simpler_alternative(self):
        reasons = find_formal_words("We must ensure we utilize the new system.")
        self.assertEqual(len(reasons), 1)
        self.assertIn("'ensure'", reasons[0].detail)
        self.assertIn("'make sure'", reasons[0].detail)
        self.assertIn("'utilize'", reasons[0].detail)
        self.assertIn("'use'", reasons[0].detail)

    def test_ignores_substring_inside_other_words(self):
        # "ensuring" is a separate entry; "ensure" must not match inside "ensuring"
        reasons = find_formal_words("She was ensuring quality throughout.")
        self.assertEqual(len(reasons), 1)
        self.assertIn("'ensuring'", reasons[0].detail)
        self.assertNotIn("'ensure'", reasons[0].detail)

    def test_plain_sentence_returns_nothing(self):
        self.assertEqual(find_formal_words("We met on Tuesday to talk about the report."), [])


class SpecificityTest(unittest.TestCase):
    def test_flags_sentence_without_detail(self):
        r = specificity_reason("The project went smoothly and the team collaborated well.")
        self.assertIsNotNone(r)
        self.assertEqual(r.type, "specificity")
        self.assertIn("concrete details", r.detail)

    def test_passes_sentence_with_number(self):
        self.assertIsNone(specificity_reason("We shipped 3 reports on 2 May 2024."))

    def test_passes_sentence_with_proper_noun(self):
        self.assertIsNone(specificity_reason("Maria led the React frontend build."))


class RepeatedPatternTest(unittest.TestCase):
    def test_flags_repeated_this_opener(self):
        sents = [
            "This helped me understand the underlying problem better.",
            "This taught me to communicate earlier with the team.",
            "This showed that planning matters more than speed.",
            "Then we finished everything else on schedule.",
        ]
        reasons = repeated_pattern_reasons(sents)
        self.assertEqual(sorted(reasons.keys()), [0, 1, 2])
        r = reasons[0][0]
        self.assertEqual(r.type, "repeated_pattern")
        self.assertIn("This", r.detail)
        self.assertIn("3", r.detail)

    def test_below_threshold_returns_nothing(self):
        sents = ["This helped once.", "This helped twice.", "Nothing else here at all."]
        self.assertEqual(repeated_pattern_reasons(sents), {})


class HighScoreFallbackTest(unittest.TestCase):
    def test_below_threshold_returns_none(self):
        self.assertIsNone(generic_high_score_reason("A smooth generic formal sentence.", 0.2, 0.5))

    def test_above_threshold_returns_generic_reason(self):
        r = generic_high_score_reason("The broad framework was covered smoothly overall.", 0.9, 0.5)
        self.assertIsNotNone(r)
        self.assertEqual(r.type, "high_score")
        self.assertIn("generic", r.detail)


class ExplainSentencesTest(unittest.TestCase):
    def test_index_aligned_with_sentences(self):
        sentences = ["First smooth formal abstract sentence overall.", "Second abstract generic summary sentence overall."]
        probs = [0.9, 0.4]
        reasons = explain_sentences(sentences, probs, [[]] * 2, 0.5)
        self.assertEqual(len(reasons), 2)

    def test_folds_existing_markers_into_reasons(self):
        sentences = ["We leveraged a robust solution for the deliverable."]
        reasons = explain_sentences(sentences, [0.8], [["leverage", "robust"]], 0.5)
        types = [r.type for r in reasons[0]]
        self.assertIn("ai_filler", types)
        self.assertIn("leverage", reasons[0][0].detail)

    def test_high_score_with_no_other_reason_gets_fallback(self):
        # Has a proper noun so specificity won't fire; no formal/generic/filler words.
        sentences = ["The Stanford committee reviewed the broad framework smoothly."]
        reasons = explain_sentences(sentences, [0.9], [[]], 0.5)
        types = [r.type for r in reasons[0]]
        self.assertEqual(types, ["high_score"])

    def test_low_score_no_markers_yields_no_reasons(self):
        sentences = ["Maria led the React frontend build last Tuesday."]
        reasons = explain_sentences(sentences, [0.1], [[]], 0.5)
        self.assertEqual(reasons[0], [])

    def test_deduplicates_reasons(self):
        # A sentence matching both a generic phrase and being flagged for specificity
        # should not duplicate identical (type, detail) entries.
        sentences = ["The cohesive story had clear internal deadlines everywhere."]
        reasons = explain_sentences(sentences, [0.8], [[]], 0.5)
        keys = [(r.type, r.detail) for r in reasons[0]]
        self.assertEqual(len(keys), len(set(keys)))

    def test_caps_reasons_per_sentence(self):
        # Pack a sentence with many distinct signals; verify the cap holds.
        sentences = [
            "Furthermore, we utilized a structured methodology ensuring professional growth, "
            "leveraging deliverables and stakeholders, however we must ensure numerous best practices."
        ]
        reasons = explain_sentences(sentences, [0.9], [[]], 0.5)
        self.assertLessEqual(len(reasons[0]), 5)


if __name__ == "__main__":
    unittest.main()
