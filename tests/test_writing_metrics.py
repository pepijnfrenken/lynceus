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


class GrammarHeuristicsTest(unittest.TestCase):
    def test_passive_voice_detected(self):
        prompt = "Write a sentence."
        passive = "The ball was thrown by John. The cake was eaten by Mary."
        active = "John threw the ball. Mary ate the cake."

        passive_result = compute_writing_metrics(prompt, passive)
        active_result = compute_writing_metrics(prompt, active)

        self.assertLess(
            passive_result["normalized"]["grammar_heuristics"],
            active_result["normalized"]["grammar_heuristics"],
        )
        self.assertGreater(
            passive_result["raw"]["grammar_heuristics"]["passive_ratio"], 0
        )

    def test_weak_modifiers_detected(self):
        prompt = "Write a sentence."
        weak = "It is very really quite good."
        strong = "It excels."

        weak_result = compute_writing_metrics(prompt, weak)
        strong_result = compute_writing_metrics(prompt, strong)

        self.assertLess(
            weak_result["normalized"]["grammar_heuristics"],
            strong_result["normalized"]["grammar_heuristics"],
        )

    def test_runon_detected(self):
        prompt = "Write a sentence."
        runon = (
            "This is a very long sentence, with many clauses, and extra words, "
            "and even more words, and it just keeps going."
        )
        short = "This is short. It is clear."

        runon_result = compute_writing_metrics(prompt, runon)
        short_result = compute_writing_metrics(prompt, short)

        self.assertLess(
            runon_result["normalized"]["grammar_heuristics"],
            short_result["normalized"]["grammar_heuristics"],
        )


class SpecificityTest(unittest.TestCase):
    def test_vague_words_detected(self):
        prompt = "Write a sentence."
        vague = "The thing was very good. The stuff was bad."
        specific = "The engine reached 120 km/h. The brake pad failed."

        vague_result = compute_writing_metrics(prompt, vague)
        specific_result = compute_writing_metrics(prompt, specific)

        self.assertLess(
            vague_result["normalized"]["specificity"],
            specific_result["normalized"]["specificity"],
        )

    def test_concrete_details_detected(self):
        prompt = "Write a sentence."
        concrete = "In 2024, Paris saw 38 million visitors."
        abstract = "Tourism increased significantly in the capital."

        concrete_result = compute_writing_metrics(prompt, concrete)
        abstract_result = compute_writing_metrics(prompt, abstract)

        self.assertGreater(
            concrete_result["raw"]["specificity"]["concrete_detail_ratio"],
            abstract_result["raw"]["specificity"]["concrete_detail_ratio"],
        )

    def test_abstract_nouns_detected(self):
        prompt = "Write a sentence."
        abstract = "The realization of the transformation required dedication."
        concrete = "The builder hammered nails into the wooden frame."

        abstract_result = compute_writing_metrics(prompt, abstract)
        concrete_result = compute_writing_metrics(prompt, concrete)

        self.assertGreater(
            abstract_result["raw"]["specificity"]["abstract_noun_ratio"],
            concrete_result["raw"]["specificity"]["abstract_noun_ratio"],
        )


class EvidenceTest(unittest.TestCase):
    def test_citations_detected(self):
        prompt = "Write a sentence."
        cited = "Smith (2019) found that the method works."
        uncited = "The method works well."

        cited_result = compute_writing_metrics(prompt, cited)
        uncited_result = compute_writing_metrics(prompt, uncited)

        self.assertGreater(
            cited_result["raw"]["evidence"]["citation_density"],
            uncited_result["raw"]["evidence"]["citation_density"],
        )

    def test_examples_detected(self):
        prompt = "Write a sentence."
        example = "For example, the bird flew south."
        no_example = "The bird flew south."

        example_result = compute_writing_metrics(prompt, example)
        no_example_result = compute_writing_metrics(prompt, no_example)

        self.assertGreater(
            example_result["raw"]["evidence"]["example_density"],
            no_example_result["raw"]["evidence"]["example_density"],
        )

    def test_experience_detected(self):
        prompt = "Write a sentence."
        experience = "In my experience, the process took three days."
        no_experience = "The process took three days."

        exp_result = compute_writing_metrics(prompt, experience)
        no_exp_result = compute_writing_metrics(prompt, no_experience)

        self.assertGreater(
            exp_result["raw"]["evidence"]["experience_density"],
            no_exp_result["raw"]["evidence"]["experience_density"],
        )

    def test_all_metrics_present_in_result(self):
        prompt = "Write a sentence."
        candidate = "The cat sat on the mat. It was very happy."
        result = compute_writing_metrics(prompt, candidate)

        self.assertIn("grammar_heuristics", result["raw"])
        self.assertIn("grammar_heuristics", result["normalized"])
        self.assertIn("specificity", result["raw"])
        self.assertIn("specificity", result["normalized"])
        self.assertIn("evidence", result["raw"])
        self.assertIn("evidence", result["normalized"])


class SentenceAnalysisTest(unittest.TestCase):
    def test_passive_flag_detected(self):
        from humanify.writing_metrics import analyze_sentences
        sentences = ["The ball was eaten by John.", "John threw the ball."]
        result = analyze_sentences(sentences)
        self.assertIn("passive_voice", result[0]["flags"])
        self.assertNotIn("passive_voice", result[1]["flags"])

    def test_weak_modifiers_flag_detected(self):
        from humanify.writing_metrics import analyze_sentences
        sentences = ["It is very really quite good."]
        result = analyze_sentences(sentences)
        flags = result[0]["flags"]
        self.assertTrue(any(f.startswith("weak_modifiers") for f in flags))

    def test_runon_flag_detected(self):
        from humanify.writing_metrics import analyze_sentences
        sentences = [
            "This is a very long sentence, with many clauses, and extra words, and even more words, and it just keeps going and going and going forever."
        ]
        result = analyze_sentences(sentences)
        self.assertIn("runon", result[0]["flags"])

    def test_no_evidence_flag(self):
        from humanify.writing_metrics import analyze_sentences
        sentences = ["The cat sat on the mat."]
        result = analyze_sentences(sentences)
        self.assertIn("no_evidence", result[0]["flags"])

    def test_citation_evidence_flag(self):
        from humanify.writing_metrics import analyze_sentences
        sentences = ["Smith (2019) found that the method works."]
        result = analyze_sentences(sentences)
        self.assertIn("citation", result[0]["flags"])
        self.assertNotIn("no_evidence", result[0]["flags"])


class TopWorstSentencesTest(unittest.TestCase):
    def test_returns_worst_sentences(self):
        from humanify.writing_metrics import get_top_worst_sentences
        sentences = [
            "The ball was thrown by John very quickly.",
            "John threw the ball.",
            "It was good.",
        ]
        worst = get_top_worst_sentences(sentences, "grammar", top_n=2)
        self.assertEqual(len(worst), 2)
        # The passive sentence should be the worst
        self.assertIn("thrown", worst[0][0])

    def test_empty_list(self):
        from humanify.writing_metrics import get_top_worst_sentences
        self.assertEqual(get_top_worst_sentences([], "grammar"), [])


class ParagraphAnalysisTest(unittest.TestCase):
    def test_split_paragraphs(self):
        from humanify.writing_metrics import split_paragraphs
        text = "The first sentence is here. The second sentence is there.\n\nThe third sentence is here. The fourth sentence is there."
        paragraphs = split_paragraphs(text)
        self.assertEqual(len(paragraphs), 2)
        self.assertEqual(len(paragraphs[0]), 2)
        self.assertEqual(len(paragraphs[1]), 2)

    def test_analyze_paragraphs_structure(self):
        from humanify.writing_metrics import analyze_paragraphs
        text = (
            "Gamification can improve learning for students. It adds game elements to education.\n\n"
            "However, poorly designed gamification can cause anxiety for students. "
            "Therefore, designers must be careful with implementation.\n\n"
            "In conclusion, gamification can help when designed well for students."
        )
        result = analyze_paragraphs(text)
        self.assertEqual(result["paragraph_count"], 3)
        self.assertGreater(result["topic_sentence_coverage"], 0.0)
        self.assertGreater(result["paragraph_transition_coverage"], 0.0)
        self.assertGreaterEqual(result["thesis_alignment"], 0.0)

    def test_empty_text(self):
        from humanify.writing_metrics import analyze_paragraphs
        result = analyze_paragraphs("")
        self.assertEqual(result["paragraph_count"], 0)

    def test_compute_writing_metrics_includes_paragraph_analysis(self):
        prompt = "Write about gamification."
        candidate = (
            "Gamification can improve learning for students. It adds game elements to education.\n\n"
            "However, poorly designed gamification can cause anxiety for students.\n\n"
            "In conclusion, gamification helps when designed well for students."
        )
        result = compute_writing_metrics(prompt, candidate)
        self.assertIn("sentence_analysis", result)
        self.assertIn("paragraph_analysis", result)
        self.assertIsInstance(result["sentence_analysis"], list)
        self.assertIsInstance(result["paragraph_analysis"], dict)


if __name__ == "__main__":
    unittest.main()


class SynthesisDetectionTest(unittest.TestCase):
    def test_synthesis_detected_as_evidence(self):
        from humanify.writing_metrics import _sentence_has_synthesis, _sentence_has_evidence
        sentences = [
            "Several studies have found that gamification improves engagement.",
            "Research generally supports the use of dashboards in education.",
            "In contrast to prior work, our findings suggest that negative effects matter.",
            "This is in line with Bitrián et al., 2021.",
        ]
        for s in sentences:
            self.assertTrue(_sentence_has_synthesis(s), f"Expected synthesis in: {s}")
            # When combined in analyze_sentences, synthesis prevents no_evidence flag

    def test_no_synthesis_in_plain_sentences(self):
        from humanify.writing_metrics import _sentence_has_synthesis
        sentences = [
            "The cat sat on the mat.",
            "Gamification is fun.",
            "This study investigates the topic.",
        ]
        for s in sentences:
            self.assertFalse(_sentence_has_synthesis(s), f"Did not expect synthesis in: {s}")


class SentenceQualityScoreTest(unittest.TestCase):
    def test_clean_sentence_perfect_score(self):
        from humanify.writing_metrics import _compute_sentence_quality_score
        self.assertEqual(_compute_sentence_quality_score([]), 10.0)
        self.assertEqual(_compute_sentence_quality_score(["synthesis"]), 10.0)

    def test_runon_penalty(self):
        from humanify.writing_metrics import _compute_sentence_quality_score
        score = _compute_sentence_quality_score(["runon"])
        self.assertEqual(score, 7.0)  # 10 - 3.0

    def test_multiple_flags_penalty(self):
        from humanify.writing_metrics import _compute_sentence_quality_score
        score = _compute_sentence_quality_score(["passive_voice", "no_evidence"])
        self.assertEqual(score, 6.0)  # 10 - 2.0 - 2.0

    def test_counted_flags_penalty(self):
        from humanify.writing_metrics import _compute_sentence_quality_score
        score = _compute_sentence_quality_score(["weak_modifiers(3)"])
        self.assertEqual(score, 7.0)  # 10 - 1.0 * 3

    def test_score_never_negative(self):
        from humanify.writing_metrics import _compute_sentence_quality_score
        score = _compute_sentence_quality_score(["runon", "passive_voice", "no_evidence", "no_concrete_details", "weak_modifiers(5)", "vague_words(5)"])
        self.assertEqual(score, 0.0)


class ParagraphCleaningTest(unittest.TestCase):
    def test_form_feed_stripped(self):
        from humanify.writing_metrics import _clean_text_for_paragraphs
        text = "First paragraph.\n\n\fSecond paragraph."
        cleaned = _clean_text_for_paragraphs(text)
        self.assertNotIn("\f", cleaned)
        self.assertIn("First paragraph", cleaned)
        self.assertIn("Second paragraph", cleaned)

    def test_page_numbers_stripped(self):
        from humanify.writing_metrics import _clean_text_for_paragraphs
        text = "First paragraph with enough words here.\n\n15\n\nSecond paragraph."
        cleaned = _clean_text_for_paragraphs(text)
        self.assertNotIn("15", cleaned)

    def test_figure_captions_stripped(self):
        from humanify.writing_metrics import _clean_text_for_paragraphs
        text = "First paragraph.\n\nFigure 3: LAD Overview\n\nSecond paragraph."
        cleaned = _clean_text_for_paragraphs(text)
        self.assertNotIn("Figure 3", cleaned)

    def test_short_header_blocks_skipped(self):
        from humanify.writing_metrics import split_paragraphs
        text = "First paragraph with enough words here.\n\n15\n\nFigure 3: Overview\n\nSecond paragraph with multiple sentences. It continues here today."
        paragraphs = split_paragraphs(text)
        self.assertEqual(len(paragraphs), 2)
        self.assertEqual(len(paragraphs[0]), 1)
        self.assertEqual(len(paragraphs[1]), 2)
