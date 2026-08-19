"""Tests for humanify.analysis.generate_quality_advice."""

import unittest

from humanify.analysis import generate_quality_advice


class QualityAdviceTest(unittest.TestCase):
    def test_strong_grammar_gets_praise(self):
        metrics = {
            "raw": {
                "grammar_heuristics": {
                    "passive_ratio": 0.05,
                    "weak_modifier_ratio": 0.01,
                    "runon_ratio": 0.0,
                },
                "specificity": {
                    "vague_word_ratio": 0.03,
                    "abstract_noun_ratio": 0.08,
                    "concrete_detail_ratio": 0.6,
                },
                "evidence": {
                    "citation_density": 0.2,
                    "example_density": 0.2,
                    "data_density": 0.1,
                    "experience_density": 0.1,
                },
            },
            "normalized": {
                "grammar_heuristics": 0.95,
                "specificity": 0.85,
                "evidence": 0.70,
            },
        }
        tips = generate_quality_advice(metrics)
        self.assertTrue(
            any("Grammar:" in tip and "\u2713" in tip for tip in tips),
            f"Expected grammar praise in tips: {tips}",
        )

    def test_weak_grammar_gets_warning(self):
        metrics = {
            "raw": {
                "grammar_heuristics": {
                    "passive_ratio": 0.6,
                    "weak_modifier_ratio": 0.2,
                    "runon_ratio": 0.6,
                },
                "specificity": {
                    "vague_word_ratio": 0.01,
                    "abstract_noun_ratio": 0.05,
                    "concrete_detail_ratio": 0.6,
                },
                "evidence": {
                    "citation_density": 0.2,
                    "example_density": 0.2,
                    "data_density": 0.1,
                    "experience_density": 0.1,
                },
            },
            "normalized": {
                "grammar_heuristics": 0.2,
                "specificity": 0.85,
                "evidence": 0.70,
            },
        }
        tips = generate_quality_advice(metrics)
        self.assertTrue(
            any("Grammar:" in tip and "\u26a0" in tip for tip in tips),
            f"Expected grammar warning in tips: {tips}",
        )

    def test_low_specificity_warns_vague_words(self):
        metrics = {
            "raw": {
                "grammar_heuristics": {
                    "passive_ratio": 0.1,
                    "weak_modifier_ratio": 0.02,
                    "runon_ratio": 0.1,
                },
                "specificity": {
                    "vague_word_ratio": 0.08,
                    "abstract_noun_ratio": 0.05,
                    "concrete_detail_ratio": 0.1,
                },
                "evidence": {
                    "citation_density": 0.2,
                    "example_density": 0.2,
                    "data_density": 0.1,
                    "experience_density": 0.1,
                },
            },
            "normalized": {
                "grammar_heuristics": 0.80,
                "specificity": 0.3,
                "evidence": 0.70,
            },
        }
        tips = generate_quality_advice(metrics)
        self.assertTrue(
            any("Specificity:" in tip and "vague" in tip for tip in tips),
            f"Expected vague-word warning in tips: {tips}",
        )

    def test_low_evidence_warns_missing_citations(self):
        metrics = {
            "raw": {
                "grammar_heuristics": {
                    "passive_ratio": 0.1,
                    "weak_modifier_ratio": 0.02,
                    "runon_ratio": 0.1,
                },
                "specificity": {
                    "vague_word_ratio": 0.02,
                    "abstract_noun_ratio": 0.05,
                    "concrete_detail_ratio": 0.6,
                },
                "evidence": {
                    "citation_density": 0.0,
                    "example_density": 0.0,
                    "data_density": 0.0,
                    "experience_density": 0.0,
                },
            },
            "normalized": {
                "grammar_heuristics": 0.80,
                "specificity": 0.85,
                "evidence": 0.05,
            },
        }
        tips = generate_quality_advice(metrics)
        self.assertTrue(
            any("Evidence:" in tip and "citations" in tip for tip in tips),
            f"Expected citation warning in tips: {tips}",
        )

    def test_balanced_text_returns_generic_tip(self):
        metrics = {
            "raw": {
                "grammar_heuristics": {
                    "passive_ratio": 0.2,
                    "weak_modifier_ratio": 0.05,
                    "runon_ratio": 0.2,
                },
                "specificity": {
                    "vague_word_ratio": 0.04,
                    "abstract_noun_ratio": 0.10,
                    "concrete_detail_ratio": 0.4,
                },
                "evidence": {
                    "citation_density": 0.15,
                    "example_density": 0.15,
                    "data_density": 0.10,
                    "experience_density": 0.10,
                },
            },
            "normalized": {
                "grammar_heuristics": 0.65,
                "specificity": 0.55,
                "evidence": 0.45,
            },
        }
        tips = generate_quality_advice(metrics)
        self.assertTrue(
            any("balanced" in tip.lower() for tip in tips),
            f"Expected generic balanced tip: {tips}",
        )

    def test_empty_metrics_returns_something(self):
        tips = generate_quality_advice({})
        self.assertIsInstance(tips, list)
        self.assertTrue(len(tips) > 0)


class AcademicStyleTest(unittest.TestCase):
    def test_academic_allows_more_passive_voice(self):
        metrics = {
            "raw": {
                "grammar_heuristics": {
                    "passive_ratio": 0.30,
                    "weak_modifier_ratio": 0.01,
                    "runon_ratio": 0.0,
                },
                "specificity": {
                    "vague_word_ratio": 0.01,
                    "abstract_noun_ratio": 0.05,
                    "concrete_detail_ratio": 0.6,
                },
                "evidence": {
                    "citation_density": 0.2,
                    "example_density": 0.2,
                    "data_density": 0.1,
                    "experience_density": 0.1,
                },
            },
            "normalized": {
                "grammar_heuristics": 0.40,
                "specificity": 0.85,
                "evidence": 0.70,
            },
        }
        # General style: grammar_norm < 0.5 and 30% passive > 20% threshold → warning
        general_tips = generate_quality_advice(metrics, style="general")
        self.assertTrue(
            any("Grammar:" in tip and "passive" in tip for tip in general_tips),
            f"Expected passive warning in general style: {general_tips}",
        )
        # Academic style: grammar_norm < 0.5 but 30% passive < 40% threshold → no warning
        academic_tips = generate_quality_advice(metrics, style="academic")
        self.assertFalse(
            any("Grammar:" in tip and "passive" in tip for tip in academic_tips),
            f"Did not expect passive warning in academic style: {academic_tips}",
        )

    def test_academic_allows_lower_evidence_threshold(self):
        metrics = {
            "raw": {
                "grammar_heuristics": {
                    "passive_ratio": 0.1,
                    "weak_modifier_ratio": 0.01,
                    "runon_ratio": 0.0,
                },
                "specificity": {
                    "vague_word_ratio": 0.01,
                    "abstract_noun_ratio": 0.05,
                    "concrete_detail_ratio": 0.6,
                },
                "evidence": {
                    "citation_density": 0.07,
                    "example_density": 0.07,
                    "data_density": 0.07,
                    "experience_density": 0.07,
                },
            },
            "normalized": {
                "grammar_heuristics": 0.85,
                "specificity": 0.85,
                "evidence": 0.25,
            },
        }
        # General style: 7% citations < 10% threshold → warning
        general_tips = generate_quality_advice(metrics, style="general")
        self.assertTrue(
            any("Evidence:" in tip and "citations" in tip for tip in general_tips),
            f"Expected citation warning in general style: {general_tips}",
        )
        # Academic style: 7% citations > 5% threshold → no warning
        academic_tips = generate_quality_advice(metrics, style="academic")
        self.assertFalse(
            any("Evidence:" in tip and "citations" in tip for tip in academic_tips),
            f"Did not expect citation warning in academic style: {academic_tips}",
        )


class ParagraphAdviceTest(unittest.TestCase):
    def test_low_topic_sentence_coverage_warns(self):
        metrics = {
            "raw": {},
            "normalized": {},
            "paragraph_analysis": {
                "paragraph_count": 4,
                "topic_sentence_coverage": 0.25,
                "paragraph_transition_coverage": 0.5,
                "thesis_alignment": 0.4,
            },
        }
        tips = generate_quality_advice(metrics)
        self.assertTrue(
            any("Paragraphs:" in tip and "topic sentence" in tip for tip in tips),
            f"Expected topic sentence warning: {tips}",
        )

    def test_high_topic_sentence_coverage_praises(self):
        metrics = {
            "raw": {},
            "normalized": {},
            "paragraph_analysis": {
                "paragraph_count": 4,
                "topic_sentence_coverage": 0.85,
                "paragraph_transition_coverage": 0.5,
                "thesis_alignment": 0.4,
            },
        }
        tips = generate_quality_advice(metrics)
        self.assertTrue(
            any("Paragraphs:" in tip and "\u2713" in tip for tip in tips),
            f"Expected paragraph praise: {tips}",
        )

    def test_low_thesis_alignment_warns(self):
        metrics = {
            "raw": {},
            "normalized": {},
            "paragraph_analysis": {
                "paragraph_count": 4,
                "topic_sentence_coverage": 0.85,
                "paragraph_transition_coverage": 0.5,
                "thesis_alignment": 0.05,
            },
        }
        tips = generate_quality_advice(metrics)
        self.assertTrue(
            any("Structure:" in tip and "introduction" in tip for tip in tips),
            f"Expected thesis alignment warning: {tips}",
        )
