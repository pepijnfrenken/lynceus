"""Tests for humanify.eval_scoring — composite scoring, aggregation, and
preference correlation."""

import math
import unittest

from humanify.eval_io import PreferencePair
from humanify.eval_scoring import (
    aggregate_samples,
    component_scores,
    composite_score,
    pearson,
    safe_mean,
    score_preference_pairs,
    spearman,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_METRICS = {
    "raw": {
        "readability": 65.0,
        "lexical_diversity": 0.65,
        "sentence_variety": {"raw_std": 2.5, "raw_range": 8.0},
        "repetition": [0.1, 0.2, 0.15],
        "coherence_rubric": {
            "prompt_keyword_coverage": 0.8,
            "adjacent_overlap": 0.6,
            "transition_balance": 0.7,
            "reference_overlap": None,
        },
        "style_consistency": {
            "terminal_punctuation_consistency": 0.9,
            "casing_consistency": 0.85,
        },
        "grammar_heuristics": {
            "passive_ratio": 0.1,
            "weak_modifier_ratio": 0.05,
            "runon_ratio": 0.0,
        },
        "specificity": {
            "vague_word_ratio": 0.05,
            "abstract_noun_ratio": 0.12,
            "concrete_detail_ratio": 0.4,
        },
        "evidence": {
            "citation_density": 0.2,
            "example_density": 0.1,
            "data_density": 0.1,
            "experience_density": 0.0,
        },
    },
    "normalized": {
        "readability": 0.65,
        "lexical_diversity": 0.55,
        "sentence_variety": 0.70,
        "repetition": 0.80,
        "coherence_rubric": 0.72,
        "style_consistency": 0.68,
        "grammar_heuristics": 0.75,
        "specificity": 0.60,
        "evidence": 0.50,
    },
}

# Expected component scores for FIXTURE_METRICS
# preference_rubric = 0.40*0.72 + 0.20*0.68 + 0.10*0.65 + 0.05*0.55 + 0.05*0.80 + 0.10*0.60 + 0.10*0.50
EXPECTED_PREFERENCE_RUBRIC = (
    0.40 * 0.72
    + 0.20 * 0.68
    + 0.10 * 0.65
    + 0.05 * 0.55
    + 0.05 * 0.80
    + 0.10 * 0.60
    + 0.10 * 0.50
)
# readability_style = mean(0.65, 0.55, 0.68, 0.75)
EXPECTED_READABILITY_STYLE = (0.65 + 0.55 + 0.68 + 0.75) / 4.0
# repetition_variety = mean(0.80, 0.70)
EXPECTED_REPETITION_VARIETY = (0.80 + 0.70) / 2.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_scores_in_range(value):
    """Check that every float leaf is None or in [0.0, 1.0]."""
    if value is None:
        return True
    if isinstance(value, float):
        return 0.0 <= value <= 1.0
    if isinstance(value, dict):
        return all(_all_scores_in_range(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_scores_in_range(v) for v in value)
    return True  # int, str, bool, etc. — not score values


# ---------------------------------------------------------------------------
# safe_mean
# ---------------------------------------------------------------------------


class SafeMeanTest(unittest.TestCase):
    def test_basic_mean(self):
        self.assertAlmostEqual(safe_mean([0.5, 0.7, 0.9]), 0.7)

    def test_ignores_none(self):
        self.assertAlmostEqual(safe_mean([0.5, None, 0.7]), 0.6)

    def test_all_none(self):
        self.assertIsNone(safe_mean([None, None]))

    def test_empty_list(self):
        self.assertIsNone(safe_mean([]))


# ---------------------------------------------------------------------------
# component_scores
# ---------------------------------------------------------------------------


class ComponentScoresTest(unittest.TestCase):
    def test_keys_and_values(self):
        components = component_scores(FIXTURE_METRICS)
        self.assertIn("preference_rubric", components)
        self.assertIn("readability_style", components)
        self.assertIn("repetition_variety", components)
        self.assertIn("humanify_diagnostic", components)

        self.assertAlmostEqual(
            components["preference_rubric"], EXPECTED_PREFERENCE_RUBRIC
        )
        self.assertAlmostEqual(
            components["readability_style"], EXPECTED_READABILITY_STYLE
        )
        self.assertEqual(
            components["repetition_variety"], EXPECTED_REPETITION_VARIETY
        )
        self.assertIsNone(components["humanify_diagnostic"])

    def test_with_detector_stats(self):
        detector_stats = {"mean": 0.25}
        components = component_scores(FIXTURE_METRICS, detector_stats)
        self.assertAlmostEqual(components["humanify_diagnostic"], 0.75)
        self.assertAlmostEqual(
            components["preference_rubric"], EXPECTED_PREFERENCE_RUBRIC
        )

    def test_all_none_normalized(self):
        metrics = {
            "raw": {},
            "normalized": {
                "readability": None,
                "lexical_diversity": None,
                "sentence_variety": None,
                "repetition": None,
                "coherence_rubric": None,
                "style_consistency": None,
            },
        }
        components = component_scores(metrics)
        self.assertIsNone(components["preference_rubric"])
        self.assertIsNone(components["readability_style"])
        self.assertIsNone(components["repetition_variety"])
        self.assertIsNone(components["humanify_diagnostic"])

    def test_output_scores_in_range(self):
        components = component_scores(FIXTURE_METRICS)
        self.assertTrue(
            _all_scores_in_range(components),
            msg=f"Scores out of range: {components}",
        )


# ---------------------------------------------------------------------------
# composite_score
# ---------------------------------------------------------------------------


class CompositeScoreTest(unittest.TestCase):
    def test_matches_40_25_20_15_weights(self):
        components = component_scores(FIXTURE_METRICS)
        result = composite_score(components)

        # humanify_diagnostic is None, so renormalize by 0.85
        numerator = (
            0.40 * EXPECTED_PREFERENCE_RUBRIC
            + 0.25 * EXPECTED_READABILITY_STYLE
            + 0.20 * EXPECTED_REPETITION_VARIETY
        )
        expected = numerator / 0.85
        self.assertAlmostEqual(result["composite"], expected, places=6)
        self.assertEqual(
            result["weights_applied"],
            ["preference_rubric", "readability_style", "repetition_variety"],
        )

    def test_no_diagnostic_renormalizes(self):
        components = {
            "preference_rubric": 0.80,
            "readability_style": 0.70,
            "repetition_variety": 0.60,
            "humanify_diagnostic": None,
        }
        result = composite_score(components)
        expected = (0.40 * 0.80 + 0.25 * 0.70 + 0.20 * 0.60) / 0.85
        self.assertAlmostEqual(result["composite"], expected, places=6)
        self.assertNotIn("humanify_diagnostic", result["weights_applied"])

    def test_all_components_scores_in_range(self):
        components = component_scores(FIXTURE_METRICS)
        result = composite_score(components)
        self.assertTrue(
            _all_scores_in_range(result),
            msg=f"Scores out of range: {result}",
        )

    def test_all_none_components(self):
        components = {
            "preference_rubric": None,
            "readability_style": None,
            "repetition_variety": None,
            "humanify_diagnostic": None,
        }
        result = composite_score(components)
        self.assertIsNone(result["composite"])
        self.assertEqual(result["weights_applied"], [])


# ---------------------------------------------------------------------------
# pearson
# ---------------------------------------------------------------------------


class PearsonTest(unittest.TestCase):
    def test_perfect_positive(self):
        r = pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        self.assertAlmostEqual(r, 1.0, places=10)

    def test_perfect_negative(self):
        r = pearson([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
        self.assertAlmostEqual(r, -1.0, places=10)

    def test_zero_variance_returns_none(self):
        self.assertIsNone(pearson([1, 2, 3], [5, 5, 5]))

    def test_single_point_returns_none(self):
        self.assertIsNone(pearson([1], [2]))

    def test_empty_lists_returns_none(self):
        self.assertIsNone(pearson([], []))

    def test_not_zero_or_nan_on_undefined(self):
        r = pearson([1, 2, 3], [5, 5, 5])
        self.assertIsNone(r)
        self.assertFalse(isinstance(r, float))


# ---------------------------------------------------------------------------
# spearman
# ---------------------------------------------------------------------------


class SpearmanTest(unittest.TestCase):
    def test_perfect_monotonic(self):
        r = spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        self.assertAlmostEqual(r, 1.0, places=10)

    def test_perfect_decreasing(self):
        r = spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
        self.assertAlmostEqual(r, -1.0, places=10)

    def test_zero_variance_returns_none(self):
        self.assertIsNone(spearman([1, 2, 3], [5, 5, 5]))

    def test_single_point_returns_none(self):
        self.assertIsNone(spearman([1], [2]))

    def test_empty_lists_returns_none(self):
        self.assertIsNone(spearman([], []))


# ---------------------------------------------------------------------------
# aggregate_samples
# ---------------------------------------------------------------------------


class AggregateSamplesTest(unittest.TestCase):
    def setUp(self):
        self.rec1 = {
            "metrics": {
                "raw": {"readability": 60.0, "lexical_diversity": 0.50},
                "normalized": {
                    "readability": 0.60,
                    "lexical_diversity": 0.50,
                    "sentence_variety": 0.70,
                    "repetition": 0.80,
                    "coherence_rubric": 0.72,
                    "style_consistency": 0.68,
                },
            },
            "errors": [],
        }
        self.rec2 = {
            "metrics": {
                "raw": {"readability": 70.0, "lexical_diversity": 0.60},
                "normalized": {
                    "readability": 0.70,
                    "lexical_diversity": 0.60,
                    "sentence_variety": 0.65,
                    "repetition": 0.75,
                    "coherence_rubric": 0.68,
                    "style_consistency": 0.72,
                },
            },
            "errors": [],
        }
        self.rec3 = {
            "metrics": {
                "raw": {"readability": 50.0, "lexical_diversity": 0.40},
                "normalized": {
                    "readability": 0.50,
                    "lexical_diversity": 0.40,
                    "sentence_variety": 0.60,
                    "repetition": 0.70,
                    "coherence_rubric": 0.66,
                    "style_consistency": 0.64,
                },
            },
            "errors": [],
        }
        self.invalid_rec = {
            "metrics": {
                "raw": {"readability": 0.0, "lexical_diversity": 0.0},
                "normalized": {
                    "readability": 0.0,
                    "lexical_diversity": 0.0,
                    "sentence_variety": 0.0,
                    "repetition": 0.0,
                    "coherence_rubric": 0.0,
                    "style_consistency": 0.0,
                },
            },
            "errors": ["parse failure"],
        }

    def test_basic_stats(self):
        records = [self.rec1, self.rec2, self.rec3]
        result = aggregate_samples(records)

        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["valid_count"], 3)
        self.assertEqual(result["invalid_count"], 0)
        self.assertEqual(result["duplicate_id_count"], 0)

        # Check normalized.readability: [0.60, 0.70, 0.50]
        nr_key = "normalized.readability"
        self.assertIn(nr_key, result["metrics"])
        nr = result["metrics"][nr_key]
        self.assertAlmostEqual(nr["mean"], 0.60)
        self.assertAlmostEqual(nr["min"], 0.50)
        self.assertAlmostEqual(nr["max"], 0.70)
        # population std: variance = ((0.00)² + (0.10)² + (-0.10)²) / 3
        expected_std = math.sqrt(0.02 / 3)
        self.assertAlmostEqual(nr["std"], expected_std)

        # Check normalized.lexical_diversity: [0.50, 0.60, 0.40]
        nl_key = "normalized.lexical_diversity"
        self.assertIn(nl_key, result["metrics"])
        nl = result["metrics"][nl_key]
        self.assertAlmostEqual(nl["mean"], 0.50)
        self.assertAlmostEqual(nl["min"], 0.40)
        self.assertAlmostEqual(nl["max"], 0.60)

        # Check raw.readability: [60.0, 70.0, 50.0] (outside 0-1, OK for raw)
        rr_key = "raw.readability"
        self.assertIn(rr_key, result["metrics"])
        rr = result["metrics"][rr_key]
        self.assertAlmostEqual(rr["mean"], 60.0)

    def test_invalid_records_excluded(self):
        records = [self.rec1, self.invalid_rec]
        result = aggregate_samples(records)

        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["valid_count"], 1)
        self.assertEqual(result["invalid_count"], 1)

        nr = result["metrics"]["normalized.readability"]
        self.assertAlmostEqual(nr["mean"], 0.60)
        self.assertAlmostEqual(nr["min"], 0.60)
        self.assertAlmostEqual(nr["max"], 0.60)

    def test_empty_records(self):
        result = aggregate_samples([])
        self.assertEqual(result["sample_count"], 0)
        self.assertEqual(result["valid_count"], 0)
        self.assertEqual(result["invalid_count"], 0)
        self.assertEqual(result["metrics"], {})

    def test_all_invalid_records(self):
        records = [self.invalid_rec]
        result = aggregate_samples(records)
        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(result["valid_count"], 0)
        self.assertEqual(result["metrics"], {})

    def test_preference_correlation_placeholder(self):
        records = [self.rec1]
        result = aggregate_samples(records)
        self.assertEqual(result["preference_correlation"], {})

    def test_weights_present(self):
        records = [self.rec1]
        result = aggregate_samples(records)
        expected_weights = {
            "preference_rubric": 0.40,
            "readability_style": 0.25,
            "repetition_variety": 0.20,
            "humanify_diagnostic": 0.15,
        }
        self.assertEqual(result["weights"], expected_weights)


# ---------------------------------------------------------------------------
# score_preference_pairs
# ---------------------------------------------------------------------------


def _make_scorer(scores: dict[tuple[str, str], float]):
    """Return a score_candidate callable backed by *scores* dict."""

    def scorer(prompt: str, candidate: str, reference: str | None = None) -> dict:
        return {"composite_score": scores.get((prompt, candidate), 0.5)}

    return scorer


class ScorePreferencePairsTest(unittest.TestCase):
    def test_two_non_tie_pairs_gives_agreement_and_correlation(self):
        pairs = [
            PreferencePair(
                id="pair1",
                prompt="p1",
                candidate_a="a1",
                candidate_b="b1",
                preferred="a",
            ),
            PreferencePair(
                id="pair2",
                prompt="p2",
                candidate_a="a2",
                candidate_b="b2",
                preferred="b",
            ),
        ]
        scorer = _make_scorer({
            ("p1", "a1"): 0.80,
            ("p1", "b1"): 0.30,
            ("p2", "a2"): 0.20,
            ("p2", "b2"): 0.70,
        })
        result = score_preference_pairs(pairs, scorer)

        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(result["non_tie_count"], 2)
        self.assertEqual(result["tie_count"], 0)
        self.assertEqual(result["agreement"], 1.0)

        # score_deltas = [0.50, -0.50], human_deltas = [1.0, -1.0]
        self.assertIsNotNone(result["pearson"])
        self.assertAlmostEqual(result["pearson"], 1.0, places=10)
        self.assertIsNotNone(result["spearman"])
        self.assertAlmostEqual(result["spearman"], 1.0, places=10)

    def test_single_non_tie_pair_null_correlation(self):
        pairs = [
            PreferencePair(
                id="pair1",
                prompt="p",
                candidate_a="a",
                candidate_b="b",
                preferred="a",
            ),
        ]
        scorer = _make_scorer({
            ("p", "a"): 0.90,
            ("p", "b"): 0.50,
        })
        result = score_preference_pairs(pairs, scorer)

        self.assertEqual(result["pair_count"], 1)
        self.assertEqual(result["non_tie_count"], 1)
        self.assertEqual(result["tie_count"], 0)
        self.assertEqual(result["agreement"], 1.0)
        self.assertIsNone(result["pearson"])
        self.assertIsNone(result["spearman"])

    def test_tie_labels_counted_but_excluded(self):
        pairs = [
            PreferencePair(
                id="pair1",
                prompt="p1",
                candidate_a="a1",
                candidate_b="b1",
                preferred="a",
            ),
            PreferencePair(
                id="pair2",
                prompt="p2",
                candidate_a="a2",
                candidate_b="b2",
                preferred="tie",
            ),
        ]
        scorer = _make_scorer({
            ("p1", "a1"): 0.90,
            ("p1", "b1"): 0.50,
            ("p2", "a2"): 0.60,
            ("p2", "b2"): 0.60,
        })
        result = score_preference_pairs(pairs, scorer)

        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(result["non_tie_count"], 1)
        self.assertEqual(result["tie_count"], 1)
        self.assertEqual(result["agreement"], 1.0)
        self.assertIsNone(result["pearson"])
        self.assertIsNone(result["spearman"])

    def test_agreement_less_than_perfect(self):
        pairs = [
            PreferencePair(
                id="pair1",
                prompt="p1",
                candidate_a="a1",
                candidate_b="b1",
                preferred="a",
            ),
            PreferencePair(
                id="pair2",
                prompt="p2",
                candidate_a="a2",
                candidate_b="b2",
                preferred="b",
            ),
        ]
        # Both predictions wrong: model says b for first, a for second
        scorer = _make_scorer({
            ("p1", "a1"): 0.30,
            ("p1", "b1"): 0.80,
            ("p2", "a2"): 0.70,
            ("p2", "b2"): 0.20,
        })
        result = score_preference_pairs(pairs, scorer)

        self.assertEqual(result["non_tie_count"], 2)
        self.assertEqual(result["agreement"], 0.0)

    def test_output_scores_in_range(self):
        pairs = [
            PreferencePair(
                id="pair1",
                prompt="p1",
                candidate_a="a1",
                candidate_b="b1",
                preferred="a",
            ),
            PreferencePair(
                id="pair2",
                prompt="p2",
                candidate_a="a2",
                candidate_b="b2",
                preferred="tie",
            ),
        ]
        scorer = _make_scorer({
            ("p1", "a1"): 0.80,
            ("p1", "b1"): 0.30,
            ("p2", "a2"): 0.60,
            ("p2", "b2"): 0.60,
        })
        result = score_preference_pairs(pairs, scorer)
        self.assertTrue(
            _all_scores_in_range(result),
            msg=f"Scores out of range: {result}",
        )


if __name__ == "__main__":
    unittest.main()
