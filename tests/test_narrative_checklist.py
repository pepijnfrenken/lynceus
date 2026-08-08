"""Tests for humanify.narrative_checklist (Task A, Phase 5.1).

No real LLM calls -- the module takes a dependency-injected callable. We assert
the gate (short text -> disabled, no call) and tolerant parsing.
"""

import json
import unittest
from unittest.mock import Mock

from humanify.narrative_checklist import (
    NARRATIVE_DIMENSIONS,
    build_narrative_prompt,
    parse_narrative_response,
    run_narrative_checklist,
)

# ~600 words of "story-ish" text, well above the 500-word gate.
LONG_TEXT = " ".join([
    "The ferry slips past the headland while gulls wheel through the fog "
    "and the engines thrum like a slow heartbeat."
    for _ in range(50)
])

VALID_JSON = {
    "theme_over_explanation": {"present": True, "evidence": "The narrator states the lesson plainly."},
    "causal_linearity": {"present": True, "evidence": "Every event follows the last directly."},
    "moral_ambiguity": {"present": False, "evidence": "The hero is clearly the good side."},
    "temporal_complexity": {"present": False, "evidence": "The tale runs in order."},
    "subplot_independence": {"present": False, "evidence": "Every subplot mirrors the theme."},
    "embodied_emotion": {"present": True, "evidence": "Fear arrives as a tight throat."},
    "intertextual_explicitness": {"present": False, "evidence": "No named texts appear."},
    "reader_address": {"present": False, "evidence": "No readers are spoken to."},
    "sensory_density": {"present": True, "evidence": "The room smells of rain and dust."},
    "protagonist_agency": {"present": True, "evidence": "The hero resolves it by choice."},
    "social_network": {"present": False, "evidence": "Only the hero exists."},
}


class NarrativeGateTest(unittest.TestCase):
    def test_short_text_is_disabled_and_no_call_made(self):
        fake = Mock(side_effect=AssertionError("llm_call must not run on short text"))
        result = run_narrative_checklist("far too short for narrative analysis.", fake)
        self.assertFalse(result["enabled"])
        self.assertEqual(result["reason"], "too_short")
        self.assertEqual(result["dimensions"], [])
        self.assertEqual(result["flagged"], [])
        self.assertEqual(result["score"], 0.0)
        fake.assert_not_called()

    def test_gate_matches_min_words(self):
        fake = Mock(return_value=json.dumps(VALID_JSON))
        # Exactly at the boundary counts as enabled (>= min_words).
        text = "word " * 500
        result = run_narrative_checklist(text, fake, min_words=500)
        self.assertTrue(result["enabled"])
        fake.assert_called_once()


class NarrativeFlagTest(unittest.TestCase):
    def setUp(self):
        self.fake = Mock(return_value=json.dumps(VALID_JSON))

    def test_dimensions_parsed_and_flags_computed(self):
        result = run_narrative_checklist(LONG_TEXT, self.fake, min_words=500)
        self.assertTrue(result["enabled"])
        self.assertEqual(len(result["dimensions"]), 11)
        by_key = {d["key"]: d for d in result["dimensions"]}

        # theme over-explanation present (ai_high) -> flagged
        self.assertTrue(by_key["theme_over_explanation"]["flagged"])
        # moral ambiguity absent (human_high) -> flagged
        self.assertTrue(by_key["moral_ambiguity"]["flagged"])
        # subplot_independence absent (human_high) -> flagged
        self.assertTrue(by_key["subplot_independence"]["flagged"])
        # reader_address present would NOT be flagged; here absent (human_high) -> flagged
        self.assertTrue(by_key["reader_address"]["flagged"])
        # temporal complexity absent (human_high) -> flagged
        self.assertTrue(by_key["temporal_complexity"]["flagged"])

        # every dimension carries its priors + direction
        for d in result["dimensions"]:
            self.assertIn("ai_prior", d)
            self.assertIn("human_prior", d)
            self.assertIn("direction", d)
            self.assertIn("present", d)
            self.assertIn("evidence", d)

        # flagged list matches the flagged entries in dimensions
        self.assertEqual(
            [d["key"] for d in result["flagged"]],
            [d["key"] for d in result["dimensions"] if d["flagged"]],
        )

    def test_score_is_fraction_flagged_in_unit_range(self):
        result = run_narrative_checklist(LONG_TEXT, self.fake, min_words=500)
        n_flagged = sum(1 for d in result["dimensions"] if d["flagged"])
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
        # all 11 have verdicts
        self.assertAlmostEqual(result["score"], n_flagged / 11, places=4)

    def test_human_high_present_is_not_flagged(self):
        only_social = dict(VALID_JSON)
        only_social = {
            k: {"present": True, "evidence": "gossip abounds"}
            if k == "social_network" else v
            for k, v in VALID_JSON.items()
        }
        self.fake.return_value = json.dumps(only_social)
        result = run_narrative_checklist(LONG_TEXT, self.fake, min_words=500)
        by_key = {d["key"]: d for d in result["dimensions"]}
        self.assertFalse(by_key["social_network"]["flagged"])

    def test_prompt_lists_dimensions(self):
        prompt = build_narrative_prompt(LONG_TEXT)
        for d in NARRATIVE_DIMENSIONS:
            self.assertIn(d["key"], prompt)
        self.assertIn("present", prompt)


class ParseTolerantTest(unittest.TestCase):
    def test_clean_json_parses(self):
        parsed = parse_narrative_response(json.dumps(VALID_JSON))
        self.assertIsNone(parsed["error"])
        self.assertEqual(set(parsed["dimensions"]), {d["key"] for d in NARRATIVE_DIMENSIONS})

    def test_malformed_json_returns_partial_with_error(self):
        raw = "This is prose about the story. {invalid json!!!}"
        parsed = parse_narrative_response(raw)
        self.assertIsNotNone(parsed["error"])
        self.assertEqual(parsed["dimensions"], {})

    def test_code_fenced_and_wrapped_parses(self):
        wrapped = '```json\n' + json.dumps({"dimensions": VALID_JSON}) + '\n```'
        parsed = parse_narrative_response(wrapped)
        self.assertIsNone(parsed["error"])
        self.assertEqual(len(parsed["dimensions"]), 11)

    def test_malformed_response_does_not_crash_checklist(self):
        self.fake = Mock(return_value="no json here buddy")
        result = run_narrative_checklist(LONG_TEXT, self.fake, min_words=500)
        self.assertTrue(result["enabled"])
        # all dimensions present with unknown verdict, no flags, error note added
        self.assertLessEqual(result["score"], 1.0)
        self.assertEqual(result["flagged"], [])
        notes = [d for d in result["dimensions"] if d["key"] == "_parse_error"]
        self.assertEqual(len(notes), 1)

    def test_verdict_string_variants(self):
        parsed = parse_narrative_response(
            json.dumps({"theme_over_explanation": "present",
                        "moral_ambiguity": "absent",
                        "reader_address": "yes"})
        )
        d = parsed["dimensions"]
        self.assertTrue(d["theme_over_explanation"]["present"])
        self.assertFalse(d["moral_ambiguity"]["present"])
        self.assertTrue(d["reader_address"]["present"])


if __name__ == "__main__":
    unittest.main()
