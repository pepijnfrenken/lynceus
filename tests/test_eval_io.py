"""Tests for humanify.eval_io."""

import json
import math
import unittest
from pathlib import Path

from humanify.eval_io import (
    SCHEMA_VERSION,
    DuplicateIdError,
    EvalSample,
    PreferencePair,
    json_dumps_record,
    load_eval_samples,
    load_preference_pairs,
    parse_eval_samples_jsonl,
    parse_preference_pairs_jsonl,
    safe_float,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class SchemaVersionTest(unittest.TestCase):
    def test_schema_version_is_1(self):
        self.assertEqual(SCHEMA_VERSION, 1)


class ParseEvalSamplesJsonlTest(unittest.TestCase):
    def test_valid_jsonl_loads_three_samples_in_order(self):
        samples = load_eval_samples(str(FIXTURES / "eval_input.jsonl"))
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].id, "s1")
        self.assertEqual(samples[0].prompt, "Write a paragraph about trees.")
        self.assertEqual(
            samples[0].candidate,
            "Trees provide oxygen and shade. They are essential for life "
            "on Earth. Many animals depend on forests for survival.",
        )
        self.assertEqual(
            samples[0].reference,
            "Trees are vital. They produce oxygen and offer shade.",
        )
        self.assertEqual(samples[1].id, "s2")
        self.assertEqual(samples[1].prompt, "Explain photosynthesis.")
        self.assertEqual(
            samples[1].candidate,
            "Photosynthesis is how plants make food using sunlight. They "
            "take in carbon dioxide and water to produce glucose.",
        )
        # s2 has no reference
        self.assertIsNone(samples[1].reference)
        self.assertEqual(samples[2].id, "s3")
        self.assertEqual(samples[2].prompt, "Describe a cat.")
        self.assertIsNone(samples[2].reference)

    def test_blank_lines_are_ignored(self):
        text = (
            '{"id": "a", "prompt": "p", "candidate": "c"}\n'
            "\n"
            '{"id": "b", "prompt": "p", "candidate": "c"}\n'
            "  \n"
            '{"id": "c", "prompt": "p", "candidate": "c"}\n'
        )
        samples = parse_eval_samples_jsonl(text, source="test")
        self.assertEqual(len(samples), 3)

    def test_malformed_json_raises_value_error_with_line_number(self):
        text = (
            '{"id": "i1", "prompt": "Test", "candidate": "Valid"}\n'
            "not valid json\n"
            '{"id": "i3", "prompt": "", "candidate": "empty prompt"}\n'
        )
        with self.assertRaises(ValueError) as ctx:
            parse_eval_samples_jsonl(text, source="test")
        self.assertIn("line 2", str(ctx.exception))

    def test_missing_required_field_raises_value_error(self):
        text = '{"id": "m1", "candidate": "missing prompt"}\n'
        with self.assertRaises(ValueError) as ctx:
            parse_eval_samples_jsonl(text, source="test")
        self.assertIn("prompt", str(ctx.exception))

    def test_non_string_required_field_raises_value_error(self):
        text = '{"id": 42, "prompt": "p", "candidate": "c"}\n'
        with self.assertRaises(ValueError) as ctx:
            parse_eval_samples_jsonl(text, source="test")
        self.assertIn("id", str(ctx.exception))
        self.assertIn("line 1", str(ctx.exception))

    def test_empty_required_string_raises_value_error(self):
        text = '{"id": "e1", "prompt": "", "candidate": "c"}\n'
        with self.assertRaises(ValueError) as ctx:
            parse_eval_samples_jsonl(text, source="test")
        self.assertIn("empty", str(ctx.exception))
        self.assertIn("prompt", str(ctx.exception))
        self.assertIn("line 1", str(ctx.exception))

    def test_duplicate_ids_raise_duplicate_id_error(self):
        with self.assertRaises(DuplicateIdError) as ctx:
            load_eval_samples(str(FIXTURES / "eval_duplicate_ids.jsonl"))
        self.assertEqual(str(ctx.exception.duplicate_id), "d1")


class ParsePreferencePairsJsonlTest(unittest.TestCase):
    def test_valid_preference_labels_load_three_pairs(self):
        pairs = load_preference_pairs(str(FIXTURES / "preference_labels.jsonl"))
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0].id, "p1")
        self.assertEqual(pairs[0].preferred, "a")
        self.assertEqual(pairs[1].id, "p2")
        self.assertEqual(pairs[1].preferred, "b")
        self.assertEqual(pairs[2].id, "p3")
        self.assertEqual(pairs[2].preferred, "tie")

    def test_invalid_preferred_value_raises_value_error(self):
        text = (
            '{"id": "x1", "prompt": "p", "candidate_a": "a", '
            '"candidate_b": "b", "preferred": "c"}\n'
        )
        with self.assertRaises(ValueError) as ctx:
            parse_preference_pairs_jsonl(text, source="test")
        self.assertIn("'c'", str(ctx.exception))

    def test_missing_preferred_field_raises_value_error(self):
        text = (
            '{"id": "x2", "prompt": "p", "candidate_a": "a", '
            '"candidate_b": "b"}\n'
        )
        with self.assertRaises(ValueError) as ctx:
            parse_preference_pairs_jsonl(text, source="test")
        self.assertIn("preferred", str(ctx.exception))

    def test_empty_preferred_string_raises_value_error(self):
        text = (
            '{"id": "x3", "prompt": "p", "candidate_a": "a", '
            '"candidate_b": "b", "preferred": ""}\n'
        )
        with self.assertRaises(ValueError) as ctx:
            parse_preference_pairs_jsonl(text, source="test")
        self.assertIn("empty", str(ctx.exception))


class SafeFloatTest(unittest.TestCase):
    def test_nan_returns_none(self):
        self.assertIsNone(safe_float(float("nan")))

    def test_infinity_returns_none(self):
        self.assertIsNone(safe_float(float("inf")))

    def test_normal_float_returns_value(self):
        self.assertEqual(safe_float(0.5), 0.5)

    def test_none_returns_none(self):
        self.assertIsNone(safe_float(None))

    def test_negative_infinity_returns_none(self):
        self.assertIsNone(safe_float(float("-inf")))


class JsonDumpsRecordTest(unittest.TestCase):
    def test_deterministic_sorted_keys(self):
        record = {"z": 1, "a": 2, "m": 3}
        result = json_dumps_record(record)
        # sort_keys=True → "a", "m", "z"
        self.assertEqual(result, '{"a": 2, "m": 3, "z": 1}')

    def test_ensure_ascii_false_preserves_unicode(self):
        record = {"msg": "héllo"}
        result = json_dumps_record(record)
        self.assertIn("héllo", result)

    def test_consistent_output(self):
        a = json_dumps_record({"b": 1, "a": 2})
        b = json_dumps_record({"a": 2, "b": 1})
        self.assertEqual(a, b)


class LoadEvalSamplesFileTest(unittest.TestCase):
    def test_load_from_file_returns_correct_count(self):
        samples = load_eval_samples(str(FIXTURES / "eval_input.jsonl"))
        self.assertEqual(len(samples), 3)

    def test_load_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_eval_samples("/nonexistent/path.jsonl")


class LoadPreferencePairsFileTest(unittest.TestCase):
    def test_load_from_file_returns_correct_count(self):
        pairs = load_preference_pairs(str(FIXTURES / "preference_labels.jsonl"))
        self.assertEqual(len(pairs), 3)

    def test_load_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_preference_pairs("/nonexistent/path.jsonl")


class DataclassTest(unittest.TestCase):
    def test_eval_sample_default_reference_is_none(self):
        s = EvalSample(id="x", prompt="p", candidate="c")
        self.assertIsNone(s.reference)
        self.assertEqual(s.id, "x")
        self.assertEqual(s.prompt, "p")
        self.assertEqual(s.candidate, "c")

    def test_preference_pair_fields(self):
        p = PreferencePair(
            id="x",
            prompt="p",
            candidate_a="a",
            candidate_b="b",
            preferred="tie",
        )
        self.assertEqual(p.id, "x")
        self.assertEqual(p.prompt, "p")
        self.assertEqual(p.candidate_a, "a")
        self.assertEqual(p.candidate_b, "b")
        self.assertEqual(p.preferred, "tie")


if __name__ == "__main__":
    unittest.main()
