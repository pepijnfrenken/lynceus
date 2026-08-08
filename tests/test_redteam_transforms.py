"""Tests for the red-team sampling attack transforms (Task A / Phase 7.1).

No Ollama / network is touched: ``_ollama_generate`` is monkeypatched and the
tests only assert the option dicts passed through to it, plus that the
``add_transitions`` tell-injecting transform is gone from ``ALL_TRANSFORMS``.
"""

import inspect
import json
import unittest
from unittest.mock import patch

from redteam import transforms as T


class SamplingOptionsPassedTest(unittest.TestCase):
    """Each sampling transform must call ``_ollama_generate`` with the right
    options dict and return its mocked output."""

    def test_sample_temperature_calls_with_temperature_option(self):
        with patch("redteam.transforms._ollama_generate",
                   return_value="MOCKED_REGEN") as gen:
            out = T.sample_temperature("some text", model="m-test", temperature=1.4)
        self.assertEqual(out, "MOCKED_REGEN")
        _args, kwargs = gen.call_args
        self.assertEqual(kwargs["model"], "m-test")
        self.assertEqual(kwargs["options"], {"temperature": 1.4})

    def test_sample_rep_penalty_calls_with_repeat_penalty_option(self):
        with patch("redteam.transforms._ollama_generate",
                   return_value="MOCKED_REP") as gen:
            out = T.sample_rep_penalty("some text", model="m-test", rep_penalty=1.08)
        self.assertEqual(out, "MOCKED_REP")
        _args, kwargs = gen.call_args
        self.assertEqual(kwargs["options"], {"repeat_penalty": 1.08})

    def test_sample_top_p_calls_with_top_p_option(self):
        with patch("redteam.transforms._ollama_generate",
                   return_value="MOCKED_P") as gen:
            out = T.sample_top_p("some text", model="m-test", top_p=0.9)
        self.assertEqual(out, "MOCKED_P")
        _args, kwargs = gen.call_args
        self.assertEqual(kwargs["options"], {"top_p": 0.9})

    def test_defaults_are_the_kb_recommended_values(self):
        # KB 03 3.2: T~1.4, mild repetition penalty 1.05-1.15, top-p ~0.9
        def _default(fn, param):
            return inspect.signature(fn).parameters[param].default
        self.assertEqual(_default(T.sample_temperature, "temperature"), 1.4)
        self.assertEqual(_default(T.sample_rep_penalty, "rep_penalty"), 1.08)
        self.assertEqual(_default(T.sample_top_p, "top_p"), 0.9)

    def test_ollama_generate_builds_merged_options_payload(self):
        """Lower the hood: _ollama_generate must merge the passed sampling
        options over the default temperature 0.7 in the HTTP payload."""
        captured = {}

        class FakeResp:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=60):
            captured["payload"] = json.loads(req.data)
            return FakeResp(json.dumps({"response": "OK REGEN"}).encode())

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = T.sample_top_p("text", model="m", top_p=0.9)

        self.assertEqual(out, "OK REGEN")
        opts = captured["payload"]["options"]
        self.assertEqual(opts["top_p"], 0.9)
        self.assertEqual(opts["temperature"], 0.7)  # default retained


class AllTransformsTest(unittest.TestCase):
    def test_add_transitions_not_in_all_transforms(self):
        """KB 02 S10: add_transitions INJECTS an AI tell; it must not ship in
        the default transform arsenal."""
        names = [n for n, _ in T.ALL_TRANSFORMS]
        self.assertNotIn("add_transitions", names)

    def test_add_transitions_still_available_for_negative_testing(self):
        # Keep the function callable (the KB suggests it can be used to build
        # a tell-injecting negative-control sample), just not auto-applied.
        self.assertTrue(callable(T.add_transitions))


class SamplingTransformsListTest(unittest.TestCase):
    def test_sampling_transforms_contains_the_three_new_attacks(self):
        names = [n for n, _ in T.SAMPLING_TRANSFORMS]
        self.assertIn("sample_temperature", names)
        self.assertIn("sample_rep_penalty", names)
        self.assertIn("sample_top_p", names)
        # names match the strategy names the loop dispatches on
        for name, _ in T.SAMPLING_TRANSFORMS:
            self.assertIn(name, T.SAMPLING_TRANSFORM_MAP)

    def test_apply_sampling_transform_routes_by_name(self):
        from unittest.mock import MagicMock
        mock_fn = MagicMock(return_value="RT")
        with patch.dict(T.SAMPLING_TRANSFORM_MAP,
                        {"sample_temperature": mock_fn}):
            out, applied = T.apply_sampling_transform("t", "sample_temperature",
                                                      model="mrm")
        mock_fn.assert_called_once()
        _a, kw = mock_fn.call_args
        self.assertEqual(kw["model"], "mrm")
        self.assertEqual(out, "RT")
        self.assertEqual(applied, ["sample_temperature"])

    def test_apply_sampling_transform_unknown_name_is_noop(self):
        out, applied = T.apply_sampling_transform("t", "not_a_transform")
        self.assertEqual(out, "t")
        self.assertEqual(applied, [])


class ComboSampleAllTest(unittest.TestCase):
    def test_combo_chains_all_three(self):
        called_opts = []

        def fake_gen(prompt, model="qwen3:1.7b", timeout=60, options=None):
            called_opts.append(dict(options or {}))
            return "REGEN_TEXT"

        with patch("redteam.transforms._ollama_generate",
                   side_effect=fake_gen) as gen:
            out = T.combo_sample_all("input", model="mll")
        self.assertEqual(out, "REGEN_TEXT")
        self.assertEqual(gen.call_count, len(T.SAMPLING_TRANSFORMS))
        # each of the three attack options must be exercised, in order
        keys = [set(o.keys()) for o in called_opts]
        self.assertIn({"temperature"}, keys)
        self.assertIn({"repeat_penalty"}, keys)
        self.assertIn({"top_p"}, keys)


if __name__ == "__main__":
    unittest.main()
