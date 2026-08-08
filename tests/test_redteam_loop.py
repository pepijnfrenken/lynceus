"""Wiring tests: the new Phase-7 strategies are recognisable by
``RedTeamLoop.get_strategy_text`` and route to the right transform. No network,
no real detector — ``_ollama_generate`` / ``llm_rewrite`` / the loop's detector
are all faked.
"""

import unittest
from unittest.mock import patch

from redteam.loop import RedTeamConfig, RedTeamLoop
from redteam.detectors import DetectionResult


def _make_loop():
    cfg = RedTeamConfig(detectors=["humanify"])
    loop = RedTeamLoop(cfg)
    # never run a real subprocess detector in tests
    loop.detect = lambda text: {
        "humanify": DetectionResult("humanify", 0.0),
    }
    return loop


class LoopStrategyDispatchTest(unittest.TestCase):
    def test_sampling_temperature_strategy_routes(self):
        loop = _make_loop()
        with patch("redteam.transforms._ollama_generate",
                   return_value="SAMP_OUT") as gen:
            out, applied = loop.get_strategy_text("sample_temperature", "t")
        gen.assert_called_once()
        self.assertEqual(gen.call_args[1]["options"]["temperature"], 1.4)
        self.assertEqual(out, "SAMP_OUT")
        self.assertEqual(applied, ["sample_temperature"])

    def test_combo_sample_all_strategy_routes(self):
        loop = _make_loop()
        called = {"n": 0}

        def fake_gen(prompt, model="q", timeout=60, options=None):
            called["n"] += 1
            return "C"
        with patch("redteam.transforms._ollama_generate", side_effect=fake_gen):
            out, applied = loop.get_strategy_text("combo_sample_all", "in")
        self.assertEqual(called["n"], 3)
        self.assertEqual(out, "C")
        self.assertEqual(applied, ["combo_sample_all"])

    def test_paraphrase_verify_strategy_routes(self):
        loop = _make_loop()
        loop.detect = lambda _t: {"humanify": DetectionResult("humanify", 0.1 if "ROUND1" in _t else 0.9)}
        with patch("redteam.transforms.llm_rewrite",
                   side_effect=lambda t, **kw: t + " ROUND1"), \
             patch("redteam.transforms._ollama_generate", return_value="x"):
            out, applied = loop.get_strategy_text("paraphrase_verify",
                                                  "Some input text here.")
        self.assertIn("ROUND1", out)
        self.assertEqual(applied, ["paraphrase_verify"])

    def test_guided_optimize_strategy_routes(self):
        loop = _make_loop()
        loop.detect = lambda _t: {"humanify": DetectionResult("humanify", 0.1 if "gen_candidate" in _t else 0.9)}
        with patch("redteam.transforms._ollama_generate",
                   return_value="gen_candidate"):
            out, applied = loop.get_strategy_text("guided_optimize", "in")
        self.assertEqual(applied, ["guided_optimize"])
        self.assertEqual(out, "gen_candidate")

    def test_new_strategies_in_builtin_strategy_list(self):
        import inspect
        src = inspect.getsource(RedTeamLoop.run)
        for strat in ("paraphrase_verify", "guided_optimize",
                      "sample_temperature", "sample_rep_penalty",
                      "sample_top_p", "combo_sample_all"):
            self.assertIn(strat, src)


if __name__ == "__main__":
    unittest.main()
