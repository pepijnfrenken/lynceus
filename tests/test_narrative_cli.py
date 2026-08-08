"""CLI-level tests for the narrative layer (Task B, Phase 5.2).

We exercise the ``--advice narrative`` wiring without any real LLM: fake
detectors (whose ``get_causal_lm`` returns a dummy causal LM so no Qwen model is
downloaded) plus a monkeypatched ``run_narrative_checklist``. os._exit is
turned into a SystemExit so the CLI run terminates cleanly in the test.
"""

import io
import json
import unittest
from unittest.mock import patch

import humanify.cli
from humanify.output import output_json


class FakeNarrativeDetector:
    """Detector stub carrying a dummy causal LM so no real model is loaded."""

    def __init__(self, p=0.5):
        self._p = p

    @property
    def name(self):
        return "fake"

    @property
    def model_id(self):
        return "fake-model"

    def load(self, device="cpu", fp16=False):
        return None

    def set_model_id(self, model_id):
        return None

    def score(self, texts, batch_size=32):
        return [self._p] * len(texts)

    def get_causal_lm(self):
        # Dummy (model, tokenizer, device); never actually invoked because the
        # checklist is monkeypatched in the enabled case, disabled in the short
        # case.
        return (object(), object(), "cpu")


LONG_TEXT = " ".join(
    "The ferry slips past the quiet headland while gulls wheel through the "
    "fog and the engines thrum like a slow human heartbeat."
    for _ in range(60)
)


def _fake_loaded(det=None):
    return [{"name": "fake", "detector": det or FakeNarrativeDetector(),
             "model_id": "fake-model"}]


class NarrativeCliOutputJsonTest(unittest.TestCase):
    def test_output_json_emits_narrative_key_when_present(self):
        narrative = {
            "enabled": True, "reason": None,
            "flagged": [{"key": "theme_over_explanation", "direction": "ai_high",
                         "evidence": "spells out the lesson"}],
            "score": 0.3, "dimensions": [],
        }
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            output_json(["sentence one here."], [0.5], [[]],
                        {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
                        0.5, "fake", narrative=narrative)
        data = json.loads(buf.getvalue())
        self.assertIn("narrative", data)
        self.assertEqual(data["narrative"]["score"], 0.3)


class NarrativeCliGateTest(unittest.TestCase):
    def _run(self, text, mm_checklist=None, detector=None):
        buf = io.StringIO()
        patchers = [
            patch("humanify.cli._load_detectors",
                  return_value=_fake_loaded(detector)),
            patch("humanify.cli.os._exit", side_effect=SystemExit),
            patch("humanify.cli.read_input", return_value=text),
            patch("sys.stdout", buf),
        ]
        if mm_checklist is not None:
            patchers.append(
                patch("humanify.narrative_checklist.run_narrative_checklist",
                      return_value=mm_checklist)
            )
        for p in patchers:
            p.start()
        try:
            with self.assertRaises(SystemExit):
                humanify.cli._main_single(
                    ["--advice", "narrative", "--json", "--file", "-"]
                )
        finally:
            for p in patchers[::-1]:
                p.stop()
        return json.loads(buf.getvalue())

    def test_narrative_on_short_input_emits_disabled_without_model(self):
        # A detector whose get_causal_lm raises proves short input never asks
        # for a causal LM (and so never triggers a Qwen download).
        class NoLmDetector(FakeNarrativeDetector):
            def get_causal_lm(self):
                raise AssertionError("causal LM must not be requested on short input")

        data = self._run("I'm far too short for narrative.", detector=NoLmDetector())
        self.assertIn("narrative", data)
        self.assertFalse(data["narrative"]["enabled"])
        self.assertEqual(data["narrative"]["reason"], "too_short")

    def test_narrative_enabled_emits_narrative_key(self):
        enabled = {"enabled": True, "reason": None, "dimensions": [],
                   "flagged": [], "score": 0.4}
        data = self._run(LONG_TEXT, mm_checklist=enabled)
        self.assertIn("narrative", data)
        self.assertTrue(data["narrative"]["enabled"])


if __name__ == "__main__":
    unittest.main()
