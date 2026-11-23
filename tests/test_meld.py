"""Tests for the MELD detector.

No weights are downloaded here: the Ettin backbone is faked via
``AutoModel.from_config``, checkpoint loading is bypassed (``load_state_dict``
no-op'd), and the hub is never touched (``snapshot_download`` is asserted
against in tests that exercise ``LYNCEUS_MELD_DIR``). The custom scoring head
(style projection, anchor/prototype distances, top-rho aggregation) runs real
torch math on the fake backbone, so determinism, bounds, calibration, and
flagging all exercise the actual head code.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from lynceus.detectors import DETECTORS
from lynceus.detectors.meld import (
    Meld,
    MeldDetector,
    _calibrate_short_text,
    MIN_RELIABLE_WORDS,
)


def _write_fake_checkpoint(tmpdir, **overrides):
    """Write a fake meld checkout dir with a minimal meld_config.json."""
    cfg = {
        "style_rank": 8,
        "backbone_hidden_size": 4,
        "n_human_anchors": 4,
        "n_families": 3,
        "n_ops": 2,
        "max_length": 64,
        "rho": 0.25,
        "tau_agg": 1.0,
        "score_offsets": {
            "overall": {"fpr_0.01": 0.0},
            "strata": {"academic": {"fpr_0.01": 0.5}},
        },
    }
    cfg.update(overrides)
    model_dir = os.path.join(tmpdir, "meld")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "meld_config.json"), "w") as f:
        json.dump(cfg, f)
    return model_dir


class FakeBackboneOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeBackbone(torch.nn.Module):
    """Deterministic content-dependent hidden states; no real weights.

    Each token's hidden row depends only on its ``input_ids`` value, so
    different texts get different style coordinates and the whole custom head
    runs real math without a downloaded backbone.
    """

    def forward(self, input_ids=None, attention_mask=None):
        h = input_ids.float().unsqueeze(-1) * torch.tensor(
            [1.0, 0.5, 0.25, 0.125]
        )
        pos = torch.arange(input_ids.shape[1], dtype=h.dtype).view(1, -1, 1)
        h = h + pos * 0.01
        return FakeBackboneOutput(h)


class FakeEncoding(dict):
    """dict-like BatchEncoding stand-in with the .to() transformers API."""

    def to(self, device):
        if device in ("cpu", None):
            return self
        return type(self)({k: v.to(device) for k, v in self.items()})


class FakeMeldTokenizer:
    """Dummy tokenizer: 4 fake tokens per text, content-dependent ids."""

    pad_token_id = 1
    eos_token_id = 2

    def __call__(self, texts, **kw):
        n, length = len(texts), 4
        ids = torch.zeros(n, length, dtype=torch.long)
        for i, t in enumerate(texts):
            ids[i] = sum(len(w) for w in t.split()) % 13
        return FakeEncoding({
            "input_ids": ids,
            "attention_mask": torch.ones(n, length, dtype=torch.long),
            "special_tokens_mask": torch.zeros(n, length, dtype=torch.long),
        })


def _make_meld_detector(tmpdir, **cfg_overrides):
    """Build a loaded MeldDetector over a fake checkpoint, no downloads."""
    model_dir = _write_fake_checkpoint(tmpdir, **cfg_overrides)
    det = MeldDetector()
    with patch("lynceus.detectors.meld.AutoConfig.from_pretrained",
               return_value=None), \
         patch("lynceus.detectors.meld.AutoModel.from_config",
               return_value=FakeBackbone()), \
         patch.object(Meld, "load_state_dict", return_value=None), \
         patch("lynceus.detectors.meld.load_file", return_value={}), \
         patch("lynceus.detectors.meld.AutoTokenizer.from_pretrained",
               return_value=FakeMeldTokenizer()), \
         patch("lynceus.detectors.meld.snapshot_download",
               side_effect=AssertionError("tests must not download weights")), \
         patch.dict(os.environ, {"LYNCEUS_MELD_DIR": model_dir}):
        det.load("cpu", fp16=False)
    return det


LONG_TEXT = " ".join(f"word{i}" for i in range(150))  # > MIN_RELIABLE_WORDS


class MeldRegistryTest(unittest.TestCase):
    def test_registry_contains_meld(self):
        self.assertIn("meld", DETECTORS)
        self.assertTrue(issubclass(DETECTORS["meld"], MeldDetector))

    def test_description_notes_raid_rank_and_license(self):
        desc = MeldDetector().description.lower()
        self.assertIn("raid", desc)
        self.assertIn("mit", desc)


class MeldModelIdTest(unittest.TestCase):
    def test_default_model_id(self):
        det = MeldDetector()
        self.assertEqual(det.model_id, "anon-review-meld-2026/meld")

    def test_set_model_id_overrides(self):
        det = MeldDetector()
        det.set_model_id("local/fake-meld")
        self.assertEqual(det.model_id, "local/fake-meld")


class MeldCalibrationTest(unittest.TestCase):
    def test_short_text_calibration_pulls_toward_neutral(self):
        # Model card: >= 100 words minimum; below ~50 words MELD flags most
        # genuine human writing, so short readings blend toward neutral 0.5.
        self.assertAlmostEqual(_calibrate_short_text(0.95, 10), 0.5 + 0.45 * 0.1)
        # At/above the threshold the raw score is returned unchanged.
        self.assertAlmostEqual(_calibrate_short_text(0.95, MIN_RELIABLE_WORDS), 0.95)
        self.assertAlmostEqual(_calibrate_short_text(0.2, 150), 0.2)
        # Monotone: more words -> closer to raw.
        self.assertLess(
            _calibrate_short_text(0.95, 20), _calibrate_short_text(0.95, 50)
        )


class MeldLoadAndScoreTest(unittest.TestCase):
    def test_load_sets_threshold_from_config(self):
        with tempfile.TemporaryDirectory() as td:
            det = _make_meld_detector(td)
        self.assertEqual(det.threshold, 0.0)
        self.assertIn("academic", det.strata_thresholds)

    def test_score_returns_bounded_float_per_text(self):
        with tempfile.TemporaryDirectory() as td:
            det = _make_meld_detector(td)
            texts = [LONG_TEXT, LONG_TEXT, LONG_TEXT]
            scores = det.score(texts, batch_size=2)
        self.assertEqual(len(scores), 3)
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)
        self.assertIsNotNone(det.last_raw_scores)
        self.assertEqual(len(det.last_raw_scores), 3)

    def test_score_is_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            det = _make_meld_detector(td)
            first = det.score([LONG_TEXT, LONG_TEXT], batch_size=2)
            second = det.score([LONG_TEXT, LONG_TEXT], batch_size=2)
        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 0)

    def test_score_calibrates_short_text_to_match_blend_formula(self):
        with tempfile.TemporaryDirectory() as td:
            det = _make_meld_detector(td)
            scores = det.score(["short text"], batch_size=1)  # 2 words
            # Recover the uncalibrated P(AI) and check the exact blend.
            raw = det._model.score(["short text"], det._tokenizer, "cpu")
        raw_p = 1.0 / (1.0 + pow(2.718281828459045, -raw[1][0]))
        expected = 0.5 + (raw_p - 0.5) * (2 / MIN_RELIABLE_WORDS)
        self.assertAlmostEqual(scores[0], expected, places=6)
        self.assertLessEqual(abs(scores[0] - 0.5), abs(raw_p - 0.5))

    def test_score_unloaded_raises_runtimeerror(self):
        det = MeldDetector()
        with self.assertRaises(RuntimeError):
            det.score(["some text"])


class MeldFlaggingTest(unittest.TestCase):
    def test_flag_units_compares_raw_to_fpr_threshold(self):
        det = MeldDetector()
        det.threshold = 0.5
        det.last_raw_scores = [0.8, 0.2, 0.5]
        self.assertEqual(det.flag_units(), [True, False, False])

    def test_flag_units_before_load_raises_runtimeerror(self):
        det = MeldDetector()
        with self.assertRaises(RuntimeError):
            det.flag_units()


class MeldScoringUnitsConventionTest(unittest.TestCase):
    def test_units_chunking_and_expansion_match_harness(self):
        from lynceus.analysis import prepare_scoring_units
        from lynceus.cli import _expand_unit_scores

        raw = " ".join(f"sentence number {i} with a few ordinary words." for i in range(60))
        sentences, units = prepare_scoring_units(raw)
        # 60 * ~7 words => several 320-word-max units covering all sentences.
        self.assertGreater(len(units), 1)
        self.assertTrue(all(len(u.text.split()) <= 320 for u in units))
        self.assertEqual(units[0].start, 0)
        self.assertEqual(units[-1].stop, len(sentences))

        with tempfile.TemporaryDirectory() as td:
            det = _make_meld_detector(td)
            unit_texts = [u.text for u in units]
            scores = det.score(unit_texts, batch_size=2)
            self.assertEqual(len(scores), len(units))
            self.assertEqual(len(det.last_raw_scores), len(units))
            # Expansion maps one detector score per unit back to one per
            # sentence, contiguously (the CLI/evader byte-identical contract).
            expanded = _expand_unit_scores(units, scores, len(sentences))
            self.assertEqual(len(expanded), len(sentences))
            self.assertEqual(expanded[0], scores[0])
            self.assertEqual(expanded[-1], scores[-1])


class MeldCliTest(unittest.TestCase):
    def test_cli_accepts_meld_model(self):
        from lynceus.cli import _load_detectors

        with patch("lynceus.detectors.meld.MeldDetector.load") as fake_load:
            loaded = _load_detectors(
                ["meld"], model_id=None, device_name="cpu", suppress_stdout=True
            )
        fake_load.assert_called_once_with("cpu", fp16=False)
        self.assertEqual(loaded[0]["name"], "meld")
        self.assertEqual(loaded[0]["model_id"], "anon-review-meld-2026/meld")

    def test_cli_unknown_model_still_rejected(self):
        from lynceus.cli import _load_detectors

        with self.assertRaises(ValueError):
            _load_detectors(["meld-typo"], model_id=None, device_name="cpu")


class MeldAxisWiringTest(unittest.TestCase):
    def test_meld_maps_to_classifier_axis(self):
        from lynceus.cli import _axis_name

        # A supervised classifier-style scorer; excluding it from the axis
        # would silently leave the per-layer "overall" disagreeing with the
        # headline detector mean (observed in the real-weights smoke test).
        self.assertEqual(_axis_name("meld"), "classifier_axis")
        self.assertIsNone(_axis_name("tmr-nope-imaginary"))


if __name__ == "__main__":
    unittest.main()