"""Tests for humanify.artifacts -- detector-of-humanizers layer (Task C, Phase 4.3).

Pure-python heuristic findings; no models are involved.
"""

import json
import unittest

from humanify.artifacts import detect_artifacts

_ZERO_WIDTH_SAMPLE = (
    "This sentence has a zero​width joiner hidden inside the word field."
    " Another invisible \u200c non-joiner and a BOM\ufeff appear later."
)

_HOMOGLYPH_SAMPLE = (
    "The сoursе was restarted and the аccount was rebuilt with thе new form."
    " The mоdulе and іntegratіon steps are reviewed агаin today."
)

# Constructed low-vocabulary / heavy mid-frequency repetition (synonym cycling).
_SYNONYM_SAMPLE = " ".join(
    "the system delivers robust output the system delivers robust"
    for _ in range(12)
)

_CLEAN_SAMPLE = (
    "The old ferry moors at the far pier each morning to offload crates of "
    "grain and timber. Children watch the ropes slow mist billow around the "
    "anchors while gulls wheel overhead. By noon the tide turns and the "
    "captains sound their worn brass horns, and the whole quay hums with the "
    "clatter of hoists and cartwheels that have done this work for decades."
)


class ArtifactsZeroWidthTest(unittest.TestCase):
    def test_zero_width_sample_fires(self):
        result = detect_artifacts(_ZERO_WIDTH_SAMPLE)
        findings = {f["name"]: f for f in result["findings"]}
        self.assertIn("zero_width_chars", findings)
        self.assertEqual(findings["zero_width_chars"]["kb_ref"], "T10")
        self.assertIn(findings["zero_width_chars"]["severity"], ("low", "medium", "high"))


class ArtifactsHomoglyphTest(unittest.TestCase):
    def test_cyrillic_homoglyph_sample_fires(self):
        result = detect_artifacts(_HOMOGLYPH_SAMPLE)
        findings = {f["name"]: f for f in result["findings"]}
        self.assertIn("unicode_homoglyphs", findings)
        self.assertEqual(findings["unicode_homoglyphs"]["kb_ref"], "T10")


class ArtifactsCleanTest(unittest.TestCase):
    def test_clean_text_has_no_findings(self):
        result = detect_artifacts(_CLEAN_SAMPLE)
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["editlens_recommended"])


class MechanicalSynonymCyclingTest(unittest.TestCase):
    def test_constructed_repeated_vocab_fires(self):
        result = detect_artifacts(_SYNONYM_SAMPLE)
        findings = {f["name"]: f for f in result["findings"]}
        self.assertIn("mechanical_synonym_cycling", findings)
        self.assertEqual(findings["mechanical_synonym_cycling"]["kb_ref"], "S4")

    def test_editlens_not_recommended_for_synonym_only(self):
        result = detect_artifacts(_SYNONYM_SAMPLE)
        # zero-width and homoglyphs absent -> no editlens recommendation
        self.assertFalse(result["editlens_recommended"])

    def test_clean_text_does_not_fire_synonym(self):
        result = detect_artifacts(_CLEAN_SAMPLE)
        self.assertNotIn("mechanical_synonym_cycling",
                         [f["name"] for f in result["findings"]])


class ArtifactsEditLensRecommendationTest(unittest.TestCase):
    def test_post_processed_zero_width_recommends_editlens(self):
        result = detect_artifacts(_ZERO_WIDTH_SAMPLE)
        self.assertTrue(result["editlens_recommended"])
        names = {f["name"] for f in result["findings"]}
        self.assertIn("high_edit_magnitude_risk", names)
        risk = next(f for f in result["findings"]
                    if f["name"] == "high_edit_magnitude_risk")
        self.assertEqual(risk["kb_ref"], "edit-magnitude")


class ArtifactsSchemaTest(unittest.TestCase):
    def test_schema_and_bounds(self):
        for doc in ("", "a", _CLEAN_SAMPLE, _HOMOGLYPH_SAMPLE):
            result = detect_artifacts(doc)
            self.assertIn("findings", result)
            self.assertIn("score", result)
            self.assertIn("editlens_recommended", result)
            self.assertIsInstance(result["findings"], list)
            self.assertGreaterEqual(result["score"], 0.0)
            self.assertLessEqual(result["score"], 1.0)
            for f in result["findings"]:
                self.assertIn("name", f)
                self.assertIn("severity", f)
                self.assertIn("evidence", f)
                self.assertIn("kb_ref", f)
                self.assertIn(f["severity"], ("low", "medium", "high"))


if __name__ == "__main__":
    unittest.main()


class ArtifactsCliWiringTest(unittest.TestCase):
    """Task C wiring: consume artifact_audit end-to-end via JSON."""

    def test_score_text_includes_artifact_audit(self):
        from humanify.cli import _score_text

        class FakeDet:
            name = "fake"
            model_id = "fake-model"
            def load(self, *a, **k): return None
            def score(self, texts, batch_size=32): return [0.5] * len(texts)
            def set_model_id(self, *a, **k): return None
            def get_causal_lm(self): return None

        raw = (
            "A perfectly normal body sentence that carries the meaning forward. "
            "Another ordinary sentence adds a little more detail here nicely."
        )
        result = _score_text(
            raw,
            loaded_detectors=[{"name": "fake", "detector": FakeDet(), "model_id": "fake"}],
            batch_size=32,
        )
        self.assertIn("artifact_audit", result)
        self.assertIn("findings", result["artifact_audit"])
        self.assertIn("score", result["artifact_audit"])

    def test_output_json_emits_artifact_audit_key(self):
        from io import StringIO
        from unittest.mock import patch
        from humanify.output import output_json

        audit = {
            "score": 0.4,
            "findings": [
                {"name": "zero_width_chars", "severity": "medium",
                 "evidence": "2 zero-width chars", "kb_ref": "T10"},
            ],
            "editlens_recommended": True,
        }
        buf = StringIO()
        with patch("sys.stdout", buf):
            output_json(["sentence."], [0.5], [[]], {"mean": 0.5, "std": 0.0,
                                                     "min": 0.5, "max": 0.5},
                        0.5, "fake", artifact_audit=audit)
        data = json.loads(buf.getvalue())
        self.assertIn("artifact_audit", data)
        self.assertEqual(data["artifact_audit"], audit)
