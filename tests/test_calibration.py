import json
import unittest



from humanify.calibration import DOMAINS, calibrate


class CalibrationTest(unittest.TestCase):
    def test_academic_adds_warning_on_low_burstiness(self):
        result = calibrate({"stats": {"std": 0.1, "mean": 0.9}}, "academic")
        self.assertEqual(result["domain"], "academic")
        joined = " ".join(result["domain_warnings"]).lower()
        self.assertIn("false-positive", joined)
        self.assertIn("burstiness", joined)

    def test_academic_high_burstiness_no_runtime_warning(self):
        # still carries the static domain warning, but not the low-burstiness one
        result = calibrate({"stats": {"std": 0.8, "mean": 0.5}}, "academic")
        # exactly one warning (the static formal-prose note), no runtime one
        self.assertEqual(len(result["domain_warnings"]), 1)
        joined = " ".join(result["domain_warnings"]).lower()
        self.assertIn("not necessarily ai", joined)
        self.assertNotIn("< 0.30", joined)

    def test_unknown_domain_raises_value_error(self):
        with self.assertRaises(ValueError):
            calibrate({}, "martian")

    def test_calibrate_never_removes_keys(self):
        original = {"a": 1, "stats": {"std": 0.2, "mean": 0.7}, "b": {"nested": 1}}
        out = calibrate(original, "social")
        for k in original:
            self.assertIn(k, out)
            self.assertEqual(out[k], original[k])
        # input dict not mutated
        self.assertNotIn("domain", original)

    def test_social_domain_has_slower_sentence_threshold(self):
        out = calibrate({"stats": {"std": 0.9, "mean": 0.3}}, "social")
        self.assertLess(out["domain_settings"]["sentence_threshold"], 0.5)

    def test_all_domains_listed_and_callable(self):
        for domain in DOMAINS:
            out = calibrate({"stats": {"std": 0.4, "mean": 0.5}}, domain)
            self.assertEqual(out["domain"], domain)
            self.assertIn("domain_settings", out)
            self.assertIn("domain_warnings", out)

class CalibrationOutputJsonTest(unittest.TestCase):
    def test_output_json_includes_domain_keys(self):
        from io import StringIO
        from unittest.mock import patch
        from humanify.output import output_json

        calibrated = calibrate({"stats": {"std": 0.12, "mean": 0.9}}, "academic")
        buf = StringIO()
        with patch("sys.stdout", buf):
            output_json(
                ["This formal sentence is quite uniform and neat."],
                [0.9], [[]],
                {"mean": 0.9, "std": 0.12, "min": 0.9, "max": 0.9},
                0.5, "fake",
                domain=calibrated["domain"],
                domain_settings=calibrated["domain_settings"],
                domain_warnings=calibrated["domain_warnings"],
            )
        data = json.loads(buf.getvalue())
        self.assertEqual(data["domain"], "academic")
        self.assertIn("domain_warnings", data)
        self.assertTrue(any("false-positive" in w for w in data["domain_warnings"]))
