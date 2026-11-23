"""Tests that README documents the evaluation mode correctly."""

from pathlib import Path
import unittest


README = Path("README.md").read_text(encoding="utf-8")
EVAL_SECTION_START = README.find("## Evaluation mode")


class ReadmeEvalDocsTest(unittest.TestCase):
    def test_readme_has_eval_mode_section(self):
        self.assertGreater(
            EVAL_SECTION_START, 0,
            "README must have an '## Evaluation mode' section",
        )

    def test_eval_section_mentions_input_flag(self):
        self.assertIn("--input", README[EVAL_SECTION_START:])

    def test_eval_section_mentions_preferences_flag(self):
        self.assertIn("--preferences", README[EVAL_SECTION_START:])

    def test_eval_section_describes_composite_weights(self):
        section = README[EVAL_SECTION_START:]
        self.assertIn("0.40", section)
        self.assertIn("0.25", section)
        self.assertIn("0.20", section)
        self.assertIn("0.15", section)

    def test_eval_section_states_metrics_only(self):
        self.assertIn("metrics-only", README[EVAL_SECTION_START:])

    def test_eval_section_does_not_advise_evasion(self):
        eval_section = README[EVAL_SECTION_START:]
        # The section explicitly says "no rewrite advice" — that's acceptable.
        # But it must not contain advice for evading detection.
        for phrase in ["avoid detection", "bypass", "evade"]:
            self.assertNotIn(phrase, eval_section.lower(),
                             f"Eval section should not contain '{phrase}'")


class ReadmeLegacyTest(unittest.TestCase):
    def test_legacy_cli_reference_unchanged(self):
        self.assertIn("--advice", README)
        self.assertIn("--file", README)

    def test_eval_subcommand_mentioned_in_cli_reference(self):
        # The CLI reference section (before ## Detectors) mentions eval
        cli_ref_section = README[:README.find("## Detectors")]
        self.assertIn("lynceus eval --help", cli_ref_section)


if __name__ == "__main__":
    unittest.main()
