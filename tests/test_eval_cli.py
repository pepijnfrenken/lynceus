"""Tests for the lynceus eval CLI subcommand dispatch."""

import io
import unittest
from unittest.mock import patch

from lynceus.cli import _main_eval, main


class EvalCliDispatchTest(unittest.TestCase):
    @patch("lynceus.cli._main_eval")
    def test_main_dispatches_to_eval_when_first_arg_is_eval(self, mock_eval):
        main(["eval", "--input", "test.jsonl"])
        mock_eval.assert_called_once_with(["--input", "test.jsonl"])

    @patch("lynceus.cli._main_single", return_value=0)
    def test_main_dispatches_to_legacy_when_not_eval(self, mock_single):
        with self.assertRaises(SystemExit):
            main(["--file", "test.txt"])
        mock_single.assert_called_once_with(["--file", "test.txt"])

    def test_eval_help_shows_input_and_preferences_but_not_advice_flag(self):
        with patch("sys.stdout", new=io.StringIO()) as stdout:
            with self.assertRaises(SystemExit):
                _main_eval(["--help"])
        help_text = stdout.getvalue()
        self.assertIn("--input", help_text)
        self.assertIn("--preferences", help_text)
        # --advice flag must not appear in eval help
        self.assertNotIn("--advice", help_text)


class LegacyHelpTest(unittest.TestCase):
    def test_legacy_help_includes_advice_flag(self):
        with patch("sys.stdout", new=io.StringIO()) as stdout:
            with self.assertRaises(SystemExit):
                main(["--help"])
        self.assertIn("--advice", stdout.getvalue())
class EvalRequiredInputTest(unittest.TestCase):
    @patch("lynceus.evaluator.run_eval", return_value=0)
    def test_eval_with_minimal_args(self, mock_run):
        with self.assertRaises(SystemExit):
            _main_eval(["--input", "test.jsonl"])
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        self.assertEqual(args[0], "test.jsonl")

    @patch("lynceus.evaluator.run_eval", return_value=0)
    def test_eval_stdin_input_uses_dash(self, mock_run):
        with self.assertRaises(SystemExit):
            _main_eval(["--input", "-"])
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        self.assertEqual(args[0], "-")

    @patch("lynceus.evaluator.run_eval", return_value=0)
    def test_eval_passes_preferences_path(self, mock_run):
        with self.assertRaises(SystemExit):
            _main_eval(["--input", "in.jsonl", "--preferences", "prefs.jsonl"])
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[1], "prefs.jsonl")

    def test_eval_requires_input(self):
        with self.assertRaises(SystemExit):
            _main_eval([])
if __name__ == "__main__":
    unittest.main()
