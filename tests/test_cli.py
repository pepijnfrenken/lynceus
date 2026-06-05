import unittest

from humanify.analysis import ScoringUnit
from humanify.cli import _expand_unit_scores, _resolve_device


class CliDeviceTest(unittest.TestCase):
    def test_cpu_device_does_not_enable_cuda(self):
        self.assertEqual(_resolve_device("cpu"), ("cpu", False))

    def test_cuda_device_is_explicit_opt_in(self):
        self.assertEqual(_resolve_device("cuda"), ("cuda", True))



class CliScoringUnitTest(unittest.TestCase):
    def test_expand_unit_scores_maps_context_scores_to_sentences(self):
        units = [
            ScoringUnit("first passage", 0, 2),
            ScoringUnit("second passage", 2, 3),
        ]

        self.assertEqual(_expand_unit_scores(units, [0.1, 0.8], 3), [0.1, 0.1, 0.8])

    def test_expand_unit_scores_rejects_missing_detector_scores(self):
        units = [ScoringUnit("first passage", 0, 1)]

        with self.assertRaisesRegex(RuntimeError, "detector returned"):
            _expand_unit_scores(units, [], 1)

if __name__ == "__main__":
    unittest.main()
