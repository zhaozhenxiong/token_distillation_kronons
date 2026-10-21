import unittest
import numpy as np

from token_distillation.evaluation.metrics import direction_metrics


class MetricTests(unittest.TestCase):
    def test_perfect_probabilities_and_failure_denominator(self):
        result = direction_metrics([[0., 1., 0.], [np.nan] * 3], np.array([1, -1]), [True, False])
        self.assertEqual(result["log_loss"], 0.)
        self.assertEqual(result["brier"], 0.)
        self.assertEqual(result["excluded"], 1)
        self.assertEqual(result["valid_fraction"], .5)
        self.assertFalse(result["qualified"])

    def test_non_normalized_probabilities_fail(self):
        with self.assertRaises(ValueError):
            direction_metrics([[.2, .2, .2]], [1])

    def test_empty_evaluation_does_not_invent_good_metrics(self):
        result = direction_metrics(np.empty((0, 3)), np.array([], dtype=int))
        self.assertIsNone(result["log_loss"])
        self.assertIsNone(result["valid_fraction"])


if __name__ == "__main__":
    unittest.main()
