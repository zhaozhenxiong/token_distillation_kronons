import importlib.util
import math
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "mechanism_holdout", Path(__file__).resolve().parents[1] / "scripts/evaluate_mechanism_controls.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MechanismHoldoutTests(unittest.TestCase):
    def rows(self):
        return [dict(symbol="600000.SH", as_of="2026-04-08", split="holdout",
                     direction_class=0, direction_probabilities=[.5, .2, .3]),
                dict(symbol="000001.SZ", as_of="2026-04-08", split="holdout",
                     direction_class=2, direction_probabilities=[.2, .2, .6])]

    def test_hand_calculated_scores_and_order_independence(self):
        identity, losses = module.prediction_losses(self.rows())
        other, reverse = module.prediction_losses(self.rows()[::-1])
        self.assertEqual(identity, other)
        self.assertAlmostEqual(losses["log_loss"].mean(), (-math.log(.5) - math.log(.6)) / 2)
        self.assertAlmostEqual(losses["brier"].mean(), .31)
        self.assertEqual(losses["accuracy"].mean(), 1.)
        for key in losses:
            self.assertEqual(losses[key].tolist(), reverse[key].tolist())

    def test_rejects_train_rows_duplicate_origins_and_bad_probabilities(self):
        for kind in ("train", "duplicate", "probabilities"):
            with self.subTest(kind=kind):
                rows = self.rows()
                if kind == "train":
                    rows[0]["split"] = "train"
                elif kind == "duplicate":
                    rows.append(rows[0].copy())
                else:
                    rows[0]["direction_probabilities"] = [.5, .3, .4]
                with self.assertRaises(ValueError):
                    module.prediction_losses(rows)
