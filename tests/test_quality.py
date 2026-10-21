import unittest
import pandas as pd

from token_distillation.data.quality import validate_raw_quality


class ActualSourceQualityTests(unittest.TestCase):
    def frame(self):
        frame = pd.DataFrame({"source": ["baostock"], "has_trade_price": [1],
                              "volume": [100.], "amount": [1000.]})
        frame.attrs.update(price_basis="unadjusted", volume_unit="shares", amount_unit="CNY")
        return frame

    def test_verified_raw_input_is_admitted_without_granting_qualification(self):
        result = validate_raw_quality(self.frame(), verified_sources={"baostock"})
        self.assertTrue(result["eligible"])
        self.assertFalse(result["formal_qualified"])

    def test_adjusted_unknown_and_unverified_data_cannot_pass_by_configuration(self):
        for issue in ("adjusted", "unknown_source", "unverified", "suspended", "missing_units"):
            with self.subTest(issue=issue):
                frame = self.frame()
                if issue == "adjusted": frame.attrs["price_basis"] = "mixed_legacy_adjustment_unverified"
                if issue == "unknown_source": frame["source"] = "unknown"
                if issue == "unverified": frame["has_trade_price"] = 0
                if issue == "suspended": frame["volume"] = 0
                if issue == "missing_units": frame.attrs.pop("volume_unit")
                with self.assertRaises(ValueError):
                    validate_raw_quality(frame, verified_sources={"baostock"})


if __name__ == "__main__":
    unittest.main()
