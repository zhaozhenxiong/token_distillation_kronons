"""Synthetic population admission cannot inspect later outcomes or completeness."""
import importlib.util
from pathlib import Path

import pandas as pd


_SPEC = importlib.util.spec_from_file_location("population_selector",
    Path(__file__).resolve().parents[1] / "scripts/select_ablation_population.py")
selector = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(selector)


def fixture():
    dates = pd.bdate_range("2023-01-02", "2024-03-01")
    frame = pd.DataFrame({"date": dates, "source": "sina_unadjusted_v1",
        "has_trade_price": 1, "volume": 1e7, "amount": 2e8})
    return frame, dates.strftime("%Y-%m-%d").tolist()


def summary(frame, calendar):
    return selector.candidate_summary(frame, calendar, train_start="2024-01-01",
        context_length=128, liquidity_start="2023-01-01", minimum_median_amount=1e8)


def test_posttraining_gaps_and_turnover_do_not_change_admission():
    frame, calendar = fixture()
    expected = summary(frame, calendar)
    changed = frame.loc[frame.date.lt("2024-02-01")].copy()
    changed.loc[changed.date.ge("2024-01-01"), "amount"] = 0.
    actual = summary(changed, calendar)
    assert expected["admitted"] and actual["admitted"]
    assert actual["pretraining_median_daily_amount_cny"] == expected["pretraining_median_daily_amount_cny"]
    assert actual["unverified_or_zero_turnover_from_first_training_context"]


def test_unverified_pretraining_prices_fail_closed():
    frame, calendar = fixture()
    frame.loc[frame.date.eq("2023-12-29"), "source"] = "legacy_fuyao"
    result = summary(frame, calendar)
    assert not result["admitted"]
    assert result["pretraining_admission_reasons"] == ["missing_or_unverified_pretraining_context"]


def test_pretraining_liquidity_cannot_be_rescued_by_future_liquidity():
    frame, calendar = fixture()
    frame.loc[frame.date.lt("2024-01-01"), "amount"] = 1e6
    frame.loc[frame.date.ge("2024-01-01"), "amount"] = 1e12
    result = summary(frame, calendar)
    assert not result["admitted"]
    assert result["pretraining_median_daily_amount_cny"] == 1e6
    assert result["pretraining_admission_reasons"] == ["insufficient_pretraining_turnover"]
