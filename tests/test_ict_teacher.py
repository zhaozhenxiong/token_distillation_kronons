"""Frozen ICT source and causal atomic-teacher tests, using synthetic bars only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import socket

import numpy as np
import pandas as pd
import pytest

from token_distillation.adapters.ict import FrozenIctAdapter, PINNED_COMMIT
from token_distillation.teachers.ict_teacher import CONTINUOUS_NAMES, EVENT_NAMES, IctTeacher


VENDOR = Path(__file__).resolve().parents[1] / "vendor/ict_trading_model"


def bars(highs, lows, closes):
    dates = pd.bdate_range("2024-01-02", periods=len(closes))
    frame = pd.DataFrame({"symbol": "600000.SH", "date": dates, "open": closes,
                          "high": highs, "low": lows, "close": closes,
                          "volume": 100.0, "amount": 1000.0, "has_trade_price": 1, "source": "tushare"})
    frame = frame.astype({name: float for name in ("open", "high", "low", "close")})
    frame.attrs.update(price_basis="unadjusted", volume_unit="shares", amount_unit="CNY")
    return frame


class SyntheticSource:
    """Implements the verified source protocol with a clearly synthetic identity."""

    def __init__(self, frame):
        self.frame = frame.copy()
        self.calendar = frame.date.dt.strftime("%Y-%m-%d").tolist()
        self.ignore_cutoff = False
        self.stale = False
        self.verified = True
        self.calls = []

    def load_bars(self, symbol, *, as_of):
        self.calls.append((symbol, as_of))
        cutoff = pd.Timestamp(as_of).tz_convert("Asia/Shanghai").tz_localize(None).normalize()
        result = self.frame.copy() if self.ignore_cutoff else self.frame.loc[self.frame.date <= cutoff].copy()
        if self.stale:
            result = result.iloc[:-1].copy()
        result.attrs = dict(self.frame.attrs)
        result.attrs["data_version"] = hashlib.sha256(self.frame.to_csv(index=False).encode()).hexdigest()
        return result.reset_index(drop=True)

    def calendar_dates(self, first, last):
        return [day for day in self.calendar if first <= day <= last]

    def calendar_identity(self):
        digest = hashlib.sha256(json.dumps(self.calendar).encode()).hexdigest()
        return {"run_id": "synthetic-only-no-real-calendar", "source": "independent-synthetic-calendar",
                "independent_from_market_bars": self.verified, "file_sha256": digest,
                "dates_hash": digest, "first_session": self.calendar[0], "last_session": self.calendar[-1],
                "source_artifacts": [{"path": "synthetic-fixture-only", "sha256": digest}]}

    def source_identity(self):
        return {"scope": "sh_sz_a_v1", "source": "synthetic-fixture-only"}


def teacher(source, count=None, width=2, vendor=VENDOR):
    return IctTeacher(source, vendor, context_bars=count or len(source.frame), swing_width=width)


def value(result, name):
    kind = "continuous" if name in CONTINUOUS_NAMES else "event"
    index = result[kind + "_names"].index(name)
    return result[kind + "_targets"][index], result[kind + "_mask"][index]


def test_frozen_source_verifies_pins_without_network_or_vendor_bytecode(tmp_path, monkeypatch):
    root = tmp_path / "ict"
    shutil.copytree(VENDOR, root, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("must stay offline"))
    adapter = FrozenIctAdapter(root)
    identity = adapter.verify()
    assert identity["commit"] == PINNED_COMMIT
    assert identity["retained_files_verified"] == 10
    assert identity["git_objects_verified"] is False
    assert adapter.module().detect_swings([], 2) == []
    assert not list(root.rglob("*.pyc"))
    assert not list(root.rglob("__pycache__"))
    # A previously cached module must not conceal a modified frozen source.
    script = root / "scripts/ict_scan.py"
    script.write_bytes(script.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        adapter.module()
    manifest = root / "source_manifest.json"
    manifest.write_text(manifest.read_text().replace(PINNED_COMMIT, "0" * 40))
    with pytest.raises(ValueError, match="manifest SHA256"):
        adapter.verify()


def test_swing_confirmation_lag_for_continuous_and_event_targets():
    frame = bars([11, 12, 15, 13, 12, 14, 17], [9, 10, 11, 10, 9, 10, 12],
                 [10, 11, 14, 12, 11, 13, 16])
    source = SyntheticSource(frame)
    # Anchor at index 2 is observed only when right-confirmation index 4 closes.
    first = teacher(source, count=5).prepare("600000", "2024-01-08")
    assert value(first, "ict_swing_high_distance_frac") == ((15 - 11) / 11, True)
    reference = first["target_metadata"]["ict_swing_high_distance_frac"]["reference"]
    assert reference["anchor_index"] == 2
    assert reference["observed_index"] == 4
    assert reference["anchor_at"] == "2024-01-04T15:00:00+08:00"
    assert reference["observed_at"] == "2024-01-08T15:00:00+08:00"
    assert value(first, "ict_buyside_sweep_now") == (None, False)
    assert value(first, "ict_close_break_high_now") == (None, False)
    # By the next origin, the same level is known and a zero event is meaningful.
    next_bar = teacher(source, count=6).prepare("600000.SH", "2024-01-09")
    assert value(next_bar, "ict_close_break_high_now") == (0.0, True)
    final = teacher(source).prepare("600000.SH", "2024-01-10")
    assert value(final, "ict_close_break_high_now") == (1.0, True)
    assert value(final, "ict_buyside_sweep_now") == (0.0, True)
    # A low newly confirmed on the current bar is valid for distance, not events.
    assert value(final, "ict_swing_low_distance_frac")[1] is True
    assert final["target_metadata"]["ict_swing_low_distance_frac"]["reference"]["observed_index"] == 6
    assert value(final, "ict_sellside_sweep_now") == (None, False)


def test_current_sweep_and_current_cross_are_distinct_atomic_events():
    source = SyntheticSource(bars([15, 14, 13, 14, 15, 14, 13], [12, 11, 8, 10, 11, 10, 7],
                                 [13, 12, 9, 12, 13, 12, 9]))
    result = teacher(source).prepare("600000.SH", source.calendar[-1])
    assert value(result, "ict_sellside_sweep_now") == (1.0, True)
    assert value(result, "ict_close_break_low_now") == (0.0, True)
    source.frame.loc[6, ["open", "close"]] = 7.5
    result = teacher(source).prepare("600000.SH", source.calendar[-1])
    assert value(result, "ict_sellside_sweep_now") == (0.0, True)
    assert value(result, "ict_close_break_low_now") == (1.0, True)


def test_fvg_is_formed_and_available_at_third_close_never_middle_bar():
    source = SyntheticSource(bars([11, 12, 11, 15, 16], [9, 10, 9, 13, 14], [10, 11, 10, 14, 15]))
    result = teacher(source).prepare("600000.SH", source.calendar[-1])
    assert value(result, "ict_fvg_bullish_now") == (1.0, True)
    assert value(result, "ict_fvg_bearish_now") == (0.0, True)
    evidence = result["target_metadata"]["ict_fvg_bullish_now"]
    assert evidence["middle_index"] == 3
    assert evidence["end_index"] == evidence["formation_index"] == 4
    assert evidence["formation_at"] == evidence["available_at"] == "2024-01-08T15:00:00+08:00"
    assert evidence["available_at"] != "2024-01-05T15:00:00+08:00"
    # Late source publication can only move availability later than formation.
    source.frame["available_at"] = source.frame.date + pd.Timedelta(hours=15)
    source.frame.loc[4, "available_at"] += pd.Timedelta(minutes=30)
    late = teacher(source).prepare("600000.SH", "2024-01-08T16:00:00+08:00")
    assert late["target_metadata"]["ict_fvg_bullish_now"]["formation_at"] == evidence["formation_at"]
    assert late["target_metadata"]["ict_fvg_bullish_now"]["available_at"] == "2024-01-08T15:30:00+08:00"


def test_future_and_older_prefix_perturbations_cannot_change_window_targets():
    frame = bars([11] * 14, [9] * 14, [10] * 14)
    frame.loc[5:11, ["open", "close"]] = np.repeat(np.array([10, 11, 14, 12, 11, 13, 16])[:, None], 2, axis=1)
    frame.loc[5:11, "high"] = [11, 12, 15, 13, 12, 14, 17]
    frame.loc[5:11, "low"] = [9, 10, 11, 10, 9, 10, 12]
    source = SyntheticSource(frame)
    origin = source.calendar[11]
    first = teacher(source, count=7).prepare("600000.SH", origin)
    source.frame.loc[[0, 12, 13], ["open", "high", "low", "close", "volume", "amount"]] *= 100
    changed = teacher(source, count=7).prepare("600000.SH", origin)
    for field in ("continuous_targets", "continuous_mask", "event_targets", "event_mask",
                  "confirmed_swings", "target_metadata"):
        assert first[field] == changed[field]
    assert first["quality"]["raw_data_version"] == changed["quality"]["raw_data_version"]
    assert first["quality"]["source_load_data_version"] != changed["quality"]["source_load_data_version"]


def test_unknown_masks_pending_scope_and_full_scanner_is_never_run(monkeypatch):
    source = SyntheticSource(bars([11] * 7, [9] * 7, [10] * 7))
    instance = teacher(source)
    native = instance.adapter.module()
    monkeypatch.setattr(native, "scan", lambda *a, **k: pytest.fail("full scanner is outside teacher scope"))
    monkeypatch.setattr(native, "build_candidate", lambda *a, **k: pytest.fail("candidate state must not be supervised"))
    result = instance.prepare("600000.SH", source.calendar[-1])
    assert result["continuous_targets"] == [None, None]
    assert result["continuous_mask"] == [False, False]
    assert result["event_targets"] == [0.0, 0.0, None, None, None, None]
    assert result["event_mask"] == [True, True, False, False, False, False]
    assert all(item["target"] is None and item["mask"] is False for item in result["pending_targets"].values())
    assert result["quality"]["full_ict_model_implemented"] is False
    assert result["quality"]["upstream_full_scanner_executed"] is False
    assert result["provenance"]["reused_primitives"] == ["detect_swings", "detect_fvgs"]
    json.dumps(result, allow_nan=False)
    source.frame.loc[2, "has_trade_price"] = 0
    unverified = instance.prepare("600000.SH", source.calendar[-1])
    assert unverified["event_mask"] == [False] * 6
    assert unverified["event_targets"] == [None] * 6
    assert unverified["quality"]["eligible"] is False


@pytest.mark.parametrize("source_value", [None, "unknown", "fuyao", "TUSHARE"])
def test_unverified_raw_source_cannot_be_made_eligible_by_unadjusted_attrs(source_value):
    source = SyntheticSource(bars([11] * 7, [9] * 7, [10] * 7))
    source.frame.loc[2, "source"] = source_value
    result = teacher(source).prepare("600000.SH", source.calendar[-1])
    assert result["quality"]["price_basis"] == "unadjusted"
    assert result["continuous_targets"] == [None, None]
    assert result["event_targets"] == [None] * 6
    assert result["event_mask"] == [False] * 6
    assert "context_contains_unverified_raw_source" in result["quality"]["reason_codes"]
    assert "tushare" in result["quality"]["verified_raw_sources"]


def test_rejects_bad_as_of_source_future_stale_short_intraday_and_late_rows():
    source = SyntheticSource(bars([11] * 7, [9] * 7, [10] * 7))
    with pytest.raises(ValueError, match="at or after Shanghai 15:00"):
        teacher(source).prepare("600000.SH", source.calendar[-1] + "T14:59:00+08:00")
    assert source.calls == []
    source.ignore_cutoff = True
    with pytest.raises(ValueError, match="future bars"):
        teacher(source, count=5).prepare("600000.SH", source.calendar[4])
    source.ignore_cutoff = False
    source.stale = True
    with pytest.raises(ValueError, match="stale history"):
        teacher(source).prepare("600000.SH", source.calendar[-1])
    source.stale = False
    with pytest.raises(ValueError, match="exactly 8 context bars"):
        teacher(source, count=8).prepare("600000.SH", source.calendar[-1])
    source.frame["available_at"] = source.frame.date + pd.Timedelta(hours=15)
    source.frame.loc[2, "available_at"] = source.frame.date.iloc[-1] + pd.Timedelta(hours=16)
    with pytest.raises(ValueError, match="unavailable at as_of"):
        teacher(source).prepare("600000.SH", source.calendar[-1])
    source.frame.loc[2, "available_at"] = source.frame.date.iloc[2] + pd.Timedelta(hours=12)
    with pytest.raises(ValueError, match="precedes Shanghai session close"):
        teacher(source).prepare("600000.SH", source.calendar[-1])


def test_rejects_unverified_calendar_gaps_mixed_symbols_and_bad_ohlc():
    source = SyntheticSource(bars([11] * 7, [9] * 7, [10] * 7))
    source.verified = False
    with pytest.raises(ValueError, match="independently verified"):
        teacher(source).prepare("600000.SH", source.calendar[-1])
    source.verified = True
    removed = source.calendar.pop(2)
    with pytest.raises(ValueError, match="exactly match the verified"):
        teacher(source).prepare("600000.SH", source.calendar[-1])
    source.calendar.insert(2, removed)
    source.frame.loc[1, "symbol"] = "000001.SZ"
    with pytest.raises(ValueError, match="different or mixed symbol"):
        teacher(source).prepare("600000.SH", source.calendar[-1])
    source.frame.loc[1, "symbol"] = "600000.SH"
    source.frame.loc[1, "high"] = 5
    with pytest.raises(ValueError, match="OHLC bounds"):
        teacher(source).prepare("600000.SH", source.calendar[-1])


def test_late_pivot_confirmation_is_not_previous_close_knowledge():
    source = SyntheticSource(bars([11, 12, 15, 13, 12, 14, 17], [9, 10, 11, 10, 9, 10, 12],
                                 [10, 11, 14, 12, 11, 13, 16]))
    source.frame["available_at"] = source.frame.date + pd.Timedelta(hours=15)
    # A supporting right-side bar was only ingested on the current origin.
    source.frame.loc[4, "available_at"] = source.frame.date.iloc[-1] + pd.Timedelta(hours=15)
    result = teacher(source).prepare("600000.SH", source.calendar[-1])
    assert value(result, "ict_swing_high_distance_frac")[1] is True
    assert value(result, "ict_close_break_high_now") == (None, False)
