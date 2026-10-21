"""Synthetic native protocol fixtures; never import the source CZSC engine."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from token_distillation.teachers.czsc_teacher import CONTINUOUS_NAMES, CzscTeacher


class SyntheticSource:
    def __init__(self, *, empty=False):
        dates = pd.bdate_range("2026-01-05", periods=25)
        self.frame = pd.DataFrame({"symbol": "600000.SH", "date": dates,
                                   "open": 10., "high": 12., "low": 8., "close": 11.,
                                   "volume": np.arange(100., 125.), "amount": 1100.,
                                   "source": "verified_fixture", "has_trade_price": 1,
                                   "available_at": [(day + pd.Timedelta(hours=15)).isoformat() for day in dates]})
        self.frame.attrs.update(price_basis="unadjusted", volume_unit="shares", amount_unit="CNY")
        self.empty = empty
        self.instances = []
        self.load_requests = []
        self.module_requests = []
        self.engine_identity = {"engine": "synthetic_protocol_fixture", "source_tree_sha256": "a" * 64}
        self.adapter = SimpleNamespace(engine_contract=lambda: self.engine_identity,
                                       native_runtime=lambda: SimpleNamespace(CZSC=self.make_native),
                                       to_raw_bar=lambda row: row)

    def load_bars(self, symbol, *, start=None, as_of):
        self.load_requests.append((symbol, start, as_of))
        day = pd.Timestamp(as_of).tz_localize(None).normalize()
        return self.frame.loc[self.frame.date <= day].copy()

    def calendar_dates(self, start, end):
        return pd.bdate_range(start, end).strftime("%Y-%m-%d").tolist()

    def calendar_identity(self):
        return {"source": "synthetic_calendar"}

    def source_identity(self):
        return {"source": "synthetic_source", "code_sha256": "b" * 64}

    def module(self, name):
        self.module_requests.append(name)
        if name.endswith("czsc_adapter"):
            return self.adapter
        assert name.endswith("czsc_trend_support")
        return SimpleNamespace(VERIFIED_SOURCES={"verified_fixture"})

    def make_native(self, bars, **kwargs):
        instance = NativeFixture(bars, kwargs, empty=self.empty)
        self.instances.append(instance)
        return instance


class NativeFixture:
    """Expose confirmed pens separately from an extending final forming pen."""
    def __init__(self, bars, config, *, empty=False):
        self.bars = list(bars)
        self.config = config
        self.empty = empty

    def update(self, bar):
        self.bars.append(bar)

    def pen(self, start, end, direction, power):
        return SimpleNamespace(fx_a=SimpleNamespace(dt=self.bars[start].dt, fx=10.),
                               fx_b=SimpleNamespace(dt=self.bars[end].dt, fx=11.),
                               length=end - start + 1, direction=direction, power_price=power,
                               raw_bars=self.bars[start:end + 1])

    @property
    def finished_bis(self):
        if self.empty:
            return []
        result = []
        for observed, args in ((7, (0, 4, "向上", 2.)), (12, (4, 9, "向下", 3.)),
                               (18, (9, 15, "向上", 4.))):
            if len(self.bars) >= observed:
                result.append(self.pen(*args))
        return result

    @property
    def zs_list(self):
        finished = self.finished_bis
        if len(finished) < 3:
            return []
        complete = SimpleNamespace(bis=finished, zd=9., zg=10., is_valid=lambda: True)
        # The native collection may end in a zone using an unfinished pen. It
        # must not hide the earlier valid finished-only geometry.
        forming = self.pen(15, len(self.bars) - 1, "向下", 500.)
        incomplete = SimpleNamespace(bis=[*finished[-2:], forming], zd=100., zg=101., is_valid=lambda: True)
        return [complete, incomplete]


def as_dict(result, field):
    return dict(zip(result["continuous_names"], result[field]))


def test_native_finished_pen_facts_and_zone_ignore_forming_structure():
    source = SyntheticSource()
    result = CzscTeacher(source, context_bars=24).prepare("600000", "2026-02-06")
    values = as_dict(result, "continuous_targets")
    assert result["continuous_names"] == list(CONTINUOUS_NAMES)
    assert values["bi1_direction"] == 1
    assert values["bi2_direction"] == -1
    assert values["bi1_return"] == pytest.approx(.1)
    assert values["bi1_power_ratio"] == 2.
    assert values["confirmed_age"] == 6.
    assert values["finished_bi_count"] == 3.
    assert values["zone_low_distance"] == pytest.approx(11. / 9. - 1)
    assert values["zone_width_ratio"] == pytest.approx(1. / 11.)
    assert all(result["continuous_mask"])
    assert result["event_names"] == result["event_targets"] == result["event_mask"] == []
    assert source.instances[0].config == {"min_bi_len": 6, "max_bi_num": 200}
    assert source.instances[0].bars[0].id == 0
    assert result["quality"]["full_history_equivalence"] is False
    assert result["quality"]["native_czsc_facts"] is True
    json.dumps(result, allow_nan=False)


def test_replaying_exact_context_ignores_earlier_prefix_and_future_rows():
    source = SyntheticSource()
    teacher = CzscTeacher(source, context_bars=24)
    before = teacher.prepare("600000", "2026-02-06")
    source.frame.loc[0, "close"] = 11.5
    extra = source.frame.iloc[[-1]].copy()
    extra["date"] += pd.Timedelta(days=3)
    extra["available_at"] = "2026-02-09T15:00:00+08:00"
    source.frame = pd.concat([source.frame, extra], ignore_index=True)
    source.frame.attrs.update(volume_unit="shares", amount_unit="CNY")
    after = teacher.prepare("600000", "2026-02-06")
    assert before["continuous_targets"] == after["continuous_targets"]
    assert before["quality"]["raw_data_version"] == after["quality"]["raw_data_version"]
    assert before["target_metadata"] == after["target_metadata"]
    assert source.instances[0] is not source.instances[1]


def test_missing_native_facts_are_masked_with_known_window_local_count():
    source = SyntheticSource(empty=True)
    result = CzscTeacher(source, context_bars=24).prepare("600000", "2026-02-06")
    assert result["continuous_targets"][:-1] == [None] * 15
    assert result["continuous_mask"][:-1] == [False] * 15
    assert result["continuous_targets"][-1] == 0.
    assert result["continuous_mask"][-1] is True
    assert result["target_metadata"]["bi1_direction"]["reason"] == "no_confirmed_finished_pen_in_student_context"


def test_confirmation_uses_prefix_availability_not_backdated_pen_endpoint():
    source = SyntheticSource()
    # Source data loaded at this origin includes a later-arriving old raw row.
    source.frame.loc[1, "available_at"] = "2026-02-06T15:00:00+08:00"
    result = CzscTeacher(source, context_bars=24).prepare("600000", "2026-02-06")
    meta = result["target_metadata"]["bi1_direction"]
    assert meta["confirmed_at"] == result["available_at"] == "2026-02-06T15:00:00+08:00"
    assert pd.Timestamp(meta["anchor_at"]).date().isoformat() < meta["confirmed_at"][:10]


def test_unverified_source_masks_all_native_targets():
    source = SyntheticSource()
    source.frame.loc[10, "source"] = "unknown"
    result = CzscTeacher(source, context_bars=24).prepare("600000", "2026-02-06")
    assert result["continuous_targets"] == [None] * 16
    assert result["continuous_mask"] == [False] * 16
    assert result["quality"]["eligible"] is False


def test_rejects_unavailable_source_and_calendar_gap_before_native_loading():
    source = SyntheticSource()
    source.frame.loc[2, "available_at"] = "2026-02-06T16:00:00+08:00"
    with pytest.raises(ValueError, match="unavailable"):
        CzscTeacher(source, context_bars=24).prepare("600000", "2026-02-06")
    assert source.module_requests == []
    source = SyntheticSource()
    source.frame = source.frame.drop(index=10)
    with pytest.raises(ValueError, match="calendar"):
        CzscTeacher(source, context_bars=24).prepare("600000", "2026-02-06")
    assert source.module_requests == []


def test_engine_verification_failure_has_no_fallback():
    source = SyntheticSource()
    def reject():
        raise RuntimeError("fixture source manifest mismatch")
    source.adapter.engine_contract = reject
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        CzscTeacher(source, context_bars=24).prepare("600000", "2026-02-06")
    assert source.instances == []


@pytest.mark.parametrize("kwargs", [{"context_bars": True}, {"min_bi_len": 2}, {"max_bi_num": 2}])
def test_invalid_native_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        CzscTeacher(SyntheticSource(), **kwargs)
