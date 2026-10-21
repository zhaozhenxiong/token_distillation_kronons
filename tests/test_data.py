"""Synthetic regression tests for leakage and daily sample contracts."""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import unittest

import numpy as np
import pandas as pd

from token_distillation.data.contracts import (
    OHLCVA_COLUMNS, StudentWindow, TeacherLabel, TeacherSample,
)
from token_distillation.data.normalization import fit_history, normalize_window, transform_with_stats
from token_distillation.data.splits import SplitRange, TimeSplits, assign_sample, assign_window
from token_distillation.data.windows import build_window, parse_timestamp


def make_frame(count=15):
    sessions = pd.bdate_range("2024-01-02", periods=count)
    base = np.arange(count, dtype=np.float64) + 10.0
    frame = pd.DataFrame({"open": base, "high": base + 2, "low": base - 1,
                          "close": base + 1, "volume": base * 100,
                          "amount": base * 1000}, index=sessions)
    return frame, sessions


def make_splits():
    names = ("train", "tune", "calibration", "policy", "holdout")
    start = datetime(2024, 1, 1)
    return TimeSplits(**{name: SplitRange(name, start + timedelta(days=i * 10),
                                          start + timedelta(days=i * 10 + 9))
                         for i, name in enumerate(names)})


class NormalizationTests(unittest.TestCase):
    def test_future_perturbation_cannot_change_history_stats_or_tokens_input(self):
        frame, _ = make_frame()
        history = frame.iloc[:7].loc[:, OHLCVA_COLUMNS].to_numpy()
        future = frame.iloc[7:12].loc[:, OHLCVA_COLUMNS].to_numpy()
        first = normalize_window(history, future, clip=None)
        perturbed = normalize_window(history, future * 100000, clip=None)
        np.testing.assert_array_equal(first.history, perturbed.history)
        np.testing.assert_array_equal(first.stats.mean, perturbed.stats.mean)
        np.testing.assert_array_equal(first.stats.std, perturbed.stats.std)
        self.assertFalse(np.array_equal(first.future, perturbed.future))
        np.testing.assert_allclose(first.future, (future - history.mean(axis=0)) /
                                   (history.std(axis=0) + first.stats.epsilon))
        self.assertEqual(first.history.dtype, np.float64)

    def test_constant_history_uses_epsilon_and_clips_future(self):
        history = np.full((5, 6), 100.0, dtype=np.float32)
        result = normalize_window(history, np.full((2, 6), 101.0), epsilon=0.1, clip=3)
        np.testing.assert_array_equal(result.history, np.zeros((5, 6)))
        np.testing.assert_array_equal(result.future, np.full((2, 6), 3.0))
        self.assertFalse(result.stats.mean.flags.writeable)
        self.assertEqual(result.stats.history_count, 5)

    def test_non_finite_empty_and_wrong_column_order_shape_rejected(self):
        for invalid in (np.zeros((0, 6)), np.zeros((3, 5)), np.full((2, 6), np.nan)):
            with self.assertRaises(ValueError):
                fit_history(invalid)
        stats = fit_history(np.ones((2, 6)))
        with self.assertRaises(ValueError):
            transform_with_stats(np.full((2, 6), np.inf), stats)


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.calendar = make_frame()

    def build(self, frame=None, **kwargs):
        params = dict(symbol="000001.SZ", as_of=self.calendar[6].date(), context_length=5,
                      horizon=3, calendar_dates=self.calendar)
        params.update(kwargs)
        return build_window(self.frame if frame is None else frame, **params)

    def test_exact_window_and_future_perturbation_invariance(self):
        window = self.build()
        mutated = self.frame.copy()
        mutated.loc[self.calendar[7]:, list(OHLCVA_COLUMNS)] *= 50
        later = self.build(mutated)
        self.assertEqual(window.context_dates, tuple(self.calendar[2:7].to_pydatetime()))
        self.assertEqual(window.future_dates, tuple(self.calendar[7:10].to_pydatetime()))
        np.testing.assert_array_equal(window.context, later.context)
        self.assertFalse(np.array_equal(window.future, later.future))
        self.assertEqual(window.future_status, "complete")
        self.assertFalse(window.context.flags.writeable)
        np.testing.assert_array_equal(window.context, self.frame.iloc[2:7].to_numpy())

    def test_requires_explicit_calendar_and_rejects_missing_context_session(self):
        with self.assertRaisesRegex(ValueError, "calendar_dates is required"):
            self.build(calendar_dates=None)
        with self.assertRaisesRegex(ValueError, "context trading sessions"):
            self.build(self.frame.drop(self.calendar[4]))
        with self.assertRaisesRegex(ValueError, "future trading sessions"):
            self.build(self.frame.drop(self.calendar[8]))

    def test_partial_future_has_no_fabricated_label(self):
        window = self.build(self.frame.iloc[:9])
        self.assertIsNone(window.future)
        self.assertEqual(window.future_dates, ())
        self.assertIsNone(window.label_end)
        self.assertIsNone(window.label_mature_at)
        self.assertEqual(window.future_status, "insufficient_future")
        self.assertEqual(window.future_status_reason, "future_bars_not_yet_available")
        calendar_end = self.build(as_of=self.calendar[-2].date())
        self.assertIsNone(calendar_end.future)
        self.assertEqual(calendar_end.future_status_reason, "calendar_horizon_unavailable")

    def test_late_context_data_and_late_future_maturity(self):
        frame = self.frame.copy()
        frame["available_at"] = self.calendar + pd.Timedelta(hours=15)
        frame.loc[self.calendar[6], "available_at"] += pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "not available at as_of"):
            self.build(frame)
        frame.loc[self.calendar[6], "available_at"] = self.calendar[6] + pd.Timedelta(hours=15)
        frame.loc[self.calendar[9], "available_at"] += pd.Timedelta(days=2)
        window = self.build(frame)
        self.assertEqual(window.label_mature_at, window.label_end + timedelta(days=2, hours=15))

    def test_full_daily_bar_is_unavailable_intraday_and_date_means_close(self):
        window = self.build(as_of="2024-01-10")
        self.assertEqual(window.as_of, datetime(2024, 1, 10, 15))
        self.assertEqual(window.available_at, window.as_of)
        self.assertEqual(window.label_mature_at, window.label_end + timedelta(hours=15))
        for precise_time in ("2024-01-10T12:00:00", datetime(2024, 1, 10),
                             pd.Timestamp("2024-01-10T14:59:59")):
            with self.assertRaisesRegex(ValueError, "not available at as_of"):
                self.build(as_of=precise_time)
        explicit_close = self.build(as_of="2024-01-10T15:00:00")
        np.testing.assert_array_equal(explicit_close.context, window.context)
        frame = self.frame.copy()
        frame["available_at"] = self.calendar + pd.Timedelta(hours=15)
        frame.loc[self.calendar[6], "available_at"] = self.calendar[6] + pd.Timedelta(hours=12)
        with self.assertRaisesRegex(ValueError, "session close at 15:00"):
            self.build(frame)

    def test_date_only_as_of_inherits_aware_source_index_timezone(self):
        frame = self.frame.copy()
        frame.index = frame.index.tz_localize("Asia/Shanghai")
        window = self.build(frame, as_of="2024-01-10", calendar_dates=frame.index)
        self.assertEqual(window.as_of.hour, 15)
        self.assertEqual(str(window.as_of.tzinfo), "Asia/Shanghai")
        self.assertEqual(window.available_at, window.as_of)
        with self.assertRaisesRegex(ValueError, "not available at as_of"):
            self.build(frame, as_of=pd.Timestamp("2024-01-10T12:00:00", tz="Asia/Shanghai"),
                       calendar_dates=frame.index)

    def test_duplicate_unsorted_invalid_ohlc_and_non_session_date_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.build(pd.concat([self.frame, self.frame.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, "sorted"):
            self.build(self.frame.iloc[::-1])
        bad = self.frame.copy()
        bad.iloc[0, bad.columns.get_loc("high")] = 1
        with self.assertRaisesRegex(ValueError, "OHLC bounds"):
            self.build(bad)
        with self.assertRaisesRegex(ValueError, "absent from calendar"):
            self.build(calendar_dates=self.calendar.delete(1))
        with self.assertRaises(ValueError):
            self.build(as_of="2024-01-13")
        with self.assertRaises(ValueError):
            self.build(as_of=datetime(2024, 1, 10, tzinfo=timezone.utc))

    def test_strict_dates_reject_numeric_ambiguous_and_nat(self):
        for value in (20240102, "01/02/2024", "20240102", pd.NaT):
            with self.assertRaises(ValueError):
                parse_timestamp(value)
        self.assertEqual(parse_timestamp(date(2024, 1, 2)), pd.Timestamp("2024-01-02"))
        with self.assertRaises(ValueError):
            replace(self.build(), as_of=date(2024, 1, 10))


class TeacherContractTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2024, 1, 2)
        self.as_of = datetime(2024, 1, 10)

    def valid_label(self, **kwargs):
        fields = dict(value=0, mask=True, available_at=self.as_of,
                      dependency_start=self.start, dependency_end=self.as_of,
                      label_end=self.as_of, label_mature_at=self.as_of)
        fields.update(kwargs)
        return TeacherLabel(**fields)

    def sample(self, label):
        return TeacherSample("000001.SZ", self.as_of, self.start, {"demand": label})

    def test_known_zero_and_unknown_are_distinct(self):
        sample = self.sample(self.valid_label())
        self.assertTrue(sample.labels["demand"].mask)
        self.assertEqual(sample.labels["demand"].value, 0)
        unknown = self.sample(TeacherLabel(reason="insufficient_history"))
        self.assertFalse(unknown.labels["demand"].mask)
        with self.assertRaisesRegex(ValueError, "value=None"):
            TeacherLabel(value=0, mask=False, reason="unknown")
        with self.assertRaisesRegex(ValueError, "explain its reason"):
            TeacherLabel()

    def test_dependency_must_fit_the_student_visible_window(self):
        with self.assertRaisesRegex(ValueError, "before the student context"):
            self.sample(self.valid_label(dependency_start=self.start - timedelta(days=1)))
        with self.assertRaisesRegex(ValueError, "after as_of"):
            self.sample(self.valid_label(available_at=self.as_of + timedelta(days=1)))
        with self.assertRaisesRegex(ValueError, "mature after dependencies"):
            self.valid_label(label_mature_at=self.as_of - timedelta(days=1),
                             label_end=self.as_of - timedelta(days=1))
        with self.assertRaises(ValueError):
            self.valid_label(value=np.nan)
        with self.assertRaises(ValueError):
            self.valid_label(available_at=self.as_of.replace(tzinfo=timezone.utc))


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.splits = make_splits()

    def decision(self, **kwargs):
        args = dict(as_of=datetime(2024, 1, 4), available_at=datetime(2024, 1, 4),
                    label_end=datetime(2024, 1, 8), label_mature_at=datetime(2024, 1, 8),
                    splits=self.splits)
        args.update(kwargs)
        return assign_sample(**args)

    def test_label_end_and_delayed_maturity_each_purge_crossing_samples(self):
        self.assertTrue(self.decision().eligible)
        crossing = self.decision(label_end=datetime(2024, 1, 11), label_mature_at=datetime(2024, 1, 11))
        self.assertFalse(crossing.eligible)
        self.assertEqual(crossing.split, "train")
        self.assertIn("label_crosses_split_end", crossing.reasons)
        delayed = self.decision(label_mature_at=datetime(2024, 1, 11))
        self.assertFalse(delayed.eligible)
        self.assertNotIn("label_crosses_split_end", delayed.reasons)
        self.assertIn("label_not_mature_in_split", delayed.reasons)

    def test_incomplete_future_and_unavailable_source_are_ineligible(self):
        missing = self.decision(label_end=None, label_mature_at=None)
        self.assertEqual(missing.reasons, ("incomplete_future",))
        unavailable = self.decision(available_at=datetime(2024, 1, 5))
        self.assertIn("source_not_available_at_origin", unavailable.reasons)
        frame, calendar = make_frame()
        window = build_window(frame, symbol="000001.SZ", as_of=calendar[-2].date(),
                              context_length=3, horizon=4, calendar_dates=calendar)
        self.assertIn("incomplete_future", assign_window(window, self.splits).reasons)

    def test_split_overlap_order_and_outside_origin(self):
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            replace(self.splits, tune=replace(self.splits.tune, start=self.splits.train.end))
        with self.assertRaisesRegex(ValueError, "corresponding"):
            replace(self.splits, tune=self.splits.train)
        self.assertEqual(self.decision(as_of=datetime(2023, 12, 31)).reasons,
                         ("origin_outside_splits",))
        config = {segment.name: {"start": segment.start.isoformat(), "end": segment.end.isoformat()}
                  for segment in self.splits.ranges}
        self.assertEqual(TimeSplits.from_mapping(config), self.splits)

    def test_date_only_split_end_includes_real_close_and_range_list_is_supported(self):
        config = {segment.name: [segment.start.date().isoformat(), segment.end.date().isoformat()]
                  for segment in self.splits.ranges}
        parsed = TimeSplits.from_mapping(config)
        self.assertEqual(parsed.train.start, datetime(2024, 1, 1))
        self.assertEqual(parsed.train.end, datetime(2024, 1, 10, 23, 59, 59, 999999))
        at_last_close = assign_sample(as_of=datetime(2024, 1, 9, 15),
                                      available_at=datetime(2024, 1, 9, 15),
                                      label_end=datetime(2024, 1, 10),
                                      label_mature_at=datetime(2024, 1, 10, 15), splits=parsed)
        self.assertTrue(at_last_close.eligible)
        config["train"] = {"start": "2024-01-01", "end": "2024-01-10T15:00:00"}
        self.assertEqual(TimeSplits.from_mapping(config).train.end, datetime(2024, 1, 10, 15))


if __name__ == "__main__":
    unittest.main()
