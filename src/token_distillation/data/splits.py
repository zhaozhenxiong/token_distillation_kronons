"""Five explicit chronological stages with label-maturity boundary purging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Mapping

from .contracts import StudentWindow, TeacherSample, require_datetime, require_same_timezone
from .windows import is_date_only, parse_timestamp


SPLIT_NAMES = ("train", "tune", "calibration", "policy", "holdout")


@dataclass(frozen=True)
class SplitRange:
    name: str
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {self.name!r}; expected one of {SPLIT_NAMES}")
        require_datetime(self.start, "start")
        require_datetime(self.end, "end")
        require_same_timezone(self.start, self.end)
        if self.start > self.end:
            raise ValueError("split start must not follow its inclusive end")


@dataclass(frozen=True)
class TimeSplits:
    train: SplitRange
    tune: SplitRange
    calibration: SplitRange
    policy: SplitRange
    holdout: SplitRange

    def __post_init__(self) -> None:
        ranges = self.ranges
        for expected, segment in zip(SPLIT_NAMES, ranges):
            if not isinstance(segment, SplitRange) or segment.name != expected:
                raise ValueError(f"{expected} must be the corresponding SplitRange")
        require_same_timezone(*(point for segment in ranges for point in (segment.start, segment.end)))
        if any(left.end >= right.start for left, right in zip(ranges, ranges[1:])):
            raise ValueError("five split ranges must be chronological and non-overlapping")

    @property
    def ranges(self) -> tuple[SplitRange, ...]:
        return tuple(getattr(self, name) for name in SPLIT_NAMES)

    @classmethod
    def from_mapping(cls, config: Mapping) -> "TimeSplits":
        if set(config) != set(SPLIT_NAMES):
            raise ValueError(f"split config must define exactly {SPLIT_NAMES}")
        ranges = {}
        for name in SPLIT_NAMES:
            segment = config[name]
            if isinstance(segment, Mapping) and set(segment) == {"start", "end"}:
                start_value, end_value = segment["start"], segment["end"]
            elif isinstance(segment, (list, tuple)) and len(segment) == 2:
                start_value, end_value = segment
            else:
                raise ValueError(f"{name} must contain start/end or a two-element date range")
            start = parse_timestamp(start_value, "start").to_pydatetime()
            end = parse_timestamp(end_value, "end").to_pydatetime()
            if is_date_only(start_value):
                start = datetime.combine(start.date(), time.min, tzinfo=start.tzinfo)
            if is_date_only(end_value):
                end = datetime.combine(end.date(), time.max, tzinfo=end.tzinfo)
            ranges[name] = SplitRange(name, start, end)
        return cls(**ranges)


@dataclass(frozen=True)
class SplitDecision:
    split: str | None
    eligible: bool
    reasons: tuple[str, ...] = ()


def assign_sample(*, as_of: datetime, available_at: datetime,
                  label_end: datetime | None, label_mature_at: datetime | None,
                  splits: TimeSplits) -> SplitDecision:
    """Filter on origin, source availability, full horizon and label maturity.

    Inclusive ranges concern prediction origins and realized labels. Historical
    context may precede a stage's start, as it would in real-time inference.
    Gaps between the five ranges are an explicit embargo, never auto-filled.
    """
    require_datetime(as_of, "as_of")
    require_datetime(available_at, "available_at")
    require_same_timezone(as_of, available_at, label_end, label_mature_at,
                          *(segment.start for segment in splits.ranges))
    segment = next((item for item in splits.ranges if item.start <= as_of <= item.end), None)
    if segment is None:
        return SplitDecision(None, False, ("origin_outside_splits",))
    reasons = []
    if available_at > as_of:
        reasons.append("source_not_available_at_origin")
    if label_end is None or label_mature_at is None:
        reasons.append("incomplete_future")
    else:
        require_datetime(label_end, "label_end")
        require_datetime(label_mature_at, "label_mature_at")
        if label_end <= as_of:
            reasons.append("label_end_not_after_origin")
        if label_mature_at < label_end:
            reasons.append("label_mature_before_end")
        if label_end > segment.end:
            reasons.append("label_crosses_split_end")
        if label_mature_at > segment.end:
            reasons.append("label_not_mature_in_split")
    return SplitDecision(segment.name, not reasons, tuple(reasons))


def assign_window(window: StudentWindow, splits: TimeSplits,
                  teacher: TeacherSample | None = None) -> SplitDecision:
    decision = assign_sample(as_of=window.as_of, available_at=window.available_at,
                             label_end=window.label_end, label_mature_at=window.label_mature_at,
                             splits=splits)
    if teacher is not None:
        if (teacher.symbol != window.symbol or teacher.as_of != window.as_of or
                teacher.context_start != window.dependency_start):
            return SplitDecision(decision.split, False, decision.reasons + ("teacher_window_mismatch",))
        if not any(label.mask for label in teacher.labels.values()):
            return SplitDecision(decision.split, False, decision.reasons + ("all_teacher_targets_masked",))
    return decision
