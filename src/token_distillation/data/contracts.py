"""Causal contracts for daily OHLCVA windows and historical teacher targets.

TeacherLabel is an as-of historical fact, never a future outcome. Realized future
outcomes belong to StudentWindow and become usable only after label_mature_at.
All timestamps in one object must share a timezone (all naive is also supported).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

import numpy as np


OHLCVA_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


def require_datetime(value: Any, field: str) -> datetime:
    """Reject ambiguous dates, strings, NaT and silently coerced numeric times."""
    if not isinstance(value, datetime) or value != value:
        raise ValueError(f"{field} must be a datetime, not a date/string/NaT")
    return value


def require_same_timezone(*values: datetime | None) -> None:
    times = [require_datetime(v, "timestamp") for v in values if v is not None]
    zones = {(v.tzinfo is not None and v.utcoffset() is not None, str(v.tzinfo)) for v in times}
    if len(zones) > 1:
        raise ValueError("timestamps must use the same timezone")


def _ordered_dates(values: tuple[datetime, ...], field: str) -> None:
    for value in values:
        require_datetime(value, field)
    require_same_timezone(*values)
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(f"{field} must be strictly increasing without duplicates")


def _matrix(value: np.ndarray, length: int, field: str) -> None:
    if not isinstance(value, np.ndarray) or value.shape != (length, len(OHLCVA_COLUMNS)):
        raise ValueError(f"{field} must be a numpy array of shape ({length}, 6)")
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{field} must contain finite numeric OHLCVA values")


@dataclass(frozen=True)
class StudentWindow:
    symbol: str
    as_of: datetime
    context_dates: tuple[datetime, ...]
    context: np.ndarray
    available_at: datetime
    future_dates: tuple[datetime, ...] = ()
    future: np.ndarray | None = None
    label_end: datetime | None = None
    label_mature_at: datetime | None = None
    future_status: str = "insufficient_future"
    future_status_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        require_datetime(self.as_of, "as_of")
        require_datetime(self.available_at, "available_at")
        if not self.context_dates:
            raise ValueError("context_dates cannot be empty")
        _ordered_dates(self.context_dates, "context_dates")
        _ordered_dates(self.future_dates, "future_dates")
        require_same_timezone(self.as_of, self.available_at, *self.context_dates,
                              *self.future_dates, self.label_end, self.label_mature_at)
        _matrix(self.context, len(self.context_dates), "context")
        if self.context_dates[-1] > self.as_of:
            raise ValueError("context contains a session after as_of")
        if not self.context_dates[-1] <= self.available_at <= self.as_of:
            raise ValueError("context available_at must be between its final session and as_of")
        if self.future is None:
            if self.future_dates or self.label_end is not None or self.label_mature_at is not None:
                raise ValueError("missing future must have no future_dates or label timestamps")
            if self.future_status != "insufficient_future":
                raise ValueError("missing future must be marked insufficient_future")
        else:
            if not self.future_dates:
                raise ValueError("future_dates cannot be empty when future is present")
            _matrix(self.future, len(self.future_dates), "future")
            require_datetime(self.label_end, "label_end")
            require_datetime(self.label_mature_at, "label_mature_at")
            if self.future_dates[0] <= self.as_of:
                raise ValueError("future sessions must be after as_of")
            if self.label_end != self.future_dates[-1]:
                raise ValueError("label_end must equal the final future session")
            if self.label_mature_at < self.label_end:
                raise ValueError("label_mature_at cannot precede label_end")
            if self.future_status != "complete":
                raise ValueError("a present future must be marked complete")

    @property
    def dependency_start(self) -> datetime:
        return self.context_dates[0]

    @property
    def has_future(self) -> bool:
        return self.future is not None


@dataclass(frozen=True)
class TeacherLabel:
    """A known historical value, or an explicitly masked unknown with a reason."""

    value: Any = None
    mask: bool = False
    available_at: datetime | None = None
    dependency_start: datetime | None = None
    dependency_end: datetime | None = None
    label_end: datetime | None = None
    label_mature_at: datetime | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mask, bool):
            raise ValueError("mask must be a bool")
        times = (self.available_at, self.dependency_start, self.dependency_end,
                 self.label_end, self.label_mature_at)
        require_same_timezone(*times)
        if not self.mask:
            if self.value is not None:
                raise ValueError("a masked unknown must have value=None; zero means known no event")
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("a masked unknown must explain its reason")
        else:
            for name in ("available_at", "dependency_start", "dependency_end", "label_end", "label_mature_at"):
                require_datetime(getattr(self, name), name)
            if self.value is None:
                raise ValueError("a valid teacher label must have a value")
            try:
                numeric = np.asarray(self.value, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("teacher value must be finite numeric data") from exc
            if numeric.size == 0 or not np.isfinite(numeric).all():
                raise ValueError("teacher value must be finite numeric data")
        if self.dependency_start is not None and self.dependency_end is not None:
            if self.dependency_start > self.dependency_end:
                raise ValueError("dependency_start cannot follow dependency_end")
        if self.mask:
            if not self.dependency_start <= self.label_end <= self.dependency_end <= self.label_mature_at:
                raise ValueError("teacher target must lie within its dependency interval and mature after dependencies")
        if self.label_end is not None and self.label_mature_at is not None:
            if self.label_mature_at < self.label_end:
                raise ValueError("label_mature_at cannot precede label_end")
        if self.available_at is not None:
            for value in (self.dependency_end, self.label_mature_at):
                if value is not None and value > self.available_at:
                    raise ValueError("available_at cannot precede dependency_end or label_mature_at")


@dataclass(frozen=True)
class TeacherSample:
    symbol: str
    as_of: datetime
    context_start: datetime
    labels: Mapping[str, TeacherLabel]

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        require_datetime(self.as_of, "as_of")
        require_datetime(self.context_start, "context_start")
        require_same_timezone(self.as_of, self.context_start)
        if self.context_start > self.as_of:
            raise ValueError("context_start cannot follow as_of")
        if not isinstance(self.labels, Mapping) or not self.labels:
            raise ValueError("labels must be a non-empty mapping")
        for name, label in self.labels.items():
            if not isinstance(name, str) or not name or not isinstance(label, TeacherLabel):
                raise ValueError("labels must map non-empty names to TeacherLabel")
            require_same_timezone(self.as_of, label.available_at, label.dependency_start,
                                  label.dependency_end, label.label_end, label.label_mature_at)
            if label.mask:
                if label.dependency_start < self.context_start:
                    raise ValueError(f"{name}: dependency starts before the student context")
                for value in (label.available_at, label.dependency_end, label.label_end, label.label_mature_at):
                    if value > self.as_of:
                        raise ValueError(f"{name}: valid historical teacher label uses information after as_of")
