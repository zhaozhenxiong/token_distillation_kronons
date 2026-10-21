"""Build causal single-symbol daily windows against an explicit session calendar.

The index is a daily session date, not an availability timestamp. A full A-share
daily bar is never available before 15:00 local time. Without ``available_at``
this builder conservatively assumes session close at 15:00; that is an assumed
close time, not evidence of actual ingestion or publication. A source with real
publication/revision times must supply that column. No calendar is inferred.
"""

from __future__ import annotations

from datetime import date, datetime, time
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import OHLCVA_COLUMNS, StudentWindow, require_datetime, require_same_timezone


_ISO_DATE_TIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:\d{2})?)?$"
)


def is_date_only(value: datetime | date | str) -> bool:
    """Datetime/Timestamp midnight remains an explicit time, never a date alias."""
    return type(value) is date or (isinstance(value, str) and
                                   re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None)


def parse_timestamp(value: datetime | date | str, field: str = "timestamp") -> pd.Timestamp:
    """Only explicit ISO date/time text, date or datetime inputs are accepted."""
    if isinstance(value, str):
        if not _ISO_DATE_TIME.fullmatch(value):
            raise ValueError(f"{field} must be an ISO YYYY-MM-DD date or datetime")
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid {field}: {value}") from exc
    elif type(value) is date:
        value = datetime.combine(value, time())
    require_datetime(value, field)
    return pd.Timestamp(value)


def _daily_index(index: pd.DatetimeIndex, name: str) -> None:
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans or len(index) == 0:
        raise ValueError(f"{name} must be a non-empty DatetimeIndex without NaT")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(f"{name} must be sorted and contain no duplicate dates")
    if not index.equals(index.normalize()):
        raise ValueError(f"{name} must contain midnight daily session dates")


def validate_frame(frame: pd.DataFrame, *, symbol: str | None = None) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame")
    _daily_index(frame.index, "frame.index")
    missing = [name for name in OHLCVA_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"frame is missing OHLCVA columns: {missing}")
    for name in OHLCVA_COLUMNS:
        if getattr(frame[name].dtype, "kind", None) not in "fiu":
            raise ValueError(f"{name} must have a numeric dtype")
    try:
        values = frame.loc[:, OHLCVA_COLUMNS].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("OHLCVA values must be finite numeric data") from exc
    if not np.isfinite(values).all():
        raise ValueError("OHLCVA values must be finite")
    opening, high, low, close, volume, amount = values.T
    if (values[:, :4] <= 0).any():
        raise ValueError("OHLC prices must be positive")
    if (high < np.maximum(opening, close)).any() or (low > np.minimum(opening, close)).any() or (low > high).any():
        raise ValueError("invalid OHLC bounds: low <= open/close <= high is required")
    if (volume < 0).any() or (amount < 0).any():
        raise ValueError("volume and amount cannot be negative")
    if "symbol" in frame and (frame["symbol"].nunique(dropna=False) != 1 or
                              (symbol is not None and frame["symbol"].iloc[0] != symbol)):
        raise ValueError("frame must contain exactly the requested single symbol")


def build_window(frame: pd.DataFrame, *, symbol: str, as_of: datetime | date | str,
                 context_length: int = 256, horizon: int = 5,
                 calendar_dates: Iterable[datetime | date | str] | None = None) -> StudentWindow:
    """Require every context session; retain no target when the future is partial.

    A missing session before the last supplied bar is a data gap and is rejected.
    A horizon extending beyond supplied bars or the explicit calendar is marked
    insufficient_future. No zero padding, forward filling or label fabrication
    is performed. Caller-owned input frames are never changed.
    """
    for name, value in (("context_length", context_length), ("horizon", horizon)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if calendar_dates is None:
        raise ValueError("calendar_dates is required; observed rows cannot prove trading-day continuity")
    validate_frame(frame, symbol=symbol)
    as_of_ts = parse_timestamp(as_of, "as_of")
    if is_date_only(as_of):
        # A date means an end-of-session prediction, in the source index zone.
        as_of_ts = as_of_ts.replace(hour=15).tz_localize(frame.index.tz)
    calendar = pd.DatetimeIndex([parse_timestamp(item, "calendar date") for item in calendar_dates])
    _daily_index(calendar, "calendar_dates")
    require_same_timezone(as_of_ts, frame.index[0], calendar[0])
    if not frame.index.isin(calendar).all():
        raise ValueError("frame contains session dates absent from calendar_dates")
    origin = as_of_ts.normalize()
    if origin not in calendar or origin not in frame.index:
        raise ValueError("as_of must identify an observed session in calendar_dates")
    origin_pos = int(calendar.get_loc(origin))
    if origin_pos + 1 < context_length:
        raise ValueError("insufficient context in the explicit calendar")
    expected_context = calendar[origin_pos - context_length + 1:origin_pos + 1]
    if not expected_context.isin(frame.index).all():
        raise ValueError("calendar gap: one or more context trading sessions are missing")

    session_closes = pd.Series([session.replace(hour=15) for session in frame.index],
                               index=frame.index)
    if "available_at" in frame:
        availability = pd.Series(
            [parse_timestamp(value, "available_at") for value in frame["available_at"]], index=frame.index
        )
        require_same_timezone(as_of_ts, *availability.tolist())
        if any(available < session_closes.loc[session] for session, available in availability.items()):
            raise ValueError("available_at cannot precede the daily bar session close at 15:00")
    else:
        availability = session_closes
    context_available_at = max(availability.loc[expected_context])
    if context_available_at > as_of_ts:
        raise ValueError("context bar was not available at as_of")

    context = frame.loc[expected_context, OHLCVA_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    context.setflags(write=False)
    expected_future = calendar[origin_pos + 1:origin_pos + 1 + horizon]
    missing = expected_future[~expected_future.isin(frame.index)]
    if any(value <= frame.index[-1] for value in missing):
        raise ValueError("calendar gap: one or more future trading sessions are missing")
    common = dict(symbol=symbol, as_of=as_of_ts.to_pydatetime(),
                  context_dates=tuple(expected_context.to_pydatetime()), context=context,
                  available_at=context_available_at.to_pydatetime())
    if len(expected_future) != horizon or len(missing):
        return StudentWindow(**common, future_status="insufficient_future",
                             future_status_reason=("calendar_horizon_unavailable" if len(expected_future) != horizon
                                                   else "future_bars_not_yet_available"))
    future = frame.loc[expected_future, OHLCVA_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    future.setflags(write=False)
    return StudentWindow(**common, future_dates=tuple(expected_future.to_pydatetime()), future=future,
                         label_end=expected_future[-1].to_pydatetime(),
                         label_mature_at=max(availability.loc[expected_future]).to_pydatetime(),
                         future_status="complete")
