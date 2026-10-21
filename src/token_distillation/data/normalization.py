"""Numpy-only normalization fitted exclusively on the observed history."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import OHLCVA_COLUMNS


def _values(values: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.ndim != 2 or values.shape[1] != len(OHLCVA_COLUMNS):
        raise ValueError(f"{name} must be a numpy array with shape (rows, 6) in OHLCVA order")
    if values.shape[0] == 0 or values.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain at least one numeric row")
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite data")
    return result


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray
    history_count: int
    epsilon: float = 1e-5
    clip: float | None = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.history_count, int) or isinstance(self.history_count, bool) or self.history_count < 1:
            raise ValueError("history_count must be a positive integer")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")
        if self.clip is not None and (not np.isfinite(self.clip) or self.clip <= 0):
            raise ValueError("clip must be positive and finite, or None")
        for name in ("mean", "std"):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if values.shape != (len(OHLCVA_COLUMNS),) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain six finite float64 values")
            if name == "std" and (values < 0).any():
                raise ValueError("std cannot be negative")
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class NormalizedWindow:
    history: np.ndarray
    future: np.ndarray | None
    stats: NormalizationStats


def fit_history(history: np.ndarray, *, epsilon: float = 1e-5,
                clip: float | None = 5.0) -> NormalizationStats:
    values = _values(history, "history")
    return NormalizationStats(mean=values.mean(axis=0, dtype=np.float64),
                              std=values.std(axis=0, dtype=np.float64, ddof=0),
                              history_count=len(values), epsilon=epsilon, clip=clip)


def transform_with_stats(values: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Apply frozen history statistics; future values can never change the fit."""
    data = _values(values, "values")
    result = (data - stats.mean) / (stats.std + stats.epsilon)
    if not np.isfinite(result).all():
        raise ValueError("normalization overflowed; inspect input scale and epsilon")
    if stats.clip is not None:
        result = np.clip(result, -stats.clip, stats.clip)
    return result


def normalize_window(history: np.ndarray, future: np.ndarray | None = None, *,
                     epsilon: float = 1e-5, clip: float | None = 5.0) -> NormalizedWindow:
    stats = fit_history(history, epsilon=epsilon, clip=clip)
    return NormalizedWindow(history=transform_with_stats(history, stats),
                            future=None if future is None else transform_with_stats(future, stats),
                            stats=stats)
