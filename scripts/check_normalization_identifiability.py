#!/usr/bin/env python3
"""Synthetic proof of lost price-level information; no model or source reads."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from token_distillation.data.normalization import normalize_window


def consistent_bars(prices):
    """OHLCVA with positive turnover and average trade price equal to close."""
    prices = np.asarray(prices, dtype=np.float64)
    volume = np.full(len(prices), 1000., dtype=np.float64)
    return np.column_stack((prices, prices + 1., prices - 1., prices, volume, prices * volume))


def counterexample():
    history = consistent_bars(np.arange(100., 228.))
    future = consistent_bars([228., 229., 230., 229., 230.])
    shifted_history, shifted_future = history.copy(), future.copy()
    for bars in (shifted_history, shifted_future):
        bars[:, :4] += 1000.
        bars[:, 5] += 1000. * bars[:, 4]
    # Same origin and synthetic session features in both worlds. Actual
    # exchange dates are irrelevant to the non-injectivity construction.
    time = np.column_stack((np.zeros(133, dtype=np.int64), np.zeros(133, dtype=np.int64),
        np.arange(133) % 5, np.arange(133) % 28 + 1, np.arange(133) // 28 + 1))
    left = normalize_window(history, future, epsilon=1e-5, clip=5.)
    right = normalize_window(shifted_history, shifted_future, epsilon=1e-5, clip=5.)
    return history, future, shifted_history, shifted_future, time, left, right


def direction_class(value, threshold=.005):
    return 2 if value > threshold else 0 if value < -threshold else 1


def _hash_array(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def build_report():
    h, f, hs, fs, time, left, right = counterexample()
    realized = float(f[-1, 3] / h[-1, 3] - 1.)
    realized_shifted = float(fs[-1, 3] / hs[-1, 3] - 1.)
    exact = {}
    for dtype in (np.float64, np.float32):
        name = np.dtype(dtype).name
        context_left, context_right = left.history.astype(dtype), right.history.astype(dtype)
        future_left, future_right = left.future.astype(dtype), right.future.astype(dtype)
        exact[name] = {"normalized_context_array_equal": bool(np.array_equal(context_left, context_right)),
            "normalized_future_array_equal": bool(np.array_equal(future_left, future_right)),
            "normalized_context_sha256_left": _hash_array(context_left),
            "normalized_context_sha256_right": _hash_array(context_right),
            "normalized_future_sha256_left": _hash_array(future_left),
            "normalized_future_sha256_right": _hash_array(future_right),
            "maximum_absolute_context_difference": float(np.max(np.abs(context_left - context_right))),
            "maximum_absolute_future_difference": float(np.max(np.abs(future_left - future_right)))}
    anchors = (32, 64)
    report = {
        "schema": "normalization_price_level_identifiability_counterexample_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_type": "synthetic_mathematical_counterexample; no empirical model attribution",
        "normalization": {"method": "current normalize_window; per-column context mean and population std",
            "epsilon": 1e-5, "clip": 5., "context_length": 128, "horizon": 5},
        "construction": {"feature_order": ["open", "high", "low", "close", "volume", "amount"],
            "history_close_start": 100., "history_close_end": 227.,
            "future_closes": f[:, 3].tolist(), "price_translation": 1000.,
            "volume": 1000., "volume_unit": "shares", "amount_unit": "CNY",
            "ohlc_rule": "open=close; high=close+1; low=close-1",
            "amount_rule": "amount=close*volume; translated_amount=amount+1000*volume",
            "finite_positive_ohlcva": bool(all(np.isfinite(x).all() and (x > 0).all() for x in (h, f, hs, fs))),
            "amount_equals_close_times_volume": bool(all(np.array_equal(x[:, 5], x[:, 3] * x[:, 4]) for x in (h, f, hs, fs))),
            "synthetic_time_inputs_identical": True,
            "synthetic_time_sha256": _hash_array(time)},
        "exact_equalities": exact,
        "normalization_stats": {"left_mean": left.stats.mean.tolist(), "right_mean": right.stats.mean.tolist(),
            "left_std": left.stats.std.tolist(), "right_std": right.stats.std.tolist(),
            "std_array_equal": bool(np.array_equal(left.stats.std, right.stats.std))},
        "direction_labels": {"threshold": .005, "class_order": ["down", "flat", "up"],
            "left_origin_close": float(h[-1, 3]), "left_final_close": float(f[-1, 3]),
            "left_exact_return_fraction": "3/227", "left_return": realized,
            "left_class": direction_class(realized),
            "right_origin_close": float(hs[-1, 3]), "right_final_close": float(fs[-1, 3]),
            "right_exact_return_fraction": "3/1227", "right_return": realized_shifted,
            "right_class": direction_class(realized_shifted)},
        "relative_price_target_example": {"semantics": "illustrative context-anchor relative change; not a native confirmed CZSC pen",
            "context_anchor_indices": list(anchors),
            "left_prices": h[list(anchors), 3].tolist(), "right_prices": hs[list(anchors), 3].tolist(),
            "left_relative_return": float(h[anchors[1], 3] / h[anchors[0], 3] - 1.),
            "right_relative_return": float(hs[anchors[1], 3] / hs[anchors[0], 3] - 1.),
            "price_order_direction_unchanged": bool(np.sign(h[anchors[1], 3] - h[anchors[0], 3]) == np.sign(hs[anchors[1], 3] - hs[anchors[0], 3]))},
        "proved_limitation": "The normalized-input mapping is non-injective for raw percentage-return targets. Even an oracle normalized future trajectory cannot determine the raw ±0.5% direction class without additional price-level information. Identical deterministic tokenizer/time inputs cannot yield different deterministic internal-head outputs.",
        "equal_weight_two_world_bound": {"maximum_deterministic_classification_accuracy": .5,
            "minimum_expected_log_loss": float(np.log(2.)),
            "scope": "only this constructed equiprobable two-world problem; not an empirical market lower bound"},
        "limits": [
            "This proves a missing-information possibility, not that this limitation caused all or any measured ablation performance loss.",
            "Real stocks have variable volume and amount; their relationships may retain partial price-scale information. This counterexample proves no universal recovery guarantee.",
            "Forecast uncertainty remains even when raw scale is available; the construction isolates information lost by normalization rather than eliminating ordinary prediction uncertainty.",
            "Translation-invariant price ordering and structural directions can remain identifiable; percentage-valued targets need separate analysis.",
            "No native CZSC structure was generated; the relative-price example is arithmetic, not a confirmed pen fact.",
            "Exact equality here is translation invariance. General scale invariance is not claimed because normalization adds epsilon.",
            "A price forecast decoder can use stored context mean/std to denormalize a normalized forecast. The current direct internal direction head receives hidden token context and is not supplied these statistics.",
            "No architecture, runtime, dataset, optimizer budget or teacher weight was changed during the running experiment.",
        ],
        "execution": {"synthetic_only": True, "network": False, "source_database_read": False,
            "torch_imported": False, "model_loaded": False, "training_started": False},
    }
    assert all(value["normalized_context_array_equal"] and value["normalized_future_array_equal"] for value in exact.values())
    assert direction_class(realized) == 2 and direction_class(realized_shifted) == 1
    return report


def main():
    report = build_report()
    output = ROOT / "artifacts/reports/normalization_identifiability.local.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "direction_labels": report["direction_labels"],
        "exact_context_and_future_float64_float32": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
