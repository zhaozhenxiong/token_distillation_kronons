"""Window-local daily facts from the verified native CZSC runtime.

This teacher replays only the student's context and reads ``finished_bis``.
Its values are native facts at that origin, not full-history CZSC equivalents,
trade labels, or outputs from the student's approximate structure head.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from token_distillation.adapters.khquant import availability_times
from token_distillation.data.contracts import OHLCVA_COLUMNS
from token_distillation.data.windows import build_window, is_date_only, parse_timestamp


TEACHER_SCHEMA = "czsc_native_window_local_confirmed_daily_teacher_v1"
CONTINUOUS_NAMES = (
    "bi1_direction", "bi1_return", "bi1_length",
    "bi2_direction", "bi2_return", "bi2_length",
    "bi3_direction", "bi3_return", "bi3_length",
    "bi1_volume_ratio", "bi1_power_ratio", "confirmed_age",
    "zone_low_distance", "zone_high_distance", "zone_width_ratio",
    "finished_bi_count",
)


def _stamp(value, field):
    stamp = parse_timestamp(value, field)
    return stamp.tz_localize("Asia/Shanghai") if stamp.tzinfo is None else stamp.tz_convert("Asia/Shanghai")


def _pen_key(pen):
    # Native getters clone their values, so retain only immutable identities.
    return (str(pen.fx_a.dt), str(pen.fx_b.dt), float(pen.fx_a.fx), float(pen.fx_b.fx))


class CzscTeacher:
    """Reset native structure state for every exact student context.

    Confirmation timestamps are first observations during this context replay.
    They conservatively include the availability of the entire consumed prefix;
    they do not assert historical database revision/ingestion reconstruction.
    Optional, absent pens and zones remain masked. No native MA-assisted signal
    or high-period direction is silently treated as a confirmed trade event.
    """

    def __init__(self, source, context_bars=256, *, min_bi_len=6, max_bi_num=200):
        for name, value, minimum in (("context_bars", context_bars, 1),
                                     ("min_bi_len", min_bi_len, 3), ("max_bi_num", max_bi_num, 3)):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.source = source
        self.context_bars = context_bars
        # These are explicit teacher policy parameters, not mutable environment
        # overrides from the source project's business strategy configuration.
        self.native_config = {"min_bi_len": min_bi_len, "max_bi_num": max_bi_num}

    def prepare(self, symbol: str, as_of: str, start: str | None = None) -> dict[str, Any]:
        import numpy as np
        import pandas as pd

        symbol = str(symbol).strip().upper()
        if re.fullmatch(r"\d{6}", symbol):
            symbol += ".SH" if symbol.startswith("6") else ".SZ"
        if not re.fullmatch(r"(?:60\d{4}|68\d{4})\.SH|(?:00\d{4}|30\d{4})\.SZ", symbol):
            raise ValueError("CZSC teacher admits only sh_sz_a_v1 Shanghai/Shenzhen A shares")
        cutoff = _stamp(as_of, "as_of")
        if is_date_only(as_of):
            cutoff = cutoff.replace(hour=15)
        if cutoff < cutoff.normalize() + pd.Timedelta(hours=15):
            raise ValueError("daily CZSC teacher as_of must be at or after Shanghai 15:00")
        loaded = self.source.load_bars(symbol, start=start, as_of=cutoff.isoformat())
        if len(loaded) < self.context_bars:
            raise ValueError(f"CZSC teacher requires exactly {self.context_bars} context bars; got {len(loaded)}")
        raw = loaded.tail(self.context_bars).copy().reset_index(drop=True)
        sessions = pd.DatetimeIndex([_stamp(day, "bar date") for day in raw["date"]])
        if (raw["symbol"].astype(str) != symbol).any():
            raise ValueError("CZSC source returned a different or mixed symbol")
        if not sessions.equals(sessions.normalize()) or not sessions.is_unique or not sessions.is_monotonic_increasing:
            raise ValueError("CZSC context dates must be midnight, unique and strictly increasing")
        if sessions[-1] != cutoff.normalize():
            raise ValueError("CZSC teacher requires the actual as_of bar; stale history rejected")
        first, last = sessions[0].date().isoformat(), sessions[-1].date().isoformat()
        closes = pd.DatetimeIndex([day.replace(hour=15) for day in sessions])
        if "available_at" in raw:
            availability = pd.DatetimeIndex(availability_times(raw["available_at"])).tz_convert("Asia/Shanghai")
            availability_basis = "explicit_source_available_at"
        else:
            availability = closes
            availability_basis = "assumed_shanghai_session_close_15h_not_ingestion_evidence"
        if availability.isna().any() or (availability > cutoff).any() or (availability < closes).any():
            raise ValueError("CZSC context has unknown, unavailable, or pre-close source available_at")
        calendar = pd.DatetimeIndex([_stamp(day, "calendar date") for day in self.source.calendar_dates(first, last)])
        if not calendar.equals(sessions):
            raise ValueError("CZSC context does not exactly match the verified trading calendar; gaps rejected")
        matrix = raw.loc[:, list(OHLCVA_COLUMNS)].copy()
        matrix.index = sessions
        matrix["available_at"] = availability
        build_window(matrix, symbol=symbol, as_of=cutoff, context_length=self.context_bars,
                     horizon=1, calendar_dates=calendar)

        adapter = self.source.module("my_strategy.adapters.czsc_adapter")
        # Both calls are the frozen adapter's actual source/distribution checks;
        # unavailable or unverified native engines fail instead of approximating.
        engine_identity = adapter.engine_contract()
        engine = adapter.native_runtime()
        verified_sources = self.source.module("my_strategy.services.czsc_trend_support").VERIFIED_SOURCES
        reasons = []
        if "source" not in raw or not raw["source"].fillna("").isin(verified_sources).all():
            reasons.append("context_contains_unverified_raw_source")
        if "has_trade_price" not in raw or not raw["has_trade_price"].eq(1).all():
            reasons.append("context_contains_unverified_trade_prices")
        if raw.attrs.get("volume_unit") != "shares" or raw.attrs.get("amount_unit") != "CNY":
            reasons.append("context_volume_amount_units_unverified")
        if ((raw["volume"] <= 0) | (raw["amount"] <= 0)).any():
            reasons.append("context_contains_zero_turnover_sessions")
        # Rebuild identifiers and clock from the sliced context. Source ids or
        # old full-prefix state must never enter the native engine.
        raw["id"] = np.arange(len(raw), dtype=int)
        raw["dt"] = [stamp.tz_localize(None) for stamp in closes]
        native = None
        confirmations, zones_seen = {}, {}
        final_finished, final_zone = [], None
        for index, row in enumerate(raw.itertuples(index=False)):
            bar = adapter.to_raw_bar(row)
            if native is None:
                native = engine.CZSC([bar], **self.native_config)
            else:
                native.update(bar)
            observed_at = max(availability[:index + 1]).isoformat()
            final_finished = list(native.finished_bis)
            keys = {_pen_key(pen) for pen in final_finished}
            for pen in final_finished:
                confirmations.setdefault(_pen_key(pen), (observed_at, index))
            complete_zones = []
            for zone in native.zs_list:
                pens = list(zone.bis)
                if len(pens) < 3 or not zone.is_valid() or not all(_pen_key(pen) in keys for pen in pens[:3]):
                    continue
                zone_key = (tuple(_pen_key(pen) for pen in pens[:3]), float(zone.zd), float(zone.zg))
                zone_time = zones_seen.setdefault(zone_key, observed_at)
                complete_zones.append((zone, zone_time))
            final_zone = complete_zones[-1] if complete_zones else None

        targets, metadata = {}, {}
        dependency_start = closes[0].isoformat()
        overall_available = max(availability).isoformat()

        def put(name, value, *, confirmed_at=None, anchor_at=None, reason=None):
            value = float(value) if value is not None else None
            valid = not reasons and value is not None and math.isfinite(value)
            targets[name] = value if valid else None
            metadata[name] = {"dependency_start": dependency_start, "dependency_end": overall_available,
                              "available_at": overall_available, "confirmed_at": confirmed_at,
                              "anchor_at": anchor_at, "label_end": None, "label_mature_at": None,
                              "reason": ";".join(reasons) if reasons else (None if valid else reason or "native_fact_unavailable")}

        for offset in range(1, 4):
            pen = final_finished[-offset] if len(final_finished) >= offset else None
            confirmed_at = confirmations[_pen_key(pen)][0] if pen is not None else None
            anchor_at = _stamp(pen.fx_b.dt, "native pen anchor").isoformat() if pen is not None else None
            direction = {"向上": 1, "向下": -1}.get(str(pen.direction)) if pen is not None else None
            for suffix, value in (("direction", direction),
                                  ("return", float(pen.fx_b.fx) / float(pen.fx_a.fx) - 1 if pen is not None else None),
                                  ("length", float(pen.length) if pen is not None else None)):
                put(f"bi{offset}_{suffix}", value, confirmed_at=confirmed_at, anchor_at=anchor_at,
                    reason="no_confirmed_finished_pen_in_student_context")
        pen = final_finished[-1] if final_finished else None
        volume_ratio = None
        if pen is not None:
            ids = [int(bar.id) for bar in pen.raw_bars]
            if not ids or min(ids) < 0 or max(ids) >= len(raw):
                raise ValueError("native pen dependency escaped the student context")
            baseline = raw["volume"].iloc[max(0, len(raw) - 21):-1].mean()
            if len(raw) >= 21 and baseline > 0:
                volume_ratio = float(raw.iloc[ids]["volume"].mean() / baseline)
        pen_time = confirmations[_pen_key(pen)][0] if pen is not None else None
        put("bi1_volume_ratio", volume_ratio, confirmed_at=pen_time,
            reason="confirmed_pen_or_prior_twenty_volume_bars_unavailable")
        power_ratio = (float(pen.power_price) / float(final_finished[-3].power_price)
                       if len(final_finished) >= 3 and float(final_finished[-3].power_price) > 0 else None)
        put("bi1_power_ratio", power_ratio, confirmed_at=pen_time,
            reason="fewer_than_three_confirmed_pens_or_zero_reference_power")
        put("confirmed_age", len(raw) - 1 - confirmations[_pen_key(pen)][1] if pen is not None else None,
            confirmed_at=pen_time, reason="no_confirmed_finished_pen_in_student_context")
        for name, value in (("zone_low_distance", None), ("zone_high_distance", None), ("zone_width_ratio", None)):
            if final_zone is not None:
                zone, zone_time = final_zone
                close = float(raw.iloc[-1]["close"])
                value = {"zone_low_distance": close / float(zone.zd) - 1,
                         "zone_high_distance": close / float(zone.zg) - 1,
                         "zone_width_ratio": (float(zone.zg) - float(zone.zd)) / close}[name]
            put(name, value, confirmed_at=final_zone[1] if final_zone else None,
                reason="no_valid_zone_with_three_finished_pens_in_student_context")
        put("finished_bi_count", len(final_finished), confirmed_at=overall_available)
        context_columns = [name for name in ("symbol", "date", *OHLCVA_COLUMNS, "source", "has_trade_price", "available_at") if name in raw]
        context_hash = hashlib.sha256(raw[context_columns].to_csv(index=False, float_format="%.17g").encode()).hexdigest()
        definition = {"schema": TEACHER_SCHEMA, "continuous_names": CONTINUOUS_NAMES, "event_names": [],
                      "native_config": self.native_config, "context_bars": self.context_bars,
                      "source_engine": engine_identity,
                      "scope": "daily_native_finished_pens_and_finished_only_zones_reset_on_student_context"}
        identity = self.source.source_identity()
        identity["native_engine"] = engine_identity
        identity["teacher_adapter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return {"schema": TEACHER_SCHEMA,
                "schema_sha256": hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest(),
                "symbol": symbol, "requested_as_of": str(as_of), "as_of": cutoff.isoformat(),
                "context_start": first, "context_end": last, "context_rows": len(raw),
                "dependency_start": dependency_start, "dependency_end": overall_available,
                "available_at": overall_available, "label_end": None, "label_mature_at": None,
                "continuous_names": list(CONTINUOUS_NAMES),
                "continuous_targets": [targets[name] for name in CONTINUOUS_NAMES],
                "continuous_mask": [targets[name] is not None for name in CONTINUOUS_NAMES],
                "event_names": [], "event_targets": [], "event_mask": [],
                "target_metadata": metadata, "native_config": dict(self.native_config),
                "quality": {"eligible": not reasons, "reason_codes": reasons,
                            "native_czsc_facts": True, "window_local_state_reset": True,
                            "full_history_equivalence": False, "formal_qualified": False,
                            "raw_data_version": context_hash, "availability_basis": availability_basis,
                            "confirmation_basis": "first_observed_during_window_replay_with_prefix_source_availability",
                            "finished_count_scope": "current_native_retained_finished_bis",
                            "teacher_scope": "single_student_context_daily_native_facts"},
                "calendar_identity": self.source.calendar_identity(), "code_identity": identity}
