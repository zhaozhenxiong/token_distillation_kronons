"""Conservative, current-bar ICT primitives; no upstream setup-state supervision."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from token_distillation.adapters.ict import FrozenIctAdapter
from token_distillation.data.contracts import OHLCVA_COLUMNS
from token_distillation.data.windows import build_window, is_date_only, parse_timestamp


TEACHER_SCHEMA = "ict_causal_atomic_teacher_v1"
CONTINUOUS_NAMES = ("ict_swing_high_distance_frac", "ict_swing_low_distance_frac")
EVENT_NAMES = ("ict_fvg_bullish_now", "ict_fvg_bearish_now", "ict_sellside_sweep_now",
               "ict_buyside_sweep_now", "ict_close_break_high_now", "ict_close_break_low_now")
# Explicit snapshot of KHQuant czsc_trend_support.VERIFIED_SOURCES. It is part
# of this adapter's schema, rather than trusting an attrs/config price-basis flag.
VERIFIED_SOURCES = frozenset({"tushare", "tushare:daily", "fuyao_unadjusted_v1", "baostock",
                              "eastmoney_unadjusted_v1", "sina_unadjusted_v1",
                              "tencent_star_unadjusted_v1"})
_PENDING = {
    "premium": "pending_validated_confirmed_dealing_range",
    "discount": "pending_validated_confirmed_dealing_range",
    "ote_long": "pending_validated_confirmed_dealing_range_and_direction",
    "ote_short": "pending_validated_confirmed_dealing_range_and_direction",
    "order_block": "pending_causal_order_block_definition_and_invalidation_validation",
    "breaker": "pending_causal_breaker_definition_and_invalidation_validation",
    "ifvg": "pending_causal_inverse_fvg_invalidation_and_retest_validation",
    "setup_status": "pending_causal_setup_sequence_and_invalidation_validation",
    "new_york_session": "not_applicable_to_A_share_daily_bars",
    "multi_timeframe_bias": "pending_independently_available_multi_timeframe_inputs",
}


def _normal_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    if re.fullmatch(r"\d{6}", symbol):
        symbol += ".SH" if symbol.startswith("6") else ".SZ"
    if not re.fullmatch(r"(?:60\d{4}|68\d{4})\.SH|(?:00\d{4}|30\d{4})\.SZ", symbol):
        raise ValueError("ICT teacher admits only sh_sz_a_v1 Shanghai/Shenzhen A shares")
    return symbol


def _shanghai(value, field):
    stamp = parse_timestamp(value, field)
    return stamp.tz_localize("Asia/Shanghai") if stamp.tzinfo is None else stamp.tz_convert("Asia/Shanghai")


def _calendar_identity(source, first: str, last: str):
    """KhquantSource has already checked retained calendar/source bytes here."""
    identity = source.calendar_identity()
    digest = re.compile(r"^[0-9a-f]{64}$")
    if (not isinstance(identity, dict) or identity.get("independent_from_market_bars") is not True
            or not identity.get("source") or not identity.get("run_id")
            or not digest.fullmatch(str(identity.get("file_sha256", "")))
            or not digest.fullmatch(str(identity.get("dates_hash", "")))
            or not identity.get("source_artifacts")
            or str(identity.get("first_session", "")) > first
            or str(identity.get("last_session", "")) < last):
        raise ValueError("ICT teacher requires an independently verified calendar identity and source hashes")
    for artifact in identity["source_artifacts"]:
        if (not isinstance(artifact, dict) or not artifact.get("path")
                or not digest.fullmatch(str(artifact.get("sha256", "")))):
            raise ValueError("ICT teacher calendar source artifact identity is incomplete")
    return identity


class IctTeacher:
    """Independently recompute only the final origin's 2 + 6 atomic targets.

    Date-only origins mean Shanghai 15:00. Explicit times retain their precision.
    Current continuous distances may use a swing confirmed on the current bar.
    Current sweep/break events require a pivot whose right-side confirmation was
    already available at the previous session's close. No historical event is
    reconstructed using the final prefix's pivots. Native candidate/setup status
    is neither executed nor copied into labels or diagnostic metadata.
    """

    def __init__(self, source, vendor_root: str | Path, context_bars: int = 256,
                 swing_width: int = 2):
        for name, value in (("context_bars", context_bars), ("swing_width", swing_width)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.source = source
        self.adapter = FrozenIctAdapter(vendor_root)
        self.context_bars = context_bars
        self.swing_width = swing_width

    def prepare(self, symbol: str, as_of: str) -> dict[str, Any]:
        import pandas as pd

        symbol = _normal_symbol(symbol)
        cutoff = _shanghai(as_of, "as_of")
        if is_date_only(as_of):
            cutoff = cutoff.replace(hour=15)
        if cutoff < cutoff.normalize() + pd.Timedelta(hours=15):
            raise ValueError("daily ICT teacher as_of must be at or after Shanghai 15:00")
        # Verify frozen code before invoking even its pure geometric primitives.
        vendor_identity = self.adapter.verify()
        native = self.adapter.module()
        raw = self.source.load_bars(symbol, as_of=cutoff.isoformat())
        if not isinstance(raw, pd.DataFrame) or raw.empty or "date" not in raw or "symbol" not in raw:
            raise ValueError("ICT source must return nonempty dated single-symbol bars")
        attrs = dict(raw.attrs)
        sessions = pd.DatetimeIndex([_shanghai(value, "bar date") for value in raw["date"]])
        if not sessions.equals(sessions.normalize()) or not sessions.is_unique or not sessions.is_monotonic_increasing:
            raise ValueError("ICT source session dates must be midnight, unique and strictly increasing")
        if (raw["symbol"].astype(str) != symbol).any():
            raise ValueError("ICT source returned a different or mixed symbol")
        if (sessions > cutoff.normalize()).any():
            raise ValueError("ICT source returned future bars after as_of")
        if sessions[-1] != cutoff.normalize():
            raise ValueError("ICT teacher requires the actual as_of bar; stale history rejected")
        if len(raw) < self.context_bars:
            raise ValueError(f"ICT teacher requires exactly {self.context_bars} context bars; got {len(raw)}")
        closes = pd.DatetimeIndex([day.replace(hour=15) for day in sessions])
        if "available_at" in raw:
            availability = pd.DatetimeIndex([_shanghai(value, "available_at") for value in raw["available_at"]])
            availability_basis = "explicit_source_available_at"
        else:
            availability = closes
            availability_basis = "assumed_shanghai_session_close_15h_not_ingestion_evidence"
        if (availability < closes).any():
            raise ValueError("ICT source daily available_at precedes Shanghai session close at 15:00")
        if (availability > cutoff).any():
            raise ValueError("ICT source returned data unavailable at as_of")

        # Slice before feature computation; older prefix state never enters the primitives.
        context = raw.tail(self.context_bars).copy().reset_index(drop=True)
        sessions = sessions[-self.context_bars:]
        availability = availability[-self.context_bars:]
        closes = closes[-self.context_bars:]
        first, last = sessions[0].date().isoformat(), sessions[-1].date().isoformat()
        calendar = self.source.calendar_dates(first, last)
        expected = pd.DatetimeIndex([_shanghai(value, "verified calendar session") for value in calendar])
        if not expected.equals(sessions):
            raise ValueError("ICT context does not exactly match the verified trading calendar; gaps rejected")
        calendar_identity = _calendar_identity(self.source, first, last)
        frame = context.loc[:, list(OHLCVA_COLUMNS)].copy()
        frame.index = sessions
        frame["available_at"] = availability
        # The shared builder checks finite OHLCVA, price bounds, timestamp order,
        # exact context length, all source times, and calendar membership.
        build_window(frame, symbol=symbol, as_of=cutoff, context_length=self.context_bars,
                     horizon=1, calendar_dates=expected)
        candles = [native.Candle(timestamp=closes[i].to_pydatetime(), open=float(row.open),
                                high=float(row.high), low=float(row.low), close=float(row.close),
                                volume=float(row.volume))
                   for i, row in enumerate(context.itertuples(index=False))]
        swings = native.detect_swings(candles, self.swing_width)
        gaps = native.detect_fvgs(candles)
        now = len(candles) - 1
        current = candles[now]
        observed = []
        for swing in swings:
            anchor = int(swing["index"])
            confirm = anchor + self.swing_width
            observed.append({"kind": swing["kind"], "price": float(swing["price"]),
                             "anchor_index": anchor, "anchor_at": closes[anchor].isoformat(),
                             "observed_index": confirm, "observed_at": closes[confirm].isoformat(),
                             "available_at": max(availability[anchor - self.swing_width:confirm + 1]).isoformat(),
                             "dependency_start": closes[anchor - self.swing_width].isoformat()})

        quality_reasons = []
        if attrs.get("price_basis") != "unadjusted":
            quality_reasons.append("unadjusted_price_basis_unverified")
        if "has_trade_price" not in context:
            quality_reasons.append("trade_price_verification_missing")
        elif not context["has_trade_price"].eq(1).all():
            quality_reasons.append("context_contains_unverified_trade_prices")
        if "source" not in context:
            quality_reasons.append("raw_source_verification_missing")
        elif not context["source"].fillna("").astype(str).isin(VERIFIED_SOURCES).all():
            quality_reasons.append("context_contains_unverified_raw_source")
        eligible = not quality_reasons
        overall_available = max(availability).isoformat()
        target_metadata = {}
        continuous_targets, continuous_masks = [], []

        def latest(kind, *, before_previous_close=False):
            candidates = [s for s in observed if s["kind"] == kind]
            if before_previous_close:
                candidates = [s for s in candidates if s["observed_index"] <= now - 1 and
                              pd.Timestamp(s["available_at"]) <= closes[now - 1]] if now else []
            return max(candidates, key=lambda item: item["anchor_index"]) if candidates else None

        def evidence(reason=None, reference=None, start=None, available=None):
            result = {"dependency_start": start or closes[0].isoformat(),
                      "dependency_end": closes[now].isoformat(),
                      "label_end": closes[now].isoformat(),
                      "label_mature_at": available or overall_available,
                      "available_at": available or overall_available,
                      "reason": (";".join(quality_reasons) if not eligible else reason)}
            if reference is not None:
                result["reference"] = dict(reference)
            return result

        for kind, name in zip(("high", "low"), CONTINUOUS_NAMES):
            pivot = latest(kind)
            valid = eligible and pivot is not None
            value = ((pivot["price"] - current.close) / current.close if kind == "high" else
                     (current.close - pivot["price"]) / current.close) if valid else None
            continuous_targets.append(value)
            continuous_masks.append(valid)
            target_metadata[name] = evidence(reason=None if pivot else "no_confirmed_swing_in_student_context",
                                             reference=pivot)

        event_values, event_masks = {}, {}
        current_gap = next((gap for gap in gaps if gap["end_index"] == now), None)
        fvg_known = eligible and len(candles) >= 3
        fvg_available = max(availability[-3:]).isoformat()
        for direction, name in zip(("bullish", "bearish"), EVENT_NAMES[:2]):
            event_values[name] = float(current_gap is not None and current_gap["direction"] == direction) if fvg_known else None
            event_masks[name] = fvg_known
            meta = evidence(reason=None if len(candles) >= 3 else "fewer_than_three_bars",
                            start=closes[max(0, now - 2)].isoformat(), available=fvg_available)
            meta.update(formation_index=now, formation_at=closes[now].isoformat(),
                        upstream_middle_bar_time_is_not_availability=True)
            if current_gap is not None:
                meta.update(start_index=int(current_gap["start_index"]),
                            middle_index=int(current_gap["middle_index"]),
                            end_index=int(current_gap["end_index"]),
                            zone=[float(value) for value in current_gap["zone"]])
            target_metadata[name] = meta

        previous = candles[now - 1] if now else None
        prior_high, prior_low = latest("high", before_previous_close=True), latest("low", before_previous_close=True)
        rules = {
            EVENT_NAMES[2]: (prior_low, lambda price: current.low < price and current.close > price),
            EVENT_NAMES[3]: (prior_high, lambda price: current.high > price and current.close < price),
            EVENT_NAMES[4]: (prior_high, lambda price: previous.close <= price < current.close),
            EVENT_NAMES[5]: (prior_low, lambda price: previous.close >= price > current.close),
        }
        for name, (reference, rule) in rules.items():
            valid = eligible and previous is not None and reference is not None
            event_values[name] = float(rule(reference["price"])) if valid else None
            event_masks[name] = valid
            target_metadata[name] = evidence(reason=None if reference else "no_pivot_known_by_previous_session_close",
                                             reference=reference)

        hash_frame = context.loc[:, [name for name in ("symbol", *OHLCVA_COLUMNS, "has_trade_price", "source") if name in context]].copy()
        hash_frame.insert(0, "date", sessions)
        hash_frame["available_at"] = availability
        context_hash = hashlib.sha256(hash_frame.to_csv(index=False, float_format="%.17g").encode()).hexdigest()
        schema_definition = {"schema": TEACHER_SCHEMA, "continuous_names": CONTINUOUS_NAMES,
                             "event_names": EVENT_NAMES, "context_bars": self.context_bars,
                             "swing_width": self.swing_width, "pending_targets": _PENDING,
                             "verified_raw_sources": sorted(VERIFIED_SOURCES),
                             "event_pivot_lag": "confirmed_and_available_by_previous_session_close",
                             "close_break_rule": "previous_close_crosses_reference_on_current_close"}
        return {"schema": TEACHER_SCHEMA,
                "schema_sha256": hashlib.sha256(json.dumps(schema_definition, sort_keys=True).encode()).hexdigest(),
                "symbol": symbol, "requested_as_of": str(as_of), "as_of": cutoff.isoformat(),
                "scope": "single_student_context_current_bar_atomic_ict",
                "context_start": first, "context_end": last, "context_rows": len(context),
                "dependency_start": closes[0].isoformat(), "dependency_end": closes[-1].isoformat(),
                "available_at": overall_available, "label_end": None, "label_mature_at": None,
                "continuous_names": list(CONTINUOUS_NAMES), "continuous_targets": continuous_targets,
                "continuous_mask": continuous_masks, "event_names": list(EVENT_NAMES),
                "event_targets": [event_values[name] for name in EVENT_NAMES],
                "event_mask": [event_masks[name] for name in EVENT_NAMES],
                "target_metadata": target_metadata, "confirmed_swings": observed,
                "pending_targets": {name: {"target": None, "mask": False, "reason": reason}
                                    for name, reason in _PENDING.items()},
                "quality": {"eligible": eligible, "reason_codes": quality_reasons,
                            "window_local_state_reset": True, "formal_qualified": False,
                            "full_ict_model_implemented": False, "native_setup_status_supervised": False,
                            "upstream_full_scanner_executed": False, "historical_events_reconstructed": False,
                            "raw_data_version": context_hash, "source_load_data_version": attrs.get("data_version"),
                            "price_basis": attrs.get("price_basis"), "availability_basis": availability_basis,
                            "verified_raw_sources": sorted(VERIFIED_SOURCES),
                            "calendar_verified_by": "KhquantSource_verified_calendar_contract"},
                "calendar_identity": calendar_identity,
                "provenance": {"vendor": vendor_identity, "source": self.source.source_identity(),
                               "teacher_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "reused_primitives": ["detect_swings", "detect_fvgs"],
                               "causal_adapter_scope": "independent_schema_not_native_setup_equivalence"}}
