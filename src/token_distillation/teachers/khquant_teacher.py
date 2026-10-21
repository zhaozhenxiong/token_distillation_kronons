"""Window-local causal W32 targets from the read-only KHQuant source."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from token_distillation.adapters.khquant import KhquantSource, availability_times


TEACHER_SCHEMA = "khquant_window_local_w32_teacher_v1"
EVENT_NAMES = ("w_spring", "w_test", "w_sos", "w_lps", "w_upthrust", "w_sow", "w_lpsy")


class KhquantTeacher:
    """Recompute a single truncated context; no full-history CZSC claim.

    Numeric/event state is reset at context_start. Unknown events are masked;
    no-event zero is used only when the source marks a known eligible row.
    Native structure supervision is explicitly pending for all 22 slots.
    """

    def __init__(self, source: KhquantSource, context_bars: int = 256):
        if not isinstance(context_bars, int) or context_bars < 1:
            raise ValueError("context_bars must be a positive integer")
        self.source = source
        self.context_bars = context_bars

    def prepare(self, symbol: str, as_of: str, start: str | None = None) -> dict[str, Any]:
        import pandas as pd
        from token_distillation.data.windows import is_date_only, parse_timestamp
        cutoff = parse_timestamp(as_of, "as_of")
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("Asia/Shanghai")
        else:
            cutoff = cutoff.tz_convert("Asia/Shanghai")
        if is_date_only(as_of):
            cutoff = cutoff.replace(hour=15)
        if pd.isna(cutoff) or cutoff < cutoff.normalize() + pd.Timedelta(hours=15):
            raise ValueError("daily teacher as_of must be at or after Shanghai 15:00")
        raw = self.source.load_bars(symbol, start=start, as_of=cutoff.isoformat())
        attrs = dict(raw.attrs)
        raw = raw.tail(self.context_bars).copy().reset_index(drop=True)
        # The frozen adapter's quality attributes summarize its entire read.
        # Recompute its documented row-quality summary on this exact context,
        # so earlier history cannot change otherwise identical student targets.
        if "source" in raw and attrs.get("price_basis") in {
                "unadjusted", "mixed_legacy_adjustment_unverified"}:
            adapter = self.source.module("my_strategy.adapters.czsc_adapter")
            if hasattr(adapter, "legacy_fuyao_source"):
                legacy = int(raw["source"].map(adapter.legacy_fuyao_source).sum())
                attrs.update(legacy_fuyao_bars=legacy,
                    price_basis="mixed_legacy_adjustment_unverified" if legacy else "unadjusted")
        if "has_trade_price" in raw:
            attrs["unverified_trade_price_bars"] = int(raw["has_trade_price"].ne(1).sum())
        raw.attrs = attrs
        first = pd.Timestamp(raw.iloc[0]["date"]).date().isoformat()
        last = pd.Timestamp(raw.iloc[-1]["date"]).date().isoformat()
        if last != cutoff.date().isoformat():
            raise ValueError("teacher requires an actual closed bar on the requested as_of date; stale history rejected")
        if len(raw) != self.context_bars:
            raise ValueError(f"teacher requires exactly {self.context_bars} context bars; got {len(raw)}")
        context_columns = [name for name in ("symbol", "date", "open", "high", "low", "close",
            "volume", "amount", "source", "has_trade_price", "available_at") if name in raw]
        context_hash = hashlib.sha256(raw[context_columns].to_csv(index=False, float_format="%.10g").encode()).hexdigest()
        calendar = self.source.calendar_dates(first, last)
        feature_module = self.source.module("my_strategy.services.czsc_price_volume_features")
        computed = feature_module.prepare_wyckoff_inputs(raw, raw=raw, config={"market_dates": calendar})
        row = computed.iloc[-1]
        eligible = bool(row["wyckoff_input_eligible"])
        reasons = list(row["wyckoff_reason_codes"])
        event = str(row["wyckoff_event"])
        event_known = eligible and event not in {"unknown", "conflict", "None", "nan"}
        names = tuple(name for name in feature_module.W_INPUT_COLUMNS if name not in EVENT_NAMES)
        values, masks = [], []
        for name in names:
            value = float(row[name])
            valid = eligible and math.isfinite(value)
            if name in {"w_event_age", "w_demand_score", "w_supply_score"}:
                valid = valid and event_known
            values.append(value if math.isfinite(value) else None)
            masks.append(valid)
        event_values = [float(row[name]) if event_known else None for name in EVENT_NAMES]
        available = last + "T15:00:00+08:00"
        availability_basis = "assumed_shanghai_session_close_15h_not_ingestion_evidence"
        # A late-arriving raw row is not available merely because its close is.
        if "available_at" in raw:
            availability_basis = "explicit_source_available_at"
            visibility = availability_times(raw["available_at"])
            if visibility.isna().any() or (visibility > cutoff).any():
                raise ValueError("teacher context includes unavailable or unknown-availability raw rows")
            maximum = max(visibility.max(), pd.Timestamp(available).tz_convert("UTC"))
            available = maximum.tz_convert("Asia/Shanghai").isoformat()
        identity = self.source.source_identity()
        identity["teacher_adapter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        schema = {"schema": TEACHER_SCHEMA, "scope": "independent_recompute_on_student_context",
            "context_bars": self.context_bars, "continuous_names": names, "event_names": EVENT_NAMES,
            "source_wyckoff_schema_hash": computed.attrs["wyckoff_schema_hash"],
            "structure_status": "pending_native_structure_adapter"}
        return {"schema": TEACHER_SCHEMA,
            "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
            "symbol": str(raw.iloc[-1]["symbol"]), "requested_as_of": str(as_of),
            "as_of": cutoff.isoformat(),
            "context_start": first, "context_end": last, "context_rows": len(raw),
            "dependency_start": first + "T15:00:00+08:00", "dependency_end": available,
            "available_at": available, "label_end": None, "label_mature_at": None,
            "continuous_names": list(names), "continuous_targets": values, "continuous_mask": masks,
            "event_names": list(EVENT_NAMES), "event_targets": event_values,
            "event_mask": [event_known] * len(EVENT_NAMES),
            "structure_targets": [None] * 22, "structure_mask": [False] * 22,
            "structure_status": "pending_native_structure_adapter",
            "quality": {"eligible": eligible, "reason_codes": reasons, "observed_event": event,
                "window_local_state_reset": True, "formal_qualified": False,
                "native_czsc_equivalence": False, "full_history_equivalence": False,
                "raw_data_version": context_hash,
                "source_load_data_version": attrs.get("data_version"),
                "price_basis": attrs.get("price_basis"),
                "availability_basis": availability_basis,
                "teacher_scope": "single_student_context_independently_recomputed"},
            "calendar_identity": self.source.calendar_identity(), "code_identity": identity}
