#!/usr/bin/env python3
"""Inspect an explicit research population using frozen read-only source inputs.

Admission uses only pretraining availability, verification and 2023 turnover.
Later coverage is reported for complete-window filtering, never for ranking or
removing symbols after observing outcomes. This is a retrospective convenience
population, not a historically reconstructed, survivor-free market universe.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from token_distillation.adapters.khquant import KhquantSource
from token_distillation.config import load_config
from token_distillation.teachers.ict_teacher import VERIFIED_SOURCES


CANDIDATES = (
    ("600519.SH", "sh_main", "consumer"),
    ("600036.SH", "sh_main", "bank"),
    ("601318.SH", "sh_main", "insurance"),
    ("600000.SH", "sh_main", "bank"),
    ("600900.SH", "sh_main", "utilities"),
    ("601398.SH", "sh_main", "bank"),
    ("600030.SH", "sh_main", "broker"),
    ("600276.SH", "sh_main", "healthcare"),
    ("601899.SH", "sh_main", "materials"),
    ("600887.SH", "sh_main", "consumer"),
    ("000001.SZ", "sz_main", "bank"),
    ("000333.SZ", "sz_main", "consumer_durables"),
    ("000651.SZ", "sz_main", "consumer_durables"),
    ("000858.SZ", "sz_main", "consumer"),
    ("002415.SZ", "sz_main", "technology"),
    ("002594.SZ", "sz_main", "automotive"),
    ("300750.SZ", "chinext", "battery"),
    ("300059.SZ", "chinext", "financial_services"),
    ("688981.SH", "star", "semiconductors"),
    ("688008.SH", "star", "technology"),
)


def candidate_summary(frame, calendar, *, train_start: str, context_length: int,
                      liquidity_start: str, minimum_median_amount: float) -> dict:
    """Admission is unchanged by changes in posttraining prices or coverage."""
    dates = frame["date"].dt.strftime("%Y-%m-%d")
    required = {"source", "has_trade_price", "volume", "amount"}
    if not required.issubset(frame.columns):
        raise ValueError("raw source/trade-price verification columns are required")
    good = (frame["source"].isin(VERIFIED_SOURCES)
            & frame["has_trade_price"].eq(1)
            & frame["volume"].gt(0) & frame["amount"].gt(0))
    previous_sessions = [day for day in calendar if day < train_start]
    if len(previous_sessions) < context_length:
        raise ValueError("verified calendar has insufficient pretraining context")
    selection_context = previous_sessions[-context_length:]
    observed_good = set(dates[good])
    pretraining_missing = sorted(set(selection_context) - observed_good)
    liquid = good & dates.ge(liquidity_start) & dates.lt(train_start)
    amount = frame.loc[liquid, "amount"].to_numpy(dtype=float)
    median = float(np.median(amount)) if len(amount) else None
    reasons = []
    if pretraining_missing:
        reasons.append("missing_or_unverified_pretraining_context")
    if median is None or median < minimum_median_amount:
        reasons.append("insufficient_pretraining_turnover")
    first_origin = next(day for day in calendar if day >= train_start)
    first_origin_index = calendar.index(first_origin)
    context_start = calendar[first_origin_index - context_length + 1]
    later_sessions = [day for day in calendar if day >= context_start]
    missing_later = sorted(set(later_sessions) - set(dates))
    bad_later = dates[(dates >= context_start) & ~good].tolist()
    return {
        "admitted": not reasons, "pretraining_admission_reasons": reasons,
        "admission_uses_posttraining_outcomes": False,
        "rows_read": len(frame), "first_stored_date_within_read": dates.min(),
        "last_stored_date_within_read": dates.max(),
        "first_verified_trade_date_within_read": dates[good].min() if good.any() else None,
        "source_counts": {str(k): int(v) for k, v in frame["source"].value_counts(dropna=False).items()},
        "source_frame_data_version": frame.attrs.get("data_version"),
        "price_basis_as_reported_by_adapter": frame.attrs.get("price_basis"),
        "selection_context_start": selection_context[0],
        "selection_context_end": selection_context[-1],
        "selection_context_missing_or_unverified": pretraining_missing,
        "liquidity_window": [liquidity_start, previous_sessions[-1]],
        "liquidity_verified_rows": int(liquid.sum()),
        "pretraining_median_daily_amount_cny": median,
        "first_training_origin": first_origin, "first_training_context_start": context_start,
        "missing_sessions_from_first_training_context": missing_later,
        "unverified_or_zero_turnover_from_first_training_context": bad_later,
        "later_gap_policy": "retain_symbol; dataset_rejects_incomplete_windows_without_imputation",
        "explicit_available_at_column_present": "available_at" in frame.columns,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/ablation.toml"))
    parser.add_argument("--history-start", default="2015-01-05")
    parser.add_argument("--end", default="2026-10-16")
    parser.add_argument("--liquidity-start", default="2023-01-01")
    parser.add_argument("--minimum-median-amount", type=float, default=100_000_000.)
    parser.add_argument("--output", default="artifacts/reports/ablation_population.local.json")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    train_start = config.values["splits"]["train"][0]
    for value in (args.history_start, args.end, args.liquidity_start):
        datetime.strptime(value, "%Y-%m-%d")
    if (args.liquidity_start >= train_start or args.history_start > args.liquidity_start
            or not np.isfinite(args.minimum_median_amount) or args.minimum_median_amount < 0):
        parser.error("liquidity must be measured before training with a finite nonnegative threshold")
    output = config.output_path(args.output)
    if output.exists():
        raise FileExistsError(f"population evidence is immutable; choose another --output: {output}")
    source = KhquantSource(config.source_app, config.raw_db, config.state_root,
                           calendar_path=config.calendar_path)
    calendar = source.calendar_dates(args.history_start, args.end)
    before_stat = config.raw_db.stat()
    rows = []
    for symbol, board, category in CANDIDATES:
        frame = source.load_bars(symbol, start=args.history_start, end=args.end)
        rows.append({"symbol": symbol, "board": board, "category": category,
                     **candidate_summary(frame, calendar, train_start=train_start,
                         context_length=config.context_length, liquidity_start=args.liquidity_start,
                         minimum_median_amount=args.minimum_median_amount)})
    after_stat = config.raw_db.stat()
    payload = {
        "schema": "read_only_ablation_population_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "related_khquant_audit_session": "20261019-134115-83f5b455",
        "universe": "sh_sz_a_v1", "candidate_order": "explicit_fixed_list_before_ablation",
        "selection_basis": "pretrain_verified_context_and_2023_liquidity_only",
        "selection_minimum_median_amount_cny": args.minimum_median_amount,
        "candidate_count": len(CANDIDATES),
        "admitted_symbols": [row["symbol"] for row in rows if row["admitted"]],
        "board_counts": dict(Counter(row["board"] for row in rows if row["admitted"])),
        "history_read_start": args.history_start, "label_through": args.end,
        "splits": config.values["splits"], "context_length": config.context_length,
        "sources_read_only": True, "network_used": False,
        "source_identity": source.source_identity(), "calendar_identity": source.calendar_identity(),
        "raw_db_stat_unchanged_during_read": (before_stat.st_size, before_stat.st_mtime_ns)
            == (after_stat.st_size, after_stat.st_mtime_ns),
        "limitations": [
            "Explicit familiar liquid names constitute a retrospective convenience population; not survivor-free.",
            "No future returns, teacher performance or ablation results are used for admission or ranking.",
            "Posttraining gaps are reported, not used to remove candidates; complete windows must be verified separately.",
            "Sources without available_at use exchange-close reconstruction; actual historical ingestion availability is unverified.",
            "The live database is not immutable; dataset generation must freeze and hash actual rows independently.",
        ],
        "candidates": rows,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                      "admitted_count": len(payload["admitted_symbols"]),
                      "symbols": payload["admitted_symbols"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
