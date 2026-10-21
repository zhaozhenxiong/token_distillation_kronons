"""Explicit, offline generation of immutable causal teacher/student datasets.

The adapter is read once per symbol. Every teacher then reads the very same
frozen, context-only rows as the student, rather than rereading a live database.
No tokenizer, model, training, network request or source output is initialized.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import date, datetime, time
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from token_distillation.adapters.khquant import KhquantSource, availability_times
from token_distillation.artifacts import jsonable, sha256_file, write_json
from token_distillation.teachers.ict_teacher import VERIFIED_SOURCES
from token_distillation.teachers.multi_teacher import MultiTeacherPreview

from .contracts import OHLCVA_COLUMNS
from .normalization import normalize_window
from .quality import validate_raw_quality
from .splits import SPLIT_NAMES, SplitRange, TimeSplits, assign_window
from .windows import build_window, parse_timestamp


SCHEMA = "causal_dual_teacher_dataset_v1"
SAMPLE_SCHEMA = "causal_teacher_student_sample_v1"
SHANGHAI = ZoneInfo("Asia/Shanghai")


def _canonical(value: Any) -> bytes:
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _day(value: str, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{field} must be an explicit YYYY-MM-DD date")
    return date.fromisoformat(value).isoformat()


def _symbol(value: str) -> str:
    value = str(value).strip().upper()
    if re.fullmatch(r"\d{6}", value):
        value += ".SH" if value.startswith("6") else ".SZ"
    if not re.fullmatch(r"(?:60\d{4}|68\d{4})\.SH|(?:00\d{4}|30\d{4})\.SZ", value):
        raise ValueError("only sh_sz_a_v1 Shanghai/Shenzhen A shares are admitted")
    return value


def _sessions(values) -> pd.DatetimeIndex:
    stamps = []
    for value in values:
        stamp = parse_timestamp(value, "session date")
        stamps.append(stamp.tz_localize("Asia/Shanghai") if stamp.tzinfo is None
                      else stamp.tz_convert("Asia/Shanghai"))
    result = pd.DatetimeIndex(stamps)
    if (len(result) == 0 or not result.is_unique or not result.is_monotonic_increasing
            or not result.equals(result.normalize())):
        raise ValueError("session dates must be nonempty, midnight, unique and increasing")
    return result


def _raw_records(frame: pd.DataFrame) -> list[dict]:
    records = []
    for row in frame.to_dict("records"):
        record = {name: row[name] for name in ("symbol", *OHLCVA_COLUMNS, "source", "has_trade_price")
                  if name in row}
        record["date"] = parse_timestamp(row["date"]).date().isoformat()
        stamp = parse_timestamp(row.get("available_at", record["date"] + "T15:00:00"))
        stamp = stamp.tz_localize("Asia/Shanghai") if stamp.tzinfo is None else stamp.tz_convert("Asia/Shanghai")
        record["available_at"] = stamp.isoformat()
        records.append(jsonable(record))
    return records


class _SnapshotSource:
    """Read-only teacher protocol over the adapter's frozen per-symbol read."""

    def __init__(self, source, frames, calendar, calendar_identity, source_identity, context_length):
        self.source = source
        self.frames = frames
        self.calendar = calendar
        self._calendar_identity = calendar_identity
        self._source_identity = source_identity
        self.context_length = context_length
        self.served_hashes: list[str] = []
        self._indices = {symbol: _sessions(frame["date"]) for symbol, frame in frames.items()}

    def module(self, name):
        return self.source.module(name)

    def source_identity(self):
        return deepcopy(self._source_identity)

    def calendar_identity(self):
        return deepcopy(self._calendar_identity)

    def calendar_dates(self, start=None, end=None):
        return [day for day in self.calendar if (start is None or day >= start) and (end is None or day <= end)]

    def load_bars(self, symbol, start=None, end=None, as_of=None):
        symbol = _symbol(symbol)
        cutoff = parse_timestamp(as_of or end, "teacher as_of")
        cutoff = cutoff.tz_localize("Asia/Shanghai") if cutoff.tzinfo is None else cutoff.tz_convert("Asia/Shanghai")
        frame = self.frames[symbol]
        index = self._indices[symbol]
        keep = index <= cutoff.normalize()
        if start is not None:
            keep &= index >= pd.Timestamp(start, tz="Asia/Shanghai")
        # Availability is checked by the causal window before teacher execution.
        # Retain missing/late rows here so the teachers also fail closed.
        result = frame.loc[keep].tail(self.context_length).copy().reset_index(drop=True)
        result.attrs = dict(frame.attrs)
        if (result.attrs.get("price_basis") in {"unadjusted", "mixed_legacy_adjustment_unverified"}
                and "source" in result and result["source"].isin(VERIFIED_SOURCES).all()):
            result.attrs["price_basis"] = "unadjusted"
        digest = _hash(_raw_records(result))
        result.attrs["data_version"] = digest
        self.served_hashes.append(digest)
        return result


def _splits(config) -> TimeSplits:
    plain = TimeSplits.from_mapping({name: config.values["splits"][name] for name in SPLIT_NAMES})
    return TimeSplits(**{name: SplitRange(name, getattr(plain, name).start.replace(tzinfo=SHANGHAI),
                                        getattr(plain, name).end.replace(tzinfo=SHANGHAI))
                         for name in SPLIT_NAMES})


def _time_features(dates) -> list[list[int]]:
    return [[stamp.minute, stamp.hour, stamp.weekday(), stamp.day, stamp.month] for stamp in dates]


def _teacher_contract(teacher: dict, window, context_sha256: str) -> dict:
    teacher = deepcopy(teacher)
    expected = {"symbol": window.symbol, "as_of": window.as_of.isoformat(),
                "context_start": window.context_dates[0].date().isoformat(),
                "context_end": window.context_dates[-1].date().isoformat(),
                "context_rows": len(window.context_dates)}
    for field, value in expected.items():
        if teacher.get(field) != value:
            raise ValueError(f"teacher/student input mismatch: {field}")
    minimum = window.context_dates[0].replace(hour=15)
    if teacher.get("dependency_start") and parse_timestamp(teacher["dependency_start"]) < minimum:
        raise ValueError("teacher dependency starts before student context")
    for field in ("available_at", "dependency_end", "confirmed_at", "observed_at"):
        if teacher.get(field) and parse_timestamp(teacher[field]) > window.as_of:
            raise ValueError(f"historical teacher uses future {field}")
    known = 0
    for kind in ("continuous", "event", "structure"):
        targets, masks = teacher.get(kind + "_targets", []), teacher.get(kind + "_mask", [])
        if len(targets) != len(masks):
            raise ValueError(f"teacher {kind} targets/masks differ in length")
        if kind != "structure" and len(teacher.get(kind + "_names", [])) != len(targets):
            raise ValueError(f"teacher {kind} target names differ in length")
        names = teacher.get(kind + "_names", [])
        if len(names) != len(set(names)) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("teacher target names must be unique nonempty strings")
        for i, (target, mask) in enumerate(zip(targets, masks)):
            if type(mask) is not bool:
                raise ValueError("teacher masks must be explicit booleans")
            if mask:
                if target is None or not np.isfinite(float(target)):
                    raise ValueError("known teacher target is nonfinite or null")
                if kind == "event" and float(target) not in (0.0, 1.0):
                    raise ValueError("known event target must be zero or one")
                known += 1
            else:
                targets[i] = None  # unknown never silently becomes a no-event zero
    for meta in teacher.get("target_metadata", {}).values():
        if meta.get("dependency_start") and parse_timestamp(meta["dependency_start"]) < minimum:
            raise ValueError("teacher target dependency starts before student context")
        for field in ("available_at", "dependency_end", "label_end", "label_mature_at", "confirmed_at", "observed_at", "anchor_at"):
            if meta.get(field):
                stamp = parse_timestamp(meta[field])
                # The native CZSC API represents exchange-local pen anchors
                # without tzinfo. Older immutable datasets retain that exact
                # anchor diagnostic; availability/confirmation stays explicit.
                if field == "anchor_at" and stamp.tzinfo is None:
                    stamp = stamp.tz_localize("Asia/Shanghai")
                if stamp.tzinfo is None or stamp > window.as_of:
                    raise ValueError(f"historical teacher target has ambiguous or future {field}")
    teacher["shared_input_sha256"] = context_sha256
    teacher["dataset_label_contract"] = {
        "semantics": "historical_fact_available_at_origin_not_future_return",
        "dependency_start": window.context_dates[0].replace(hour=15).isoformat(),
        "dependency_end": window.context_dates[-1].replace(hour=15).isoformat(),
        "label_end": window.context_dates[-1].replace(hour=15).isoformat(),
        "label_mature_at": teacher.get("available_at", window.available_at.isoformat()),
        "available_at": teacher.get("available_at", window.available_at.isoformat()),
        "known_target_count": known, "unknown_target_value": None,
    }
    return teacher


def _teacher_signature(teachers: dict) -> dict:
    return {name: {"schema": teacher.get("schema"), "schema_sha256": teacher.get("schema_sha256"),
                   **{kind + "_names": teacher.get(kind + "_names", []) for kind in ("continuous", "event")},
                   "structure_dim": len(teacher.get("structure_mask", []))}
            for name, teacher in teachers.items()}


def _coverage(samples_by_split: dict) -> dict:
    result = {"by_split": {}, "overall": {}}
    for split, samples in samples_by_split.items():
        counts = {}
        for sample in samples:
            for name, teacher in sample["teachers"].items():
                target = counts.setdefault(name, {})
                overall = result["overall"].setdefault(name, {})
                for kind in ("continuous", "event", "structure"):
                    mask = teacher.get(kind + "_mask", [])
                    for counter in (target, overall):
                        counter[kind + "_known"] = counter.get(kind + "_known", 0) + sum(mask)
                        counter[kind + "_total"] = counter.get(kind + "_total", 0) + len(mask)
        result["by_split"][split] = counts
    for counter in result["overall"].values():
        counter["status"] = ("valid_supervision_present" if sum(counter.get(kind + "_known", 0)
                             for kind in ("continuous", "event", "structure")) else "no_valid_supervision")
    return result


def _candidate_priority(candidates: list, maximum: int | None) -> list:
    if maximum is None or maximum >= len(candidates):
        return candidates
    chosen = list(dict.fromkeys(np.linspace(0, len(candidates) - 1, maximum).round().astype(int).tolist()))
    chosen_set = set(chosen)
    return [candidates[i] for i in chosen] + [item for i, item in enumerate(candidates) if i not in chosen_set]


def _write_bytes(config, path: Path, content: bytes) -> None:
    path = config.output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if config.output_path(path) != path:
        raise ValueError("output path changed during dataset write")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".dataset-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_dataset(config, *, symbols: list[str], start: str, end: str, label_through: str,
                  output_dir: str | Path, stride: int = 5, max_per_split: int | None = None) -> dict:
    """Generate actual teacher labels and realized future targets, explicitly.

    Origin dates are inclusive; ``label_through`` is the final date whose
    realized labels may be read. Late or crossing labels and incomplete windows
    are rejected with recorded reasons. Caps sample dates evenly within a split
    before deterministic fallback attempts, rather than taking the first dates.
    """
    start, end, label_through = (_day(value, field) for value, field in
                                 ((start, "start"), (end, "end"), (label_through, "label_through")))
    if not start <= end <= label_through:
        raise ValueError("require start <= end <= label_through")
    if type(stride) is not int or stride < 1 or (max_per_split is not None and
                                               (type(max_per_split) is not int or max_per_split < 1)):
        raise ValueError("stride and max_per_split must be positive integers")
    symbols = sorted({_symbol(item) for item in symbols})
    if not symbols:
        raise ValueError("at least one explicit A-share symbol is required")
    if config.values["ict"]["enabled"] is not True:
        raise ValueError("dual-teacher dataset generation requires ict.enabled=true")
    output = config.output_path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"dataset output must be new or empty: {output}")
    source = KhquantSource(config.source_app, config.raw_db, config.state_root,
                           calendar_path=config.calendar_path)
    calendar_days = source.calendar_dates()
    calendar_identity, source_identity = source.calendar_identity(), source.source_identity()
    calendar = _sessions(calendar_days)
    if start < calendar_days[0] or label_through > calendar_days[-1]:
        raise ValueError("requested origins/labels exceed the independently verified calendar")
    origin_positions = [i for i, day in enumerate(calendar_days) if start <= day <= end][::stride]
    if not origin_positions:
        raise ValueError("no trading-session origins in requested date range")
    read_start = calendar_days[max(0, origin_positions[0] - config.context_length + 1)]
    frames, snapshots, frame_indices = {}, {}, {}
    for symbol in symbols:
        frame = source.load_bars(symbol, start=read_start, end=label_through, as_of=label_through)
        if frame.empty or "date" not in frame or "symbol" not in frame:
            raise ValueError(f"source returned no dated bars for {symbol}")
        sessions = _sessions(frame["date"])
        if (frame["symbol"].astype(str) != symbol).any() or not sessions.isin(calendar).all():
            raise ValueError("source bars violate requested symbol or verified calendar")
        if sessions[-1].date().isoformat() > label_through:
            raise ValueError("source returned bars after label_through")
        frame = frame.copy(deep=True).reset_index(drop=True)
        if "available_at" in frame and availability_times(frame["available_at"]).isna().any():
            raise ValueError("source contains unknown available_at")
        frames[symbol] = frame
        frame_indices[symbol] = sessions
        snapshots[symbol] = {"sha256": _hash(_raw_records(frame)), "row_count": len(frame),
                             "first_session": sessions[0].date().isoformat(),
                             "last_session": sessions[-1].date().isoformat(),
                             "adapter_data_version": frame.attrs.get("data_version")}
    snapshot = _SnapshotSource(source, frames, calendar_days, calendar_identity,
                               source_identity, config.context_length)
    preview = MultiTeacherPreview(snapshot, config.ict_vendor_root, context_bars=config.context_length,
                                  swing_width=config.values["ict"]["swing_width"], include_ict=True)
    from token_distillation.teachers.czsc_teacher import CzscTeacher
    native_czsc = CzscTeacher(snapshot, context_bars=config.context_length)
    splits = _splits(config)
    threshold = float(config.values["data"].get("direction_threshold", 0.005))
    if not np.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError("direction_threshold must be finite and in [0,1)")
    candidate_groups = {name: [] for name in SPLIT_NAMES}
    outside_splits = 0
    for pos in origin_positions:
        origin = calendar[pos].replace(hour=15).to_pydatetime()
        split = next((item.name for item in splits.ranges if item.start <= origin <= item.end), None)
        if split is None:
            outside_splits += len(symbols)
        else:
            candidate_groups[split].extend((pos, symbol) for symbol in symbols)
    samples_by_split, skips, attempted = {name: [] for name in SPLIT_NAMES}, Counter(), Counter()
    rejection_examples = []
    for split in SPLIT_NAMES:
        for pos, symbol in _candidate_priority(candidate_groups[split], max_per_split):
            if max_per_split is not None and len(samples_by_split[split]) >= max_per_split:
                break
            attempted[split] += 1
            origin = calendar[pos].replace(hour=15)
            try:
                if pos + 1 < config.context_length:
                    raise ValueError("insufficient_context_in_verified_calendar")
                expected = calendar[pos - config.context_length + 1:pos + config.horizon + 1]
                raw = frames[symbol]
                indexed = frame_indices[symbol]
                local = raw.loc[indexed.isin(expected)].copy()
                local.attrs = dict(raw.attrs)
                if (local.attrs.get("price_basis") in {"unadjusted", "mixed_legacy_adjustment_unverified"}
                        and "source" in local and local["source"].isin(VERIFIED_SOURCES).all()):
                    local.attrs["price_basis"] = "unadjusted"
                quality = validate_raw_quality(local, verified_sources=VERIFIED_SOURCES)
                frame = local.copy()
                frame.index = _sessions(frame["date"])
                if "available_at" in frame:
                    frame["available_at"] = list(availability_times(frame["available_at"]).dt.tz_convert("Asia/Shanghai"))
                window = build_window(frame, symbol=symbol, as_of=origin,
                                      context_length=config.context_length, horizon=config.horizon,
                                      calendar_dates=calendar)
                decision = assign_window(window, splits)
                if not decision.eligible:
                    raise ValueError(";".join(decision.reasons))
                cutoff = pd.Timestamp(label_through, tz="Asia/Shanghai") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                if window.label_mature_at > cutoff:
                    raise ValueError("label_not_mature_at_label_through")
            except (ValueError, KeyError) as exc:
                reason = str(exc)
                skips[reason] += 1
                if len(rejection_examples) < 50:
                    rejection_examples.append({"symbol": symbol, "as_of": origin.isoformat(), "split": split, "reason": reason})
                continue
            raw_context = local.loc[_sessions(local["date"]) <= origin.normalize()]
            raw_future = local.loc[_sessions(local["date"]) > origin.normalize()]
            context_sha = _hash(_raw_records(raw_context))
            snapshot.served_hashes.clear()
            # Teacher/code/provenance errors abort generation; they are never
            # disguised as a harmless skipped market-data window.
            teachers = preview.prepare(symbol, origin.isoformat())["teachers"]
            teachers["czsc"] = native_czsc.prepare(symbol, origin.isoformat())
            if not snapshot.served_hashes or any(item != context_sha for item in snapshot.served_hashes):
                raise ValueError("teacher/student frozen raw context hashes differ")
            teachers = {name: _teacher_contract(teacher, window, context_sha) for name, teacher in teachers.items()}
            if not any(t["dataset_label_contract"]["known_target_count"] for t in teachers.values()):
                skips["all_teacher_targets_masked"] += 1
                continue
            normalized = normalize_window(window.context, window.future,
                epsilon=config.values["data"]["normalization_epsilon"], clip=config.values["data"]["normalization_clip"])
            realized_return = float(window.future[-1, 3] / window.context[-1, 3] - 1)
            sample = {"schema": SAMPLE_SCHEMA, "split": split, "symbol": symbol,
                "as_of": window.as_of.isoformat(), "available_at": window.available_at.isoformat(),
                "dependency_start": window.dependency_start.isoformat(),
                "label_end": window.label_end.isoformat(), "label_mature_at": window.label_mature_at.isoformat(),
                "context_dates": [stamp.isoformat() for stamp in window.context_dates],
                "future_dates": [stamp.isoformat() for stamp in window.future_dates],
                "raw_context": window.context.tolist(), "raw_future": window.future.tolist(),
                "normalized_context": normalized.history.tolist(), "normalized_future": normalized.future.tolist(),
                "time_context": _time_features(window.context_dates), "time_future": _time_features(window.future_dates),
                "normalization": jsonable(normalized.stats), "teachers": teachers,
                "realized_return": realized_return,
                "direction_class": 2 if realized_return > threshold else 0 if realized_return < -threshold else 1,
                "direction_threshold": threshold, "quality": quality,
                "provenance": {"raw_context_sha256": context_sha,
                    "raw_future_sha256": _hash(_raw_records(raw_future)),
                    "symbol_snapshot_sha256": snapshots[symbol]["sha256"],
                    "context_source_rows": _raw_records(raw_context), "future_source_rows": _raw_records(raw_future),
                    "source_code_sha256": source_identity.get("code_sha256"),
                    "calendar_dates_sha256": calendar_identity.get("dates_hash"),
                    "availability_basis": "explicit_source_available_at" if "available_at" in raw_context
                        else "assumed_shanghai_session_close_15h_not_ingestion_evidence"}}
            sample["sample_sha256"] = _hash(sample)
            samples_by_split[split].append(sample)
    if not any(samples_by_split.values()):
        raise ValueError(f"no eligible samples; rejected={dict(skips)}")
    if source.source_identity() != source_identity or source.calendar_identity() != calendar_identity:
        raise ValueError("read-only source code or verified calendar changed during dataset generation")
    shards = {}
    for split in SPLIT_NAMES:
        samples = sorted(samples_by_split[split], key=lambda sample: (sample["as_of"], sample["symbol"]))
        content = b"".join(_canonical(sample) + b"\n" for sample in samples)
        _write_bytes(config, output / f"{split}.jsonl", content)
        shards[split] = {"path": f"{split}.jsonl", "sha256": hashlib.sha256(content).hexdigest(),
                          "row_count": len(samples), "first_as_of": samples[0]["as_of"] if samples else None,
                          "last_as_of": samples[-1]["as_of"] if samples else None}
    signature = _teacher_signature(next(sample for samples in samples_by_split.values() for sample in samples)["teachers"])
    if any(_teacher_signature(sample["teachers"]) != signature for samples in samples_by_split.values() for sample in samples):
        raise ValueError("teacher schemas changed within the generated dataset")
    manifest = {"schema": SCHEMA, "sample_schema": SAMPLE_SCHEMA, "universe": "sh_sz_a_v1", "frequency": "1d",
        "symbols": symbols, "start": start, "end": end, "label_through": label_through,
        "context_length": config.context_length, "horizon": config.horizon, "features": list(OHLCVA_COLUMNS),
        "time_feature_order": ["minute", "hour", "weekday", "day", "month"], "stride": stride,
        "max_per_split": max_per_split, "direction_threshold": threshold,
        "config_sha256": _hash(config.values), "config": deepcopy(config.values),
        "source_identity": source_identity, "calendar_identity": calendar_identity,
        "symbol_snapshots": snapshots, "generator_sha256": sha256_file(Path(__file__)),
        "splits": shards, "split_ranges": {item.name: [item.start.isoformat(), item.end.isoformat()] for item in splits.ranges},
        "split_status": config.values["splits"].get("status", "draft_not_frozen"),
        "holdout_claim": config.values["splits"].get("claim", "retrospective_development_only"),
        "population_sha256": _hash([sample["sample_sha256"] for split in SPLIT_NAMES
                                      for sample in sorted(samples_by_split[split], key=lambda s: (s["as_of"], s["symbol"]))]),
        "total_samples": sum(item["row_count"] for item in shards.values()),
        "teacher_schemas": signature, "teacher_coverage": _coverage(samples_by_split),
        "admission": {"candidate_counts": {name: len(candidate_groups[name]) for name in SPLIT_NAMES},
            "attempted_counts": {name: attempted[name] for name in SPLIT_NAMES},
            "skipped_counts": dict(sorted(skips.items())), "origins_outside_splits": outside_splits,
            "rejection_examples": rejection_examples},
        "causal_contract": {"source_read_only": True, "teachers_share_frozen_student_context": True,
            "context_only_normalization": True, "split_end_label_maturity_purge": True,
            "unknown_is_masked_null": True, "tokenizer_loaded": False, "training_started": False,
            "formal_qualified": False, "survivorship_and_corporate_actions_audited": False}}
    manifest["manifest_sha256"] = _hash(manifest)
    write_json(config, output / "manifest.json", manifest)
    return manifest


def load_dataset(config, manifest_path: str | Path, *, compact=None) -> tuple[dict, list[dict]]:
    """Read only workspace-local, content-verified immutable dataset shards."""
    path = config.output_path(manifest_path)
    if path.is_dir():
        path = config.output_path(path / "manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("universe") != "sh_sz_a_v1":
        raise ValueError("unsupported dataset manifest")
    claimed = manifest.get("manifest_sha256")
    if claimed != _hash({key: value for key, value in manifest.items() if key != "manifest_sha256"}):
        raise ValueError("dataset manifest SHA256 mismatch")
    if set(manifest.get("splits", {})) != set(SPLIT_NAMES):
        raise ValueError("dataset must define all five explicit splits")
    if manifest.get("config_sha256") != _hash(manifest["config"]):
        raise ValueError("dataset configuration SHA256 mismatch")
    if manifest.get("features") != list(OHLCVA_COLUMNS):
        raise ValueError("dataset feature ordering mismatch")
    if compact is None: compact = config.values["training"].get("compact_dataset", False)
    from .compact import compact_sample
    samples = []
    population_hashes = []
    identities = set()
    signature = manifest.get("teacher_schemas")
    splits = TimeSplits(**{name: SplitRange(name, *(datetime.fromisoformat(value)
                                      for value in manifest["split_ranges"][name])) for name in SPLIT_NAMES})
    from .splits import assign_sample
    for split in SPLIT_NAMES:
        shard = manifest["splits"][split]
        relative = Path(shard["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("dataset shard path must be relative without traversal")
        shard_path = config.output_path(path.parent / relative)
        if not shard_path.is_relative_to(path.parent) or sha256_file(shard_path) != shard["sha256"]:
            raise ValueError("dataset shard SHA256 mismatch or escaped dataset directory")
        first = last = previous = None
        count = 0
        def stream():
            with shard_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip(): yield json.loads(line)
        for sample in stream():
            count += 1
            key = (sample["as_of"], sample["symbol"])
            if previous is not None and key < previous:
                raise ValueError("dataset shard is not chronologically ordered")
            previous = key
            if first is None: first = sample["as_of"]
            last = sample["as_of"]
            if sample.get("sample_sha256") != _hash({k: v for k, v in sample.items() if k != "sample_sha256"}):
                raise ValueError("dataset sample SHA256 mismatch")
            if sample.get("schema") != SAMPLE_SCHEMA or sample.get("split") != split:
                raise ValueError("sample schema or split mismatch")
            identity = sample["symbol"], sample["as_of"]
            if identity in identities:
                raise ValueError("duplicate symbol/origin dataset sample")
            identities.add(identity)
            if _symbol(sample["symbol"]) != sample["symbol"] or sample["symbol"] not in manifest["symbols"]:
                raise ValueError("sample symbol not in explicit A-share population")
            decision = assign_sample(as_of=datetime.fromisoformat(sample["as_of"]),
                available_at=datetime.fromisoformat(sample["available_at"]),
                label_end=datetime.fromisoformat(sample["label_end"]),
                label_mature_at=datetime.fromisoformat(sample["label_mature_at"]), splits=splits)
            if not decision.eligible or decision.split != split:
                raise ValueError("dataset sample violates split or maturity contract")
            for field, shape in (("normalized_context", (manifest["context_length"], 6)),
                                 ("normalized_future", (manifest["horizon"], 6)),
                                 ("time_context", (manifest["context_length"], 5)),
                                 ("time_future", (manifest["horizon"], 5))):
                value = np.asarray(sample[field], dtype=np.float64)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"dataset sample {field} shape or finite-value failure")
            if sample["direction_class"] not in (0, 1, 2):
                raise ValueError("invalid direction class")
            _validate_loaded_sample(sample, manifest)
            current_signature = _teacher_signature(sample["teachers"])
            if signature is None:
                signature = current_signature
            if current_signature != signature:
                raise ValueError("inconsistent teacher schemas within dataset")
            population_hashes.append(sample["sample_sha256"])
            samples.append(compact_sample(sample) if compact else sample)
        if count != shard["row_count"]:
            raise ValueError("dataset shard row count mismatch")
        if count and (first != shard["first_as_of"] or last != shard["last_as_of"]):
            raise ValueError("dataset shard origin bounds mismatch")
    if len(samples) != manifest["total_samples"] or _hash(population_hashes) != manifest["population_sha256"]:
        raise ValueError("dataset population count or SHA256 mismatch")
    return manifest, samples


def _validate_loaded_sample(sample: dict, manifest: dict) -> None:
    """Recheck causal semantics even when bytes have valid updated hashes."""
    if set(sample["teachers"]) != {"w32", "ict", "czsc"}:
        raise ValueError("dataset lacks a required CZSC/W32/ICT teacher")
    context_rows = sample["provenance"]["context_source_rows"]
    future_rows = sample["provenance"]["future_source_rows"]
    if (_hash(context_rows) != sample["provenance"]["raw_context_sha256"] or
            _hash(future_rows) != sample["provenance"]["raw_future_sha256"]):
        raise ValueError("sample source row SHA256 mismatch")
    frame = pd.DataFrame(context_rows + future_rows)
    frame.index = _sessions(frame["date"])
    frame["available_at"] = [parse_timestamp(stamp).tz_convert("Asia/Shanghai") for stamp in frame["available_at"]]
    frame.attrs.update(price_basis="unadjusted", volume_unit="shares", amount_unit="CNY")
    validate_raw_quality(frame, verified_sources=VERIFIED_SOURCES)
    window = build_window(frame, symbol=sample["symbol"], as_of=parse_timestamp(sample["as_of"]).tz_convert("Asia/Shanghai"),
                          context_length=manifest["context_length"], horizon=manifest["horizon"],
                          calendar_dates=frame.index)
    expected = {"available_at": window.available_at.isoformat(), "dependency_start": window.dependency_start.isoformat(),
                "label_end": window.label_end.isoformat(), "label_mature_at": window.label_mature_at.isoformat(),
                "context_dates": [stamp.isoformat() for stamp in window.context_dates],
                "future_dates": [stamp.isoformat() for stamp in window.future_dates],
                "time_context": _time_features(window.context_dates), "time_future": _time_features(window.future_dates),
                "raw_context": window.context.tolist(), "raw_future": window.future.tolist()}
    if any(sample[field] != value for field, value in expected.items()):
        raise ValueError("sample raw values or causal timestamp provenance mismatch")
    normalization = normalize_window(window.context, window.future,
        epsilon=manifest["config"]["data"]["normalization_epsilon"],
        clip=manifest["config"]["data"]["normalization_clip"])
    if (sample["normalization"] != jsonable(normalization.stats) or
            sample["normalized_context"] != normalization.history.tolist() or
            sample["normalized_future"] != normalization.future.tolist()):
        raise ValueError("sample normalization differs from context-only statistics")
    realized = float(window.future[-1, 3] / window.context[-1, 3] - 1)
    threshold = manifest["direction_threshold"]
    direction = 2 if realized > threshold else 0 if realized < -threshold else 1
    if (sample["direction_threshold"] != threshold or sample["realized_return"] != realized or
            sample["direction_class"] != direction):
        raise ValueError("sample realized direction target mismatch")
    cutoff = pd.Timestamp(manifest["label_through"], tz="Asia/Shanghai") + pd.Timedelta(days=1)
    if window.label_mature_at >= cutoff:
        raise ValueError("sample label matures after dataset label_through")
    for teacher in sample["teachers"].values():
        checked = _teacher_contract(teacher, window, sample["provenance"]["raw_context_sha256"])
        if checked != teacher:
            raise ValueError("teacher unknown masks or shared context contract mismatch")
