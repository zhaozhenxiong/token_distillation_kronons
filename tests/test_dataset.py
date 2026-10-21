"""Synthetic-only end-to-end dataset admission, causality and integrity tests."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from token_distillation.config import load_config
from token_distillation.data import dataset
from token_distillation.data.splits import SPLIT_NAMES


class Source:
    def __init__(self, frames, calendar):
        self.frames = frames
        self.calendar = calendar
        self.reads = []

    def load_bars(self, symbol, start=None, end=None, as_of=None):
        self.reads.append((symbol, start, end, as_of))
        frame = self.frames[symbol]
        dates = frame.date.dt.strftime("%Y-%m-%d")
        result = frame.loc[(dates >= start) & (dates <= end)].copy().reset_index(drop=True)
        result.attrs = dict(frame.attrs)
        result.attrs["data_version"] = hashlib.sha256(result.to_csv(index=False).encode()).hexdigest()
        return result

    def calendar_dates(self, start=None, end=None):
        return [day for day in self.calendar if (start is None or day >= start) and (end is None or day <= end)]

    def calendar_identity(self):
        digest = hashlib.sha256(str(self.calendar).encode()).hexdigest()
        return {"run_id": "synthetic", "source": "synthetic-independent-calendar",
                "independent_from_market_bars": True, "file_sha256": digest, "dates_hash": digest,
                "first_session": self.calendar[0], "last_session": self.calendar[-1],
                "source_artifacts": [{"path": "synthetic", "sha256": digest}]}

    def source_identity(self):
        return {"scope": "sh_sz_a_v1", "code_sha256": "a" * 64, "source": "synthetic-only"}


class Teacher:
    def __init__(self, source, context_bars=16, **kwargs):
        self.source = source
        self.context = context_bars

    def prepare(self, symbol, as_of):
        raw = self.source.load_bars(symbol, as_of=as_of)
        first, last = raw.date.iloc[0].date().isoformat(), raw.date.iloc[-1].date().isoformat()
        return {"schema": "synthetic-teacher", "symbol": symbol, "as_of": as_of,
                "context_start": first, "context_end": last, "context_rows": len(raw),
                "available_at": last + "T15:00:00+08:00", "dependency_end": last + "T15:00:00+08:00",
                "continuous_names": ["last_close", "unknown"],
                "continuous_targets": [float(raw.close.iloc[-1]), 123.0], "continuous_mask": [True, False],
                "event_names": ["known_no_event"], "event_targets": [0.0], "event_mask": [True],
                "quality": {"eligible": True}}


class Preview:
    def __init__(self, source, vendor, context_bars=16, **kwargs):
        self.teacher = Teacher(source, context_bars)

    def prepare(self, symbol, as_of):
        return {"teachers": {"w32": self.teacher.prepare(symbol, as_of),
                             "ict": self.teacher.prepare(symbol, as_of)}}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    config = load_config()
    values = deepcopy(config.values)
    dates = pd.bdate_range("2024-01-02", periods=150)
    calendar = dates.strftime("%Y-%m-%d").tolist()
    values["data"].update(context_length=16, horizon=3)
    for i, name in enumerate(SPLIT_NAMES):
        values["splits"][name] = [calendar[20 + i * 25], calendar[44 + i * 25]]
    config = replace(config, root=tmp_path, state_root=tmp_path / "artifacts/state", values=values)
    frames = {}
    for symbol, offset in (("600000.SH", 0), ("000001.SZ", 5)):
        prices = np.arange(len(dates), dtype=float) * .1 + 10 + offset
        frame = pd.DataFrame({"date": dates, "symbol": symbol, "open": prices,
            "high": prices + 1, "low": prices - 1, "close": prices + .2,
            "volume": prices * 100, "amount": prices * 1000,
            "source": "tushare", "has_trade_price": 1})
        frame.attrs.update(price_basis="unadjusted", volume_unit="shares", amount_unit="CNY")
        frames[symbol] = frame
    source = Source(frames, calendar)
    monkeypatch.setattr(dataset, "KhquantSource", lambda *a, **kw: source)
    monkeypatch.setattr(dataset, "MultiTeacherPreview", Preview)
    import token_distillation.teachers.czsc_teacher as native
    monkeypatch.setattr(native, "CzscTeacher", Teacher)
    return SimpleNamespace(config=config, source=source, calendar=calendar,
        params={"symbols": ["600000", "000001.SZ"], "start": calendar[20], "end": calendar[144],
                "label_through": calendar[-1], "stride": 1})


def test_dataset_is_deterministic_frozen_and_preserves_unknowns(fixture):
    f = fixture
    first = dataset.build_dataset(f.config, **f.params, output_dir="data/one", max_per_split=4)
    manifest, samples = dataset.load_dataset(f.config, "data/one")
    assert len(f.source.reads) == 2  # every teacher sees a snapshot; no database rereads
    second = dataset.build_dataset(f.config, **f.params, output_dir="data/two", max_per_split=4)
    assert first == second == manifest
    assert manifest["total_samples"] == 20
    assert set(manifest["splits"]) == set(SPLIT_NAMES)
    assert manifest["teacher_coverage"]["overall"]["w32"]["continuous_known"] == 20
    assert manifest["teacher_coverage"]["overall"]["ict"]["status"] == "valid_supervision_present"
    for split in SPLIT_NAMES:
        assert manifest["splits"][split]["row_count"] == 4
        left = f.config.root / "data/one" / f"{split}.jsonl"
        right = f.config.root / "data/two" / f"{split}.jsonl"
        assert left.read_bytes() == right.read_bytes()
    for sample in samples:
        assert sample["normalization"]["history_count"] == 16
        assert np.asarray(sample["normalized_context"]).shape == (16, 6)
        assert np.asarray(sample["normalized_future"]).shape == (3, 6)
        assert sample["as_of"] < sample["label_mature_at"]
        for teacher in sample["teachers"].values():
            assert teacher["continuous_targets"][1] is None
            assert teacher["event_targets"][0] == 0
            assert teacher["event_mask"][0] is True
            assert teacher["shared_input_sha256"] == sample["provenance"]["raw_context_sha256"]
    # Uniform cap reaches far into each split, rather than its first four days.
    assert manifest["splits"]["train"]["last_as_of"][:10] >= f.calendar[35]
    with pytest.raises(FileExistsError):
        dataset.build_dataset(f.config, **f.params, output_dir="data/one", max_per_split=4)


def test_split_crossing_future_labels_and_late_bars_are_purged(fixture):
    f = fixture
    # Late explicit source availability also prevents a historical origin from
    # consuming that row; complete-window admission records why it is skipped.
    for frame in f.source.frames.values():
        frame["available_at"] = frame.date + pd.Timedelta(hours=15)
        frame.loc[44, "available_at"] += pd.Timedelta(days=5)
    result = dataset.build_dataset(f.config, **f.params, output_dir="data/purged")
    _, samples = dataset.load_dataset(f.config, "data/purged/manifest.json")
    reasons = result["admission"]["skipped_counts"]
    assert any("label_crosses_split_end" in reason for reason in reasons)
    assert any("not available at as_of" in reason for reason in reasons)
    for sample in samples:
        assert sample["label_mature_at"] <= result["split_ranges"][sample["split"]][1]


def test_future_prices_cannot_change_context_normalization_or_teachers(fixture):
    f = fixture
    dataset.build_dataset(f.config, **f.params, output_dir="data/before", max_per_split=3)
    _, before = dataset.load_dataset(f.config, "data/before")
    chosen = next(sample for sample in before if sample["split"] == "train")
    altered_day = chosen["future_dates"][-1][:10]
    frame = f.source.frames[chosen["symbol"]]
    mask = frame.date.dt.strftime("%Y-%m-%d") == altered_day
    frame.loc[mask, ["open", "high", "low", "close"]] *= 3
    dataset.build_dataset(f.config, **f.params, output_dir="data/after", max_per_split=3)
    _, after = dataset.load_dataset(f.config, "data/after")
    changed = next(s for s in after if (s["symbol"], s["as_of"]) == (chosen["symbol"], chosen["as_of"]))
    assert changed["normalization"] == chosen["normalization"]
    assert changed["normalized_context"] == chosen["normalized_context"]
    assert changed["teachers"] == chosen["teachers"]
    assert changed["normalized_future"] != chosen["normalized_future"]
    assert changed["realized_return"] != chosen["realized_return"]


def test_dataset_load_rejects_modified_shards_and_path_escape(fixture):
    f = fixture
    dataset.build_dataset(f.config, **f.params, output_dir="data/immutable", max_per_split=1)
    shard = f.config.root / "data/immutable/train.jsonl"
    shard.write_bytes(shard.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        dataset.load_dataset(f.config, "data/immutable")
    with pytest.raises(ValueError, match="inside token_distillation"):
        dataset.build_dataset(f.config, **f.params, output_dir=f.config.root.parent / "outside")


def _rewrite_with_valid_hashes(root: Path, change):
    """Simulate internally rehashed but semantically invalid imported data."""
    manifest = json.loads((root / "manifest.json").read_text())
    records = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
    change(records)
    for sample in records:
        sample["sample_sha256"] = dataset._hash({key: value for key, value in sample.items() if key != "sample_sha256"})
    raw = b"".join(dataset._canonical(sample) + b"\n" for sample in records)
    (root / "train.jsonl").write_bytes(raw)
    manifest["splits"]["train"].update(sha256=hashlib.sha256(raw).hexdigest(), row_count=len(records),
        first_as_of=records[0]["as_of"], last_as_of=records[-1]["as_of"])
    population = [json.loads(line)["sample_sha256"] for name in SPLIT_NAMES
                  for line in (root / f"{name}.jsonl").read_text().splitlines()]
    manifest.update(total_samples=len(population), population_sha256=dataset._hash(population))
    manifest["manifest_sha256"] = dataset._hash({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    (root / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("mutation,reason", [
    (lambda rows: rows.append(deepcopy(rows[0])), "duplicate"),
    (lambda rows: rows[0]["teachers"]["w32"].update(as_of="2024-01-01T15:00:00+08:00"), "input mismatch"),
    (lambda rows: rows[0]["teachers"]["w32"].update(dependency_start="2020-01-01T15:00:00+08:00"), "before student context"),
    (lambda rows: rows[0]["teachers"]["w32"]["continuous_targets"].__setitem__(1, 0.0), "unknown masks"),
    (lambda rows: rows[0]["teachers"]["w32"]["continuous_names"].__setitem__(0, "different_feature"), "inconsistent teacher schemas"),
    (lambda rows: rows[0]["normalized_future"][0].__setitem__(0, 123.0), "normalization differs"),
])
def test_semantic_validation_rejects_rehashed_invalid_samples(fixture, mutation, reason):
    f = fixture
    dataset.build_dataset(f.config, **f.params, output_dir="data/adversarial", max_per_split=1)
    root = f.config.root / "data/adversarial"
    _rewrite_with_valid_hashes(root, mutation)
    with pytest.raises(ValueError, match=reason):
        dataset.load_dataset(f.config, root)


def test_schema_matches_generated_manifest(fixture):
    jsonschema = pytest.importorskip("jsonschema")
    f = fixture
    manifest = dataset.build_dataset(f.config, **f.params, output_dir="data/schema", max_per_split=1)
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/dataset_index.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(manifest, schema)
