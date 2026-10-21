"""Synthetic-only future-label gap admission; no source reads or training."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from token_distillation.config import load_config, SPLIT_NAMES
from token_distillation.data import dataset as dataset_module
from token_distillation.data import filtering


def sample(symbol="600000.SH", *, first_open=100., second_open=100., split="train"):
    teacher = {"continuous_targets": [0.], "continuous_mask": [True],
               "event_targets": [None], "event_mask": [False]}
    row = {"symbol": symbol, "split": split, "as_of": "2024-01-02T15:00:00+08:00",
        "label_mature_at": "2024-01-04T15:00:00+08:00",
        "future_dates": ["2024-01-03T00:00:00+08:00", "2024-01-04T00:00:00+08:00"],
        "raw_context": [[100., 101., 99., 100., 1000., 100000.]],
        "raw_future": [[first_open, first_open + 1, first_open - 1, first_open, 1000., 1e5],
                       [second_open, second_open + 1, second_open - 1, second_open, 1000., 1e5]],
        "teachers": {name: deepcopy(teacher) for name in ("w32", "ict", "czsc")}}
    row["sample_sha256"] = dataset_module._hash(row)
    return row


@pytest.mark.parametrize("symbol,opening,reject", [
    ("600000.SH", 116., True), ("000001.SZ", 84., True),
    ("300750.SZ", 120., False), ("688981.SH", 126., True),
    ("600000.SH", 115., False), ("688981.SH", 125., False),
])
def test_frozen_board_cutoffs_and_strict_greater_than(symbol, opening, reject):
    assert bool(filtering.future_label_gap_findings(sample(symbol, first_open=opening, second_open=opening))) is reject


def test_every_future_bar_uses_previous_future_close_and_origin_last_close():
    row = sample(first_open=110., second_open=130.)
    hits = filtering.future_label_gap_findings(row)
    assert len(hits) == 1 and hits[0]["future_bar_index"] == 1
    assert hits[0]["previous_close"] == 110.
    row["raw_context"].insert(0, [1000., 1001., 999., 1000., 1e3, 1e6])
    assert filtering.future_label_gap_findings(row) == hits  # context gaps stay untouched


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    config = replace(load_config(), root=tmp_path, state_root=tmp_path / "artifacts/state")
    parent_path = tmp_path / "data/parent/manifest.json"
    parent_path.parent.mkdir(parents=True)
    rows = [sample(split=name) for name in SPLIT_NAMES]
    rows.append(sample("000001.SZ", first_open=30., second_open=30., split="tune"))
    parent = {"schema": dataset_module.SCHEMA, "config": deepcopy(config.values),
        "source_identity": {"code_sha256": "a" * 64, "files": {filtering.POLICY_SOURCE_FILE: "b" * 64}},
        "symbol_snapshots": {"synthetic": {"sha256": "c" * 64}},
        "population_sha256": dataset_module._hash([r["sample_sha256"] for r in rows]),
        "admission": {"candidate_counts": {name: 2 for name in SPLIT_NAMES}},
        "splits": {}, "total_samples": len(rows)}
    for name in SPLIT_NAMES:
        records = [row for row in rows if row["split"] == name]
        # Noncanonical spaces deliberately verify literal byte preservation.
        content = b"".join(json.dumps(row, sort_keys=True).encode() + b"\r\n" for row in records)
        (parent_path.parent / (name + ".jsonl")).write_bytes(content)
        parent["splits"][name] = {"path": name + ".jsonl", "sha256": hashlib.sha256(content).hexdigest()}
    parent["manifest_sha256"] = dataset_module._hash(parent)
    parent_path.write_text(json.dumps(parent))
    monkeypatch.setattr(filtering, "load_dataset", lambda *_: (deepcopy(parent), deepcopy(rows)))
    return config, parent_path, parent, rows


def test_derivation_preserves_parent_and_admitted_sample_bytes(frozen):
    config, parent_path, parent, rows = frozen
    before = {p.name: p.read_bytes() for p in parent_path.parent.iterdir()}
    result = filtering.derive_dataset(config, dataset=parent_path, output_dir=Path("data/derived"))
    assert result["total_samples"] == 5
    assert result["config"] == parent["config"]
    assert result["source_identity"] == parent["source_identity"]
    assert result["symbol_snapshots"] == parent["symbol_snapshots"]
    assert before == {p.name: p.read_bytes() for p in parent_path.parent.iterdir()}
    for name in SPLIT_NAMES:
        admitted = [line for line in before[name + ".jsonl"].splitlines(keepends=True)
                    if json.loads(line)["symbol"] == "600000.SH"]
        assert (config.root / "data/derived" / (name + ".jsonl")).read_bytes() == b"".join(admitted)
    lineage = result["derivation"]
    assert lineage["parent_manifest_sha256"] == hashlib.sha256(before["manifest.json"]).hexdigest()
    assert lineage["rejected_counts_by_split"] == {name: int(name == "tune") for name in SPLIT_NAMES}
    assert lineage["information_available_after_label_mature"]
    assert lineage["policy"]["known_at_prediction_origin"] is False
    assert lineage["no_price_adjustment"] and lineage["adjustment_factors_read"] is False
    assert lineage["admitted_sample_bytes_unchanged"]
    assert result["admission"]["parent_generation_admission"] == parent["admission"]
    assert result["manifest_sha256"] == dataset_module._hash({k: v for k, v in result.items() if k != "manifest_sha256"})


def test_derived_output_must_be_new_and_inside_project(frozen):
    config, parent_path, _, _ = frozen
    output = config.root / "data/existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="must be new"):
        filtering.derive_dataset(config, dataset=parent_path, output_dir=output)
    with pytest.raises(ValueError, match="inside token_distillation"):
        filtering.derive_dataset(config, dataset=parent_path, output_dir=config.root.parent / "escaped")


def test_changed_parent_shard_is_rejected_before_output(frozen):
    config, parent_path, _, _ = frozen
    (parent_path.parent / "train.jsonl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Parent shard changed"):
        filtering.derive_dataset(config, dataset=parent_path, output_dir=Path("data/derived"))
    assert not (config.root / "data/derived").exists()
