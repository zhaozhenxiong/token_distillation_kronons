"""Explicit immutable label-quality derivation over already frozen datasets.

This is an ex-post research admission rule using future daily bars. It cannot
be applied as an at-origin trading filter. Suspected price discontinuities are
not confirmed corporate actions and no prices or existing teacher masks change.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from ..artifacts import sha256_file, write_json
from ..config import ProjectConfig, SPLIT_NAMES
from .dataset import _coverage, _hash, _symbol, _write_bytes, load_dataset


POLICY_SOURCE_FILE = "my_strategy/services/czsc_price_volume_features.py"
POLICY = {
    "schema": "frozen_w32_future_label_overnight_gap_policy_v1",
    "formula": "abs(future_open / previous_raw_close - 1) > board_cutoff",
    "cutoff_by_board": {"sh_main_60": 0.15, "sz_main_00": 0.15,
                        "chinext_30": 0.25, "star_68": 0.25},
    "comparison": "strict_greater_than; same_float_ratio_rule_as_frozen_W32_quality_policy",
    "scope": "future_label_bars_only; first_previous_close_is_origin_context_last_close",
    "meaning": "suspected_raw_price_discontinuity_or_possible_corporate_action; cause_unconfirmed",
    "known_at_prediction_origin": False,
    "price_adjustment": False,
}


def future_label_gap_findings(sample: dict) -> list[dict]:
    """Inspect future opens against preceding closes, including the origin."""
    symbol = _symbol(sample["symbol"])
    cutoff = 0.15 if ((symbol.startswith("60") and symbol.endswith(".SH"))
                       or (symbol.startswith("00") and symbol.endswith(".SZ"))) else 0.25
    previous_close = float(sample["raw_context"][-1][3])
    hits = []
    for index, (session, bar) in enumerate(zip(sample["future_dates"], sample["raw_future"])):
        opening, closing = float(bar[0]), float(bar[3])
        gap = opening / previous_close - 1.
        if abs(gap) > cutoff:
            hits.append({"future_bar_index": index, "session": session,
                "previous_close": previous_close, "open": opening,
                "overnight_open_to_previous_close_return": gap, "absolute_cutoff": cutoff})
        previous_close = closing
    return hits


def derive_dataset(config: ProjectConfig, *, dataset: Path, output_dir: Path) -> dict:
    """Write a new, fully linked dataset without changing admitted sample bytes."""
    parent_path = config.output_path(dataset)
    if parent_path.is_dir():
        parent_path = config.output_path(parent_path / "manifest.json")
    output = config.output_path(output_dir)
    if output.exists():
        raise FileExistsError(f"Derived dataset output must be new: {output}")
    parent_sha256 = sha256_file(parent_path)
    parent, samples = (load_dataset(config, parent_path, compact=False)
                       if config.values["training"].get("compact_dataset") else load_dataset(config, parent_path))
    groups = {name: [] for name in SPLIT_NAMES}
    rejected_counts = {name: 0 for name in SPLIT_NAMES}
    rejections = []
    for sample in samples:
        hits = future_label_gap_findings(sample)
        if hits:
            rejected_counts[sample["split"]] += 1
            rejections.append({"symbol": sample["symbol"], "as_of": sample["as_of"],
                "split": sample["split"], "sample_sha256": sample["sample_sha256"],
                "label_mature_at": sample["label_mature_at"],
                "information_available_at": sample["label_mature_at"],
                "reason": "suspected_raw_future_label_price_discontinuity; cause_unconfirmed",
                "hits": hits})
        else:
            groups[sample["split"]].append(sample)
    if not any(groups.values()):
        raise ValueError("Label-gap policy rejected every sample; no derived dataset written")
    # Preserve each admitted original JSONL line literally, including whitespace.
    # Revalidate the shard hash from the exact bytes used for this derivation.
    contents = {}
    for name in SPLIT_NAMES:
        source_path = config.output_path(parent_path.parent / parent["splits"][name]["path"])
        raw = source_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != parent["splits"][name]["sha256"]:
            raise ValueError("Parent shard changed during label-quality derivation")
        admitted = {sample["sample_sha256"] for sample in groups[name]}
        lines = [line for line in raw.splitlines(keepends=True) if line.strip()
                 and json.loads(line)["sample_sha256"] in admitted]
        if len(lines) != len(groups[name]):
            raise ValueError("Admitted sample bytes differ from the verified parent population")
        contents[name] = b"".join(lines)
    if sha256_file(parent_path) != parent_sha256:
        raise ValueError("Parent manifest changed during label-quality derivation")
    policy = {**deepcopy(POLICY), "source_policy_file": POLICY_SOURCE_FILE,
        "source_policy_file_sha256": parent["source_identity"].get("files", {}).get(POLICY_SOURCE_FILE),
        "parent_source_code_sha256": parent["source_identity"].get("code_sha256")}
    manifest = deepcopy(parent)
    manifest.pop("manifest_sha256")
    manifest["derivation"] = {
        "schema": "frozen_label_gap_dataset_derivation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_manifest": str(parent_path), "parent_manifest_sha256": parent_sha256,
        "parent_population_sha256": parent["population_sha256"],
        "policy": policy, "policy_sha256": _hash(policy),
        "derivation_code_sha256": sha256_file(Path(__file__)),
        "rejected_counts_by_split": rejected_counts, "rejections": rejections,
        "information_available_after_label_mature": True,
        "information_timing": "conservative_full_label_maturity; unavailable_at_prediction_origin",
        "no_price_adjustment": True, "adjustment_factors_read": False,
        "context_rows_or_teacher_masks_changed": False, "admitted_sample_bytes_unchanged": True,
        "cap_refill_performed": False,
        "expost_quality_selection_bias": "Research estimand is conditional on future-label gap admission. This ex-post selection is unavailable at the prediction origin and is not a deployable trading filter.",
        "model_results_used_for_filtering": False,
    }
    manifest["admission"] = {**deepcopy(parent["admission"]),
        "parent_generation_admission": deepcopy(parent["admission"]),
        "parent_generation_counters_preserved": True,
        "label_gap_rejected_counts_by_split": rejected_counts,
        "derived_admitted_counts_by_split": {name: len(groups[name]) for name in SPLIT_NAMES}}
    manifest["splits"] = {name: {"path": name + ".jsonl",
        "sha256": hashlib.sha256(contents[name]).hexdigest(), "row_count": len(groups[name]),
        "first_as_of": groups[name][0]["as_of"] if groups[name] else None,
        "last_as_of": groups[name][-1]["as_of"] if groups[name] else None}
        for name in SPLIT_NAMES}
    manifest["total_samples"] = sum(len(groups[name]) for name in SPLIT_NAMES)
    manifest["population_sha256"] = _hash([sample["sample_sha256"]
        for name in SPLIT_NAMES for sample in groups[name]])
    manifest["teacher_coverage"] = _coverage(groups)
    manifest["manifest_sha256"] = _hash(manifest)
    output.mkdir(parents=True, exist_ok=False)
    for name in SPLIT_NAMES:
        _write_bytes(config, output / (name + ".jsonl"), contents[name])
    write_json(config, output / "manifest.json", manifest)
    return manifest
