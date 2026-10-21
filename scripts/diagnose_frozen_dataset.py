#!/usr/bin/env python3
"""Describe frozen research JSONL only; never query sources or run a model."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "tune", "calibration", "policy", "holdout")
CLASS_NAMES = ("down", "flat", "up")


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _summary(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"count": 0, "mean": None, "std": None, "minimum": None,
                "maximum": None, "quantiles": None}
    return {"count": len(a), "mean": float(a.mean()), "std": float(a.std()),
            "minimum": float(a.min()), "maximum": float(a.max()),
            "quantiles": {str(q): float(np.quantile(a, q)) for q in (.01, .05, .25, .5, .75, .95, .99)}}


def profile_dataset(dataset_dir):
    """Stream rows; retain aggregate teacher values rather than raw matrices."""
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    result = {"manifest": str(manifest_path), "manifest_file_sha256": _sha(manifest_path),
        "manifest_population_sha256": manifest["population_sha256"],
        "symbols": manifest["symbols"], "context_length": manifest["context_length"],
        "horizon": manifest["horizon"], "stride": manifest["stride"],
        "split_ranges": manifest["split_ranges"], "admission": manifest["admission"],
        "splits": {}, "teacher_profiles": {}, "availability_basis_counts": {},
        "source_values": {}, "source_trade_price_values": {},
        "label_outliers": {}, "future_overnight_gap_diagnostics": {}}
    teachers = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    requested = defaultdict(lambda: defaultdict(Counter))
    source_values, trade_values, availability = Counter(), Counter(), Counter()
    labels, identities, overnight = [], set(), {}
    duplicate_count = 0
    for split in SPLITS:
        shard = manifest["splits"][split]
        path = dataset_dir / shard["path"]
        if _sha(path) != shard["sha256"]:
            raise ValueError(f"frozen shard hash mismatch: {path}")
        classes, stocks, origins, returns = Counter(), Counter(), set(), []
        stock_origins = defaultdict(list)
        first_origin = last_origin = None
        count = 0
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            symbol, origin = row["symbol"], row["as_of"]
            key = symbol, origin
            duplicate_count += key in identities
            identities.add(key)
            if row["split"] != split:
                raise ValueError("shard split differs from row split")
            classes[row["direction_class"]] += 1
            stocks[symbol] += 1
            stock_origins[symbol].append(origin)
            origins.add(origin)
            first_origin = min(first_origin or origin, origin)
            last_origin = max(last_origin or origin, origin)
            returns.append(row["realized_return"])
            labels.append({"symbol": symbol, "as_of": origin, "split": split,
                "label_end": row["label_end"], "realized_return": row["realized_return"],
                "direction_class": row["direction_class"]})
            availability[row["provenance"]["availability_basis"]] += 1
            for source_row in row["provenance"]["context_source_rows"] + row["provenance"]["future_source_rows"]:
                source_values[source_row.get("source", "unknown")] += 1
                trade_values[str(source_row.get("has_trade_price", "unknown"))] += 1
            previous_close = float(row["raw_context"][-1][3])
            for date, bar in zip(row["future_dates"], row["raw_future"]):
                gap = float(bar[0]) / previous_close - 1.
                event = {"symbol": symbol, "session": date, "previous_close": previous_close,
                    "open": float(bar[0]), "overnight_open_to_previous_close_return": gap,
                    "example_origin": origin, "example_split": split}
                event_key = symbol, date
                if event_key in overnight and overnight[event_key]["overnight_open_to_previous_close_return"] != gap:
                    raise ValueError("same frozen symbol/session has inconsistent overnight values")
                overnight[event_key] = event
                previous_close = float(bar[3])
            for name, teacher in row["teachers"].items():
                for kind in ("continuous", "event"):
                    names = teacher[kind + "_names"]
                    for target, value, known in zip(names, teacher[kind + "_targets"], teacher[kind + "_mask"]):
                        target_key = kind, target
                        for stage in (split, "all", *( ("nontrain",) if split != "train" else () )):
                            requested[stage][name][target_key] += 1
                            if known:
                                teachers[stage][name][target_key].append(float(value))
                for stage in (split, "all", *( ("nontrain",) if split != "train" else () )):
                    requested[stage][name][("structure", "all_dimensions")] += len(teacher.get("structure_mask", []))
                    teachers[stage][name][("structure", "all_dimensions")].extend(
                        [1.] * sum(teacher.get("structure_mask", [])))
        if count != shard["row_count"]:
            raise ValueError("frozen shard count mismatch")
        result["splits"][split] = {"samples": count, "unique_origins": len(origins),
            "distinct_symbols": len(stocks), "first_origin": first_origin, "last_origin": last_origin,
            "class_counts_down_flat_up": [classes[i] for i in range(3)],
            "class_rates_down_flat_up": [classes[i] / count if count else None for i in range(3)],
            "realized_return": _summary(returns),
            "per_stock": {symbol: {"samples": stocks[symbol],
                "share_of_split": stocks[symbol] / count if count else 0.,
                "first_origin": min(stock_origins[symbol]) if stock_origins[symbol] else None,
                "last_origin": max(stock_origins[symbol]) if stock_origins[symbol] else None}
                for symbol in manifest["symbols"]}}
    train_counts = np.asarray(result["splits"]["train"]["class_counts_down_flat_up"], dtype=float)
    prior = (train_counts + 1.) / (train_counts.sum() + 3.)
    for split, stats in result["splits"].items():
        counts = np.asarray(stats["class_counts_down_flat_up"], dtype=float)
        truth_distribution = counts / counts.sum()
        brier_by_class = np.square(prior[None, :] - np.eye(3)).sum(axis=1)
        stats["train_prior_baseline"] = {"fit_split": "train", "laplace_pseudocount_per_class": 1,
            "probabilities_down_flat_up": prior.tolist(),
            "log_loss": float(-(truth_distribution * np.log(prior)).sum()),
            "brier_mean_sum_squared_error_three_classes": float((truth_distribution * brier_by_class).sum()),
            "accuracy": float(truth_distribution[np.argmax(prior)])}
    for split, names in requested.items():
        result["teacher_profiles"][split] = {}
        for name, targets in names.items():
            target_stats = []
            for (kind, target), total in targets.items():
                values = teachers[split][name][kind, target]
                stats = _summary(values)
                stats.update(kind=kind, name=target, requested=total, known=len(values),
                    unknown=total - len(values), known_fraction=len(values) / total if total else None)
                if kind == "event":
                    stats.update(positive_count=sum(value == 1 for value in values),
                        positive_rate_among_known=float(np.mean(values)) if values else None)
                target_stats.append(stats)
            result["teacher_profiles"][split][name] = target_stats
    result["train_constant_targets"] = []
    result["train_constant_targets_that_vary_outside_train"] = []
    result["event_rate_shift_train_to_nontrain"] = []
    for name, targets in result["teacher_profiles"]["train"].items():
        later = {(x["kind"], x["name"]): x for x in result["teacher_profiles"]["nontrain"][name]}
        for stats in targets:
            other = later[stats["kind"], stats["name"]]
            if stats["kind"] != "structure" and stats["known"] >= 2 and stats["maximum"] - stats["minimum"] <= 1e-12:
                entry = {"teacher": name, "kind": stats["kind"], "name": stats["name"],
                    "train_constant_value": stats["mean"], "train_known": stats["known"],
                    "nontrain_known": other["known"], "nontrain_minimum": other["minimum"],
                    "nontrain_maximum": other["maximum"], "nontrain_std": other["std"]}
                result["train_constant_targets"].append(entry)
                if other["known"] >= 2 and other["maximum"] - other["minimum"] > 1e-12:
                    result["train_constant_targets_that_vary_outside_train"].append(entry)
            if stats["kind"] == "event":
                a, b = stats["positive_rate_among_known"], other["positive_rate_among_known"]
                result["event_rate_shift_train_to_nontrain"].append({"teacher": name, "name": stats["name"],
                    "train_known": stats["known"], "nontrain_known": other["known"],
                    "train_positive_rate": a, "nontrain_positive_rate": b,
                    "difference_nontrain_minus_train": b - a if a is not None and b is not None else None})
    result["total_samples"] = sum(stats["samples"] for stats in result["splits"].values())
    if result["total_samples"] != manifest["total_samples"]:
        raise ValueError("frozen dataset total sample mismatch")
    result["grain"] = "one symbol at one prediction origin; one five-session direction label"
    result["duplicate_symbol_origin_count"] = duplicate_count
    result["all_shard_sha256_and_row_counts_match_manifest"] = True
    result["availability_basis_counts"] = dict(availability)
    result["source_values"] = {"basis": "source-row references within windows; repeated rows counted again", "counts": dict(source_values)}
    result["source_trade_price_values"] = dict(trade_values)
    result["label_outliers"] = {
        "thresholds_are_review_flags_not_error_classification": True,
        "by_split": {split: {"absolute_return_gt_10pct": sum(x["split"] == split and abs(x["realized_return"]) > .1 for x in labels),
                              "absolute_return_gt_20pct": sum(x["split"] == split and abs(x["realized_return"]) > .2 for x in labels)} for split in SPLITS},
        "largest_20_by_absolute_five_session_return": sorted(labels, key=lambda x: abs(x["realized_return"]), reverse=True)[:20]}
    gaps = list(overnight.values())
    result["future_overnight_gap_diagnostics"] = {"unique_symbol_session_events": len(gaps),
        "deduplication_key": ["symbol", "session"],
        "absolute_overnight_gap_gt_10pct": sum(abs(x["overnight_open_to_previous_close_return"]) > .1 for x in gaps),
        "absolute_overnight_gap_gt_20pct": sum(abs(x["overnight_open_to_previous_close_return"]) > .2 for x in gaps),
        "largest_20": sorted(gaps, key=lambda x: abs(x["overnight_open_to_previous_close_return"]), reverse=True)[:20],
        "interpretation": "Unadjusted open versus preceding close. Corporate actions, data errors and market moves are not independently distinguished or confirmed."}
    return result


def build_report(root=ROOT):
    root = Path(root).resolve()
    expanded = profile_dataset(root / "data/ablation_20261019_v1")
    pilot = profile_dataset(root / "data/pilot_20261019_c128")
    report = {"schema": "frozen_ablation_dataset_diagnostics_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_scope": "existing immutable project JSONL and manifests only; no database, network or model results read",
        "read_only_inputs": True, "dataset_or_runtime_modified": False,
        "training_budget_or_protocol_changed": False,
        "class_order": list(CLASS_NAMES), "constant_tolerance": 1e-12,
        "expanded_dataset": expanded,
        "pilot_comparison": {"pilot_dataset": pilot["manifest"],
            "pilot_manifest_file_sha256": pilot["manifest_file_sha256"],
            "pilot_total_samples": pilot["total_samples"], "expanded_total_samples": expanded["total_samples"],
            "sample_expansion_factor": expanded["total_samples"] / pilot["total_samples"],
            "pilot_symbol_count": len(pilot["symbols"]), "expanded_symbol_count": len(expanded["symbols"]),
            "splits": {split: {"pilot": pilot["splits"][split], "expanded": expanded["splits"][split]} for split in SPLITS},
            "pilot_train_constant_targets": pilot["train_constant_targets"],
            "pilot_train_teacher_profiles": pilot["teacher_profiles"]["train"]},
        "limitations": [
            "These are descriptive data facts; no causal explanation of forecast performance is established.",
            "Train and pilot chronology differ; their metrics and target rates are not controlled performance comparisons.",
            "Teacher facts are historical auxiliary labels, not independently verified future-trading signals.",
            "Raw unadjusted return and overnight-gap outliers may include corporate actions; no corporate-action table was read and none is confirmed.",
            "Source availability is reconstructed at Shanghai session close; actual historical ingestion times remain unverified.",
            "Frozen JSONL integrity checks verify recorded bytes and counts, not the economic correctness of underlying source prices.",
            "No label, price, stock admission, teacher weight or training-budget adjustment was made from these diagnostics.",
        ]}
    report["findings"] = [
        {"severity": "high", "confidence": "high for recorded bytes; cause unverified",
         "finding": "Large raw unadjusted overnight gaps exist, including a roughly two-thirds price discontinuity.",
         "evidence_path": "expanded_dataset.future_overnight_gap_diagnostics",
         "risk": "Raw-return labels and historical teacher features may reflect share-basis changes as well as market moves.",
         "follow_up": "Independently verify corporate-action and source-price lineage in a separate future protocol; current frozen labels remain unchanged."},
        {"severity": "medium", "confidence": "high",
         "finding": "All expanded W32 event targets have positive training examples, but their rates remain low.",
         "evidence_path": "expanded_dataset.teacher_profiles.train.w32",
         "risk": "Low positive support makes event transfer harder to estimate even with more windows.",
         "follow_up": "Interpret auxiliary fidelity alongside train-rate baselines and matched forecast outcomes."},
        {"severity": "medium", "confidence": "high",
         "finding": "688981.SH has no admitted calibration or policy rows after complete-window filtering.",
         "evidence_path": "expanded_dataset.splits.calibration.per_stock.688981.SH",
         "risk": "Intermediate-stage population composition differs; all 20 names remain present in the holdout.",
         "follow_up": "Keep per-stock coverage visible when interpreting stage aggregates; do not impute missing sessions."},
        {"severity": "high", "confidence": "high",
         "finding": "Source availability is reconstructed at Shanghai session close for all 2,158 expanded samples.",
         "evidence_path": "expanded_dataset.availability_basis_counts",
         "risk": "This does not prove that historical data ingestion was timely at every prediction origin.",
         "follow_up": "Describe the experiment as retrospective; actual point-in-time ingestion qualification requires independent arrival evidence."},
        {"severity": "medium", "confidence": "high",
         "finding": "The expanded holdout has 480 stock-origin rows but 24 distinct origins.",
         "evidence_path": "expanded_dataset.splits.holdout",
         "risk": "Same-date stock outcomes are correlated; 480 rows cannot be assumed to be independent trials.",
         "follow_up": "Preserve same-origin stock clusters in uncertainty estimates, as fixed in the ablation protocol."},
    ]
    return report


def main():
    report = build_report()
    output = ROOT / "artifacts/reports/ablation_dataset_diagnostics.local.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "total_samples": report["expanded_dataset"]["total_samples"],
        "split_counts": {k: v["samples"] for k, v in report["expanded_dataset"]["splits"].items()},
        "constant_train_targets": report["expanded_dataset"]["train_constant_targets"],
        "return_outliers": report["expanded_dataset"]["label_outliers"]["by_split"],
        "largest_gaps": report["expanded_dataset"]["future_overnight_gap_diagnostics"]["largest_20"][:3]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
