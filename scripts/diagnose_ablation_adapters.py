#!/usr/bin/env python3
"""Read-only CPU diagnostics of selected adapters in a completed ablation grid.

This script neither initializes a model nor performs optimization, inference,
database access, merging, promotion, or external writes. LoRA products remain
low rank: their Frobenius norms use small Gram matrices in CPU float64.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
import sys

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from token_distillation.artifacts import sha256_file, write_json
from token_distillation.config import load_config

ARMS = ("none", "czsc", "ict", "dual")
PROJECTION = re.compile(r"transformer\.\d+\.self_attn\.(q_proj|v_proj)")


def _factors(a, b, scaling):
    if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
            or a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]
            or min(*a.shape, *b.shape) < 1
            or isinstance(scaling, bool) or not isinstance(scaling, (float, int))
            or not math.isfinite(scaling)):
        raise ValueError("Expected finite scaling and compatible nonempty LoRA A/B matrices")
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("LoRA factors must be finite")
    return a, b, float(scaling)


def low_rank_frobenius_squared(a, b, scaling=1.0):
    """Return ||scaling * B @ A||_F² without forming B @ A."""
    a, b, scaling = _factors(a, b, scaling)
    return max(0., float(((b.T @ b) * (a @ a.T)).sum()) * scaling**2)


def low_rank_difference_squared(a, b, scaling, other_a, other_b, other_scaling):
    """Return ||s*B@A - t*D@C||_F², including different factor ranks."""
    a, b, scaling = _factors(a, b, scaling)
    other_a, other_b, other_scaling = _factors(other_a, other_b, other_scaling)
    if a.shape[1] != other_a.shape[1] or b.shape[0] != other_b.shape[0]:
        raise ValueError("Compared LoRA products must have identical dense shapes")
    if scaling == other_scaling and torch.equal(a, other_a) and torch.equal(b, other_b):
        return 0.
    left = low_rank_frobenius_squared(a, b, scaling)
    right = low_rank_frobenius_squared(other_a, other_b, other_scaling)
    inner = float(((b.T @ other_b) * (a @ other_a.T)).sum()) * scaling * other_scaling
    result = left + right - 2 * inner
    if result < -1e-10 * max(left + right, 1.):
        raise ValueError("Unexpected negative LoRA difference squared norm")
    return max(0., result)


def validate_complete_grid(manifest):
    if manifest.get("schema") != "distillation_ablation_experiment_v1" or manifest.get("status") != "completed":
        raise ValueError("Adapter diagnostics require a completed experiment manifest")
    protocol, runs = manifest["protocol"], manifest["runs"]
    arms, seeds = protocol["arms"], protocol["seeds"]
    if len(arms) != 4 or set(arms) != set(ARMS) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Protocol must identify four arms and unique seeds")
    pairs = [(row["arm"], row["seed"]) for row in runs]
    expected = {(arm, seed) for arm in arms for seed in seeds}
    if set(pairs) != expected or len(pairs) != len(expected) or any(row.get("status") != "completed" for row in runs):
        raise ValueError("Partial, duplicate, or failed ablation grids are not supported")
    return {(row["arm"], row["seed"]): row for row in runs}


def _inside(root, path):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Experiment inputs must stay inside token_distillation")
    return path


def _read_bound_run(entry, config, manifest):
    directory = _inside(config.root, entry["run_dir"])
    summary_path = directory / "training_summary.json"
    if sha256_file(summary_path) != entry["training_summary_sha256"]:
        raise ValueError("Training summary SHA does not match experiment manifest")
    summary = json.loads(summary_path.read_text())
    checkpoint = _inside(config.root, summary["checkpoint"])
    digest = sha256_file(checkpoint)
    if digest != summary["checkpoint_sha256"] or digest != entry["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA does not match completed training records")
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if payload.get("schema") != "distilled_kronos_lora_checkpoint_v1" or payload["bindings"] != summary["bindings"]:
        raise ValueError("Checkpoint schema or training bindings changed")
    params, binding = payload["parameters"], payload["bindings"]
    if (params != summary["parameters"] or params["teacher_mode"] != entry["arm"] or params["seed"] != entry["seed"]
            or payload["optimizer_step"] != summary["selected_optimizer_step"]
            or payload["optimizer_step"] != entry["selected_optimizer_step"]):
        raise ValueError("Checkpoint parameters or selected steps differ from experiment records")
    if (binding["config_sha256"] != sha256_file(config.config_path)
            or binding["dataset_manifest_sha256"] != manifest["dataset_manifest_sha256"]):
        raise ValueError("Checkpoint config or dataset identity mismatch")
    for name in ("initial_trainable_state_sha256", "sample_order_sha256"):
        if summary[name] != entry[name]:
            raise ValueError(f"Experiment and training {name} differ")
    if params["initial_trainable_state_sha256"] != summary["initial_trainable_state_sha256"]:
        raise ValueError("Checkpoint initialization identity differs from training")
    state = payload["trainable_state"]
    if not state or any(not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                        or not value.is_floating_point() or not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Checkpoint trainable state must contain finite CPU tensors")
    return {"summary": summary, "binding": binding, "state": state, "checkpoint_sha256": digest,
            "selected_step": payload["optimizer_step"]}


def _adapters(state, rank):
    adapters = {}
    for key, value in state.items():
        if not key.startswith("backbone."):
            continue
        match = re.fullmatch(r"backbone\.(transformer\.\d+\.self_attn\.(?:q_proj|v_proj))\.lora_([AB])", key)
        if match is None:
            raise ValueError(f"Unexpected trainable backbone tensor: {key}")
        adapters.setdefault(match[1], {})[match[2]] = value
    if not adapters or any(set(pair) != {"A", "B"} or pair["A"].ndim != 2 or pair["B"].ndim != 2
                           or pair["A"].shape[0] != rank or pair["B"].shape[1] != rank
                           for pair in adapters.values()):
        raise ValueError("Missing LoRA pair or inconsistent configured rank")
    return adapters


def _head_differences(state, reference):
    if set(state) != set(reference):
        raise ValueError("Paired checkpoints must retain the same complete parameter architecture")
    result = {}
    for group in ("direction_head", "teacher_heads", "all_heads"):
        names = [name for name in state if not name.startswith("backbone.")
                 and (group == "all_heads" or (name.startswith("direction_head.") == (group == "direction_head")))]
        delta = reference_squared = 0.
        for name in names:
            if state[name].shape != reference[name].shape:
                raise ValueError("Paired head dimensions differ")
            left, right = state[name].double(), reference[name].double()
            delta += float((left - right).square().sum())
            reference_squared += float(right.square().sum())
        result[group] = {"tensor_count": len(names), "difference_l2_vs_none": math.sqrt(delta),
                         "none_l2": math.sqrt(reference_squared),
                         "relative_difference_l2_vs_none": math.sqrt(delta / reference_squared) if reference_squared else None}
    return result


def diagnose_adapters(experiment_dir, *, config=None):
    experiment_dir = Path(experiment_dir).resolve()
    manifest_path = experiment_dir / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = config or load_config(manifest["config"])
    _inside(config.root, experiment_dir)
    entries = validate_complete_grid(manifest)
    if sha256_file(config.config_path) != manifest["protocol"]["config_sha256"]:
        raise ValueError("Experiment protocol config SHA changed")
    dataset = _inside(config.root, manifest["dataset"])
    if sha256_file(dataset) != manifest["dataset_manifest_sha256"]:
        raise ValueError("Frozen experiment dataset manifest SHA changed")
    runs = {key: _read_bound_run(entry, config, manifest) for key, entry in entries.items()}
    bindings = [run["binding"] for run in runs.values()]
    if any(binding != bindings[0] for binding in bindings):
        raise ValueError("All ablation runs must share identical dataset/config/assets/runtime bindings")
    asset = bindings[0]["assets"]["files"]["model/model.safetensors"]
    base_path = (config.kronos_root / "model.safetensors").resolve()
    if Path(asset["path"]).resolve() != base_path or sha256_file(base_path) != asset["sha256"]:
        raise ValueError("Configured frozen base asset SHA/path differs from checkpoint binding")
    rank = config.values["student"]["lora_rank"]
    scaling = config.values["student"]["lora_alpha"] / rank
    for run in runs.values():
        run["adapters"] = _adapters(run["state"], rank)
    with safe_open(str(base_path), framework="pt", device="cpu") as base:
        expected = {key.removesuffix(".weight") for key in base.keys()
                    if key.endswith(".weight") and PROJECTION.fullmatch(key.removesuffix(".weight"))}
        if not expected or any(set(run["adapters"]) != expected for run in runs.values()):
            raise ValueError("Checkpoint LoRA targets differ from the frozen base q/v projections")
        base_norms = {}
        for name in sorted(expected):
            weight = base.get_tensor(name + ".weight")
            if not bool(torch.isfinite(weight).all()):
                raise ValueError("Frozen base contains nonfinite q/v weights")
            base_norms[name] = float(weight.double().square().sum())
            for run in runs.values():
                pair = run["adapters"][name]
                if pair["B"].shape[0] != weight.shape[0] or pair["A"].shape[1] != weight.shape[1]:
                    raise ValueError("LoRA dimensions differ from frozen base projection")
    denominator = sum(base_norms.values())
    if denominator <= 0:
        raise ValueError("Frozen q/v projection norm must be positive")
    rows = []
    for seed in manifest["protocol"]["seeds"]:
        none = runs["none", seed]
        for arm in ARMS:
            run = runs[arm, seed]
            for field in ("initial_trainable_state_sha256", "sample_order_sha256"):
                if run["summary"][field] != none["summary"][field]:
                    raise ValueError(f"Same-seed paired {field} mismatch")
            layers, total, difference = [], 0., 0.
            for name, pair in sorted(run["adapters"].items()):
                reference = none["adapters"][name]
                norm = low_rank_frobenius_squared(pair["A"], pair["B"], scaling)
                diff = low_rank_difference_squared(pair["A"], pair["B"], scaling, reference["A"], reference["B"], scaling)
                total += norm
                difference += diff
                layers.append({"name": name, "base_weight_frobenius": math.sqrt(base_norms[name]),
                               "effective_delta_frobenius": math.sqrt(norm),
                               "effective_delta_difference_vs_none_frobenius": math.sqrt(diff)})
            matched = run["selected_step"] == none["selected_step"]
            rows.append({"arm": arm, "seed": seed, "selected_optimizer_step": run["selected_step"],
                "none_selected_optimizer_step": none["selected_step"], "selected_steps_match": matched,
                "step_confound": None if matched else "selected_checkpoints_have_different_optimizer_steps; cannot_attribute_difference_solely_to_teacher",
                "initialization_and_full_optimization_sample_order_match": True,
                "checkpoint_sha256": run["checkpoint_sha256"], "projection_count": len(layers),
                "base_qv_frobenius": math.sqrt(denominator), "effective_lora_delta_frobenius": math.sqrt(total),
                "effective_lora_delta_relative_to_base_qv": math.sqrt(total / denominator),
                "effective_lora_difference_vs_none_frobenius": math.sqrt(difference),
                "effective_lora_difference_vs_none_relative_to_base_qv": math.sqrt(difference / denominator),
                "head_differences_vs_none": _head_differences(run["state"], none["state"]), "layers": layers})
    by_arm = {}
    for arm in ARMS:
        members = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {"seed_count": len(members), "all_selected_steps_match_none": all(row["selected_steps_match"] for row in members)}
        for field in ("effective_lora_delta_relative_to_base_qv", "effective_lora_difference_vs_none_relative_to_base_qv"):
            values = [row[field] for row in members]
            by_arm[arm][field] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None}
    return {"schema": "controlled_ablation_selected_adapter_diagnostics_v1", "status": "completed_analysis_only",
        "created_at": datetime.now(timezone.utc).isoformat(), "experiment_manifest": str(manifest_path),
        "experiment_manifest_sha256": sha256_file(manifest_path), "base_asset": {"path": str(base_path), "sha256": asset["sha256"]},
        "device": "cpu", "arithmetic": "float64_low_rank_Gram_trace; no_dense_LoRA_product_materialized",
        "lora_rank": rank, "lora_alpha": config.values["student"]["lora_alpha"], "lora_scaling": scaling,
        "definitions": {"effective_delta": "(alpha/rank)*B@A; B_initialized_zero",
            "relative_norm": "sqrt(sum_projection_Frobenius_squared_delta / sum_projection_Frobenius_squared_base_weight)",
            "paired_difference": "effective_delta_arm_minus_effective_delta_same_seed_no_teacher",
            "heads": "selected_checkpoint_parameter_difference_vs_same_seed_no_teacher"},
        "interpretation": "nonzero_LoRA_difference_confirms_different_selected_backbone_adaptation; does_not_prove_predictive_utility_or_causal_teacher_effect",
        "limitations": ["MPS_and_other_backend_numerical_nondeterminism_can_contribute_to_small_differences",
            "different_selected_steps_confound_teacher_and_optimization_time_effects",
            "only_selected_checkpoints_are_available; full_budget_final_adapters_are_not_compared",
            "auxiliary_head_learning_alone_does_not_establish_backbone_or_prediction_improvement"],
        "optimizer_steps_performed": 0, "gpu_used": False, "database_accessed": False,
        "external_sources_modified": False, "model_promoted": False, "rows": rows, "by_arm": by_arm}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.experiment_dir / "experiment_manifest.json").read_text())
    config = load_config(args.config or manifest["config"])
    target = config.output_path(args.output)
    if target.exists():
        raise FileExistsError(f"Refusing to replace an existing diagnostic: {target}")
    result = diagnose_adapters(args.experiment_dir, config=config)
    write_json(config, target, result)
    print(json.dumps({"output": str(target), "status": result["status"], "rows": len(result["rows"]), "optimizer_steps_performed": 0}))


if __name__ == "__main__":
    main()
