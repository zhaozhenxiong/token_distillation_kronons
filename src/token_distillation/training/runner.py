"""Explicit offline research training and locked, chronological evaluation.

Importing this module performs no data generation, model loading or optimizer
steps. Only ``train`` starts optimization. All persisted weights are LoRA and
internal heads, and all five chronological stages retain their identities.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..artifacts import sha256_file, write_json
from ..config import ProjectConfig, SPLIT_NAMES
from ..evaluation.metrics import direction_metrics
from ..models.runtime import (asset_identity, canonical_hash, causal_decode_s2, frozen_base,
                              load_official_assets, make_student, resolve_device,
                              restore_trainable_state, trainable_state)
from .losses import masked_bce_with_logits_loss, masked_cross_entropy_loss, masked_huber_loss, masked_kl_div_loss

DEFAULT_LOSS_WEIGHTS = {"ar_s1": 1.0, "ar_s2": 1.0, "direction": 1.0,
                        "w32_continuous": 0.2, "w32_events": 0.2,
                        "ict_continuous": 0.2, "ict_events": 0.2,
                        "czsc_continuous": 0.2, "czsc_events": 0.0, "base_kl": 0.05}
TEACHER_MODES = ("none", "czsc", "ict", "dual")


def _manifest_path(dataset: Path) -> Path:
    dataset = Path(dataset).resolve()
    return dataset / "manifest.json" if dataset.is_dir() else dataset


def teacher_specification(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("Dataset contains no samples")
    spec = {name: {"schema": teacher["schema"], "schema_sha256": teacher["schema_sha256"],
                   "continuous_names": teacher["continuous_names"], "event_names": teacher["event_names"]}
            for name, teacher in samples[0]["teachers"].items()}
    for sample in samples:
        if set(sample["teachers"]) != set(spec):
            raise ValueError("Dataset teacher set changed across samples")
        for name, teacher in sample["teachers"].items():
            actual = {key: teacher[key] for key in spec[name]}
            if actual != spec[name]:
                raise ValueError(f"Dataset teacher schema/order changed: {name}")
            for kind in ("continuous", "event"):
                targets, mask = teacher[kind + "_targets"], teacher[kind + "_mask"]
                if len(targets) != len(spec[name][kind + "_names"]) or len(mask) != len(targets):
                    raise ValueError(f"Teacher target/mask dimensions changed: {name}")
                for value, valid in zip(targets, mask):
                    if type(valid) is not bool or (valid and (isinstance(value, bool)
                            or not isinstance(value, (int, float)) or not math.isfinite(value))):
                        raise ValueError(f"Teacher valid targets must be finite with explicit boolean masks: {name}")
                    if kind == "event" and valid and not 0 <= value <= 1:
                        raise ValueError("Valid event targets must be in [0,1]")
    return spec


def fit_teacher_statistics(train_samples: list[dict], teacher_spec: dict) -> dict:
    """Continuous teacher scaling is fitted on train's known targets only."""
    if any(sample["split"] != "train" for sample in train_samples):
        raise ValueError("Teacher statistics may be fitted on the training split only")
    result = {}
    for name, spec in teacher_spec.items():
        means, scales, counts = [], [], []
        for column in range(len(spec["continuous_names"])):
            values = [sample["teachers"][name]["continuous_targets"][column] for sample in train_samples
                      if sample["teachers"][name]["continuous_mask"][column]]
            means.append(float(np.mean(values)) if values else 0.0)
            scales.append(max(float(np.std(values)), 1e-5) if values else 1.0)
            counts.append(len(values))
        result[name] = {"mean": means, "scale": scales, "valid_count": counts,
                        "event_valid_count": [sum(sample["teachers"][name]["event_mask"][column]
                                                   for sample in train_samples)
                                              for column in range(len(spec["event_names"]))],
                        "event_rate": [float(np.mean(values)) if values else 0.5
                                       for column in range(len(spec["event_names"]))
                                       for values in [[sample["teachers"][name]["event_targets"][column]
                                                       for sample in train_samples
                                                       if sample["teachers"][name]["event_mask"][column]]]],
                        "fit_split": "train", "unknown_targets": "masked"}
    return result


def encode_samples(tokenizer, samples: list[dict], config: ProjectConfig, device: torch.device,
                   *, chunk_size: int = 8, use_cache: bool = True) -> tuple[list[dict], dict]:
    """Tokenize frozen official assets and reject any future-to-prefix leakage."""
    if use_cache and getattr(config, "values", {}).get("training", {}).get("token_cache"):
        from .token_cache import cached_tokens
        return cached_tokens(config, samples)
    encoded = []
    tokenizer.eval()
    c, h = config.context_length, config.horizon
    for start in range(0, len(samples), chunk_size):
        batch = samples[start:start + chunk_size]
        context = torch.tensor(np.stack([row["normalized_context"] for row in batch]), dtype=torch.float32, device=device)
        future = torch.tensor(np.stack([row["normalized_future"] for row in batch]), dtype=torch.float32, device=device)
        if context.shape != (len(batch), c, 6) or future.shape != (len(batch), h, 6):
            raise ValueError("Dataset context/horizon shapes differ from frozen configuration")
        if not bool(torch.isfinite(context).all() and torch.isfinite(future).all()):
            raise ValueError("Dataset normalized inputs must be finite")
        with torch.no_grad():
            prefix = tokenizer.encode(context, half=True)
            full = tokenizer.encode(torch.cat([context, future], dim=1), half=True)
        if (len(full) != 2 or any(item.shape != (len(batch), c + h) for item in full)
                or any(not torch.equal(left, right[:, :c]) for left, right in zip(prefix, full))):
            raise ValueError("Official tokenizer failed exact context prefix invariance; refusing training/evaluation")
        for index, sample in enumerate(batch):
            row = dict(sample)
            row["_s1"], row["_s2"] = full[0][index].cpu(), full[1][index].cpu()
            encoded.append(row)
        if not use_cache and (len(encoded) % 1024 == 0 or len(encoded) == len(samples)):
            print(f"tokenizer verified_samples={len(encoded)}/{len(samples)} exact_prefix=True", file=sys.stderr, flush=True)
    return encoded, {"checked_samples": len(samples), "prefix_token_ids_exact": True,
                     "context_encoded_alone_and_with_future": True, "tokenizer_frozen": True}


def _batch(rows: list[dict], teacher_spec: dict, teacher_statistics: dict,
           device: torch.device, context_length: int) -> dict:
    s1 = torch.stack([row["_s1"] for row in rows]).to(device)
    s2 = torch.stack([row["_s2"] for row in rows]).to(device)
    stamp = torch.tensor([row["time_context"] + row["time_future"] for row in rows], device=device, dtype=torch.long)
    if stamp.shape != (len(rows), s1.shape[1], 5):
        raise ValueError("Temporal features must use [minute,hour,weekday,day,month]")
    ar_mask = torch.zeros_like(s1[:, 1:], dtype=torch.bool)
    ar_mask[:, context_length - 1:] = True
    result = {"s1_inputs": s1[:, :-1], "s2_inputs": s2[:, :-1], "s1_targets": s1[:, 1:],
              "s2_targets": s2[:, 1:], "stamp": stamp[:, :-1], "ar_mask": ar_mask,
              "direction": torch.tensor([row["direction_class"] for row in rows], dtype=torch.long, device=device),
              "teachers": {}}
    if "context_features" in rows[0]:
        result["context_features"] = torch.tensor([row["context_features"] for row in rows], device=device, dtype=torch.float32)
    for name in teacher_spec:
        teacher = {}
        for kind in ("continuous", "event"):
            mask = torch.tensor([row["teachers"][name][kind + "_mask"] for row in rows], dtype=torch.bool, device=device)
            target = torch.tensor([[0.0 if value is None else value for value in row["teachers"][name][kind + "_targets"]]
                                   for row in rows], dtype=torch.float32, device=device)
            if kind == "continuous":
                stats = teacher_statistics[name]
                target = ((target - target.new_tensor(stats["mean"])) / target.new_tensor(stats["scale"]))
            teacher[kind + "_targets"], teacher[kind + "_mask"] = target, mask
        result["teachers"][name] = teacher
    return result


def forward_objective(student, batch: dict, *, context_length: int, loss_weights: dict,
                      include_base: bool = False) -> tuple[torch.Tensor, dict, dict]:
    """Score future AR tokens, with teacher/direction heads at the origin only."""
    s1_logits, hidden = student.backbone.decode_s1(batch["s1_inputs"], batch["s2_inputs"],
                                                  stamp=batch["stamp"], padding_mask=None)
    s2_logits = causal_decode_s2(student.backbone, hidden, batch["s1_targets"])
    origin = student.head_input(hidden[:, context_length - 1], batch.get("context_features"))
    outputs = {"s1_logits": s1_logits, "s2_logits": s2_logits,
               "direction_logits": student.direction_head(origin),
               "teachers": {"w32": {"continuous": student.continuous_head(origin),
                                      "event_logits": student.event_head(origin)}}}
    for name, heads in student.extra_teacher_heads.items():
        outputs["teachers"][name] = {"continuous": heads["continuous"](origin),
                                     "event_logits": heads["event_logits"](origin) if "event_logits" in heads
                                     else origin.new_empty(origin.shape[0], 0)}
    losses = {"ar_s1": masked_cross_entropy_loss(s1_logits, batch["s1_targets"], batch["ar_mask"]),
              "ar_s2": masked_cross_entropy_loss(s2_logits, batch["s2_targets"], batch["ar_mask"]),
              "direction": masked_cross_entropy_loss(outputs["direction_logits"], batch["direction"])}
    for name, target in batch["teachers"].items():
        prediction = outputs["teachers"][name]
        losses[name + "_continuous"] = (masked_huber_loss(prediction["continuous"], target["continuous_targets"], target["continuous_mask"])
            if target["continuous_mask"].numel() and bool(target["continuous_mask"].any()) else prediction["continuous"].sum() * 0)
        # MPS BCE kernels reject empty buffers. Continuous-only native CZSC
        # has no event columns, and any teacher may have an all-unknown batch.
        # Build its differentiable zero from the nonempty origin instead.
        losses[name + "_events"] = (masked_bce_with_logits_loss(prediction["event_logits"], target["event_targets"], target["event_mask"])
            if target["event_mask"].numel() and bool(target["event_mask"].any()) else origin.sum() * 0)
    if include_base or loss_weights.get("base_kl", 0):
        with frozen_base(student) as backbone:
            base_s1, base_hidden = backbone.decode_s1(batch["s1_inputs"], batch["s2_inputs"], stamp=batch["stamp"], padding_mask=None)
            base_s2 = causal_decode_s2(backbone, base_hidden, batch["s1_targets"])
        losses["base_kl"] = (masked_kl_div_loss(s1_logits, base_s1, batch["ar_mask"])
                              + masked_kl_div_loss(s2_logits, base_s2, batch["ar_mask"])) / 2
        outputs["base_ar_s1"] = masked_cross_entropy_loss(base_s1, batch["s1_targets"], batch["ar_mask"])
        outputs["base_ar_s2"] = masked_cross_entropy_loss(base_s2, batch["s2_targets"], batch["ar_mask"])
    else:
        losses["base_kl"] = s1_logits.sum() * 0
    # Excluding zero-weight branches entirely leaves their gradients None;
    # AdamW must not silently decay heads in the no-teacher control.
    active = [losses[name] * loss_weights.get(name, 0.) for name in losses if loss_weights.get(name, 0.) > 0]
    if not active:
        raise ValueError("At least one objective weight must be positive")
    total = sum(active)
    if not bool(torch.isfinite(total)):
        raise ValueError("Nonfinite distillation objective")
    return total, losses, outputs


def _loss_weights(config: ProjectConfig, teacher_spec: dict, *, teacher_mode: str = "dual",
                  auxiliary_budget: float = 0.6) -> dict:
    """Equal total auxiliary weight for every enabled teacher ablation.

    CZSC allocates half its budget to W32, and half to native structure.
    Dual divides the same total budget between CZSC and ICT. Enabled targets
    inside each branch retain the explicitly configured relative weights.
    """
    if teacher_mode not in TEACHER_MODES:
        raise ValueError(f"teacher_mode must be one of {TEACHER_MODES}")
    if (isinstance(auxiliary_budget, bool) or not isinstance(auxiliary_budget, (int, float))
            or not math.isfinite(auxiliary_budget) or auxiliary_budget <= 0):
        raise ValueError("auxiliary_budget must be positive and finite")
    weights = dict(DEFAULT_LOSS_WEIGHTS)
    overrides = config.values["training"].get("loss_weights", {})
    if not isinstance(overrides, dict) or set(overrides) - set(weights):
        raise ValueError("Unknown distillation loss weight")
    weights.update(overrides)
    for name, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Loss weights must be finite nonnegative numbers: {name}")
    for name in teacher_spec:
        weights.setdefault(name + "_continuous", 0.2)
        weights.setdefault(name + "_events", 0.2)
    enabled = {"none": {}, "czsc": {"w32": 0.5, "czsc": 0.5},
               "ict": {"ict": 1.0}, "dual": {"w32": 0.25, "czsc": 0.25, "ict": 0.5}}[teacher_mode]
    for name in teacher_spec:
        keys = [name + "_continuous", name + "_events"]
        total = sum(weights[key] for key in keys)
        budget = auxiliary_budget * enabled.get(name, 0.)
        if budget and (not total or name not in teacher_spec):
            raise ValueError(f"Enabled teacher branch has no positive objective weight: {name}")
        for key in keys:
            weights[key] = budget * weights[key] / total if total else 0.0
    if set(enabled) - set(teacher_spec):
        raise ValueError(f"Enabled teacher branches absent from dataset: {sorted(set(enabled) - set(teacher_spec))}")
    return weights


def evaluate_samples(student, samples: list[dict], *, context_length: int, teacher_spec: dict,
                     teacher_statistics: dict, loss_weights: dict, device: torch.device,
                     direction_prior: list[float], batch_size: int = 4, include_base: bool = True) -> tuple[dict, list[dict]]:
    """Pure finite forward evaluation; no fitting, optimizer or policy changes."""
    if not samples:
        return {"requested": 0, "valid": 0, "status": "empty_split", "qualified": False}, []
    student.eval()
    losses_sum, loss_counts, requested = {}, {}, 0
    predictions, probabilities, targets = [], [], []
    teacher_totals = {name: {"continuous_absolute_error": 0., "continuous_normalized_absolute_error": 0., "continuous_valid": 0,
                             "event_bce_sum": 0., "event_valid": 0,
                             "train_mean_normalized_absolute_error": 0., "train_rate_event_bce_sum": 0.,
                             "continuous_requested": 0, "event_requested": 0,
                             "per_target_error": [0.] * len(spec["continuous_names"]),
                             "per_target_valid": [0] * len(spec["continuous_names"])} for name, spec in teacher_spec.items()}
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            rows = samples[start:start + batch_size]
            batch = _batch(rows, teacher_spec, teacher_statistics, device, context_length)
            total, components, output = forward_objective(student, batch, context_length=context_length,
                                                           loss_weights=loss_weights, include_base=include_base)
            values = dict(components)
            if include_base:
                values.update(frozen_base_ar_s1=output["base_ar_s1"], frozen_base_ar_s2=output["base_ar_s2"])
            for name, value in values.items():
                if name in {"ar_s1", "ar_s2", "base_kl", "frozen_base_ar_s1", "frozen_base_ar_s2"}:
                    valid_elements = int(batch["ar_mask"].sum().cpu())
                elif name == "direction":
                    valid_elements = len(rows)
                else:
                    teacher_name, kind = name.rsplit("_", 1)
                    mask = batch["teachers"][teacher_name]["continuous_mask" if kind == "continuous" else "event_mask"]
                    valid_elements = int(mask.sum().cpu()) if mask.numel() else 0
                losses_sum[name] = losses_sum.get(name, 0.) + float(value.cpu()) * valid_elements
                loss_counts[name] = loss_counts.get(name, 0) + valid_elements
            p = F.softmax(output["direction_logits"], dim=-1).cpu().tolist()
            probabilities.extend(p)
            targets.extend(row["direction_class"] for row in rows)
            requested += len(rows)
            for name, prediction in output["teachers"].items():
                target = batch["teachers"][name]
                cm, em = target["continuous_mask"], target["event_mask"]
                stats = teacher_totals[name]
                continuous_valid = int(cm.sum().cpu()) if cm.numel() else 0
                event_valid = int(em.sum().cpu()) if em.numel() else 0
                if continuous_valid:
                    scale = prediction["continuous"].new_tensor(teacher_statistics[name]["scale"])
                    normalized_error = (prediction["continuous"] - target["continuous_targets"])
                    raw_error = normalized_error * scale
                    stats["continuous_absolute_error"] += float(raw_error[cm].abs().sum().cpu())
                    stats["continuous_normalized_absolute_error"] += float(normalized_error[cm].abs().sum().cpu())
                    stats["train_mean_normalized_absolute_error"] += float(target["continuous_targets"][cm].abs().sum().cpu())
                    for column in range(cm.shape[1]):
                        column_valid = int(cm[:, column].sum().cpu())
                        if column_valid:
                            stats["per_target_error"][column] += float(raw_error[:, column][cm[:, column]].abs().sum().cpu())
                            stats["per_target_valid"][column] += column_valid
                rates = teacher_statistics[name].get("event_rate")
                if event_valid and rates is not None:
                    reference = target["event_targets"].new_tensor(rates).clamp(1e-6, 1 - 1e-6)
                    reference = reference[None, :].expand_as(target["event_targets"])
                    stats["train_rate_event_bce_sum"] += float(F.binary_cross_entropy(
                        reference[em], target["event_targets"][em], reduction="sum").cpu())
                stats["continuous_valid"] += continuous_valid
                stats["continuous_requested"] += cm.numel()
                if event_valid:
                    stats["event_bce_sum"] += float(F.binary_cross_entropy_with_logits(
                        prediction["event_logits"][em], target["event_targets"][em], reduction="sum").cpu())
                stats["event_valid"] += event_valid
                stats["event_requested"] += em.numel()
            for index, row in enumerate(rows):
                predictions.append({"symbol": row["symbol"], "as_of": row["as_of"], "split": row["split"],
                                     "direction_class": row["direction_class"], "direction_probabilities": p[index],
                                     "predicted_direction_class": int(np.argmax(p[index])),
                                     "direction_log_loss": float(-np.log(max(p[index][row["direction_class"]], 1e-12))),
                                     "realized_return": row.get("realized_return"),
                                     "output_semantics": "student_internal_predictions_not_native_confirmed_facts"})
    direction = direction_metrics(probabilities, np.asarray(targets, dtype=np.int64))
    direction["accuracy"] = float(np.mean(np.argmax(probabilities, axis=1) == targets))
    direction["target_class_counts"] = np.bincount(targets, minlength=3).tolist()
    direction["predicted_class_counts"] = np.bincount(np.argmax(probabilities, axis=1), minlength=3).tolist()
    direction["mean_probabilities"] = np.asarray(probabilities).mean(axis=0).tolist()
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (targets, np.argmax(probabilities, axis=1)), 1)
    direction["confusion_true_rows_predicted_columns"] = confusion.tolist()
    prior = direction_metrics([direction_prior] * len(targets), np.asarray(targets, dtype=np.int64))
    prior["fit_split"] = "train"
    for name, stats in teacher_totals.items():
        stats["continuous_mae_original_units"] = (stats.pop("continuous_absolute_error") / stats["continuous_valid"]
                                                    if stats["continuous_valid"] else None)
        stats["continuous_mae_normalized"] = (stats.pop("continuous_normalized_absolute_error") / stats["continuous_valid"]
                                               if stats["continuous_valid"] else None)
        stats["per_continuous_target"] = [
            {"name": target, "valid": valid, "mae_original_units": error / valid if valid else None}
            for target, valid, error in zip(teacher_spec[name]["continuous_names"],
                                            stats.pop("per_target_valid"), stats.pop("per_target_error"))]
        stats["mixed_original_units_average_interpretation"] = "diagnostic_only; compare_per_target_or_normalized_MAE"
        stats["event_bce"] = stats.pop("event_bce_sum") / stats["event_valid"] if stats["event_valid"] else None
        stats["train_constant_baseline"] = {
            "fit_split": "train", "continuous_prediction": "train_known_target_mean",
            "continuous_mae_normalized": stats.pop("train_mean_normalized_absolute_error") / stats["continuous_valid"]
                if stats["continuous_valid"] else None,
            "event_prediction": "train_known_target_rate_clipped_1e-6",
            "event_bce": stats.pop("train_rate_event_bce_sum") / stats["event_valid"]
                if stats["event_valid"] and "event_rate" in teacher_statistics[name] else None}
    mean_losses = {name: value / loss_counts[name] if loss_counts[name] else 0. for name, value in losses_sum.items()}
    mean_losses["total"] = sum(mean_losses[name] * loss_weights.get(name, 0.) for name in components)
    return {"requested": requested, "valid": requested, "excluded": 0,
            "mean_losses": mean_losses, "loss_valid_elements": loss_counts,
            "loss_aggregation": "global_masked_element_mean_per_component_then_fixed_weighted_total",
            "direction": direction, "direction_train_prior_baseline": prior, "teachers": teacher_totals,
            "ar_scoring": "future_only_teacher_forced_s1_and_causal_s2; not_free_running_forecast_metrics",
            "qualified": False}, predictions


def _bindings(config: ProjectConfig, manifest_path: Path, assets: dict, teacher_spec: dict) -> dict:
    package = Path(__file__).resolve().parents[1]
    runtime_files = ("training/runner.py", "training/losses.py", "models/runtime.py", "models/student.py", "models/lora.py", "data/dataset.py", "data/compact.py", "training/token_cache.py")
    return {"dataset_manifest_sha256": sha256_file(manifest_path), "config_sha256": sha256_file(config.config_path),
            "assets": assets, "teacher_spec_sha256": canonical_hash(teacher_spec),
            "token_cache_manifest_sha256": (sha256_file(config.output_path(config.values["training"]["token_cache"]) / "manifest.json")
                if config.values["training"].get("token_cache") else None),
            "runtime_sha256": {name: sha256_file(package / name) for name in runtime_files}}


def _atomic_checkpoint(config: ProjectConfig, target: Path, payload: dict) -> None:
    target = config.output_path(target)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".checkpoint-", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _new_output(config: ProjectConfig, output_dir: Path) -> Path:
    output = config.output_path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Bind initialization without persisting another set of model tensors."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _parameter_movement(initial: dict, final: dict) -> dict:
    result = {}
    for group in ("lora", "direction_head", "teacher_heads"):
        names = [name for name in initial if (group == "lora" and ".lora_" in name)
                 or (group == "direction_head" and name.startswith("direction_head."))
                 or (group == "teacher_heads" and not name.startswith(("backbone.", "direction_head.")))]
        squared = sum(float((final[name].double() - initial[name].double()).square().sum()) for name in names)
        reference = sum(float(initial[name].double().square().sum()) for name in names)
        result[group] = {"changed_tensors": sum(not torch.equal(initial[name], final[name]) for name in names),
                         "delta_l2": math.sqrt(squared), "initial_l2": math.sqrt(reference),
                         "relative_delta_l2": math.sqrt(squared / reference) if reference else None}
    return result


def gradient_diagnostics(student, batch: dict, *, context_length: int, loss_weights: dict) -> dict:
    """One train-only backward diagnostic; no optimizer or parameter mutation."""
    student.eval()
    _, components, _ = forward_objective(student, batch, context_length=context_length, loss_weights=loss_weights)
    parameters = [parameter for name, parameter in student.named_parameters()
                  if parameter.requires_grad and ".lora_" in name]
    groups = {"ar": ("ar_s1", "ar_s2"), "direction": ("direction",),
              "czsc": ("w32_continuous", "w32_events", "czsc_continuous", "czsc_events"),
              "ict": ("ict_continuous", "ict_events")}
    gradients, result = {}, {}
    for name, keys in groups.items():
        terms = [components[key] * loss_weights.get(key, 0.) for key in keys
                 if key in components and loss_weights.get(key, 0.) > 0]
        if not terms or not parameters:
            result[name] = {"active": False, "weighted_shared_lora_gradient_l2": 0.0}
            continue
        gradients[name] = torch.autograd.grad(sum(terms), parameters, retain_graph=True, allow_unused=True)
        norm = math.sqrt(sum(float(value.detach().cpu().double().square().sum())
                             for value in gradients[name] if value is not None))
        result[name] = {"active": True, "weighted_shared_lora_gradient_l2": norm}
    cosines = {}
    for left in gradients:
        for right in gradients:
            if left >= right:
                continue
            dot = sum(float((a.detach().cpu().double() * b.detach().cpu().double()).sum())
                      for a, b in zip(gradients[left], gradients[right]) if a is not None and b is not None)
            denominator = result[left]["weighted_shared_lora_gradient_l2"] * result[right]["weighted_shared_lora_gradient_l2"]
            cosines[left + "__" + right] = dot / denominator if denominator else None
    return {"fit_split": "train", "sample_count": len(batch["direction"]),
            "parameter_scope": "shared_backbone_lora_only", "groups": result, "gradient_cosines": cosines,
            "interpretation": "single_train_batch_local_diagnostic_not_causal_proof"}


def _selection_value(metrics: dict, metric: str) -> float:
    if metric != "direction_log_loss":
        raise ValueError("Checkpoint selection supports the common tune direction_log_loss only")
    value = metrics["direction"]["log_loss"]
    if value is None or not math.isfinite(value):
        raise ValueError("Checkpoint selection requires finite tune direction LogLoss")
    return value


def shuffle_teacher_targets(rows, seed):
    """Negative control: per-target permutation of KNOWN train values only."""
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Teacher permutation is train-only")
    result=[{**row,"teachers":copy.deepcopy(row["teachers"])} for row in rows]
    rng=random.Random(seed+9173)
    for name in result[0]["teachers"]:
        for kind in ("continuous","event"):
            for col in range(len(result[0]["teachers"][name][kind+"_mask"])):
                known=[i for i,row in enumerate(result) if row["teachers"][name][kind+"_mask"][col]]
                values=[result[i]["teachers"][name][kind+"_targets"][col] for i in known]
                rng.shuffle(values)
                for i,value in zip(known,values):result[i]["teachers"][name][kind+"_targets"][col]=value
    return result


def train(config: ProjectConfig, *, dataset: Path, output_dir: Path, device: str = "auto",
          max_steps: int = 24, learning_rate: float = 1e-4, batch_size: int = 1,
          gradient_accumulation_steps: int = 1, seed: int = 42, teacher_mode: str = "dual",
          auxiliary_budget: float = 0.6, selection_metric: str = "direction_log_loss",
          evaluation_interval: int | None = None, evaluation_batch_size: int = 4,
          mechanism_control: str = "none") -> dict:
    """Optimize a new research candidate; never resume or promote source state."""
    wall_start = time.perf_counter()
    if config.values["training"].get("execution_enabled") is not True:
        raise ValueError("Explicit training requires training.execution_enabled=true in the supplied config")
    for name, value in (("max_steps", max_steps), ("batch_size", batch_size),
                        ("gradient_accumulation_steps", gradient_accumulation_steps),
                        ("evaluation_batch_size", evaluation_batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if evaluation_interval is not None and (type(evaluation_interval) is not int or evaluation_interval < 1):
        raise ValueError("evaluation_interval must be a positive integer")
    if selection_metric != "direction_log_loss":
        raise ValueError("Checkpoint selection supports the common tune direction_log_loss only")
    if (type(seed) is not int or not 0 <= seed <= 2**32 - 1 or isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float)) or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Invalid seed or learning rate")
    if mechanism_control not in ("none", "shuffle", "frozen_backbone"):
        raise ValueError("Unknown mechanism control")
    resolved = resolve_device(device)
    from ..data.dataset import load_dataset
    manifest_path = _manifest_path(dataset)
    from .token_cache import load_model_dataset
    manifest, samples = load_model_dataset(config, manifest_path)
    teacher_spec = teacher_specification(samples)
    groups = {name: [row for row in samples if row["split"] == name] for name in SPLIT_NAMES}
    if not groups["train"] or not groups["tune"]:
        raise ValueError("Training requires nonempty train and tune splits; tune alone selects checkpoints")
    if manifest.get("context_length") != config.context_length or manifest.get("horizon") != config.horizon:
        raise ValueError("Dataset context/horizon do not match training config")
    weights = _loss_weights(config, teacher_spec, teacher_mode=teacher_mode, auxiliary_budget=auxiliary_budget)
    statistics = fit_teacher_statistics(groups["train"], teacher_spec)
    for name, stats in statistics.items():
        active_continuous = weights.get(name + "_continuous", 0.2) > 0
        active_events = weights.get(name + "_events", 0.2) > 0
        if ((sum(stats["valid_count"]) if active_continuous else 0)
                + (sum(stats["event_valid_count"]) if active_events else 0) == 0
                and (active_continuous or active_events)):
            raise ValueError(f"Active teacher branch has zero valid train supervision: {name}; increase context or repair data")
    output = _new_output(config, output_dir)
    torch.manual_seed(seed)
    if resolved.type == "mps":
        torch.mps.manual_seed(seed)
    rng = random.Random(seed)
    backbone, tokenizer, assets = load_official_assets(config, device=resolved)
    student = make_student(config, backbone, teacher_spec)
    # Only train and tune are even tokenized during fitting/selection.
    encoded, prefix_check = encode_samples(tokenizer, groups["train"] + groups["tune"], config, resolved)
    train_rows = [row for row in encoded if row["split"] == "train"]
    if mechanism_control == "shuffle":
        train_rows = shuffle_teacher_targets(train_rows, seed)
    if mechanism_control == "frozen_backbone":
        for parameter in student.backbone.parameters(): parameter.requires_grad_(False)
    tune_rows = [row for row in encoded if row["split"] == "tune"]
    counts = np.bincount([row["direction_class"] for row in train_rows], minlength=3).astype(float) + 1.0
    prior = (counts / counts.sum()).tolist()
    binding = _bindings(config, manifest_path, assets, teacher_spec)
    trainables = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainables, lr=learning_rate)
    initial = trainable_state(student)
    tune_interval = evaluation_interval or max(1, max_steps // 4)
    params = {"max_steps": max_steps, "learning_rate": learning_rate, "batch_size": batch_size,
              "gradient_accumulation_steps": gradient_accumulation_steps, "seed": seed,
              "device": str(resolved), "loss_weights": weights, "selection_split": "tune",
              "selection_metric": selection_metric, "teacher_mode": teacher_mode,
              "auxiliary_budget": auxiliary_budget, "auxiliary_budget_definition": "sum_of_teacher_component_weights",
              "evaluation_interval": tune_interval, "evaluation_batch_size": evaluation_batch_size,
              "initial_trainable_state_sha256": _state_sha256(initial),
              "seeded_bitwise_reproducibility_asserted": False, "mechanism_control": mechanism_control}
    write_json(config, output / "run_started.json", {"schema": "distillation_training_started_v1",
                "bindings": binding, "parameters": params, "teacher_spec": teacher_spec})
    baseline, _ = evaluate_samples(student, tune_rows, context_length=config.context_length, teacher_spec=teacher_spec,
                    teacher_statistics=statistics, loss_weights=weights, device=resolved, direction_prior=prior,
                    batch_size=evaluation_batch_size, include_base=True)
    trace, tune_trace = [], [{"optimizer_step": 0, "metrics": baseline}]
    best_loss, best_step = _selection_value(baseline, selection_metric), 0
    checkpoint = output / "adapter_checkpoint.pt"
    def save_checkpoint(step):
        payload = {"schema": "distilled_kronos_lora_checkpoint_v1", "bindings": binding,
                   "teacher_spec": teacher_spec, "teacher_statistics": statistics,
                   "direction_train_prior": prior, "parameters": params, "optimizer_step": step,
                   "trainable_state": trainable_state(student), "optimizer_state": copy.deepcopy(optimizer.state_dict()),
                   "torch_rng_state": torch.get_rng_state(),
                   "qualification": "research_ablation_pending; no_model_promotion"}
        _atomic_checkpoint(config, checkpoint, payload)
    save_checkpoint(0)
    # Fixed train-only subset keeps diagnostics comparable and bounded. This
    # subset is never used to select checkpoints or tune the auxiliary budget.
    diagnostic_indices = np.linspace(0, len(train_rows) - 1, min(64, len(train_rows)), dtype=np.int64).tolist()
    diagnostic_rows = [train_rows[index] for index in diagnostic_indices]
    diagnostic_batch = _batch(diagnostic_rows[:batch_size], teacher_spec, statistics, resolved, config.context_length)
    initial_gradient = gradient_diagnostics(student, diagnostic_batch, context_length=config.context_length,
                                            loss_weights=weights)
    initial_train, _ = evaluate_samples(student, diagnostic_rows, context_length=config.context_length,
        teacher_spec=teacher_spec, teacher_statistics=statistics, loss_weights=weights, device=resolved,
        direction_prior=prior, batch_size=evaluation_batch_size, include_base=False)
    order, position = list(range(len(train_rows))), 0
    rng.shuffle(order)
    sample_order_digest = hashlib.sha256()
    for step in range(1, max_steps + 1):
        student.train()
        student.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        component_totals = {}
        for _ in range(gradient_accumulation_steps):
            indices = []
            for _ in range(batch_size):
                if position == len(order):
                    rng.shuffle(order)
                    position = 0
                indices.append(order[position])
                position += 1
            sample_order_digest.update(np.asarray(indices, dtype=np.int64).tobytes())
            batch = _batch([train_rows[index] for index in indices], teacher_spec, statistics, resolved, config.context_length)
            total, components, _ = forward_objective(student, batch, context_length=config.context_length,
                                                      loss_weights=weights)
            (total / gradient_accumulation_steps).backward()
            for name, value in {"total": total, **components}.items():
                component_totals[name] = component_totals.get(name, 0.) + float(value.detach().cpu()) / gradient_accumulation_steps
        norm = torch.nn.utils.clip_grad_norm_(trainables, 1.0, error_if_nonfinite=True)
        optimizer.step()
        trace.append({"optimizer_step": step, "losses": component_totals, "gradient_norm_before_clip": float(norm.cpu())})
        if step % 128 == 0 and step % tune_interval != 0 and step != max_steps:
            print(f"distillation optimizer_step={step}/{max_steps} train_loss={component_totals['total']:.6g}", file=sys.stderr, flush=True)
        if step % tune_interval == 0 or step == max_steps:
            metrics, _ = evaluate_samples(student, tune_rows, context_length=config.context_length, teacher_spec=teacher_spec,
                            teacher_statistics=statistics, loss_weights=weights, device=resolved, direction_prior=prior,
                            batch_size=evaluation_batch_size, include_base=True)
            tune_trace.append({"optimizer_step": step, "metrics": metrics})
            print(f"distillation optimizer_step={step}/{max_steps} train_loss={component_totals['total']:.6g} "
                  f"tune_direction_log_loss={metrics['direction']['log_loss']:.6g}", file=sys.stderr, flush=True)
            if _selection_value(metrics, selection_metric) < best_loss:
                best_loss, best_step = _selection_value(metrics, selection_metric), step
                save_checkpoint(step)
    final = trainable_state(student)
    changed = [name for name in final if not torch.equal(final[name], initial[name])]
    if not changed or best_step is None:
        raise RuntimeError("Training produced no changed adapter/head tensors or selected checkpoint")
    final_train, _ = evaluate_samples(student, diagnostic_rows, context_length=config.context_length,
        teacher_spec=teacher_spec, teacher_statistics=statistics, loss_weights=weights, device=resolved,
        direction_prior=prior, batch_size=evaluation_batch_size, include_base=False)
    final_gradient = gradient_diagnostics(student, diagnostic_batch, context_length=config.context_length,
                                          loss_weights=weights)
    selected_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restore_trainable_state(student, selected_payload["trainable_state"])
    selected_train, _ = evaluate_samples(student, diagnostic_rows, context_length=config.context_length,
        teacher_spec=teacher_spec, teacher_statistics=statistics, loss_weights=weights, device=resolved,
        direction_prior=prior, batch_size=evaluation_batch_size, include_base=False)
    summary = {"schema": "distillation_training_run_v1", "status": "completed_research_pilot",
               "wall_seconds": time.perf_counter() - wall_start,
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
               "completed_at": datetime.now(timezone.utc).isoformat(), "training_started": True,
               "bindings": binding, "parameters": params, "teacher_spec": teacher_spec,
               "direction_train_prior": prior,
               "teacher_statistics": statistics, "tokenizer_prefix_check": prefix_check,
               "split_counts": {name: len(rows) for name, rows in groups.items()},
               "optimizer_steps": max_steps, "selected_optimizer_step": best_step,
               "selection_split": "tune", "calibration_policy_holdout_used_for_fitting": False,
               "selection_metric": selection_metric, "selected_tune_direction_log_loss": best_loss,
               "initial_trainable_state_sha256": params["initial_trainable_state_sha256"],
               "sample_order_sha256": sample_order_digest.hexdigest(),
               "sample_order_hash_definition": "ordered_int64_train_row_indices_for_all_optimization_microbatches",
               "trainable_parameter_count": sum(parameter.numel() for parameter in trainables),
               "total_parameter_count": sum(parameter.numel() for parameter in student.parameters()),
               "changed_tensor_count": len(changed), "changed_tensor_names": changed,
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
               "checkpoint_contents": "LoRA_and_internal_heads_plus_optimizer; external_base_and_tokenizer_not_copied",
               "trace": trace, "tune_trace": tune_trace,
               "train_diagnostics": {"fit_split": "train", "selection_use": False,
                   "subset": "up_to_64_evenly_spaced_train_rows_in_dataset_order", "row_indices": diagnostic_indices,
                   "sample_count": len(diagnostic_rows),
                   "initial": initial_train, "final_optimizer": final_train, "selected_checkpoint": selected_train},
               "gradient_diagnostics": {"initial": initial_gradient, "final_optimizer": final_gradient},
               "parameter_movement": {"final_optimizer": _parameter_movement(initial, final),
                   "selected_checkpoint": _parameter_movement(initial, selected_payload["trainable_state"])},
               "qualification": "pending; small retrospective integration pilot only", "promoted": False}
    write_json(config, output / "training_summary.json", summary)
    return summary


def evaluate(config: ProjectConfig, *, dataset: Path, checkpoint: Path, output_dir: Path,
             split: str = "holdout", device: str = "auto", batch_size: int = 4) -> dict:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Evaluation batch_size must be a positive integer")
    if split not in SPLIT_NAMES and split != "all":
        raise ValueError("Evaluation split must be one chronological stage or all non-training stages")
    from ..data.dataset import load_dataset
    manifest_path = _manifest_path(dataset)
    from .token_cache import load_model_dataset
    _, samples = load_model_dataset(config, manifest_path)
    teacher_spec = teacher_specification(samples)
    resolved = resolve_device(device)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_relative_to(config.root):
        raise ValueError("Adapter checkpoint must be inside token_distillation")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") != "distilled_kronos_lora_checkpoint_v1":
        raise ValueError("Unsupported adapter checkpoint schema")
    current_assets = asset_identity(config)
    binding = _bindings(config, manifest_path, current_assets, teacher_spec)
    if payload["bindings"] != binding or payload["teacher_spec"] != teacher_spec:
        raise ValueError("Checkpoint bindings changed: dataset/config/assets/schema/runtime must match exactly")
    import json
    training_summary = json.loads((checkpoint.parent / "training_summary.json").read_text())
    checkpoint_sha256 = sha256_file(checkpoint)
    if training_summary["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Adapter checkpoint hash does not match its completed training summary")
    output = _new_output(config, output_dir)
    backbone, tokenizer, _ = load_official_assets(config, device=resolved)
    student = make_student(config, backbone, teacher_spec)
    if payload["parameters"].get("mechanism_control") == "frozen_backbone":
        for parameter in student.backbone.parameters(): parameter.requires_grad_(False)
    restore_trainable_state(student, payload["trainable_state"])
    names = SPLIT_NAMES[1:] if split == "all" else (split,)
    selected = [row for row in samples if row["split"] in names]
    encoded, prefix_check = encode_samples(tokenizer, selected, config, resolved)
    reports, predictions = {}, []
    for name in names:
        report, rows = evaluate_samples(student, [row for row in encoded if row["split"] == name],
                    context_length=config.context_length, teacher_spec=teacher_spec,
                    teacher_statistics=payload["teacher_statistics"], loss_weights=payload["parameters"]["loss_weights"],
                    device=resolved, direction_prior=payload["direction_train_prior"], include_base=True,
                    batch_size=batch_size)
        reports[name] = report
        predictions.extend(rows)
    result = {"schema": "distillation_evaluation_run_v1", "status": "completed_research_pilot",
              "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256,
              "bindings": binding, "device": str(resolved), "selection_split": "tune",
              "selection_metric": payload["parameters"].get("selection_metric"),
              "teacher_mode": payload["parameters"].get("teacher_mode"), "batch_size": batch_size,
              "selected_optimizer_step": payload["optimizer_step"], "evaluation_splits": list(names),
              "stages": reports, "tokenizer_prefix_check": prefix_check,
              "fit_or_calibration_performed": False, "policy_or_trading_backtest_performed": False,
              "qualification": "pending; retrospective_small_sample_integration_only",
              "output_semantics": "student_internal_predictions_not_native_CZSC_confirmed_facts", "promoted": False}
    write_json(config, output / "predictions.json", {"schema": "distillation_direction_predictions_v1", "rows": predictions})
    write_json(config, output / "evaluation_summary.json", result)
    return result
