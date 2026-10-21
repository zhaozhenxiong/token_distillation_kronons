"""Offline, paired ablation analysis; this module never loads a model or source DB.

Uncertainty resamples contiguous origin-date blocks, keeping all stocks and all
seeds in a date together.  Its interval is conditional on the requested seed set;
seed variability is separately reported, never disguised as independent rows.
"""
from __future__ import annotations

from collections import defaultdict
import html
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path

import numpy as np

from ..artifacts import sha256_file, write_json
from ..config import SPLIT_NAMES
from .metrics import direction_metrics

ARMS = ("none", "czsc", "ict", "dual")
CLASS_NAMES = ("down", "flat", "up")


def _mean_std(values) -> dict:
    values = [float(v) for v in values if v is not None]
    return {"mean": float(np.mean(values)) if values else None,
            "std_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "values": values, "seed_count": len(values)}


def _key(row: dict) -> tuple:
    return row["symbol"], row["as_of"]


def prediction_diagnostics(rows: list[dict]) -> dict:
    """Three-class calibration and collapse diagnostics with explicit denominators."""
    if any(type(row["direction_class"]) is not int or row["direction_class"] not in range(3) for row in rows):
        raise ValueError("Prediction targets must be integer class IDs 0..2")
    p = np.asarray([row["direction_probabilities"] for row in rows], dtype=float).reshape(-1, 3)
    y = np.asarray([row["direction_class"] for row in rows], dtype=int)
    result = direction_metrics(p, y)
    if not len(rows):
        return result
    predicted = p.argmax(1)
    confusion = np.zeros((3, 3), dtype=int)
    np.add.at(confusion, (y, predicted), 1)
    confidence, correct = p.max(1), predicted == y
    bins, ece = [], 0.
    for index in range(10):
        mask = (confidence >= index / 10) & (confidence < (index + 1) / 10)
        if index == 9:
            mask |= confidence == 1.
        count = int(mask.sum())
        gap = abs(float(confidence[mask].mean() - correct[mask].mean())) if count else None
        ece += (count / len(y)) * (gap or 0.)
        bins.append({"lower": index / 10, "upper": (index + 1) / 10, "count": count,
                     "confidence": float(confidence[mask].mean()) if count else None,
                     "accuracy": float(correct[mask].mean()) if count else None})
    classes = []
    for index, name in enumerate(CLASS_NAMES):
        actual, chosen, true_positive = int(confusion[index].sum()), int(confusion[:, index].sum()), int(confusion[index, index])
        precision = true_positive / chosen if chosen else None
        recall = true_positive / actual if actual else None
        classes.append({"class": index, "name": name, "actual": actual, "predicted": chosen,
                        "precision": precision, "recall": recall,
                        "f1": 2 * true_positive / (actual + chosen) if actual + chosen else None})
    probability_std = p.std(0)
    result.update(accuracy=float(correct.mean()), class_names=list(CLASS_NAMES),
                  confusion_matrix_actual_rows_predicted_columns=confusion.tolist(), per_class=classes,
                  actual_class_counts=np.bincount(y, minlength=3).tolist(),
                  predicted_class_counts=np.bincount(predicted, minlength=3).tolist(),
                  mean_probability=p.mean(0).tolist(), std_probability=probability_std.tolist(),
                  mean_confidence=float(confidence.mean()),
                  mean_entropy_nats=float((-p * np.log(np.clip(p, 1e-12, 1.))).sum(1).mean()),
                  maximum_entropy_nats=math.log(3), expected_calibration_error_10_bins=ece,
                  calibration_bins=bins, predicted_class_dominance=float(np.bincount(predicted, minlength=3).max() / len(y)),
                  single_predicted_class=bool(len(np.unique(predicted)) == 1),
                  nearly_constant_probabilities=bool(probability_std.max() < 0.01),
                  balanced_accuracy=float(np.mean([item["recall"] for item in classes if item["recall"] is not None])))
    return result


def date_block_interval(values, dates, *, block_length: int = 2,
                        resamples: int = 2000, seed: int = 1729) -> dict:
    """Percentile moving-block interval for paired losses [seeds, sample rows].

    Blocks are noncircular contiguous *observed origin dates*.  Every row within
    a sampled date is retained, and one sampled date sequence is shared by seeds.
    With irregular origins, block length is measured in dates rather than days.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None]
    if (values.ndim != 2 or values.shape[1] != len(dates) or not values.size
            or not np.isfinite(values).all()):
        raise ValueError("Expected finite paired values [seeds,samples] and one date per sample")
    if type(block_length) is not int or block_length < 1 or type(resamples) is not int or resamples < 100:
        raise ValueError("Positive integer block length and at least 100 resamples required")
    dates = [str(date)[:10] for date in dates]
    unique = sorted(set(dates))
    date_index = {date: index for index, date in enumerate(unique)}
    indices = np.asarray([date_index[date] for date in dates])
    sums = np.zeros((values.shape[0], len(unique)))
    counts = np.bincount(indices, minlength=len(unique))
    for index in range(values.shape[0]):
        np.add.at(sums[index], indices, values[index])
    result = {"mean": float(values.mean(1).mean()), "date_clusters": len(unique),
              "row_count": len(dates), "seed_count": values.shape[0],
              "block_length_origins": block_length, "resamples": resamples,
              "method": "paired_non_circular_moving_origin_date_block_percentile_bootstrap",
              "estimand": "mean_of_seed_specific_row_mean_loss_differences",
              "all_symbols_and_seeds_share_resampled_date_sequence": True,
              "seed_uncertainty": "conditional_on_fixed_seed_set; see separately reported seed SD",
              "nominal_interval": 0.95, "ci95": None,
              "approximate_nonoverlapping_blocks": len(unique) / block_length,
              "adequate_independent_block_support": len(unique) >= 20 * block_length,
              "formal_significance_or_generalization_claim": False}
    if len(unique) <= block_length:
        result["warning"] = "too_few_origin_dates_for_a_meaningful_block_interval"
        return result
    generator = np.random.default_rng(seed)
    starts = generator.integers(0, len(unique) - block_length + 1,
                               size=(resamples, math.ceil(len(unique) / block_length)))
    sampled = (starts[..., None] + np.arange(block_length)).reshape(resamples, -1)[:, :len(unique)]
    denominator = counts[sampled].sum(1)
    estimates = (sums[:, sampled].sum(2) / denominator[None]).mean(0)
    result["ci95"] = np.quantile(estimates, [.025, .975]).tolist()
    if not result["adequate_independent_block_support"]:
        result["warning"] = "few_effective_time_blocks; interval_is_descriptive_and_may_be_unstable"
    return result


def teacher_constant_baselines(train_rows: list[dict], rows: list[dict], statistics: dict) -> dict:
    """Known targets only, train-fitted constants; no unknown-as-zero conversion."""
    result = {}
    if any(row["split"] != "train" for row in train_rows):
        raise ValueError("Teacher constant baselines must be fitted on train only")
    for name, teacher in (rows[0]["teachers"].items() if rows else []):
        stats = statistics[name]
        absolute_sum, normalized_sum, median_absolute_sum, median_normalized_sum, valid = 0., 0., 0., 0., 0
        per_target, events = [], []
        for index, target_name in enumerate(teacher["continuous_names"]):
            train_known = [row["teachers"][name]["continuous_targets"][index] for row in train_rows
                           if row["teachers"][name]["continuous_mask"][index]]
            train_median = float(np.median(train_known)) if train_known else 0.
            known = [row["teachers"][name]["continuous_targets"][index] for row in rows
                     if row["teachers"][name]["continuous_mask"][index]]
            if any(value is None for value in known):
                raise ValueError("Known teacher target cannot be null")
            error_sum = sum(abs(value - stats["mean"][index]) for value in known)
            median_error_sum = sum(abs(value - train_median) for value in known)
            absolute_sum += error_sum
            normalized_sum += error_sum / stats["scale"][index]
            median_absolute_sum += median_error_sum
            median_normalized_sum += median_error_sum / stats["scale"][index]
            valid += len(known)
            per_target.append({"name": target_name, "valid": len(known), "constant": stats["mean"][index],
                               "train_valid": stats["valid_count"][index],
                               "mae_original_units": error_sum / len(known) if known else None,
                               "mae_normalized": error_sum / stats["scale"][index] / len(known) if known else None,
                               "train_median_constant": train_median,
                               "median_mae_original_units": median_error_sum / len(known) if known else None,
                               "median_mae_normalized": median_error_sum / stats["scale"][index] / len(known) if known else None})
        event_loss, event_valid = 0., 0
        for index, target_name in enumerate(teacher["event_names"]):
            fitted = [row["teachers"][name]["event_targets"][index] for row in train_rows
                      if row["teachers"][name]["event_mask"][index]]
            known = [row["teachers"][name]["event_targets"][index] for row in rows
                     if row["teachers"][name]["event_mask"][index]]
            if any(value is None for value in fitted + known):
                raise ValueError("Known teacher event cannot be null")
            prevalence = (sum(fitted) + 1.) / (len(fitted) + 2.)
            loss = sum(-value * math.log(prevalence) - (1 - value) * math.log(1 - prevalence) for value in known)
            event_loss += loss
            event_valid += len(known)
            events.append({"name": target_name, "train_valid": len(fitted), "train_positive": sum(fitted),
                           "train_laplace_prevalence": prevalence, "valid": len(known),
                           "positive": sum(known), "bce": loss / len(known) if known else None})
        result[name] = {"fit_split": "train", "unknown_targets": "excluded_by_explicit_mask",
                        "continuous_baseline": "training_mean_per_target",
                        "continuous_valid": valid,
                        "continuous_mae_original_units": absolute_sum / valid if valid else None,
                        "continuous_mae_normalized": normalized_sum / valid if valid else None,
                        "continuous_train_median_baseline": "train_known_target_median_per_target; optimal_constant_for_train_absolute_error",
                        "continuous_train_median_mae_original_units": median_absolute_sum / valid if valid else None,
                        "continuous_train_median_mae_normalized": median_normalized_sum / valid if valid else None,
                        "per_continuous_target": per_target,
                        "event_baseline": "training_per_target_prevalence_Laplace_1_1",
                        "event_valid": event_valid, "event_bce": event_loss / event_valid if event_valid else None,
                        "per_event_target": events}
    return result


def _teacher_comparison(measured: dict, baseline: dict) -> dict:
    result = {}
    for name, constant in baseline.items():
        metric = measured.get(name, {})
        item = {"student": metric, "train_fitted_constant": constant}
        for field in ("continuous_mae_normalized", "event_bce"):
            current, reference = metric.get(field), constant.get(field)
            item[field + "_minus_constant"] = current - reference if current is not None and reference is not None else None
        current, reference = metric.get("continuous_mae_normalized"), constant.get("continuous_train_median_mae_normalized")
        item["continuous_mae_normalized_minus_train_median"] = current - reference if current is not None and reference is not None else None
        per_target = {target["name"]: target for target in constant["per_continuous_target"]}
        item["per_continuous_target_minus_train_mean"] = [
            {"name": target["name"], "valid": target["valid"],
             "mae_original_units_difference": target["mae_original_units"] - per_target[target["name"]]["mae_original_units"]}
            for target in metric.get("per_continuous_target", [])
            if target.get("valid") and target["name"] in per_target and per_target[target["name"]]["valid"]]
        item["per_continuous_target_minus_train_median"] = [
            {"name": target["name"], "valid": target["valid"],
             "mae_original_units_difference": target["mae_original_units"] - per_target[target["name"]]["median_mae_original_units"]}
            for target in metric.get("per_continuous_target", [])
            if target.get("valid") and target["name"] in per_target and per_target[target["name"]]["valid"]]
        result[name] = item
    return result


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _losses(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray([row["direction_probabilities"] for row in rows], dtype=float)
    y = np.asarray([row["direction_class"] for row in rows], dtype=int)
    return (-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.)),
            np.square(p - np.eye(3)[y]).sum(1), (p.argmax(1) == y).astype(float))


def _paired_effect(values: list[np.ndarray], dates: list[str], *, block_length: int, resamples: int) -> dict:
    array = np.asarray(values)
    result = _mean_std(array.mean(1))
    result.update(lower_is_better=True, improved_seed_count=int((array.mean(1) < 0).sum()),
                  date_block_uncertainty=date_block_interval(array, dates, block_length=block_length, resamples=resamples))
    result["date_block_sensitivity"] = {str(length): date_block_interval(array, dates, block_length=length, resamples=resamples)
                                         for length in sorted({block_length, max(5, block_length)})}
    return result


def _grouped_diagnostics(rows: list[dict], field: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[field][:10] if field == "as_of" else row[field]].append(row)
    return [{field: key, **prediction_diagnostics(value)} for key, value in sorted(grouped.items())]


def _run_diagnostics(training: dict, evaluation: dict, rows: list[dict], train_rows: list[dict],
                     dataset_rows: list[dict]) -> dict:
    stages = {}
    for split, metrics in evaluation["stages"].items():
        selected = sorted([row for row in rows if row["split"] == split], key=_key)
        targets = [row for row in dataset_rows if row["split"] == split]
        baseline = teacher_constant_baselines(train_rows, targets, training["teacher_statistics"])
        losses = metrics.get("mean_losses", {})
        stages[split] = {"direction": prediction_diagnostics(selected),
                         "direction_train_prior_baseline": metrics["direction_train_prior_baseline"],
                         "teachers": _teacher_comparison(metrics.get("teachers", {}), baseline),
                         "ar_delta_vs_frozen_original": {
                             part: losses.get(part, 0.) - losses["frozen_base_" + part]
                             if "frozen_base_" + part in losses else None for part in ("ar_s1", "ar_s2")},
                         "ar_conditional_nll": {part: losses.get(part) for part in
                             ("ar_s1", "ar_s2", "frozen_base_ar_s1", "frozen_base_ar_s2")},
                         "ar_metric_semantics": "teacher_forced_future_tokens; not_free_running_price_forecasts",
                         "per_symbol": _grouped_diagnostics(selected, "symbol"),
                         "per_date": _grouped_diagnostics(selected, "as_of")}
    tune_trace = [{"optimizer_step": item["optimizer_step"],
                   "direction_log_loss": item["metrics"]["direction"]["log_loss"],
                   "weighted_total": item["metrics"]["mean_losses"].get("total")}
                  for item in training.get("tune_trace", [])]
    result = {"optimizer_steps": training["optimizer_steps"], "selected_optimizer_step": training["selected_optimizer_step"],
            "parameters": training["parameters"], "split_counts": training["split_counts"],
            "approximate_training_sample_presentations": training["optimizer_steps"] * training["parameters"]["batch_size"]
                * training["parameters"].get("gradient_accumulation_steps", 1),
            "tune_trace": tune_trace, "stages": stages}
    for name in ("train_diagnostics", "gradient_diagnostics", "parameter_movement", "initial_trainable_state_sha256", "sample_order_sha256"):
        if name in training:
            result[name] = training[name]
    train_diagnostics = training.get("train_diagnostics")
    if train_diagnostics and "row_indices" in train_diagnostics:
        subset = [train_rows[index] for index in train_diagnostics["row_indices"]]
        constant = teacher_constant_baselines(train_rows, subset, training["teacher_statistics"])
        result["teacher_training_trajectory"] = {
            "evaluation_scope": "fixed_train_only_diagnostic_subset; not_out_of_sample_acquisition",
            "sample_count": len(subset),
            "initial": _teacher_comparison(train_diagnostics["initial"]["teachers"], constant),
            "final_optimizer": _teacher_comparison(train_diagnostics["final_optimizer"]["teachers"], constant),
            "selected_checkpoint": _teacher_comparison(train_diagnostics["selected_checkpoint"]["teachers"], constant),
            "interpretation": "final_optimizer_learning_can_be_discarded_by_tune_direction_selection; selected_holdout_fidelity_does_not_prove_optimizer_learning_failure"}
    result["approximate_passes_over_training_population"] = result["approximate_training_sample_presentations"] / len(train_rows)
    return result


def diagnose_pilot(*, training: dict, evaluation: dict, rows: list[dict], dataset_rows: list[dict]) -> dict:
    """Evidence from an old integration pilot, without treating hypotheses as facts."""
    train_rows = [row for row in dataset_rows if row["split"] == "train"]
    diagnostics = _run_diagnostics(training, evaluation, rows, train_rows, dataset_rows)
    holdout = diagnostics["stages"].get("holdout", {})
    dates = len({row["as_of"][:10] for row in rows if row["split"] == "holdout"})
    findings = [
        {"kind": "fact", "finding": "small_training_budget_and_population",
         "evidence": {"optimizer_steps": training["optimizer_steps"], "train_samples": len(train_rows),
                      "sample_presentations": diagnostics["approximate_training_sample_presentations"]}},
        {"kind": "fact", "finding": "holdout_rows_are_not_independent_temporal_replicates",
         "evidence": {"holdout_rows": len([row for row in rows if row["split"] == "holdout"]), "origin_dates": dates}},
        {"kind": "fact", "finding": "selection_objective", "evidence": training["parameters"].get("selection_metric")},
        {"kind": "architectural_fact", "finding": "direction_head_is_a_new_supervised_linear_head",
         "implication": "direction_head_quality_is_not_the_original_Kronos_free_running_forecast_quality"},
        {"kind": "hypothesis", "finding": "auxiliary_gradient_competition_and_random_head_undertraining",
         "validation": "requires_matched_no_teacher_and_single_teacher_runs; loss_magnitude_alone_does_not_measure_gradient_conflict"}]
    if holdout:
        findings.append({"kind": "fact", "finding": "holdout_direction_log_loss_minus_train_prior",
                         "evidence": holdout["direction"]["log_loss"] - holdout["direction_train_prior_baseline"]["log_loss"]})
        findings.append({"kind": "fact", "finding": "holdout_prediction_shape", "evidence": {
            key: holdout["direction"].get(key) for key in ("single_predicted_class", "predicted_class_counts", "actual_class_counts",
                "mean_entropy_nats", "std_probability", "expected_calibration_error_10_bins")}})
    diagnostics.update(findings=findings, causal_attribution_proven=False,
                       conclusion="integration_pilot_cannot_establish_distillation_effectiveness")
    return diagnostics


def _verify_population(rows: list[dict], expected: dict) -> None:
    actual = {}
    for row in rows:
        identity = (row["split"], *_key(row))
        if identity in actual:
            raise ValueError("Duplicate prediction population identity")
        if identity not in expected or row["direction_class"] != expected[identity]["direction_class"]:
            raise ValueError("Prediction population or labels differ from frozen dataset")
        actual[identity] = row
    if set(actual) != set(expected):
        raise ValueError("Prediction population is selectively missing or contains extra rows")
    for split in {row["split"] for row in rows}:
        prediction_diagnostics([row for row in rows if row["split"] == split])


def _verify_teacher_statistics(statistics: dict, train_rows: list[dict]) -> None:
    """Verify saved scaling did use known train targets rather than later splits."""
    for name, stats in statistics.items():
        for index, mean in enumerate(stats["mean"]):
            values = [row["teachers"][name]["continuous_targets"][index] for row in train_rows
                      if row["teachers"][name]["continuous_mask"][index]]
            expected_mean = float(np.mean(values)) if values else 0.
            expected_scale = max(float(np.std(values)), 1e-5) if values else 1.
            if (stats["valid_count"][index] != len(values) or not math.isclose(mean, expected_mean, abs_tol=1e-8)
                    or not math.isclose(stats["scale"][index], expected_scale, rel_tol=1e-7, abs_tol=1e-8)):
                raise ValueError("Teacher statistics differ from known train-only targets")


def _verify_protocol_run(protocol: dict, run: dict, training: dict) -> None:
    parameters = training["parameters"]
    if "teacher_mode" in parameters and parameters["teacher_mode"] != run["arm"]:
        raise ValueError("Run teacher mode differs from requested protocol")
    requested_steps = protocol.get("max_steps", protocol.get("steps"))
    if requested_steps is not None and training["optimizer_steps"] != requested_steps:
        raise ValueError("Run optimizer steps differ from the predeclared budget")
    for field in ("batch_size", "gradient_accumulation_steps", "learning_rate", "auxiliary_budget", "evaluation_interval", "evaluation_batch_size"):
        if field in protocol and parameters.get(field) != protocol[field]:
            raise ValueError(f"Run {field} differs from the predeclared budget")
    weights = parameters.get("loss_weights")
    if weights is not None:
        expected = {"none": set(), "czsc": {"czsc", "w32"}, "ict": {"ict"}, "dual": {"czsc", "w32", "ict"}}[run["arm"]]
        active = {name.rsplit("_", 1)[0] for name, weight in weights.items()
                  if name.endswith(("_continuous", "_events")) and weight > 0}
        if expected != active:
            raise ValueError("Run active auxiliary branches differ from the requested teacher arm")
        budget = sum(value for name, value in weights.items() if name.endswith(("_continuous", "_events")))
        if not math.isclose(budget, parameters["auxiliary_budget"] if expected else 0., abs_tol=1e-8):
            raise ValueError("Enabled teachers must share the predeclared total auxiliary budget")
        if "base_kl" in protocol and weights.get("base_kl") != protocol["base_kl"]:
            raise ValueError("Run base KL weight differs from the predeclared common objective")


def _teacher_acquisition(diagnostics: dict, seeds: list, split: str = "holdout") -> dict:
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for teacher in ("czsc", "w32", "ict"):
            if not any(teacher in diagnostics[f"{arm}_seed{seed}"]["stages"].get(split, {}).get("teachers", {}) for seed in seeds):
                continue
            result[arm][teacher] = {}
            for metric in ("continuous_mae_normalized_minus_constant", "continuous_mae_normalized_minus_train_median", "event_bce_minus_constant"):
                values = [diagnostics[f"{arm}_seed{seed}"]["stages"][split]["teachers"].get(teacher, {}).get(metric) for seed in seeds]
                result[arm][teacher][metric] = _mean_std(values)
            for metric in ("continuous_mae_normalized", "event_bce"):
                differences = []
                for seed in seeds:
                    current = diagnostics[f"{arm}_seed{seed}"]["stages"][split]["teachers"].get(teacher, {}).get("student", {}).get(metric)
                    control = diagnostics[f"none_seed{seed}"]["stages"][split]["teachers"].get(teacher, {}).get("student", {}).get(metric)
                    differences.append(current - control if current is not None and control is not None else None)
                result[arm][teacher][metric + "_minus_matched_no_teacher_head"] = _mean_std(differences)
            result[arm][teacher]["lower_is_better"] = True
            result[arm][teacher]["semantics"] = "student_internal_reconstruction_of_known_origin_facts; not_future_trade_labels"
            result[arm][teacher]["uncertainty"] = "seed_SD_only; per_row_teacher_predictions_are_not_stored_for_date_block_intervals"
            result[arm][teacher]["checkpoint_scope"] = "tune_direction_selected_checkpoint; includes_step_zero_if_selected"
            result[arm][teacher]["optimizer_learning_failure_attribution"] = False
    return result


def _write_svg(config, path: Path, comparison: dict, *, comparator: str = "vs_none") -> None:
    """Small dependency-free, standalone scientific comparison figure."""
    data = comparison["stages"].get("holdout")
    if not data:
        return
    points = [(arm, data["comparisons"][arm][comparator]["log_loss"]) for arm in ARMS]
    label = "matched no-teacher" if comparator == "vs_none" else "train prior"
    extents = [0.]
    for _, effect in points:
        extents.extend(effect["date_block_uncertainty"]["ci95"] or [effect["mean"]])
        extents.extend(effect["values"])
    lo, hi = min(extents), max(extents)
    padding = max((hi - lo) * .2, .001)
    lo, hi = lo - padding, hi + padding
    transform = lambda value: 150 + (value - lo) / (hi - lo) * 560
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="820" height="320" viewBox="0 0 820 320">',
             '<rect width="820" height="320" fill="white"/>',
             '<g font-family="sans-serif" fill="#16263a">',
             f'<text x="30" y="30" font-size="18">Holdout log loss difference vs {label}</text>',
             '<text x="30" y="53" font-size="12">Negative is better. Large dot: mean. Small dots: seeds. Line: 95% date-block interval.</text>',
             f'<line x1="{transform(0):.2f}" x2="{transform(0):.2f}" y1="70" y2="246" stroke="#aaa" stroke-dasharray="4 4"/>']
    for index, (arm, effect) in enumerate(points):
        y = 92 + index * 43
        interval = effect["date_block_uncertainty"]["ci95"]
        color = "#20678e" if arm != "dual" else "#a95522"
        parts.append(f'<text x="30" y="{y + 5}" font-size="14">{arm}</text>')
        if interval:
            parts.append(f'<line x1="{transform(interval[0]):.2f}" x2="{transform(interval[1]):.2f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        seed_offsets = np.linspace(-8, 8, len(effect["values"]))
        for value, offset in zip(effect["values"], seed_offsets):
            parts.append(f'<circle cx="{transform(value):.2f}" cy="{y + offset:.1f}" r="2.8" fill="{color}" fill-opacity=".5"/>')
        parts.append(f'<circle cx="{transform(effect["mean"]):.2f}" cy="{y}" r="5" fill="{color}"/>')
        parts.append(f'<text x="730" y="{y + 5}" font-size="12">{effect["mean"]:+.4f}</text>')
    for value in np.linspace(lo, hi, 5):
        parts.append(f'<text x="{transform(value):.2f}" y="268" text-anchor="middle" font-size="11">{value:+.3f}</text>')
    parts.append('<text x="30" y="300" font-size="11">Retrospective development comparison. Few time blocks can make intervals unstable.</text></g></svg>')
    path = config.output_path(path)
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_figures(config, output: Path, comparison: dict) -> dict:
    """Use standard scientific plotting when available, without installing it."""
    names = (("ablation_comparison", "vs_none", "matched no-teacher"),
             ("ablation_prior_comparison", "vs_train_prior", "train prior"))
    if find_spec("matplotlib") is None:
        for name, comparator, _ in names:
            _write_svg(config, output / (name + ".svg"), comparison, comparator=comparator)
        return {"backend": "dependency_free_svg_fallback", "optional_dependency": "matplotlib_unavailable_in_analysis_runtime",
                "installation_performed": False, "primary": "ablation_comparison.svg", "secondary": "ablation_prior_comparison.svg"}
    cache = config.output_path("artifacts/source_runtime/matplotlib")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    for name, comparator, label in names:
        figure = Figure(figsize=(9.4, 4.4), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        data = comparison["stages"]["holdout"]["comparisons"]
        for index, arm in enumerate(ARMS):
            effect = data[arm][comparator]["log_loss"]
            interval = effect["date_block_uncertainty"]["ci95"]
            color = "#20678e" if arm != "dual" else "#a95522"
            if interval:
                axis.hlines(index, interval[0], interval[1], color=color, linewidth=2.5)
            axis.scatter(effect["values"], index + np.linspace(-.13, .13, len(effect["values"])),
                         s=19, color=color, alpha=.5, label="Individual seeds" if index == 0 else None)
            axis.scatter([effect["mean"]], [index], s=55, color=color, zorder=4,
                         label="Seed mean and 95% date-block interval" if index == 0 else None)
        axis.axvline(0., color=".6", linestyle="--", linewidth=1.)
        axis.set_yticks(range(4), ARMS)
        axis.invert_yaxis()
        axis.set_xlabel("Direction log loss difference (negative is better)")
        axis.set_title("Holdout comparison vs " + label)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="best", fontsize=8)
        figure.text(.5, .005, "Retrospective development; intervals conditional on fixed seeds. Few time blocks limit precision.",
                    ha="center", fontsize=8, color=".35")
        figure.savefig(config.output_path(output / (name + ".svg")), metadata={"Date": None})
        figure.savefig(config.output_path(output / (name + ".png")), dpi=180)
        figure.clear()
    return {"backend": "matplotlib_Agg", "version": matplotlib.__version__, "cache": str(cache),
            "primary": "ablation_comparison.png", "secondary": "ablation_prior_comparison.png",
            "vector_primary": "ablation_comparison.svg", "vector_secondary": "ablation_prior_comparison.svg"}


def _html_number(value, *, signed: bool = True) -> str:
    return "—" if value is None else format(value, "+.4f" if signed else ".4f")


def _html_seed_metric(metric: dict) -> str:
    value = _html_number(metric.get("mean"))
    deviation = metric.get("std_across_seeds")
    return value + (" ± " + format(deviation, ".4f") if deviation is not None else "")


def _html_evidence(comparison: dict) -> str:
    """Present already-computed evidence; never select or recompute an experiment."""
    parts = []
    holdout = comparison["stages"].get("holdout")
    if holdout:
        seeds = comparison["protocol"]["seeds"]
        parts.append(f'<h2>蒸馏是否带来额外预测收益</h2><p>留出段共有 {holdout["rows"]} 个股票窗口、{holdout["origin_dates"]} 个预测日期、{holdout["symbols"]} 只股票，使用 {len(seeds)} 个固定随机种子。主要证据是相对同种子无教师的配对 LogLoss 差值，负数表示改善。</p>')
        parts.append('<table><tr><th>教师组</th><th>Δ LogLoss ± 种子 SD</th><th>改善的种子数</th><th>2 原点块区间</th><th>5 原点块敏感性区间</th><th>可支持的解读</th></tr>')
        for arm in ("czsc", "ict", "dual"):
            effect = holdout["comparisons"][arm]["vs_none"]["log_loss"]
            primary = effect["date_block_uncertainty"]["ci95"]
            sensitivity = effect["date_block_sensitivity"].get("5", {}).get("ci95")
            interval = lambda value: "—" if value is None else f'[{value[0]:+.4f}, {value[1]:+.4f}]'
            interpretation = ("该回溯区间出现改善信号" if primary and primary[1] < 0 else
                              "该回溯区间出现退化信号" if primary and primary[0] > 0 else "额外预测收益尚未确认")
            parts.append(f'<tr><td>{arm}</td><td>{_html_seed_metric(effect)}</td><td>{effect["improved_seed_count"]}/{len(seeds)}</td><td>{interval(primary)}</td><td>{interval(sensitivity)}</td><td>{interpretation}</td></tr>')
        parts.append('</table><p>同日股票并非独立市场观测，固定种子用于检查优化敏感性。连续日期块的数量仍有限，区间为条件于这些固定种子的描述性区间，未校正多重比较；不能据此宣称正式样本外有效。</p>')
    acquisition = comparison.get("teacher_acquisition_holdout", {})
    if acquisition:
        names = {"czsc": "CZSC 结构", "w32": "CZSC 量价 W32", "ict": "ICT"}
        groups = comparison["protocol"].get("teacher_groups", {"none": [], "czsc": ["czsc", "w32"], "ict": ["ict"], "dual": ["czsc", "w32", "ict"]})
        parts.append('<h2>是否获取了教师知识：所选检查点的留出重建</h2><p>连续目标采用归一化 MAE，事件采用 BCE；均只统计已知目标。下表是模型减去训练段拟合的常数基线，负数表示更好，显示种子均值 ± SD。连续目标保守地要求优于训练中位数；仅优于均值或随机未训练头，不能证明学到了有用结构。事件基线为训练频率加 Laplace 平滑。</p>')
        parts.append('<table><tr><th>组 / 教师</th><th>该教师监督</th><th>连续 MAE Δ 训练均值</th><th>连续 MAE Δ 训练中位数</th><th>事件 BCE Δ 训练频率</th></tr>')
        for arm in ARMS:
            for teacher, metric in acquisition.get(arm, {}).items():
                parts.append(f'<tr><td>{arm} / {names.get(teacher, teacher)}</td><td>{"启用" if teacher in groups.get(arm, []) else "未启用（控制）"}</td><td>{_html_seed_metric(metric.get("continuous_mae_normalized_minus_constant", {}))}</td><td>{_html_seed_metric(metric.get("continuous_mae_normalized_minus_train_median", {}))}</td><td>{_html_seed_metric(metric.get("event_bce_minus_constant", {}))}</td></tr>')
        parts.append('</table><p>留出重建评价的是按调参段方向 LogLoss 选中的检查点。最终优化步可能学会训练子集上的教师重建，却因方向选择指标而没有被保留；训练获取、所选检查点的重建泛化与未来方向转移必须分别判断。</p>')
        parts.append('<details><summary>训练获取与检查点保留：逐种子轨迹</summary><p>括号内依次为初始 / 最终优化步 / 所选检查点。训练子集只用于诊断，不是新的留出证据。所有差值越小越好。</p><table><tr><th>组 / 种子 / 教师</th><th>训练连续 Δ 中位数（初 / 终 / 选）</th><th>训练事件 Δ 频率（初 / 终 / 选）</th><th>所选留出连续 Δ 中位数</th><th>所选留出事件 Δ 频率</th></tr>')
        for arm in ARMS:
            for seed in comparison["protocol"]["seeds"]:
                diagnostic = comparison.get("run_diagnostics", {}).get(f"{arm}_seed{seed}", {})
                trajectory = diagnostic.get("teacher_training_trajectory", {})
                held = diagnostic.get("stages", {}).get("holdout", {}).get("teachers", {})
                for teacher in groups.get(arm, []):
                    if not trajectory or teacher not in held:
                        continue
                    cells = []
                    for metric in ("continuous_mae_normalized_minus_train_median", "event_bce_minus_constant"):
                        cells.append(" / ".join(_html_number(trajectory.get(stage, {}).get(teacher, {}).get(metric))
                                                for stage in ("initial", "final_optimizer", "selected_checkpoint")))
                    parts.append(f'<tr><td>{arm} / {seed} / {names.get(teacher, teacher)}</td><td>{cells[0]}</td><td>{cells[1]}</td><td>{_html_number(held[teacher].get("continuous_mae_normalized_minus_train_median"))}</td><td>{_html_number(held[teacher].get("event_bce_minus_constant"))}</td></tr>')
        parts.append('</table></details>')
    if any("ar_conditional_nll" in value.get("stages", {}).get("holdout", {})
           for value in comparison.get("run_diagnostics", {}).values()):
        parts.append('<h2>Kronos 条件 AR 评分与官方冻结模型</h2><p>以下为固定种子间的描述性均值 ± SD，NLL 越低越好。评分使用未来 token 的 teacher forcing 与因果 s2 解码；它不是自由生成的价格走势预测，也不是交易回测。方向头与条件 AR 结果需要分别解读。</p><table><tr><th>组</th><th>AR s1 NLL</th><th>AR s2 NLL</th><th>官方冻结 s1</th><th>官方冻结 s2</th><th>Δ s1 相对官方</th><th>Δ s2 相对官方</th></tr>')
        for arm in ARMS:
            stages = [comparison.get("run_diagnostics", {}).get(f"{arm}_seed{seed}", {}).get("stages", {}).get("holdout", {})
                      for seed in comparison["protocol"]["seeds"]]
            cells = []
            for metric in ("ar_s1", "ar_s2", "frozen_base_ar_s1", "frozen_base_ar_s2"):
                cells.append(_html_seed_metric(_mean_std([stage.get("ar_conditional_nll", {}).get(metric) for stage in stages])))
            for metric in ("ar_s1", "ar_s2"):
                cells.append(_html_seed_metric(_mean_std([stage.get("ar_delta_vs_frozen_original", {}).get(metric) for stage in stages])))
            parts.append(f'<tr><td>{arm}</td>' + ''.join(f'<td>{value}</td>' for value in cells) + '</tr>')
        parts.append('</table>')
    selection = comparison.get("checkpoint_selection", {})
    if selection.get("selected_steps_by_arm_and_seed"):
        parts.append('<details><summary>各组实际选中的优化步数</summary><p>固定优化预算内按共同调参指标选择，允许第 0 步；比较的不是强制相同选中步数。</p><table><tr><th>组</th>' + ''.join(f'<th>seed {seed}</th>' for seed in comparison["protocol"]["seeds"]) + '</tr>')
        for arm, steps in selection["selected_steps_by_arm_and_seed"].items():
            parts.append(f'<tr><td>{arm}</td>' + ''.join(f'<td>{steps[str(seed)]}</td>' for seed in comparison["protocol"]["seeds"]) + '</tr>')
        parts.append('</table></details>')
    pilot = comparison.get("prior_pilot_diagnosis")
    if pilot and "holdout" in pilot["stages"]:
        held = pilot["stages"]["holdout"]
        metric, prior = held["direction"], held["direction_train_prior_baseline"]
        date_count = len(held.get("per_date", []))
        actual, predicted = metric["actual_class_counts"], metric["predicted_class_counts"]
        parts.append(f'<h2>原试跑弱于基线：已经观察到的事实</h2><p>旧试跑只有 {pilot["split_counts"]["train"]} 个训练窗口、{pilot["optimizer_steps"]} 次更新；留出 {metric["valid"]} 行仅来自 {date_count} 个预测日期。留出 LogLoss 为 {metric["log_loss"]:.4f}，训练先验为 {prior["log_loss"]:.4f}，差值 {metric["log_loss"] - prior["log_loss"]:+.4f}。实际下跌 {actual[0]} 次，模型预测下跌 {predicted[0]} 次。</p>')
        parts.append('<table><tr><th>旧试跑教师</th><th>留出连续 MAE Δ 训练均值</th><th>留出连续 MAE Δ 训练中位数</th><th>留出事件 BCE Δ 训练频率</th></tr>')
        for teacher, teacher_metric in held.get("teachers", {}).items():
            parts.append(f'<tr><td>{teacher}</td><td>{_html_number(teacher_metric.get("continuous_mae_normalized_minus_constant"))}</td><td>{_html_number(teacher_metric.get("continuous_mae_normalized_minus_train_median"))}</td><td>{_html_number(teacher_metric.get("event_bce_minus_constant"))}</td></tr>')
        old_selection = html.escape(str(pilot["parameters"].get("selection_metric")))
        parts.append(f'</table><p>旧检查点选择指标为 <code>{old_selection}</code>，方向头是新建的线性监督头，其指标不等于官方 Kronos 自由生成的价格预测。旧留出教师重建与常数基线的对比不足以支持已获取教师知识的结论。少量训练与辅助梯度竞争是机制假说，不能仅凭损失量级认定。扩大样本和更新步数的收益要看无教师组；蒸馏本身的额外收益要看本轮同种子配对比较。</p>')
    return "\n".join(parts)


def _write_html(config, path: Path, comparison: dict) -> None:
    figures = comparison.get("figures", {})
    pieces = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Distillation ablation</title>',
              '<style>body{font:15px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#172536}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #d8e0e6;padding:9px;text-align:right}th:first-child,td:first-child{text-align:left}small{color:#566}img{max-width:100%}</style>',
              '<h1>CZSC / ICT 蒸馏消融</h1><p>固定数据、时间切分、训练预算和调参选择指标。置信区间按连续预测原点日期块重采样，同日股票与固定随机种子共同保留。</p>',
              '<p>教师重建能力与未来方向预测收益分别检验；本结果属于回溯研发验证。各组使用相同训练预算，并按调参段方向 LogLoss 选择检查点，允许选择第 0 步，因此最终选择的训练步数可以不同。</p>',
              '<p>教师轨迹区分初始化、最终优化步与所选检查点的训练子集重建，以及所选检查点的留出段重建。所选检查点重建较差不能直接推断整个优化过程未学习教师。</p>']
    if comparison.get("dataset_derivation"):
        pieces.append('<p><strong>数据口径：</strong>本轮数据按未来标签区间的原始价格跳空规则派生。该信息在预测原点未知，属于事后质量筛选；比较只适用于满足该准入规则的数据，不能当作可在预测时实施的筛选策略。未做价格复权或确认公司行动原因。各阶段剔除数：' +
                      html.escape(json.dumps(comparison["dataset_derivation"].get("rejected_counts_by_split", {}), ensure_ascii=False)) + '</p>')
    pieces.append(_html_evidence(comparison))
    pieces.extend([f'<h2>主要比较图：教师相对同种子无教师</h2><img src="{figures.get("primary", "ablation_comparison.svg")}" alt="Holdout difference vs matched no-teacher">',
                   f'<h2>辅助比较图：相对训练先验</h2><img src="{figures.get("secondary", "ablation_prior_comparison.svg")}" alt="Holdout difference vs train prior">'])
    for split, stage in comparison["stages"].items():
        pieces.append(f'<h2>{html.escape(split)}</h2><table><tr><th>arm</th><th>LogLoss mean ± seed SD</th><th>Brier</th><th>Accuracy</th><th>Δ vs none [date-block 95%]</th></tr>')
        for arm in ARMS:
            metrics, effects = stage["arms"][arm], stage["comparisons"][arm]
            ll = metrics["log_loss"]
            effect = effects["vs_none"]["log_loss"]
            ci = effect["date_block_uncertainty"]["ci95"]
            interval = f'[{ci[0]:+.4f}, {ci[1]:+.4f}]' if ci else 'unavailable'
            sd = f'{ll["std_across_seeds"]:.4f}' if ll["std_across_seeds"] is not None else 'NA'
            pieces.append(f'<tr><td>{arm}</td><td>{ll["mean"]:.4f} ± {sd}</td><td>{metrics["brier"]["mean"]:.4f}</td><td>{metrics["accuracy"]["mean"]:.3f}</td><td>{effect["mean"]:+.4f} {interval}</td></tr>')
        pieces.append('</table>')
    pieces.append('<details><summary>完整解释字段与分析范围</summary><pre>' + html.escape(json.dumps(comparison["effectiveness"], ensure_ascii=False, indent=2)) + '</pre></details><p><a href="ablation_comparison.json">完整指标、逐股票/日期诊断、教师常数基线与审计绑定 JSON</a></p></html>')
    config.output_path(path).write_text("\n".join(pieces), encoding="utf-8")


def analyze_ablation(config, *, experiment_dir: Path, dataset: Path, output_dir: Path) -> dict:
    """Analyze a complete predeclared four-arm seed grid, refusing cherry-picking."""
    from ..data.dataset import load_dataset
    experiment_dir = config.output_path(experiment_dir)
    manifest_path = experiment_dir / "experiment_manifest.json"
    manifest = _read_json(manifest_path)
    if "status" in manifest and manifest["status"] != "completed":
        raise ValueError("Ablation experiment must complete before final comparison")
    protocol = manifest.get("protocol", manifest)
    seeds = protocol.get("seeds", manifest.get("seeds"))
    modes = protocol.get("modes", protocol.get("arms", list(ARMS)))
    if (not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds) or set(modes) != set(ARMS) or len(modes) != len(ARMS)):
        raise ValueError("Protocol must declare unique seeds and exactly none/czsc/ict/dual arms")
    requested = {(arm, seed) for arm in ARMS for seed in seeds}
    runs = manifest["runs"]
    pairs = [(run["arm"], run["seed"]) for run in runs]
    if set(pairs) != requested or len(pairs) != len(requested) or any(run.get("status") != "completed" for run in runs):
        raise ValueError("Complete requested four-arm x seed grid is required; incomplete/duplicate runs cannot be selected away")
    dataset_manifest_path = config.output_path(Path(dataset))
    if dataset_manifest_path.is_dir():
        dataset_manifest_path /= "manifest.json"
    dataset_manifest, samples = load_dataset(config, dataset_manifest_path)
    expected = {(row["split"], *_key(row)): row for row in samples if row["split"] != "train"}
    train_rows = [row for row in samples if row["split"] == "train"]
    counts = np.bincount([row["direction_class"] for row in train_rows], minlength=3).astype(float) + 1.
    prior = counts / counts.sum()
    dataset_hash = sha256_file(dataset_manifest_path)
    if "dataset_manifest_sha256" in manifest and manifest["dataset_manifest_sha256"] != dataset_hash:
        raise ValueError("Experiment and analyzer dataset manifests differ")
    for field in ("context_length", "horizon"):
        if field in protocol and protocol[field] != dataset_manifest.get(field):
            raise ValueError("Experiment protocol context/horizon differs from dataset")
    loaded, diagnostics, identities = {}, {}, []
    within_seed, common_parameters = {}, None
    shared_binding = None
    for run in runs:
        run_dir, report_dir = (config.output_path(Path(run[field])) for field in ("run_dir", "report_dir"))
        training, evaluation = _read_json(run_dir / "training_summary.json"), _read_json(report_dir / "evaluation_summary.json")
        rows = _read_json(report_dir / "predictions.json")["rows"]
        if training.get("bindings") != evaluation.get("bindings"):
            raise ValueError("Training/evaluation bindings differ")
        binding = training["bindings"]
        if binding["dataset_manifest_sha256"] != dataset_hash:
            raise ValueError("Run does not bind to the frozen comparison dataset")
        if shared_binding is None:
            shared_binding = binding
        elif binding != shared_binding:
            raise ValueError("Ablation runs must share config, source assets, runtime and dataset bindings")
        if training["checkpoint_sha256"] != evaluation["checkpoint_sha256"]:
            raise ValueError("Evaluation checkpoint differs from completed training artifact")
        if "checkpoint" in training and sha256_file(config.output_path(training["checkpoint"])) != training["checkpoint_sha256"]:
            raise ValueError("Completed checkpoint artifact hash changed")
        for field, path in (("training_summary_sha256", run_dir / "training_summary.json"),
                            ("evaluation_summary_sha256", report_dir / "evaluation_summary.json")):
            if field in run and run[field] != sha256_file(path):
                raise ValueError("Completed run summary hash differs from frozen experiment manifest")
        if training.get("selection_split") != "tune" or training["parameters"].get("selection_metric") not in {"direction_log_loss", "direction_logloss"}:
            raise ValueError("All ablation checkpoints must be selected on tune direction log loss")
        if training["parameters"]["seed"] != run["seed"]:
            raise ValueError("Run seed differs from requested protocol")
        _verify_protocol_run(protocol, run, training)
        _verify_teacher_statistics(training["teacher_statistics"], train_rows)
        params = training["parameters"]
        shared_parameters = {"optimizer_steps": training["optimizer_steps"], **{
            field: params.get(field) for field in ("learning_rate", "batch_size", "gradient_accumulation_steps", "evaluation_interval", "evaluation_batch_size")}}
        if "loss_weights" in params:
            shared_parameters["common_objective_weights"] = {key: params["loss_weights"].get(key)
                                                              for key in ("ar_s1", "ar_s2", "direction", "base_kl")}
        if common_parameters is None:
            common_parameters = shared_parameters
        elif common_parameters != shared_parameters:
            raise ValueError("All arms and seeds must share training and checkpoint-selection budgets")
        matching = {field: training.get(field, params.get(field)) for field in ("sample_order_sha256", "initial_trainable_state_sha256")}
        if run["seed"] in within_seed and within_seed[run["seed"]] != matching:
            raise ValueError("Matched arms must share initialization and sampled training rows within each seed")
        within_seed[run["seed"]] = matching
        expected_stages = {row["split"] for row in samples if row["split"] != "train"}
        if set(evaluation["stages"]) != expected_stages:
            raise ValueError("Evaluation must include all frozen non-training stages")
        _verify_population(rows, expected)
        for split, summary in evaluation["stages"].items():
            selected = [row for row in rows if row["split"] == split]
            actual = prediction_diagnostics(selected)
            if not math.isclose(actual["log_loss"], summary["direction"]["log_loss"], abs_tol=1e-6):
                raise ValueError("Reported direction log loss differs from saved predictions")
            baseline_ll = -np.log(prior[[row["direction_class"] for row in selected]]).mean()
            if not math.isclose(baseline_ll, summary["direction_train_prior_baseline"]["log_loss"], abs_tol=1e-6):
                raise ValueError("Run prior baseline is not the shared train-only Laplace prior")
        pair = run["arm"], run["seed"]
        loaded[pair] = {split: sorted([row for row in rows if row["split"] == split], key=_key) for split in SPLIT_NAMES[1:]}
        diagnostics[f'{run["arm"]}_seed{run["seed"]}'] = _run_diagnostics(training, evaluation, rows, train_rows, samples)
        identities.append({**run, "training_summary_sha256": sha256_file(run_dir / "training_summary.json"),
                           "evaluation_summary_sha256": sha256_file(report_dir / "evaluation_summary.json"),
                           "predictions_sha256": sha256_file(report_dir / "predictions.json")})
    stride = max(1, int(dataset_manifest.get("stride", 1)))
    block_length = max(2, math.ceil(dataset_manifest["horizon"] / stride))
    resamples = int(protocol.get("bootstrap_resamples", 2000))
    stages = {}
    for split in SPLIT_NAMES[1:]:
        reference = loaded[("none", seeds[0])][split]
        if not reference:
            continue
        dates = [row["as_of"] for row in reference]
        target = np.asarray([row["direction_class"] for row in reference])
        prior_loss = -np.log(prior[target])
        prior_brier = np.square(prior - np.eye(3)[target]).sum(1)
        arm_metrics, comparisons = {}, {}
        arrays = {arm: [_losses(loaded[(arm, seed)][split]) for seed in seeds] for arm in ARMS}
        for arm in ARMS:
            arm_metrics[arm] = {metric: _mean_std([value[index].mean() for value in arrays[arm]])
                                for index, metric in enumerate(("log_loss", "brier", "accuracy"))}
            comparisons[arm] = {}
            for comparator in ("none", "train_prior"):
                pair = {}
                for index, metric in enumerate(("log_loss", "brier")):
                    differences = [arrays[arm][seed_index][index] -
                                   (arrays["none"][seed_index][index] if comparator == "none" else (prior_loss, prior_brier)[index])
                                   for seed_index in range(len(seeds))]
                    pair[metric] = _paired_effect(differences, dates, block_length=block_length, resamples=resamples)
                comparisons[arm]["vs_" + comparator] = pair
        # The dual/single contrasts identify whether combining teachers helps beyond one teacher.
        comparisons["dual"]["vs_single_teachers"] = {
            arm: _paired_effect([arrays["dual"][index][0] - arrays[arm][index][0] for index in range(len(seeds))],
                                dates, block_length=block_length, resamples=resamples) for arm in ("czsc", "ict")}
        stages[split] = {"rows": len(reference), "origin_dates": len(set(date[:10] for date in dates)),
                         "symbols": len(set(row["symbol"] for row in reference)),
                         "actual_class_counts": np.bincount(target, minlength=3).tolist(),
                         "train_prior": prior.tolist(), "prior_log_loss": float(prior_loss.mean()),
                         "prior_brier": float(prior_brier.mean()), "arms": arm_metrics, "comparisons": comparisons}
    effectiveness = {"formal_qualified": False, "retrospective_development_only": True,
                     "teacher_acquisition": "separate_final_optimizer_train_only_fidelity_from_tune_selected_checkpoint_train_and_holdout_fidelity; require_beating_train_median_MAE_for_conservative_continuous_acquisition; inactive_heads_are_controls",
                     "selected_checkpoint_fidelity_does_not_establish_optimizer_learning_failure": True,
                     "selection_estimand": "best_by_common_tune_direction_log_loss_within_equal_optimizer_budget; step_zero_is_eligible; selected_step_need_not_match_between_arms",
                     "teacher_acquisition_train_trajectory_path": "run_diagnostics.<arm_seed>.teacher_training_trajectory",
                     "teacher_acquisition_holdout_path": "teacher_acquisition_holdout",
                     "predictive_transfer": {},
                     "inference_scope": "fixed_universe_and_chronology; no_preregistered_untouched_external_test; 95% intervals are descriptive, unadjusted for multiple comparisons",
                     "seed_variability": f"{len(seeds)} fixed seeds are optimizer sensitivity checks, not independent market histories"}
    for arm in ("czsc", "ict", "dual"):
        if "holdout" not in stages:
            break
        effect = stages["holdout"]["comparisons"][arm]["vs_none"]["log_loss"]
        interval = effect["date_block_uncertainty"]["ci95"]
        effectiveness["predictive_transfer"][arm] = (
            "retrospective_improvement_with_date_block_interval_below_zero" if interval and interval[1] < 0 else
            "retrospective_degradation_with_date_block_interval_above_zero" if interval and interval[0] > 0 else
            "not_established; interval_spans_zero_or_time_support_is_insufficient")
    effectiveness["block_length_sensitivity"] = {
        arm: {"log_loss_mean_minus_none": stages["holdout"]["comparisons"][arm]["vs_none"]["log_loss"]["mean"],
              "all_requested_seeds_improve": stages["holdout"]["comparisons"][arm]["vs_none"]["log_loss"]["improved_seed_count"] == len(seeds),
              "interval_below_zero_for_all_predeclared_block_lengths": all(
                  item["ci95"] is not None and item["ci95"][1] < 0
                  for item in stages["holdout"]["comparisons"][arm]["vs_none"]["log_loss"]["date_block_sensitivity"].values())}
        for arm in ("czsc", "ict", "dual")} if "holdout" in stages else {}
    comparison = {"schema": "distillation_ablation_comparison_v1", "status": "completed_retrospective_comparison",
                  "experiment_manifest_sha256": sha256_file(manifest_path), "protocol": protocol,
                  "dataset_manifest_sha256": dataset_hash, "population_sha256": dataset_manifest.get("population_sha256"),
                  "holdout_claim": dataset_manifest.get("holdout_claim"), "runs": identities,
                  "uncertainty": {"block_length_origins": block_length, "origin_stride": stride,
                                  "sensitivity_block_lengths": sorted({block_length, max(5, block_length)}),
                                  "label_horizon_trading_sessions": dataset_manifest["horizon"],
                                  "block_length_policy": "max(2,ceil(horizon/stride)); preserves common market-date clustering",
                                  "iid_stock_row_bootstrap": False, "selection_split": "tune"},
                  "stages": stages, "run_diagnostics": diagnostics,
                  "teacher_acquisition_holdout": _teacher_acquisition(diagnostics, seeds),
                  "checkpoint_selection": {"equal_optimizer_budget": True, "selection_metric": "tune_direction_log_loss",
                      "step_zero_eligible": protocol.get("selection_includes_initialization", True),
                      "selected_steps_by_arm_and_seed": {arm: {str(seed): diagnostics[f"{arm}_seed{seed}"]["selected_optimizer_step"]
                                                                  for seed in seeds} for arm in ARMS},
                      "interpretation": "effect_of_teacher_objective_plus_common_tune_checkpoint_selection; not_a_fixed_selected_training_step_contrast"},
                  "matched_seed_initialization_and_sample_order": within_seed,
                  "effectiveness": effectiveness, "promoted": False}
    if dataset_manifest.get("derivation"):
        comparison["dataset_derivation"] = dataset_manifest["derivation"]
        comparison["effectiveness"]["dataset_admission_scope"] = {
            "estimand": "contrast_conditional_on_frozen_ex_post_future_label_gap_quality_admission",
            "information_known_at_prediction_origin": dataset_manifest["derivation"].get("policy", {}).get("known_at_prediction_origin"),
            "rejected_counts_by_split": dataset_manifest["derivation"].get("rejected_counts_by_split"),
            "selection_bias": dataset_manifest["derivation"].get("expost_quality_selection_bias"),
            "deployable_origin_time_screen_asserted": False,
            "price_adjustment_or_corporate_action_confirmation_asserted": False}
    pilot = manifest.get("prior_pilot")
    if pilot:
        _, pilot_samples = load_dataset(config, config.output_path(pilot["dataset"]))
        comparison["prior_pilot_diagnosis"] = diagnose_pilot(
            training=_read_json(config.output_path(pilot["run_dir"]) / "training_summary.json"),
            evaluation=_read_json(config.output_path(pilot["report_dir"]) / "evaluation_summary.json"),
            rows=_read_json(config.output_path(pilot["report_dir"]) / "predictions.json")["rows"], dataset_rows=pilot_samples)
    output = config.output_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "ablation_comparison.json").exists():
        raise FileExistsError("Ablation comparison already exists; use a new output directory")
    comparison["figures"] = _write_figures(config, output, comparison)
    write_json(config, output / "ablation_comparison.json", comparison)
    _write_html(config, output / "index.html", comparison)
    return comparison
