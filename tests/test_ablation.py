"""Synthetic analysis only: no source database, model, optimizer or network."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from token_distillation.artifacts import sha256_file
from token_distillation.evaluation.ablation import (analyze_ablation, date_block_interval,
    prediction_diagnostics, teacher_constant_baselines, _verify_population, _verify_protocol_run, _run_diagnostics, _write_html)
from token_distillation.evaluation.metrics import direction_metrics


def prediction(symbol, date, y, p, split="holdout"):
    return {"symbol": symbol, "as_of": date, "split": split,
            "direction_class": y, "direction_probabilities": p}


def test_class_collapse_calibration_and_confusion_are_explicit():
    rows = [prediction("A", "2025-01-01", y, [.1, .2, .7]) for y in [0, 1, 2]]
    result = prediction_diagnostics(rows)
    assert result["confusion_matrix_actual_rows_predicted_columns"] == [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
    assert result["single_predicted_class"]
    assert result["nearly_constant_probabilities"]
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["expected_calibration_error_10_bins"] == pytest.approx(.7 - 1 / 3)
    assert sum(bin_["count"] for bin_ in result["calibration_bins"]) == 3


def test_date_blocks_preserve_shared_market_shocks_instead_of_iid_rows():
    dates = [f"2025-01-{index + 1:02}" for index in range(12)]
    shocks = np.array([-1.] * 6 + [1.] * 6)
    one_stock = date_block_interval(shocks, dates, block_length=2, resamples=500)
    twenty_stocks = date_block_interval(np.repeat(shocks, 20), np.repeat(dates, 20), block_length=2, resamples=500)
    assert one_stock["ci95"] == pytest.approx(twenty_stocks["ci95"])
    assert twenty_stocks["date_clusters"] == 12
    assert twenty_stocks["row_count"] == 240
    assert not twenty_stocks["adequate_independent_block_support"]


def test_block_bootstrap_is_paired_across_seeds_and_constant_effect_is_exact():
    dates = [f"2025-01-{index + 1:02}" for index in range(10)]
    result = date_block_interval([[-.1] * 10, [-.2] * 10], dates, block_length=2, resamples=100)
    assert result["mean"] == pytest.approx(-.15)
    assert result["ci95"] == pytest.approx([-.15, -.15])
    assert result["seed_count"] == 2
    assert result["all_symbols_and_seeds_share_resampled_date_sequence"]
    insufficient = date_block_interval([1., 2.], dates[:2], block_length=2, resamples=100)
    assert insufficient["ci95"] is None


def sample(value, event, *, split="train", known=True):
    return {"split": split, "teachers": {"ict": {"continuous_names": ["distance"], "event_names": ["sweep"],
        "continuous_targets": [value], "continuous_mask": [known], "event_targets": [event], "event_mask": [known]}}}


def test_teacher_baselines_fit_train_only_and_unknowns_never_become_zero():
    train = [sample(1., 0.), sample(3., 1.), sample(None, None, known=False)]
    rows = [sample(4., 1., split="holdout"), sample(None, None, split="holdout", known=False)]
    stats = {"ict": {"mean": [2.], "scale": [2.], "valid_count": [2]}}
    result = teacher_constant_baselines(train, rows, stats)["ict"]
    assert result["continuous_valid"] == 1
    assert result["continuous_mae_normalized"] == 1.
    assert result["event_valid"] == 1
    assert result["event_bce"] == pytest.approx(np.log(2.))
    assert result["per_event_target"][0]["train_valid"] == 2
    with pytest.raises(ValueError, match="train only"):
        teacher_constant_baselines(rows, rows, stats)


def test_teacher_mae_median_sensitivity_does_not_reward_only_learning_a_constant():
    train = [sample(0., 0.), sample(0., 0.), sample(9., 1.)]
    held = [sample(0., 0., split="holdout")]
    stats = {"ict": {"mean": [3.], "scale": [1.], "valid_count": [3]}}
    result = teacher_constant_baselines(train, held, stats)["ict"]
    assert result["continuous_mae_normalized"] == 3.
    assert result["continuous_train_median_mae_normalized"] == 0.
    assert result["per_continuous_target"][0]["train_median_constant"] == 0.


def test_pairing_refuses_missing_duplicate_or_relabeled_predictions():
    row = prediction("A", "2025-01-01", 0, [.5, .2, .3])
    expected = {("holdout", "A", "2025-01-01"): row}
    _verify_population([row], expected)
    with pytest.raises(ValueError, match="selectively missing"):
        _verify_population([], expected)
    with pytest.raises(ValueError, match="Duplicate"):
        _verify_population([row, row], expected)
    changed = {**row, "direction_class": 2}
    with pytest.raises(ValueError, match="labels"):
        _verify_population([changed], expected)


def fixture_experiment(tmp_path, monkeypatch):
    root = tmp_path
    def output_path(path):
        result = (root / path).resolve()
        if not result.is_relative_to(root):
            raise ValueError("escaped test root")
        return result
    config = SimpleNamespace(root=root, output_path=output_path)
    dataset = root / "data"
    dataset.mkdir()
    dataset_path = dataset / "manifest.json"
    dataset_path.write_text('{"synthetic": true}')
    rows = []
    for index in range(6):
        row = {**sample(float(index), float(index % 2)), "symbol": "A",
               "as_of": f"2024-01-{index + 1:02}", "direction_class": index % 3}
        rows.append(row)
    for index in range(12):
        row = {**sample(float(index), float(index % 2), split="holdout"), "symbol": str(index % 2),
               "as_of": f"2025-01-{index // 2 + 1:02}", "direction_class": index % 3}
        rows.append(row)
    dataset_manifest = {"horizon": 5, "stride": 5, "holdout_claim": "retrospective", "population_sha256": "population"}
    monkeypatch.setattr("token_distillation.data.dataset.load_dataset", lambda *args: (dataset_manifest, rows))
    experiment = root / "experiment"
    experiment.mkdir()
    runs = []
    for arm in ("none", "czsc", "ict", "dual"):
        for seed in (42, 123):
            run_dir = root / arm / str(seed) / "run"
            report_dir = root / arm / str(seed) / "report"
            run_dir.mkdir(parents=True)
            report_dir.mkdir()
            p = [.2, .3, .5] if arm == "none" else [.3, .3, .4]
            predictions = [{**row, "direction_probabilities": p} for row in rows if row["split"] == "holdout"]
            binding = {"dataset_manifest_sha256": sha256_file(dataset_path), "config_sha256": "same", "runtime": "same"}
            training = {"parameters": {"seed": seed, "batch_size": 4, "selection_metric": "direction_log_loss"},
                "optimizer_steps": 16, "selected_optimizer_step": 8, "selection_split": "tune", "checkpoint_sha256": "checkpoint",
                "bindings": binding, "teacher_statistics": {"ict": {"mean": [2.5], "scale": [float(np.std(np.arange(6)))], "valid_count": [6]}},
                "split_counts": {"train": 6, "holdout": 12}, "tune_trace": []}
            targets = np.asarray([row["direction_class"] for row in predictions])
            measured = direction_metrics([p] * 12, targets)
            prior = direction_metrics([[1 / 3] * 3] * 12, targets)
            evaluation = {"bindings": binding, "checkpoint_sha256": "checkpoint", "stages": {"holdout": {
                "direction": measured, "direction_train_prior_baseline": prior, "mean_losses": {}, "teachers": {}}}}
            for path, value in ((run_dir / "training_summary.json", training),
                                (report_dir / "evaluation_summary.json", evaluation),
                                (report_dir / "predictions.json", {"rows": predictions})):
                path.write_text(json.dumps(value))
            runs.append({"arm": arm, "seed": seed, "run_dir": str(run_dir), "report_dir": str(report_dir), "status": "completed"})
    manifest = {"protocol": {"modes": ["none", "czsc", "ict", "dual"], "seeds": [42, 123], "bootstrap_resamples": 100}, "runs": runs}
    (experiment / "experiment_manifest.json").write_text(json.dumps(manifest))
    return config, experiment, dataset, root / "output", manifest


def test_complete_grid_offline_analyzer_exports_reproducible_comparison(tmp_path, monkeypatch):
    config, experiment, dataset, output, _ = fixture_experiment(tmp_path, monkeypatch)
    result = analyze_ablation(config, experiment_dir=experiment, dataset=dataset, output_dir=output)
    assert result["stages"]["holdout"]["rows"] == 12
    assert result["stages"]["holdout"]["origin_dates"] == 6
    assert result["stages"]["holdout"]["comparisons"]["none"]["vs_none"]["log_loss"]["mean"] == 0.
    assert not result["effectiveness"]["formal_qualified"]
    assert not result["uncertainty"]["iid_stock_row_bootstrap"]
    assert "<svg" in (output / "ablation_comparison.svg").read_text()
    assert "matched no-teacher" in (output / "ablation_comparison.svg").read_text()
    assert (output / "ablation_prior_comparison.svg").exists()
    assert "selected_step_need_not_match" in result["effectiveness"]["selection_estimand"]
    assert result["teacher_acquisition_holdout"]["ict"]["ict"]["optimizer_learning_failure_attribution"] is False
    assert (output / "index.html").exists()


def test_analyzer_refuses_cherry_picked_grid(tmp_path, monkeypatch):
    config, experiment, dataset, output, manifest = fixture_experiment(tmp_path, monkeypatch)
    manifest["runs"].pop()
    (experiment / "experiment_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Complete requested"):
        analyze_ablation(config, experiment_dir=experiment, dataset=dataset, output_dir=output)


def test_analyzer_refuses_changed_config_binding(tmp_path, monkeypatch):
    config, experiment, dataset, output, manifest = fixture_experiment(tmp_path, monkeypatch)
    for filename, key in (("training_summary.json", "run_dir"), ("evaluation_summary.json", "report_dir")):
        path = Path(manifest["runs"][-1][key]) / filename
        value = json.loads(path.read_text())
        value["bindings"]["config_sha256"] = "different"
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="share config"):
        analyze_ablation(config, experiment_dir=experiment, dataset=dataset, output_dir=output)


def test_analyzer_verifies_train_only_teacher_statistics(tmp_path, monkeypatch):
    config, experiment, dataset, output, manifest = fixture_experiment(tmp_path, monkeypatch)
    path = Path(manifest["runs"][-1]["run_dir"]) / "training_summary.json"
    value = json.loads(path.read_text())
    value["teacher_statistics"]["ict"]["mean"] = [999.]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="train-only targets"):
        analyze_ablation(config, experiment_dir=experiment, dataset=dataset, output_dir=output)


def test_fairness_checks_reject_hidden_teacher_and_different_step_budget():
    protocol = {"max_steps": 512, "auxiliary_budget": .6, "base_kl": 0.}
    training = {"optimizer_steps": 512, "parameters": {"teacher_mode": "none", "auxiliary_budget": .6,
        "loss_weights": {"ar_s1": 1., "ar_s2": 1., "direction": 1., "base_kl": 0., "ict_events": 0.}}}
    _verify_protocol_run(protocol, {"arm": "none"}, training)
    training["parameters"]["loss_weights"]["ict_events"] = .6
    with pytest.raises(ValueError, match="active auxiliary"):
        _verify_protocol_run(protocol, {"arm": "none"}, training)
    training["parameters"]["loss_weights"]["ict_events"] = 0.
    training["optimizer_steps"] = 1024
    with pytest.raises(ValueError, match="predeclared budget"):
        _verify_protocol_run(protocol, {"arm": "none"}, training)


def test_analyzer_refuses_unmatched_initializations(tmp_path, monkeypatch):
    config, experiment, dataset, output, manifest = fixture_experiment(tmp_path, monkeypatch)
    for index, run in enumerate(manifest["runs"]):
        path = Path(run["run_dir"]) / "training_summary.json"
        value = json.loads(path.read_text())
        value["initial_trainable_state_sha256"] = "same" if index < 7 else "changed"
        value["sample_order_sha256"] = "same"
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="share initialization"):
        analyze_ablation(config, experiment_dir=experiment, dataset=dataset, output_dir=output)


def test_prediction_diagnostics_rejects_fractional_or_boolean_classes():
    for label in (.5, True):
        with pytest.raises(ValueError, match="integer class"):
            prediction_diagnostics([prediction("A", "2025-01-01", label, [.2, .3, .5])])


def test_final_optimizer_teacher_learning_is_distinct_from_selected_holdout_fidelity():
    train = [sample(1., 0.), sample(3., 1.)]
    held = {**sample(4., 1., split="holdout"), **prediction("A", "2025-01-01", 0, [.4, .3, .3])}
    def teacher_metrics(mae):
        return {"ict": {"continuous_mae_normalized": mae, "event_bce": .8,
                        "per_continuous_target": [{"name": "distance", "valid": 2, "mae_original_units": mae}]}}
    training = {"optimizer_steps": 4, "selected_optimizer_step": 0, "parameters": {"batch_size": 1},
                "split_counts": {"train": 2, "holdout": 1}, "tune_trace": [],
                "teacher_statistics": {"ict": {"mean": [2.], "scale": [1.], "valid_count": [2]}},
                "train_diagnostics": {"row_indices": [0, 1], "sample_count": 2,
                    "initial": {"teachers": teacher_metrics(2.)},
                    "final_optimizer": {"teachers": teacher_metrics(.1)},
                    "selected_checkpoint": {"teachers": teacher_metrics(2.)}}}
    evaluation = {"stages": {"holdout": {"mean_losses": {}, "direction_train_prior_baseline": {}, "teachers": teacher_metrics(3.)}}}
    result = _run_diagnostics(training, evaluation, [held], train, train + [held])
    trajectory = result["teacher_training_trajectory"]
    assert trajectory["final_optimizer"]["ict"]["continuous_mae_normalized_minus_constant"] == pytest.approx(-.9)
    assert trajectory["selected_checkpoint"]["ict"]["continuous_mae_normalized_minus_constant"] == pytest.approx(1.)
    assert result["stages"]["holdout"]["teachers"]["ict"]["continuous_mae_normalized_minus_constant"] == pytest.approx(1.)
    assert "not_out_of_sample" in trajectory["evaluation_scope"]
    assert "does_not_prove_optimizer_learning_failure" in trajectory["interpretation"]


def test_future_label_quality_filter_is_reported_as_expost_admission(tmp_path, monkeypatch):
    config, experiment, dataset, output, _ = fixture_experiment(tmp_path, monkeypatch)
    from token_distillation.data.dataset import load_dataset
    manifest, _ = load_dataset(config, dataset)
    manifest["derivation"] = {"schema": "frozen_label_gap_dataset_derivation_v1", "parent_manifest_sha256": "parent",
        "policy": {"known_at_prediction_origin": False}, "rejected_counts_by_split": {"tune": 1},
        "expost_quality_selection_bias": "conditional_on_future_label_gap_admission"}
    result = analyze_ablation(config, experiment_dir=experiment, dataset=dataset, output_dir=output)
    admission = result["effectiveness"]["dataset_admission_scope"]
    assert admission["information_known_at_prediction_origin"] is False
    assert admission["deployable_origin_time_screen_asserted"] is False
    assert result["dataset_derivation"]["parent_manifest_sha256"] == "parent"
    assert "事后质量筛选" in (output / "index.html").read_text()


def test_report_surfaces_uncertain_transfer_teacher_baselines_and_pilot_facts(tmp_path, monkeypatch):
    config, experiment, dataset, output, _ = fixture_experiment(tmp_path, monkeypatch)
    result = analyze_ablation(config, experiment_dir=experiment, dataset=dataset, output_dir=output)
    result["prior_pilot_diagnosis"] = result["run_diagnostics"]["none_seed42"]
    target = output / "presentation_check.html"
    _write_html(config, target, result)
    report = target.read_text()
    assert "额外预测收益尚未确认" in report
    assert "连续 MAE Δ 训练中位数" in report
    assert "所选检查点的留出重建" in report
    assert "训练获取与检查点保留" in report
    assert "6 个预测日期" in report
    assert "2 个固定随机种子" in report
    assert "原试跑弱于基线：已经观察到的事实" in report
    assert "实际下跌 4 次，模型预测下跌 0 次" in report
    assert "Kronos 条件 AR 评分与官方冻结模型" in report
    assert "teacher forcing" in report
    assert report.index("蒸馏是否带来额外预测收益") < report.index("主要比较图")


def test_report_preserves_conditional_ar_numbers_and_separates_them_from_direction():
    train = [sample(1., 0.), sample(3., 1.)]
    held = {**sample(4., 1., split="holdout"), **prediction("A", "2025-01-01", 0, [.4, .3, .3])}
    training = {"optimizer_steps": 4, "selected_optimizer_step": 0, "parameters": {"batch_size": 1},
                "split_counts": {"train": 2, "holdout": 1}, "tune_trace": [],
                "teacher_statistics": {"ict": {"mean": [2.], "scale": [1.], "valid_count": [2]}}}
    losses = {"ar_s1": 2.8, "ar_s2": 3.1, "frozen_base_ar_s1": 3., "frozen_base_ar_s2": 3.05}
    evaluation = {"stages": {"holdout": {"mean_losses": losses, "direction_train_prior_baseline": {}, "teachers": {}}}}
    result = _run_diagnostics(training, evaluation, [held], train, train + [held])
    assert result["stages"]["holdout"]["ar_conditional_nll"] == losses
    assert result["stages"]["holdout"]["ar_delta_vs_frozen_original"] == pytest.approx({"ar_s1": -.2, "ar_s2": .05})
    assert "not_free_running_price_forecasts" in result["stages"]["holdout"]["ar_metric_semantics"]
