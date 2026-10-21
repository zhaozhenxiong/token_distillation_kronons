"""Synthetic subprocess stubs: orchestration never trains or opens source data."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from token_distillation.artifacts import sha256_file
from token_distillation.config import load_config
from token_distillation.data import dataset as dataset_module
from token_distillation.training import ablation


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/ablation.toml")
    config = replace(config, root=tmp_path, state_root=tmp_path / "artifacts/source_runtime")
    dataset_path = tmp_path / "data/synthetic/manifest.json"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text('{"fixture":"synthetic_only"}')
    manifest = {"context_length": config.context_length, "horizon": config.horizon,
        "config": deepcopy(config.values),
        "split_ranges": {name: config.values["splits"][name] for name in
                         ("train", "tune", "calibration", "policy", "holdout")},
        "splits": {name: {"row_count": 2} for name in
                   ("train", "tune", "calibration", "policy", "holdout")}}
    rows = [{"split": name} for name in manifest["splits"] for _ in range(2)]
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *_: (manifest, rows))
    return config, dataset_path, manifest


def install_children(monkeypatch, *, mismatch=None):
    calls = []

    def child(config, arguments, stdout_path, stderr_path, label):
        calls.append(arguments)
        output = Path(arguments[arguments.index("--output-dir") + 1])
        output.mkdir(parents=True)
        if arguments[0] == "train":
            arm = arguments[arguments.index("--teacher-mode") + 1]
            seed = arguments[arguments.index("--seed") + 1]
            steps = int(arguments[arguments.index("--max-steps") + 1])
            checkpoint = output / "adapter_checkpoint.pt"
            checkpoint.write_bytes(b"synthetic_checkpoint_without_model_tensors")
            summary = {"optimizer_steps": steps, "selected_optimizer_step": steps // 2,
                "parameters": {"teacher_mode": arm}, "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "initial_trainable_state_sha256": "initial_" + seed,
                "sample_order_sha256": "order_" + seed}
            if arm == "czsc" and mismatch in ("initialization", "sample_order"):
                key = "initial_trainable_state_sha256" if mismatch == "initialization" else "sample_order_sha256"
                summary[key] = "mismatched_" + seed
            if mismatch == "budget":
                summary["optimizer_steps"] -= 1
            (output / "training_summary.json").write_text(json.dumps(summary))
        else:
            checkpoint = Path(arguments[arguments.index("--checkpoint") + 1])
            (output / "evaluation_summary.json").write_text(json.dumps({
                "checkpoint_sha256": "different" if mismatch == "checkpoint" else sha256_file(checkpoint)}))

    monkeypatch.setattr(ablation, "_run_child", child)
    return calls


def test_plan_is_deterministic_and_creates_no_outputs(experiment, monkeypatch):
    config, _, _ = experiment
    before = sorted(path.relative_to(config.root) for path in config.root.rglob("*"))
    monkeypatch.setattr(ablation.subprocess, "Popen", lambda *_, **__: pytest.fail("plan started a child"))
    first = ablation.ablation_plan(config, seeds=(3, 4), max_steps=8)
    assert first == ablation.ablation_plan(config, seeds=(3, 4), max_steps=8)
    assert first["selection_metric"] == "direction_log_loss"
    assert first["teacher_groups"]["none"] == [] and first["base_kl"] == 0
    assert first["training_started"] is False
    assert before == sorted(path.relative_to(config.root) for path in config.root.rglob("*"))


def test_equal_options_and_complete_four_arm_seed_grid(experiment, monkeypatch):
    config, dataset_path, _ = experiment
    calls = install_children(monkeypatch)
    result = ablation.run_ablation(config, dataset=dataset_path, output_dir=Path("artifacts/synthetic"),
        seeds=(3, 4), max_steps=8, batch_size=2, evaluation_interval=4)
    assert result["status"] == "completed"
    assert len(calls) == 16 and len(result["runs"]) == 8
    assert {(row["arm"], row["seed"]) for row in result["runs"]} == {
        (arm, seed) for arm in ablation.ARMS for seed in (3, 4)}
    assert result["split_counts"] == {name: 2 for name in ("train", "tune", "calibration", "policy", "holdout")}
    for arguments in calls:
        if arguments[0] == "train":
            for name, value in (("--max-steps", "8"), ("--batch-size", "2"),
                                ("--gradient-accumulation-steps", "1"),
                                ("--selection-metric", "direction_log_loss"),
                                ("--auxiliary-budget", "0.6")):
                assert arguments[arguments.index(name) + 1] == value
        else:
            assert arguments[arguments.index("--split") + 1] == "all"
    assert result["paired_initializations_verified"] and result["paired_sample_orders_verified"]


@pytest.mark.parametrize("mismatch,expected", [
    ("initialization", "initialization or consumed"),
    ("sample_order", "initialization or consumed"),
    ("budget", "fixed budget/mode"),
    ("checkpoint", "checkpoint differs"),
])
def test_arm_mismatch_rejects_and_records_failed_experiment(experiment, monkeypatch, mismatch, expected):
    config, dataset_path, _ = experiment
    calls = install_children(monkeypatch, mismatch=mismatch)
    with pytest.raises(ValueError, match=expected):
        ablation.run_ablation(config, dataset=dataset_path, output_dir=Path("artifacts/synthetic"),
            seeds=(3,), max_steps=8)
    record = json.loads((config.root / "artifacts/synthetic/experiment_manifest.json").read_text())
    assert record["status"] == "failed" and record["error"]
    assert record["completed_at"] >= record["created_at"]
    assert len(calls) < 8


def test_shape_mismatch_fails_before_child_or_experiment_output(experiment, monkeypatch):
    config, dataset_path, manifest = experiment
    manifest["context_length"] += 1
    monkeypatch.setattr(ablation, "_run_child", lambda *_, **__: pytest.fail("mismatch started a child"))
    with pytest.raises(ValueError, match="shape differs"):
        ablation.run_ablation(config, dataset=dataset_path, output_dir=Path("artifacts/synthetic"), seeds=(3,))
    assert not (config.root / "artifacts/synthetic").exists()


@pytest.mark.parametrize("section", ("data", "splits", "teacher", "ict"))
def test_dataset_contract_mismatch_fails_before_children(experiment, monkeypatch, section):
    config, dataset_path, manifest = experiment
    manifest["config"][section]["synthetic_mismatch"] = True
    monkeypatch.setattr(ablation, "_run_child", lambda *_, **__: pytest.fail("mismatch started a child"))
    with pytest.raises(ValueError, match=f"Dataset {section} contract differs"):
        ablation.run_ablation(config, dataset=dataset_path, output_dir=Path("artifacts/synthetic"), seeds=(3,))
    assert not (config.root / "artifacts/synthetic").exists()


def test_plan_requires_explicit_no_kl_and_distinct_seeds(experiment):
    config, _, _ = experiment
    values = deepcopy(config.values)
    values["training"]["loss_weights"].pop("base_kl")
    with pytest.raises(ValueError, match="explicit base_kl=0"):
        ablation.ablation_plan(replace(config, values=values))
    with pytest.raises(ValueError, match="distinct"):
        ablation.ablation_plan(config, seeds=(3, 3))
