"""Explicit, sequential, equal-budget experiment orchestration.

Planning imports no model runtime. Execution uses a fresh process per fit/eval,
so official weights and accelerator allocations cannot leak between arms.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from ..artifacts import sha256_file, write_json
from ..config import ProjectConfig

ARMS = ("none", "czsc", "ict", "dual")


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ablation_plan(config: ProjectConfig, *, seeds=(42, 123, 2026), max_steps=512,
                  batch_size=4, auxiliary_budget=0.6, learning_rate=1e-4,
                  evaluation_interval=128, evaluation_batch_size=4) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed <= 2**32 - 1 for seed in seeds):
        raise ValueError("Seeds must be distinct nonnegative integer seeds")
    for name, value in (("max_steps", max_steps), ("batch_size", batch_size),
                        ("evaluation_interval", evaluation_interval), ("evaluation_batch_size", evaluation_batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    import math
    if (isinstance(auxiliary_budget, bool) or not isinstance(auxiliary_budget, (int, float))
            or not math.isfinite(auxiliary_budget) or auxiliary_budget <= 0
            or isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Auxiliary budget and learning rate must be positive finite numbers")
    if config.values["training"].get("loss_weights", {}).get("base_kl") != 0:
        raise ValueError("The controlled ablation requires explicit base_kl=0 in every arm")
    protocol = {
        "schema": "controlled_distillation_ablation_protocol_v1",
        "arms": list(ARMS), "seeds": list(seeds), "max_steps": max_steps,
        "batch_size": batch_size, "gradient_accumulation_steps": 1,
        "learning_rate": learning_rate, "auxiliary_budget": auxiliary_budget,
        "base_kl": 0.0, "selection_metric": "direction_log_loss", "selection_split": "tune",
        "evaluation_interval": evaluation_interval, "evaluation_batch_size": evaluation_batch_size,
        "selection_includes_initialization": True,
        "teacher_groups": {"none": [], "czsc": ["w32", "czsc"], "ict": ["ict"], "dual": ["w32", "czsc", "ict"]},
        "teacher_budget_policy": "equal_total_auxiliary_budget_across_active_arms; dual_splits_between_CZSC_and_ICT",
        "common_objective": ["ar_s1", "ar_s2", "direction"],
        "paired_controls": ["dataset", "seed", "initial_trainable_tensors", "training_sample_order", "LoRA_capacity", "all_internal_heads", "optimizer_budget"],
        "primary_outcome": "holdout_direction_log_loss_difference_vs_same_seed_no_teacher",
        "secondary_outcomes": ["direction_Brier", "accuracy", "AR_conditional_NLL", "teacher_fidelity_vs_train_constants"],
        "uncertainty": "seed_variation_and_paired_date_block_bootstrap; stocks_at_same_origin_stay_together",
        "context_length": config.context_length, "horizon": config.horizon,
        "split_ranges": config.values["splits"],
        "config_sha256": sha256_file(config.config_path),
        "retrospective_only": True, "adaptation_using_holdout": False,
        "training_started": False, "model_promotion": False,
    }
    return {**protocol, "protocol_sha256": _hash(protocol)}


def _run_child(config: ProjectConfig, arguments: list[str], stdout_path: Path, stderr_path: Path, label: str) -> None:
    command = [sys.executable, str(config.root / "run.py"), "--config", str(config.config_path), *arguments]
    stdout_path, stderr_path = config.output_path(stdout_path), config.output_path(stderr_path)
    with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
        child = subprocess.Popen(command, cwd=config.root, stdout=stdout, stderr=subprocess.PIPE, text=True, bufsize=1)
        try:
            assert child.stderr is not None
            for line in child.stderr:
                stderr.write(line)
                stderr.flush()
                print(f"[{label}] {line.rstrip()}", file=sys.stderr, flush=True)
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    if code:
        raise RuntimeError(f"{label} exited with code {code}; inspect {stderr_path}")


def run_ablation(config: ProjectConfig, *, dataset: Path, output_dir: Path, device="mps", **options) -> dict:
    if config.values["training"]["execution_enabled"] is not True:
        raise ValueError("Ablation execution requires execution_enabled=true")
    protocol = ablation_plan(config, **options)
    from ..data.dataset import load_dataset
    dataset_path = config.output_path(dataset)
    if dataset_path.is_dir():
        dataset_path = config.output_path(dataset_path / "manifest.json")
    from .token_cache import load_model_dataset
    manifest, rows = load_model_dataset(config, dataset_path)
    if manifest["context_length"] != config.context_length or manifest["horizon"] != config.horizon:
        raise ValueError("Dataset shape differs from ablation configuration")
    for section in ("data", "splits", "teacher", "ict"):
        if manifest["config"][section] != config.values[section]:
            raise ValueError(f"Dataset {section} contract differs from the frozen ablation protocol")
    if any(not any(row["split"] == split for row in rows) for split in ("train", "tune", "calibration", "policy", "holdout")):
        raise ValueError("Ablation needs all five nonempty chronological stages")
    output = config.output_path(output_dir)
    if output.exists():
        raise FileExistsError(f"Ablation output already exists; select a new experiment directory: {output}")
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    record = {"schema": "distillation_ablation_experiment_v1", "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
        "dataset": str(dataset_path), "dataset_manifest_sha256": sha256_file(dataset_path),
        "config": str(config.config_path), "device": device,
        "orchestrator_sha256": sha256_file(Path(__file__)),
        "split_counts": {name: shard["row_count"] for name, shard in manifest["splits"].items()},
        "prior_pilot": {"dataset": str(config.root / "data/pilot_20261019_c128"),
            "run_dir": str(config.root / "artifacts/runs/pilot_20261019_c128"),
            "report_dir": str(config.root / "artifacts/reports/pilot_20261019_c128")},
        "runs": []}
    manifest_path = output / "experiment_manifest.json"
    write_json(config, manifest_path, record)
    del rows  # Parent must not retain the training population while a child runs.
    initial_by_seed, order_by_seed = {}, {}
    try:
        # Interleave arms within seeds; do not inspect holdout to set budgets.
        for seed in protocol["seeds"]:
            for arm in ARMS:
                label = f"{arm}_seed{seed}"
                run_dir, report_dir = output / label, output / (label + "_evaluation")
                entry = {"arm": arm, "seed": seed, "run_dir": str(run_dir),
                         "report_dir": str(report_dir), "status": "running"}
                record["runs"].append(entry)
                write_json(config, manifest_path, record, replace=True)
                print(f"ablation starting {label} steps={protocol['max_steps']}", file=sys.stderr, flush=True)
                _run_child(config, ["train", "--dataset", str(dataset_path), "--output-dir", str(run_dir),
                    "--device", device, "--teacher-mode", arm, "--max-steps", str(protocol["max_steps"]),
                    "--learning-rate", str(protocol["learning_rate"]), "--batch-size", str(protocol["batch_size"]),
                    "--gradient-accumulation-steps", "1", "--seed", str(seed),
                    "--auxiliary-budget", str(protocol["auxiliary_budget"]),
                    "--selection-metric", "direction_log_loss", "--evaluation-interval", str(protocol["evaluation_interval"]),
                    "--evaluation-batch-size", str(protocol["evaluation_batch_size"])],
                    output / "logs" / (label + "_train.json"), output / "logs" / (label + "_train.log"), label)
                training_path = run_dir / "training_summary.json"
                training = json.loads(training_path.read_text())
                if training["optimizer_steps"] != protocol["max_steps"] or training["parameters"]["teacher_mode"] != arm:
                    raise ValueError("Ablation arm did not complete the fixed budget/mode")
                init, order = training["initial_trainable_state_sha256"], training["sample_order_sha256"]
                if seed in initial_by_seed and (initial_by_seed[seed] != init or order_by_seed[seed] != order):
                    raise ValueError("Matched seed arms differ in initialization or consumed training sample order")
                initial_by_seed[seed], order_by_seed[seed] = init, order
                _run_child(config, ["evaluate", "--dataset", str(dataset_path), "--checkpoint", training["checkpoint"],
                    "--output-dir", str(report_dir), "--split", "all", "--device", device,
                    "--batch-size", str(protocol["evaluation_batch_size"])],
                    output / "logs" / (label + "_eval.json"), output / "logs" / (label + "_eval.log"), label + " eval")
                evaluation_path = report_dir / "evaluation_summary.json"
                evaluation = json.loads(evaluation_path.read_text())
                if evaluation["checkpoint_sha256"] != training["checkpoint_sha256"]:
                    raise ValueError("Ablation evaluation checkpoint differs from training")
                entry.update(status="completed", optimizer_steps=training["optimizer_steps"],
                    selected_optimizer_step=training["selected_optimizer_step"],
                    training_summary_sha256=sha256_file(training_path), evaluation_summary_sha256=sha256_file(evaluation_path),
                    checkpoint_sha256=training["checkpoint_sha256"],
                    initial_trainable_state_sha256=init, sample_order_sha256=order)
                write_json(config, manifest_path, record, replace=True)
                print(f"ablation completed {label}", file=sys.stderr, flush=True)
        record.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat(),
            paired_initializations_verified=True, paired_sample_orders_verified=True, promoted=False)
    except BaseException as exc:
        if record["runs"] and record["runs"][-1]["status"] == "running":
            record["runs"][-1].update(status="failed", error=str(exc))
        record.update(status="failed", error=str(exc), completed_at=datetime.now(timezone.utc).isoformat())
        write_json(config, manifest_path, record, replace=True)
        raise
    write_json(config, manifest_path, record, replace=True)
    return record
