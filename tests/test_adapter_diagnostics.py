"""Synthetic CPU algebra and integrity checks; never train or load real assets."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from token_distillation.artifacts import sha256_file

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_ablation_adapters.py"
SPEC = importlib.util.spec_from_file_location("adapter_diagnostics_script", SCRIPT)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


@pytest.mark.parametrize("rank,other_rank", [(1, 1), (2, 3), (4, 2)])
def test_low_rank_gram_norms_match_dense_products(rank, other_rank):
    rng = torch.Generator().manual_seed(24)
    a = torch.randn(rank, 7, generator=rng, dtype=torch.float64)
    b = torch.randn(5, rank, generator=rng, dtype=torch.float64)
    other_a = torch.randn(other_rank, 7, generator=rng, dtype=torch.float64)
    other_b = torch.randn(5, other_rank, generator=rng, dtype=torch.float64)
    actual = 2.5 * b @ a
    reference = 1.3 * other_b @ other_a
    assert diagnostics.low_rank_frobenius_squared(a, b, 2.5) == pytest.approx(float(actual.square().sum()), rel=1e-12)
    assert diagnostics.low_rank_difference_squared(a, b, 2.5, other_a, other_b, 1.3) == pytest.approx(
        float((actual - reference).square().sum()), rel=1e-12)
    assert diagnostics.low_rank_difference_squared(a, b, 2.5, a, b, 2.5) == 0.
    assert diagnostics.low_rank_frobenius_squared(a, b * 0, 2.5) == 0.


def test_low_rank_shapes_and_nonfinite_rejected():
    with pytest.raises(ValueError, match="compatible"):
        diagnostics.low_rank_frobenius_squared(torch.ones(2, 3), torch.ones(3, 4))
    with pytest.raises(ValueError, match="finite"):
        diagnostics.low_rank_frobenius_squared(torch.full((2, 3), float("nan")), torch.ones(3, 2))
    with pytest.raises(ValueError, match="identical dense shapes"):
        diagnostics.low_rank_difference_squared(torch.ones(2, 3), torch.ones(3, 2), 1.,
                                               torch.ones(2, 4), torch.ones(3, 2), 1.)


def _synthetic_experiment(tmp_path):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    config_path = tmp_path / "config.toml"
    config_path.write_text("synthetic_identity_only=true")
    base_path = tmp_path / "synthetic_frozen_base" / "model.safetensors"
    base_path.parent.mkdir()
    targets = [f"transformer.0.self_attn.{name}_proj" for name in ("q", "v")]
    save_file({name + ".weight": torch.eye(3) for name in targets}, str(base_path))
    dataset = tmp_path / "data" / "manifest.json"
    write(dataset, {"synthetic": True})
    config = SimpleNamespace(root=tmp_path, config_path=config_path, kronos_root=base_path.parent,
                             values={"student": {"lora_rank": 2, "lora_alpha": 4.}})
    binding = {"config_sha256": sha256_file(config_path), "dataset_manifest_sha256": sha256_file(dataset),
               "assets": {"files": {"model/model.safetensors": {"path": str(base_path), "sha256": sha256_file(base_path)}}}}
    manifest = {"schema": "distillation_ablation_experiment_v1", "status": "completed",
        "protocol": {"arms": list(diagnostics.ARMS), "seeds": [7], "config_sha256": sha256_file(config_path)},
        "config": str(config_path), "dataset": str(dataset), "dataset_manifest_sha256": sha256_file(dataset), "runs": []}
    directory = tmp_path / "artifacts" / "experiment"
    for index, arm in enumerate(diagnostics.ARMS):
        run_dir = directory / arm
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "adapter_checkpoint.pt"
        state = {}
        for name in targets:
            state["backbone." + name + ".lora_A"] = torch.full((2, 3), 0.1)
            state["backbone." + name + ".lora_B"] = torch.full((3, 2), (index + 1) * .01)
        state["direction_head.weight"] = torch.ones(3, 3) * (1 + index * .01)
        state["continuous_head.weight"] = torch.ones(2, 3) * (1 + index * .1)
        params = {"teacher_mode": arm, "seed": 7, "initial_trainable_state_sha256": "synthetic_init"}
        step = 2 if arm == "dual" else 1
        payload = {"schema": "distilled_kronos_lora_checkpoint_v1", "bindings": binding,
                   "parameters": params, "optimizer_step": step, "trainable_state": state}
        torch.save(payload, checkpoint)
        summary = {"bindings": binding, "parameters": params, "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint), "selected_optimizer_step": step,
            "initial_trainable_state_sha256": "synthetic_init", "sample_order_sha256": "synthetic_order"}
        summary_path = run_dir / "training_summary.json"
        write(summary_path, summary)
        manifest["runs"].append({"arm": arm, "seed": 7, "run_dir": str(run_dir), "status": "completed",
            "training_summary_sha256": sha256_file(summary_path), "checkpoint_sha256": sha256_file(checkpoint),
            "selected_optimizer_step": step, "initial_trainable_state_sha256": "synthetic_init",
            "sample_order_sha256": "synthetic_order"})
    write(directory / "experiment_manifest.json", manifest)
    return directory, manifest, config


def test_complete_synthetic_analysis_has_backbone_and_head_deltas_and_step_confound(tmp_path):
    directory, manifest, config = _synthetic_experiment(tmp_path)
    checkpoint_paths = [Path(row["run_dir"]) / "adapter_checkpoint.pt" for row in manifest["runs"]]
    hashes = [sha256_file(path) for path in checkpoint_paths]
    result = diagnostics.diagnose_adapters(directory, config=config)
    rows = {row["arm"]: row for row in result["rows"]}
    assert result["optimizer_steps_performed"] == 0
    assert not result["gpu_used"] and not result["database_accessed"]
    assert rows["none"]["effective_lora_difference_vs_none_relative_to_base_qv"] == 0.
    assert rows["ict"]["effective_lora_difference_vs_none_relative_to_base_qv"] > 0.
    assert rows["ict"]["head_differences_vs_none"]["direction_head"]["difference_l2_vs_none"] > 0.
    assert rows["ict"]["selected_steps_match"]
    assert rows["dual"]["step_confound"] is not None
    assert not rows["dual"]["selected_steps_match"]
    assert rows["dual"]["projection_count"] == 2
    assert [sha256_file(path) for path in checkpoint_paths] == hashes


def test_partial_grid_and_changed_checkpoint_or_base_are_rejected(tmp_path):
    directory, manifest, config = _synthetic_experiment(tmp_path)
    partial = deepcopy(manifest)
    partial["runs"].pop()
    with pytest.raises(ValueError, match="Partial"):
        diagnostics.validate_complete_grid(partial)
    partial = deepcopy(manifest)
    partial["status"] = "running"
    with pytest.raises(ValueError, match="completed experiment"):
        diagnostics.validate_complete_grid(partial)
    checkpoint = Path(manifest["runs"][0]["run_dir"]) / "adapter_checkpoint.pt"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="Checkpoint SHA"):
        diagnostics.diagnose_adapters(directory, config=config)
    checkpoint.write_bytes(original)
    base = config.kronos_root / "model.safetensors"
    base.write_bytes(base.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="base asset SHA"):
        diagnostics.diagnose_adapters(directory, config=config)


def test_paired_initialization_digest_mismatch_is_rejected(tmp_path):
    directory, manifest, config = _synthetic_experiment(tmp_path)
    row = manifest["runs"][1]
    summary_path = Path(row["run_dir"]) / "training_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["sample_order_sha256"] = "changed_order"
    summary_path.write_text(json.dumps(summary))
    row["sample_order_sha256"] = "changed_order"
    row["training_summary_sha256"] = sha256_file(summary_path)
    (directory / "experiment_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="paired sample_order_sha256 mismatch"):
        diagnostics.diagnose_adapters(directory, config=config)
