"""Synthetic component checks only; never run a training optimizer loop."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from token_distillation.models.runtime import (causal_decode_s2, frozen_base, make_student,
                                               resolve_device, restore_trainable_state, trainable_state)
from token_distillation.models.student import KronosStudent
from token_distillation.training.runner import (_batch, _loss_weights, _parameter_movement, _selection_value,
                                                _state_sha256, DEFAULT_LOSS_WEIGHTS, encode_samples,
                                                evaluate_samples, fit_teacher_statistics,
                                                forward_objective, gradient_diagnostics, teacher_specification)


class Attention(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.q_proj, self.v_proj = nn.Linear(size, size), nn.Linear(size, size)

    def forward(self, value):
        return self.q_proj(value) + self.v_proj(value)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()

    def forward(self, value):
        return value + self.self_attn(value).cumsum(1) / torch.arange(1, value.shape[1] + 1, device=value.device)[None, :, None]


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_dropout_p = 0.37

    def forward(self, query, context):
        return F.scaled_dot_product_attention(query[:, None], context[:, None], context[:, None],
                                              is_causal=self.training,
                                              dropout_p=self.attn_dropout_p if self.training else 0.)[:, 0]


class SyntheticBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 8
        self.embedding = nn.Embedding(16, 8)
        self.transformer = nn.ModuleList([Block()])
        self.head = nn.Linear(8, 16)
        self.dep_layer = nn.Module()
        self.dep_layer.cross_attn = CrossAttention()

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        value = self.embedding(s1_ids) + self.embedding(s2_ids)
        for block in self.transformer:
            value = block(value)
        return self.head(value), value

    def decode_s2(self, context, s1_ids, padding_mask=None):
        query = self.embedding(s1_ids)
        return self.head(self.dep_layer.cross_attn(query, context) + context)


class SyntheticTokenizer(nn.Module):
    def encode(self, inputs, half=False):
        ids = ((inputs[..., 0] * 3).round().long() % 16)
        return ids, (ids + 1) % 16


def samples():
    teacher = lambda name, c, e: {"schema": name + "_v1", "schema_sha256": name * 3,
        "continuous_names": [f"c{i}" for i in range(c)], "event_names": [f"e{i}" for i in range(e)],
        "continuous_targets": [float(i) for i in range(c)], "continuous_mask": [True] * c,
        "event_targets": [0.] * e, "event_mask": [True] * e}
    records = []
    for index in range(3):
        records.append({"symbol": "000001.SZ", "as_of": f"2025-01-0{index+1}", "split": "train",
            "normalized_context": [[float(t % 4)] * 6 for t in range(4)],
            "normalized_future": [[0.5] * 6, [0.2] * 6],
            "time_context": [[0, 15, 1, 2, 1]] * 4, "time_future": [[0, 15, 2, 3, 1]] * 2,
            "direction_class": index, "realized_return": (index - 1) / 100,
            "teachers": {"w32": teacher("w32", 3, 7), "ict": teacher("ict", 2, 6), "czsc": teacher("czsc", 16, 0)}})
    records[1]["teachers"]["ict"]["continuous_mask"][0] = False
    records[1]["teachers"]["ict"]["continuous_targets"][0] = None
    return records


def fixture():
    torch.manual_seed(20)
    rows = samples()
    spec = teacher_specification(rows)
    statistics = fit_teacher_statistics(rows, spec)
    student = KronosStudent(SyntheticBackbone(), 3, lora_rank=2,
                            extra_teacher_dims={"ict": (2, 6), "czsc": (16, 0)}).eval()
    config = SimpleNamespace(context_length=4, horizon=2)
    encoded, check = encode_samples(SyntheticTokenizer(), rows, config, torch.device("cpu"))
    return student, encoded, spec, statistics, check


def test_continuous_only_teacher_does_not_create_empty_linear_and_forwards():
    student, rows, _, _, _ = fixture()
    assert "event_logits" not in student.extra_teacher_heads["czsc"]
    output = student(rows[0]["_s1"][None, :4], rows[0]["_s2"][None, :4])
    assert output["extra_teachers"]["czsc"]["continuous"].shape == (1, 16)
    assert output["extra_teachers"]["czsc"]["event_logits"].shape == (1, 0)


def test_causal_s2_future_perturbation_cannot_change_prefix_and_flags_restored():
    backbone = SyntheticBackbone().eval()
    context = torch.randn(2, 7, 8)
    ids = torch.randint(0, 16, (2, 7))
    initial = causal_decode_s2(backbone, context, ids)
    modified = context.clone()
    modified[:, 4:] += 40
    modified_ids = ids.clone()
    modified_ids[:, 4:] = (modified_ids[:, 4:] + 7) % 16
    actual = causal_decode_s2(backbone, modified, modified_ids)
    torch.testing.assert_close(initial[:, :4], actual[:, :4], rtol=0, atol=0)
    assert not backbone.dep_layer.cross_attn.training
    assert backbone.dep_layer.cross_attn.attn_dropout_p == 0.37
    with pytest.raises(ValueError, match="equal"):
        causal_decode_s2(backbone, context, ids[:, :3])


def test_origin_heads_have_no_future_access_and_future_only_ar_mask():
    student, rows, spec, stats, _ = fixture()
    batch = _batch(rows, spec, stats, torch.device("cpu"), 4)
    assert batch["ar_mask"].tolist() == [[False, False, False, True, True]] * 3
    _, losses, before = forward_objective(student, batch, context_length=4,
                                          loss_weights=DEFAULT_LOSS_WEIGHTS)
    perturbed = deepcopy(batch)
    perturbed["s1_inputs"][:, 4:] = (perturbed["s1_inputs"][:, 4:] + 7) % 16
    perturbed["s2_inputs"][:, 4:] = (perturbed["s2_inputs"][:, 4:] + 7) % 16
    _, _, after = forward_objective(student, perturbed, context_length=4,
                                    loss_weights=DEFAULT_LOSS_WEIGHTS)
    torch.testing.assert_close(before["direction_logits"], after["direction_logits"], rtol=0, atol=0)
    for name in spec:
        torch.testing.assert_close(before["teachers"][name]["continuous"], after["teachers"][name]["continuous"], rtol=0, atol=0)
    assert set(losses) == {"ar_s1", "ar_s2", "direction", "w32_continuous", "w32_events",
                           "ict_continuous", "ict_events", "czsc_continuous", "czsc_events", "base_kl"}
    assert all(torch.isfinite(value) for value in losses.values())
    assert float(losses["czsc_events"].detach()) == 0.


def test_only_train_valid_continuous_targets_fit_statistics():
    rows = samples()
    rows[0]["teachers"]["ict"]["continuous_targets"][0] = 100
    stats = fit_teacher_statistics(rows, teacher_specification(rows))
    assert stats["ict"]["mean"][0] == 50
    assert stats["ict"]["valid_count"][0] == 2
    assert stats["ict"]["fit_split"] == "train"


def test_tokenizer_prefix_invariance_required_before_any_training():
    class LeakingTokenizer(SyntheticTokenizer):
        def encode(self, inputs, half=False):
            left, right = super().encode(inputs, half)
            return (left + inputs.shape[1]) % 16, right
    with pytest.raises(ValueError, match="prefix invariance"):
        encode_samples(LeakingTokenizer(), samples(), SimpleNamespace(context_length=4, horizon=2), torch.device("cpu"))


def test_adapter_checkpoint_contains_no_base_and_restore_rejects_mismatch(tmp_path):
    student, rows, _, _, _ = fixture()
    saved = trainable_state(student)
    assert saved and all(".lora_" in name or not name.startswith("backbone.") for name in saved)
    checkpoint = tmp_path / "adapter.pt"
    torch.save({"schema": "synthetic", "trainable_state": saved, "optimizer_state": {}}, checkpoint)
    loaded = torch.load(checkpoint, weights_only=True)
    restored = KronosStudent(SyntheticBackbone(), 3, lora_rank=2,
                              extra_teacher_dims={"ict": (2, 6), "czsc": (16, 0)})
    restore_trainable_state(restored, loaded["trainable_state"])
    for name, value in trainable_state(restored).items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="names"):
        restore_trainable_state(restored, {**saved, "backbone.fake_weight": torch.zeros(1)})
    wrong = dict(saved)
    key = next(iter(wrong))
    wrong[key] = wrong[key].flatten()[:1]
    with pytest.raises(ValueError, match="shape/dtype"):
        restore_trainable_state(restored, wrong)


def test_frozen_reference_restores_scaling_and_is_not_a_second_model():
    student, _, _, _, _ = fixture()
    adapters = [module for module in student.backbone.modules() if hasattr(module, "scaling")]
    scaling = [module.scaling for module in adapters]
    with pytest.raises(RuntimeError):
        with frozen_base(student) as reference:
            assert reference is student.backbone
            assert all(module.scaling == 0. for module in adapters)
            raise RuntimeError("synthetic failure")
    assert [module.scaling for module in adapters] == scaling


def test_synthetic_evaluation_reports_unknown_denominators_and_baseline():
    student, rows, spec, statistics, prefix = fixture()
    assert prefix["prefix_token_ids_exact"]
    report, predictions = evaluate_samples(student, rows, context_length=4, teacher_spec=spec,
        teacher_statistics=statistics, loss_weights=DEFAULT_LOSS_WEIGHTS, device=torch.device("cpu"),
        direction_prior=[1/3] * 3, batch_size=2)
    assert report["requested"] == report["valid"] == 3
    assert len(predictions) == 3
    assert report["teachers"]["ict"]["continuous_requested"] == 6
    assert report["teachers"]["ict"]["continuous_valid"] == 5
    assert report["teachers"]["czsc"]["event_valid"] == 0
    assert report["direction_train_prior_baseline"]["fit_split"] == "train"
    assert report["mean_losses"]["frozen_base_ar_s1"] > 0
    assert not report["qualified"]


def test_evaluation_masked_means_are_independent_of_batch_composition():
    student, rows, spec, statistics, _ = fixture()
    # Distinct samples and differently sparse masks expose batch-mean bias.
    rows[0]["teachers"] = deepcopy(rows[0]["teachers"])
    rows[0]["teachers"]["ict"]["continuous_targets"][0] = 0.001
    rows[0]["_s1"] = (rows[0]["_s1"] + 2) % 16
    reports = []
    for batch_size in (1, 2, 3):
        report, _ = evaluate_samples(student, rows, context_length=4, teacher_spec=spec,
            teacher_statistics=statistics, loss_weights=DEFAULT_LOSS_WEIGHTS, device=torch.device("cpu"),
            direction_prior=[1/3] * 3, batch_size=batch_size)
        reports.append(report)
    for report in reports[1:]:
        assert report["loss_valid_elements"] == reports[0]["loss_valid_elements"]
        for name, value in report["mean_losses"].items():
            assert value == pytest.approx(reports[0]["mean_losses"][name], rel=1e-6, abs=1e-6)


def test_explicit_unavailable_device_errors(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")
    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device("mps")


def test_four_arm_auxiliary_budget_and_common_objectives():
    _, _, spec, _, _ = fixture()
    config = SimpleNamespace(values={"training": {"loss_weights": {"base_kl": 0.}}})
    for mode in ("none", "czsc", "ict", "dual"):
        weights = _loss_weights(config, spec, teacher_mode=mode, auxiliary_budget=0.6)
        assert [weights[key] for key in ("ar_s1", "ar_s2", "direction", "base_kl")] == [1., 1., 1., 0.]
        teacher_total = sum(value for key, value in weights.items() if key.endswith(("_continuous", "_events")))
        assert teacher_total == pytest.approx(0. if mode == "none" else 0.6)
        if mode == "czsc":
            assert weights["w32_continuous"] == pytest.approx(0.15)
            assert weights["w32_events"] == pytest.approx(0.15)
            assert weights["czsc_continuous"] == pytest.approx(0.3)
            assert weights["ict_continuous"] == weights["ict_events"] == 0.
        if mode == "dual":
            assert weights["czsc_continuous"] == pytest.approx(0.15)
            assert weights["ict_continuous"] == weights["ict_events"] == pytest.approx(0.15)
    with pytest.raises(ValueError, match="teacher_mode"):
        _loss_weights(config, spec, teacher_mode="bad")
    with pytest.raises(ValueError, match="auxiliary_budget"):
        _loss_weights(config, spec, auxiliary_budget=float("nan"))


def test_no_teacher_control_is_invariant_to_teacher_targets_and_has_no_teacher_gradients():
    student, rows, spec, stats, _ = fixture()
    weights = _loss_weights(SimpleNamespace(values={"training": {"loss_weights": {"base_kl": 0.}}}),
                            spec, teacher_mode="none")
    batch = _batch(rows, spec, stats, torch.device("cpu"), 4)
    initial = trainable_state(student)
    total, _, _ = forward_objective(student, batch, context_length=4, loss_weights=weights)
    altered = deepcopy(batch)
    for targets in altered["teachers"].values():
        targets["continuous_targets"] += 100
        targets["event_targets"] = 1. - targets["event_targets"]
    other, _, _ = forward_objective(student, altered, context_length=4, loss_weights=weights)
    torch.testing.assert_close(total, other, rtol=0, atol=0)
    total.backward()
    for name, parameter in student.named_parameters():
        if parameter.requires_grad and not name.startswith(("backbone.", "direction_head.")):
            assert parameter.grad is None
    assert student.direction_head.weight.grad is not None
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum()) for name, parameter in student.named_parameters()
               if ".lora_" in name)
    assert _state_sha256(trainable_state(student)) == _state_sha256(initial)


def test_teacher_modes_preserve_initialization_and_diagnostic_backward_does_not_update_state():
    hashes = []
    for mode in ("none", "czsc", "ict", "dual"):
        student, rows, spec, stats, _ = fixture()
        before = trainable_state(student)
        hashes.append(_state_sha256(before))
        weights = _loss_weights(SimpleNamespace(values={"training": {"loss_weights": {"base_kl": 0.}}}),
                                spec, teacher_mode=mode)
        batch = _batch(rows, spec, stats, torch.device("cpu"), 4)
        result = gradient_diagnostics(student, batch, context_length=4, loss_weights=weights)
        assert result["fit_split"] == "train"
        assert result["groups"]["czsc"]["active"] == (mode in ("czsc", "dual"))
        assert result["groups"]["ict"]["active"] == (mode in ("ict", "dual"))
        assert all(value is None or -1.00001 <= value <= 1.00001 for value in result["gradient_cosines"].values())
        assert _state_sha256(trainable_state(student)) == _state_sha256(before)
        assert all(parameter.grad is None for parameter in student.parameters())
        assert all(group["delta_l2"] == 0. for group in _parameter_movement(before, before).values())
    assert len(set(hashes)) == 1


def test_selection_uses_common_direction_loss_and_teacher_constant_baselines_are_train_only():
    assert _selection_value({"direction": {"log_loss": 0.7}, "mean_losses": {"total": 99.}}, "direction_log_loss") == 0.7
    with pytest.raises(ValueError, match="common"):
        _selection_value({"direction": {"log_loss": 0.7}}, "total_distillation_objective")
    student, rows, spec, stats, _ = fixture()
    # All known synthetic events are negative; the baseline must use the
    # training event rates rather than the evaluation targets.
    assert stats["ict"]["event_rate"] == [0.] * 6
    altered = deepcopy(rows)
    altered[0]["teachers"]["ict"]["event_targets"][0] = 1.
    report, predictions = evaluate_samples(student, altered, context_length=4, teacher_spec=spec,
        teacher_statistics=stats, loss_weights=DEFAULT_LOSS_WEIGHTS, device=torch.device("cpu"),
        direction_prior=[1/3] * 3, batch_size=2)
    baseline = report["teachers"]["ict"]["train_constant_baseline"]
    assert baseline["fit_split"] == "train"
    assert baseline["event_bce"] > 0.7
    assert report["direction"]["log_loss"] == pytest.approx(sum(row["direction_log_loss"] for row in predictions) / 3)
    assert sum(report["direction"]["target_class_counts"]) == 3


def test_zero_event_and_all_unknown_teachers_never_call_bce_with_empty_buffers(monkeypatch):
    student, rows, spec, stats, _ = fixture()
    for row in rows:
        row["teachers"] = deepcopy(row["teachers"])
        for teacher in row["teachers"].values():
            teacher["event_mask"] = [False] * len(teacher["event_mask"])
        row["teachers"]["ict"]["continuous_mask"] = [False, False]
    def reject_bce(*args, **kwargs):
        pytest.fail("BCE must not be invoked for zero event columns or all-unknown event targets")
    monkeypatch.setattr(F, "binary_cross_entropy", reject_bce)
    monkeypatch.setattr(F, "binary_cross_entropy_with_logits", reject_bce)
    report, _ = evaluate_samples(student, rows, context_length=4, teacher_spec=spec,
        teacher_statistics=stats, loss_weights=DEFAULT_LOSS_WEIGHTS, device=torch.device("cpu"),
        direction_prior=[1/3] * 3, batch_size=2)
    for name in spec:
        assert report["teachers"][name]["event_valid"] == 0
        assert report["teachers"][name]["event_bce"] is None
        assert report["mean_losses"][name + "_events"] == 0.
    assert report["teachers"]["czsc"]["event_requested"] == 0
    assert report["teachers"]["ict"]["continuous_valid"] == 0
    assert report["teachers"]["ict"]["continuous_mae_normalized"] is None
