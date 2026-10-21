"""Synthetic CPU tests: no vendor imports, model downloads, or real training."""

from __future__ import annotations

import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from token_distillation.models.lora import LoRALinear, inject_lora, merge_lora
from token_distillation.models.student import KronosStudent
from token_distillation.training.losses import (
    masked_bce_with_logits_loss,
    masked_cross_entropy_loss,
    masked_huber_loss,
    masked_kl_div_loss,
)


class _Attention(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.q_proj = nn.Linear(size, size)
        self.k_proj = nn.Linear(size, size)
        self.v_proj = nn.Linear(size, size)

    def forward(self, x):
        return self.q_proj(x) + torch.tanh(self.v_proj(x)) + 0.1 * self.k_proj(x)


class _Block(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.self_attn = _Attention(size)
        self.norm = nn.LayerNorm(size)

    def forward(self, x):
        return self.norm(x + self.self_attn(x))


class _SyntheticKronos(nn.Module):
    """Only the interface and exact target paths mirror the external model."""

    def __init__(self):
        super().__init__()
        self.d_model = 8
        self.embedding = nn.Embedding(16, 8)
        self.transformer = nn.ModuleList([_Block(), _Block()])
        self.dep_layer = _Attention()
        self.head = nn.Linear(8, 16)

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        x = self.embedding(s1_ids) + self.embedding(s2_ids)
        for layer in self.transformer:
            x = layer(x)
        return self.head(x), x


class LoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_zero_initialized_adapter_is_exactly_base(self):
        base = nn.Linear(5, 4)
        inputs = torch.randn(2, 3, 5)
        expected = base(inputs).detach().clone()
        layer = LoRALinear(base, rank=2, alpha=4)
        torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
        self.assertFalse(layer.base_layer.weight.requires_grad)
        self.assertFalse(layer.base_layer.bias.requires_grad)
        self.assertEqual(torch.count_nonzero(layer.lora_B).item(), 0)
        self.assertGreater(torch.count_nonzero(layer.lora_A).item(), 0)

    def test_hyperparameters_reject_invalid_values(self):
        for rank in (0, -1, 1.5, True):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                LoRALinear(nn.Linear(2, 2), rank=rank)
        for alpha in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                LoRALinear(nn.Linear(2, 2), alpha=alpha)

    def test_injection_is_exactly_scoped_and_freezes_all_base_weights(self):
        backbone = _SyntheticKronos()
        dependency_q = backbone.dep_layer.q_proj
        targets = inject_lora(backbone, rank=2, alpha=4)
        self.assertEqual(len(targets), 4)
        self.assertEqual(targets[0], "transformer.0.self_attn.q_proj")
        self.assertIs(backbone.dep_layer.q_proj, dependency_q)
        trainable = [name for name, parameter in backbone.named_parameters() if parameter.requires_grad]
        self.assertEqual(len(trainable), 8)
        self.assertTrue(all(".lora_A" in name or ".lora_B" in name for name in trainable))

    def test_duplicate_and_missing_targets_fail_before_mutating(self):
        backbone = _SyntheticKronos()
        inject_lora(backbone, rank=2)
        original = backbone.transformer[0].self_attn.q_proj
        with self.assertRaisesRegex(ValueError, "already"):
            inject_lora(backbone, rank=2)
        self.assertIs(backbone.transformer[0].self_attn.q_proj, original)
        for broken in (nn.Linear(2, 2), nn.Module()):
            with self.assertRaises(ValueError):
                inject_lora(broken)
        empty = nn.Module()
        empty.transformer = nn.ModuleList()
        with self.assertRaisesRegex(ValueError, "No transformer"):
            inject_lora(empty)
        broken = _SyntheticKronos()
        broken.transformer[1].self_attn.v_proj = nn.Identity()
        with self.assertRaises(TypeError):
            inject_lora(broken)
        self.assertTrue(all(parameter.requires_grad for parameter in broken.parameters()))
        self.assertIsInstance(broken.transformer[0].self_attn.q_proj, nn.Linear)

    def test_merge_nonzero_adapter_removes_modules_and_preserves_forward(self):
        backbone = _SyntheticKronos().eval()
        inject_lora(backbone, rank=3, alpha=6)
        with torch.no_grad():
            for module in backbone.modules():
                if isinstance(module, LoRALinear):
                    module.lora_B.normal_(std=0.07)
        ids = torch.randint(0, 16, (3, 7))
        expected_logits, expected_hidden = backbone.decode_s1(ids, ids)
        paths = merge_lora(backbone)
        self.assertEqual(len(paths), 4)
        self.assertFalse(any(isinstance(module, LoRALinear) for module in backbone.modules()))
        actual_logits, actual_hidden = backbone.decode_s1(ids, ids)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=2e-5, atol=2e-6)
        self.assertFalse(any(parameter.requires_grad for parameter in backbone.parameters()))
        self.assertFalse(any("lora_" in key for key in backbone.state_dict()))
        with self.assertRaisesRegex(ValueError, "unmerged"):
            merge_lora(backbone)

    def test_standalone_merge_does_not_double_apply(self):
        adapter = LoRALinear(nn.Linear(3, 4), rank=2)
        with torch.no_grad():
            adapter.lora_B.normal_()
        inputs = torch.randn(2, 3)
        expected = adapter(inputs)
        base = adapter.merge()
        torch.testing.assert_close(base(inputs), expected)
        torch.testing.assert_close(adapter(inputs), expected)
        with self.assertRaisesRegex(ValueError, "already"):
            adapter.merge()


class StudentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_student_uses_last_decode_context_and_exposes_logits(self):
        student = KronosStudent(_SyntheticKronos(), continuous_dim=5, lora_rank=2)
        ids = torch.randint(0, 16, (3, 6))
        output = student(ids, ids)
        _, context = student.backbone.decode_s1(ids, ids)
        self.assertEqual(output["continuous"].shape, (3, 5))
        self.assertEqual(output["event_logits"].shape, (3, 7))
        self.assertEqual(output["direction_logits"].shape, (3, 3))
        self.assertEqual(output["s1_logits"].shape, (3, 6, 16))
        torch.testing.assert_close(output["hidden"], context[:, -1])
        with self.assertRaises(ValueError):
            student(ids[:, :0], ids[:, :0])
        with self.assertRaises(ValueError):
            KronosStudent(_SyntheticKronos(), continuous_dim=0)

    def test_only_lora_and_heads_receive_gradients_and_update(self):
        student = KronosStudent(_SyntheticKronos(), continuous_dim=4, lora_rank=2)
        before = {name: value.detach().clone() for name, value in student.named_parameters()}
        optimizer = torch.optim.SGD((p for p in student.parameters() if p.requires_grad), lr=0.1)
        ids = torch.randint(0, 16, (4, 5))
        for _ in range(3):
            optimizer.zero_grad()
            outputs = student(ids, ids)
            loss = (
                masked_huber_loss(outputs["continuous"], torch.ones(4, 4))
                + masked_bce_with_logits_loss(outputs["event_logits"], torch.ones(4, 7))
                + masked_cross_entropy_loss(outputs["direction_logits"], torch.tensor([0, 1, 2, 1]))
            )
            loss.backward()
            for name, parameter in student.named_parameters():
                if parameter.requires_grad:
                    self.assertTrue(".lora_" in name or "_head." in name, name)
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertIsNone(parameter.grad, name)
            optimizer.step()
        changed = {name for name, p in student.named_parameters() if not torch.equal(before[name], p)}
        self.assertTrue(any(".lora_A" in name for name in changed))
        self.assertTrue(any(".lora_B" in name for name in changed))
        for prefix in ("continuous_head.", "event_head.", "direction_head."):
            self.assertTrue(any(name.startswith(prefix) for name in changed))
        self.assertTrue(all(".lora_" in name or "_head." in name for name in changed))
        expected = {key: tensor.detach().clone() for key, tensor in student(ids, ids).items()}
        student.merge_adapters()
        for key, tensor in student(ids, ids).items():
            torch.testing.assert_close(tensor, expected[key], rtol=2e-5, atol=2e-6)

    def test_padding_masks_are_explicitly_rejected(self):
        student = KronosStudent(_SyntheticKronos(), continuous_dim=4, lora_rank=2)
        ids = torch.randint(0, 16, (2, 5))
        for mask in (torch.zeros_like(ids, dtype=torch.bool), torch.ones_like(ids, dtype=torch.bool)):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "equal-length"):
                student(ids, ids, padding_mask=mask)

    def test_default_and_empty_extra_teachers_preserve_original_outputs_and_parameters(self):
        default = KronosStudent(_SyntheticKronos(), continuous_dim=4, lora_rank=2)
        empty = KronosStudent(_SyntheticKronos(), continuous_dim=4, lora_rank=2, extra_teacher_dims={})
        ids = torch.randint(0, 16, (2, 5))
        expected_keys = {"hidden", "continuous", "event_logits", "direction_logits", "s1_logits"}
        for student in (default, empty):
            self.assertEqual(set(student(ids, ids)), expected_keys)
            self.assertTrue(all(isinstance(value, torch.Tensor) for value in student(ids, ids).values()))
            self.assertEqual(len(student.extra_teacher_heads), 0)
        self.assertEqual(set(default.state_dict()), set(empty.state_dict()))
        self.assertEqual(sum(p.numel() for p in default.parameters()), sum(p.numel() for p in empty.parameters()))

    def test_extra_teacher_names_and_dimensions_fail_before_backbone_mutation(self):
        invalid = (
            {"ict.smc": (2, 3)}, {"": (2, 3)}, {"items": (2, 3)}, {"forward": (2, 3)},
            {"backbone": (2, 3)}, {"extra_teachers": (2, 3)}, {"_ict": (2, 3)},
            {2: (2, 3)}, {"ict": (0, 3)}, {"ict": (2, -1)}, {"ict": (True, 3)},
            {"ict": (2.0, 3)}, {"ict": (2,)}, {"ict": [2, 3]},
        )
        for dimensions in invalid:
            with self.subTest(dimensions=dimensions):
                backbone = _SyntheticKronos()
                with self.assertRaises(ValueError):
                    KronosStudent(backbone, continuous_dim=4, extra_teacher_dims=dimensions)
                self.assertTrue(all(p.requires_grad for p in backbone.parameters()))
                self.assertFalse(any(isinstance(module, LoRALinear) for module in backbone.modules()))
        with self.assertRaises(TypeError):
            KronosStudent(_SyntheticKronos(), continuous_dim=4, extra_teacher_dims=[("ict", (2, 3))])

    def test_joint_teachers_share_one_backbone_receive_gradients_and_survive_merge(self):
        backbone = _SyntheticKronos()
        student = KronosStudent(backbone, continuous_dim=4, lora_rank=2, extra_teacher_dims={"ict": (9, 5)})
        self.assertIs(student.backbone, backbone)
        self.assertEqual(sum(isinstance(module, _SyntheticKronos) for module in student.modules()), 1)
        before = {name: parameter.detach().clone() for name, parameter in student.named_parameters()}
        optimizer = torch.optim.SGD((p for p in student.parameters() if p.requires_grad), lr=0.1)
        ids = torch.randint(0, 16, (4, 5))
        for _ in range(3):
            optimizer.zero_grad()
            outputs = student(ids, ids)
            ict = outputs["extra_teachers"]["ict"]
            self.assertEqual(ict["continuous"].shape, (4, 9))
            self.assertEqual(ict["event_logits"].shape, (4, 5))
            torch.testing.assert_close(ict["continuous"], student.extra_teacher_heads["ict"]["continuous"](outputs["hidden"]))
            loss = (
                masked_huber_loss(outputs["continuous"], torch.ones(4, 4))
                + masked_bce_with_logits_loss(outputs["event_logits"], torch.ones(4, 7))
                + masked_cross_entropy_loss(outputs["direction_logits"], torch.tensor([0, 1, 2, 1]))
                + masked_huber_loss(ict["continuous"], torch.ones(4, 9))
                + masked_bce_with_logits_loss(ict["event_logits"], torch.ones(4, 5))
            )
            loss.backward()
            for name, parameter in student.named_parameters():
                if parameter.requires_grad:
                    self.assertTrue(".lora_" in name or "_head." in name or "extra_teacher_heads." in name, name)
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertIsNone(parameter.grad, name)
            optimizer.step()
        for name, parameter in student.named_parameters():
            if ".lora_" not in name and name.startswith("backbone."):
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        for prefix in ("continuous_head.", "event_head.", "direction_head.",
                       "extra_teacher_heads.ict.continuous.", "extra_teacher_heads.ict.event_logits."):
            self.assertTrue(any(name.startswith(prefix) and not torch.equal(parameter, before[name])
                                for name, parameter in student.named_parameters()), prefix)
        with torch.no_grad():
            expected = student(ids, ids)
            heads_before = {name: value.clone() for name, value in student.extra_teacher_heads.state_dict().items()}
            student.merge_adapters()
            actual = student(ids, ids)
            for key in ("hidden", "continuous", "event_logits", "direction_logits", "s1_logits"):
                torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-6)
            for key in ("continuous", "event_logits"):
                torch.testing.assert_close(actual["extra_teachers"]["ict"][key],
                                           expected["extra_teachers"]["ict"][key], rtol=2e-5, atol=2e-6)
            for name, value in student.extra_teacher_heads.state_dict().items():
                torch.testing.assert_close(value, heads_before[name], rtol=0, atol=0)


class MaskedLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def test_regression_and_bce_ignore_nan_values_before_arithmetic(self):
        for fn, target in (
            (masked_huber_loss, torch.tensor([0.5, float("nan"), -1.0])),
            (masked_bce_with_logits_loss, torch.tensor([1.0, float("nan"), 0.0])),
        ):
            with self.subTest(loss=fn.__name__):
                prediction = torch.tensor([0.2, float("nan"), -0.3], requires_grad=True)
                mask = torch.tensor([True, False, True])
                loss = fn(prediction, target, mask)
                expected = fn(prediction[mask], target[mask])
                torch.testing.assert_close(loss, expected)
                loss.backward()
                self.assertTrue(torch.isfinite(prediction.grad).all())
                self.assertEqual(prediction.grad[1].item(), 0)

    def test_classification_masks_unknown_labels_and_nan_rows(self):
        logits = torch.tensor([[0.2, 0.1, -0.3], [math.nan] * 3], requires_grad=True)
        target = torch.tensor([1, -100])
        mask = torch.tensor([True, False])
        loss = masked_cross_entropy_loss(logits, target, mask)
        torch.testing.assert_close(loss, F.cross_entropy(logits[:1], target[:1]))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[1].abs().sum().item(), 0)

    def test_kl_masks_nan_rows_and_detaches_teacher(self):
        logits = torch.tensor([[0.1, 0.8, -0.5], [math.nan] * 3], requires_grad=True)
        teacher = torch.tensor([[0.7, 0.0, -0.2], [math.nan] * 3], requires_grad=True)
        loss = masked_kl_div_loss(logits, teacher, torch.tensor([True, False]), temperature=2.0)
        reference = F.kl_div(F.log_softmax(logits[:1] / 2, -1), F.softmax(teacher[:1].detach() / 2, -1), reduction="batchmean") * 4
        torch.testing.assert_close(loss, reference)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[1].abs().sum().item(), 0)

    def test_empty_masks_return_differentiable_zero_with_zero_finite_gradients(self):
        cases = (
            (masked_huber_loss, (2,), torch.full((2,), math.nan)),
            (masked_bce_with_logits_loss, (2,), torch.full((2,), math.nan)),
            (masked_cross_entropy_loss, (2, 3), torch.tensor([-100, -100])),
            (masked_kl_div_loss, (2, 3), torch.full((2, 3), math.nan)),
        )
        for fn, shape, target in cases:
            with self.subTest(loss=fn.__name__):
                prediction = torch.full(shape, math.nan, requires_grad=True)
                loss = fn(prediction, target, torch.zeros(2, dtype=torch.bool))
                self.assertEqual(loss.item(), 0)
                loss.backward()
                torch.testing.assert_close(prediction.grad, torch.zeros(shape))

    def test_valid_nonfinite_predictions_or_targets_fail(self):
        for fn in (masked_huber_loss, masked_bce_with_logits_loss):
            with self.subTest(loss=fn.__name__):
                with self.assertRaisesRegex(ValueError, "finite"):
                    fn(torch.tensor([math.nan]), torch.tensor([0.0]))
                with self.assertRaisesRegex(ValueError, "finite"):
                    fn(torch.tensor([0.0]), torch.tensor([math.inf]))
        with self.assertRaisesRegex(ValueError, "finite"):
            masked_cross_entropy_loss(torch.tensor([[0.0, math.inf]]), torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, "finite"):
            masked_kl_div_loss(torch.tensor([[0.0, math.nan]]), torch.zeros(1, 2))
        with self.assertRaisesRegex(ValueError, "finite"):
            masked_kl_div_loss(torch.zeros(1, 2), torch.tensor([[0.0, math.inf]]))

    def test_shapes_mask_types_and_target_ranges_are_validated(self):
        with self.assertRaises(ValueError):
            masked_huber_loss(torch.zeros(2), torch.zeros(3))
        with self.assertRaises(ValueError):
            masked_huber_loss(torch.zeros(2), torch.zeros(2), torch.ones(2))
        with self.assertRaises(ValueError):
            masked_huber_loss(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            masked_bce_with_logits_loss(torch.zeros(2), torch.tensor([0.0, 2.0]))
        with self.assertRaises(ValueError):
            masked_cross_entropy_loss(torch.zeros(1, 3), torch.tensor([3]))
        with self.assertRaises(TypeError):
            masked_cross_entropy_loss(torch.zeros(1, 3), torch.tensor([0.0]))
        for value in (0, -1, math.nan, math.inf):
            with self.subTest(temperature=value), self.assertRaises(ValueError):
                masked_kl_div_loss(torch.zeros(1, 2), torch.zeros(1, 2), temperature=value)
        with self.assertRaises(ValueError):
            masked_huber_loss(torch.zeros(1), torch.zeros(1), delta=0)


if __name__ == "__main__":
    unittest.main()
