"""Masked primitive losses; unknown teacher targets never act as negatives.

Masks are boolean tensors with exactly the loss-element shape. Huber/BCE use
the prediction shape; CE/KL use the prediction shape without its class axis.
Select valid elements *before* arithmetic, preventing NaN placeholders from
contaminating either loss or gradients. Empty masks return differentiable zero.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _mask(mask: torch.Tensor | None, shape: torch.Size, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != shape:
        raise ValueError(f"mask must be boolean with exact shape {tuple(shape)}")
    return mask.to(device=device)


def _finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Valid {name} values must be finite")


def _pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shape")
    if prediction.device != target.device:
        raise ValueError("prediction and target must share a device")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating-point tensors")


def masked_huber_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, *, delta: float = 1.0
) -> torch.Tensor:
    _positive(delta, "delta")
    _pair(prediction, target)
    valid = _mask(mask, prediction.shape, prediction.device)
    selected_prediction, selected_target = prediction[valid], target[valid].detach()
    _finite(selected_prediction, "prediction")
    _finite(selected_target, "target")
    if selected_prediction.numel() == 0:
        return selected_prediction.sum()
    return F.huber_loss(selected_prediction, selected_target, delta=float(delta), reduction="mean")


def masked_bce_with_logits_loss(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    _pair(logits, target)
    valid = _mask(mask, logits.shape, logits.device)
    selected_logits, selected_target = logits[valid], target[valid].detach()
    _finite(selected_logits, "logit")
    _finite(selected_target, "target")
    if bool(((selected_target < 0) | (selected_target > 1)).any()):
        raise ValueError("Valid BCE targets must be in [0, 1]")
    if selected_logits.numel() == 0:
        return selected_logits.sum()
    return F.binary_cross_entropy_with_logits(selected_logits, selected_target, reduction="mean")


def masked_cross_entropy_loss(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    if logits.ndim < 1 or logits.shape[-1] == 0 or target.shape != logits.shape[:-1]:
        raise ValueError("CE expects logits [..., classes] and targets [...]")
    if not logits.is_floating_point() or target.dtype != torch.long:
        raise TypeError("CE expects floating-point logits and int64 targets")
    if logits.device != target.device:
        raise ValueError("logits and target must share a device")
    valid = _mask(mask, target.shape, logits.device)
    selected_logits, selected_target = logits[valid], target[valid]
    _finite(selected_logits, "logit")
    if bool(((selected_target < 0) | (selected_target >= logits.shape[-1])).any()):
        raise ValueError("Valid CE targets must name an existing class")
    if selected_target.numel() == 0:
        return selected_logits.sum()
    return F.cross_entropy(selected_logits, selected_target, reduction="mean")


def masked_kl_div_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Mean KL(teacher || student), including the usual temperature squared."""
    _positive(temperature, "temperature")
    _pair(student_logits, teacher_logits)
    if student_logits.ndim < 1 or student_logits.shape[-1] == 0:
        raise ValueError("KL expects logits [..., classes] with a nonempty class axis")
    valid = _mask(mask, student_logits.shape[:-1], student_logits.device)
    selected_student = student_logits[valid]
    selected_teacher = teacher_logits[valid].detach()
    _finite(selected_student, "student logit")
    _finite(selected_teacher, "teacher logit")
    if selected_student.numel() == 0:
        return selected_student.sum()
    log_student = F.log_softmax(selected_student / temperature, dim=-1)
    teacher = F.softmax(selected_teacher / temperature, dim=-1)
    return F.kl_div(log_student, teacher, reduction="none").sum(dim=-1).mean() * temperature**2
