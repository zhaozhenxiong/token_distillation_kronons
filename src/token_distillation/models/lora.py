"""Small, vendor-independent LoRA adapters for the Kronos transformer.

Only ``transformer.<index>.self_attn.{q_proj,v_proj}`` is eligible.  This
deliberately excludes the tokenizer and Kronos' dependency attention layer.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _validate_hyperparameters(rank: int, alpha: float) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("LoRA alpha must be a positive finite number")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha must be a positive finite number")


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank residual.

    B starts at zero, so the first forward exactly equals the original layer.
    ``merge()`` folds the residual into the frozen base layer in place and
    returns that layer. Use ``merge_lora`` to also replace adapter modules.
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        _validate_hyperparameters(rank, alpha)
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear requires an nn.Linear base layer")
        if not base_layer.weight.is_floating_point():
            raise TypeError("LoRA requires floating-point base weights")
        if base_layer.weight.device.type == "meta":
            raise ValueError("Materialize the base weights before injecting LoRA")
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.lora_A = nn.Parameter(base_layer.weight.new_empty(rank, self.in_features))
        self.lora_B = nn.Parameter(base_layer.weight.new_zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self._merged = False

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base_layer.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(inputs)
        if self._merged:
            return result
        return result + F.linear(F.linear(inputs, self.lora_A), self.lora_B) * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        if self._merged:
            raise ValueError("This LoRA adapter has already been merged")
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base_layer.weight.add_(delta)
        self._merged = True
        return self.base_layer


def _projection_slots(backbone: nn.Module) -> list[tuple[str, nn.Module, str, nn.Module]]:
    transformer = getattr(backbone, "transformer", None)
    if not isinstance(transformer, (nn.ModuleList, nn.Sequential)):
        raise ValueError("Expected a Kronos backbone with a transformer ModuleList/Sequential")
    slots = []
    for index, block in enumerate(transformer):
        attention = getattr(block, "self_attn", None)
        for name in ("q_proj", "v_proj"):
            layer = getattr(attention, name, None)
            if layer is None:
                raise ValueError(f"Missing transformer.{index}.self_attn.{name}")
            slots.append((f"transformer.{index}.self_attn.{name}", attention, name, layer))
    if not slots:
        raise ValueError("No transformer q_proj/v_proj layers matched")
    return slots


def inject_lora(backbone: nn.Module, rank: int = 8, alpha: float = 16.0) -> tuple[str, ...]:
    """Freeze the whole backbone, then adapt its transformer q/v projections.

    Preflight checks run before any mutation. Repeated injection, absent
    targets, and unexpected projection types fail loudly.
    """
    _validate_hyperparameters(rank, alpha)
    if any(isinstance(module, LoRALinear) for module in backbone.modules()):
        raise ValueError("Backbone already contains LoRA adapters; refusing duplicate injection")
    slots = _projection_slots(backbone)
    for path, _, _, layer in slots:
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"{path} must be nn.Linear")
        if not layer.weight.is_floating_point() or layer.weight.device.type == "meta":
            raise ValueError(f"{path} must have materialized floating-point weights")
    backbone.requires_grad_(False)
    for _, parent, name, layer in slots:
        adapter = LoRALinear(layer, rank=rank, alpha=alpha)
        adapter.train(layer.training)
        setattr(parent, name, adapter)
    return tuple(path for path, _, _, _ in slots)


def merge_lora(backbone: nn.Module) -> tuple[str, ...]:
    """Fold every transformer q/v adapter into nn.Linear and remove adapters.

    The operation is in place and terminal for that optimizer. Create a new
    optimizer if adapting the resulting backbone again.
    """
    slots = _projection_slots(backbone)
    for path, _, _, layer in slots:
        if not isinstance(layer, LoRALinear) or layer._merged:
            raise ValueError(f"{path} is not an unmerged LoRA adapter")
    for _, parent, name, adapter in slots:
        setattr(parent, name, adapter.merge())
    return tuple(path for path, _, _, _ in slots)
