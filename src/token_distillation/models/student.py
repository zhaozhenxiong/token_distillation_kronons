"""Kronos student heads without importing KHQuant or a vendor model package."""

from __future__ import annotations

from collections.abc import Mapping
import re

import torch
from torch import nn

from .lora import inject_lora, merge_lora


class KronosStudent(nn.Module):
    """Adapt a supplied Kronos instance and predict teacher targets internally.

    The caller loads the external backbone and frozen tokenizer. Inputs are
    already tokenized; ``decode_s1`` supplies the final context representation.
    Windows must be equal-length and end at the prediction origin: the final
    position is selected literally. Any padding mask is rejected, since the
    external vendor's boolean mask convention is not assumed here.
    Event and direction values are raw logits, not CZSC confirmed facts.
    Optional additional teachers add internal linear heads over the same
    hidden context. Their dimensions are supplied by the target schema, and
    their outputs appear under ``extra_teachers`` only when configured.
    """

    EVENT_COUNT = 7
    DIRECTION_COUNT = 3

    def __init__(
        self,
        backbone: nn.Module,
        continuous_dim: int,
        *,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        extra_teacher_dims: Mapping[str, tuple[int, int]] | None = None,
        context_feature_dim: int = 0,
    ):
        super().__init__()
        hidden_dim = getattr(backbone, "d_model", None)
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("Backbone must expose a positive integer d_model")
        if not callable(getattr(backbone, "decode_s1", None)):
            raise TypeError("Backbone must expose decode_s1(s1_ids, s2_ids, stamp, padding_mask)")
        if isinstance(continuous_dim, bool) or not isinstance(continuous_dim, int) or continuous_dim <= 0:
            raise ValueError("continuous_dim must be a positive integer")
        dimensions = self._validate_extra_teacher_dims(extra_teacher_dims)
        if type(context_feature_dim) is not int or context_feature_dim < 0:
            raise ValueError("Invalid context feature dimension")
        self.context_feature_dim = context_feature_dim
        head_dim = hidden_dim + context_feature_dim
        self.backbone = backbone
        self.lora_targets = inject_lora(backbone, rank=lora_rank, alpha=lora_alpha)
        floating = next((parameter for parameter in backbone.parameters() if parameter.is_floating_point()), None)
        factory = {"device": floating.device, "dtype": floating.dtype} if floating is not None else {}
        self.continuous_head = nn.Linear(head_dim, continuous_dim, **factory)
        self.event_head = nn.Linear(head_dim, self.EVENT_COUNT, **factory)
        self.direction_head = nn.Linear(head_dim, self.DIRECTION_COUNT, **factory)
        self.extra_teacher_heads = nn.ModuleDict({
            name: nn.ModuleDict({
                "continuous": nn.Linear(head_dim, dims[0], **factory),
                **({"event_logits": nn.Linear(head_dim, dims[1], **factory)} if dims[1] else {}),
            })
            for name, dims in dimensions.items()
        })

    def _validate_extra_teacher_dims(
        self, dimensions: Mapping[str, tuple[int, int]] | None
    ) -> dict[str, tuple[int, int]]:
        if dimensions is None:
            return {}
        if not isinstance(dimensions, Mapping):
            raise TypeError("extra_teacher_dims must be a mapping of teacher name to (continuous, event) dimensions")
        reserved = set(dir(nn.ModuleDict())) | set(dir(self)) | {
            "backbone", "lora_targets", "continuous_head", "event_head", "direction_head", "extra_teacher_heads",
            "hidden", "continuous", "event_logits", "direction_logits", "s1_logits", "extra_teachers",
        }
        validated = {}
        for name, dims in dimensions.items():
            if not isinstance(name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None or name in reserved:
                raise ValueError("Teacher names must be unreserved module identifiers without dots")
            if (not isinstance(dims, tuple) or len(dims) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in dims)
                    or dims[0] <= 0 or dims[1] < 0):
                raise ValueError(f"Teacher {name} needs a positive continuous dimension and nonnegative event dimension")
            validated[name] = dims
        return validated

    def head_input(self, hidden, context_features=None):
        if not self.context_feature_dim:
            return hidden
        if context_features is None or context_features.shape != (hidden.shape[0], self.context_feature_dim):
            raise ValueError("Causal context features required for internal heads")
        if not bool(torch.isfinite(context_features).all()):
            raise ValueError("Nonfinite context features")
        return torch.cat([hidden, context_features.to(device=hidden.device, dtype=hidden.dtype)], dim=-1)

    def forward(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        stamp: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        context_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, dict[str, torch.Tensor]]]:
        if padding_mask is not None:
            raise ValueError("Padding masks are unsupported; supply equal-length windows ending at the prediction origin")
        if s1_ids.ndim != 2 or s2_ids.shape != s1_ids.shape or s1_ids.shape[1] == 0:
            raise ValueError("Token tensors must share nonempty [batch, context] shape")
        s1_logits, context = self.backbone.decode_s1(
            s1_ids, s2_ids, stamp=stamp, padding_mask=None
        )
        if context.ndim != 3 or context.shape[:2] != s1_ids.shape:
            raise ValueError("decode_s1 context must have shape [batch, context, d_model]")
        hidden = context[:, -1, :]
        head_hidden = self.head_input(hidden, context_features)
        output: dict[str, torch.Tensor | dict[str, dict[str, torch.Tensor]]] = {
            "hidden": hidden,
            "continuous": self.continuous_head(head_hidden),
            "event_logits": self.event_head(head_hidden),
            "direction_logits": self.direction_head(head_hidden),
            "s1_logits": s1_logits,
        }
        if self.extra_teacher_heads:
            output["extra_teachers"] = {
                name: {"continuous": heads["continuous"](head_hidden),
                       "event_logits": heads["event_logits"](head_hidden) if "event_logits" in heads
                       else hidden.new_empty(hidden.shape[0], 0)}
                for name, heads in self.extra_teacher_heads.items()
            }
        return output

    def merge_adapters(self) -> tuple[str, ...]:
        """Merge backbone LoRA weights; preserve all student heads."""
        return merge_lora(self.backbone)
