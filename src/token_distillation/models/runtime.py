"""Offline, identity-bound official Kronos assets; no KHQuant service imports.

Only the two frozen vendor math modules are imported in an isolated namespace.
There is no Hugging Face loader, network request, source sys.path mutation or
source bytecode write. Base weights are referenced, never copied to checkpoints.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import torch

from ..artifacts import sha256_file
from ..config import ProjectConfig
from .lora import LoRALinear
from .student import KronosStudent

OFFICIAL_PINS = {
    "model/model.safetensors": "abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83",
    "model/config.json": "77ebc3038b647709b92be002f801d72e1a385f4c8c2c5aa1cc6cf21fcfe44eb2",
    "tokenizer/model.safetensors": "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee",
    "tokenizer/config.json": "2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2",
    "vendor/source_manifest.json": "45eaf905e092e39695442da9bcd7d4be0a3b83c462aadef8d07d58d2d824fb83",
}


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def resolve_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "mps", "cuda", "cuda:0"}:
        raise ValueError(f"Unsupported device: {requested}")
    if requested == "auto":
        requested = ("cuda" if torch.cuda.is_available() else "mps"
                     if torch.backends.mps.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Explicit CUDA device requested but CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Explicit MPS device requested but MPS is unavailable")
    return torch.device(requested)


def asset_identity(config: ProjectConfig) -> dict:
    vendor = config.source_app / "vendor" / "kronos"
    source_manifest = vendor / "source_manifest.json"
    manifest = json.loads(source_manifest.read_text())
    expected = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    files = {}
    for name in ("module.py", "kronos.py", "LICENSE"):
        path = vendor / name
        digest = sha256_file(path)
        if digest != expected.get(name):
            raise ValueError(f"Frozen Kronos source manifest mismatch: {name}")
        files[f"vendor/{name}"] = {"path": str(path), "sha256": digest}
    for name, path in {
        "vendor/source_manifest.json": source_manifest,
        "model/config.json": config.kronos_root / "config.json",
        "model/model.safetensors": config.kronos_root / "model.safetensors",
        "tokenizer/config.json": config.kronos_root / "tokenizer" / "config.json",
        "tokenizer/model.safetensors": config.kronos_root / "tokenizer" / "model.safetensors",
    }.items():
        digest = sha256_file(path)
        if digest != OFFICIAL_PINS[name]:
            raise ValueError(f"Pinned official Kronos asset mismatch: {name}")
        files[name] = {"path": str(path), "sha256": digest}
    return {"schema": "offline_official_kronos_assets_v1", "files": files,
            "model_revision": "NeoQuasar/Kronos-base@2b554741eca47781b64468546e77fef3e85130e6",
            "tokenizer_revision": "NeoQuasar/Kronos-Tokenizer-base@0e0117387f39004a9016484a186a908917e22426",
            "sha256": canonical_hash(files)}


def _vendor_classes(config: ProjectConfig, identity: dict):
    vendor = config.source_app / "vendor" / "kronos"
    namespace = "_td_frozen_kronos_" + identity["sha256"][:20]
    if namespace + ".kronos" not in sys.modules:
        package = ModuleType(namespace)
        package.__path__ = [str(vendor)]
        sys.modules[namespace] = package
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            for name in ("module", "kronos"):
                spec = importlib.util.spec_from_file_location(namespace + "." + name, vendor / (name + ".py"))
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load frozen Kronos source: {name}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
        except Exception:
            for name in (namespace, namespace + ".module", namespace + ".kronos"):
                sys.modules.pop(name, None)
            raise
        finally:
            sys.dont_write_bytecode = previous
    module = sys.modules[namespace + ".kronos"]
    return module.Kronos, module.KronosTokenizer


def load_official_assets(config: ProjectConfig, *, device: str | torch.device = "auto"):
    """Materialize verified local tensors with a frozen tokenizer/backbone."""
    from safetensors.torch import load_file
    resolved = resolve_device(str(device))
    identity = asset_identity(config)
    Kronos, Tokenizer = _vendor_classes(config, identity)
    backbone = Kronos(**json.loads((config.kronos_root / "config.json").read_text()))
    tokenizer = Tokenizer(**json.loads((config.kronos_root / "tokenizer" / "config.json").read_text()))
    backbone.load_state_dict(load_file(str(config.kronos_root / "model.safetensors")), strict=True)
    tokenizer.load_state_dict(load_file(str(config.kronos_root / "tokenizer" / "model.safetensors")), strict=True)
    backbone.requires_grad_(False).eval().to(resolved)
    tokenizer.requires_grad_(False).eval().to(resolved)
    return backbone, tokenizer, identity


def make_student(config: ProjectConfig, backbone, teacher_spec: dict) -> KronosStudent:
    if "w32" not in teacher_spec or len(teacher_spec["w32"]["event_names"]) != KronosStudent.EVENT_COUNT:
        raise ValueError("W32 teacher with the exact seven event targets is required")
    settings = config.values["student"]
    student = KronosStudent(backbone, len(teacher_spec["w32"]["continuous_names"]),
                            lora_rank=settings["lora_rank"], lora_alpha=settings["lora_alpha"],
                            context_feature_dim=20 if settings.get("context_statistics", False) else 0,
                            extra_teacher_dims={name: (len(spec["continuous_names"]), len(spec["event_names"]))
                                                for name, spec in teacher_spec.items() if name != "w32"})
    # Frozen vendor dropout is disabled; LoRA and heads still receive gradients.
    student.backbone.eval()
    return student


def causal_decode_s2(backbone, context: torch.Tensor, s1_targets: torch.Tensor) -> torch.Tensor:
    """Force vendor dependency attention causal for teacher-forced train/eval.

    The frozen source ties its cross-attention causal flag to ``training``.
    Set only that flag, with attention dropout explicitly zero, leaving every
    dropout module in evaluation mode. This is external runtime adaptation;
    vendor source and weights remain untouched. Equal query/key lengths are
    required because PyTorch's non-square causal alignment differs.
    """
    if context.shape[:2] != s1_targets.shape:
        raise ValueError("Causal s2 scoring requires equal full-sequence query/key lengths")
    attention = getattr(getattr(backbone, "dep_layer", None), "cross_attn", None)
    if attention is None or not hasattr(attention, "attn_dropout_p"):
        raise TypeError("Official Kronos dependency cross-attention interface is required")
    previous_training, previous_dropout = attention.training, attention.attn_dropout_p
    try:
        attention.training, attention.attn_dropout_p = True, 0.0
        return backbone.decode_s2(context, s1_targets, padding_mask=None)
    finally:
        attention.training, attention.attn_dropout_p = previous_training, previous_dropout


@contextmanager
def frozen_base(student: KronosStudent):
    """Evaluate the original base without allocating another large checkpoint."""
    modules = [(module, module.scaling) for module in student.backbone.modules()
               if isinstance(module, LoRALinear)]
    was_training = student.backbone.training
    try:
        student.backbone.eval()
        for module, _ in modules:
            module.scaling = 0.0
        with torch.no_grad():
            yield student.backbone
    finally:
        for module, scaling in modules:
            module.scaling = scaling
        student.backbone.train(was_training)


def trainable_state(student: KronosStudent) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in student.named_parameters()
            if parameter.requires_grad}


def restore_trainable_state(student: KronosStudent, tensors: dict) -> None:
    expected = {name: parameter for name, parameter in student.named_parameters() if parameter.requires_grad}
    if set(tensors) != set(expected):
        raise ValueError("Checkpoint trainable tensor names do not match the LoRA/head architecture")
    for name, parameter in expected.items():
        value = tensors[name]
        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
            raise ValueError(f"Checkpoint tensor shape/dtype mismatch: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Checkpoint contains nonfinite tensors: {name}")
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(tensors[name].to(parameter.device))
