"""Explicit external references and output containment, without filesystem writes."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import os
from pathlib import Path
import tomllib
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES = ("open", "high", "low", "close", "volume", "amount")
SPLIT_NAMES = ("train", "tune", "calibration", "policy", "holdout")


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    config_path: Path
    source_app: Path
    raw_db: Path
    kronos_root: Path
    source_python: Path
    state_root: Path
    calendar_path: Path | None
    ict_vendor_root: Path
    values: dict[str, Any]

    @property
    def context_length(self) -> int:
        return self.values["data"]["context_length"]

    @property
    def horizon(self) -> int:
        return self.values["data"]["horizon"]

    def output_path(self, relative: str | Path) -> Path:
        """A symlink or '..' must never redirect a generated file outside here."""
        path = (self.root / relative).resolve()
        if path == self.root or not path.is_relative_to(self.root):
            raise ValueError("Output must be a file or directory inside token_distillation")
        if path.relative_to(self.root).parts[0] not in {"data", "artifacts", "logs", "audit"}:
            raise ValueError("Generated outputs are restricted to data, artifacts, logs and audit")
        sources = [self.source_app, self.kronos_root, self.raw_db, self.ict_vendor_root]
        if self.calendar_path is not None:
            sources.append(self.calendar_path)
        for source in sources:
            if path == source or path.is_relative_to(source):
                raise ValueError("Output overlaps a read-only source")
        return path


def _path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Configured paths must be nonempty strings")
    result = Path(value).expanduser()
    return (result if result.is_absolute() else root / result).resolve()


def _integer(value: Any, name: str, lower: int, upper: int) -> None:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


def load_config(path: str | Path | None = None, *, root: Path | None = None) -> ProjectConfig:
    root = Path(root or PROJECT_ROOT).resolve()
    config_path = Path(path or root / "configs/default.toml").expanduser().resolve()
    with config_path.open("rb") as handle:
        values = tomllib.load(handle)
    if values.get("schema_version") != "token_distillation_config_v1":
        raise ValueError("Unsupported configuration schema")
    for section in ("paths", "data", "teacher", "student", "ict", "training", "splits"):
        if not isinstance(values.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    data, student, training = (values[x] for x in ("data", "student", "training"))
    if (data.get("universe") != "sh_sz_a_v1" or data.get("frequency") != "1d"
            or tuple(data.get("features", ())) != FEATURES
            or data.get("volume_unit") != "shares" or data.get("amount_unit") != "CNY"
            or data.get("price_basis") != "unadjusted"):
        raise ValueError("Unsupported data universe, feature order or units")
    _integer(data.get("context_length"), "context_length", 16, 512)
    _integer(data.get("horizon"), "horizon", 1, 10)
    for name in ("normalization_epsilon", "normalization_clip"):
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    _integer(student.get("lora_rank"), "lora_rank", 1, 64)
    alpha = student.get("lora_alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("lora_alpha must be finite and positive")
    if (student.get("freeze_tokenizer") is not True
            or student.get("target_modules") != ["q_proj", "v_proj"]
            or student.get("direction_classes") != 3):
        raise ValueError("The scaffold supports a frozen tokenizer, q/v LoRA and three direction classes")
    _integer(training.get("max_candidates"), "max_candidates", 1, 6)
    _integer(training.get("max_kronos_configurations"), "max_kronos_configurations", 1, 2)
    _integer(training.get("micro_batch_size"), "micro_batch_size", 1, 1024)
    _integer(training.get("gradient_accumulation_steps"), "gradient_accumulation_steps", 1, 1024)
    _integer(training.get("seed"), "seed", 0, 2**32 - 1)
    if training.get("device") not in ("auto", "cpu", "mps", "cuda", "cuda:0"):
        raise ValueError("Unsupported device")
    if type(training.get("execution_enabled")) is not bool:
        raise ValueError("training.execution_enabled must be boolean")
    if training.get("stage") not in {"scaffold_only", "pilot_ready"}:
        raise ValueError("Unsupported training stage")
    if training.get("stage") == "scaffold_only" and training["execution_enabled"]:
        raise ValueError("scaffold_only cannot enable training")
    threshold = data.get("direction_threshold", 0.005)
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError("direction_threshold must be finite in [0, 1)")
    ict = values["ict"]
    if type(ict.get("enabled")) is not bool:
        raise ValueError("ict.enabled must be boolean")
    _integer(ict.get("swing_width"), "ict.swing_width", 1, 20)
    if ict.get("continuous_dim") != 2 or ict.get("event_dim") != 6:
        raise ValueError("The initial causal ICT schema has two continuous and six event targets")
    last_end = None
    for name in SPLIT_NAMES:
        bounds = values["splits"].get(name)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"Invalid split range: {name}")
        start, end = (date.fromisoformat(day) for day in bounds)
        if start > end or (last_end is not None and start <= last_end):
            raise ValueError("Splits must be ordered and nonoverlapping")
        last_end = end
    paths = values["paths"]
    # Only these explicit overrides are honored; no external .env is loaded.
    external = {
        "source_app": _path(root, os.environ.get("TD_SOURCE_APP") or paths["source_app"]),
        "raw_db": _path(root, os.environ.get("TD_RAW_DB") or paths["raw_db"]),
        "kronos_root": _path(root, os.environ.get("TD_KRONOS_ROOT") or paths["kronos_root"]),
        "source_python": _path(root, os.environ.get("TD_SOURCE_PYTHON") or paths["source_python"]),
    }
    state_root = _path(root, paths["state_root"])
    calendar_value = os.environ.get("TD_CALENDAR_PATH") or paths.get("calendar_path")
    config = ProjectConfig(root, config_path, state_root=state_root,
                           calendar_path=_path(root, calendar_value) if calendar_value else None,
                           ict_vendor_root=_path(root, os.environ.get("TD_ICT_VENDOR_ROOT") or paths["ict_vendor_root"]),
                           values=values, **external)
    config.output_path(state_root)
    # A source parent must never encompass this workspace's own outputs.
    if root.is_relative_to(config.source_app) or root.is_relative_to(config.kronos_root):
        raise ValueError("Workspace must be outside read-only source and model directories")
    return config
