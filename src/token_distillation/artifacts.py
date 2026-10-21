"""Strict JSON and atomic workspace-local metadata writes."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .config import ProjectConfig


def jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        return jsonable(value.item())
    return value


def dumps(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(config: ProjectConfig, target: str | Path, value: Any, *, replace: bool = False) -> Path:
    path = config.output_path(target)
    raw = (dumps(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Recheck resolved containment after directory creation, including links.
    if config.output_path(path) != path:
        raise ValueError("Output path changed during write")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".metadata-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # atomic create, refuses existing files
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
