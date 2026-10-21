"""Load the exact frozen ICT primitives, offline and without vendor bytecode writes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import threading
import types
from typing import Any


PINNED_REPOSITORY = "https://github.com/carreypro/ict-trading-model"
PINNED_COMMIT = "7162437e28b1f59a4b25ea88615ee99902c9cc75"
PINNED_SCRIPT_SHA256 = "e47249ea426d8b513c3e82e9e5cacaee77429424c269a225df3f0383a49802b5"
PINNED_MANIFEST_SHA256 = "1956f82ff6b2fa2d0bbf25aa78a66279db93b70b9535885dd822d1f78807b298"
_LOCK = threading.RLock()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenIctAdapter:
    """The vendor's scanner is preserved; only detect_swings/detect_fvgs are used.

    The archive commit is checked against a pinned manifest and every retained
    upstream file's pinned hash. This is a frozen archive, not a local Git
    checkout, and this class does not claim to re-verify absent Git objects.
    Hashes are checked again on each load, including after an in-process cache.
    """

    def __init__(self, vendor_root: str | Path):
        self.vendor_root = Path(vendor_root).expanduser().resolve()
        self._module: types.ModuleType | None = None

    def verify(self) -> dict[str, Any]:
        manifest_path = self.vendor_root / "source_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"frozen ICT manifest missing: {manifest_path}")
        if _sha(manifest_path) != PINNED_MANIFEST_SHA256:
            raise ValueError("frozen ICT manifest SHA256 differs from the pinned archive manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("commit") != PINNED_COMMIT or manifest.get("repository") != PINNED_REPOSITORY):
            raise ValueError("frozen ICT repository/commit does not match the pin")
        if manifest.get("files", {}).get("scripts/ict_scan.py") != PINNED_SCRIPT_SHA256:
            raise ValueError("frozen ICT primitive source SHA256 does not match the pin")
        for relative, digest in manifest["files"].items():
            path = (self.vendor_root / relative).resolve()
            if not path.is_relative_to(self.vendor_root) or not path.is_file():
                raise ValueError(f"frozen ICT file is missing or escapes vendor_root: {relative}")
            if _sha(path) != digest:
                raise ValueError(f"frozen ICT source SHA256 mismatch: {relative}")
        return {"schema": manifest["schema"], "repository": PINNED_REPOSITORY,
                "commit": PINNED_COMMIT, "manifest_sha256": PINNED_MANIFEST_SHA256,
                "script_sha256": PINNED_SCRIPT_SHA256, "vendor_root": str(self.vendor_root),
                "retained_files_verified": len(manifest["files"]),
                "commit_verification": "pinned_archive_manifest_and_retained_source_hashes",
                "git_objects_verified": False, "offline_only": True, "vendor_read_only": True,
                "adapter_sha256": _sha(Path(__file__))}

    def module(self) -> types.ModuleType:
        """Compile verified bytes directly; Python's source loader writes no pyc."""
        with _LOCK:
            self.verify()
            if self._module is not None:
                return self._module
            source_path = self.vendor_root / "scripts/ict_scan.py"
            source_bytes = source_path.read_bytes()
            # Check the exact bytes that will be executed, avoiding a hash/read race.
            if hashlib.sha256(source_bytes).hexdigest() != PINNED_SCRIPT_SHA256:
                raise ValueError("frozen ICT source changed while being loaded")
            suffix = hashlib.sha256(str(self.vendor_root).encode()).hexdigest()[:16]
            name = f"_token_distillation_frozen_ict_{PINNED_SCRIPT_SHA256[:16]}_{suffix}"
            module = types.ModuleType(name)
            module.__file__ = str(source_path)
            module.__package__ = ""
            sys.modules[name] = module  # dataclass resolves its module through sys.modules.
            try:
                exec(compile(source_bytes, str(source_path), "exec"), module.__dict__)
            except BaseException:
                sys.modules.pop(name, None)
                raise
            self._module = module
            return module
