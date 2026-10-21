#!/usr/bin/env python3
"""Package allowlisted research sources, without importing data/model code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = (
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "AGENTS.md",
    ".gitignore", "pyproject.toml", "run.py",
)
SOURCE_DIRS = ("src", "tests", "scripts", "configs", "schemas", "docs", "vendor")


def source_files(root: Path = ROOT) -> list[Path]:
    paths = [root / name for name in TOP_FILES]
    for name in SOURCE_DIRS:
        paths.extend((root / name).rglob("*"))
    paths.extend((root / ".github/workflows").glob("*.yml"))
    paths.extend((root / "audit").glob("*.json"))
    # Preserve only empty-directory markers in business output directories.
    for name in ("data", "artifacts", "logs"):
        paths.extend((root / name).rglob(".gitkeep"))
    accepted = set()
    for path in paths:
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"Source package refuses symlink: {relative}")
        if not path.is_file():
            continue
        if set(relative.parts) & {"__pycache__", ".git", ".pytest_cache", "node_modules"}:
            continue
        if (path.name == ".DS_Store" or path.suffix in {".pyc", ".pyo"}
                or ".local." in path.name or path.name.startswith(".env")):
            continue
        if path.suffix.lower() in {".db", ".pt", ".pth", ".safetensors", ".pem", ".key", ".zip"}:
            raise ValueError(f"Unexpected data, weight, credential or archive: {relative}")
        if path.stat().st_size > 5 * 1024**2:
            raise ValueError(f"Source file exceeds 5 MiB: {relative}")
        # A nested publication checkout must not recursively enter the package.
        if relative.parts[0] == "artifacts" and "publication" in relative.parts:
            continue
        accepted.add(path)
    missing = [name for name in TOP_FILES if root / name not in accepted]
    if missing:
        raise ValueError(f"Missing required top-level files: {missing}")
    return sorted(accepted, key=lambda path: path.relative_to(root).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="0.1.0")
    args = parser.parse_args()
    if not args.version or any(ch not in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-_" for ch in args.version):
        raise ValueError("Invalid archive version")
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.suffix != ".zip":
        raise ValueError("Archive must be a new .zip inside this project's artifacts directory")
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Choose a new output path")
    files = source_files()
    contents = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in files}
    manifest = {
        "schema": "source_only_publication_v1", "version": args.version,
        "repository": "https://github.com/zhaozhenxiong/token_distillation_kronons",
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                  for name, data in contents.items()},
        "raw_market_data_included": False, "model_weights_included": False,
        "training_started": False,
    }
    metadata = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"token_distillation_kronons-{args.version}/"
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in {**contents, "SOURCE_MANIFEST.json": metadata}.items():
            info = zipfile.ZipInfo(prefix + name, date_time=(2026, 10, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if name.endswith(".sh") else 0o644
            info.external_attr = (0o100000 | mode) << 16
            archive.writestr(info, data)
    with manifest_path.open("xb") as stream:
        stream.write(metadata)
    print(json.dumps({"archive": str(output.relative_to(ROOT)), "files": len(contents),
                      "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "manifest": str(manifest_path.relative_to(ROOT)),
                      "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
