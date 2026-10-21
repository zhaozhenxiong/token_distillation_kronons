"""Read-only, explicitly bound KHQuant inputs; no source project initialization."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import threading
import types
from typing import Any


_IMPORT_LOCK = threading.RLock()
_BINDING: tuple[str, str, str] | None = None
_IDENTITY_FILES = (
    "my_strategy/adapters/czsc_adapter.py",
    "my_strategy/services/czsc_price_volume_features.py",
    "my_strategy/services/czsc_price_volume_events.py",
    "my_strategy/services/czsc_research_calendar.py",
    "my_strategy/services/czsc_trend_support.py",
    "my_strategy/core/paths.py",
    "my_strategy/core/run_context.py",
    "my_strategy/core/tz.py",
    "my_strategy/runtime_env.py",
    "my_strategy/services/kronos_assets.py",
)


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def availability_times(values):
    """Interpret naive source timestamps in Shanghai, and aware stamps exactly."""
    import pandas as pd
    converted = []
    for value in values:
        stamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(stamp):
            converted.append(pd.NaT)
        else:
            converted.append((stamp.tz_localize("Asia/Shanghai") if stamp.tzinfo is None else stamp).tz_convert("UTC"))
    return pd.Series(converted, index=values.index, dtype="datetime64[ns, UTC]")


class KhquantSource:
    """One KHQuant source and isolated output binding per Python process.

    A package spec deliberately skips ``my_strategy.__init__`` because that
    initializer reads the source checkout's .env. Actual adapter/service files
    are imported normally from their original location, without being copied.
    Binding disables bytecode writes in this process, including delayed imports,
    to avoid creating source-checkout __pycache__ files.
    """

    def __init__(self, app_root: str | Path, raw_db: str | Path, state_root: str | Path,
                 *, calendar_path: str | Path | None = None,
                 runs_root: str | Path | None = None):
        self.app_root = Path(app_root).expanduser().resolve()
        self.raw_db = Path(raw_db).expanduser().resolve()
        self.state_root = Path(state_root).expanduser().resolve()
        self.calendar_path = Path(calendar_path).expanduser().resolve() if calendar_path else None
        self.runs_root = Path(runs_root).expanduser().resolve() if runs_root else self.app_root / "my_strategy/artifacts/runs"
        if not (self.app_root / "my_strategy/adapters/czsc_adapter.py").is_file():
            raise FileNotFoundError(f"KHQuant app root lacks frozen adapter: {self.app_root}")
        if self.state_root.is_relative_to(self.app_root):
            raise ValueError("state_root must be outside the read-only KHQuant app")
        self._calendar_metadata: dict[str, Any] | None = None

    @property
    def _environment(self) -> dict[str, str]:
        data = self.state_root / "data"
        return {"KHQUANT_PROJECT_ROOT": str(self.app_root),
                "KHQUANT_DATA_ROOT": str(data),
                "KHQUANT_ARTIFACT_ROOT": str(self.state_root / "artifacts"),
                "KHQUANT_METADATA_ROOT": str(data / "metadata"),
                "KHQUANT_LOG_ROOT": str(self.state_root / "logs"),
                "KHQUANT_RAW_DB": str(self.raw_db),
                "KHQUANT_BASIC_KB_DB": str(data / "raw/stock_basic/basic_kb.db"),
                "KHQUANT_BASIC_FLUENT_DB": str(data / "processed/stock_fluent/basic_fluent.db"),
                "CZSC_HOME": str(self.state_root / "czsc-cache")}

    def _check_namespace(self) -> None:
        package_root = self.app_root / "my_strategy"
        for name, module in tuple(sys.modules.items()):
            if name != "my_strategy" and not name.startswith("my_strategy."):
                continue
            if module is None:
                raise RuntimeError(f"KHQuant namespace has an incomplete module: {name}")
            locations = list(getattr(module, "__path__", ()))
            filename = getattr(module, "__file__", None)
            if filename:
                locations.append(filename)
            if not locations or any(not Path(p).resolve().is_relative_to(package_root) for p in locations):
                raise RuntimeError(f"KHQuant namespace conflict: {name}")
        paths = sys.modules.get("my_strategy.core.paths")
        if paths:
            for key in ("PROJECT_ROOT", "DATA_ROOT", "ARTIFACT_ROOT", "METADATA_ROOT", "LOG_ROOT"):
                expected = Path(self._environment["KHQUANT_" + key]).resolve()
                if Path(getattr(paths, key)).resolve() != expected:
                    raise RuntimeError(f"KHQuant already has a conflicting path binding: {key}")

    def module(self, name: str):
        """Lazy internal import with source identity and output binding checks."""
        global _BINDING
        if not name.startswith("my_strategy."):
            raise ValueError("only KHQuant my_strategy modules can be loaded")
        binding = (str(self.app_root), str(self.raw_db), str(self.state_root))
        with _IMPORT_LOCK:
            self._check_namespace()
            if _BINDING is not None and _BINDING != binding:
                raise RuntimeError("KHQuant process is already bound to another source/state root")
            if _BINDING is not None:
                if any(os.environ.get(key) != value for key, value in self._environment.items()):
                    raise RuntimeError("KHQuant environment binding changed after initialization")
            os.environ.update(self._environment)
            sys.dont_write_bytecode = True
            if "my_strategy" not in sys.modules:
                package_path = self.app_root / "my_strategy"
                package = types.ModuleType("my_strategy")
                package.__file__ = str(package_path / "__init__.py")
                package.__package__ = "my_strategy"
                package.__path__ = [str(package_path)]
                package.__spec__ = importlib.machinery.ModuleSpec("my_strategy", loader=None, is_package=True)
                package.__spec__.submodule_search_locations = list(package.__path__)
                sys.modules["my_strategy"] = package
            _BINDING = binding
            result = importlib.import_module(name)
            self._check_namespace()
            return result

    @contextmanager
    def connection(self):
        """Never creates or writes a database, including when the path is absent."""
        if not self.raw_db.is_file():
            raise FileNotFoundError(self.raw_db)
        connection = sqlite3.connect(self.raw_db.as_uri() + "?mode=ro", uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        try:
            yield connection
        finally:
            connection.close()

    def load_bars(self, symbol: str, start: str | None = None, end: str | None = None,
                  as_of: str | None = None):
        adapter = self.module("my_strategy.adapters.czsc_adapter")
        normalized = adapter.normalize_symbol(symbol)
        if not re.fullmatch(r"(?:60\d{4}|68\d{4})\.SH|(?:00\d{4}|30\d{4})\.SZ", normalized):
            raise ValueError("only sh_sz_a_v1 Shanghai/Shenzhen A shares are admitted")
        frame = adapter.load_bars(normalized, start=start, end=end, as_of=as_of, db_path=self.raw_db)
        if "available_at" in frame and (end is not None or as_of is not None):
            import pandas as pd
            cutoffs = [adapter.date_cutoff(value) for value in (end, as_of) if value is not None]
            cutoff = min(cutoffs).tz_localize("Asia/Shanghai").tz_convert("UTC")
            visibility = availability_times(frame["available_at"])
            if visibility.isna().any():
                raise ValueError("raw rows contain unknown available_at")
            frame = frame.loc[visibility <= cutoff].copy().reset_index(drop=True)
            if frame.empty:
                raise adapter.MissingHistoricalBars(f"{normalized} has no available raw rows at cutoff")
            frame.attrs["data_version"] = hashlib.sha256(frame.to_csv(index=False, float_format="%.10g").encode()).hexdigest()
        return frame

    def _calendar(self) -> tuple[list[str], dict[str, Any]]:
        loader = self.module("my_strategy.services.czsc_research_calendar")
        candidates = [self.calendar_path] if self.calendar_path else sorted(self.runs_root.glob("*/reports/calendar.json"))
        valid, rejected = [], []
        for path in candidates:
            try:
                if path.name != "calendar.json" or path.parent.name != "reports":
                    raise ValueError("calendar_path must be <runs>/<run_id>/reports/calendar.json")
                dates, metadata = loader.load_verified_calendar(path.parent.parent.name,
                    runs_root=path.parent.parent.parent, require_sources=True)
                # Parent lineage must retain independently verified provenance too.
                for parent in metadata.get("ancestors", []):
                    loader.load_verified_calendar(parent["run_id"], runs_root=path.parent.parent.parent, require_sources=True)
                valid.append((dates, metadata))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                if self.calendar_path:
                    raise ValueError(f"explicit verified calendar failed: {path}: {exc}") from exc
                rejected.append({"path": str(path), "reason": str(exc)})
        if not valid:
            raise ValueError(f"no independently verified calendar with intact source hashes: {rejected}")
        dates, metadata = max(valid, key=lambda item: (item[0][-1], len(item[0]), item[1]["run_id"]))
        self._calendar_metadata = {**metadata, "rejected_candidates": rejected}
        return dates, self._calendar_metadata

    def calendar_dates(self, start: str | None = None, end: str | None = None) -> list[str]:
        from datetime import date
        dates, _ = self._calendar()
        start = date.fromisoformat(start).isoformat() if start else dates[0]
        end = date.fromisoformat(end).isoformat() if end else dates[-1]
        if start > end:
            raise ValueError("calendar start must not exceed end")
        if start < dates[0] or end > dates[-1]:
            raise ValueError(f"requested dates exceed verified calendar coverage {dates[0]}..{dates[-1]}")
        return [day for day in dates if start <= day <= end]

    def calendar_identity(self) -> dict[str, Any]:
        _, metadata = self._calendar()
        return dict(metadata)

    def source_identity(self) -> dict[str, Any]:
        files = {name: _sha(self.app_root / name) for name in _IDENTITY_FILES if (self.app_root / name).is_file()}
        contract = {"app_root": str(self.app_root), "files": files,
                    "bridge_sha256": _sha(Path(__file__)),
                    "scope": "sh_sz_a_v1", "raw_db": str(self.raw_db),
                    "raw_db_hash_policy": "per_loaded_frame_data_version;not_full_database_hash"}
        contract["code_sha256"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        return contract

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {"app_root": str(self.app_root), "raw_db": str(self.raw_db),
            "state_root": str(self.state_root), "read_only": True, "available": False,
            "calendar_basis": "independent_verified_run_artifact", "source_identity": self.source_identity()}
        try:
            with self.connection() as connection:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(stock_daily_normalized)")}
                required = {"stock", "date", "open", "high", "low", "close", "volume", "amount"}
                result.update(available=required <= columns, columns=sorted(columns),
                              query_only=bool(connection.execute("PRAGMA query_only").fetchone()[0]))
                if result["available"]:
                    result["latest_date"] = connection.execute("SELECT MAX(date) FROM stock_daily_normalized").fetchone()[0]
                if not result["available"]:
                    result["reason"] = "raw database schema missing required columns"
        except (OSError, sqlite3.Error) as exc:
            result["reason"] = str(exc)
        return result

    def asset_status(self, model_root: str | Path) -> dict[str, Any]:
        """Verify actual official pinned asset bytes, offline and without torch."""
        root = Path(model_root).expanduser().resolve()
        module = self.module("my_strategy.services.kronos_assets")
        return {**module.asset_status(root), "root": str(root), "local_only": True, "loaded": False}
