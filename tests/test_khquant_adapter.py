"""Synthetic source fixtures never touch the user's raw database or models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import types

import pytest

from token_distillation.adapters import khquant
from token_distillation.adapters.khquant import KhquantSource
from token_distillation.teachers.khquant_teacher import KhquantTeacher


ADAPTER = '''
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import pandas as pd
from my_strategy.core.paths import DATA_ROOT
class MissingHistoricalBars(ValueError): pass
def normalize_symbol(symbol):
    return symbol if "." in symbol else symbol + (".SH" if symbol.startswith("6") else ".SZ")
def date_cutoff(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo: stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return stamp + pd.Timedelta(hours=23,minutes=59,seconds=59) if len(value) in (8,10) else stamp
def load_bars(symbol,start=None,end=None,as_of=None,*,db_path=None):
    with sqlite3.connect(Path(db_path).as_uri()+"?mode=ro",uri=True) as conn:
        conn.execute("PRAGMA query_only=ON")
        frame=pd.read_sql_query("SELECT stock AS symbol,date,open,high,low,close,volume,amount,available_at FROM stock_daily_normalized WHERE stock=? ORDER BY date",conn,params=[symbol])
    frame["date"]=pd.to_datetime(frame["date"])
    frame["dt"]=frame["date"]+pd.Timedelta(hours=15)
    for cutoff in (end,as_of):
        if cutoff: frame=frame.loc[frame["dt"]<=date_cutoff(cutoff)].copy()
    if start: frame=frame.loc[frame["date"]>=pd.Timestamp(start)].copy()
    if frame.empty: raise MissingHistoricalBars("no data")
    frame.attrs.update(price_basis="unadjusted",volume_unit="shares",amount_unit="CNY",data_version="fixture-visible-frame")
    return frame.reset_index(drop=True)
'''

PATHS = '''
import os
from pathlib import Path
for name in ("PROJECT_ROOT","DATA_ROOT","ARTIFACT_ROOT","METADATA_ROOT","LOG_ROOT"):
    globals()[name]=Path(os.environ["KHQUANT_"+name]).resolve()
'''


@pytest.fixture(autouse=True)
def isolated_namespace():
    previous = {k: v for k, v in sys.modules.items() if k == "my_strategy" or k.startswith("my_strategy.")}
    environment = dict(os.environ)
    binding = khquant._BINDING
    bytecode_setting = sys.dont_write_bytecode
    for key in previous:
        sys.modules.pop(key)
    khquant._BINDING = None
    yield
    for key in list(sys.modules):
        if key == "my_strategy" or key.startswith("my_strategy."):
            sys.modules.pop(key)
    sys.modules.update(previous)
    os.environ.clear()
    os.environ.update(environment)
    khquant._BINDING = binding
    sys.dont_write_bytecode = bytecode_setting


@pytest.fixture
def source(tmp_path):
    app = tmp_path / "source/app"
    for name in ("adapters", "core", "services"):
        (app / "my_strategy" / name).mkdir(parents=True)
    (app / "my_strategy/__init__.py").write_text('raise RuntimeError("must not initialize source or read .env")')
    (app / "my_strategy/adapters/czsc_adapter.py").write_text(ADAPTER)
    (app / "my_strategy/core/paths.py").write_text(PATHS)
    raw = tmp_path / "source/raw.db"
    with sqlite3.connect(raw) as conn:
        conn.execute("CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,available_at TEXT)")
        conn.executemany("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?)", [
            ("600000.SH", "2026-01-05", 10., 11., 9., 10., 100., 1000., "2026-01-05T15:00:00+08:00"),
            ("600000.SH", "2026-01-06", 10., 11., 9., 10., 100., 1000., "2026-01-07T16:00:00+08:00"),
            ("600000.SH", "2026-01-07", 10., 11., 9., 10., 100., 1000., "2026-01-07T15:00:00+08:00")])
    return KhquantSource(app, raw, tmp_path / "output/state")


def test_read_only_and_missing_path_never_creates_database(source, tmp_path):
    before = hashlib.sha256(source.raw_db.read_bytes()).hexdigest()
    assert source.status()["query_only"] is True
    with source.connection() as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM stock_daily_normalized")
    assert hashlib.sha256(source.raw_db.read_bytes()).hexdigest() == before
    missing = tmp_path / "does-not-exist.db"
    other = KhquantSource(source.app_root, missing, source.state_root)
    assert other.status()["available"] is False
    assert not missing.exists()


def test_lazy_binding_skips_source_init_and_overrides_external_state(source, monkeypatch):
    monkeypatch.setenv("KHQUANT_DATA_ROOT", "/danger/source-data")
    assert "my_strategy" not in sys.modules
    frame = source.load_bars("600000", as_of="2026-01-07")
    assert len(frame) == 3
    paths = sys.modules["my_strategy.core.paths"]
    assert paths.DATA_ROOT == source.state_root / "data"
    assert os.environ["KHQUANT_ARTIFACT_ROOT"] == str(source.state_root / "artifacts")
    assert not source.state_root.exists()


def test_as_of_excludes_future_close_and_late_available_rows(source):
    frame = source.load_bars("600000", as_of="2026-01-06")
    assert frame.date.dt.strftime("%Y-%m-%d").tolist() == ["2026-01-05"]
    with pytest.raises(ValueError):
        source.load_bars("600000", as_of="2026-01-05T14:59:00+08:00")


def test_naive_source_available_at_uses_shanghai_timezone(source):
    with sqlite3.connect(source.raw_db) as conn:
        conn.execute("UPDATE stock_daily_normalized SET available_at=? WHERE date=?",
                     ("2026-01-05 15:00:00", "2026-01-05"))
    assert len(source.load_bars("600000", as_of="2026-01-05T15:01:00+08:00")) == 1


@pytest.mark.parametrize("symbol", ["430001.BJ", "900901.SH", "200001.SZ", "510300.SH"])
def test_population_excludes_beijing_b_shares_and_etfs(source, symbol):
    with pytest.raises(ValueError, match="sh_sz_a_v1"):
        source.load_bars(symbol)


def test_namespace_from_other_root_is_rejected(source, tmp_path):
    package = types.ModuleType("my_strategy")
    package.__path__ = [str(tmp_path / "another/my_strategy")]
    sys.modules["my_strategy"] = package
    with pytest.raises(RuntimeError, match="namespace conflict"):
        source.load_bars("600000")


def test_existing_path_binding_conflict_is_rejected(source):
    source.load_bars("600000")
    sys.modules["my_strategy.core.paths"].DATA_ROOT = Path("/other/state")
    with pytest.raises(RuntimeError, match="conflicting path binding"):
        source.load_bars("600000")


def test_environment_mutation_and_rebind_are_rejected(source, tmp_path):
    source.load_bars("600000")
    other = KhquantSource(source.app_root, source.raw_db, tmp_path / "another-state")
    with pytest.raises(RuntimeError, match="conflicting path binding|already bound"):
        other.load_bars("600000")
    os.environ["KHQUANT_LOG_ROOT"] = "/other/log"
    with pytest.raises(RuntimeError, match="environment binding changed"):
        source.load_bars("600000")


def test_calendar_never_falls_back_to_observed_raw_dates(source, monkeypatch):
    loader = types.SimpleNamespace(load_verified_calendar=lambda *a, **k: pytest.fail("no calendar files exist"))
    monkeypatch.setattr(source, "module", lambda name: loader)
    with pytest.raises(ValueError, match="no independently verified calendar"):
        source.calendar_dates("2026-01-05", "2026-01-07")


def test_calendar_demands_sources_and_checks_requested_coverage(source, monkeypatch):
    path = source.runs_root / "verified-fixture/reports/calendar.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    calls = []
    def load(run_id, *, runs_root, require_sources):
        calls.append(require_sources)
        return dates, {"run_id": run_id, "source": "independent-exchange-fixture", "ancestors": []}
    monkeypatch.setattr(source, "module", lambda name: types.SimpleNamespace(load_verified_calendar=load))
    assert source.calendar_dates("2026-01-05", "2026-01-06") == dates[:2]
    assert calls == [True]
    with pytest.raises(ValueError, match="exceed verified calendar coverage"):
        source.calendar_dates("2026-01-05", "2026-01-08")


def _teacher_fixture(source, monkeypatch):
    import pandas as pd
    calls = []
    names = ("w_relative_volume20", "w_demand_score", "w_event_age", "w_spring", "w_test", "w_sos", "w_lps", "w_upthrust", "w_sow", "w_lpsy")
    def prepare(frame, **kwargs):
        calls.append(frame.date.dt.strftime("%Y-%m-%d").tolist())
        result = frame.copy()
        for name in names:
            result[name] = float("nan") if name == "w_event_age" else 0.
        result["wyckoff_input_eligible"] = True
        result["wyckoff_reason_codes"] = [[] for _ in range(len(frame))]
        result["wyckoff_event"] = "unknown"
        result.attrs["wyckoff_schema_hash"] = "fixture-w32-schema"
        return result
    actual_module = source.module
    feature_module = types.SimpleNamespace(W_INPUT_COLUMNS=names, prepare_wyckoff_inputs=prepare)
    monkeypatch.setattr(source, "module", lambda name: feature_module if name.endswith("czsc_price_volume_features") else actual_module(name))
    monkeypatch.setattr(source, "calendar_dates", lambda start, end: pd.date_range(start, end).strftime("%Y-%m-%d").tolist())
    monkeypatch.setattr(source, "calendar_identity", lambda: {"source": "verified-fixture"})
    teacher = KhquantTeacher(source, context_bars=2)
    return teacher, calls


def test_teacher_is_window_local_masks_unknown_and_structure(source, monkeypatch):
    teacher, calls = _teacher_fixture(source, monkeypatch)
    with sqlite3.connect(source.raw_db) as conn:
        conn.execute("UPDATE stock_daily_normalized SET available_at=? WHERE date=?",
                     ("2026-01-07T16:00:00+08:00", "2026-01-07"))
    with pytest.raises(ValueError, match="stale history rejected"):
        teacher.prepare("600000", as_of="2026-01-07")
    result = teacher.prepare("600000", as_of="2026-01-07T08:00:00Z")
    assert calls == [["2026-01-06", "2026-01-07"]]
    assert result["context_rows"] == 2
    assert result["dependency_start"] == "2026-01-06T15:00:00+08:00"
    assert result["available_at"] == "2026-01-07T16:00:00+08:00"
    assert result["as_of"] == "2026-01-07T16:00:00+08:00"
    assert result["requested_as_of"] == "2026-01-07T08:00:00Z"
    assert result["quality"]["availability_basis"] == "explicit_source_available_at"
    assert result["event_mask"] == [False] * 7
    assert result["event_targets"] == [None] * 7
    assert result["continuous_mask"] == [True, False, False]
    assert result["structure_mask"] == [False] * 22
    assert result["quality"]["full_history_equivalence"] is False
    json.dumps(result, allow_nan=False)
    with sqlite3.connect(source.raw_db) as conn:
        conn.execute("UPDATE stock_daily_normalized SET close=10.5 WHERE date='2026-01-05'")
        conn.execute("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?)", (
            "600000.SH", "2026-01-08", 10., 11., 9., 10., 100., 1000., "2026-01-08T15:00:00+08:00"))
    changed = teacher.prepare("600000", as_of="2026-01-07T08:00:00Z")
    assert changed["quality"]["raw_data_version"] == result["quality"]["raw_data_version"]
    assert changed["quality"]["source_load_data_version"] != result["quality"]["source_load_data_version"]
    assert changed["continuous_targets"] == result["continuous_targets"]
    assert changed["event_mask"] == result["event_mask"]


def test_teacher_date_only_means_15h_and_passes_normalized_cutoff(source, monkeypatch):
    teacher, calls = _teacher_fixture(source, monkeypatch)
    actual_load = source.load_bars
    cutoffs = []
    def capture_load(symbol, **kwargs):
        cutoffs.append(kwargs["as_of"])
        return actual_load(symbol, **kwargs)
    monkeypatch.setattr(source, "load_bars", capture_load)
    result = teacher.prepare("600000", as_of="2026-01-07")
    assert cutoffs == ["2026-01-07T15:00:00+08:00"]
    assert result["as_of"] == "2026-01-07T15:00:00+08:00"
    assert result["requested_as_of"] == "2026-01-07"
    assert result["available_at"] == "2026-01-07T15:00:00+08:00"
    # The Jan 6 row was not published until Jan 7 16:00, so cannot enter 15:00.
    assert calls == [["2026-01-05", "2026-01-07"]]


def test_teacher_discloses_assumed_close_when_source_availability_is_absent(source, monkeypatch):
    teacher, _ = _teacher_fixture(source, monkeypatch)
    actual_load = source.load_bars
    def no_publication_times(symbol, **kwargs):
        return actual_load(symbol, **kwargs).drop(columns="available_at")
    monkeypatch.setattr(source, "load_bars", no_publication_times)
    result = teacher.prepare("600000", as_of="2026-01-07")
    assert result["quality"]["availability_basis"] == "assumed_shanghai_session_close_15h_not_ingestion_evidence"


@pytest.mark.parametrize("as_of,context_bars,reason", [
    ("2026-01-07T14:59:00+08:00", 1, "at or after Shanghai 15:00"),
    ("2026-01-08", 2, "stale history rejected"),
    ("2026-01-07", 256, "exactly 256 context bars"),
])
def test_teacher_rejects_stale_intraday_and_short_context(source, as_of, context_bars, reason):
    with pytest.raises(ValueError, match=reason):
        KhquantTeacher(source, context_bars=context_bars).prepare("600000", as_of)


def test_asset_inspection_delegates_to_official_byte_verifier(source, tmp_path, monkeypatch):
    root = tmp_path / "models"
    calls = []
    def verify(path):
        calls.append(path)
        return {"ready": False, "missing": ["model.safetensors"], "errors": [], "offline_only": True}
    def module(name):
        assert name == "my_strategy.services.kronos_assets"
        return types.SimpleNamespace(asset_status=verify)
    monkeypatch.setattr(source, "module", module)
    result = source.asset_status(root)
    assert calls == [root]
    assert result["ready"] is False
    assert result["loaded"] is False
