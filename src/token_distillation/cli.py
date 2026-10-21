"""Read-only inspection plus explicit dataset, training and evaluation commands."""
from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import platform
import sys

from . import __version__
from .artifacts import dumps, write_json
from .config import ProjectConfig, load_config


def _source(config: ProjectConfig):
    from .adapters.khquant import KhquantSource
    return KhquantSource(config.source_app, config.raw_db, config.state_root,
                         calendar_path=config.calendar_path)


def doctor(config: ProjectConfig, *, verify_assets: bool = False) -> dict:
    external_paths = {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in (("source_app", config.source_app), ("raw_db", config.raw_db),
                           ("kronos_root", config.kronos_root), ("source_python", config.source_python))
    }
    report = {
        "schema": "distillation_doctor_v1", "version": __version__,
        "workspace": str(config.root), "python": sys.version.split()[0],
        "python_executable": sys.executable, "machine": platform.machine(),
        "external_paths": external_paths, "training_started": False,
        "models_loaded": False, "source_write_enabled": False,
        "stage": config.values["training"]["stage"], "training_ready": False,
        "pipeline_implemented": True,
    }
    report["installed_versions"] = {}
    for name in ("numpy", "pandas", "torch", "safetensors", "czsc"):
        try:
            report["installed_versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["installed_versions"][name] = None
    source = _source(config)
    report["source"] = source.status()
    report["calendar"] = source.calendar_identity()
    report["assets"] = (source.asset_status(config.kronos_root) if verify_assets else
                         {"verification": "not_requested", "models_loaded": False})
    if config.values["ict"]["enabled"]:
        from .adapters.ict import FrozenIctAdapter
        report["ict_source"] = FrozenIctAdapter(config.ict_vendor_root).verify()
    report["status"] = "ready_for_data_preparation"
    if not report["source"].get("available"):
        report["status"] = "raw_data_unavailable"
    if verify_assets and not report["assets"].get("ready"):
        report["status"] = "assets_unavailable"
    if not all(item["exists"] for item in external_paths.values()):
        report["status"] = "external_path_missing"
    report["training_ready"] = bool(report["status"] == "ready_for_data_preparation"
        and verify_assets and report["assets"].get("ready")
        and config.values["training"]["execution_enabled"])
    return report


def source_references(config: ProjectConfig, *, verify_assets: bool = False) -> dict:
    source = _source(config)
    info = config.raw_db.stat()
    result = {
        "schema": "distillation_source_references_v1", "read_only": True,
        "source_app": str(config.source_app), "raw_db": str(config.raw_db),
        "kronos_root": str(config.kronos_root),
        "raw_db_identity": {"kind": "file_stat_not_content_hash", "size": info.st_size,
                            "mtime_ns": info.st_mtime_ns, "inode": info.st_ino},
        "raw_db_warning": "Live SQLite/WAL is not an immutable training snapshot; freeze actual row content per shard before training.",
        "code": source.source_identity(), "calendar": source.calendar_identity(),
        "assets": source.asset_status(config.kronos_root) if verify_assets else {"verification": "not_requested"},
        "training_started": False, "models_loaded": False,
    }
    if config.values["ict"]["enabled"]:
        from .adapters.ict import FrozenIctAdapter
        result["ict_source"] = FrozenIctAdapter(config.ict_vendor_root).verify()
    return result


def inspect_window(config: ProjectConfig, args) -> dict:
    import pandas as pd
    from .data.windows import build_window
    from .data.normalization import normalize_window
    from .data.quality import validate_raw_quality
    source = _source(config)
    through = args.label_through or args.as_of
    if through[:10] < args.as_of[:10]:
        raise ValueError("label-through cannot precede as-of")
    frame = source.load_bars(args.symbol, end=through)
    dates = source.calendar_dates(str(frame.date.min().date()))
    indexed = frame.set_index(pd.DatetimeIndex(frame["date"]))
    origin = pd.Timestamp(args.as_of)
    normalized_origin = (origin.tz_convert("Asia/Shanghai").tz_localize(None).to_pydatetime()
                         if origin.tzinfo is not None else args.as_of)
    if "available_at" in indexed:
        indexed["available_at"] = [
            (stamp.tz_convert("Asia/Shanghai").tz_localize(None).to_pydatetime()
             if stamp.tzinfo is not None else stamp.to_pydatetime())
            for stamp in map(pd.Timestamp, indexed["available_at"])
        ]
    symbol = str(frame["symbol"].iloc[-1])
    window = build_window(indexed, symbol=symbol, as_of=normalized_origin,
                          context_length=config.context_length, horizon=config.horizon,
                          calendar_dates=dates)
    selected = indexed.loc[[*window.context_dates, *window.future_dates]].copy()
    selected.attrs = dict(frame.attrs)
    verified_sources = source.module("my_strategy.services.czsc_trend_support").VERIFIED_SOURCES
    quality = validate_raw_quality(selected, verified_sources=verified_sources)
    normalized = normalize_window(window.context, window.future,
                                  epsilon=config.values["data"]["normalization_epsilon"],
                                  clip=config.values["data"]["normalization_clip"])
    result = {
        "schema": "distillation_window_preview_v1", "symbol": window.symbol,
        "as_of": window.as_of, "available_at": window.available_at,
        "context_shape": list(window.context.shape), "context_start": window.context_dates[0],
        "context_end": window.context_dates[-1], "future_dates": window.future_dates,
        "label_end": window.label_end, "label_mature_at": window.label_mature_at,
        "future_status": window.future_status,
        "future_status_reason": window.future_status_reason,
        "availability_basis": ("source_available_at" if "available_at" in indexed else
                               "assumed_15h_session_close_no_ingestion_timestamp"),
        "normalization": normalized,
        "quality": quality, "training_started": False, "teacher_called": False,
    }
    if args.include_values:
        result["raw_context"] = window.context
        result["raw_future"] = window.future
    else:
        # Never print all input arrays in an ordinary inspection.
        result["normalization"] = {"history_only": True, "stats": normalized.stats}
    return result


def teacher_preview(config: ProjectConfig, args) -> dict:
    from .teachers.khquant_teacher import KhquantTeacher
    teacher = KhquantTeacher(_source(config), context_bars=config.context_length)
    result = teacher.prepare(args.symbol, args.as_of)
    return {"schema": "distillation_teacher_preview_v1", "training_started": False,
            "preview": result}


def ict_preview(config: ProjectConfig, args) -> dict:
    if not config.values["ict"]["enabled"]:
        raise ValueError("ICT teacher is disabled in configuration")
    from .teachers.ict_teacher import IctTeacher
    teacher = IctTeacher(_source(config), config.ict_vendor_root,
                         context_bars=config.context_length,
                         swing_width=config.values["ict"]["swing_width"])
    return {"schema": "distillation_ict_preview_v1", "training_started": False,
            "preview": teacher.prepare(args.symbol, args.as_of)}


def teachers_preview(config: ProjectConfig, args) -> dict:
    from .teachers.multi_teacher import MultiTeacherPreview
    teachers = MultiTeacherPreview(_source(config), config.ict_vendor_root,
                                    context_bars=config.context_length,
                                    swing_width=config.values["ict"]["swing_width"],
                                    include_ict=config.values["ict"]["enabled"])
    return teachers.prepare(args.symbol, args.as_of)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="CZSC → Kronos 蒸馏工程：只读检查、因果预览与训练准备")
    result.add_argument("--config", type=Path, help="TOML configuration; relative data paths remain project-relative")
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    for name, help_text in (("doctor", "检查只读来源、日历和依赖"),
                            ("sources", "生成外部路径和来源身份清单"),
                            ("inspect-window", "预览一个因果行情窗口"),
                            ("teacher-preview", "预览一个窗口内量价教师样本"),
                            ("ict-preview", "预览一个因果ICT原子教师样本"),
                            ("teachers-preview", "预览同一窗口的量价与ICT联合监督"),
                            ("train-plan", "显示训练配方与资格限制，不启动训练")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--output", type=Path, help="Optional JSON output, restricted to this workspace")
        command.add_argument("--replace", action="store_true", help="Replace an existing generated JSON only")
        if name in ("doctor", "sources"):
            command.add_argument("--verify-assets", action="store_true", help="Hash official asset bytes without loading tensors")
        if name in ("inspect-window", "teacher-preview", "ict-preview", "teachers-preview"):
            command.add_argument("--symbol", required=True, help="Explicit SH/SZ A-share code, for example 600519.SH")
            command.add_argument("--as-of", required=True, help="Decision date or timestamp in Asia/Shanghai")
        if name == "inspect-window":
            command.add_argument("--label-through", help="Explicit stored observation cutoff for mature labels")
            command.add_argument("--include-values", action="store_true", help="Include this preview's actual arrays")
    build = commands.add_parser("build-dataset", help="显式生成只读来源的因果双教师数据分片")
    build.add_argument("--symbols", nargs="+", required=True, help="Explicit SH/SZ A-share population")
    build.add_argument("--start", required=True, help="First prediction origin, YYYY-MM-DD")
    build.add_argument("--end", required=True, help="Last prediction origin, YYYY-MM-DD")
    build.add_argument("--label-through", required=True, help="Stored observation cutoff for future-label maturity")
    build.add_argument("--output-dir", type=Path, required=True, help="New dataset directory inside data or artifacts")
    build.add_argument("--stride", type=int, default=5, help="Trading-session spacing between origins per symbol")
    build.add_argument("--max-per-split", type=int, help="Evenly selected cap for each chronological stage")
    filtered = commands.add_parser("filter-dataset", help="显式派生数据集，排除未来标签区间内可疑未复权价格断点")
    filtered.add_argument("--dataset", type=Path, required=True)
    filtered.add_argument("--output-dir", type=Path, required=True)
    train = commands.add_parser("train", help="显式运行冻结 tokenizer、LoRA 与双教师蒸馏")
    train.add_argument("--dataset", type=Path, required=True, help="Dataset manifest or its directory")
    train.add_argument("--output-dir", type=Path, required=True, help="New checkpoint/run directory inside this workspace")
    train.add_argument("--device", choices=("auto", "cpu", "mps", "cuda", "cuda:0"))
    train.add_argument("--max-steps", type=int, default=24, help="Actual optimizer-update budget")
    train.add_argument("--learning-rate", type=float, default=0.0001)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--gradient-accumulation-steps", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--mechanism-control", choices=("none", "shuffle", "frozen_backbone"), default="none")
    train.add_argument("--teacher-mode", choices=("none", "czsc", "ict", "dual"), default="dual")
    train.add_argument("--auxiliary-budget", type=float, default=0.6)
    train.add_argument("--selection-metric", choices=("direction_log_loss",), default="direction_log_loss")
    train.add_argument("--evaluation-interval", type=int)
    train.add_argument("--evaluation-batch-size", type=int, default=4)
    evaluate = commands.add_parser("evaluate", help="显式重载学生检查点并评估冻结阶段")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--split", choices=("tune", "calibration", "policy", "holdout", "all"), default="holdout")
    evaluate.add_argument("--device", choices=("auto", "cpu", "mps", "cuda", "cuda:0"))
    evaluate.add_argument("--batch-size", type=int, default=4)
    for name in ("ablation-plan", "run-ablation"):
        command = commands.add_parser(name, help="预览固定消融协议" if name == "ablation-plan" else "显式执行四组多种子消融")
        command.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
        command.add_argument("--max-steps", type=int, default=512)
        command.add_argument("--batch-size", type=int, default=4)
        command.add_argument("--auxiliary-budget", type=float, default=0.6)
        command.add_argument("--learning-rate", type=float, default=0.0001)
        command.add_argument("--evaluation-interval", type=int, default=128)
        command.add_argument("--evaluation-batch-size", type=int, default=4)
        if name == "run-ablation":
            command.add_argument("--dataset", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
            command.add_argument("--device", choices=("auto", "cpu", "mps", "cuda", "cuda:0"), default="mps")
        else:
            command.add_argument("--output", type=Path)
            command.add_argument("--replace", action="store_true")
    analyze = commands.add_parser("analyze-ablation", help="核验消融完整性并计算配对比较与诊断")
    analyze.add_argument("--experiment-dir", type=Path, required=True)
    analyze.add_argument("--dataset", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            result = doctor(config, verify_assets=args.verify_assets)
        elif args.command == "sources":
            result = source_references(config, verify_assets=args.verify_assets)
        elif args.command == "inspect-window":
            result = inspect_window(config, args)
        elif args.command == "teacher-preview":
            result = teacher_preview(config, args)
        elif args.command == "ict-preview":
            result = ict_preview(config, args)
        elif args.command == "teachers-preview":
            result = teachers_preview(config, args)
        elif args.command == "build-dataset":
            from .data.dataset import build_dataset
            result = build_dataset(config, symbols=args.symbols, start=args.start, end=args.end,
                label_through=args.label_through, output_dir=args.output_dir, stride=args.stride,
                max_per_split=args.max_per_split)
        elif args.command == "filter-dataset":
            from .data.filtering import derive_dataset
            result = derive_dataset(config, dataset=args.dataset, output_dir=args.output_dir)
        elif args.command == "train":
            from .training.runner import train
            settings = config.values["training"]
            result = train(config, dataset=args.dataset, output_dir=args.output_dir,
                           mechanism_control=args.mechanism_control,
                device=args.device or settings["device"], max_steps=args.max_steps,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size if args.batch_size is not None else settings["micro_batch_size"],
                gradient_accumulation_steps=(args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None
                    else settings["gradient_accumulation_steps"]),
                seed=args.seed if args.seed is not None else settings["seed"],
                teacher_mode=args.teacher_mode, auxiliary_budget=args.auxiliary_budget,
                selection_metric=args.selection_metric, evaluation_interval=args.evaluation_interval,
                evaluation_batch_size=args.evaluation_batch_size)
        elif args.command == "evaluate":
            from .training.runner import evaluate
            result = evaluate(config, dataset=args.dataset, checkpoint=args.checkpoint,
                output_dir=args.output_dir, split=args.split,
                device=args.device or config.values["training"]["device"], batch_size=args.batch_size)
        elif args.command in {"ablation-plan", "run-ablation"}:
            from .training.ablation import ablation_plan, run_ablation
            options = dict(seeds=args.seeds, max_steps=args.max_steps, batch_size=args.batch_size,
                auxiliary_budget=args.auxiliary_budget, learning_rate=args.learning_rate,
                evaluation_interval=args.evaluation_interval, evaluation_batch_size=args.evaluation_batch_size)
            result = (ablation_plan(config, **options) if args.command == "ablation-plan" else
                run_ablation(config, dataset=args.dataset, output_dir=args.output_dir, device=args.device, **options))
        elif args.command == "analyze-ablation":
            from .evaluation.ablation import analyze_ablation
            result = analyze_ablation(config, experiment_dir=args.experiment_dir,
                dataset=args.dataset, output_dir=args.output_dir)
        else:
            from .training.planning import training_plan
            result = training_plan(config)
        if getattr(args, "output", None):
            output = write_json(config, args.output, result, replace=args.replace)
            print(dumps({"output": str(output), "schema": result.get("schema"),
                         "status": result.get("status", "saved"),
                         "training_started": result.get("training_started", False)}))
        else:
            print(dumps(result))
        return 2 if args.command == "doctor" and result["status"] != "ready_for_data_preparation" else 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(dumps({"status": "error", "error": str(exc),
                     "execution_attempted": args.command in {"build-dataset", "filter-dataset", "train", "evaluate", "run-ablation"},
                     "training_completed": False,
                     "training_started": None if args.command in {"train", "run-ablation"} else False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
