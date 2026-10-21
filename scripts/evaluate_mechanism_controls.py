#!/usr/bin/env python3
"""Score already-selected 80-stock controls after the complete 240-stock study.

Evaluation only. A separate, predeclared protocol pins every input checkpoint.
No fitting, selection, calibration, promotion, or source database access.
"""
import argparse
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from token_distillation.artifacts import sha256_file, write_json
from token_distillation.config import load_config
from token_distillation.data.dataset import load_dataset
from token_distillation.evaluation.ablation import (
    date_block_interval, teacher_constant_baselines, _teacher_comparison,
)

LABELS = ("none", "czsc", "ict", "dual", "dual_shuffled", "heads_only", "dual_repeat")


def prediction_losses(rows):
    if not rows or any(r["split"] != "holdout" for r in rows):
        raise ValueError("Only a nonempty holdout population is supported")
    rows = sorted(rows, key=lambda r: (r["symbol"], r["as_of"]))
    identity = [(r["symbol"], r["as_of"], r["direction_class"]) for r in rows]
    if len(set((s, t) for s, t, _ in identity)) != len(rows):
        raise ValueError("Duplicate prediction origins")
    p = np.asarray([r["direction_probabilities"] for r in rows], dtype=float)
    y = np.asarray([r["direction_class"] for r in rows], dtype=int)
    if (p.shape != (len(rows), 3) or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(1), 1., atol=1e-6, rtol=0.)
            or (y < 0).any() or (y > 2).any()):
        raise ValueError("Invalid three-class probabilities or labels")
    return identity, {
        "log_loss": -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)),
        "brier": ((p - np.eye(3)[y]) ** 2).sum(1),
        "accuracy": (p.argmax(1) == y).astype(float),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/ablation240.toml")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    c = load_config(args.config)
    protocol_path = c.output_path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    if (protocol["schema"] != "mechanism80_holdout_evaluation_protocol_v1"
            or protocol["config_sha256"] != sha256_file(c.config_path)
            or protocol["script_sha256"] != sha256_file(Path(__file__))
            or tuple(r["label"] for r in protocol["runs"]) != LABELS):
        raise ValueError("Frozen evaluation protocol differs from the current inputs")
    study = c.output_path(protocol["study"])
    dataset = c.output_path(protocol["dataset"])
    if sha256_file(dataset / "manifest.json") != protocol["dataset_manifest_sha256"]:
        raise ValueError("Frozen 80-stock dataset manifest changed")
    if not args.execute:
        print(json.dumps({"status": "plan_only", "training_started": False,
                          "evaluation_started": False, "runs": len(LABELS)}))
        return
    if json.loads((study / "study_status.json").read_text())["status"] != "completed":
        raise ValueError("240-stock training and analysis must complete before this evaluation")
    output = c.output_path(protocol["output_dir"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    status = {"schema": "mechanism80_holdout_execution_v1", "status": "running",
              "started_at": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": sha256_file(protocol_path), "runs": [],
              "training_performed": False, "selection_performed": False, "promoted": False}
    status_path = output / "execution.json"
    def save():
        write_json(c, status_path, status, replace=status_path.exists())
    save()
    summaries = {}
    try:
        for run in protocol["runs"]:
            label = run["label"]
            training_path = c.output_path(run["training_summary"])
            if sha256_file(training_path) != run["training_summary_sha256"]:
                raise ValueError("Training summary changed: " + label)
            training = json.loads(training_path.read_text())
            checkpoint = c.output_path(training["checkpoint"])
            if sha256_file(checkpoint) != run["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint changed: " + label)
            report = output / (label + "_evaluation")
            command = [sys.executable, str(c.root / "run.py"), "--config", str(c.config_path),
                       "evaluate", "--dataset", str(dataset), "--checkpoint", str(checkpoint),
                       "--output-dir", str(report), "--split", "holdout", "--device", "mps",
                       "--batch-size", "32"]
            entry = {"label": label, "status": "running", "command": command,
                     "started_at": datetime.now(timezone.utc).isoformat()}
            status["runs"].append(entry)
            save()
            print("mechanism holdout starting " + label, file=sys.stderr, flush=True)
            with (output / (label + "_stdout.json")).open("x") as stdout, (output / (label + "_stderr.log")).open("x") as stderr:
                result = subprocess.run(command, cwd=c.root, stdout=stdout, stderr=stderr)
            if result.returncode:
                raise RuntimeError(f"{label} evaluation failed, exit={result.returncode}")
            evaluation_path = report / "evaluation_summary.json"
            evaluation = json.loads(evaluation_path.read_text())
            if evaluation["checkpoint_sha256"] != run["checkpoint_sha256"]:
                raise ValueError("Evaluated checkpoint differs from the frozen selection")
            entry.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat(),
                         evaluation_summary_sha256=sha256_file(evaluation_path),
                         predictions_sha256=sha256_file(report / "predictions.json"))
            summaries[label] = (training, evaluation)
            save()
            print("mechanism holdout completed " + label, file=sys.stderr, flush=True)
        _, samples = load_dataset(c, dataset)
        train = [r for r in samples if r["split"] == "train"]
        held = [r for r in samples if r["split"] == "holdout"]
        reference, losses, metrics = None, {}, {}
        for label in LABELS:
            training, evaluation = summaries[label]
            prediction_path = output / (label + "_evaluation") / "predictions.json"
            identity, loss = prediction_losses(json.loads(prediction_path.read_text())["rows"])
            if reference is None:
                reference = identity
            if identity != reference:
                raise ValueError("Mechanism controls are not paired on the same holdout rows")
            if len(identity) != len(held):
                raise ValueError("Evaluation population differs from the frozen holdout")
            measured = evaluation["stages"]["holdout"]
            for name in ("log_loss", "brier", "accuracy"):
                if not math.isclose(float(loss[name].mean()), measured["direction"][name], abs_tol=1e-6):
                    raise ValueError("Raw probability recomputation differs from evaluation summary")
            constants = teacher_constant_baselines(train, held, training["teacher_statistics"])
            metrics[label] = {"selected_optimizer_step": training["selected_optimizer_step"],
                              "direction": {k: float(v.mean()) for k, v in loss.items()},
                              "teacher_fidelity": _teacher_comparison(measured["teachers"], constants),
                              "training_summary_sha256": sha256_file(c.output_path(protocol["runs"][LABELS.index(label)]["training_summary"]))}
            losses[label] = loss
        dates = [r[1] for r in reference]
        pairs = [(a, "none") for a in LABELS if a != "none"] + [
            ("dual", "dual_shuffled"), ("dual", "heads_only"), ("dual", "dual_repeat")]
        comparisons = {}
        for left, right in pairs:
            comparisons[left + "_minus_" + right] = {
                name: {"block2": date_block_interval(losses[left][name] - losses[right][name], dates),
                       "block5": date_block_interval(losses[left][name] - losses[right][name], dates, block_length=5)}
                for name in ("log_loss", "brier")}
        report = {"schema": "mechanism80_supplementary_holdout_review_v1",
                  "status": "completed", "protocol_sha256": sha256_file(protocol_path),
                  "rows": len(reference), "symbols": len({r[0] for r in reference}),
                  "origin_dates": len(set(dates)), "seed": 42, "metrics": metrics,
                  "paired_comparisons": comparisons, "training_performed": False,
                  "limitations": ["supplementary single-seed mechanism comparison; no multiplicity adjustment",
                                  "nested 80-stock subset shares market dates with the 240-stock study; not independent replication",
                                  "checkpoints selected previously by the same tune rule; selected optimizer steps can differ",
                                  "frozen-backbone control changes both backbone adaptation and its optimization path",
                                  "one exact-objective repeat pair cannot characterize all backend numerical variability",
                                  "current dates were used by earlier research; retrospective only"],
                  "qualified": False, "promoted": False}
        write_json(c, output / "comparison.json", report)
        parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>80股机制留出对照</title><style>body{font:15px system-ui;max-width:1100px;margin:32px auto;padding:20px;color:#172536}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #d8e0e6;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}</style><h1>80股机制留出对照</h1>',
                 f'<p>固定种子42；{len(reference)}个窗口、{len(set(dates))}个预测日期。仅评估已经选好的检查点，没有新增训练、选点或校准。80股为240股嵌套子集，与主实验共享市场日期，不能视为独立复现。</p>',
                 '<table><tr><th>配置</th><th>选中步数</th><th>LogLoss</th><th>Brier</th><th>准确率</th><th>Δ LogLoss vs 无教师</th></tr>']
        for label, row in metrics.items():
            m = row["direction"]
            delta = m["log_loss"] - metrics["none"]["direction"]["log_loss"]
            parts.append(f'<tr><td>{html.escape(label)}</td><td>{row["selected_optimizer_step"]}</td><td>{m["log_loss"]:.6f}</td><td>{m["brier"]:.6f}</td><td>{m["accuracy"]:.2%}</td><td>{delta:+.6f}</td></tr>')
        parts.append('</table><h2>真实双教师与机制控制</h2><table><tr><th>差值（越小越好）</th><th>Δ LogLoss</th><th>2原点块95%描述区间</th><th>5原点块敏感性区间</th></tr>')
        for key in ("dual_minus_dual_shuffled", "dual_minus_heads_only", "dual_minus_dual_repeat"):
            m = comparisons[key]["log_loss"]
            parts.append(f'<tr><td>{html.escape(key)}</td><td>{m["block2"]["mean"]:+.6f}</td><td>{html.escape(str(m["block2"]["ci95"]))}</td><td>{html.escape(str(m["block5"]["ci95"]))}</td></tr>')
        parts.append('</table><p>单种子与有限时间块不足以证明普遍有效。辅助头重建不单独证明主干语义吸收；冻结主干与可训练主干的选中步数及优化路径可能不同。完整教师重建和配对区间见<a href="comparison.json">证据JSON</a>；<a href="../ablation240_20261021/index.html">240股主实验</a>。</p></html>')
        (output / "index.html").write_text("\n".join(parts))
        status.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat(),
                      comparison_sha256=sha256_file(output / "comparison.json"))
        save()
        print(json.dumps({"status": "completed", "report": str(output / "index.html")}))
    except BaseException as exc:
        status.update(status="failed", error=str(exc), completed_at=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__ == "__main__":
    main()
