#!/usr/bin/env python3
"""Render reviewed comparison JSON using an existing Matplotlib environment.

This renderer reads scalar analysis results only. It cannot load market data,
models or optimizer state, and it never installs packages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from token_distillation.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/ablation.toml"))
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    source, output = config.output_path(args.comparison), config.output_path(args.output_dir)
    comparison = json.loads(source.read_text())
    if (comparison.get("schema") != "distillation_ablation_comparison_v1"
            or comparison.get("status") != "completed_retrospective_comparison"):
        raise ValueError("Only a completed ablation comparison can be rendered")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = config.output_path("artifacts/source_runtime/matplotlib")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    settings = {"font.family": "DejaVu Sans", "font.size": 11,
                "axes.spines.top": False, "axes.spines.right": False,
                "svg.hashsalt": source_sha}
    figures = []
    holdout = comparison["stages"]["holdout"]
    colors = {"none": "#6C7882", "czsc": "#376BA4", "ict": "#247F70", "dual": "#BF6628"}
    names = {"none": "No teacher", "czsc": "CZSC", "ict": "ICT", "dual": "Dual"}
    for reference, arms, stem, title in (
        ("none", ("czsc", "ict", "dual"), "ablation_transfer_matplotlib", "Does distillation improve direction prediction?"),
        ("train_prior", ("none", "czsc", "ict", "dual"), "ablation_prior_matplotlib", "Direction models versus the training prior"),
    ):
        targets = [output / (stem + suffix) for suffix in (".png", ".svg")]
        if any(path.exists() for path in targets):
            raise FileExistsError("Figure outputs already exist; use a new output directory")
        effects = [holdout["comparisons"][arm]["vs_" + reference]["log_loss"] for arm in arms]
        extent = [0.]
        for effect in effects:
            extent.extend(effect["values"])
            extent.extend(effect["date_block_uncertainty"]["ci95"] or [effect["mean"]])
        span = max(max(extent) - min(extent), 1e-5)
        limits = min(extent) - span * .15, max(extent) + span * .15
        with plt.rc_context(settings):
            fig, ax = plt.subplots(figsize=(10, 4.6))
            fig.subplots_adjust(left=.12, right=.82, bottom=.25, top=.78)
            fig.suptitle(title, x=.12, y=.96, ha="left", fontsize=16, fontweight="bold")
            fig.text(.12, .885, "Holdout log loss difference: negative is better. Same-seed paired comparison.", fontsize=10)
            ax.axvline(0, color="#84909A", lw=1, ls="--", zorder=0)
            for index, (arm, effect) in enumerate(zip(arms, effects)):
                y = len(arms) - index - 1
                color = colors[arm]
                interval = effect["date_block_uncertainty"]["ci95"]
                if interval:
                    ax.hlines(y, *interval, colors=color, lw=5, alpha=.35)
                    ax.vlines(interval, y - .075, y + .075, colors=color, lw=1.3)
                seed_effects = effect["values"]
                ax.scatter(seed_effects, y + np.linspace(-.10, .10, len(seed_effects)),
                           s=30, marker="x", color=color, zorder=2)
                ax.scatter([effect["mean"]], [y], s=62, color=color, edgecolors="white", lw=.7, zorder=3)
                ax.text(1.025, y, f'{effect["mean"]:+.5f}', transform=ax.get_yaxis_transform(),
                        va="center", fontsize=10, color=color)
            ax.set(yticks=range(len(arms)), yticklabels=[names[arm] for arm in reversed(arms)],
                   ylim=(-.45, len(arms) - .5), xlim=limits,
                   xlabel="Change in direction LogLoss")
            ax.xaxis.set_major_formatter(FormatStrFormatter("%+.4f"))
            ax.xaxis.grid(True, alpha=.15)
            ax.set_axisbelow(True)
            ax.legend(handles=[Line2D([], [], color="#4D5963", lw=5, alpha=.35, label="95% date-block interval"),
                               Line2D([], [], color="#4D5963", marker="o", ls="", label="Fixed-seed mean"),
                               Line2D([], [], color="#4D5963", marker="x", ls="", label="Individual seed")],
                      loc="upper left", bbox_to_anchor=(-.01, -.25), frameon=False, ncol=3, fontsize=8)
            method = effects[0]["date_block_uncertainty"]
            fig.text(.12, .035, f'Retrospective cohort | {holdout["rows"]} rows / {holdout["origin_dates"]} origin dates | '
                     f'block length {method["block_length_origins"]} origins | intervals conditional on fixed seeds', fontsize=8, color="#53616D")
            for target in targets:
                metadata = ({"Date": None, "Creator": "token_distillation / Matplotlib"} if target.suffix == ".svg"
                            else {"Software": "token_distillation / Matplotlib", "Comparison SHA256": source_sha})
                fig.savefig(target, dpi=180, facecolor="white", metadata=metadata)
                figures.append({"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
            plt.close(fig)
    metadata = {"schema": "ablation_matplotlib_render_v1", "comparison": str(source),
                "comparison_sha256": source_sha, "renderer_python": sys.executable,
                "matplotlib": matplotlib.__version__, "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "statistics_recomputed": False, "figures": figures}
    (output / "matplotlib_render.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
