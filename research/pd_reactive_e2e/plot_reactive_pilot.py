"""Absolute metrics for the bounded reactive pilot; never infer qualification."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.comparison.read_text())
    by = {row["mechanism"]: row for row in result["runs"]}
    rows = [by[name] for name in ("inplace", "replica")]
    if not all(row.get("complete_workload") for row in rows):
        raise RuntimeError("Both finite workloads must be complete before plotting")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.labelcolor": "#243746",
        "xtick.color": "#243746", "ytick.color": "#52616b",
        "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    panels = [
        ("Mean request E2E", "seconds", [r["request_e2e_ms"]["mean"] / 1000 for r in rows]),
        ("Mean task E2E", "seconds", [r["task_e2e_s"]["mean"] for r in rows]),
        ("Worker GPU time", "GPU-minutes", [r["gpu_seconds"]["worker_service_through_scale_down"] / 60 for r in rows]),
    ]
    for ax, (title, unit, values) in zip(axes, panels):
        bars = ax.bar([0, 1], values, width=.52, color=["#386f9b", "#d49b4e"], zorder=3)
        ax.set_title(title, loc="left", fontweight="bold", pad=14)
        ax.set_ylabel(unit)
        ax.set_xticks([0, 1], ["In-place\nPP2 → PP3", "Replica\nPP2 → 2 × PP2"])
        ax.set_ylim(0, max(values) * 1.24)
        ax.yaxis.grid(True, color="#e3e8ec", linewidth=.7, zorder=0)
        ax.tick_params(axis="both", length=0)
        ax.spines["bottom"].set_color("#c7d0d8")
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold")
    fig.suptitle("122B on A100 · P/D disaggregation · short reactive pilot", x=.07,
        ha="left", fontsize=12, fontweight="bold", y=.98)
    fig.text(.07, .055, "One run per arm; one active request per D instance; fixed KV capacity.", fontsize=8.5, color="#52616b")
    fig.text(.07, .012, "In-place measurements completed; its supervisor failed during later cleanup. Descriptive observations only.", fontsize=8, color="#52616b")
    fig.subplots_adjust(left=.07, right=.99, top=.78, bottom=.29, wspace=.46)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix), dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
