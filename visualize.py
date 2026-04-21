import math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from .sim import SimulationRun, WorkloadResult
from .analyze import suite_summary

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})

PALETTE = [
    "#2D6BE4", "#E84545", "#27AE60", "#F39C12",
    "#8E44AD", "#16A085", "#C0392B", "#2980B9",
]


# ---------------------------------------------------------------------------
# Roofline plot
# ---------------------------------------------------------------------------

def plot_roofline(
    runs: list[SimulationRun],
    title: str = "Roofline model",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    ai_range = np.logspace(-2, 4, 400)

    for idx, run in enumerate(runs):
        color    = PALETTE[idx % len(PALETTE)]
        rf_model = run.hardware.roofline_model()

        peak_perf_gflops = rf_model.peak_flops / 1e9
        peak_bw_gb_s     = rf_model.peak_bw    / 1e9
        ridge            = rf_model.ridge_point

        roof_perf = np.minimum(
            peak_perf_gflops * np.ones_like(ai_range),
            ai_range * peak_bw_gb_s,
        )

        ax.plot(
            ai_range, roof_perf,
            color=color, linewidth=2.0, linestyle="--", alpha=0.7,
            label=f"{run.hardware.name} (roof)",
        )

        # Annotate ridge point.
        ax.axvline(ridge, color=color, linewidth=0.6, alpha=0.4, linestyle=":")
        ax.text(
            ridge * 1.05, peak_perf_gflops * 0.5,
            f"ridge={ridge:.1f}",
            color=color, fontsize=8, va="center", alpha=0.8,
        )

        # Scatter the workload points.
        for r in run.results:
            ai       = r.arithmetic_intensity
            achieved = r.effective_tflops * 1000   # convert to GFLOPs/s
            marker   = "o" if r.bottleneck == "compute" else "s"
            ax.scatter(
                ai, achieved,
                color=color, marker=marker, s=60, zorder=5, alpha=0.85,
            )
            ax.annotate(
                r.workload_label,
                xy=(ai, achieved),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color=color, alpha=0.9,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity  (FLOPs / byte)", fontsize=11)
    ax.set_ylabel("Performance  (GFLOPs / s)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], marker="o", color="gray", label="compute-bound", linestyle="none", markersize=7),
        Line2D([0], [0], marker="s", color="gray", label="memory-bound",  linestyle="none", markersize=7),
    ]
    hw_handles = [
        mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=run.hardware.name)
        for i, run in enumerate(runs)
    ]
    ax.legend(handles=hw_handles + legend_handles, fontsize=8, loc="upper left")

    ax.set_xlim(0.05, 3000)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Runtime comparison bar chart
# ---------------------------------------------------------------------------

def plot_runtime_comparison(
    runs: list[SimulationRun],
    title: str = "Runtime per workload",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    all_labels = []
    for run in runs:
        for r in run.results:
            if r.workload_label not in all_labels:
                all_labels.append(r.workload_label)

    n_labels = len(all_labels)
    n_hw     = len(runs)
    x        = np.arange(n_labels)
    width    = 0.8 / n_hw

    fig, ax = plt.subplots(figsize=(max(10, n_labels * 1.4), 6))

    for idx, run in enumerate(runs):
        runtimes = []
        for label in all_labels:
            match = next((r.runtime_ms for r in run.results if r.workload_label == label), 0.0)
            runtimes.append(match)

        bars = ax.bar(
            x + idx * width - 0.4 + width / 2,
            runtimes,
            width=width * 0.9,
            color=PALETTE[idx % len(PALETTE)],
            label=run.hardware.name,
            alpha=0.85,
        )

        for bar, val in zip(bars, runtimes):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, alpha=0.8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Runtime  (ms)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Cache hit-rate waterfall
# ---------------------------------------------------------------------------

def plot_cache_hit_rates(
    run: SimulationRun,
    title: str = "",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    labels = [r.workload_label for r in run.results]
    hits   = [r.cache_hit_rate  for r in run.results]
    stalls = [r.memory_stall_cycles / max(1, r.compute_cycles) for r in run.results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(labels))

    colors = [
        ("#27AE60" if h >= 0.90 else ("#F39C12" if h >= 0.70 else "#E84545"))
        for h in hits
    ]

    ax1.barh(x, hits, color=colors, alpha=0.85)
    ax1.set_yticks(x)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Cache hit rate", fontsize=10)
    ax1.set_xlim(0, 1.05)
    ax1.axvline(0.90, color="#27AE60", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.axvline(0.70, color="#F39C12", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_title(f"Cache hit rate — {run.hardware.name}", fontsize=11, fontweight="bold")

    ax2.barh(x, stalls, color=PALETTE[0], alpha=0.75)
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Memory stall / compute cycles  (ratio)", fontsize=10)
    ax2.set_title("Memory stall ratio", fontsize=11, fontweight="bold")

    fig.suptitle(title or run.hardware.name, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Arithmetic intensity scatter (all hardware overlaid)
# ---------------------------------------------------------------------------

def plot_ai_scatter(
    runs: list[SimulationRun],
    title: str = "Arithmetic intensity by workload",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))

    seen_labels: set[str] = set()

    for idx, run in enumerate(runs):
        color = PALETTE[idx % len(PALETTE)]
        for r in run.results:
            ax.scatter(
                r.arithmetic_intensity,
                idx,
                s=max(20, min(500, r.flops / 1e8)),
                color=color, alpha=0.75, zorder=4,
            )
            if r.workload_label not in seen_labels:
                ax.annotate(
                    r.workload_label,
                    xy=(r.arithmetic_intensity, idx),
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=7, ha="center", rotation=25, alpha=0.85,
                )
                seen_labels.add(r.workload_label)

    ax.set_xscale("log")
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([r.hardware.name for r in runs], fontsize=9)
    ax.set_xlabel("Arithmetic intensity  (FLOPs / byte)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.axvline(1,  color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.axvline(10, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.text(1.05,  -0.5, "AI=1",  fontsize=7, color="gray")
    ax.text(10.5, -0.5, "AI=10", fontsize=7, color="gray")

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Bottleneck heatmap
# ---------------------------------------------------------------------------

def plot_bottleneck_heatmap(
    runs: list[SimulationRun],
    out_path: Optional[Path] = None,
) -> plt.Figure:
    hw_names = [r.hardware.name for r in runs]
    wl_names = [r.workload_label for r in runs[0].results]

    data = np.zeros((len(hw_names), len(wl_names)))
    for i, run in enumerate(runs):
        lut = {r.workload_label: r for r in run.results}
        for j, label in enumerate(wl_names):
            r = lut.get(label)
            if r:
                data[i, j] = r.compute_utilization

    fig, ax = plt.subplots(figsize=(max(8, len(wl_names) * 1.2), max(4, len(hw_names) * 1.2)))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(wl_names)))
    ax.set_xticklabels(wl_names, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(hw_names)))
    ax.set_yticklabels(hw_names, fontsize=9)

    for i in range(len(hw_names)):
        for j in range(len(wl_names)):
            val = data[i, j]
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=8, color="black" if 0.3 < val < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="Compute utilization")
    ax.set_title("Compute utilization heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def save_all(
    runs: list[SimulationRun],
    out_dir: Path,
    prefix: str = "",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = lambda name: out_dir / f"{prefix}{name}.png"

    plot_roofline(runs,             out_path=p("roofline"))
    plot_runtime_comparison(runs,   out_path=p("runtime_comparison"))
    plot_ai_scatter(runs,           out_path=p("ai_scatter"))
    plot_bottleneck_heatmap(runs,   out_path=p("bottleneck_heatmap"))

    for run in runs:
        slug = run.hardware.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        plot_cache_hit_rates(run,   out_path=p(f"cache_{slug}"))

    plt.close("all")
