"""Create presentation-sized class maps from a completed notebook run.

Example:
    python kaggle/make_focus_figures.py --results-dir notebook_output/results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FOCUS = {
    "HAM10000": ["PCA", "Isomap", "UMAP", "PCA + UMAP"],
    "Blood cell": ["PCA", "LLE", "PCA + t-SNE", "PCA + UMAP"],
}


def make(results: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(results / "runs.csv")
    for dataset, methods in FOCUS.items():
        stem = dataset.replace(" ", "_")
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.1))
        for ax, method in zip(axes.flat, methods):
            raw = results / "raw" / f"{stem}_0_{method.replace(' ', '_')}_0.npz"
            with np.load(raw) as saved:
                z, y = saved["embedding_test"], saved["y"]
            auc = runs[(runs.dataset == dataset) & (runs.seed == 0) &
                       (runs.method == method) & (runs.setting == 0)].auc_roc.iloc[0]
            ax.scatter(z[y == 0, 0], z[y == 0, 1], s=15, alpha=.62,
                       color="#23668A", label="negative")
            ax.scatter(z[y == 1, 0], z[y == 1, 1], s=23, alpha=.8,
                       color="#D17A42", label="positive")
            ax.set_title(f"{method}  |  split AUC {auc:.3f}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        axes.flat[0].legend(loc="best", fontsize=9)
        positive = "melanoma" if dataset == "HAM10000" else "anomaly"
        fig.suptitle(f"{dataset}: held-out class maps  |  orange = {positive}", fontsize=16)
        fig.tight_layout()
        fig.savefig(output / f"{stem}_focus_classes.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    summary = pd.read_csv(results / "primary_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharex=True)
    for ax, dataset in zip(axes, FOCUS):
        part = summary[summary.dataset == dataset].sort_values("auc_roc_mean")
        colors = ["#D17A42" if m == "PCA + UMAP" else
                  "#153B50" if m == part.iloc[-1].method else "#6D98AC"
                  for m in part.method]
        ax.barh(part.method, part.auc_roc_mean, color=colors, height=.72)
        for y, score in enumerate(part.auc_roc_mean):
            ax.text(score + .002, y, f"{score:.3f}", va="center", fontsize=10)
        ax.set_title(dataset, fontsize=16, color="#153B50", pad=12)
        ax.set_xlim(.5, .685)
        ax.set_xlabel("Mean held-out ROC AUC", fontsize=11)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Eleven methods | three shared splits | setting A", fontsize=17)
    fig.tight_layout()
    fig.savefig(output / "eleven_method_auc_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharex=True)
    for ax, dataset in zip(axes, FOCUS):
        part = summary[summary.dataset == dataset].sort_values("cpu_seconds_mean")
        colors = ["#D17A42" if m == "PCA + UMAP" else "#6D98AC"
                  for m in part.method]
        ax.barh(part.method, part.cpu_seconds_mean, color=colors, height=.72)
        for y, seconds in enumerate(part.cpu_seconds_mean):
            ax.text(seconds * 1.08, y, f"{seconds:.2f}s", va="center", fontsize=9)
        ax.set_title(dataset, fontsize=16, color="#153B50", pad=12)
        ax.set_xscale("log")
        ax.set_xlim(.01, 15)
        ax.set_xlabel("Mean process CPU seconds (log scale)", fontsize=11)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Eleven methods | CPU time for 750 train + 300 test", fontsize=17)
    fig.tight_layout()
    fig.savefig(output / "eleven_method_cpu_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "example_outputs")
    args = parser.parse_args()
    make(args.results_dir, args.output_dir)
