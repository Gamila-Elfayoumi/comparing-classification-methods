"""Figures for the measured feature-selection extension."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

METHODS = ["Full", "GA", "PSO", "DE", "GA + PSO", "GA + DE"]
COLORS = dict(zip(METHODS, ["#153B50", "#4E8A9E", "#D17A42", "#72935B",
                            "#8B6AA8", "#BA5C5C"]))
DATASETS = ["HAM10000", "Blood cell"]


def save(fig, path):
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def paired_bars(summary, column, ylabel, path, log=False):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        part = summary[summary.dataset == dataset].set_index("method").reindex(METHODS)
        vals = part[column].to_numpy()
        ax.bar(np.arange(len(METHODS)), vals, color=[COLORS[m] for m in METHODS])
        if log:
            ax.set_yscale("log")
            ax.set_ylim(max(min(vals) * .75, .005), max(vals) * 1.55)
        ax.set_xticks(range(len(METHODS)), METHODS, rotation=35, ha="right")
        ax.set_title(dataset, fontsize=15)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        for i, v in enumerate(vals):
            label = (f"{v:.3f}" if column in ("auc_roc_mean", "accuracy_mean", "log_loss_mean") else
                     f"{v:.2f}" if log else f"{v:.1f}")
            ax.text(i, v * (1.08 if log else 1.01), label, ha="center", fontsize=8)
        if column == "auc_roc_mean":
            ax.set_ylim(0, max(vals) + .1)
        elif column == "selected_features_mean":
            ax.set_ylim(0, max(vals) * 1.17)
    fig.tight_layout()
    save(fig, path)


def make(out):
    out = Path(out)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    runs = pd.read_csv(out / "runs.csv")
    summary = pd.read_csv(out / "primary_summary.csv")
    traces = pd.read_csv(out / "trajectories.csv")
    selected = pd.read_csv(out / "selected_features.csv")
    predictions = pd.read_csv(out / "predictions.csv")
    paired_bars(summary, "auc_roc_mean", "Mean held-out ROC AUC", figures / "auc_comparison.png")
    paired_bars(summary, "accuracy_mean", "Mean held-out accuracy", figures / "accuracy_comparison.png")
    paired_bars(summary, "log_loss_mean", "Mean held-out log loss (lower is better)",
                figures / "loss_comparison.png")
    paired_bars(summary, "selected_features_mean", "Selected features (mean)",
                figures / "feature_count.png")
    paired_bars(summary, "cpu_seconds_mean", "Process CPU seconds (log scale)",
                figures / "cpu_comparison.png", log=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, dataset in zip(axes, DATASETS):
        part = summary[summary.dataset == dataset].set_index("method")
        for m in METHODS:
            row = part.loc[m]
            ax.scatter(row.selected_features_mean, row.auc_roc_mean,
                       color=COLORS[m], s=105, label=m)
            ax.annotate(m, (row.selected_features_mean, row.auc_roc_mean),
                        xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_title(dataset)
        ax.set_xlabel("Mean feature count")
        ax.set_ylabel("Mean held-out ROC AUC")
        ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, figures / "auc_vs_features.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, dataset in zip(axes, DATASETS):
        part = traces[(traces.dataset == dataset) & (traces.setting == 0)]
        for m in METHODS[1:]:
            mean = part[part.method == m].groupby("generation").best_fitness.mean()
            ax.plot(mean.index, mean.values, marker="o", markersize=3,
                    color=COLORS[m], label=m)
        ax.set_title(dataset)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best inner fitness (mean of 3 searches)")
        ax.grid(alpha=.2)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, figures / "search_convergence.png")

    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
        part = predictions[(predictions.dataset == dataset) &
                           (predictions.seed == 0) & (predictions.setting == 0)]
        for m in METHODS:
            d = part[part.method == m]
            fpr, tpr, _ = roc_curve(d.y, d.probability)
            mean_auc = summary[(summary.dataset == dataset) &
                               (summary.method == m)].auc_roc_mean.iloc[0]
            axes[0].plot(fpr, tpr, color=COLORS[m],
                         label=f"{m} (mean {mean_auc:.3f})")
        axes[0].plot([0, 1], [0, 1], ls="--", color="gray", alpha=.5)
        axes[0].set(xlabel="False positive rate", ylabel="Sensitivity",
                    title="ROC curves: first held-out split")
        axes[0].legend(fontsize=8)
        best = summary[summary.dataset == dataset].sort_values("auc_roc_mean").iloc[-1].method
        for j, m in enumerate(("Full", best)):
            d = part[part.method == m]
            tn = int(((d.y == 0) & (d.pred == 0)).sum())
            fp = int(((d.y == 0) & (d.pred == 1)).sum())
            fn = int(((d.y == 1) & (d.pred == 0)).sum())
            tp = int(((d.y == 1) & (d.pred == 1)).sum())
            axes[1].text(.08, .77 - j * .43,
                         f"{m}\nTN {tn}   FP {fp}\nFN {fn}   TP {tp}",
                         transform=axes[1].transAxes, fontsize=15,
                         va="top", color=COLORS[m])
        axes[1].set(title="Confusion counts at 0.5 threshold")
        axes[1].axis("off")
        fig.tight_layout()
        save(fig, figures / f"{dataset.replace(' ', '_')}_roc_confusion.png")

    for dataset in DATASETS:
        part = selected[(selected.dataset == dataset) &
                        (selected.setting == 0) & (selected.method != "Full")]
        counts = {}
        for encoded in part.features:
            for feature in json.loads(encoded):
                counts[feature] = counts.get(feature, 0) + 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:18]
        fig, ax = plt.subplots(figsize=(9.4, 5.2))
        ax.barh([p[0] for p in top][::-1], [p[1] for p in top][::-1],
                color="#4E8A9E")
        ax.set_xlabel("Selections among 15 primary searches (5 methods × 3 seeds)")
        ax.set_title(f"{dataset}: frequently selected features")
        ax.grid(axis="x", alpha=.2)
        fig.tight_layout()
        save(fig, figures / f"{dataset.replace(' ', '_')}_feature_frequency.png")

    sensitivity = runs[runs.method != "Full"].groupby(
        ["dataset", "method", "sparsity_penalty"], as_index=False).agg(
            auc=("auc_roc", "mean"), features=("n_features", "mean"))
    sensitivity.to_csv(out / "parameter_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, dataset in zip(axes, DATASETS):
        part = sensitivity[sensitivity.dataset == dataset]
        for method in METHODS[1:]:
            p = part[part.method == method].sort_values("sparsity_penalty")
            ax.plot(p.sparsity_penalty, p.auc, marker="o", color=COLORS[method],
                    label=method)
        ax.set(title=dataset, xlabel="Feature-count penalty",
               ylabel="Mean held-out ROC AUC")
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, figures / "parameter_sensitivity.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(Path(__file__).resolve().parents[1] /
                                                  "results" / "feature_selection"))
    make(parser.parse_args().results)
