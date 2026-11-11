"""Figures and exploratory paired comparisons for the paper adaptation."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_rlwoa_sofl"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
runs = pd.read_csv(OUT / "runs.csv")
pred = pd.read_csv(OUT / "predictions.csv")
trace = pd.read_csv(OUT / "trajectories.csv")
methods = ["WOA + LR", "Adaptive WOA + LR", "Adaptive WOA + fuzzy", "Full + fuzzy"]
colors = ["#23668A", "#D17A42", "#6C4F8C", "#4B8F72"]

fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
for i, dataset in enumerate(["HAM10000", "Blood cell"]):
    sub = runs[runs.dataset == dataset]
    for j, (col, title) in enumerate([("auc_roc", "Held-out ROC AUC"),
                                      ("n_features", "Selected features"),
                                      ("cpu_seconds", "Process CPU seconds")]):
        ax = axs[i, j]
        vals = [sub[sub.method == m][col].mean() for m in methods]
        sd = [sub[sub.method == m][col].std() for m in methods]
        ax.bar(range(4), vals, color=colors, yerr=sd, capsize=3)
        ax.set_xticks(range(4), ["WOA\nLR", "Adaptive\nLR", "Adaptive\nfuzzy", "Full\nfuzzy"])
        ax.set_title(f"{dataset}: {title}")
        ax.grid(axis="y", alpha=.2)
        if col == "auc_roc": ax.set_ylim(0, 1)
fig.savefig(FIG / "model_comparison.png", dpi=180)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for ax, dataset in zip(axs, ["HAM10000", "Blood cell"]):
    for method, color in zip(methods[:3], colors):
        group = trace[(trace.dataset == dataset) & (trace.method == method)]
        mean = group.groupby("iteration").best_fitness.mean()
        ax.plot(mean.index, mean.values, label=method, color=color, linewidth=2)
    ax.set(title=dataset, xlabel="Iteration", ylabel="Best inner fitness")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.savefig(FIG / "search_convergence.png", dpi=180)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for ax, dataset in zip(axs, ["HAM10000", "Blood cell"]):
    for method, color in zip(methods, colors):
        group = pred[(pred.dataset == dataset) & (pred.method == method) & (pred.seed == 0)]
        fpr, tpr, _ = roc_curve(group.y, group.probability)
        auc = runs[(runs.dataset == dataset) & (runs.method == method) & (runs.seed == 0)].auc_roc.iloc[0]
        ax.plot(fpr, tpr, label=f"{method}: {auc:.3f}", color=color)
    ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
    ax.set(title=f"{dataset}: seed 0", xlabel="False positive rate", ylabel="Sensitivity")
    ax.legend(fontsize=8); ax.grid(alpha=.2)
fig.savefig(FIG / "heldout_roc.png", dpi=180)
plt.close(fig)

fig, axs = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
for i, dataset in enumerate(["HAM10000", "Blood cell"]):
    for j, method in enumerate(methods):
        group = pred[(pred.dataset == dataset) & (pred.method == method) & (pred.seed == 0)]
        cm = confusion_matrix(group.y, group.pred, labels=[0, 1])
        ax = axs[i, j]
        ax.imshow(cm, cmap="Blues")
        for (row, col), value in np.ndenumerate(cm):
            ax.text(col, row, str(value), ha="center", va="center", fontsize=14)
        ax.set(title=f"{dataset}: {method}", xlabel="Predicted", ylabel="Actual",
               xticks=[0, 1], yticks=[0, 1])
fig.savefig(FIG / "heldout_confusions.png", dpi=180)
plt.close(fig)

other = ROOT / "results" / "paper_rlwoa_sofl_alpha02" / "summary.csv"
if other.exists():
    left = pd.read_csv(OUT / "summary.csv")
    right = pd.read_csv(other)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, dataset in zip(axs, ["HAM10000", "Blood cell"]):
        vals = []
        for table in [left, right]:
            row = table[(table.dataset == dataset) &
                        (table.method == "Adaptive WOA + fuzzy")].iloc[0]
            vals.append((row.auc_roc_mean, row.n_features_mean))
        ax.scatter([v[1] for v in vals], [v[0] for v in vals],
                   s=140, c=["#D17A42", "#23668A"])
        for label, (auc, count) in zip(["alpha 0.8", "alpha 0.2"], vals):
            ax.annotate(label, (count, auc), xytext=(8, 7), textcoords="offset points")
        ax.set(title=dataset, xlabel="Mean selected features", ylabel="Mean held-out AUC",
               ylim=(0, 1))
        ax.grid(alpha=.2)
    fig.savefig(FIG / "alpha_sensitivity.png", dpi=180)
    plt.close(fig)

tests = []
for dataset in ["HAM10000", "Blood cell"]:
    base = runs[(runs.dataset == dataset) & (runs.method == "WOA + LR")].sort_values("seed")
    for method in methods[1:]:
        other = runs[(runs.dataset == dataset) & (runs.method == method)].sort_values("seed")
        a, b = base.auc_roc.to_numpy(), other.auc_roc.to_numpy()
        tests.append(dict(dataset=dataset, comparator=method, reference="WOA + LR",
                          mean_auc_difference=float(np.mean(b - a)),
                          paired_t_p=float(ttest_rel(b, a).pvalue),
                          wilcoxon_p=float(wilcoxon(b, a).pvalue)))
pd.DataFrame(tests).to_csv(OUT / "exploratory_tests.csv", index=False)
