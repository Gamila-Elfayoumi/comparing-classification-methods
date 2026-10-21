"""Summarize measured runs and make figures and exploratory statistical tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COLORS = ["#153B50", "#23668A", "#4D9CB9", "#D17A42", "#AA4A3C", "#77619B", "#72944B", "#B28A32", "#576D77"]
METHODS = ["PCA", "Kernel PCA", "SNE", "t-SNE", "PCA + t-SNE", "UMAP", "MDS", "PCoA", "Isomap"]

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=220, bbox_inches="tight")
    plt.close()

def main():
    df = pd.read_csv(OUT / "runs.csv")
    primary = df[df.setting == 0].copy()
    cols = ["accuracy", "error", "sensitivity", "specificity", "precision",
            "recall", "f_measure", "auc_roc", "kappa", "informedness",
            "markedness", "log_loss", "cpu_seconds", "wall_seconds"]
    summary = primary.groupby(["dataset", "method"])[cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a,b in summary.columns]
    summary.reset_index().to_csv(OUT / "primary_summary.csv", index=False)
    df.groupby(["dataset", "method", "setting", "params"])[cols].mean().reset_index().to_csv(
        OUT / "parameter_comparison.csv", index=False)

    for dataset in primary.dataset.unique():
        part = primary[primary.dataset == dataset]
        ordered = part.groupby("method").accuracy.mean().reindex(METHODS)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        bars = ax.bar(METHODS, ordered, color=COLORS)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Held-out accuracy")
        ax.set_title(f"{dataset}: accuracy by embedding")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.tick_params(axis="x", rotation=35)
        savefig(f"{dataset.replace(' ','_')}_accuracy.png")

        means = part.groupby("method").cpu_seconds.mean().reindex(METHODS)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        bars = ax.bar(METHODS, means, color=COLORS)
        ax.set_ylabel("CPU seconds per fit and test")
        ax.set_title(f"{dataset}: process CPU time")
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.tick_params(axis="x", rotation=35)
        savefig(f"{dataset.replace(' ','_')}_cpu.png")

        curves = json.loads((OUT / "roc_curves.json").read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        for method, color in zip(METHODS, COLORS):
            items = [r for r in curves if r["dataset"] == dataset and r["seed"] == 0
                     and r["method"] == method and r["setting"] == 0]
            r = items[0]
            auc = float(part[(part.method == method) & (part.seed == 0)].auc_roc.iloc[0])
            ax.plot(r["fpr"], r["tpr"], label=f"{method} ({auc:.3f})", color=color)
        ax.plot([0, 1], [0, 1], ls="--", color="#999999")
        ax.set(xlabel="False positive rate", ylabel="Sensitivity", title=f"{dataset}: ROC on split 1")
        ax.legend(fontsize=8, loc="lower right")
        savefig(f"{dataset.replace(' ','_')}_roc.png")

        fig, axes = plt.subplots(3, 3, figsize=(10, 9.4))
        for ax, method in zip(axes.flat, METHODS):
            row = part[(part.method == method) & (part.seed == 0)].iloc[0]
            cm = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=13)
            ax.set(title=method, xticks=[0,1], yticks=[0,1], xlabel="Predicted", ylabel="True")
        fig.suptitle(f"{dataset}: held-out confusion matrices, split 1")
        savefig(f"{dataset.replace(' ','_')}_confusion.png")

        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        for method, color in zip(METHODS, COLORS):
            sub = df[(df.dataset == dataset) & (df.method == method)]
            means = sub.groupby("setting").auc_roc.mean()
            ax.plot([0,1], means.values, marker="o", label=method, color=color)
        ax.set(xticks=[0,1], xticklabels=["Setting A", "Setting B"],
               ylabel="Mean ROC AUC", title=f"{dataset}: parameter sensitivity")
        ax.legend(ncol=3, fontsize=8)
        savefig(f"{dataset.replace(' ','_')}_parameters.png")

        loss = part.groupby("method").log_loss.mean().reindex(METHODS)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        bars = ax.bar(METHODS, loss, color=COLORS)
        ax.set(ylabel="Held-out logistic log loss", title=f"{dataset}: common classifier loss")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.tick_params(axis="x", rotation=35)
        savefig(f"{dataset.replace(' ','_')}_loss.png")

    traces = json.loads((OUT / "optimization_traces.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, dataset in zip(axes, ["HAM10000", "Blood cell"]):
        trace = next(x["trace"] for x in traces if x["dataset"] == dataset and
                     x["seed"] == 0 and x["method"] == "SNE" and x["setting"] == 0)
        ax.plot([t[0] for t in trace], [t[1] for t in trace], color=COLORS[2])
        ax.set(xlabel="SNE optimization step", ylabel="Gaussian KL divergence", title=dataset)
    savefig("sne_objective.png")

    # This is intentionally an applicability figure: no lesion localization or
    # multiple candidate detections exist in the available tables.
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.5, .7, "FROC AUC cannot be estimated from these labels", ha="center",
            fontsize=17, weight="bold", color="#153B50")
    ax.text(0.5, .35, "Required: lesion locations and scored candidate detections per image.\n"
            "Available: one binary label and one score per record.", ha="center", fontsize=12)
    savefig("froc_applicability.png")

    stat_rows = []
    for dataset in primary.dataset.unique():
        part = primary[primary.dataset == dataset]
        pivot = part.pivot(index="seed", columns="method", values="auc_roc")[METHODS]
        base = pivot["PCA"].to_numpy()
        pvals = []
        pending = []
        for method in METHODS[1:]:
            other = pivot[method].to_numpy()
            t = stats.ttest_rel(other, base)
            try:
                w = stats.wilcoxon(other - base)
                wp = float(w.pvalue)
            except ValueError:
                wp = np.nan
            pending.append((method, float(np.mean(other-base)), float(t.statistic), float(t.pvalue), wp))
            pvals.append(float(t.pvalue))
        adjusted = multipletests(pvals, method="holm")[1]
        for (method, difference, tstat, tp, wp), adj in zip(pending, adjusted):
            stat_rows.append(dict(dataset=dataset, comparison=f"{method} vs PCA",
                                  metric="AUC ROC", mean_difference=difference,
                                  paired_t=tstat, paired_t_p=tp, paired_t_holm_p=float(adj),
                                  wilcoxon_p=wp))
    pd.DataFrame(stat_rows).to_csv(OUT / "paired_tests.csv", index=False)

    # Repeated-measures ANOVA respects the shared splits. The MANOVA below is
    # requested as a descriptive sensitivity analysis, not confirmatory evidence.
    a = primary.copy()
    a["subject"] = a.dataset + "_" + a.seed.astype(str)
    try:
        anova = AnovaRM(a, depvar="auc_roc", subject="subject", within=["method"]).fit()
        (OUT / "anova.txt").write_text(str(anova), encoding="utf-8")
    except Exception as exc:
        (OUT / "anova.txt").write_text(f"Not estimable: {exc}", encoding="utf-8")
    try:
        manova = MANOVA.from_formula("auc_roc + log_loss ~ method + dataset", data=a).mv_test()
        (OUT / "manova.txt").write_text(str(manova), encoding="utf-8")
    except Exception as exc:
        (OUT / "manova.txt").write_text(f"Not estimable: {exc}", encoding="utf-8")

    # First split, discordant paired predictions: McNemar's large-sample z.
    zrows = []
    for dataset in primary.dataset.unique():
        stem = dataset.replace(" ", "_")
        base = np.load(OUT / "raw" / f"{stem}_0_PCA_0.npz")
        for method in METHODS[1:]:
            other = np.load(OUT / "raw" / f"{stem}_0_{method.replace(' ','_')}_0.npz")
            y = base["y"]
            b = np.sum((base["pred"] == y) & (other["pred"] != y))
            c = np.sum((base["pred"] != y) & (other["pred"] == y))
            z = (c-b) / np.sqrt(b+c) if b+c else np.nan
            p = 2*stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
            zrows.append(dict(dataset=dataset, comparison=f"{method} vs PCA",
                              discordant_pca_only=int(b), discordant_other_only=int(c),
                              mcnemar_z=z, two_sided_p=p))
    pd.DataFrame(zrows).to_csv(OUT / "mcnemar_z.csv", index=False)
    print(summary.round(3).to_string())

if __name__ == "__main__":
    main()
