"""Generate a self-contained notebook from the benchmark's tested functions."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = (ROOT / "src/benchmark.py").read_text(encoding="utf-8")
ANALYZE = (ROOT / "src/analyze.py").read_text(encoding="utf-8")


def source_function(source: str, name: str) -> str:
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def source_between(source: str, first: str, last: str) -> str:
    return source[source.index(first):source.index(last)].strip()


def source_assignment(source: str, name: str) -> str:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.get_source_segment(source, node)
    raise KeyError(name)


cells = []


def add(kind: str, content: str) -> None:
    item = {"cell_type": kind, "id": f"aml-{len(cells)+1:02d}", "metadata": {},
            "source": content.strip().splitlines(True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)


add("markdown", """# Full two-dataset dimensionality reduction benchmark

This notebook runs **PCA, kernel PCA, SNE, t-SNE, PCA + t-SNE, UMAP, MDS,
PCoA, Isomap, LLE, and a new PCA + UMAP hybrid** on HAM10000 and Blood Cell
Anomaly Detection 2025. It contains preprocessing, eleven algorithms, two settings per algorithm,
held-out classification, metrics, plots, timing, and exploratory tests in
the notebook itself. No repository attachment or Fashion-MNIST data is needed.

Attach the public Kaggle datasets `kmader/skin-cancer-mnist-ham10000` and
`alitaqishah/blood-cell-anomaly-detection-2025` before running all cells.
This is a computational pilot, not a clinical diagnostic evaluation.
""")
add("markdown", """## 1. Environment and input files

The default run uses three shared splits, 750 training records, and 300 test
records per split. Set the `AML_NOTEBOOK_*` environment variables for a
smaller local check. Missing dependencies are installed only when needed.
""")
add("code", """import importlib.util, os, subprocess, sys
from pathlib import Path
for module, package in [("umap", "umap-learn>=0.5.9"),
                        ("statsmodels", "statsmodels>=0.14"),
                        ("threadpoolctl", "threadpoolctl>=3.5")]:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(name, "1")
default_root = (Path("/kaggle/working/aml-benchmark") if Path("/kaggle/working").exists()
                else Path.cwd() / "notebook_output")
ROOT = Path(os.getenv("AML_NOTEBOOK_OUTPUT", str(default_root))).resolve()
DATA, RESULTS = ROOT / "data", ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "numba_cache")
SEEDS = int(os.getenv("AML_NOTEBOOK_SEEDS", "3"))
TRAIN_N = int(os.getenv("AML_NOTEBOOK_TRAIN_N", "750"))
TEST_N = int(os.getenv("AML_NOTEBOOK_TEST_N", "300"))
assert SEEDS >= 1 and TRAIN_N >= 100 and TEST_N >= 40
print(f"Output: {ROOT}; seeds={SEEDS}; training={TRAIN_N}; test={TEST_N}")
""")
add("code", """import hashlib, shutil
import pandas as pd
from IPython.display import Image, display

INPUT_OVERRIDES = {"HAM10000_metadata.csv": None,
                   "hmnist_8_8_RGB.csv": None,
                   "blood_cell_anomaly_detection.csv": None}
targets = {"HAM10000_metadata.csv": DATA / "ham/HAM10000_metadata.csv",
           "hmnist_8_8_RGB.csv": DATA / "ham/hmnist_8_8_RGB.csv",
           "blood_cell_anomaly_detection.csv": DATA / "blood/blood_cell_anomaly_detection.csv"}
def find_input(filename):
    if INPUT_OVERRIDES[filename]:
        path = Path(INPUT_OVERRIDES[filename])
        if not path.is_file(): raise FileNotFoundError(path)
        return path
    locations = []
    for root in (Path("/kaggle/input"), Path.cwd() / "data"):
        if root.exists(): locations += list(root.rglob(filename))
    locations = list({str(p.resolve()): p for p in locations}.values())
    if len(locations) != 1:
        raise RuntimeError(f"Expected one {filename}, found {locations}. Set INPUT_OVERRIDES.")
    return locations[0]
manifest = []
for filename, target in targets.items():
    source = find_input(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve(): shutil.copy2(source, target)
    manifest.append({"file": filename, "bytes": target.stat().st_size,
                     "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
display(pd.DataFrame(manifest))
""")
add("markdown", """## 2. Dataset checks and feature selection

HAM10000 uses 192 values from the 8 × 8 RGB pixel CSV. We check exact
pixel/metadata label alignment and split by `lesion_id`. The positive class
is melanoma. Blood cells use 22 measured morphology, color, and routine
blood-count variables. We exclude cell type, disease category, and supplied
anomaly/confidence scores that could reveal the answer. The blood-cell file
has no patient ID; row splits cannot prove patient-level generalization.
""")
add("code", """import json, time
import numpy as np
import matplotlib.pyplot as plt
import umap
from scipy.optimize import minimize
from scipy import stats
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
from sklearn.metrics import pairwise_distances, confusion_matrix, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multitest import multipletests

""" + source_between(BENCH, "METHODS = [", "\ndef load_dataset"))
add("code", source_function(BENCH, "load_dataset"))
add("code", """profiles = []
for dataset in ("HAM10000", "Blood cell"):
    x, y, groups, features = load_dataset(dataset)
    profiles.append({"dataset": dataset, "rows": len(y), "features": len(features),
                     "positive_n": int(y.sum()), "positive_rate": float(y.mean()),
                     "missing_features": int(np.isnan(x).sum()),
                     "group_count": len(np.unique(groups)) if groups is not None else "unavailable"})
profile = pd.DataFrame(profiles)
display(profile)
ax = profile.plot.bar(x="dataset", y="positive_rate", legend=False,
                      color=["#23668A", "#D17A42"], figsize=(7, 3.5))
ax.set(ylabel="Positive-class share", ylim=(0, 1), title="Class prevalence")
plt.tight_layout(); plt.show()
""")
add("markdown", """## 3. Shared preprocessing and algorithm settings

Every method sees the same held-out records within a dataset and seed.
Standardization is fit on training records only. Setting A is the primary
comparison; setting B measures parameter sensitivity. Selecting the better
setting using the test results would bias performance upward.
""")
add("code", source_function(BENCH, "balanced_subsample") + "\n\n" + source_function(BENCH, "split_data"))
add("markdown", """## 4. Embedding and held-out transformation

PCA preserves variance; kernel PCA uses nonlinear RBF similarity. MDS
preserves Euclidean distances, PCoA decomposes a centered distance matrix,
and Isomap uses neighbor-graph geodesics. SNE, t-SNE, and UMAP emphasize
local neighborhoods. PCA + t-SNE filters features before t-SNE.

PCA, kernel PCA, UMAP, and Isomap transform unseen records directly. PCoA
uses a Gower extension. SNE, t-SNE, and MDS use a documented five-neighbor
approximation because these implementations lack a test transform. Their
measured scores include this approximation.
""")
add("code", source_function(BENCH, "sne_embedding"))
add("code", source_function(BENCH, "pcoa_transform") + "\n\n" + source_function(BENCH, "embed"))
add("markdown", """### Notebook extensions: LLE and PCA + UMAP

Locally Linear Embedding (LLE) reconstructs each point from nearby points
and seeks low-dimensional coordinates preserving those reconstruction
weights. The new PCA + UMAP hybrid removes low-variance input directions
before learning UMAP's neighborhood graph. Both are exploratory additions;
neither is guaranteed to improve held-out performance. PCA + t-SNE remains
in the main method list and is evaluated alongside them.
""")
add("code", """METHODS += ["LLE", "PCA + UMAP"]
GRIDS["LLE"] = [{"n_neighbors": 15}, {"n_neighbors": 30}]
GRIDS["PCA + UMAP"] = [{"n_neighbors": 15}, {"n_neighbors": 30}]
base_embed = embed
def embed(method, params, x_train, x_test, seed):
    if method == "LLE":
        model = LocallyLinearEmbedding(n_components=2,
                                       n_neighbors=params["n_neighbors"],
                                       method="standard", eigen_solver="arpack",
                                       random_state=seed, n_jobs=1)
        return model.fit_transform(x_train), model.transform(x_test), None
    if method == "PCA + UMAP":
        pre = PCA(n_components=min(30, x_train.shape[1] - 1), random_state=seed)
        xtr, xte = pre.fit_transform(x_train), pre.transform(x_test)
        model = umap.UMAP(n_components=2, n_neighbors=params["n_neighbors"],
                          min_dist=0.1, random_state=seed, n_jobs=1)
        return model.fit_transform(xtr), model.transform(xte), None
    return base_embed(method, params, x_train, x_test, seed)
print("Methods:", METHODS)
""")
add("code", "display(pd.DataFrame([{'method': m, 'setting_A': GRIDS[m][0], 'setting_B': GRIDS[m][1]} for m in METHODS]))")
add("markdown", """## 5. Classification, metrics, and CPU time

The same balanced logistic regression fits every two-coordinate training
map. A fixed 0.5 threshold gives a confusion matrix, accuracy, error,
sensitivity/recall, specificity, precision, F1, Cohen's kappa,
informedness, and markedness. ROC AUC and log loss use held-out
probabilities. CPU time covers embedding, test transform, classifier fit,
and prediction. UMAP compilation warmup is reported separately.
""")
add("code", source_function(BENCH, "binary_metrics"))
add("code", source_function(BENCH, "run"))
add("code", """from argparse import Namespace
run(Namespace(seeds=SEEDS, train_n=TRAIN_N, test_n=TEST_N,
              methods="all", append=False))
runs = pd.read_csv(RESULTS / "runs.csv")
assert len(runs) == 2 * SEEDS * len(METHODS) * 2
display(runs[["dataset", "seed", "method", "setting", "accuracy",
              "auc_roc", "log_loss", "cpu_seconds"]].head(18))
""")
add("markdown", """## 6. Figures and statistical comparisons

The next cells generate accuracy, loss, ROC, confusion-matrix, CPU-time,
parameter-sensitivity, and SNE-objective plots. The **held-out classifier
log loss** is comparable across all methods; native embedding objectives
have different meanings and scales.

Paired t and Wilcoxon comparisons use shared split AUCs against PCA,
with Holm-adjusted t-test p values. Repeated-measures ANOVA tests the
method term. MANOVA is descriptive only because repeated splits violate
its independence assumption. McNemar z compares discordant first-split
predictions. Three overlapping splits give weak inferential power.
""")
add("code", "OUT = RESULTS\nFIG = RESULTS / 'figures'\nFIG.mkdir(parents=True, exist_ok=True)\n" +
    source_assignment(ANALYZE, "COLORS") + "\nCOLORS += ['#A35F76', '#4B817A']")
add("code", source_function(ANALYZE, "savefig"))
analysis_code = source_function(ANALYZE, "main").replace("def main():", "def analyze_results():", 1)
analysis_code = analysis_code.replace("plt.subplots(3, 3, figsize=(10, 9.4))",
                                      "plt.subplots(3, 4, figsize=(12, 9.4))")
analysis_code = analysis_code.replace("        fig.suptitle(f\"{dataset}: held-out confusion matrices, split 1\")",
    "        for unused in list(axes.flat)[len(METHODS):]: unused.axis('off')\n"
    "        fig.suptitle(f\"{dataset}: held-out confusion matrices, split 1\")")
add("code", analysis_code)
add("code", """analyze_results()
summary = pd.read_csv(RESULTS / "primary_summary.csv")
display(summary[["dataset", "method", "accuracy_mean", "error_mean",
                 "sensitivity_mean", "specificity_mean", "precision_mean",
                 "f_measure_mean", "auc_roc_mean", "kappa_mean",
                 "informedness_mean", "markedness_mean", "log_loss_mean",
                 "cpu_seconds_mean"]].round(3))
""")
add("code", """for dataset in ("HAM10000", "Blood cell"):
    part = summary[summary.dataset == dataset].set_index("method").reindex(METHODS)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    bars = ax.bar(METHODS, part.auc_roc_mean, color=COLORS)
    ax.set(ylabel="Mean held-out ROC AUC", ylim=(0, 1),
           title=f"{dataset}: ranking quality across 11 embeddings")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIG / f"{dataset.replace(' ', '_')}_auc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
""")
add("code", """for dataset in ("HAM10000", "Blood cell"):
    stem = dataset.replace(" ", "_")
    print(f"\\n{dataset}: AUC, accuracy, loss, ROC, and confusion matrices")
    for kind in ("auc", "accuracy", "loss", "roc", "confusion"):
        display(Image(filename=str(FIG / f"{stem}_{kind}.png")))
    print(f"{dataset}: CPU time and parameter sensitivity")
    for kind in ("cpu", "parameters"):
        display(Image(filename=str(FIG / f"{stem}_{kind}.png")))
display(Image(filename=str(FIG / "sne_objective.png")))
""")
add("markdown", """### Where the classes lie in each two-dimensional map

Blue points are negative records and orange points are positive records
on the first held-out split under setting A. Axes differ by method and
must not be compared numerically across panels. Overlap suggests a simple
linear classifier will struggle; apparent separation still needs
held-out AUC, precision, and calibration checks.
""")
add("code", """for dataset in ("HAM10000", "Blood cell"):
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    stem = dataset.replace(" ", "_")
    for ax, method in zip(axes.flat, METHODS):
        path = RESULTS / "raw" / f"{stem}_0_{method.replace(' ', '_')}_0.npz"
        with np.load(path) as result:
            z, y = result["embedding_test"], result["y"]
        ax.scatter(z[y == 0, 0], z[y == 0, 1], s=13, alpha=.6,
                   color="#23668A", label="negative")
        ax.scatter(z[y == 1, 0], z[y == 1, 1], s=19, alpha=.8,
                   color="#D17A42", label="positive")
        auc = runs[(runs.dataset == dataset) & (runs.seed == 0) &
                   (runs.method == method) & (runs.setting == 0)].auc_roc.iloc[0]
        ax.set_title(f"{method} | AUC {auc:.3f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in list(axes.flat)[len(METHODS):]: ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"{dataset}: class-colored held-out embeddings", fontsize=17)
    fig.tight_layout()
    image_path = FIG / f"{stem}_class_embeddings.png"
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(image_path)))
""")
add("code", """hybrids = summary[summary.method.isin(["PCA", "t-SNE", "PCA + t-SNE",
                                             "UMAP", "PCA + UMAP", "LLE"])]
display(hybrids[["dataset", "method", "auc_roc_mean", "accuracy_mean",
                 "cpu_seconds_mean"]].sort_values(["dataset", "auc_roc_mean"],
                                                   ascending=[True, False]).round(3))
""")
add("markdown", """## 7. Interpretation and limitations

The best AUC method need not have the best accuracy at 0.5 or the lowest
loss. With rare melanoma, accuracy can hide low positive precision. AUCs
near 0.6 show only modest discrimination in this pilot. Proposed geometric
reasons for a method's result are hypotheses rather than proven causes.
""")
add("code", """winners = []
for dataset in ("HAM10000", "Blood cell"):
    part = summary[summary.dataset == dataset]
    auc = part.loc[part.auc_roc_mean.idxmax()]
    acc = part.loc[part.accuracy_mean.idxmax()]
    cpu = part.loc[part.cpu_seconds_mean.idxmin()]
    winners.append({"dataset": dataset, "best_AUC": auc.method,
                    "AUC": auc.auc_roc_mean, "best_accuracy": acc.method,
                    "accuracy": acc.accuracy_mean, "fastest": cpu.method,
                    "CPU_seconds": cpu.cpu_seconds_mean})
display(pd.DataFrame(winners).round(3))
for dataset in ("HAM10000", "Blood cell"):
    part = summary[summary.dataset == dataset].set_index("method")
    novel = part.loc["PCA + UMAP"]
    other = part.drop(index="PCA + UMAP").sort_values("auc_roc_mean", ascending=False).iloc[0]
    print(f"{dataset}: PCA + UMAP AUC {novel.auc_roc_mean:.3f}, "
          f"best other method {other.name} {other.auc_roc_mean:.3f}; "
          f"PCA + UMAP CPU {novel.cpu_seconds_mean:.2f}s.")
print("Small AUC gaps over three overlapping splits do not establish an improvement.")
print("LLE's accuracy should also be read with its AUC, sensitivity, and precision.")
display(pd.read_csv(RESULTS / "paired_tests.csv").round(4))
display(pd.read_csv(RESULTS / "mcnemar_z.csv").round(4))
print((RESULTS / "anova.txt").read_text(encoding="utf-8"))
print("MANOVA is descriptive, not a confirmatory significance test.")
print((RESULTS / "manova.txt").read_text(encoding="utf-8"))
""")
add("markdown", """### FROC AUC cannot be estimated from these tables

FROC needs scored candidate locations per image to plot sensitivity
against false positives **per image**. These files contain one binary
label and one score per record. Calling ROC a FROC would invent a
localization result. The figure below explains the missing requirement.
""")
add("code", "display(Image(filename=str(FIG / 'froc_applicability.png')))\nprint(f'All results: {RESULTS}')")
add("markdown", """## 8. Next study and data sources

The `results/` directory contains per-run metrics, raw held-out
predictions, plots, parameter tables, and statistical outputs. A stronger
study needs original lesion images or validated features, patient-level
IDs for the blood-cell cohort, external cohorts, nested tuning, and
uncertainty intervals. Detection annotations are needed if FROC is a
target.

[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
and [Blood Cell Anomaly Detection 2025](https://www.kaggle.com/datasets/alitaqishah/blood-cell-anomaly-detection-2025)
retain their respective source terms. Cite them separately.
""")

notebook = {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}
output = ROOT / "kaggle/run_benchmark.ipynb"
output.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {output} ({len(cells)} cells)")
