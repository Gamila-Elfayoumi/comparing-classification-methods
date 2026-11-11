"""Generate a self-contained notebook for the RLWOA-SOFL adaptation."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_of(path, names):
    source = path.read_text(encoding="utf-8")
    return "\n\n".join(ast.get_source_segment(source, item) for item in ast.parse(source).body
                       if isinstance(item, ast.FunctionDef) and item.name in names)


def cell(kind, source):
    item = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
    if kind == "code": item.update(execution_count=None, outputs=[])
    return item


helpers = source_of(ROOT / "src" / "benchmark.py",
                    {"load_dataset", "balanced_subsample", "split_data", "binary_metrics"})
evaluate = source_of(ROOT / "src" / "feature_selection.py", {"evaluate"})
algorithm = (ROOT / "src" / "paper_rlwoa_sofl.py").read_text(encoding="utf-8")
algorithm = algorithm.replace("from benchmark import binary_metrics, load_dataset, split_data\n", "")
algorithm = algorithm.replace("from feature_selection import evaluate\n", "")
algorithm = algorithm.replace("ROOT = Path(__file__).resolve().parents[1]", "ROOT = Path.cwd()")
algorithm = algorithm.split('if __name__ == "__main__":')[0]

cells = [
    cell("markdown", """# RLWOA-SOFL paper adaptation: HAM10000 and blood cells

The source paper (Anter and Zhang, 2023, DOI 10.1109/TAFFC.2023.3285997) studies fMRI pain decoding. This notebook adapts its binary WOA feature selection, per-agent learned exploration, and self-organizing fuzzy prototypes to **two different binary datasets**. It does not reproduce the paper's clinical experiment or exact AnYa classifier. Attach Kaggle inputs `kmader/skin-cancer-mnist-ham10000` and `alitaqishah/blood-cell-anomaly-detection-2025`. No Fashion-MNIST is used.
"""),
    cell("code", """from pathlib import Path
import os, shutil
root = Path.cwd()
required = {'HAM10000_metadata.csv': root/'data'/'ham',
            'hmnist_8_8_RGB.csv': root/'data'/'ham',
            'blood_cell_anomaly_detection.csv': root/'data'/'blood'}
for name, directory in required.items():
    target = directory/name
    if not target.exists():
        matches = list(Path('/kaggle/input').rglob(name))
        if len(matches) != 1: raise FileNotFoundError(f'Expected one {name}; got {len(matches)}')
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(matches[0], target)
print('Data ready')
"""),
    cell("code", """import argparse, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit, softmax
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split, StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
DATA = Path.cwd()/'data'
""" + helpers + "\n\n" + evaluate),
    cell("markdown", """## Method

Five agents search for 20 iterations (105 fitness attempts). The paper's fitness is inner five-neighbor accuracy plus `alpha × (1 − selected/total)`. Standard WOA uses a linear exploration schedule; the adaptive method learns one three-action neural policy per agent from recent fitness changes. The fuzzy classifier updates class-specific cosine prototypes online. The training-only search and held-out test use lesion-separated splits for HAM10000 and stratified rows for blood cells. The fuzzy prototype implementation is an explicit approximation of the source paper's more elaborate AnYa rules.
"""),
    cell("code", algorithm),
    cell("code", """from argparse import Namespace
for alpha, suffix in [(0.8, 'alpha08'), (0.2, 'alpha02')]:
    settings = Namespace(output=str(root/'aml-paper-adaptation'/suffix), seeds=3,
                         train_n=750, test_n=300, agents=5, iterations=20,
                         alpha=alpha, radius=.25)
    run(settings)
"""),
    cell("code", """from IPython.display import display
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(13, 4))
for ax, (alpha, suffix) in zip(axs, [(0.8,'alpha08'), (0.2,'alpha02')]):
    summary = pd.read_csv(root/'aml-paper-adaptation'/suffix/'summary.csv')
    display(summary.round(3))
    for dataset, marker in [('HAM10000','o'), ('Blood cell','s')]:
        sub = summary[summary.dataset == dataset]
        ax.plot(sub.n_features_mean, sub.auc_roc_mean, marker=marker, label=dataset)
    ax.set(title=f'alpha={alpha}', xlabel='Mean feature count', ylabel='Mean held-out ROC AUC')
    ax.grid(alpha=.2); ax.legend()
plt.show()
"""),
    cell("markdown", """## Interpretation

Read ROC AUC together with log loss, confusion metrics, feature count, and process CPU time. The paper's `alpha=0.8` heavily favors sparse masks; this may remove useful blood features. The `alpha=0.2` run is a sensitivity analysis, not a test-selected winner. Three overlapping splits do not establish significance or patient-level generalization. FROC cannot be computed because neither table includes scored detection candidates and image locations.
"""),
]
for i, item in enumerate(cells, 1): item["id"] = f"paper-{i:02d}"
notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3",
                   language="python", name="python3"), language_info=dict(name="python")),
                nbformat=4, nbformat_minor=5)
target = ROOT / "kaggle" / "paper_rlwoa_sofl.ipynb"
target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(target, len(cells), "cells")
