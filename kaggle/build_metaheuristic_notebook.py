"""Build a self-contained Kaggle notebook from the tested feature-selection code."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code(source):
    return dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
                source=source.splitlines(keepends=True))


def markdown(source):
    return dict(cell_type="markdown", metadata={}, source=source.splitlines(keepends=True))


def function_source(path, names):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    return "\n\n".join(ast.get_source_segment(source, node)
                     for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in names)


benchmark_helpers = function_source(ROOT / "src" / "benchmark.py",
                                    {"load_dataset", "balanced_subsample",
                                     "split_data", "binary_metrics"})
selector = (ROOT / "src" / "feature_selection.py").read_text(encoding="utf-8")
selector = selector.replace("from benchmark import binary_metrics, load_dataset, split_data\n", "")
selector = selector.replace("ROOT = Path(__file__).resolve().parents[1]",
                            "ROOT = Path.cwd()")
selector = selector.split('if __name__ == "__main__":')[0]
plotter = (ROOT / "src" / "plot_feature_selection.py").read_text(encoding="utf-8")
plotter = plotter.split('if __name__ == "__main__":')[0]

cells = [
    markdown("""# GA, PSO, DE, and hybrid feature selection on two biomedical tables

This notebook is self-contained. Attach the Kaggle datasets `kmader/skin-cancer-mnist-ham10000` and `alitaqishah/blood-cell-anomaly-detection-2025`. It evaluates **full features, GA, PSO, DE, GA + PSO, and GA + DE** on HAM10000 and the blood-cell anomaly table. No Fashion-MNIST is used.

The outer test records never guide feature selection. HAM10000 uses lesion-group splits; blood cells use stratified rows because patient IDs are unavailable. The default is three shared 750/300 splits, two feature penalties, eight candidates, and six generations. Search CPU time and selected feature names are saved for every run.
"""),
    code("""from pathlib import Path
import os, shutil
root = Path.cwd()
inputs = Path('/kaggle/input')
required = {
    'HAM10000_metadata.csv': root / 'data' / 'ham',
    'hmnist_8_8_RGB.csv': root / 'data' / 'ham',
    'blood_cell_anomaly_detection.csv': root / 'data' / 'blood',
}
for filename, destination in required.items():
    target = destination / filename
    if not target.exists():
        matches = list(inputs.rglob(filename)) if inputs.exists() else []
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected one {filename}; found {len(matches)}. Attach both datasets or place local files under data/.')
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(matches[0], target)
print('Input files ready:', [str(d / f) for f, d in required.items()])
"""),
    code("""import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score
DATA = Path.cwd() / 'data'
""" + benchmark_helpers),
    markdown("""## Search protocol

Each feature mask is evaluated by two inner folds on the training partition. The fitness is inner ROC AUC minus `penalty × fraction of features selected`. GA uses crossover and mutation; PSO uses personal and global best masks; DE mutates and crosses real-valued vectors before thresholding. Both hybrids run three GA generations followed by three generations of the named second optimizer. Every search makes 56 attempted fitness evaluations, with repeated masks cached.
"""),
    code(selector),
    code(plotter),
    code("""from argparse import Namespace
settings = Namespace(
    seeds=int(os.getenv('AML_FS_SEEDS', '3')),
    train_n=int(os.getenv('AML_FS_TRAIN_N', '750')),
    test_n=int(os.getenv('AML_FS_TEST_N', '300')),
    penalties=os.getenv('AML_FS_PENALTIES', '0.02,0.05'),
    population=int(os.getenv('AML_FS_POPULATION', '8')),
    generations=int(os.getenv('AML_FS_GENERATIONS', '6')),
    output=str(root / 'aml-feature-selection' / 'results'),
)
run(settings)
make(settings.output)
"""),
    code("""from IPython.display import display, Image
result_dir = Path(settings.output)
display(pd.read_csv(result_dir / 'primary_summary.csv').round(3))
display(pd.read_csv(result_dir / 'parameter_summary.csv').round(3))
display(pd.read_csv(result_dir / 'exploratory_tests.csv').round(3))
for name in ['auc_comparison.png', 'accuracy_comparison.png', 'loss_comparison.png',
             'feature_count.png', 'cpu_comparison.png',
             'auc_vs_features.png', 'search_convergence.png',
             'parameter_sensitivity.png', 'HAM10000_roc_confusion.png',
             'Blood_cell_roc_confusion.png', 'HAM10000_feature_frequency.png',
             'Blood_cell_feature_frequency.png']:
    display(Image(filename=str(result_dir / 'figures' / name)))
"""),
    markdown("""## Interpretation

Compare AUC, sensitivity, precision, feature count, and CPU time together. A smaller mask is useful only if the held-out performance and study goal justify the added search cost. Three overlapping splits and small inner folds make the exploratory p-values weak evidence. Do not choose the better penalty using the held-out test results. FROC AUC is unavailable because these classification tables do not have scored candidate lesion locations and false positives per image. The 8 × 8 lesion pixels and missing blood-cell patient IDs prevent clinical generalization claims.
"""),
]

for index, cell in enumerate(cells, start=1):
    cell["id"] = f"aml-meta-{index:02d}"

notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3",
                    language="python", name="python3"), language_info=dict(name="python")),
                nbformat=4, nbformat_minor=5)
output = ROOT / "kaggle" / "metaheuristic_feature_selection.ipynb"
output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(output, len(cells), "cells")
