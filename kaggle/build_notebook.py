"""Create a portable Kaggle notebook from the verified local workflow."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent

def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": [s + "\n" for s in source.strip().split("\n")]}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result

cells = [
    cell("markdown", """# Dimensionality reduction on HAM10000 and blood cell anomaly data

Nine methods, two parameter settings each, shared held-out splits, binary confusion metrics, ROC AUC, log loss, and CPU time. The study uses the attached public Kaggle datasets. FROC AUC is not estimable without candidate lesion locations. See the repository README and report for limitations."""),
    cell("code", """from pathlib import Path
import shutil
import subprocess
import sys

# Attach the GitHub repository as a Kaggle Dataset, then set its input path here.
PROJECT = Path('/kaggle/input/aml-benchmark-source')
WORK = Path('/kaggle/working/AML')
if not PROJECT.exists():
    raise FileNotFoundError('Attach the AML repository as a Kaggle Dataset and set PROJECT to its path.')
shutil.copytree(PROJECT, WORK, dirs_exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(WORK/'requirements.txt')], check=True)"""),
    cell("code", """from pathlib import Path
import shutil

inputs = {
    'HAM10000_metadata.csv': WORK/'data/ham/HAM10000_metadata.csv',
    'hmnist_8_8_RGB.csv': WORK/'data/ham/hmnist_8_8_RGB.csv',
    'blood_cell_anomaly_detection.csv': WORK/'data/blood/blood_cell_anomaly_detection.csv',
}
for filename, target in inputs.items():
    matches = list(Path('/kaggle/input').rglob(filename))
    matches = [p for p in matches if not str(p).startswith(str(PROJECT))]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one Kaggle input for {filename}, found {matches}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], target)
    print(filename, target.stat().st_size, 'bytes')"""),
    cell("code", """subprocess.run([sys.executable, str(WORK/'src/benchmark.py'), '--seeds', '3', '--train-n', '750', '--test-n', '300'], check=True)
subprocess.run([sys.executable, str(WORK/'src/analyze.py')], check=True)"""),
    cell("code", """import pandas as pd
from IPython.display import display, Image
display(pd.read_csv(WORK/'results/primary_summary.csv'))
display(Image(filename=str(WORK/'results/figures/HAM10000_roc.png')))
display(Image(filename=str(WORK/'results/figures/Blood_cell_roc.png')))"""),
]
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}
(here / "run_benchmark.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
