# Full Kaggle notebook

`run_benchmark.ipynb` is a self-contained, 35-cell notebook. It contains the preprocessing, algorithm implementations, evaluation, figures, and exploratory statistical comparisons. It needs only these two Kaggle input datasets:

- `kmader/skin-cancer-mnist-ham10000`
- `alitaqishah/blood-cell-anomaly-detection-2025`

Attach both datasets to the Kaggle notebook and run all cells. The input locator finds `HAM10000_metadata.csv`, `hmnist_8_8_RGB.csv`, and `blood_cell_anomaly_detection.csv` under `/kaggle/input`. If a name occurs more than once, set its exact path in `INPUT_OVERRIDES`. On a local computer, place the same files under `data/ham/` and `data/blood/` and run the notebook from the repository root.

The default run uses three seeds, 750 training records, and 300 test records for each dataset. It evaluates two settings each for PCA, kernel PCA, SNE, t-SNE, PCA + t-SNE, UMAP, MDS, PCoA, Isomap, LLE, and an exploratory PCA + UMAP hybrid. Set `AML_NOTEBOOK_SEEDS`, `AML_NOTEBOOK_TRAIN_N`, `AML_NOTEBOOK_TEST_N`, or `AML_NOTEBOOK_OUTPUT` in the environment before running to change the experiment. The complete benchmark can take several minutes, depending on the Kaggle runtime.

The notebook saves `runs.csv`, predictions, summary tables, ROC and confusion plots, class-colored two-dimensional embeddings, CPU-time plots, parameter comparisons, and statistical outputs in `/kaggle/working/aml-benchmark/results/` (or `notebook_output/results/` locally). FROC AUC is marked unavailable because the source files lack lesion-location detections. No Fashion-MNIST data are used.

`example_outputs/` contains a completed default-size local run: the 11-method summary, parameter table, protocol, ANOVA output, mean-AUC charts, and class-colored embedding panels for both datasets. The PCA + UMAP hybrid is competitive but does not establish a reliable improvement: mean ROC AUC is 0.648 on HAM10000 versus Isomap's 0.649, and 0.632 on blood cells versus PCA's 0.630, with higher CPU use. These are preliminary three-split comparisons.

`build_full_notebook.py` regenerates the notebook from the tested repository functions and adds LLE, PCA + UMAP, and class visualization. The notebook runs without attaching the repository itself.

Kaggle publication still requires signing in to the account, attaching the two source datasets, and checking their terms and the notebook's visibility. No Kaggle page is claimed to exist yet.
