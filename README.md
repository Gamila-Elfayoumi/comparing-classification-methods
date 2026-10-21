# Dimensionality reduction for two biomedical classification tables

This repository compares PCA, RBF kernel PCA, Gaussian SNE, t-SNE, PCA followed by t-SNE, UMAP, metric MDS, PCoA, and Isomap. A fixed logistic regression classifier tests whether each two-dimensional embedding retains information useful for a binary decision. It reports predictive performance, parameter sensitivity, and process CPU time. The report and slides are in `deliverables/`.

## Study data

1. **HAM10000**: 10,015 lesion images represented by the Kaggle 8 × 8 RGB pixel CSV (192 pixel features). The binary endpoint is melanoma (`mel`) versus every other HAM10000 diagnosis. `HAM10000_metadata.csv` supplies `lesion_id` for group-separated evaluation. This compressed representation is useful for a tractable algorithm comparison but loses clinical image detail. The source is [Skin Cancer MNIST HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), based on the [HAM10000 paper](https://arxiv.org/abs/1803.10417). Kaggle labels its mirror CC BY-NC-SA 4.0. Check the original data terms before reuse.
2. **Blood Cell Anomaly Detection 2025**: 5,880 records, 22 selected measured morphology, color, and routine blood-count variables. The endpoint is `anomaly_label`. Direct target proxies (`cell_type`, `disease_category`, and anomaly/confidence scores) are excluded. The [Kaggle source](https://www.kaggle.com/datasets/alitaqishah/blood-cell-anomaly-detection-2025) does not provide a patient identifier or independently verified cohort provenance in the downloaded table. Therefore, its row-level split cannot establish patient-level generalization. Results are methodological demonstrations, not diagnostic claims.

No raw dataset is committed. `data/download_data.py` fetches the three required public source files, validates archive extraction, and prints SHA-256 hashes. It intentionally has no Fashion-MNIST dependency.

## Reproduce

Tested on Windows with Python 3.13. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python data/download_data.py
python src/benchmark.py --seeds 3 --train-n 750 --test-n 300
python src/analyze.py
```

The parameters above reproduce the accompanying pilot. Increase `--seeds`, `--train-n`, and `--test-n` for a larger study; runtime grows quickly for SNE and t-SNE. `results/protocol.json` records the run sizes and UMAP compilation warmup. `results/runs.csv` has every setting and split. `results/raw/` contains test labels, scores, predictions, and test embeddings; it is gitignored to keep the public repository small. The source data and outputs contain no patient names, but verify terms before publishing data-derived artifacts.

## Evaluation protocol

- The same split and test records are used for every method and setting within a dataset and seed. HAM10000 uses `GroupShuffleSplit` on `lesion_id`; blood-cell records use stratified row splits because no patient ID is available.
- Training and test sets are subsampled with class stratification. Standardization fits only on the training subset. The 2D embedding fits only on training observations.
- PCA, kernel PCA, UMAP, and Isomap use their native test transforms. Gaussian SNE, t-SNE, and MDS lack native transforms in the chosen implementations, so a five-neighbor distance-weighted regressor maps unseen samples into the training embedding. This approximation is part of the measured pipeline and should be interpreted as such. PCoA uses a Gower out-of-sample extension.
- A class-balanced logistic regression is fitted to each training embedding. The 0.5 threshold is fixed in advance. ROC AUC and log loss use held-out probabilities. CPU seconds use `time.process_time()` for embedding, test transform, classifier fit, and prediction. UMAP's first Numba compilation is warmed separately and reported in the protocol.
- The primary comparison uses setting A for each method. Setting B is a sensitivity check; selecting the best setting on the test set would bias performance upward.

| Method | Setting A | Setting B |
|---|---|---|
| PCA | whiten false | whiten true |
| Kernel PCA | RBF gamma = 0.5 / features | RBF gamma = 2 / features |
| Gaussian SNE | perplexity 15 | perplexity 30 |
| t-SNE | perplexity 15 | perplexity 30 |
| PCA + t-SNE | PCA to up to 30 features, perplexity 15 | same, perplexity 30 |
| UMAP | 15 neighbors | 30 neighbors |
| Metric MDS | 80 iterations | 150 iterations |
| PCoA | Cosine distance | Manhattan distance |
| Isomap | 10 neighbors | 25 neighbors |

PCoA is classical multidimensional scaling of a distance matrix. Its result depends on that distance; Euclidean PCoA would largely duplicate PCA, so the primary setting uses cosine distance.

## Metrics and statistics

Positive means melanoma for HAM10000 and anomaly for the blood-cell table. Sensitivity equals recall. Precision is positive predictive value. Error is `1 - accuracy`. F-measure is F1. Cohen's kappa corrects agreement for chance. Informedness is `sensitivity + specificity - 1`. Markedness is `precision + negative predictive value - 1`. Each run records the 2 × 2 confusion matrix, ROC AUC, and held-out classifier log loss.

**FROC AUC is not estimable from these files.** FROC requires lesion locations and scored candidate detections per image so sensitivity can be plotted against false positives per image. One class label and one score per record produce an ROC curve, not a valid free-response ROC. The report includes an applicability figure rather than a fabricated FROC value.

`src/analyze.py` calculates exploratory paired t tests and Wilcoxon signed-rank tests for AUC against PCA, Holm-adjusted paired t p values, repeated-measures ANOVA, a descriptive MANOVA of AUC and log loss, and a McNemar large-sample z statistic on the first shared test split. Three repeated splits have low inferential power and share training observations. The MANOVA treats repeated observations as independent and must **not** be used for confirmatory inference. The methods are included for transparent comparison, not significance claims.


## Files

- `src/benchmark.py`: data checks, split, embeddings, metrics, timing.
- `src/analyze.py`: summaries, figures, statistical comparisons.
- `data/download_data.py`: source acquisition and hashes.
- `results/`: measured CSV tables, protocol, figures, and statistical outputs.
- `deliverables/`: research report and presentation.
- `kaggle/`: Kaggle notebook handoff instructions.

Code is released under the MIT license. Source datasets retain their own licenses and must be cited separately.
