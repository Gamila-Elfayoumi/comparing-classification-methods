# Model and representation summary

## What is the model?

The predictive model is **class-balanced logistic regression on two coordinates**. PCA, kernel PCA, SNE, t-SNE, PCA + t-SNE, UMAP, MDS, PCoA, Isomap, LLE, and PCA + UMAP produce the coordinates. They are feature-extraction methods, not the final classifier. Keeping the classifier fixed makes this a comparison of complete representation pipelines under one simple decision rule.

For each dataset, seed, and parameter setting, the workflow is:

1. Select 750 training and 300 held-out records. HAM10000 splits by `lesion_id` to keep related images together; the blood-cell table uses stratified rows because it lacks patient identifiers.
2. Fit `StandardScaler` on training features and apply it to both partitions. Fit the unsupervised embedding on training records only, producing two coordinates per record.
3. Map test records into the fitted representation. PCA, kernel PCA, UMAP, Isomap, and LLE use their supported transforms. PCoA uses a Gower extension. SNE, t-SNE, and MDS use a five-neighbor distance-weighted regression approximation because their selected implementations do not provide a native transform.
4. Fit `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=100+seed)` on the two training coordinates and training labels. Obtain held-out positive-class scores with `predict_proba`; predict positive at the fixed 0.5 threshold.
5. Compute confusion counts, accuracy, error, sensitivity/recall, specificity, precision, F1, ROC AUC, Cohen's kappa, informedness, markedness, and probability log loss. Measure process CPU time for embedding fit, test transform, classifier fit, and prediction. UMAP compilation warmup is recorded separately.

The classifier learns a **linear boundary in the two-dimensional map**. If a representation folds positive and negative cases together, this simple classifier cannot recover the lost information. Class weighting compensates for class imbalance in fitting, but it does not guarantee calibrated probabilities or a good 0.5 decision threshold.

## Inputs and endpoints

| Dataset | Input features | Positive class | Split limitation |
|---|---|---|---|
| HAM10000 | 8 × 8 RGB lesion pixels (192 values) | Melanoma | Lesion-separated, but image resolution is too low for a clinical claim |
| Blood Cell Anomaly Detection 2025 | 22 selected morphology, color, and blood-count features | Anomaly | Row split; patient-separated generalization cannot be tested |

Features that directly encode the blood-cell target or predicted anomaly score are excluded. The source data are downloaded separately; raw datasets are not stored in this repository. Fashion-MNIST is not used.

## How to read the results

The original report and core presentation analyses cover nine methods. The full notebook adds **LLE** and **PCA + UMAP**, making eleven methods. Setting A is the prespecified primary comparison; setting B measures sensitivity to parameters. The figures in `kaggle/example_outputs/` show the completed default-size notebook run on three shared splits.

| Dataset | Reference | Highest mean ROC AUC in 11-method run | Interpretation |
|---|---:|---:|---|
| HAM10000 | PCA 0.602 | Isomap 0.649; PCA + UMAP 0.648 | Nonlinear maps improve ranking modestly, but positive precision is low and the leaders differ by only 0.0001 before rounding. |
| Blood cell | PCA 0.630 | PCA + UMAP 0.632 | The hybrid exceeds PCA by about 0.002 AUC while using about 1.72 versus 0.02 CPU seconds per run. This is not reliable evidence of a better method. |

LLE has mean AUC 0.606 on HAM10000 and 0.597 on blood cells. Its blood-cell accuracy of 0.660 is relatively high, but its AUC is below PCA's 0.630; one threshold can look favorable while ranking across thresholds is weaker. The class-colored plots show overlap between labels even where a method creates distinct geometric clusters. A neat embedding is not proof of a useful classifier.

The reported CPU values are for small, fixed-size runs on one machine and are not hardware-independent complexity benchmarks. The three splits overlap, so the exploratory statistical tests do not establish method superiority. A valid FROC AUC cannot be calculated: these files have one label and one score per record, without scored candidate lesion locations or false positives per image.

## Files to inspect

- `kaggle/run_benchmark.ipynb`: complete notebook, all eleven methods, preprocessing, metrics, plots, and exploratory tests.
- `kaggle/example_outputs/primary_summary.csv`: measured three-split summary for setting A.
- `kaggle/example_outputs/*_focus_classes.png`: enlarged class maps for the updated slides.
- `kaggle/example_outputs/eleven_method_*_comparison.png`: AUC and CPU charts for all methods.
- `deliverables/AML_presentation.pptx`: study design, method interpretation, confusion matrices, ROC, loss, timing, class maps, and extension results.
- `deliverables/AML_research_report.docx`: original nine-method pilot report.
