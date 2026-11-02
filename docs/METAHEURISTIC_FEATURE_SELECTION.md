# Metaheuristic feature selection on two biomedical tables

## Research question

Can a small wrapper search retain or improve held-out ranking while reducing the number of input features? This extension compares a genetic algorithm (GA), binary particle swarm optimization (PSO), differential evolution (DE), and two staged hybrids (GA + PSO and GA + DE) against logistic regression using every eligible feature. The feature-selection results are **separate from the earlier two-dimensional embedding comparison**: the classifier here receives selected original features, so AUC values across the two studies are not directly attributable to one algorithmic difference.

## Data and leakage controls

- **HAM10000:** 10,015 source records with 192 features from 8 × 8 RGB lesion images; melanoma is positive. The outer train/test split and the inner fitness folds separate `lesion_id` groups. The low-resolution pixels are unsuitable for clinical deployment claims.
- **Blood Cell Anomaly Detection 2025:** 5,880 records with 22 eligible numeric morphology, color, and blood-count features; anomaly is positive. Target proxies are excluded. No patient identifier is available, so the row split cannot test patient-level generalization; independent cohort provenance is not established.
- Three shared outer splits use 750 training and 300 held-out records each. Feature selection sees **only outer training labels**. Each candidate mask is scored in two inner folds; `StandardScaler` and class-balanced logistic regression are refit inside each fold. The held-out test set is evaluated once for each prespecified method and parameter setting.

## Algorithms and objective

A binary vector indicates which features to retain. Every candidate retains at least two. The wrapper objective is

`fitness(mask) = mean inner-fold ROC AUC − λ × (selected features / total features)`.

Setting A uses `λ = 0.02`; setting B uses `λ = 0.05` as a sensitivity check. All searches use 8 candidates initially and 6 generations of 8 candidates: **56 attempted evaluations per run**. Caching reduces repeated model fits; unique masks are logged. The algorithms differ in how they propose the next masks:

| Search | Proposal mechanism |
|---|---|
| GA | Tournament selection, uniform crossover, bit mutation, and best-mask preservation |
| PSO | Binary swarm using personal and global best masks with sigmoid-sampled velocities |
| DE | Real-valued mutation/crossover followed by binary thresholding and one-to-one replacement |
| GA + PSO | Three GA generations followed by three PSO generations, with the current population and best mask handed over |
| GA + DE | Three GA generations followed by three DE generations, with the current population and best mask handed over |

The final classifier is `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=100+seed)` trained on the selected, training-standardized original features. Its positive-class `predict_proba` score yields ROC AUC and log loss; a fixed 0.5 threshold yields confusion metrics. CPU time includes selection and final fit/prediction and excludes data loading and figure generation. The full-feature baseline has no search overhead.

## Measured results (setting A, mean of three shared splits)

| Dataset | Method | ROC AUC | Selected features | Process CPU seconds |
|---|---|---:|---:|---:|
| HAM10000 | Full | 0.713 | 192.0 | 0.08 |
| HAM10000 | GA | 0.712 | 48.7 | 0.85 |
| HAM10000 | PSO | 0.723 | 55.7 | 1.13 |
| HAM10000 | DE | 0.706 | 65.0 | 1.13 |
| HAM10000 | GA + PSO | **0.735** | **40.0** | 0.98 |
| HAM10000 | GA + DE | 0.687 | 49.3 | 0.91 |
| Blood cell | Full | 0.809 | 22.0 | 0.02 |
| Blood cell | GA | 0.793 | 11.0 | 0.34 |
| Blood cell | PSO | 0.808 | 12.3 | 0.42 |
| Blood cell | DE | **0.813** | **10.7** | 0.41 |
| Blood cell | GA + PSO | 0.809 | 12.0 | 0.37 |
| Blood cell | GA + DE | 0.810 | 12.7 | 0.38 |

On HAM10000, GA + PSO retains about 21% of the original features and raises mean AUC by 0.022, but takes roughly 13 times the baseline CPU time. On blood cells, DE retains about 49% of the features and raises AUC by only 0.003 while costing roughly 26 times the baseline CPU time. GA + DE is **not** a general improvement: its mean HAM AUC is 0.687. These observations do not establish that a hybrid is consistently better.

HAM10000 has marked split variability: the full-feature AUC across the three test splits is 0.762, 0.794, and 0.584. Feature masks also vary by seed. The plotted top feature frequencies are descriptive search outputs, not biological feature-importance estimates; 8 × 8 pixels are especially hard to interpret biologically. All paired t and Wilcoxon comparisons against full features have p > 0.15 in this run. With only three overlapping splits, the tests have little power and do not support superiority claims. Parameter setting B is reported in `parameter_summary.csv`; choosing whichever setting scores higher on the test set would bias a final estimate.

## Reproduce

After `python data/download_data.py` and installing `requirements.txt`:

```powershell
python src/feature_selection.py --seeds 3 --train-n 750 --test-n 300 --penalties 0.02,0.05 --population 8 --generations 6
python src/plot_feature_selection.py
python -m pytest tests/test_feature_selection.py -q
```

Results are in `results/feature_selection/`: per-run metrics, selected feature names, test probabilities and predictions, search trajectories, the protocol, exploratory tests, and figures. The figures show AUC, accuracy, log loss, selected feature count, CPU time, AUC versus feature count, convergence, feature frequency, ROC and confusion counts, and parameter sensitivity. `runs.csv` also records error, sensitivity/recall, specificity, precision, F1, Cohen's kappa, informedness, and markedness. A valid FROC requires candidate lesion locations and false positives per image; these record-level classification tables cannot supply it.

For a stronger journal study, use external cohorts, more independent outer groups, training-only tuning of the search budget and sparsity penalty, confidence intervals, calibration analysis, and clinically meaningful input features. A search on tiny inner folds can overfit the fitness score even when the outer test set is untouched.
