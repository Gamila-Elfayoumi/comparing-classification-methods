# Methodology diagrams

The presentation contains editable versions of these diagrams. The first two workflows use the same experiment sizes but different split rules. In both, feature selection and standardization use training records only.

## HAM10000

```mermaid
flowchart LR
    A[10,015 lesion records<br/>192 values from 8 × 8 RGB pixels] --> B[Outer split by lesion ID<br/>750 train / 300 held out]
    B --> C[Training-only wrapper search<br/>2 group-separated inner folds]
    C --> D[GA, PSO, DE, GA + PSO, GA + DE<br/>56 attempted masks per run]
    D --> E[Selected original features<br/>StandardScaler + balanced logistic regression]
    E --> F[Held-out AUC, log loss,<br/>confusion metrics and CPU time]
```

Related images from the same lesion stay on one side of the outer split. The inner folds also keep lesion groups separate. The positive label is melanoma.

## Blood-cell anomaly table

```mermaid
flowchart LR
    A[5,880 records<br/>22 eligible numeric variables] --> B[Outer stratified row split<br/>750 train / 300 held out]
    B --> C[Training-only wrapper search<br/>2 stratified inner folds]
    C --> D[GA, PSO, DE, GA + PSO, GA + DE<br/>56 attempted masks per run]
    D --> E[Selected original features<br/>StandardScaler + balanced logistic regression]
    E --> F[Held-out AUC, log loss,<br/>confusion metrics and CPU time]
```

The table has no patient identifier, so this row split cannot test generalization to new patients. The positive label is anomaly.

## Which models can be compared directly?

```mermaid
flowchart LR
    X[Original features] --> A[Unsupervised 2D embedding]
    A --> B[Balanced logistic regression on 2 coordinates]
    X --> C[Keep all original features]
    C --> D[Balanced logistic regression on all features]
    X --> E[Inner-fold metaheuristic feature selection]
    E --> F[Balanced logistic regression on selected original features]
```

The direct feature-selection comparison is **all features versus each selected-feature model on the same outer split**. The earlier two-dimensional embedding study changes the feature representation and is a separate experiment. Every path ends in a balanced logistic-regression classifier, but the inputs differ.

| Dataset | Full features: AUC / count / CPU s | Best setting-A selector: AUC / count / CPU s |
|---|---|---|
| HAM10000 | 0.713 / 192 / 0.08 | GA + PSO: 0.735 / 40 / 0.98 |
| Blood cells | 0.809 / 22 / 0.02 | DE: 0.813 / 10.7 / 0.41 |

Values are means of three shared outer splits. CPU times are process seconds on one machine. The AUC differences are small relative to split variation; three overlapping splits do not establish superiority. See [`METAHEURISTIC_FEATURE_SELECTION.md`](METAHEURISTIC_FEATURE_SELECTION.md) for the objective, metrics, parameters and limitations.
