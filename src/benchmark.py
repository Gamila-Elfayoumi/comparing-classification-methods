"""Reproducible two-dataset dimensionality reduction benchmark.

The downstream model is a fixed, balanced logistic regression on two-dimensional
embeddings. Manifold methods are fit only on training data. t-SNE, SNE, and MDS
use a documented five-neighbor out-of-sample map; UMAP and Isomap transform().
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_tmp = Path(__file__).resolve().parents[1] / ".tmp"
_tmp.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_tmp)
os.environ["TEMP"] = str(_tmp)
os.environ["TMP"] = str(_tmp)
os.environ["NUMBA_CACHE_DIR"] = str(_tmp / "numba")

import numpy as np
import pandas as pd
import umap
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.metrics import pairwise_distances
from sklearn.metrics import (accuracy_score, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
METHODS = ["PCA", "Kernel PCA", "SNE", "t-SNE", "PCA + t-SNE", "UMAP", "MDS", "PCoA", "Isomap"]
GRIDS = {
    "PCA": [{"whiten": False}, {"whiten": True}],
    "Kernel PCA": [{"gamma_factor": 0.5}, {"gamma_factor": 2.0}],
    "SNE": [{"perplexity": 15}, {"perplexity": 30}],
    "t-SNE": [{"perplexity": 15}, {"perplexity": 30}],
    "PCA + t-SNE": [{"perplexity": 15}, {"perplexity": 30}],
    "UMAP": [{"n_neighbors": 15}, {"n_neighbors": 30}],
    "MDS": [{"max_iter": 80}, {"max_iter": 150}],
    "PCoA": [{"distance": "cosine"}, {"distance": "manhattan"}],
    "Isomap": [{"n_neighbors": 10}, {"n_neighbors": 25}],
}


def load_dataset(name: str):
    if name == "HAM10000":
        pixels = pd.read_csv(DATA / "ham" / "hmnist_8_8_RGB.csv")
        meta = pd.read_csv(DATA / "ham" / "HAM10000_metadata.csv")
        mapping = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3,
                   "nv": 4, "vasc": 5, "mel": 6}
        if len(pixels) != len(meta) or not np.array_equal(
            pixels["label"].to_numpy(), meta["dx"].map(mapping).to_numpy()
        ):
            raise ValueError("HAM pixel/metadata row alignment failed")
        x = pixels.drop(columns="label").to_numpy(dtype=np.float32) / 255.0
        y = (meta["dx"] == "mel").astype(int).to_numpy()
        groups = meta["lesion_id"].astype(str).to_numpy()
        return x, y, groups, list(pixels.columns[:-1])
    if name == "Blood cell":
        df = pd.read_csv(DATA / "blood" / "blood_cell_anomaly_detection.csv")
        # Restrict to measured cell morphology, colour, and routine blood counts.
        # Exclude cell_type, disease_category, and three model/labeller scores.
        features = ["cell_diameter_um", "nucleus_area_pct", "chromatin_density",
                    "cytoplasm_ratio", "circularity", "eccentricity",
                    "granularity_score", "lobularity_score", "membrane_smoothness",
                    "cell_area_px", "perimeter_px", "mean_r", "mean_g", "mean_b",
                    "stain_intensity", "wbc_count_per_ul",
                    "rbc_count_millions_per_ul", "hemoglobin_g_dl",
                    "hematocrit_pct", "platelet_count_per_ul", "mcv_fl", "mchc_g_dl"]
        if df["cell_id"].duplicated().any() or df[features].isna().any().any():
            raise ValueError("Unexpected duplicate ID or missing blood feature")
        return df[features].to_numpy(dtype=np.float32), df["anomaly_label"].to_numpy(dtype=int), None, features
    raise ValueError(name)


def balanced_subsample(indices, labels, n, seed):
    if len(indices) <= n:
        return indices
    chosen, _ = train_test_split(indices, train_size=n, stratify=labels[indices], random_state=seed)
    return np.sort(chosen)


def split_data(x, y, groups, seed, train_n, test_n):
    idx = np.arange(len(y))
    if groups is None:
        tr, te = train_test_split(idx, test_size=0.28, stratify=y, random_state=seed)
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.28, random_state=seed)
        tr, te = next(splitter.split(x, y, groups))
    tr = balanced_subsample(tr, y, train_n, seed)
    te = balanced_subsample(te, y, test_n, seed + 100)
    if groups is not None and set(groups[tr]) & set(groups[te]):
        raise AssertionError("Lesion leakage")
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        raise ValueError("Split has only one class")
    scaler = StandardScaler().fit(x[tr])
    return scaler.transform(x[tr]), scaler.transform(x[te]), y[tr], y[te], tr, te


def sne_embedding(x, perplexity, seed, steps=130):
    """Symmetric Gaussian SNE, with Gaussian low-dimensional affinity."""
    from sklearn.metrics import pairwise_distances
    distance = pairwise_distances(x, metric="sqeuclidean")
    n = len(x)
    cond = np.zeros((n, n), dtype=np.float32)
    target_entropy = np.log(perplexity)
    for i in range(n):
        lo, hi = 1e-5, 1e3
        d = np.delete(distance[i], i)
        for _ in range(24):
            beta = (lo + hi) / 2
            p = np.exp(-beta * (d - d.min()))
            p /= p.sum()
            entropy = -(p * np.log(p + 1e-12)).sum()
            if entropy > target_entropy:
                lo = beta
            else:
                hi = beta
        cond[i, np.arange(n) != i] = p
    p_sym = (cond + cond.T) / (2 * n)
    np.fill_diagonal(p_sym, 0)
    y0 = PCA(n_components=2, random_state=seed).fit_transform(x).astype(np.float64)
    y0 = 0.1 * y0 / max(float(np.std(y0)), 1e-12)
    trace = []
    iterations = [0]
    def objective(flat):
        y = flat.reshape(n, 2)
        d = pairwise_distances(y, metric="sqeuclidean")
        q = np.exp(-d / 2)
        np.fill_diagonal(q, 0)
        q /= q.sum()
        loss = np.sum(p_sym * (np.log(p_sym + 1e-12) - np.log(q + 1e-12)))
        a = p_sym - q
        grad = 2 * (a.sum(axis=1, keepdims=True) * y - a @ y)
        return float(loss), grad.ravel()
    def callback(intermediate):
        iterations[0] += 1
        if iterations[0] % 10 == 0:
            trace.append([iterations[0], objective(intermediate)[0]])
    result = minimize(objective, y0.ravel(), jac=True, method="L-BFGS-B",
                      callback=callback, options={"maxiter": steps})
    trace.append([int(result.nit), float(result.fun)])
    return result.x.reshape(n, 2), trace


def pcoa_transform(x_train, x_test, metric):
    """Classical principal coordinates with Gower out-of-sample extension."""
    d = pairwise_distances(x_train, metric=metric)
    d2 = d ** 2
    row_mean = d2.mean(axis=1)
    grand_mean = d2.mean()
    gram = -0.5 * (d2 - row_mean[:, None] - row_mean[None, :] + grand_mean)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:2]
    lam = eigenvalues[order]
    vectors = eigenvectors[:, order]
    scale = np.sqrt(np.maximum(lam, 0))
    inverse_scale = np.divide(1.0, scale, out=np.zeros_like(scale), where=scale > 1e-10)
    z_train = vectors * scale
    test_d2 = pairwise_distances(x_test, x_train, metric=metric) ** 2
    centered = -0.5 * (test_d2 - row_mean[None, :] - test_d2.mean(axis=1, keepdims=True) + grand_mean)
    z_test = centered @ (vectors * inverse_scale)
    return z_train, z_test


def embed(method, params, x_train, x_test, seed):
    trace = None
    if method == "PCA":
        model = PCA(n_components=2, whiten=params["whiten"], random_state=seed)
        return model.fit_transform(x_train), model.transform(x_test), trace
    if method == "Kernel PCA":
        model = KernelPCA(n_components=2, kernel="rbf",
                          gamma=params["gamma_factor"] / x_train.shape[1],
                          eigen_solver="arpack", random_state=seed)
        return model.fit_transform(x_train), model.transform(x_test), trace
    if method == "UMAP":
        model = umap.UMAP(n_components=2, n_neighbors=params["n_neighbors"],
                          min_dist=0.1, random_state=seed, n_jobs=1)
        return model.fit_transform(x_train), model.transform(x_test), trace
    if method == "PCoA":
        z_train, z_test = pcoa_transform(x_train, x_test, params["distance"])
        return z_train, z_test, trace
    if method == "Isomap":
        model = Isomap(n_components=2, n_neighbors=params["n_neighbors"], n_jobs=1)
        return model.fit_transform(x_train), model.transform(x_test), trace
    if method == "MDS":
        model = MDS(n_components=2, metric=True, n_init=1, max_iter=params["max_iter"],
                    normalized_stress="auto", random_state=seed, n_jobs=1)
        z_train = model.fit_transform(x_train)
        mapper = KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=1)
        mapper.fit(x_train, z_train)
        return z_train, mapper.predict(x_test), [[int(model.n_iter_), float(model.stress_)]]
    source_train, source_test = x_train, x_test
    if method == "PCA + t-SNE":
        pre = PCA(n_components=min(30, x_train.shape[1] - 1), random_state=seed)
        source_train = pre.fit_transform(x_train)
        source_test = pre.transform(x_test)
    if method == "SNE":
        z_train, trace = sne_embedding(source_train, params["perplexity"], seed)
    else:
        model = TSNE(n_components=2, perplexity=params["perplexity"],
                     init="pca", learning_rate="auto", max_iter=500,
                     method="barnes_hut", random_state=seed)
        z_train = model.fit_transform(source_train)
        trace = [[500, float(model.kl_divergence_)]]
    # Explicit approximation because sklearn t-SNE and this SNE have no transform.
    mapper = KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=1)
    mapper.fit(source_train, z_train)
    return z_train, mapper.predict(source_test), trace


def binary_metrics(y, prediction, probability):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else np.nan
    acc = (tp + tn) / len(y)
    prevalence = (tp + fn) / len(y)
    predicted = (tp + fp) / len(y)
    chance = prevalence * predicted + (1 - prevalence) * (1 - predicted)
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                accuracy=acc, error=1-acc, sensitivity=sensitivity,
                specificity=specificity, precision=precision, recall=sensitivity,
                f_measure=f1, auc_roc=roc_auc_score(y, probability),
                kappa=(acc-chance)/(1-chance),
                informedness=sensitivity+specificity-1,
                markedness=precision+npv-1,
                log_loss=log_loss(y, probability, labels=[0, 1]))


def run(args):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "raw").mkdir(exist_ok=True)
    selected_methods = METHODS if args.methods == "all" else [x.strip() for x in args.methods.split(",")]
    if any(method not in METHODS for method in selected_methods):
        raise ValueError(f"Unknown methods: {selected_methods}")
    warm_cpu = 0.0
    if "UMAP" in selected_methods:
        # Numba's first-call compilation is environment setup, not per-run fitting.
        warm = np.random.default_rng(7).normal(size=(50, 8)).astype(np.float32)
        warm_start = time.process_time()
        warm_model = umap.UMAP(n_neighbors=15, n_components=2, random_state=7, n_jobs=1)
        warm_model.fit(warm)
        warm_model.transform(warm[:5])
        warm_cpu = time.process_time() - warm_start
        print(f"UMAP JIT warmup CPU={warm_cpu:.2f}s", flush=True)
    rows, curves, traces = [], [], []
    if args.append:
        rows = pd.read_csv(RESULTS / "runs.csv").to_dict("records")
        curves = json.loads((RESULTS / "roc_curves.json").read_text(encoding="utf-8"))
        traces = json.loads((RESULTS / "optimization_traces.json").read_text(encoding="utf-8"))
        warm_cpu = json.loads((RESULTS / "protocol.json").read_text(encoding="utf-8")).get("umap_warmup_cpu_seconds", warm_cpu)
    for dataset in ["HAM10000", "Blood cell"]:
        x, y, groups, features = load_dataset(dataset)
        print(f"{dataset}: n={len(y)}, features={len(features)}, positives={int(y.sum())}", flush=True)
        for seed in range(args.seeds):
            xtr, xte, ytr, yte, tr, te = split_data(x, y, groups, 100 + seed,
                                                    args.train_n, args.test_n)
            for method in selected_methods:
                for setting, params in enumerate(GRIDS[method]):
                    if any(r["dataset"] == dataset and int(r["seed"]) == seed and
                           r["method"] == method and int(r["setting"]) == setting for r in rows):
                        continue
                    start_cpu = time.process_time()
                    start_wall = time.perf_counter()
                    with threadpool_limits(limits=1):
                        ztr, zte, trace = embed(method, params, xtr, xte, 100 + seed)
                        classifier = LogisticRegression(class_weight="balanced", max_iter=1000,
                                                        random_state=100 + seed)
                        classifier.fit(ztr, ytr)
                        prob = classifier.predict_proba(zte)[:, 1]
                        pred = (prob >= 0.5).astype(int)
                    cpu = time.process_time() - start_cpu
                    wall = time.perf_counter() - start_wall
                    metrics = binary_metrics(yte, pred, prob)
                    row = dict(dataset=dataset, seed=seed, method=method, setting=setting,
                               params=json.dumps(params, sort_keys=True), n_train=len(tr),
                               n_test=len(te), prevalence=float(yte.mean()), cpu_seconds=cpu,
                               wall_seconds=wall, **metrics)
                    rows.append(row)
                    fpr, tpr, thresholds = roc_curve(yte, prob)
                    curves.append(dict(dataset=dataset, seed=seed, method=method,
                                       setting=setting, fpr=fpr.tolist(), tpr=tpr.tolist()))
                    if trace is not None:
                        traces.append(dict(dataset=dataset, seed=seed, method=method,
                                           setting=setting, trace=trace))
                    np.savez_compressed(RESULTS / "raw" / f"{dataset.replace(' ','_')}_{seed}_{method.replace(' ','_')}_{setting}.npz",
                                        y=yte, pred=pred, prob=prob, test_index=te,
                                        embedding_test=zte)
                    print(f"{dataset} seed={seed} {method} {params}: AUC={metrics['auc_roc']:.3f} CPU={cpu:.2f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "runs.csv", index=False)
    (RESULTS / "roc_curves.json").write_text(json.dumps(curves), encoding="utf-8")
    (RESULTS / "optimization_traces.json").write_text(json.dumps(traces), encoding="utf-8")
    (RESULTS / "protocol.json").write_text(json.dumps({**vars(args), "all_methods": METHODS,
                                                         "umap_warmup_cpu_seconds": warm_cpu}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--train-n", type=int, default=450)
    parser.add_argument("--test-n", type=int, default=180)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--append", action="store_true")
    run(parser.parse_args())
