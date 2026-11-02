"""Leakage-aware wrapper feature selection on the two AML datasets.

GA, binary PSO, real-coded DE, GA->PSO, and GA->DE share the same fitness
function and attempted evaluation budget. The outer test split is untouched
until a feature mask has been selected using inner training folds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from benchmark import binary_metrics, load_dataset, split_data

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("Full", "GA", "PSO", "DE", "GA + PSO", "GA + DE")


def repair(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    if mask.sum() < 2:
        mask[rng.choice(mask.size, size=2, replace=False)] = True
    return mask


def initial_population(rng, size, features):
    return np.vstack([repair(rng.random(features) < rng.uniform(.15, .55), rng)
                      for _ in range(size)])


class Fitness:
    def __init__(self, x, y, groups, seed, penalty):
        self.x, self.y, self.penalty = x, y, penalty
        if groups is None:
            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            self.folds = list(cv.split(x, y))
        else:
            cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
            self.folds = list(cv.split(x, y, groups))
            assert all(set(groups[tr]).isdisjoint(groups[va]) for tr, va in self.folds)
        self.cache = {}
        self.attempts = 0

    def __call__(self, mask):
        self.attempts += 1
        key = np.asarray(mask, dtype=bool).tobytes()
        if key in self.cache:
            return self.cache[key]
        if np.count_nonzero(mask) < 2:
            return -np.inf
        scores = []
        for tr, va in self.folds:
            if len(np.unique(self.y[tr])) < 2 or len(np.unique(self.y[va])) < 2:
                return -np.inf
            scaler = StandardScaler().fit(self.x[tr][:, mask])
            xt = scaler.transform(self.x[tr][:, mask])
            xv = scaler.transform(self.x[va][:, mask])
            clf = LogisticRegression(class_weight="balanced", max_iter=1000,
                                     random_state=42)
            clf.fit(xt, self.y[tr])
            from sklearn.metrics import roc_auc_score
            scores.append(roc_auc_score(self.y[va], clf.predict_proba(xv)[:, 1]))
        result = (float(np.mean(scores) - self.penalty * np.mean(mask)),
                  float(np.mean(scores)))
        self.cache[key] = result
        return result


def optimize(method, fitness, seed, population_size=8, generations=6):
    """Equal attempted evaluations: population_size * (generations + 1)."""
    rng = np.random.default_rng(seed)
    d = fitness.x.shape[1]
    population = initial_population(rng, population_size, d)
    scores = np.array([fitness(mask)[0] for mask in population])
    best_idx = int(np.argmax(scores))
    best, best_score = population[best_idx].copy(), float(scores[best_idx])
    trajectory = [dict(generation=0, best_fitness=best_score,
                       best_features=int(best.sum()))]
    velocities = rng.normal(0, .3, size=(population_size, d))
    personal = population.copy()
    personal_scores = scores.copy()
    real_population = np.where(population, .75, .25).astype(float)

    for generation in range(1, generations + 1):
        phase = method
        if method == "GA + PSO":
            phase = "GA" if generation <= generations // 2 else "PSO"
        elif method == "GA + DE":
            phase = "GA" if generation <= generations // 2 else "DE"
        candidates = []
        if phase == "GA":
            candidates.append(best.copy())
            while len(candidates) < population_size:
                i, j = rng.integers(population_size, size=2)
                a = population[i] if scores[i] >= scores[j] else population[j]
                i, j = rng.integers(population_size, size=2)
                b = population[i] if scores[i] >= scores[j] else population[j]
                child = np.where(rng.random(d) < .5, a, b)
                child = np.logical_xor(child, rng.random(d) < max(1 / d, .02))
                candidates.append(repair(child, rng))
        elif phase == "PSO":
            velocities = (.55 * velocities + 1.3 * rng.random((population_size, d)) *
                          (personal.astype(float) - population.astype(float)) +
                          1.3 * rng.random((population_size, d)) *
                          (best.astype(float) - population.astype(float)))
            velocities = np.clip(velocities, -4, 4)
            candidates = [repair(rng.random(d) < expit(v), rng)
                          for v in velocities]
        elif phase == "DE":
            for i in range(population_size):
                a, b, c = rng.choice([j for j in range(population_size) if j != i],
                                     size=3, replace=False)
                donor = np.clip(real_population[a] + .6 *
                                (real_population[b] - real_population[c]), 0, 1)
                crossing = rng.random(d) < .7
                crossing[rng.integers(d)] = True
                trial_real = np.where(crossing, donor, real_population[i])
                candidates.append(repair(trial_real >= .5, rng))
        else:
            raise ValueError(method)
        candidate_masks = np.vstack(candidates)
        candidate_scores = np.array([fitness(mask)[0] for mask in candidate_masks])
        if phase == "DE":
            keep = candidate_scores > scores
            population[keep] = candidate_masks[keep]
            scores[keep] = candidate_scores[keep]
            real_population[keep] = np.where(population[keep], .75, .25)
        else:
            population, scores = candidate_masks, candidate_scores
            real_population = np.where(population, .75, .25).astype(float)
        better = scores > personal_scores
        personal[better], personal_scores[better] = population[better], scores[better]
        idx = int(np.argmax(scores))
        if scores[idx] > best_score:
            best, best_score = population[idx].copy(), float(scores[idx])
        trajectory.append(dict(generation=generation, best_fitness=best_score,
                               best_features=int(best.sum())))
    assert fitness.attempts == population_size * (generations + 1)
    return best, trajectory


def evaluate(xtr, ytr, xte, yte, mask, seed):
    scaler = StandardScaler().fit(xtr[:, mask])
    clf = LogisticRegression(class_weight="balanced", max_iter=1000,
                             random_state=seed)
    clf.fit(scaler.transform(xtr[:, mask]), ytr)
    probability = clf.predict_proba(scaler.transform(xte[:, mask]))[:, 1]
    prediction = (probability >= .5).astype(int)
    return binary_metrics(yte, prediction, probability), probability, prediction


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, traces, selections, predictions = [], [], [], []
    penalties = [float(v) for v in args.penalties.split(",")]
    if not penalties or any(v < 0 for v in penalties):
        raise ValueError("Penalties must be nonnegative")
    for dataset in ("HAM10000", "Blood cell"):
        x, y, groups, feature_names = load_dataset(dataset)
        for seed in range(args.seeds):
            split_seed = 100 + seed
            _, _, ytr, yte, tr, te = split_data(x, y, groups, split_seed,
                                                args.train_n, args.test_n)
            xtr, xte = x[tr], x[te]
            inner_groups = None if groups is None else groups[tr]
            for method in METHODS:
                settings = [None] if method == "Full" else penalties
                for setting, penalty in enumerate(settings):
                    start_cpu = time.process_time()
                    with threadpool_limits(limits=1):
                        if method == "Full":
                            mask = np.ones(x.shape[1], dtype=bool)
                            trajectory = []
                            fitness = None
                        else:
                            fitness = Fitness(xtr, ytr, inner_groups,
                                              split_seed, penalty)
                            mask, trajectory = optimize(method, fitness,
                                                       10000 * (seed + 1) +
                                                       100 * setting + METHODS.index(method),
                                                       args.population, args.generations)
                        metrics, prob, pred = evaluate(xtr, ytr, xte, yte,
                                                       mask, split_seed)
                    cpu = time.process_time() - start_cpu
                    inner_auc = (np.nan if fitness is None else
                                 fitness.cache[mask.tobytes()][1])
                    row = dict(dataset=dataset, seed=seed, method=method,
                               setting=setting, sparsity_penalty=penalty,
                               n_train=len(tr), n_test=len(te),
                               n_features=int(mask.sum()), feature_fraction=float(mask.mean()),
                               evaluations=0 if fitness is None else fitness.attempts,
                               unique_masks=0 if fitness is None else len(fitness.cache),
                               inner_auc=inner_auc,
                               cpu_seconds=cpu, **metrics)
                    rows.append(row)
                    selections.append(dict(dataset=dataset, seed=seed,
                                           method=method, setting=setting,
                                           features=json.dumps([feature_names[i]
                                                                for i in np.flatnonzero(mask)])))
                    for point in trajectory:
                        traces.append(dict(dataset=dataset, seed=seed,
                                           method=method, setting=setting, **point))
                    predictions.extend(dict(dataset=dataset, seed=seed,
                                            method=method, setting=setting,
                                            test_index=int(te[i]), y=int(yte[i]),
                                            probability=float(prob[i]), pred=int(pred[i]))
                                       for i in range(len(te)))
                    print(f"{dataset} seed={seed} {method} penalty={penalty}: "
                          f"AUC={metrics['auc_roc']:.3f}, features={mask.sum()}, CPU={cpu:.1f}s",
                          flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    pd.DataFrame(traces).to_csv(out / "trajectories.csv", index=False)
    pd.DataFrame(selections).to_csv(out / "selected_features.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
    primary = df[(df.method == "Full") | (df.setting == 0)]
    primary.groupby(["dataset", "method"], sort=False).agg(
        auc_roc_mean=("auc_roc", "mean"), auc_roc_sd=("auc_roc", "std"),
        accuracy_mean=("accuracy", "mean"), sensitivity_mean=("sensitivity", "mean"),
        specificity_mean=("specificity", "mean"), precision_mean=("precision", "mean"),
        f_measure_mean=("f_measure", "mean"), log_loss_mean=("log_loss", "mean"),
        selected_features_mean=("n_features", "mean"),
        cpu_seconds_mean=("cpu_seconds", "mean")).reset_index().to_csv(
            out / "primary_summary.csv", index=False)
    tests = []
    for dataset in ("HAM10000", "Blood cell"):
        base = primary[(primary.dataset == dataset) & (primary.method == "Full")].sort_values("seed")
        for method in METHODS[1:]:
            other = primary[(primary.dataset == dataset) & (primary.method == method)].sort_values("seed")
            d = other.auc_roc.to_numpy() - base.auc_roc.to_numpy()
            tests.append(dict(dataset=dataset, method=method,
                              mean_auc_delta=float(d.mean()),
                              paired_t_p=float(ttest_rel(other.auc_roc, base.auc_roc).pvalue),
                              wilcoxon_p=float(wilcoxon(d).pvalue)))
    pd.DataFrame(tests).to_csv(out / "exploratory_tests.csv", index=False)
    (out / "protocol.json").write_text(json.dumps(dict(
        datasets=["HAM10000", "Blood cell"], methods=METHODS,
        seeds=args.seeds, train_n=args.train_n, test_n=args.test_n,
        penalties=penalties, population=args.population,
        generations=args.generations, inner_folds=2,
        objective="mean inner ROC AUC - penalty * feature fraction",
        threshold=.5, group_split="lesion_id for HAM10000; rows for blood cell",
        note="Exploratory comparison; three overlapping splits are not independent."
    ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--train-n", type=int, default=750)
    parser.add_argument("--test-n", type=int, default=300)
    parser.add_argument("--penalties", default="0.02,0.05")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--output", default=str(ROOT / "results" / "feature_selection"))
    args = parser.parse_args()
    if args.population < 4 or args.generations < 2 or args.seeds < 1:
        parser.error("population >= 4, generations >= 2, seeds >= 1 required")
    run(args)
