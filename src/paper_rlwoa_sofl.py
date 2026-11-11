"""Auditable RLWOA-SOFL adaptation for the two AML benchmark tables.

This is an adaptation of Anter & Zhang (2023), DOI 10.1109/TAFFC.2023.3285997,
not a claim of exact reproduction: the source fMRI data and full SOFL reference
implementation are unavailable. Search fitness follows its KNN accuracy plus
feature sparsity equation; the learned action policy and fuzzy prototype model
are explicit, reproducible approximations documented in docs/PAPER_ADAPTATION.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from benchmark import binary_metrics, load_dataset, split_data
from feature_selection import evaluate

ROOT = Path(__file__).resolve().parents[1]


def repair(mask, rng):
    mask = np.asarray(mask, dtype=bool).copy()
    if mask.sum() < 2:
        mask[rng.choice(mask.size, 2, replace=False)] = True
    return mask


class PaperFitness:
    """Paper Eq. 14, evaluated on training-only inner folds."""

    def __init__(self, x, y, groups, seed, alpha=.8):
        self.x, self.y, self.alpha = x, y, alpha
        cv = (StratifiedKFold(2, shuffle=True, random_state=seed) if groups is None
              else StratifiedGroupKFold(2, shuffle=True, random_state=seed))
        self.folds = list(cv.split(x, y, groups))
        if groups is not None:
            assert all(set(groups[a]).isdisjoint(groups[b]) for a, b in self.folds)
        self.cache = {}
        self.attempts = 0

    def __call__(self, mask):
        self.attempts += 1
        key = mask.tobytes()
        if key not in self.cache:
            scores = []
            for tr, va in self.folds:
                scaler = StandardScaler().fit(self.x[tr][:, mask])
                model = KNeighborsClassifier(n_neighbors=5)
                model.fit(scaler.transform(self.x[tr][:, mask]), self.y[tr])
                scores.append(model.score(scaler.transform(self.x[va][:, mask]), self.y[va]))
            accuracy = float(np.mean(scores))
            self.cache[key] = (accuracy + self.alpha * (1 - mask.mean()), accuracy)
        return self.cache[key]


class AgentPolicy:
    """Small online neural state-action policy, trained from reward per agent."""

    def __init__(self, rng, hidden=7):
        self.w1 = rng.normal(0, .2, (3, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, .2, (hidden, 3))
        self.b2 = np.zeros(3)

    def choose(self, state, rng):
        h = np.tanh(state @ self.w1 + self.b1)
        p = softmax(h @ self.w2 + self.b2)
        return int(rng.choice(3, p=p)), h, p

    def learn(self, state, action, h, p, reward, rate=.035):
        # REINFORCE gradient on log policy; reward is sign of fitness change.
        delta = -p.copy()
        delta[action] += 1
        delta *= reward
        old_w2 = self.w2.copy()
        self.w2 += rate * np.outer(h, delta)
        self.b2 += rate * delta
        dh = (old_w2 @ delta) * (1 - h * h)
        self.w1 += rate * np.outer(state, dh)
        self.b1 += rate * dh


def optimize_woa(fitness, seed, adaptive, agents=5, iterations=20):
    rng = np.random.default_rng(seed)
    d = fitness.x.shape[1]
    positions = rng.random((agents, d))
    masks = np.vstack([repair(p >= .5, rng) for p in positions])
    values = np.array([fitness(m)[0] for m in masks])
    best_i = int(values.argmax())
    best, best_value = positions[best_i].copy(), float(values[best_i])
    best_mask = masks[best_i].copy()
    rates = np.full(agents, 2.0)
    policies = [AgentPolicy(rng) for _ in range(agents)]
    history = np.zeros((agents, 3))
    trace = [dict(iteration=0, best_fitness=best_value, selected=int(best_mask.sum()),
                  positive_rewards=0, negative_rewards=0)]
    positive = negative = 0
    for t in range(1, iterations + 1):
        next_positions = np.empty_like(positions)
        next_masks = np.empty_like(masks)
        next_values = np.empty_like(values)
        for i in range(agents):
            state = history[i].copy()
            if adaptive:
                action, hidden, probs = policies[i].choose(state, rng)
                rates[i] = np.clip(rates[i] * (1 + .1 * (action - 1)), .05, 2.0)
                a = rates[i]
            else:
                a = 2 * (1 - (t - 1) / iterations)
            r = rng.random(d)
            A = 2 * a * r - a
            C = 2 * rng.random(d)
            if rng.random() < .5:
                reference = positions[rng.integers(agents)] if np.mean(np.abs(A)) > 1 else best
                new_position = reference - A * np.abs(C * reference - positions[i])
            else:
                l = rng.uniform(-1, 1)
                new_position = np.abs(best - positions[i]) * np.exp(l) * np.cos(2 * np.pi * l) + best
            next_positions[i] = np.clip(new_position, 0, 1)
            # Paper uses a sigmoid transfer to map real coordinates to bits.
            next_masks[i] = repair(rng.random(d) < expit(4 * (next_positions[i] - .5)), rng)
            next_values[i] = fitness(next_masks[i])[0]
            reward = float(np.sign(next_values[i] - values[i]))
            positive += reward > 0
            negative += reward < 0
            history[i] = np.roll(history[i], -1)
            history[i, -1] = reward
            if adaptive:
                policies[i].learn(state, action, hidden, probs, reward)
        positions, masks, values = next_positions, next_masks, next_values
        j = int(values.argmax())
        if values[j] > best_value:
            best, best_mask, best_value = positions[j].copy(), masks[j].copy(), float(values[j])
        trace.append(dict(iteration=t, best_fitness=best_value,
                          selected=int(best_mask.sum()), positive_rewards=int(positive),
                          negative_rewards=int(negative)))
    assert fitness.attempts == agents * (iterations + 1)
    return best_mask, trace


class PrototypeFuzzyClassifier:
    """Online class prototypes with cosine similarity and soft fuzzy decisions.

    A tractable SOFL-inspired classifier, not the source paper's full AnYa model.
    """

    def __init__(self, radius=.25, max_prototypes=20, temperature=.12):
        self.radius = radius
        self.max_prototypes = max_prototypes
        self.temperature = temperature

    @staticmethod
    def _unit(x):
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norm, 1e-12)

    def fit(self, x, y):
        x = self._unit(np.asarray(x, float))
        self.classes_ = np.unique(y)
        self.prototypes_ = {}
        for c in self.classes_:
            centers, counts = [], []
            for row in x[y == c]:
                if not centers:
                    centers.append(row.copy()); counts.append(1); continue
                sim = np.array(centers) @ row
                j = int(sim.argmax())
                if 1 - sim[j] > self.radius and len(centers) < self.max_prototypes:
                    centers.append(row.copy()); counts.append(1)
                else:
                    counts[j] += 1
                    centers[j] = (centers[j] * (counts[j] - 1) + row) / counts[j]
                    centers[j] /= max(np.linalg.norm(centers[j]), 1e-12)
            self.prototypes_[c] = np.array(centers)
        return self

    def predict_proba(self, x):
        x = self._unit(np.asarray(x, float))
        similarity = np.column_stack([np.max(x @ self.prototypes_[c].T, axis=1)
                                      for c in self.classes_])
        return softmax(similarity / self.temperature, axis=1)


def evaluate_fuzzy(xtr, ytr, xte, yte, mask, radius):
    scaler = StandardScaler().fit(xtr[:, mask])
    model = PrototypeFuzzyClassifier(radius=radius).fit(
        scaler.transform(xtr[:, mask]), ytr)
    prob = model.predict_proba(scaler.transform(xte[:, mask]))[:, 1]
    pred = (prob >= .5).astype(int)
    return binary_metrics(yte, pred, prob), prob, pred, sum(len(p) for p in model.prototypes_.values())


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    runs, traces, predictions, selections = [], [], [], []
    for dataset in ("HAM10000", "Blood cell"):
        x, y, groups, feature_names = load_dataset(dataset)
        for seed in range(args.seeds):
            split_seed = 100 + seed
            _, _, ytr, yte, tr, te = split_data(x, y, groups, split_seed,
                                                args.train_n, args.test_n)
            xtr, xte = x[tr], x[te]
            for method in ("WOA + LR", "Adaptive WOA + LR", "Adaptive WOA + fuzzy", "Full + fuzzy"):
                start = time.process_time()
                with threadpool_limits(limits=1):
                    if method == "Full + fuzzy":
                        mask = np.ones(x.shape[1], bool)
                        trace = []
                        fit = None
                    else:
                        fit = PaperFitness(xtr, ytr, None if groups is None else groups[tr],
                                           split_seed, args.alpha)
                        mask, trace = optimize_woa(fit, 1000 + seed * 10 +
                                                   (0 if method == "WOA + LR" else 1),
                                                   method != "WOA + LR", args.agents, args.iterations)
                    if "fuzzy" in method:
                        metrics, prob, pred, n_prototypes = evaluate_fuzzy(
                            xtr, ytr, xte, yte, mask, args.radius)
                    else:
                        metrics, prob, pred = evaluate(xtr, ytr, xte, yte, mask, split_seed)
                        n_prototypes = 0
                cpu = time.process_time() - start
                runs.append(dict(dataset=dataset, seed=seed, method=method,
                                 alpha=args.alpha, radius=args.radius, n_train=len(tr),
                                 n_test=len(te), n_features=int(mask.sum()),
                                 n_prototypes=n_prototypes,
                                 evaluations=0 if fit is None else fit.attempts,
                                 inner_accuracy=np.nan if fit is None else fit.cache[mask.tobytes()][1],
                                 cpu_seconds=cpu, **metrics))
                selections.append(dict(dataset=dataset, seed=seed, method=method,
                                       features=json.dumps([feature_names[j] for j in np.flatnonzero(mask)])))
                traces.extend(dict(dataset=dataset, seed=seed, method=method, **point)
                              for point in trace)
                predictions.extend(dict(dataset=dataset, seed=seed, method=method,
                                        test_index=int(te[j]), y=int(yte[j]),
                                        probability=float(prob[j]), pred=int(pred[j]))
                                   for j in range(len(te)))
                print(f"{dataset} seed={seed} {method}: AUC={metrics['auc_roc']:.3f} "
                      f"features={mask.sum()} CPU={cpu:.2f}s", flush=True)
    df = pd.DataFrame(runs)
    df.to_csv(out / "runs.csv", index=False)
    pd.DataFrame(traces).to_csv(out / "trajectories.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(out / "selected_features.csv", index=False)
    df.groupby(["dataset", "method"], sort=False).agg(
        auc_roc_mean=("auc_roc", "mean"), auc_roc_sd=("auc_roc", "std"),
        accuracy_mean=("accuracy", "mean"), sensitivity_mean=("sensitivity", "mean"),
        specificity_mean=("specificity", "mean"), f_measure_mean=("f_measure", "mean"),
        log_loss_mean=("log_loss", "mean"), n_features_mean=("n_features", "mean"),
        cpu_seconds_mean=("cpu_seconds", "mean")).reset_index().to_csv(out / "summary.csv", index=False)
    (out / "protocol.json").write_text(json.dumps(vars(args), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "results" / "paper_rlwoa_sofl"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--train-n", type=int, default=750)
    parser.add_argument("--test-n", type=int, default=300)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=.8)
    parser.add_argument("--radius", type=float, default=.25)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
