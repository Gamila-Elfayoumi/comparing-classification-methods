"""Checks for the search budget and grouped inner validation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from feature_selection import Fitness, METHODS, optimize


def test_equal_attempted_budget_and_valid_masks():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, 8))
    y = np.repeat([0, 1], 20)
    for method in METHODS[1:]:
        fitness = Fitness(x, y, None, seed=7, penalty=.02)
        mask, trace = optimize(method, fitness, seed=3, population_size=4,
                               generations=2)
        assert mask.sum() >= 2
        assert fitness.attempts == 12
        assert len(trace) == 3


def test_grouped_folds_keep_groups_apart():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(40, 6))
    y = np.tile([0, 1], 20)
    groups = np.repeat(np.arange(20), 2)
    fitness = Fitness(x, y, groups, seed=8, penalty=.02)
    assert all(set(groups[tr]).isdisjoint(groups[va])
               for tr, va in fitness.folds)
