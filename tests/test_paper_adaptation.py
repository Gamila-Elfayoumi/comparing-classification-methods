"""Invariants specific to the published-method adaptation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paper_rlwoa_sofl import PaperFitness, PrototypeFuzzyClassifier, optimize_woa


def test_training_group_folds_and_search_budget():
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(20), 2)
    y = np.tile([0, 1], 20)
    x = rng.normal(size=(40, 8))
    for adaptive in (False, True):
        f = PaperFitness(x, y, groups, seed=4)
        assert all(set(groups[a]).isdisjoint(groups[b]) for a, b in f.folds)
        mask, trajectory = optimize_woa(f, seed=5, adaptive=adaptive,
                                        agents=4, iterations=2)
        assert f.attempts == 12
        assert mask.sum() >= 2
        assert len(trajectory) == 3


def test_fuzzy_probabilities_are_valid_and_class_sensitive():
    x = np.array([[1., 0.], [.95, .05], [0., 1.], [.05, .95]])
    y = np.array([0, 0, 1, 1])
    model = PrototypeFuzzyClassifier(radius=.1).fit(x, y)
    prob = model.predict_proba(np.array([[1., 0.], [0., 1.]]))
    assert np.allclose(prob.sum(axis=1), 1)
    assert prob[0, 0] > prob[0, 1]
    assert prob[1, 1] > prob[1, 0]
