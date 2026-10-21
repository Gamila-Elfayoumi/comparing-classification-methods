"""Small numerical and leakage checks for the benchmark protocol."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from benchmark import binary_metrics, load_dataset, split_data, pcoa_transform


def test_binary_metrics():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 0])
    prob = np.array([.1, .8, .9, .2])
    m = binary_metrics(y, pred, prob)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["accuracy"] == .5
    assert m["sensitivity"] == m["recall"] == .5
    assert m["specificity"] == m["precision"] == m["f_measure"] == .5
    assert m["kappa"] == 0
    assert m["informedness"] == m["markedness"] == 0


def test_ham_group_split():
    x, y, groups, _ = load_dataset("HAM10000")
    _, _, _, _, train_idx, test_idx = split_data(x, y, groups, 100, 750, 300)
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
    assert len(train_idx) == 750 and len(test_idx) == 300


def test_pcoa_extension_reproduces_training_coordinates():
    x = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 1.]])
    training, extended = pcoa_transform(x, x, "cosine")
    assert np.allclose(training, extended, atol=1e-7)
