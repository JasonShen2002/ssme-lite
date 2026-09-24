"""SemiSupervisedSplit keeps the benchmark permutation prefix."""
import numpy as np
import pytest
from ssme_lite.splits import SemiSupervisedSplit


def test_permutation_prefix_and_partition():
    y = np.arange(80)
    parts = SemiSupervisedSplit(7, 30, random_state=3).indices(len(y))
    order = np.random.default_rng(3).permutation(len(y))
    assert np.array_equal(parts["labeled"], order[:7])
    assert np.array_equal(parts["unlabeled"], order[7:37])
    assert np.array_equal(parts["estimation"], order[:37])
    assert np.array_equal(parts["heldout"], order[37:])
    covered = np.concatenate([parts["estimation"], parts["heldout"]])
    assert sorted(covered.tolist()) == list(range(80))
    assert len(np.intersect1d(parts["labeled"], parts["unlabeled"])) == 0
    assert len(np.intersect1d(parts["estimation"], parts["heldout"])) == 0


def test_larger_label_budget_extends_the_same_prefix():
    small = SemiSupervisedSplit(5, 20, random_state=1).indices(60)
    large = SemiSupervisedSplit(12, 20, random_state=1).indices(60)
    assert np.array_equal(large["labeled"][:5], small["labeled"])


def test_partial_labels_hide_the_unlabeled_prefix():
    y = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    splitter = SemiSupervisedSplit(3, 4, random_state=0)
    parts = splitter.indices(len(y))
    partial = splitter.partial_labels(y, parts)
    assert np.array_equal(partial[:3], y[parts["labeled"]])
    assert np.all(partial[3:] == -1)


def test_rejects_a_split_without_heldout():
    with pytest.raises(ValueError):
        SemiSupervisedSplit(5, 5, random_state=0).indices(10)
    with pytest.raises(ValueError):
        SemiSupervisedSplit(0, 5)
