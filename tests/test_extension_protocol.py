"""Check scientific invariants without fitting large experiment grids."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('extension_common', Path(__file__).resolve().parents[1]/'experiments/extensions/common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def test_budget_changes_do_not_change_pool_or_truth():
    y = np.arange(624) % 2
    pool, heldout = common.nested_split(len(y), 420, 100)
    assert len(heldout) == 204
    assert not set(pool) & set(heldout)
    assert len(set(pool) | set(heldout)) == len(y)
    previous = set()
    for budget in (20, 50, 100):
        partial = common.partial_labels(y[pool], budget)
        selected = set(pool[partial >= 0])
        assert previous <= selected
        previous = selected
        assert (partial[budget:] == -1).all()
        np.testing.assert_array_equal(partial[:budget], y[pool[:budget]])
        pool2, heldout2 = common.nested_split(len(y), 420, 100)
        np.testing.assert_array_equal(heldout, heldout2)
        np.testing.assert_array_equal(pool, pool2)


def test_exact_accuracy_and_selection_ties():
    p = np.array([[[.8,.2],[.3,.7]],[[.4,.6],[.4,.6]]])
    q = np.array([[.7,.3],[.2,.8]])
    values, _ = common.posterior_metrics(p,q)
    np.testing.assert_allclose(values['accuracy'], [.75,.55])
    result = common.selection(np.array([.8,.8]),np.array([.7,.9]))
    assert result['top1'] == .5
    assert result['selection_regret'] == pytest.approx(.1)
    assert common.selection(np.array([np.nan,.8]),np.array([.7,.9]))['mae'] is None


def test_paired_bounds_retain_inconclusive_results():
    inconclusive = common.paired_statistics([-.01,.01,-.01,.01])
    assert inconclusive['adjusted_upper'] > 0
    assert not inconclusive['supports_lower_error']
    improving = common.paired_statistics([-.02,-.021,-.022]*10)
    assert improving['supports_lower_error']
    assert improving['adjusted_upper'] > improving['mean_difference']
