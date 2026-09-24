import json
import numpy as np
import pytest
from ssme_lite import SSMEEstimator
from ssme_lite.contribution import model_contribution
from ssme_lite.experiments import benchmark
from ssme_lite.report_data import histogram


def inputs():
    rng = np.random.default_rng(4)
    scores = rng.uniform(.05, .95, (36, 3))
    scores[:, 2] = .5
    labels = np.full(36, -1)
    labels[:4] = [0, 1, 0, 1]
    return scores, labels


def test_fixed_epochs_ignore_tolerance_and_default_to_100():
    p, y = inputs()
    fixed = SSMEEstimator(tol=2).fit(p, y)
    assert fixed.n_iter_ == len(fixed.history_) == 100
    assert fixed.stop_reason_ == 'fixed_epochs'
    early = SSMEEstimator(max_iter=100, tol=2, early_stopping=True).fit(p, y)
    assert early.n_iter_ == 1 and early.stop_reason_ == 'tolerance'
    forced = SSMEEstimator(max_iter=20, tol=2).fit(p, y)
    assert forced.n_iter_ == 20
    assert forced.report(n_draws=2).metadata['early_stopping'] is False


def test_controlled_ablation_constant_view_and_exact_accuracy():
    p, y = inputs()
    est = SSMEEstimator(max_iter=3, bandwidth=.8).fit(p, y)
    saved = est.posterior_.copy()
    c = est.report(n_draws=2, contribution=True).data['contribution']
    assert c['n_refits'] == 3 and c['n_samples'] == int((y < 0).sum())
    assert c['rows'][2]['posterior_total_variation'] < 1e-12
    assert all(r['epochs'] == 3 and r['bandwidth'] == .8 for r in c['rows'])
    for i in range(3):
        assert c['accuracy_delta_matrix'][i][i] is None
        assert 0 <= c['rows'][i]['posterior_total_variation'] <= 1
        assert 0 <= c['rows'][i]['posterior_js_divergence_nats'] <= np.log(2)
    keep = [1, 2]
    reduced = SSMEEstimator(max_iter=3, bandwidth=.8).fit(p[:, keep], y,
                  initial_responsibilities=est.initial_responsibilities_)
    u = y < 0
    predictions = est.scores_[u][:, keep].argmax(2)
    delta = (reduced.posterior_[u][np.arange(u.sum())[:, None], predictions]
             -est.posterior_[u][np.arange(u.sum())[:, None], predictions]).mean(0)
    np.testing.assert_allclose(c['accuracy_delta_matrix'][0][1:], delta, atol=1e-12)
    np.testing.assert_array_equal(est.posterior_, saved)
    json.dumps(c, allow_nan=False)


def test_initial_responsibilities_validation_and_no_mutation():
    p, y = inputs()
    initial = np.full((len(y), 2), .5)
    copy = initial.copy()
    est = SSMEEstimator(max_iter=2).fit(p, y, initial_responsibilities=initial)
    np.testing.assert_array_equal(initial, copy)
    np.testing.assert_array_equal(est.initial_responsibilities_[:4], np.eye(2)[y[:4]])
    with pytest.raises(ValueError, match='normalized'):
        SSMEEstimator().fit(p, y, initial_responsibilities=initial*.5)
    with pytest.raises(ValueError, match='unlabeled'):
        model_contribution(SSMEEstimator(max_iter=1).fit(p, np.arange(len(p))%2))


def test_benchmark_forced_epochs_and_report_contribution(tmp_path):
    p, _ = inputs()
    y = np.arange(len(p))%2
    benchmark(p, y, ['a','b','c'], tmp_path, seeds=(0,), n_labeled=12,
              n_unlabeled=12, n_draws=2, max_iter=2, tol=2, contribution=True)
    status = json.loads((tmp_path/'run_status.json').read_text())[0]
    assert status['iterations'] == 2 and status['stop_reason'] == 'fixed_epochs'
    d = json.loads((tmp_path/'report/report_data.json').read_text())
    assert d['contribution']['n_refits'] == 3
    assert (tmp_path/'report/contribution.csv').exists()


def test_histogram_handles_machine_precision_spread():
    a = np.array([.1, np.nextafter(.1, 1.)])
    h = histogram(a)
    assert h['point_mass'] == pytest.approx(.1)
    assert sum(h['counts']) == 2
