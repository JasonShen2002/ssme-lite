"""Statistical contracts for the real-world report, including degenerate cases."""
import json
import numpy as np
import pytest
from scipy.special import softmax, logsumexp
from ssme_lite import SSMEEstimator
from ssme_lite.prediction import alr
from ssme_lite.reporting.data import sampling_statistics


def test_fractional_ties_and_pairwise_denominators():
    draws = np.array([[1., 1., 0.], [0., 1., 1.], [np.nan, 0., 1.]])
    result = sampling_statistics(draws)
    np.testing.assert_allclose(result['rank_frequencies'].sum(axis=1), 1)
    np.testing.assert_allclose(result['rank_frequencies'].sum(axis=0), 1)
    np.testing.assert_allclose(result['rank1'], [.25, .5, .25])
    assert result['rank_valid_draws'] == 2
    pair = result['pairs']['0:1']
    assert pair['n'] == 2
    assert pair['a_better'] == 0 and pair['tie'] == .5 and pair['b_better'] == .5
    assert result['pairs']['1:2']['n'] == 3
    inverted = sampling_statistics(draws, lower_is_better=True)
    assert inverted['pairs']['0:1']['a_better'] == .5


def test_no_valid_rankings_do_not_become_zero_certainty():
    result = sampling_statistics(np.full((3, 2), np.nan))
    assert result['rank_valid_draws'] == 0
    assert np.isnan(result['rank1']).all()
    assert result['median_rank'] == [None, None]


@pytest.mark.parametrize('k', [2, 3])
def test_report_scopes_and_observed_objective(k):
    rng = np.random.default_rng(9)
    p = softmax(rng.normal(size=(60, 3, k)), axis=-1)
    y = np.full(60, -1)
    y[:k*2] = np.arange(k*2) % k
    est = SSMEEstimator(max_iter=1, tol=2).fit(p, y)
    _, joint = est._posterior(alr(p), return_log_joint=True)
    observed = logsumexp(joint, axis=1)
    known = y >= 0
    observed[known] = joint[known, y[known]]
    weights = np.where(known, est.labeled_weight, 1)
    assert est.history_[0]['objective'] == pytest.approx(np.average(observed, weights=weights))
    report = est.report(n_draws=8, target='unlabeled', sample_ids=[f'id-{i}' for i in range(60)])
    d = report.data
    assert d['metadata']['n_samples'] == 60-k*2
    assert d['metadata']['n_labeled'] == 0
    assert d['fit_pool']['n_labeled'] == k*2
    assert d['posterior']['n_unlabeled'] == 60-k*2
    assert sum(d['posterior']['confidence_histogram']['counts']) == 60-k*2
    assert len(d['prediction_space']['points']) == 60
    assert all(int(x['id'].split('-')[1]) >= k*2 for x in d['posterior']['ambiguous'])
    assert len(d['metrics']['auc']['samples']) == 3
    json.dumps(d, allow_nan=False)


def test_degenerate_single_model_report_and_html_escaping(tmp_path):
    p = np.full((12, 1, 2), .5)
    y = np.arange(12) % 2
    name = '</script><script>alert(1)</script>'
    report = SSMEEstimator().fit(p, y, model_names=[name]).report(n_draws=3)
    d = report.data
    assert d['landscape']['correlation'] == [[None]]
    assert d['metrics']['accuracy']['rank1'] == [1.]
    assert d['metrics']['accuracy']['distributions'][0]['density'] == []
    assert d['posterior']['ambiguous'] == []
    assert d['prediction_space']['variance'] == [0., 0., 0.]
    path = report.save(tmp_path)
    html = path.read_text()
    assert name not in html
    assert '\\u003c/script>' in html
    assert '<script src=' not in html
    assert json.loads((tmp_path/'report_data.json').read_text())['models'] == [name]


def test_invalid_primary_metric():
    p = np.full((12, 1, 2), .5)
    with pytest.raises(ValueError, match='primary_metric'):
        SSMEEstimator().fit(p, np.arange(12)%2).report(primary_metric='precision')


def test_public_report_contribution_survives_module_reorganization(tmp_path):
    rng = np.random.default_rng(3)
    scores = rng.uniform(.1, .9, (20, 2))
    labels = np.full(20, -1)
    labels[:4] = [0, 1, 0, 1]
    estimator = SSMEEstimator(max_iter=2).fit(scores, labels)
    report = estimator.report(n_draws=4, contribution=True)
    assert len(report.data["contribution"]["rows"]) == 2
    assert report.save(tmp_path).is_file()
    assert (tmp_path / "contribution.csv").is_file()
