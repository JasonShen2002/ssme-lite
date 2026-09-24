import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, softmax
from sklearn.exceptions import NotFittedError
from ssme_lite import PredictionMatrixGenerator, SSMEEstimator
from ssme_lite.prediction import probabilities, alr
from ssme_lite.metrics import metric_values


def sample(k=2, n=100):
    rng = np.random.default_rng(21)
    y = np.arange(n) % k
    p = softmax(rng.normal(0, 0.3, (n, 3, k)) + np.eye(k)[y, None, :] * 2, axis=-1)
    partial = y.copy()
    partial[k * 3:] = -1
    return p, partial


def test_alr_and_no_mutation():
    p = np.array([[0., 1.], [0.25, 0.75]])
    copy = p.copy()
    z = alr(probabilities(p))
    np.testing.assert_array_equal(p, copy)
    assert np.isfinite(z).all()
    np.testing.assert_allclose(expit(-z), probabilities(p)[..., 1])


def test_default_tolerance():
    assert SSMEEstimator().tol == 1e-3
    assert SSMEEstimator().labeled_weight == 10.0
    assert SSMEEstimator().max_iter == 100
    assert SSMEEstimator().early_stopping is False


def test_official_bandwidth_matches_reference():
    pytest.importorskip("statsmodels")
    from statsmodels.nonparametric.kde import KDEUnivariate
    p, y = sample()
    widths = []
    for column in alr(p).T:
        kde = KDEUnivariate(column)
        kde.fit()
        widths.append(kde.bw / 4)
    est = SSMEEstimator(bandwidth="official", max_iter=2, tol=1.1).fit(p, y)
    assert est.bandwidth_ == pytest.approx(min(widths), rel=1e-12)
    assert est.get_params()["bandwidth"] == "official"
    meta = est.report(n_draws=2).metadata
    assert meta["labeled_weight"] == 10
    assert meta["bandwidth_rule"] == "official"


@pytest.mark.parametrize("k", [2, 3])
def test_em_clamping_and_predict(k):
    p, y = sample(k)
    est = SSMEEstimator(max_iter=50).fit(p, y)
    np.testing.assert_allclose(est.posterior_.sum(1), 1)
    np.testing.assert_allclose(est.posterior_[y >= 0], np.eye(k)[y[y >= 0]])
    np.testing.assert_allclose(est.priors_.sum(), 1)
    np.testing.assert_allclose(est.predict_proba(p)[y < 0], est.posterior_[y < 0])
    assert est.predict_proba(p).shape == (len(p), k)
    again = SSMEEstimator(max_iter=50).fit(p, y)
    np.testing.assert_allclose(est.posterior_, again.posterior_)


def test_weighted_m_step():
    est = SSMEEstimator(bandwidth=0.4)
    est.bandwidth_ = 0.4
    z = np.array([[0.], [1.], [2.]])
    gamma = np.array([[1., 0.], [0., 1.], [.25, .75]])
    weights = np.array([2., 2., 1.])
    est._m_step(z, gamma, weights)
    np.testing.assert_allclose(est.priors_, [2.25/5, 2.75/5])
    # Direct Gaussian mixture density matches the weighted KDE.
    expected = np.sum(np.exp(-z[:, 0]**2/(2*0.4**2)) / (0.4*np.sqrt(2*np.pi)) * gamma[:, 0]*weights)/2.25
    np.testing.assert_allclose(np.exp(est.kdes_[0].score_samples([[0.]]))[0], expected)


def test_all_labeled_exact_metrics():
    p, _ = sample()
    y = np.arange(len(p)) % 2
    est = SSMEEstimator().fit(p, y)
    report = est.report(n_draws=5)
    for j, name in enumerate(est.model_names_):
        expected = metric_values(y, p[:, j])
        for row in report.table[report.table.model == name].itertuples():
            assert row.estimate == pytest.approx(expected[row.metric])
            assert row.interval_low == pytest.approx(row.interval_high)
    with pytest.raises(ValueError, match="empty"):
        est.report(target="unlabeled")


def test_separate_fit_and_unknown_labels():
    p, y = sample()
    est = SSMEEstimator().fit(p[y >= 0], y[y >= 0], p[y < 0])
    assert len(est.y_) == len(y)
    with pytest.raises(ValueError, match="per class"):
        SSMEEstimator().fit(p, np.full(len(p), -1))
    with pytest.raises(ValueError):
        SSMEEstimator().fit(p, np.full(len(p), .5))
    with pytest.raises(NotFittedError):
        SSMEEstimator().predict_proba(p)


@pytest.mark.parametrize("bad", [np.array([[np.nan]]), np.array([[1.1]]), np.zeros((2, 2, 3))])
def test_bad_probabilities(bad):
    with pytest.raises(ValueError):
        probabilities(bad)


def test_class_alignment():
    class Model:
        def __init__(self, reverse=False):
            self.classes_ = np.array(["yes", "no"] if reverse else ["no", "yes"])
            self.reverse = reverse
        def predict_proba(self, X):
            p = np.tile([.2, .8], (len(X), 1))
            return p[:, ::-1] if self.reverse else p
    X = pd.DataFrame({"x": [1, 2]}, index=[42, 99])
    generator = PredictionMatrixGenerator({"a": Model(), "b": Model(True)})
    p = generator.fit_transform(X, [-1, 1])
    np.testing.assert_allclose(p, .8)
    np.testing.assert_array_equal(generator.sample_ids_, [42, 99])
    np.testing.assert_array_equal(generator.unlabeled_indices_, [0])


def test_metric_edges():
    scores = np.array([[.8, .2], [.2, .8]])
    values = metric_values([0, 1], scores)
    assert values["accuracy"] == values["auc"] == values["auprc"] == 1
    assert values["ece"] == pytest.approx(.2)
    assert np.isnan(metric_values([0, 0], scores)["auc"])


def test_report_export_and_ranking(tmp_path):
    p, y = sample()
    report = SSMEEstimator().fit(p, y).report(n_draws=5)
    assert report.ranking("ece").estimate.is_monotonic_increasing
    path = report.save(tmp_path)
    assert path.exists() and (tmp_path / "accuracy.png").exists()
    assert "conditional" in path.read_text()


def test_multiclass_metrics_and_batching():
    p, y = sample(3)
    est = SSMEEstimator(batch_size=7).fit(p, y)
    q = est.predict_proba(p)
    est.batch_size = 1000
    np.testing.assert_allclose(q, est.predict_proba(p))
    report = est.report(n_draws=5)
    assert np.isfinite(report.table.estimate).all()
    assert ((report.table.estimate >= 0) & (report.table.estimate <= 1)).all()
    assert est.report(target="unlabeled", n_draws=5).metadata["n_labeled"] == 0


def test_cli_custom_npz(tmp_path, monkeypatch):
    import json
    from ssme_lite.cli import main
    p, y = sample()
    np.savez(tmp_path / "input.npz", scores=p, y=y, model_names=np.array(["a", "b", "c"]))
    config = tmp_path / "run.json"
    config.write_text(json.dumps({"source": "npz", "data": "input.npz", "output": "report", "report": {"n_draws": 3}}))
    monkeypatch.setattr("sys.argv", ["ssme-lite", str(config)])
    main()
    assert (tmp_path / "report/report.html").exists()
    assert (tmp_path / "report/input_config.json").exists()


def test_selection_ties():
    from ssme_lite.benchmark import selection_stats
    a = pd.DataFrame({"model": ["a", "b"], "metric": ["accuracy"]*2, "value": [.5, .5]})
    b = a.copy()
    b["value"] = [.6, .8]
    result = selection_stats(a, b, "accuracy")
    assert result["top1_correct"] == .5
    assert result["selection_regret"] == pytest.approx(.1)
