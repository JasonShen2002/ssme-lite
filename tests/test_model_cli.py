import json
import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from ssme_lite import PredictionMatrixGenerator, SSMEEstimator
from ssme_lite.cli import load_config, main, run_config


@pytest.fixture
def fitted(tmp_path):
    X, y = make_classification(n_samples=160, n_features=6, random_state=29)
    models = {'lr': LogisticRegression().fit(X[:80], y[:80]),
              'rf': RandomForestClassifier(n_estimators=8, random_state=5).fit(X[:80], y[:80])}
    # Exercise non-numeric real class names, while partial y uses class positions.
    for model in models.values():
        model.classes_ = np.array(['negative', 'positive'])
    observed = y[80:].copy()
    observed[20:] = -1
    np.savez(tmp_path/'evaluation.npz', X=X[80:], y=observed,
             classes=np.array(['negative', 'positive']), sample_ids=np.arange(80, 160))
    for name, model in models.items():
        joblib.dump(model, tmp_path/f'{name}.joblib')
    config = {'source': 'models', 'data': 'evaluation.npz', 'output': 'out',
              'models': [{'name': n, 'path': n+'.joblib'} for n in models],
              'prediction': {'n_jobs': 2, 'batch_size': 13},
              'estimator': {'max_iter': 3, 'random_state': 5},
              'report': {'metrics': ['accuracy'], 'n_draws': 5, 'target': 'unlabeled'}}
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    return models, X[80:], observed, path, config


def test_model_cli_equals_python_from_other_directory(fitted, tmp_path, monkeypatch):
    models, X, y, path, config = fitted
    other = tmp_path/'other'
    other.mkdir()
    monkeypatch.chdir(other)
    out = run_config(path, trust_model_files=True)
    generator = PredictionMatrixGenerator(models)
    p = generator.fit_transform(X)
    expected = SSMEEstimator(**config['estimator']).fit(p, y, model_names=list(models)).report(**config['report'])
    with np.load(out/'predictions.npz') as f:
        np.testing.assert_allclose(f['scores'], p, atol=1e-14)
        np.testing.assert_array_equal(f['y'], y)
    actual = pd.read_csv(out/'metrics.csv')
    np.testing.assert_allclose(actual.estimate, expected.table.estimate)
    assert (out/'report.html').is_file()
    receipt = json.loads((out/'input_provenance.json').read_text())
    assert receipt['classes'] == ['negative', 'positive']
    assert all(len(m['sha256']) == 64 for m in receipt['models'])


def test_preflight_and_untrusted_run_never_deserialize(fitted, monkeypatch):
    *_, path, config = fitted
    def forbidden(*a, **kw):
        pytest.fail('preflight/untrusted run deserialized a model')
    monkeypatch.setattr(joblib, 'load', forbidden)
    load_config(path)
    with pytest.raises(ValueError, match='trust-model-files'):
        run_config(path)
    config['models'][1]['name'] = config['models'][0]['name']
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='unique'):
        load_config(path)


def test_label_axis_mismatch_is_rejected(fitted):
    _, X, y, path, _ = fitted
    np.savez(path.parent/'evaluation.npz', X=X, y=y, classes=['positive', 'negative'])
    with pytest.raises(ValueError, match='classes_ order'):
        run_config(path, trust_model_files=True)


@pytest.mark.parametrize('kind', ['array', 'pandas', 'sparse'])
def test_batched_prediction_preserves_rows(fitted, kind):
    models, X, _, _, _ = fitted
    if kind == 'pandas':
        X = pd.DataFrame(X)
    elif kind == 'sparse':
        X = csr_matrix(X)
    full = PredictionMatrixGenerator(models).fit_transform(X)
    batches = PredictionMatrixGenerator(models, n_jobs=2, batch_size=7).fit_transform(X)
    np.testing.assert_allclose(batches, full, atol=1e-14)


def test_template_refuses_overwrite_and_legacy_cli_works(fitted):
    models, X, y, path, _ = fitted
    template = path.parent/'template.json'
    main(['init-config', str(template)])
    before = template.read_bytes()
    with pytest.raises(SystemExit):
        main(['init-config', str(template)])
    assert template.read_bytes() == before
    np.savez(path.parent/'scores.npz', scores=PredictionMatrixGenerator(models).fit_transform(X), y=y)
    path.write_text(json.dumps({'source': 'npz', 'data': 'scores.npz', 'output': 'legacy',
                                'estimator': {'max_iter': 2},
                                'report': {'metrics': ['accuracy'], 'n_draws': 2}}))
    main([str(path)])
    assert (path.parent/'legacy/report.html').is_file()
