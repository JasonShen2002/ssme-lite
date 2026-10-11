"""One isolated condition at a time; hidden labels used only by external scoring."""
import time
import numpy as np
from scipy.special import softmax
from ssme_lite import SSMEEstimator, PredictionMatrixGenerator
from ssme_lite.prediction import probabilities
from .common import (ROOT, frozen, nested_split, partial_labels, mean_posterior,
                     posterior_metrics, observed_metrics, selection, array_hash)


def evaluate_condition(p, y, names, job, candidate_count=None):
    pool, heldout = nested_split(len(y), job['pool_size'], job['seed'])
    pp, yy = p[pool], y[pool]
    partial = partial_labels(yy, job['budget'])
    count = len(names) if candidate_count is None else candidate_count
    metrics = ('accuracy', 'auc') if job['study'] == 'stress' else ('accuracy',)
    truth = {target: observed_metrics(p[idx, :count], y[idx], metrics)
             for target, idx in [('heldout', heldout), ('fixed_pool', pool)]}
    means, valid = posterior_metrics(pp[:, :count], mean_posterior(pp, partial), metrics,
                                    draws=job['draws'], seed=job['seed'])
    estimates = {'labeled_only': observed_metrics(pp[:job['budget'], :count], yy[:job['budget']], metrics),
                 'mean_probability': means}
    statuses = {'labeled_only': 'ok', 'mean_probability': 'ok'}
    diagnostics = dict(input_sha256=array_hash(p), labeled_indices=pool[:job['budget']].tolist(),
                       pool_indices=pool.tolist(), heldout_indices=heldout.tolist(),
                       n_models=p.shape[1], n_classes=p.shape[2], alr_dimension=p.shape[1]*(p.shape[2]-1))
    if len(np.unique(partial[partial >= 0])) != p.shape[2]:
        statuses['ssme'] = 'missing_labeled_class'
    else:
        start = time.perf_counter()
        est = SSMEEstimator(max_iter=job['iterations'], random_state=job['seed']).fit(pp, partial, model_names=names)
        fit_seconds = time.perf_counter()-start
        estimates['ssme'], counts = posterior_metrics(pp[:, :count], est.posterior_, metrics,
                                                      draws=job['draws'], seed=job['seed'])
        statuses['ssme'] = 'ok'
        q = est.posterior_[partial < 0]
        diagnostics.update(fit_seconds=fit_seconds, iterations=est.n_iter_, bandwidth=est.bandwidth_,
                           final_delta=est.history_[-1]['max_responsibility_change'],
                           mean_max_probability=float(q.max(1).mean()),
                           mean_entropy=float(-(q*np.log(np.maximum(q, 1e-300))).sum(1).mean()),
                           valid_draws={m: a.tolist() for m, a in counts.items()})
    rows, details = [], []
    for method, values in estimates.items():
        for target, target_values in truth.items():
            for metric, a in values.items():
                b = target_values[metric]
                rows.append(dict(method=method, target=target, metric=metric, **selection(a, b)))
                for j in range(count):
                    details.append(dict(method=method, target=target, metric=metric, model=names[j],
                                        estimate=float(a[j]) if np.isfinite(a[j]) else None,
                                        truth=float(b[j]) if np.isfinite(b[j]) else None))
    diagnostics['mean_probability_valid_draws'] = {m: a.tolist() for m, a in valid.items()}
    return dict(status='ok', method_status=statuses, diagnostics=diagnostics, rows=rows, details=details)


def labels(job):
    p, y, names = frozen(job['pipeline'])
    return evaluate_condition(probabilities(p), y, names, job)


def stress(job):
    with np.load(ROOT / 'experiments/extensions/inputs' / f"digits{job['classes']}.npz", allow_pickle=False) as f:
        p, y, names = f['scores'], f['y'], f['model_names'].tolist()
    if job['regime'] == 'three_diverse':
        p, names = p[:, :3], names[:3]
    elif job['regime'] == 'three_duplicated':
        p = np.concatenate([p[:, :3], p[:, :3]], axis=1)
        names = names[:3] + [n+'_duplicate' for n in names[:3]]
    return evaluate_condition(probabilities(p), y, names, job, candidate_count=3)


def performance(job):
    if job['kind'] == 'prediction':
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        X, y = make_classification(n_samples=job['n']+500, n_features=20, random_state=731)
        models = {'logistic': LogisticRegression(max_iter=500).fit(X[:500], y[:500]),
                  'forest': RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=731).fit(X[:500], y[:500]),
                  'naive_bayes': GaussianNB().fit(X[:500], y[:500])}
        start = time.perf_counter()
        p = PredictionMatrixGenerator(models, n_jobs=job['workers'], batch_size=job['batch']).fit_transform(X[500:])
        return dict(status='ok', prediction_seconds=time.perf_counter()-start,
                    input_sha256=array_hash(X[500:]), posterior_hash=array_hash(p),
                    comparison_values=p.tolist())
    if job['kind'] == 'real':
        p, y, names = frozen('checkpoints')
        p = probabilities(p)
    else:
        rng = np.random.default_rng(731)
        y = np.arange(job['n']) % job['classes']
        logits = rng.normal(0, 1.2, (job['n'], job['models'], job['classes']))
        logits += np.eye(job['classes'])[y, None, :]*.8
        p = probabilities(softmax(logits, axis=-1))
        names = [f'view{j}' for j in range(job['models'])]
    partial = partial_labels(y, max(20, job['classes']*2))
    start = time.perf_counter()
    est = SSMEEstimator(max_iter=job['iterations'], n_jobs=job['workers'],
                        batch_size=job['batch'], random_state=731).fit(p, partial, model_names=names)
    fit_time = time.perf_counter()-start
    start = time.perf_counter()
    posterior_metrics(p, est.posterior_, ('accuracy', 'auc'), draws=10, seed=731)
    metric_time = time.perf_counter()-start
    start = time.perf_counter()
    inferred = est.predict_proba(p)
    inference_time = time.perf_counter()-start
    import tempfile
    from pathlib import Path
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='ssme-report-') as directory:
        est.report(metrics=('accuracy', 'auc'), n_draws=10).save(directory)
        report_bytes = sum(f.stat().st_size for f in Path(directory).iterdir() if f.is_file())
    report_time = time.perf_counter()-start
    return dict(status='ok', input_sha256=array_hash(p), fit_seconds=fit_time,
                metric_seconds=metric_time, inference_seconds=inference_time,
                report_seconds=report_time, report_bytes=report_bytes,
                iterations=est.n_iter_, bandwidth=est.bandwidth_,
                final_delta=est.history_[-1]['max_responsibility_change'],
                posterior_hash=array_hash(est.posterior_), inference_hash=array_hash(inferred),
                comparison_values=est.posterior_.tolist())
