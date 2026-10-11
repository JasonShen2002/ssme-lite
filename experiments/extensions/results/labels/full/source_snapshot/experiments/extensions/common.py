"""Shared audit/statistics helpers for the predeclared extension experiments."""
import hashlib
import importlib.metadata
import json
import platform
import os
from pathlib import Path
import numpy as np
from scipy.stats import t
from ssme_lite.metrics import metric_values

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def environment():
    return dict(python=platform.python_version(), platform=platform.platform(),
                machine=platform.machine(), logical_cpus=os.cpu_count(), versions={name: importlib.metadata.version(name)
                for name in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'joblib']})


def frozen(name):
    path = ROOT / 'reproduction/inputs' / (name + '.npz')
    manifest = json.loads((path.parent / 'manifest.json').read_text())
    expected = next(f['sha256'] for f in manifest['files'] if f['file'] == path.name)
    if digest(path) != expected:
        raise ValueError('frozen input hash mismatch: ' + name)
    with np.load(path, allow_pickle=False) as f:
        return f['scores'], f['y'], f['model_names'].tolist()


def nested_split(n, pool_size, seed):
    if not 0 < pool_size < n:
        raise ValueError('reserve a nonempty disjoint held-out set')
    order = np.random.default_rng(seed).permutation(n)
    # Prefix order is the uniformly random, nested annotation order.
    return order[:pool_size], order[pool_size:]


def partial_labels(y, budget):
    if not 0 < budget <= len(y):
        raise ValueError('invalid label budget')
    result = np.full(len(y), -1, dtype=int)
    result[:budget] = y[:budget]
    return result


def mean_posterior(p, partial):
    q = p.mean(axis=1)
    known = partial >= 0
    q[known] = np.eye(p.shape[-1])[partial[known]]
    return q


def posterior_metrics(p, q, metrics=('accuracy',), draws=30, seed=0):
    """Score the fixed pool without constructing report plots/PCA."""
    result, counts = {}, {}
    if 'accuracy' in metrics:
        result['accuracy'] = np.array([q[np.arange(len(p)), p[:, j].argmax(1)].mean()
                                       for j in range(p.shape[1])])
        counts['accuracy'] = np.ones(p.shape[1], dtype=int)
    others = [m for m in metrics if m != 'accuracy']
    if others:
        rng = np.random.default_rng(seed)
        samples = {m: [] for m in others}
        for _ in range(draws):
            y = np.minimum((rng.random((len(q), 1)) > q.cumsum(1)).sum(1), q.shape[1]-1)
            per_model = [metric_values(y, p[:, j], others) for j in range(p.shape[1])]
            for m in others:
                samples[m].append([v[m] for v in per_model])
        for m in others:
            values = np.asarray(samples[m])
            valid = np.isfinite(values)
            counts[m] = valid.sum(0)
            result[m] = np.divide(np.where(valid, values, 0).sum(0), counts[m],
                                  out=np.full(p.shape[1], np.nan), where=counts[m] > 0)
    return result, counts


def observed_metrics(p, y, metrics=('accuracy',)):
    rows = [metric_values(y, p[:, j], metrics) for j in range(p.shape[1])]
    return {m: np.array([r[m] for r in rows]) for m in metrics}


def selection(estimate, truth):
    # Do not silently drop undefined candidates and change the selection task.
    if not np.isfinite(estimate).all() or not np.isfinite(truth).all():
        return dict(mae=None, selection_regret=None, top1=None)
    selected = np.flatnonzero(np.isclose(estimate, estimate.max(), rtol=0, atol=1e-12))
    best = truth.max()
    return dict(mae=float(np.abs(estimate-truth).mean()),
                selection_regret=float(np.mean(best-truth[selected])),
                top1=float(np.isclose(truth[selected], best, atol=1e-12, rtol=0).mean()))


def paired_statistics(differences, family=6):
    a = np.asarray(differences, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return dict(n=len(a), mean_difference=None, ci_low=None, ci_high=None,
                    adjusted_upper=None, supports_lower_error=False)
    mean, se = float(a.mean()), float(a.std(ddof=1)/np.sqrt(len(a)))
    half = float(t.ppf(.975, len(a)-1)*se)
    upper = float(mean+t.ppf(1-.05/family, len(a)-1)*se)
    return dict(n=len(a), mean_difference=mean, ci_low=mean-half, ci_high=mean+half,
                adjusted_upper=upper, supports_lower_error=upper < 0)
