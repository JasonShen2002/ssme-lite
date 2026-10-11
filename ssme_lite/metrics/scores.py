"""Consistent metrics shared by estimates, baselines and ground truth."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

METRICS = ("accuracy", "auc", "auprc", "ece")


def metric_values(y, p, metrics=METRICS, n_bins=10):
    y = np.asarray(y, dtype=int)
    if any(m not in METRICS for m in metrics):
        raise ValueError(f"metrics must be drawn from {METRICS}")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if not len(y):
        raise ValueError("cannot evaluate an empty target")
    k = p.shape[-1]
    result = {}
    for metric in metrics:
        if metric == "accuracy":
            result[metric] = float(np.mean(p.argmax(1) == y))
        elif metric == "ece":
            # Binary positive-class calibration; multiclass top-label calibration.
            confidence = p[:, 1] if k == 2 else p.max(1)
            outcome = y if k == 2 else (p.argmax(1) == y)
            bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
            result[metric] = sum(np.mean(bins == b) * abs(
                outcome[bins == b].mean() - confidence[bins == b].mean())
                for b in range(n_bins) if np.any(bins == b))
        elif len(np.unique(y)) != k:
            result[metric] = np.nan
        elif metric == "auc":
            result[metric] = roc_auc_score(y, p[:, 1]) if k == 2 else roc_auc_score(
                y, p, multi_class="ovr", average="macro", labels=np.arange(k))
        else:
            result[metric] = average_precision_score(y, p[:, 1]) if k == 2 else average_precision_score(
                np.eye(k)[y], p, average="macro")
    return result


def point_metrics(y, p, names, metrics=METRICS):
    return pd.DataFrame([{"model": name, "metric": metric, "value": value}
                         for j, name in enumerate(names)
                         for metric, value in metric_values(y, p[:, j], metrics).items()])

