"""Prediction alignment and non-mutating probability validation."""
from collections.abc import Mapping
import numpy as np
from joblib import Parallel, delayed


def probabilities(scores, clip=1e-6):
    """Accept binary (N,M) or full (N,M,K); return normalized (N,M,K)."""
    p = np.array(scores, dtype=float, copy=True)
    if not 0 < clip < 0.5:
        raise ValueError("clip must be between 0 and 0.5")
    if p.ndim == 2:
        p = np.stack([1 - p, p], axis=-1)
    if p.ndim != 3 or min(p.shape) < 1 or p.shape[-1] < 2:
        raise ValueError("scores must have shape (N,M) or (N,M,K), N,M>=1 and K>=2")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must be finite and within [0,1]")
    if not np.allclose(p.sum(-1), 1, atol=1e-5):
        raise ValueError("class probabilities must sum to one")
    p = np.clip(p, clip, 1 - clip)
    return p / p.sum(-1, keepdims=True)


def alr(scores):
    """Last class is the reference; flatten M*(K-1) features."""
    return np.log(scores[..., :-1] / scores[..., -1:]).reshape(len(scores), -1)


class PredictionMatrixGenerator:
    """Wrap already fitted sklearn-style classifiers; never train them.

    Binary output is (N,M), positive class classes_[1]. Multiclass is (N,M,K).
    y_partial uses -1 for unknown and integer positions in classes_ otherwise.
    """
    def __init__(self, models, clip=1e-6, n_jobs=1):
        self.models = models
        self.clip = clip
        self.n_jobs = n_jobs

    def fit(self, X=None, y=None):
        items = list(self.models.items()) if isinstance(self.models, Mapping) else [
            (f"model_{i}", m) for i, m in enumerate(self.models)]
        if not items:
            raise ValueError("at least one fitted model is required")
        self.model_names_ = [str(name) for name, _ in items]
        if len(set(self.model_names_)) != len(items):
            raise ValueError("model names must be unique")
        self.models_ = [m for _, m in items]
        for m in self.models_:
            if not hasattr(m, "classes_") or not callable(getattr(m, "predict_proba", None)):
                raise ValueError("each fitted model needs classes_ and predict_proba")
        self.classes_ = np.asarray(self.models_[0].classes_).copy()
        if len(self.classes_) < 2 or len(set(self.classes_.tolist())) != len(self.classes_):
            raise ValueError("at least two distinct classes are required")
        self.orders_ = []
        for m in self.models_:
            classes = list(m.classes_)
            if len(classes) != len(self.classes_) or set(classes) != set(self.classes_):
                raise ValueError("all models must predict the same class set")
            self.orders_.append([classes.index(c) for c in self.classes_])
        return self

    def transform(self, X, y_partial=None):
        if not hasattr(self, "models_"):
            raise ValueError("call fit or fit_transform first")
        blocks = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(m.predict_proba)(X) for m in self.models_)
        if any(np.asarray(b).shape != (len(X), len(self.classes_)) for b in blocks):
            raise ValueError("each predict_proba result must have shape (len(X), n_classes)")
        p = probabilities(np.stack([np.asarray(b)[:, o] for b, o in zip(blocks, self.orders_)], axis=1), self.clip)
        self.sample_ids_ = np.asarray(X.index).copy() if hasattr(X, "index") and not callable(X.index) else np.arange(len(p))
        self.labeled_indices_ = self.unlabeled_indices_ = None
        if y_partial is not None:
            y = np.asarray(y_partial)
            if y.shape != (len(p),):
                raise ValueError("y_partial length must match X")
            self.labeled_indices_ = np.flatnonzero(y != -1)
            self.unlabeled_indices_ = np.flatnonzero(y == -1)
        return p[..., 1] if p.shape[-1] == 2 else p

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X, y)
