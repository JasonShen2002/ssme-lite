"""ALR + weighted Gaussian KDE + clamped semi-supervised EM."""
import warnings
import numpy as np
from scipy.special import softmax, logsumexp
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed
from .prediction import probabilities, alr


class SSMEEstimator(BaseEstimator):
    """Fit on precomputed scores. Unknown labels are -1; known labels are 0..K-1.

    Fixed-bandwidth KDE updates are a nonparametric EM-style procedure, not a
    guarantee of monotonic maximization of a finite-dimensional likelihood.
    """
    def __init__(self, density="kde", bandwidth="scott", max_iter=100,
                 tol=1e-3, labeled_weight=10.0, clip=1e-6,
                 random_state=0, n_jobs=1, batch_size=2048, early_stopping=False):
        self.density = density
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.labeled_weight = labeled_weight
        self.clip = clip
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.early_stopping = early_stopping

    def fit(self, scores, y, scores_unlabeled=None, model_names=None,
            initial_responsibilities=None):
        p = probabilities(scores, self.clip)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(p),) or not np.isfinite(y).all() or not np.equal(y, np.floor(y)).all():
            raise ValueError("y must contain integer class indices or -1, one per row")
        if scores_unlabeled is not None:
            u = probabilities(scores_unlabeled, self.clip)
            if u.shape[1:] != p.shape[1:]:
                raise ValueError("labeled and unlabeled scores must have matching model/class dimensions")
            p = np.concatenate([p, u])
            y = np.concatenate([y, np.full(len(u), -1)])
        y = y.astype(int)
        k = p.shape[-1]
        if ((y < -1) | (y >= k)).any():
            raise ValueError("labels must be -1 or class indices 0..K-1")
        if not isinstance(self.max_iter, (int, np.integer)) or isinstance(self.max_iter, bool):
            raise ValueError("max_iter must be a positive integer number of EM epochs")
        if not isinstance(self.early_stopping, (bool, np.bool_)):
            raise ValueError("early_stopping must be boolean")
        if self.density != "kde" or self.max_iter < 1 or self.tol <= 0 or self.labeled_weight <= 0 or self.batch_size < 1:
            raise ValueError("use kde and positive max_iter, tol, labeled_weight, batch_size")
        labeled = y >= 0
        if set(y[labeled]) != set(range(k)):
            raise ValueError("at least one labeled example per class is required")
        names = list(model_names) if model_names is not None else [f"model_{j}" for j in range(p.shape[1])]
        if len(names) != p.shape[1] or len(set(names)) != len(names):
            raise ValueError("model_names must be unique and match the model axis")
        self.scores_, self.y_, self.model_names_ = p, y, names
        self.classes_ = np.arange(k)
        self.labeled_indices_, self.unlabeled_indices_ = np.flatnonzero(labeled), np.flatnonzero(~labeled)
        z = alr(p)
        self.n_features_in_ = z.shape[1]
        # Scott scaling in raw ALR space; no hidden label-based bandwidth tuning.
        if self.bandwidth == "scott":
            self.bandwidth_ = max(float(np.std(z, axis=0).mean() * len(z) ** (-1 / (z.shape[1] + 4))), 1e-3)
        elif self.bandwidth == "official":
            # Match the public repository, not the paper's claimed ISJ rule.
            # Explicit normal_reference is KDEUnivariate.fit's upstream default.
            try:
                from statsmodels.nonparametric.kde import KDEUnivariate
            except ImportError as exc:
                raise ImportError("bandwidth='official' requires pip install 'ssme-lite[official]'") from exc
            widths = []
            for column in z.T:
                kde = KDEUnivariate(column)
                kde.fit(bw="normal_reference")
                widths.append(kde.bw / 4.0)
            self.bandwidth_ = float(min(widths))
            if not np.isfinite(self.bandwidth_) or self.bandwidth_ <= 0:
                raise ValueError("official bandwidth must be finite and positive")
        else:
            self.bandwidth_ = float(self.bandwidth)
            if not np.isfinite(self.bandwidth_) or self.bandwidth_ <= 0:
                raise ValueError("bandwidth must be 'scott', 'official' or a positive finite number")
        rng = np.random.default_rng(self.random_state)
        mean_p = p.mean(1)
        assignments = (rng.random((len(p), 1)) > np.cumsum(mean_p, axis=1)).sum(1)
        gamma = np.eye(k)[np.minimum(assignments, k - 1)]
        if initial_responsibilities is not None:
            gamma = np.asarray(initial_responsibilities, dtype=float).copy()
            if (gamma.shape != (len(p), k) or not np.isfinite(gamma).all()
                    or (gamma < 0).any() or not np.allclose(gamma.sum(1), 1, rtol=0, atol=1e-10)):
                raise ValueError("initial_responsibilities must be a normalized nonnegative (N, K) matrix")
        gamma[labeled] = np.eye(k)[y[labeled]]
        self.initial_responsibilities_ = gamma.copy()
        weights = np.where(labeled, self.labeled_weight, 1.0)
        self.history_, self.converged_ = [], False
        for iteration in range(self.max_iter):
            self._m_step(z, gamma, weights)
            updated, log_joint = self._posterior(z, return_log_joint=True)
            updated[labeled] = np.eye(k)[y[labeled]]
            delta = float(np.max(np.abs(updated - gamma)))
            # Observed-data diagnostic at this M-step: labeled log joint plus
            # unlabeled log marginal, with the same annotation weights as fit.
            log_observed = logsumexp(log_joint, axis=1)
            log_observed[labeled] = log_joint[labeled, y[labeled]]
            self.history_.append({
                "iteration": iteration + 1, "max_responsibility_change": delta,
                "objective": float(np.dot(weights, log_observed) / weights.sum()),
                "class_priors": self.priors_.tolist(),
                "mean_posterior_confidence": float(updated.max(axis=1).mean()),
                "responsibility_mass": updated.mean(axis=0).tolist(),
            })
            gamma = updated
            self.converged_ = delta < self.tol
            if self.early_stopping and self.converged_:
                break
        self.n_iter_ = iteration + 1
        self.posterior_ = gamma
        self.stop_reason_ = "tolerance" if self.early_stopping and self.converged_ else "max_iter" if self.early_stopping else "fixed_epochs"
        if self.early_stopping and not self.converged_:
            warnings.warn("SSME reached max_iter; inspect history_ and bandwidth sensitivity", RuntimeWarning)
        return self

    def _m_step(self, z, gamma, weights):
        effective = gamma * weights[:, None]
        counts = effective.sum(0)
        self.priors_ = counts / counts.sum()
        def fit_one(w):
            mask = w > 0
            return KernelDensity(bandwidth=self.bandwidth_, kernel="gaussian").fit(z[mask], sample_weight=w[mask])
        self.kdes_ = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(fit_one)(effective[:, c]) for c in range(gamma.shape[1]))

    def _posterior(self, z, return_log_joint=False):
        chunks, joints = [], []
        for start in range(0, len(z), self.batch_size):
            block = z[start:start + self.batch_size]
            log_density = np.column_stack([kde.score_samples(block) for kde in self.kdes_])
            joint = log_density + np.log(self.priors_)
            chunks.append(softmax(joint, axis=1))
            if return_log_joint:
                joints.append(joint)
        posterior = np.concatenate(chunks)
        return (posterior, np.concatenate(joints)) if return_log_joint else posterior

    def predict_proba(self, scores):
        check_is_fitted(self, "kdes_")
        p = probabilities(scores, self.clip)
        if p.shape[1:] != self.scores_.shape[1:]:
            raise ValueError("model and class dimensions must match training scores")
        return self._posterior(alr(p))

    def evaluate(self, metrics=("accuracy", "auc", "auprc", "ece"), n_draws=100,
                 confidence=0.95, target="all", random_state=None,
                 title=None, sample_ids=None, primary_metric=None, contribution=False):
        from .reporting.report import make_report
        check_is_fitted(self, "posterior_")
        return make_report(self, metrics, n_draws, confidence, target, random_state,
                           title=title, sample_ids=sample_ids, primary_metric=primary_metric,
                           contribution=contribution)

    def report(self, **kwargs):
        return self.evaluate(**kwargs)

    def contribution(self, n_draws=30):
        """Controlled leave-one-model-out sensitivity; n_draws kept for compatibility.

        Uses exact expected Accuracy, so no label sampling is needed. Full-fit
        initialization, scalar bandwidth and epoch count are held fixed.
        """
        import pandas as pd
        from .contribution import model_contribution
        check_is_fitted(self, "posterior_")
        return pd.DataFrame(model_contribution(self)["rows"])
