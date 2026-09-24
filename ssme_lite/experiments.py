"""Reproducible comparisons, with labels hidden before fitting the evaluator."""
from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
from scipy.special import expit
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from joblib import Parallel, delayed
from .prediction import PredictionMatrixGenerator, probabilities
from .estimator import SSMEEstimator
from .metrics import point_metrics, METRICS
from .splits import SemiSupervisedSplit


def ten_models(seed=0):
    """Ten distinct classifier families; classifier fitting is example-only."""
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=1),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=80, random_state=seed),
        "adaboost": AdaBoostClassifier(n_estimators=60, random_state=seed),
        "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=seed),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=21)),
        "gaussian_nb": GaussianNB(),
        "lda": LinearDiscriminantAnalysis(),
        "qda": QuadraticDiscriminantAnalysis(reg_param=0.1),
    }


def synthetic_scores(seed=1729):
    X, y = make_classification(n_samples=7000, n_features=16, n_informative=8,
                               n_redundant=3, class_sep=1.3, flip_y=0.05, random_state=seed)
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, train_size=2000, random_state=seed, stratify=y)
    models = ten_models(seed)
    for model in models.values():
        model.fit(X_train, y_train)
    generator = PredictionMatrixGenerator(models)
    scores = generator.fit_transform(X_eval)
    return probabilities(scores), y_eval, generator.model_names_


def gaussian_scores(seed=1729):
    """Controlled binary task: independent Gaussian score views given the label.

    Predeclared signal strengths, no selection based on estimator performance.
    This diagnostic is distinct from the ten trained sklearn-family benchmark.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 5000)
    strength = np.linspace(0.5, 2.0, 10)
    X = rng.normal(size=(len(y), len(strength))) + (2*y[:, None]-1)*strength
    class ScoreModel:
        classes_ = np.array([0, 1])
        def __init__(self, column):
            self.column = column
        def predict_proba(self, features):
            positive = expit(features[:, self.column])
            return np.column_stack([1-positive, positive])
    models = {f"gaussian_view_{j}": ScoreModel(j) for j in range(len(strength))}
    generator = PredictionMatrixGenerator(models)
    return probabilities(generator.fit_transform(X)), y, generator.model_names_


def official_scores(root, dataset="CivilComments"):
    root = Path(root) / dataset
    if dataset == "CivilComments":
        names = ["alg_CORAL", "alg_ERM", "alg_IRM", "alg_ERM_seed1", "alg_ERM_seed2", "alg_IRM_seed1", "alg_IRM_seed2"]
    elif dataset == "MultiNLI":
        names = ["alg_ReWeight", "alg_SqrtReWeight", "alg_IRM", "alg_ReSample"]
    else:
        raise ValueError("official dataset must be CivilComments or MultiNLI")
    scores, labels = [], None
    for name in names:
        y = np.load(root / name / "labels.npy", allow_pickle=False).reshape(-1)
        if labels is not None and not np.array_equal(y, labels):
            raise ValueError("official model labels are not row-aligned")
        labels = y
        scores.append(np.load(root / name / "preds.npy", allow_pickle=False))
    return probabilities(np.stack(scores, axis=1)), labels, names


def selection_stats(estimates, truth, metric):
    merged = estimates.merge(truth, on=["model", "metric"], suffixes=("_estimate", "_truth")).dropna()
    if merged.empty:
        return dict(mae=np.nan, spearman=np.nan, top1_correct=np.nan, selection_regret=np.nan)
    a, b = merged.value_estimate.to_numpy(), merged.value_truth.to_numpy()
    order = 1 if metric == "ece" else -1
    selected = np.flatnonzero(np.isclose(a, a.min() if order == 1 else a.max(), atol=1e-12, rtol=0))
    best = b.min() if order == 1 else b.max()
    # Ties are treated as uniform random selection among tied candidates.
    return dict(mae=float(abs(a-b).mean()),
                spearman=float(spearmanr(a, b).statistic) if np.ptp(a) and np.ptp(b) else np.nan,
                top1_correct=float(np.isclose(b[selected], best, atol=1e-12, rtol=0).mean()),
                selection_regret=float(np.mean(abs(b[selected]-best))))


def _run_one_seed(p, y, names, seed, n_labeled, n_unlabeled, n_draws, max_iter, bandwidth, dataset, contribution=False, tol=1e-3, labeled_weight=10.0, early_stopping=False):
    """Fit and score one split; pure in (p, y, seed, hyperparameters), safe to run in any worker."""
    # Uniform annotation sampling; never use hidden labels to repair the split.
    splitter = SemiSupervisedSplit(n_labeled, n_unlabeled, seed)
    parts = splitter.indices(len(y))
    labeled, idx, heldout = parts["labeled"], parts["estimation"], parts["heldout"]
    if splitter.labeled_class_count(y, parts) != p.shape[-1]:
        return {"seed": seed, "status": "missing_labeled_class"}
    partial = splitter.partial_labels(y, parts)
    est = SSMEEstimator(max_iter=max_iter, tol=tol, bandwidth=bandwidth, labeled_weight=labeled_weight, random_state=seed, early_stopping=early_stopping).fit(p[idx], partial, model_names=names)
    report = est.report(n_draws=n_draws, title=f"{dataset} · Model Evaluation", sample_ids=idx, contribution=contribution)
    report.metadata.update(dataset=dataset, split_seed=seed)
    estimated = report.table[["model", "metric", "estimate"]].rename(columns={"estimate": "value"})
    baseline = point_metrics(y[labeled], p[labeled], names)
    targets = {"heldout": point_metrics(y[heldout], p[heldout], names),
               "estimation_pool": point_metrics(y[idx], p[idx], names)}
    rows, summaries = [], []
    for target, truth in targets.items():
        for method, predictions in [("ssme", estimated), ("labeled_only", baseline)]:
            detail = predictions.merge(truth, on=["model", "metric"], suffixes=("_estimate", "_truth"))
            detail["absolute_error"] = abs(detail.value_estimate-detail.value_truth)
            detail = detail.assign(seed=seed, method=method, target=target, dataset=dataset)
            rows.append(detail)
            for metric in METRICS:
                summaries.append(dict(seed=seed, method=method, target=target, metric=metric,
                                      **selection_stats(predictions[predictions.metric == metric], truth[truth.metric == metric], metric)))
    result = {"seed": seed, "status": "ok", "converged": est.converged_, "iterations": est.n_iter_, "bandwidth": est.bandwidth_, "stop_reason": est.stop_reason_,
              "final_responsibility_change": est.history_[-1]["max_responsibility_change"],
              "history": pd.DataFrame(est.history_),
              "report": report, "rows": pd.concat(rows), "summaries": pd.DataFrame(summaries),
              "splits": {"estimation": idx.tolist(), "labeled": labeled.tolist(), "heldout": heldout.tolist()},
              "contribution": pd.DataFrame(report.data['contribution']['rows']) if contribution else None}
    del est  # the fitted KDEs are large; drop before the worker ships the result back
    return result


def benchmark(scores, y, names, output, seeds=(0, 1, 2, 3, 4), n_labeled=20,
              n_unlabeled=1000, n_draws=100, max_iter=100, bandwidth="scott", contribution=False,
              dataset="custom", n_jobs=1, tol=1e-3, labeled_weight=10.0, early_stopping=False):
    p = probabilities(scores)
    y = np.asarray(y)
    if y.shape != (len(p),) or not np.isin(y, np.arange(p.shape[-1])).all():
        raise ValueError("benchmark needs complete ground-truth class indices")
    if n_labeled < 1 or n_unlabeled < 1 or n_labeled + n_unlabeled >= len(y):
        raise ValueError("reserve positive labeled, unlabeled and held-out sample counts")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide distinct nonempty seeds")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{dataset}: running {len(seeds)} seeds with n_jobs={n_jobs}", flush=True)
    # Seeds are independent (split and EM randomness derive from the seed alone),
    # so process-level parallelism reproduces the serial result exactly.
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_run_one_seed)(p, y, names, seed, n_labeled, n_unlabeled, n_draws, max_iter, bandwidth, dataset, contribution, tol, labeled_weight, early_stopping)
        for seed in seeds)
    rows, summaries, splits, statuses = [], [], {}, []
    for result in results:
        if result["status"] != "ok":
            statuses.append(result)
            continue
        seed = result["seed"]
        print(f"{dataset}: seed={seed} done, converged={result['converged']}, iterations={result['iterations']}", flush=True)
        rows.append(result["rows"])
        summaries.append(result["summaries"])
        splits[str(seed)] = result["splits"]
        statuses.append({"seed": seed, "status": "ok", "converged": result["converged"], "iterations": result["iterations"], "stop_reason": result["stop_reason"],
                         "bandwidth": result["bandwidth"], "final_responsibility_change": result["final_responsibility_change"]})
        result["history"].to_csv(out / f"convergence_seed{seed}.csv", index=False)
        if result["contribution"] is not None:
            result["contribution"].to_csv(out / f"contribution_seed{seed}.csv", index=False)
        if seed == 0:
            result["report"].save(out / "report")
            if result["contribution"] is not None:
                result["contribution"].to_csv(out / "contribution.csv", index=False)
    (out / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    (out / "run_status.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    import importlib.metadata
    config = dict(dataset=dataset, seeds=list(seeds), n_labeled=n_labeled, n_unlabeled=n_unlabeled,
                  n_draws=n_draws, max_iter=max_iter, tol=tol, early_stopping=early_stopping, bandwidth=bandwidth, labeled_weight=labeled_weight, model_names=names,
                  versions={name: importlib.metadata.version(name) for name in ["numpy", "scipy", "pandas", "scikit-learn", "ssme-lite"]},
                  python=platform.python_version(), platform=platform.platform())
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if not rows:
        raise ValueError("no usable split: see run_status.json; collect more labels")
    pd.concat(rows).to_csv(out / "details.csv", index=False)
    runs = pd.concat(summaries, ignore_index=True)
    runs.to_csv(out / "per_seed.csv", index=False)
    aggregates = []
    for keys, group in runs.groupby(["target", "method", "metric"]):
        record = dict(zip(["target", "method", "metric"], keys))
        for statistic in ["mae", "spearman", "top1_correct", "selection_regret"]:
            values = group[statistic].dropna().to_numpy()
            mean = values.mean() if len(values) else np.nan
            se = values.std(ddof=1)/np.sqrt(len(values)) if len(values) > 1 else np.nan
            half = t.ppf(0.975, len(values)-1)*se if len(values) > 1 else np.nan
            record.update({statistic: mean, f"{statistic}_se": se, f"{statistic}_ci_low": mean-half,
                           f"{statistic}_ci_high": mean+half, f"{statistic}_n": len(values)})
        aggregates.append(record)
    aggregate = pd.DataFrame(aggregates)
    aggregate.to_csv(out / "summary.csv", index=False)
    plot_comparison(aggregate, out / "comparison.png")
    return aggregate


def plot_comparison(table, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for metric, ax in zip(METRICS, axes):
        t = table[(table.target == "heldout") & (table.metric == metric)]
        ax.bar(t.method, t.mae, color=["#a8b7c5" if m == "labeled_only" else "#277da8" for m in t.method])
        ax.set_title(metric)
        ax.set_ylabel("Mean absolute estimation error")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Held-out ground truth | lower error is better")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
