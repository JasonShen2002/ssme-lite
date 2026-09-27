"""CivilComments figure from this package, on the paper's half-split protocol.

Half the comments (66,891) are the held-out ground truth. Each seed draws 20
labels and 1,000 unlabeled comments from the other half with
``SemiSupervisedSplit``. SSME estimates come from ``SSMEEstimator``
(``bandwidth="official"``, 20 EM epochs, labeled weight 10), not from the
public ``SSME_KDE`` function.

Baselines follow Appendix B.2 and are scored with this package's metrics.
The three extra predictors are mixtures of the seven released scores. They are
scored with labels drawn from this package's posterior; they are not extra
inputs to EM.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

PACKAGE = Path(__file__).resolve().parents[2]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from ssme_lite import SSMEEstimator, SemiSupervisedSplit
from ssme_lite.benchmark import official_scores
from ssme_lite.metrics import METRICS, metric_values, point_metrics

INPUTS = PACKAGE.parent / "official" / "SSME" / "inputs"
NAMES = [
    "alg_CORAL", "alg_ERM", "alg_IRM",
    "alg_ERM_seed1", "alg_ERM_seed2", "alg_IRM_seed1", "alg_IRM_seed2",
]
MIXTURES = ("ensemble", "mix_erm_coral", "mix_irm_erm")
N_DRAWS = 100


def _mixture_scores(positive):
    coral, erm, irm = positive[:, 0], positive[:, 1], positive[:, 2]
    return np.column_stack([
        positive.mean(axis=1),
        0.5 * (erm + coral),
        0.5 * (irm + erm),
    ])


def _as_probabilities(positive):
    positive = np.asarray(positive, dtype=float)
    return np.stack([1.0 - positive, positive], axis=-1)


def _dawid_skene(votes, max_iter=100, tol=1e-5):
    """Binary Dawid-Skene on discretized votes. Shape (n, m), values in {0, 1}."""
    n, m = votes.shape
    classes = 2
    posterior = np.column_stack([1.0 - votes.mean(axis=1), votes.mean(axis=1)])
    last = posterior.copy()
    for _ in range(max_iter):
        pi = np.clip(posterior.mean(axis=0), 1e-6, None)
        pi /= pi.sum()
        theta = np.zeros((m, classes, classes))
        for annotator in range(m):
            for true_class in range(classes):
                weight = posterior[:, true_class]
                for observed in range(classes):
                    theta[annotator, true_class, observed] = np.sum(weight * (votes[:, annotator] == observed))
                total = theta[annotator, true_class].sum()
                theta[annotator, true_class] = (theta[annotator, true_class] + 1e-3) / (total + 2e-3)
        log_post = np.broadcast_to(np.log(pi), (n, classes)).copy()
        for annotator in range(m):
            observed = votes[:, annotator]
            log_post += np.log(theta[annotator][:, observed].T + 1e-12)
        log_post -= log_post.max(axis=1, keepdims=True)
        posterior = np.exp(log_post)
        posterior /= posterior.sum(axis=1, keepdims=True)
        if np.max(np.abs(posterior - last)) < tol:
            break
        last = posterior.copy()
    return posterior.argmax(axis=1)


def _absolute_errors(estimate, truth):
    merged = estimate.merge(truth, on=["model", "metric"], suffixes=("_estimate", "_truth"))
    merged["absolute_error"] = (merged.value_estimate - merged.value_truth).abs()
    return merged


def _posterior_metrics(posterior, labels, model_scores, names, seed, n_draws=N_DRAWS):
    """Match ``make_report``: exact Accuracy, Monte Carlo for the other metrics."""
    rng = np.random.default_rng(seed)
    known = labels >= 0
    k = model_scores.shape[-1]
    nonlinear = [metric for metric in METRICS if metric != "accuracy"]
    draws = {metric: np.empty((n_draws, len(names))) for metric in nonlinear}
    for draw in range(n_draws):
        sampled = np.minimum((rng.random((len(labels), 1)) > np.cumsum(posterior, 1)).sum(1), k - 1)
        sampled[known] = labels[known]
        for j, name in enumerate(names):
            result = metric_values(sampled, model_scores[:, j], nonlinear)
            for metric in nonlinear:
                draws[metric][draw, j] = result[metric]
    predicted = model_scores.argmax(axis=-1)
    accuracy = np.array([posterior[np.arange(len(posterior)), predicted[:, j]].mean() for j in range(len(names))])
    rows = [{"model": name, "metric": "accuracy", "value": float(accuracy[j])} for j, name in enumerate(names)]
    for metric in nonlinear:
        mean = np.nanmean(draws[metric], axis=0)
        rows.extend({"model": name, "metric": metric, "value": float(mean[j])} for j, name in enumerate(names))
    return pd.DataFrame(rows)


def _one_seed(scores, y, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_eval = len(y) // 2
    test, train = perm[:n_eval], perm[n_eval:]
    splitter = SemiSupervisedSplit(20, 1000, seed)
    local = splitter.indices(len(train))
    if splitter.labeled_class_count(y[train], local) != scores.shape[-1]:
        return None
    partial = splitter.partial_labels(y[train], local)
    pool = scores[train[local["estimation"]]]
    labeled_scores = scores[train[local["labeled"]]]
    y_labeled = y[train[local["labeled"]]]
    estimator = SSMEEstimator(
        max_iter=20, early_stopping=False, bandwidth="official",
        labeled_weight=10.0, random_state=seed, n_jobs=1,
    )
    estimator.fit(pool, partial, model_names=NAMES)
    ssme = estimator.report(n_draws=N_DRAWS, primary_metric="accuracy").table.rename(columns={"estimate": "value"})
    ssme = ssme[["model", "metric", "value"]]
    truth = point_metrics(y[test], scores[test], NAMES)
    labeled_only = point_metrics(y_labeled, labeled_scores, NAMES)

    positive = pool[:, :, 1]
    votes = (positive >= 0.5).astype(int)
    weights = np.clip(np.array([np.mean(votes[:20, j] == y_labeled) for j in range(votes.shape[1])]), 1e-6, None)
    majority = (votes @ weights / weights.sum() >= 0.5).astype(int)
    majority[:20] = y_labeled
    pseudo = LogisticRegression().fit(positive[:20], y_labeled).predict(positive).astype(int)
    pseudo[:20] = y_labeled
    dawid = _dawid_skene(votes)
    methods = {
        "ssme": ssme,
        "labeled_only": labeled_only,
        "majority_vote": point_metrics(majority, pool, NAMES),
        "pl": point_metrics(pseudo, pool, NAMES),
        "dawid_skene": point_metrics(dawid, pool, NAMES),
    }
    rows = []
    for method, frame in methods.items():
        errors = _absolute_errors(frame, truth)
        for record in errors.itertuples(index=False):
            rows.append(dict(seed=seed, method=method, metric=record.metric, model=record.model,
                             absolute_error=float(record.absolute_error), n_draws=N_DRAWS,
                             estimator="ssme_lite.SSMEEstimator"))

    mixture_pool = _as_probabilities(_mixture_scores(positive))
    mixture_test = _as_probabilities(_mixture_scores(scores[test][:, :, 1]))
    mixture_labeled = _as_probabilities(_mixture_scores(labeled_scores[:, :, 1]))
    mixture_truth = point_metrics(y[test], mixture_test, MIXTURES)
    mixture_methods = {
        "ssme": _posterior_metrics(estimator.posterior_, estimator.y_, mixture_pool, MIXTURES, seed),
        "labeled_only": point_metrics(y_labeled, mixture_labeled, MIXTURES),
    }
    for method, frame in mixture_methods.items():
        errors = _absolute_errors(frame, mixture_truth)
        for record in errors.itertuples(index=False):
            rows.append(dict(seed=seed, method=method, metric=record.metric, model=record.model,
                             absolute_error=float(record.absolute_error), n_draws=N_DRAWS,
                             estimator="ssme_lite.SSMEEstimator"))
    return rows


def plot(details_path, out_dir):
    details = pd.read_csv(details_path)
    base = details[details.model.isin(NAMES)]
    mae = base.groupby(["method", "metric"])["absolute_error"].mean()
    labeled = mae.xs("labeled_only", level="method")
    # Display order is top to bottom. METRICS itself is accuracy, auc, auprc, ece.
    order = ["accuracy", "ece", "auc", "auprc"]
    methods = [
        ("majority_vote", "Majority Vote", "#d94f4f", "x"),
        ("pl", "PL", "#e8923a", "x"),
        ("dawid_skene", "Dawid-Skene", "#d4a017", "x"),
        ("ssme", "SSME", "#5c5c5c", "o"),
    ]
    import matplotlib.pyplot as plt
    style = {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 0.9,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.4, 3.15))
        y_pos = np.arange(len(order))[::-1]
        ax.axvline(1, color="#8a8a8a", linestyle=(0, (1.4, 1.5)), linewidth=1.5,
                   label="Labeled", zorder=1)
        for method, label, color, marker in methods:
            values = [mae.loc[(method, name)] / labeled.loc[name] for name in order]
            ax.scatter(values, y_pos, s=130 if marker == "o" else 95, color=color, marker=marker,
                       linewidths=2.2, zorder=3, label=label)
        ax.set_yticks(y_pos, ["ACC", "ECE", "AUC", "AUPRC"])
        ax.set_ylim(-0.55, len(order) - 0.45)
        ax.set_xlim(0, 1.72)
        ax.set_xticks(np.arange(0, 1.51, 0.25))
        ax.set_xlabel("RMAE")
        ax.set_title("Relative estimation error (RMAE)", pad=8)
        ax.grid(axis="x", color="#e4e4e4", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  handletextpad=0.5, labelspacing=0.55, borderaxespad=0.2)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "rmae.png", dpi=300, bbox_inches="tight", pad_inches=0.08,
                    facecolor="white")
        plt.close(fig)

    model_order = [
        ("alg_CORAL", "CORAL"),
        ("alg_ERM", "ERM"),
        ("alg_IRM", "IRM"),
        ("alg_ERM_seed1", "ERM s1"),
        ("alg_ERM_seed2", "ERM s2"),
        ("alg_IRM_seed1", "IRM s1"),
        ("alg_IRM_seed2", "IRM s2"),
        ("ensemble", "Ensemble"),
        ("mix_erm_coral", "mix ERM+CORAL"),
        ("mix_irm_erm", "mix IRM+ERM"),
    ]
    per_model = details.groupby(["model", "metric", "method"], as_index=False)["absolute_error"].mean()
    per_model["pp"] = per_model.absolute_error * 100
    labeled_color, ssme_color = "#c5c5c5", "#2c5f8a"

    def panel(metric):
        rows = per_model[per_model.metric == metric].set_index(["model", "method"])
        labeled_pp = [rows.loc[(name, "labeled_only"), "pp"] for name, _ in model_order]
        ssme_pp = [rows.loc[(name, "ssme"), "pp"] for name, _ in model_order]
        return np.array(labeled_pp), np.array(ssme_pp)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 6.2), sharey=True)
    positions = np.arange(len(model_order))
    for ax, metric, title in zip(axes, ["auc", "accuracy"], ["AUC error (percentage points)", "ACC error (percentage points)"]):
        labeled_pp, ssme_pp = panel(metric)
        ax.barh(positions - 0.18, ssme_pp, height=0.32, color=ssme_color, label="SSME", zorder=2)
        ax.barh(positions + 0.18, labeled_pp, height=0.32, color=labeled_color, label="Labeled only", zorder=2)
        for y, value in zip(positions - 0.18, ssme_pp):
            ax.text(value + 0.12, y, f"{value:.1f}", va="center", fontsize=8, color=ssme_color)
        for y, value in zip(positions + 0.18, labeled_pp):
            ax.text(value + 0.12, y, f"{value:.1f}", va="center", fontsize=8, color="#555555")
        ax.set_yticks(positions, [label for _, label in model_order])
        ax.set_xlim(0, 12)
        ax.set_xlabel("(estimate − holdout), percentage points")
        ax.set_title(title)
        ax.grid(axis="x", color="#e6e6e6", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    legend_order = [labels.index("Labeled only"), labels.index("SSME")]
    fig.legend([handles[i] for i in legend_order], [labels[i] for i in legend_order],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("10 models: SSME vs labeled data only", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "model_errors.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(n_success=50, n_jobs=8):
    scores, y, names = official_scores(INPUTS, "CivilComments")
    if names != NAMES:
        raise ValueError(f"unexpected model order: {names}")
    y = np.asarray(y)
    collected, seed = [], 0
    # Draw seeds until 50 splits contain both classes. The package does not
    # look at hidden labels to repair a one-class labeled draw.
    pending = []
    while len(collected) < n_success:
        batch = list(range(seed, seed + n_jobs * 2))
        seed = batch[-1] + 1
        pieces = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_one_seed)(scores, y, item) for item in batch)
        for item, piece in zip(batch, pieces):
            pending.append((item, piece))
        pending.sort()
        collected = [piece for _, piece in pending if piece]
        print(f"valid splits {len(collected)} / attempts {pending[-1][0] + 1}", flush=True)
    details = pd.DataFrame([row for piece in collected[:n_success] for row in piece])
    out = PACKAGE.parent / "results" / "civilcomments_package_half_split"
    out.mkdir(parents=True, exist_ok=True)
    details.to_csv(out / "details.csv", index=False)
    base = details[details.model.isin(NAMES)]
    summary = (base.groupby(["method", "metric"])["absolute_error"].mean()
               .unstack("metric")
               .reindex(columns=list(METRICS)))
    rmae = summary.div(summary.loc["labeled_only"], axis=1)
    summary.to_csv(out / "mae.csv")
    rmae.to_csv(out / "rmae.csv")
    plot(out / "details.csv", Path(__file__).resolve().parent / "results")
    print(out)
    print((summary * 100).round(2))
    print(rmae.round(3))
    print("seeds", sorted(details.seed.unique()))


if __name__ == "__main__":
    main()
