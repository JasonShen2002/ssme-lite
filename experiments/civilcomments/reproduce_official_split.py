"""CivilComments figure on the official half-split protocol.

The public SSME code holds out half the rows (66,891 on this release) as the
ground-truth evaluation split, then draws 20 labeled and 1,000 unlabeled
comments from the other half. Metric error is the estimate on that estimation
split minus the metric on the held-out half.

Baselines follow Appendix B.2: logistic regression pseudo-labels, Dawid-Skene on
predictions thresholded at 1/2, and an accuracy-weighted majority vote. The three
extra predictors on the bar chart are mixtures of the seven released scores.
They are scored with the labels SSME imputes; they are not extra inputs to EM.
"""
from pathlib import Path
import os
import sys

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

OFFICIAL = Path(__file__).resolve().parents[3] / "official" / "SSME"
sys.path.insert(0, str(OFFICIAL))

from baselines import labeled_data_alone  # noqa: E402
from model import SSME_KDE  # noqa: E402
from utils import (  # noqa: E402
    DATASET_INFO,
    N_DRAWS,
    create_metrics_df,
    get_model_values_df,
    sample_data,
)

NAMES = DATASET_INFO["CivilComments"]["model_names"]
METRICS = ("acc", "ece", "auc", "auprc")
MIXTURES = ("ensemble", "mix_erm_coral", "mix_irm_erm")


def _mixture_scores(positive):
    coral, erm, irm = positive[:, 0], positive[:, 1], positive[:, 2]
    return np.column_stack([
        positive.mean(axis=1),
        0.5 * (erm + coral),
        0.5 * (irm + erm),
    ])


def _dawid_skene(votes, labeled_index, labeled_y, max_iter=100, tol=1e-5):
    """Binary Dawid-Skene. Known labels stay fixed. votes are 0/1, shape (n, m)."""
    n, m = votes.shape
    classes = 2
    posterior = np.full((n, classes), 0.5)
    if len(labeled_index):
        posterior[labeled_index] = 0.0
        posterior[labeled_index, labeled_y.astype(int)] = 1.0
    else:
        posterior[:, 1] = votes.mean(axis=1)
        posterior[:, 0] = 1.0 - posterior[:, 1]
    last = posterior.copy()
    for _ in range(max_iter):
        pi = posterior.mean(axis=0)
        pi = np.clip(pi, 1e-6, None)
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
        if len(labeled_index):
            posterior[labeled_index] = 0.0
            posterior[labeled_index, labeled_y.astype(int)] = 1.0
        if np.max(np.abs(posterior - last)) < tol:
            break
        last = posterior.copy()
    return posterior.argmax(axis=1)


def _metric_frame(labels, positive, dataset="CivilComments"):
    groups = [np.array(["global"] * len(labels))]
    frame = create_metrics_df(np.asarray(labels), np.asarray(positive, dtype=float), groups, dataset=dataset)
    return frame.sort_values("model_idx")


def _one_seed(df, seed):
    np.random.seed(seed)
    train = df.sample(frac=0.5, random_state=seed)
    test = df.loc[~df.index.isin(train.index)]
    sampled, partial, _, _ = sample_data(train, 20, 1000, NAMES, seed, 2)
    labeled = np.flatnonzero(partial != -1)
    unlabeled = np.flatnonzero(partial == -1)
    y_labeled = partial[labeled].astype(int)
    groups = [np.array(["global"] * len(partial))]
    test_positive = test[NAMES].to_numpy(dtype=float)
    test_groups = [np.array(["global"] * len(test))]
    config = {"dataset": "CivilComments"}

    draws = {"count": 0, "mixture": []}
    import utils
    original = utils.create_metrics_df

    def capture(labels, predictions, demographics_list, dataset=None):
        frame = original(labels, predictions, demographics_list, dataset=dataset)
        draws["count"] += 1
        if draws["count"] > 1:
            mixture = _mixture_scores(predictions)
            drawn = original(labels, mixture, demographics_list, dataset=dataset)
            draws["mixture"].append(drawn.sort_values("model_idx"))
        return frame

    utils.create_metrics_df = capture
    import model as model_module
    model_module.create_metrics_df = capture
    try:
        ssme = SSME_KDE(
            (sampled[labeled], [groups[0][labeled]], y_labeled),
            (sampled[unlabeled], [groups[0][unlabeled]], partial[unlabeled]),
            config,
        ).sort_values("model_idx")
    finally:
        utils.create_metrics_df = original
        model_module.create_metrics_df = original

    gt = labeled_data_alone((test_positive, test_groups, test["label"].to_numpy()), config).sort_values("model_idx")
    labeled_only = _metric_frame(y_labeled, sampled[labeled])
    votes = (sampled >= 0.5).astype(int)
    weights = np.array([np.mean(votes[labeled, j] == y_labeled) for j in range(votes.shape[1])])
    weights = np.clip(weights, 1e-6, None)
    majority = (votes @ weights / weights.sum() >= 0.5).astype(int)
    majority[labeled] = y_labeled
    majority_metrics = _metric_frame(majority, sampled)
    pseudo = LogisticRegression().fit(sampled[labeled], y_labeled).predict(sampled).astype(int)
    pseudo[labeled] = y_labeled
    pseudo_metrics = _metric_frame(pseudo, sampled)
    # Appendix B.2 runs Dawid-Skene on discretized votes only. Clamping the 20
    # known labels into the posterior moves accuracy error from 4.98 to 6.42 pp.
    dawid = _dawid_skene(votes, np.array([], dtype=int), np.array([], dtype=int))
    dawid_metrics = _metric_frame(dawid, sampled)

    mixture_draws = (pd.concat(draws["mixture"])
                     .groupby("model_idx")[list(METRICS)]
                     .mean()
                     .reset_index())
    mixture_positive_test = _mixture_scores(test_positive)
    mixture_gt = _metric_frame(test["label"].to_numpy(), mixture_positive_test)
    mixture_labeled = _metric_frame(y_labeled, _mixture_scores(sampled[labeled]))

    rows = []
    methods = {
        "ssme": ssme,
        "labeled_only": labeled_only,
        "majority_vote": majority_metrics,
        "pl": pseudo_metrics,
        "dawid_skene": dawid_metrics,
    }
    for method, frame in methods.items():
        for metric in METRICS:
            error = np.abs(frame[metric].to_numpy() - gt[metric].to_numpy())
            for model_idx, value in enumerate(error):
                rows.append(dict(seed=seed, method=method, metric=metric, model=NAMES[model_idx],
                                 absolute_error=float(value), n_draws=N_DRAWS))
    for method, frame in (("ssme", mixture_draws), ("labeled_only", mixture_labeled)):
        for metric in METRICS:
            error = np.abs(frame[metric].to_numpy() - mixture_gt[metric].to_numpy())
            for model_idx, value in enumerate(error):
                rows.append(dict(seed=seed, method=method, metric=metric, model=MIXTURES[model_idx],
                                 absolute_error=float(value), n_draws=int(draws["count"] - 1)))
    return rows


def plot(details_path, out_dir):
    details = pd.read_csv(details_path)
    base_names = set(NAMES)
    base = details[details.model.isin(base_names)]
    mae = base.groupby(["method", "metric"])["absolute_error"].mean()
    labeled = mae.xs("labeled_only", level="method")
    order = ["accuracy", "ece", "auc", "auprc"]
    metric_key = {"accuracy": "acc", "ece": "ece", "auc": "auc", "auprc": "auprc"}
    methods = [
        ("majority_vote", "Majority Vote", "#d94f4f", "x"),
        ("pl", "PL", "#e8923a", "x"),
        ("dawid_skene", "Dawid-Skene", "#d4a017", "x"),
        ("ssme", "SSME", "#5c5c5c", "o"),
    ]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    y_pos = np.arange(len(order))[::-1]
    ax.axvline(1, color="#8a8a8a", linestyle=(0, (1.2, 1.4)), linewidth=1.3, label="Labeled", zorder=1)
    for method, label, color, marker in methods:
        values = [mae.loc[(method, metric_key[name])] / labeled.loc[metric_key[name]] for name in order]
        ax.scatter(values, y_pos, s=64 if marker == "o" else 46, color=color, marker=marker,
                   linewidths=1.6, zorder=3, label=label)
    ax.set_yticks(y_pos, ["ACC", "ECE", "AUC", "AUPRC"])
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("RMAE")
    ax.set_title("Relative estimation error (RMAE)")
    ax.grid(axis="x", color="#e6e6e6", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "rmae.png", dpi=160, bbox_inches="tight")
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
    per_model = (details.groupby(["model", "metric", "method"], as_index=False)["absolute_error"].mean())
    per_model["pp"] = per_model.absolute_error * 100
    labeled_color, ssme_color = "#c5c5c5", "#2c5f8a"

    def panel(metric):
        rows = per_model[per_model.metric == metric].set_index(["model", "method"])
        labeled_pp = [rows.loc[(name, "labeled_only"), "pp"] for name, _ in model_order]
        ssme_pp = [rows.loc[(name, "ssme"), "pp"] for name, _ in model_order]
        return np.array(labeled_pp), np.array(ssme_pp)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 6.2), sharey=True)
    positions = np.arange(len(model_order))
    for ax, metric, title in zip(axes, ["auc", "acc"], ["AUC error (percentage points)", "ACC error (percentage points)"]):
        labeled_pp, ssme_pp = panel(metric)
        ax.barh(positions - 0.18, ssme_pp, height=0.32, color=ssme_color, label="SSME", zorder=2)
        ax.barh(positions + 0.18, labeled_pp, height=0.32, color=labeled_color, label="Labeled only", zorder=2)
        for y, value in zip(positions - 0.18, ssme_pp):
            ax.text(value + 0.12, y, f"{value:.1f}", va="center", fontsize=8, color=ssme_color)
        for y, value in zip(positions + 0.18, labeled_pp):
            ax.text(value + 0.12, y, f"{value:.1f}", va="center", fontsize=8, color="#555555")
        ax.set_yticks(positions, [label for _, label in model_order])
        ax.set_xlim(0, 10)
        ax.set_xlabel("(estimate − holdout), percentage points")
        ax.set_title(title)
        ax.grid(axis="x", color="#e6e6e6", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index("Labeled only"), labels.index("SSME")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("10 models: SSME vs labeled data only", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "model_errors.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(seeds=range(50), n_jobs=8):
    os.chdir(OFFICIAL)
    df = get_model_values_df("CivilComments", NAMES)
    pieces = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_one_seed)(df, seed) for seed in seeds)
    out = Path(__file__).resolve().parents[3] / "results" / "civilcomments_official_split"
    out.mkdir(parents=True, exist_ok=True)
    details = pd.DataFrame([row for piece in pieces for row in piece])
    details.to_csv(out / "details.csv", index=False)
    base = details[details.model.isin(NAMES)]
    summary = (base.groupby(["method", "metric"])["absolute_error"].mean()
               .unstack("metric")
               .reindex(columns=list(METRICS)))
    labeled = summary.loc["labeled_only"]
    rmae = summary.div(labeled, axis=1)
    summary.to_csv(out / "mae.csv")
    rmae.to_csv(out / "rmae.csv")
    figure_dir = Path(__file__).resolve().parent / "results"
    plot(out / "details.csv", figure_dir)
    print(out)
    print((summary * 100).round(2))
    print(rmae.round(3))


if __name__ == "__main__":
    main()
