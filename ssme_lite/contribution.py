"""Controlled model-view ablation, without hidden ground truth or metric MC."""
import numpy as np
from scipy.special import xlogy
from .report_data import json_safe


def model_contribution(estimator):
    """Hold initialization and scalar bandwidth fixed while removing each view.

    This is conditional sensitivity, not out-of-sample utility or a Shapley
    value. The full model's initialization includes all input views. The same
    actual number of EM updates is used for all ablations, even if the original
    fit used early stopping. All effects are measured on unlabeled samples.
    """
    from .estimator import SSMEEstimator
    if estimator.scores_.shape[1] < 2:
        raise ValueError("contribution needs at least two models")
    u = estimator.unlabeled_indices_
    if not len(u):
        raise ValueError("contribution needs unlabeled samples")
    names = estimator.model_names_
    full = estimator.posterior_[u]
    predicted = estimator.scores_[u].argmax(axis=2)
    accuracy = full[np.arange(len(u))[:, None], predicted].mean(axis=0)
    rows, impacts = [], []
    for j, name in enumerate(names):
        keep = np.arange(len(names)) != j
        params = estimator.get_params()
        params.update(bandwidth=estimator.bandwidth_, max_iter=estimator.n_iter_, early_stopping=False)
        reduced = SSMEEstimator(**params).fit(
            estimator.scores_[:, keep], estimator.y_,
            model_names=np.asarray(names)[keep].tolist(),
            initial_responsibilities=estimator.initial_responsibilities_,
        )
        posterior = reduced.posterior_[u]
        tv = .5*np.abs(full-posterior).sum(axis=1)
        midpoint = (full+posterior)/2
        js = .5*(xlogy(full, np.divide(full, midpoint, out=np.ones_like(full), where=midpoint>0)).sum(1)
                 + xlogy(posterior, np.divide(posterior, midpoint, out=np.ones_like(full), where=midpoint>0)).sum(1))
        # Compare only remaining models: removing the winner is not itself a
        # rank change attributable to a new posterior.
        reduced_accuracy = posterior[np.arange(len(u))[:, None], predicted[:, keep]].mean(0)
        base = accuracy[keep]
        delta = reduced_accuracy-base
        before_top = set(np.flatnonzero(np.isclose(base, base.max(), atol=1e-12, rtol=0)))
        after_top = set(np.flatnonzero(np.isclose(reduced_accuracy, reduced_accuracy.max(), atol=1e-12, rtol=0)))
        rank_before = np.array([1+np.sum(base > x+1e-12) for x in base])
        rank_after = np.array([1+np.sum(reduced_accuracy > x+1e-12) for x in reduced_accuracy])
        rows.append({
            "removed_model": name,
            "posterior_total_variation": float(tv.mean()),
            "posterior_js_divergence_nats": float(np.maximum(js, 0).mean()),
            "posterior_class_flip_rate": float(np.mean(full.argmax(1) != posterior.argmax(1))),
            "posterior_mean_abs_change": float(np.abs(full-posterior).mean()),
            "remaining_accuracy_mean_abs_change": float(np.abs(delta).mean()),
            "remaining_accuracy_max_abs_change": float(np.abs(delta).max()),
            "remaining_rank_mean_abs_change": float(np.abs(rank_after-rank_before).mean()),
            "remaining_top1_changed": before_top != after_top,
            "mean_entropy_change_nats": float((-xlogy(posterior, posterior).sum(1)+xlogy(full, full).sum(1)).mean()),
            "epochs": reduced.n_iter_, "bandwidth": reduced.bandwidth_,
        })
        impact = np.full(len(names), np.nan)
        impact[keep] = delta
        impacts.append(impact)
    return json_safe({
        "method": "controlled_leave_one_model_out", "n_refits": len(names),
        "target": "unlabeled", "n_samples": len(u), "rows": rows,
        "accuracy_delta_matrix": impacts,
        "order": np.argsort([-r['posterior_total_variation'] for r in rows], kind='stable'),
        "protocol": {"fixed_bandwidth": estimator.bandwidth_, "fixed_epochs": estimator.n_iter_,
                     "initialization": "shared full-fit initial responsibilities (contains all model views)",
                     "labels": "same visible labels, clamped in every refit",
                     "accuracy": "exact posterior expectation on unlabeled samples; remaining candidates only",
                     "delta": "ablated minus full; positive Accuracy delta means a higher estimate, not greater truth accuracy",
                     "interpretation": "Conditional influence, not improvement in evaluation accuracy; does not quantify interactions or Shapley values."},
    })
