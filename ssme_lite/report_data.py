"""Reproducible report statistics. No hidden labels, HTML, or plotting state."""
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from .prediction import alr


def json_safe(value):
    """Strict JSON: undefined statistics are null, never JavaScript NaN."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def histogram(values, bins=24, limits=None):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"edges": [], "counts": [], "n": 0}
    # Near-identical floating-point values may not admit 24 distinct bin edges.
    # Preserve the point-mass semantics instead of inventing a visible spread.
    if limits is None and np.ptp(values) <= 1e-12:
        center = float(values.mean())
        pad = max(abs(center)*1e-6, 1e-6)
        return {"edges": [center-pad, center+pad], "counts": [len(values)],
                "n": len(values), "point_mass": center}
    counts, edges = np.histogram(values, bins=bins, range=limits)
    return {"edges": edges.tolist(), "counts": counts.tolist(), "n": len(values),
            "point_mass": float(values[0]) if np.ptp(values) <= 1e-12 else None}


def bounded_density(values, lower, upper):
    """Reflected Gaussian KDE on a bounded domain, with explicit point masses."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"x": [], "density": [], "n": 0}
    if np.ptp(values) <= 1e-12:
        return {"x": [], "density": [], "n": len(values), "point_mass": float(values.mean())}
    kde = gaussian_kde(values)
    bandwidth = float(np.sqrt(kde.covariance[0, 0]))
    lo, hi = max(lower, values.min()-4*bandwidth), min(upper, values.max()+4*bandwidth)
    x = np.linspace(lo, hi, 512)
    density = kde(x) + kde(2*lower-x) + kde(2*upper-x)
    return {"x": x.tolist(), "density": density.tolist(), "n": len(values),
            "bandwidth": bandwidth, "bounds": [lower, upper], "method": "Scott bandwidth; boundary reflection"}


def distribution(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "quantiles": [], "x": [], "density": []}
    q = np.quantile(values, [0, .25, .5, .75, 1])
    result = {"n": len(values), "quantiles": q.tolist(), "mean": float(values.mean()),
              "x": [], "density": []}
    # A point mass is displayed honestly as a point, not a made-up violin.
    if len(values) > 1 and np.ptp(values) > 1e-12:
        x = np.linspace(values.min(), values.max(), 64)
        result.update(x=x.tolist(), density=gaussian_kde(values)(x).tolist())
    return result


def sampling_statistics(draws, lower_is_better=False):
    """Fractional ranks allocate ties equally across occupied positions.

    Ranking uses draws valid for *all* models; pairwise uses draws valid for
    that pair. Strict wins, losses and ties partition each pair's sample set.
    """
    n_models = draws.shape[1]
    complete = draws[np.isfinite(draws).all(axis=1)]
    rank = np.zeros((n_models, n_models))
    for row in complete:
        utility = -row if lower_is_better else row
        for j in range(n_models):
            tied = np.isclose(utility, utility[j], atol=1e-12, rtol=0)
            better = int(((utility > utility[j]) & ~tied).sum())
            count = int(tied.sum())
            rank[j, better:better+count] += 1 / count
    if len(complete):
        rank /= len(complete)
    else:
        rank[:] = np.nan
    wins = np.full((n_models, n_models), np.nan)
    ties = wins.copy()
    pairs = {}
    for i in range(n_models):
        for j in range(n_models):
            valid = np.isfinite(draws[:, i]) & np.isfinite(draws[:, j])
            delta = draws[valid, i] - draws[valid, j]
            if len(delta):
                tie = np.isclose(delta, 0, atol=1e-12, rtol=0)
                utility = -delta if lower_is_better else delta
                wins[i, j] = np.mean((utility > 0) & ~tie)
                ties[i, j] = np.mean(tie)
            if i != j:
                pairs[f"{i}:{j}"] = {
                    "a": i, "b": j, "n": int(valid.sum()),
                    "a_better": wins[i, j], "tie": ties[i, j],
                    "b_better": 1-wins[i, j]-ties[i, j],
                    "delta": histogram(delta),
                    "delta_mean": float(delta.mean()) if len(delta) else np.nan,
                }
    return {"rank_frequencies": rank, "rank_valid_draws": len(complete),
            "rank1": rank[:, 0], "top2": rank[:, :min(2, n_models)].sum(axis=1),
            "median_rank": [int(np.searchsorted(np.cumsum(r), .5)+1)
                            if len(complete) else None for r in rank],
            "pairwise_wins": wins, "pairwise_ties": ties, "pairs": pairs}


def build_report_data(estimator, table, values, metrics, metadata, sample_ids=None):
    p, gamma, y = estimator.scores_, estimator.posterior_, estimator.y_
    n, m, k = p.shape
    ids = list(range(n)) if sample_ids is None else list(sample_ids)
    if len(ids) != n:
        raise ValueError("sample_ids must match the full fitted pool")
    names = list(estimator.model_names_)
    metric_data = {}
    for mi, metric in enumerate(metrics):
        rows = table[table.metric == metric].set_index("model").loc[names].reset_index()
        order = rows.estimate.sort_values(ascending=metric == "ece", kind="stable").index.tolist()
        draws = values[:, :, mi]
        metric_data[metric] = {
            "rows": rows.to_dict("records"), "order": order,
            "samples": draws.T, "distributions": [distribution(draws[:, j]) for j in range(m)],
            **sampling_statistics(draws, metric == "ece"),
        }
    # Binary correlation uses positive-class probabilities; multiclass flattens
    # the aligned sample × class probability vector for each model.
    features = p[:, :, 1].T if k == 2 else p.transpose(1, 0, 2).reshape(m, -1)
    corr = np.full((m, m), np.nan)
    nonconstant = np.std(features, axis=1) > 1e-15
    if nonconstant.sum() > 1:
        corr[np.ix_(nonconstant, nonconstant)] = np.corrcoef(features[nonconstant])
    for j in np.flatnonzero(nonconstant):
        corr[j, j] = 1
    predictions = p.argmax(axis=2)
    agreement = np.mean(predictions[:, :, None] == predictions[:, None, :], axis=0)
    pair_list = [(i, j) for i in range(m) for j in range(i+1, m)]
    comparable = [(i, j) for i, j in pair_list if np.isfinite(corr[i, j])]
    landscape = {"correlation": corr, "agreement": agreement,
                 "correlation_definition": "positive-class probabilities" if k == 2 else "flattened sample × class probabilities",
                 "mean_correlation": np.mean([corr[i, j] for i, j in comparable]) if comparable else None,
                 "mean_agreement": np.mean([agreement[i, j] for i, j in pair_list]) if pair_list else None,
                 "most_similar": max(comparable, key=lambda ij: corr[ij]) if comparable else None,
                 "most_different": min(comparable, key=lambda ij: corr[ij]) if comparable else None}
    z = alr(p)
    components = min(3, *z.shape)
    if np.var(z, axis=0).sum() > 0:
        pca = PCA(n_components=components, svd_solver="full").fit(z)
        coordinates = pca.transform(z)
        variance = pca.explained_variance_ratio_
    else:
        coordinates, variance = np.zeros((n, components)), np.zeros(components)
    coordinates = np.pad(coordinates, ((0, 0), (0, 3-components)))
    variance = np.pad(variance, (0, 3-components))
    confidence = gamma.max(axis=1)
    entropy = -(gamma * np.log(np.clip(gamma, 1e-300, 1))).sum(axis=1)
    unlabeled = np.flatnonzero(y < 0)
    ambiguous = unlabeled[np.argsort(-entropy[unlabeled], kind="stable")[:12]]
    posterior = {
        "n_unlabeled": len(unlabeled), "threshold": .90,
        "mean_confidence": float(confidence[unlabeled].mean()) if len(unlabeled) else None,
        "min_confidence": float(confidence[unlabeled].min()) if len(unlabeled) else None,
        "max_confidence": float(confidence[unlabeled].max()) if len(unlabeled) else None,
        "min_entropy": float(entropy[unlabeled].min()) if len(unlabeled) else None,
        "max_entropy": float(entropy[unlabeled].max()) if len(unlabeled) else None,
        "expected_nonmodal_labels_per_draw": float((1-confidence[unlabeled]).sum()),
        "concentrated_fraction": float(np.mean(confidence[unlabeled] >= .90)) if len(unlabeled) else None,
        "confidence_histogram": histogram(confidence[unlabeled], limits=(0, 1)),
        "entropy_histogram": histogram(entropy[unlabeled], limits=(0, float(np.log(k)))),
        "confidence_density": bounded_density(confidence[unlabeled], 1/k, 1),
        "entropy_density": bounded_density(entropy[unlabeled], 0, float(np.log(k))),
        "class_counts": np.bincount(gamma[unlabeled].argmax(axis=1), minlength=k),
        "class_priors": estimator.priors_,
        "ambiguous": [{"id": str(ids[i]), "posterior": gamma[i],
                       "confidence": confidence[i], "entropy": entropy[i]} for i in ambiguous],
    }
    # Deterministic display cap; all posterior summaries still use the full pool.
    anchors = np.flatnonzero(y >= 0)
    budget = max(0, 3000-len(anchors))
    shown_u = unlabeled if len(unlabeled) <= budget else unlabeled[np.linspace(0, len(unlabeled)-1, budget, dtype=int)]
    shown = np.concatenate([shown_u, anchors])
    space = {"variance": variance, "alr_dimensions": z.shape[1], "total": n,
             "points": [{"id": str(ids[i]), "x": coordinates[i, 0], "y": coordinates[i, 1],
                         "z": coordinates[i, 2], "class": int(gamma[i].argmax()), "confidence": confidence[i],
                         "entropy": entropy[i], "labeled": bool(y[i] >= 0)} for i in shown]}
    return json_safe({
        "schema_version": 1, "metadata": metadata, "models": names, "metrics": metric_data,
        "fit_pool": {"n_samples": n, "n_labeled": int((y >= 0).sum()), "n_unlabeled": len(unlabeled), "n_classes": k},
        "landscape": landscape, "prediction_space": space, "posterior": posterior,
        "fit_history": estimator.history_,
        "definitions": {
            "ranking": "Complete-case sampled evaluations; ties split equally across occupied rank positions (absolute tolerance 1e-12).",
            "pairwise": "Pairwise-valid sampled evaluations; strict better / worse / tie frequencies sum to one. ECE uses lower is better.",
            "objective": "Weighted mean observed-data log density in ALR space: labeled log joint + unlabeled log marginal, at each M-step. Not a held-out score; monotonicity is not guaranteed.",
            "scope": "Performance uses the selected evaluation target. Landscape and PCA use the fitted pool; posterior uncertainty summaries use unlabeled samples only.",
        },
    })
