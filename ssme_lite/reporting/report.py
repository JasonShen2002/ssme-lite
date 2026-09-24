"""Monte Carlo label uncertainty and portable model selection reports."""
from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
from ..metrics import metric_values, METRICS

INTERVAL_NOTE = ("Intervals are conditional latent-label uncertainty given a fitted posterior. "
                 "They exclude density-fit, bandwidth and population sampling uncertainty. "
                 "mc_standard_error measures numerical Monte Carlo error only.")


class EvaluationReport:
    def __init__(self, table, metadata=None, data=None):
        self.table = table
        self.metadata = metadata or {}
        self.data = data

    def ranking(self, metric="accuracy"):
        result = self.table[self.table.metric == metric].copy()
        if result.empty:
            raise ValueError("metric is not present in report")
        result["rank"] = result.estimate.rank(ascending=(metric == "ece"), method="min")
        return result.sort_values(["rank", "model"], na_position="last")

    def show(self):
        print(self.table.to_string(index=False))
        print("Converged:", self.metadata.get("converged"), "| Iterations:", self.metadata.get("n_iter"))
        print(INTERVAL_NOTE)
        return self

    def plot(self, metric="accuracy", path=None):
        import matplotlib.pyplot as plt
        t = self.ranking(metric).sort_values("rank", ascending=False)
        fig, ax = plt.subplots(figsize=(8, max(3, len(t) * 0.4)))
        positions = np.arange(len(t))
        ax.hlines(positions, t.interval_low, t.interval_high, color="#3182bd", linewidth=3)
        ax.scatter(t.estimate, positions, color="#153c5b", zorder=3)
        ax.set_yticks(positions, t.model)
        ax.set_xlabel(metric)
        ax.set_title("Estimated performance | conditional label intervals")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=160)
            plt.close(fig)
        return fig

    def save(self, directory):
        """Write the report directory."""
        import json
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(directory / "metrics.csv", index=False)
        (directory / "metadata.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        panels = []
        for metric in self.table.metric.unique():
            self.plot(metric, directory / f"{metric}.png")
            panels.append(f"<h2>{escape(metric)}</h2>" + self.ranking(metric).to_html(index=False, float_format=lambda x: f"{x:.4f}") + f'<img src="{metric}.png" alt="{metric} estimates">')
        if self.data is not None:
            from .html import render_report
            from .data import json_safe
            self.data["metadata"] = self.metadata
            if self.data.get("contribution"):
                pd.DataFrame(self.data["contribution"]["rows"]).to_csv(directory / "contribution.csv", index=False)
            payload = json.dumps(json_safe(self.data), ensure_ascii=False, allow_nan=False)
            (directory / "report_data.json").write_text(payload, encoding="utf-8")
            (directory / "report.html").write_text(render_report(payload), encoding="utf-8")
            return directory / "report.html"
        (directory / "report.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>SSME-Lite report</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#183047}'
            'table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #d8e0e7}img{max-width:100%}</style>'
            '<h1>SSME-Lite Model Evaluation</h1><p>' + escape(INTERVAL_NOTE) + '</p>'
            + '<h2>Fit status</h2><pre>' + escape(json.dumps(self.metadata, indent=2)) + '</pre>'
            + ("<p><strong>Iteration limit reached: treat estimates as provisional and check convergence/bandwidth.</strong></p>" if not self.metadata.get("converged", True) else "")
            + ''.join(panels) + '</html>', encoding="utf-8")
        return directory / "report.html"


def make_report(estimator, metrics, n_draws, confidence, target, random_state,
                title=None, sample_ids=None, primary_metric=None, contribution=False):
    metrics = tuple(metrics)
    if not metrics or len(set(metrics)) != len(metrics) or any(m not in METRICS for m in metrics):
        raise ValueError(f"provide distinct metrics from {METRICS}")
    if primary_metric is not None and primary_metric not in metrics:
        raise ValueError("primary_metric must be included in metrics")
    if n_draws < 2 or not 0 < confidence < 1 or target not in ("all", "unlabeled"):
        raise ValueError("n_draws>=2, 0<confidence<1, target='all' or 'unlabeled' required")
    idx = np.arange(len(estimator.y_)) if target == "all" else estimator.unlabeled_indices_
    if not len(idx):
        raise ValueError("evaluation target is empty")
    rng = np.random.default_rng(estimator.random_state if random_state is None else random_state)
    p = estimator.scores_[idx]
    gamma = estimator.posterior_[idx]
    labels = estimator.y_[idx]
    known = labels >= 0
    values = np.empty((n_draws, p.shape[1], len(metrics)))
    for b in range(n_draws):
        y = np.minimum((rng.random((len(idx), 1)) > np.cumsum(gamma, 1)).sum(1), p.shape[-1] - 1)
        y[known] = labels[known]
        for j in range(p.shape[1]):
            result = metric_values(y, p[:, j], metrics)
            values[b, j] = [result[m] for m in metrics]
    alpha = (1 - confidence) / 2
    rows = []
    for j, name in enumerate(estimator.model_names_):
        for mi, metric in enumerate(metrics):
            draws = values[:, j, mi]
            draws = draws[np.isfinite(draws)]
            mean = float(draws.mean()) if len(draws) else np.nan
            # Accuracy is linear in labels; use its exact conditional expectation.
            if metric == "accuracy":
                mean = float(gamma[np.arange(len(p)), p[:, j].argmax(1)].mean())
            sd = float(draws.std(ddof=1)) if len(draws) > 1 else np.nan
            lo, hi = np.quantile(draws, [alpha, 1-alpha]) if len(draws) else [np.nan, np.nan]
            rows.append(dict(model=name, metric=metric, estimate=mean, label_std=sd,
                             mc_standard_error=0.0 if metric == "accuracy" else sd / np.sqrt(len(draws)) if len(draws) else np.nan,
                             interval_low=lo, interval_high=hi, valid_draws=len(draws)))
    table = pd.DataFrame(rows)
    metadata = {
        "title": title or "Model Evaluation Report", "primary_metric": primary_metric or metrics[0],
        "random_state": estimator.random_state if random_state is None else random_state,
        "target": target, "n_samples": len(idx), "n_labeled": int(known.sum()),
        "n_draws": n_draws, "confidence": confidence, "interval_type": "conditional_latent_label",
        "note": INTERVAL_NOTE, "bandwidth": estimator.bandwidth_, "n_iter": estimator.n_iter_,
        "bandwidth_rule": getattr(estimator, "bandwidth", None),
        "labeled_weight": getattr(estimator, "labeled_weight", None),
        "tol": getattr(estimator, "tol", None),
        "max_iter": estimator.max_iter, "early_stopping": estimator.early_stopping,
        "stop_reason": estimator.stop_reason_,
        "converged": estimator.converged_}
    from .data import build_report_data
    data = build_report_data(estimator, table, values, metrics, metadata, sample_ids)
    if contribution:
        from ..contribution import model_contribution
        data["contribution"] = model_contribution(estimator)
    return EvaluationReport(table, metadata, data)
