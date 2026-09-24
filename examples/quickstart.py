"""Run from any directory after pip install -e ."""
from pathlib import Path
import numpy as np
from ssme_lite import SSMEEstimator
from ssme_lite.experiments import gaussian_scores


def main():
    # Controlled diagnostic. See configs/synthetic.json for ten trained sklearn families.
    scores, y, names = gaussian_scores()
    indices = np.random.default_rng(0).permutation(len(y))[:1020]
    partial = np.full(1020, -1)
    partial[:20] = y[indices[:20]]
    estimator = SSMEEstimator(random_state=0).fit(scores[indices], partial, model_names=names)
    report = estimator.report(n_draws=100)
    report.show()
    print(report.ranking("accuracy")[["rank", "model", "estimate"]])
    print(report.save(Path(__file__).resolve().parents[1] / "results" / "quickstart"))


if __name__ == "__main__":
    main()
