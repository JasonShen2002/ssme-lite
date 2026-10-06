"""Self-contained quickstart. Run from any directory after installing ssme-lite."""
from pathlib import Path
import json
import numpy as np
from ssme_lite import SSMEEstimator


def main():
    rng = np.random.default_rng(7)
    y = np.tile([0, 1], 120)
    rng.shuffle(y)
    strength = np.array([0.7, 1.1, 1.5])
    logits = (2 * y[:, None] - 1) * strength + rng.normal(size=(len(y), 3))
    scores = 1 / (1 + np.exp(-logits))
    partial = np.full(len(y), -1, dtype=int)
    partial[:24] = y[:24]
    assert set(partial[:24]) == {0, 1}
    names = ["model_a", "model_b", "model_c"]
    estimator = SSMEEstimator(max_iter=20, random_state=7)
    estimator.fit(scores, partial, model_names=names)
    report = estimator.report(n_draws=50, primary_metric="accuracy")
    print(report.ranking("accuracy")[["model", "estimate", "rank"]].to_string(index=False))
    output = Path("results/quickstart")
    print(report.save(output))
    np.savez(output / "predictions.npz", scores=scores, y=partial,
             model_names=np.asarray(names, dtype=str))
    config = {"source": "npz", "mode": "evaluate", "data": "predictions.npz",
              "output": "cli_report", "estimator": {"max_iter": 20, "random_state": 7},
              "report": {"n_draws": 50, "primary_metric": "accuracy"}}
    (output / "evaluation.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
