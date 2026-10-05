# Five-minute quickstart

**Goal:** estimate three classifiers' performance and create an offline report, without downloading data.

Install the [core package](installation.md), then save the following as `quickstart.py` and run `python quickstart.py`. The same script is included under `examples/` in the repository.

```python
import numpy as np
from ssme_lite import SSMEEstimator

rng = np.random.default_rng(7)
y = np.tile([0, 1], 120)
rng.shuffle(y)
strength = np.array([0.7, 1.1, 1.5])
logits = (2 * y[:, None] - 1) * strength + rng.normal(size=(len(y), 3))
scores = 1 / (1 + np.exp(-logits))

# Only these 24 labels are passed to the estimator.
y_partial = np.full(len(y), -1, dtype=int)
y_partial[:24] = y[:24]
assert set(y_partial[:24]) == {0, 1}

estimator = SSMEEstimator(max_iter=20, random_state=7)
estimator.fit(scores, y_partial, model_names=["model_a", "model_b", "model_c"])
report = estimator.report(n_draws=50, primary_metric="accuracy")
print(report.ranking("accuracy")[["model", "estimate", "rank"]].to_string(index=False))
print(report.save("results/quickstart"))
```

## What just happened?

`scores` has 240 rows and three columns. Each column contains one model's positive-class probabilities on the same samples. Twenty-four labels are visible; the other 216 are marked `-1`. The synthetic truth is used to construct this teaching example; hidden labels are never passed to `fit()`.

`fit()` estimates class posteriors from these inputs. `report()` evaluates the input classifiers using that posterior. It does not retrain the classifiers.

## Expected output

You should see three ranking rows and the path `results/quickstart/report.html`. Open that file in a browser. The output directory also contains `metrics.csv`, `metadata.json`, `report_data.json`, and four PNG metric plots. A verified run produced:

```text
  model  estimate  rank
model_c  0.922928   1.0
model_b  0.907622   2.0
model_a  0.751096   3.0
```

Exact decimal values can vary with dependency versions.

The synthetic example demonstrates usage, not measured gains on real data. It uses 20 iterations for a quick run; the estimator default is 100.

## Next steps

- [Use your own models or probability matrices](guides/inputs.md).
- [Understand estimates, intervals, and rankings](guides/reports.md).
- [Try CivilComments](tutorials/civilcomments.md) with released research predictions.
