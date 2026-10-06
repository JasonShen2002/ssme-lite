# API reference

The stable public imports are:

```python
from ssme_lite import (
    SSMEEstimator, PredictionMatrixGenerator,
    SemiSupervisedSplit, EvaluationReport,
)
```

## SSMEEstimator

```python
SSMEEstimator(density="kde", bandwidth="scott", max_iter=100,
              tol=1e-3, labeled_weight=10.0, clip=1e-6,
              random_state=0, n_jobs=1, batch_size=2048,
              early_stopping=False)
```

| Parameter | Meaning |
| --- | --- |
| `density` | Currently only `"kde"` is supported |
| `bandwidth` | `"scott"`, `"official"`, or a positive scalar in ALR space |
| `max_iter` | Positive integer iteration count or early-stopping limit |
| `tol` | Positive threshold for maximum posterior change |
| `labeled_weight` | Positive weight on observed samples; unlabeled weight is 1 |
| `clip` | Probability clipping threshold, strictly between 0 and 0.5 |
| `random_state` | Seed for initialization and default report sampling |
| `n_jobs` | joblib parallelism for class-density fitting; 1 is serial |
| `batch_size` | Positive batch size for posterior evaluation |
| `early_stopping` | Whether to stop when posterior change falls below `tol` |

### fit

```python
fit(scores, y, scores_unlabeled=None, model_names=None,
    initial_responsibilities=None)
```

Returns the fitted estimator. `scores` is binary `(N, M)` or full `(N, M, K)`. `y` has shape `(N,)`, contains class indices or `-1`, and includes every class among observed entries. If supplied, `scores_unlabeled` is appended after the first array. `model_names` contains unique names for the model axis; default names are `model_0`, etc.

Advanced: `initial_responsibilities` is a nonnegative normalized `(N_total, K)` posterior used for controlled initialization. Observed labels are still clamped. Ordinary use should leave it unset.

Invalid shapes, probabilities, labels, unsupported density choices, or duplicate names raise `ValueError`. The `official` bandwidth raises `ImportError` without statsmodels.

### Fitted attributes

| Attribute | Meaning |
| --- | --- |
| `posterior_` | `(N_total, K)` latent-class probabilities, with observed rows clamped |
| `scores_`, `y_` | Normalized full probability tensor and aligned partial labels |
| `model_names_`, `classes_` | Candidate names and class indices |
| `labeled_indices_`, `unlabeled_indices_` | Indices in the combined fitted pool |
| `bandwidth_`, `priors_` | Effective scalar bandwidth and fitted class priors |
| `history_` | Per-iteration posterior-change, objective, and class summaries |
| `n_iter_`, `converged_`, `stop_reason_` | Actual iterations, tolerance status, and stopping reason |

### predict_proba

```python
predict_proba(scores)
```

Returns `(N_new, K)` posterior probabilities from the fitted densities. Input model and class axes must match the fit. This returns inferred labels for samples, not per-model performance. It does not clamp labels for new rows. Use `posterior_` when you need the fitted pool with observed-label clamping.

### evaluate / report

```python
evaluate(metrics=("accuracy", "auc", "auprc", "ece"), n_draws=100,
         confidence=0.95, target="all", random_state=None,
         title=None, sample_ids=None, primary_metric=None,
         contribution=False)
report(**kwargs)
```

Returns `EvaluationReport`; `report` is an alias for `evaluate`. Metrics must be distinct supported keys. `n_draws >= 2`, `0 < confidence < 1`, and target is `"all"` or `"unlabeled"`. The default report seed inherits the estimator seed. `sample_ids`, if provided, identify all fitted rows. `primary_metric` must be among the requested metrics and defaults to the first. Contribution requires at least two models and some unlabeled samples.

### contribution

`contribution(n_draws=30)` returns a pandas DataFrame of leave-one-model-out diagnostics. `n_draws` is retained for compatibility; this diagnostic uses exact expected Accuracy and does not sample labels. See [interpretation](guides/reports.md).

## PredictionMatrixGenerator

```python
PredictionMatrixGenerator(models, clip=1e-6, n_jobs=1)
```

`models` is a mapping of names to already fitted classifiers, or a sequence assigned automatic names. Each classifier must expose `classes_` and `predict_proba(X)`.

- `fit(X=None, y=None)` inspects models and aligns class columns; returns self.
- `transform(X, y_partial=None)` collects probabilities; optional labels record labeled/unlabeled indices.
- `fit_transform(X, y=None)` combines the two steps without training classifiers.

Attributes include `model_names_`, `classes_`, and, after transformation, `sample_ids_`. Binary output is `(N, M)` for `classes_[1]`; multiclass output is `(N, M, K)`.

## SemiSupervisedSplit

```python
SemiSupervisedSplit(n_labeled, n_unlabeled, random_state=0)
```

Both counts must be positive integers. `indices(n_samples)` returns `labeled`, `unlabeled`, `estimation`, and `heldout` index arrays, requiring a nonempty held-out set. `partial_labels(y, parts)` returns the partial-label array aligned to `estimation`. `labeled_class_count(y, parts)` counts represented classes. Labels do not affect the permutation.

## EvaluationReport

Usually obtain this object from `estimator.report()`.

- `table`: pandas DataFrame of estimates and uncertainty summaries.
- `metadata`: settings and interpretation notes.
- `data`: structured interactive-report payload.
- `ranking(metric="accuracy")`: sorted DataFrame with a `rank` column; ECE ascending, other metrics descending.
- `show()`: print estimates and fit status; return self.
- `plot(metric="accuracy", path=None)`: return a Matplotlib figure and optionally save it.
- `save(directory)`: create outputs and return the HTML `pathlib.Path`.

See the [report guide](guides/reports.md) for table fields and saved-file definitions.
