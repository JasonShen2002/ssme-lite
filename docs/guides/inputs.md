# Use your own predictions

## Input contract

| Input | Shape / values | Meaning |
| --- | --- | --- |
| Binary scores | `(N, M)`, probabilities in `[0, 1]` | Probability of class index 1 |
| Full probabilities | `(N, M, K)` | Class probabilities summing to one on the last axis |
| Partial labels | `(N,)`, `-1` or integers `0..K-1` | Unknown labels or observed class indices |
| Model names | `M` unique strings | Names shown in tables and reports |

All models must predict the same samples in the same order and the same class set. Every class needs at least one observed label. Pass probabilities, not logits. Inputs are copied, clipped away from zero and one, and normalized internally.

## Runnable sklearn example

From the checkout root, run:

```bash
python examples/sklearn_models.py
```

This self-contained script generates data, trains logistic regression and a random forest on 300 training samples, and collects their predictions on 300 different samples. It then fits SSME with 20 observed labels and 180 unlabeled samples, reserving 100 held-out rows. The output is `results/sklearn_models/report.html`, including model-influence diagnostics. No external data or model weights are required.

## Already fitted classifiers

```python
from ssme_lite import PredictionMatrixGenerator, SSMEEstimator

# Each value is already trained, and exposes classes_ and predict_proba(X).
generator = PredictionMatrixGenerator({"logistic": fitted_logistic, "forest": fitted_forest})
scores = generator.fit_transform(X_evaluation)
estimator = SSMEEstimator(random_state=0).fit(
    scores, y_partial, model_names=generator.model_names_
)
report = estimator.report()
```

The generator aligns probability columns to the first model's `classes_`. Its `fit()` inspects models; it does not train them. Binary output uses `generator.classes_[1]` as the positive class.

If original labels are strings or nonconsecutive numbers, encode observed labels as positions in `generator.classes_`. Reserve `-1` for unknown entries. For example, classes `['healthy', 'pneumonia']` map to `[0, 1]`.

## Separate labeled and unlabeled arrays

```python
estimator.fit(
    labeled_scores, labeled_y,
    scores_unlabeled=unlabeled_scores,
    model_names=model_names,
)
```

This appends the unlabeled rows internally. Model and class dimensions must match. Reports default to all fitted rows; use `report(target="unlabeled")` to evaluate only the unlabeled portion.

## Controlled benchmark splits

When full labels are available for a benchmark, use `SemiSupervisedSplit` to hide them reproducibly:

```python
from ssme_lite import SemiSupervisedSplit

splitter = SemiSupervisedSplit(20, 400, random_state=0)
parts = splitter.indices(len(y))
y_partial = splitter.partial_labels(y, parts)
estimator.fit(scores[parts["estimation"]], y_partial, model_names=model_names)
```

The remaining samples form `parts['heldout']`. They are not used by the estimator. The splitter uses one uniform permutation and does not inspect labels to repair a missing-class draw. Check `splitter.labeled_class_count(y, parts)` when auditing benchmark splits, and document exclusions or additional annotation. Keep classifier training separate from this evaluation pool.
