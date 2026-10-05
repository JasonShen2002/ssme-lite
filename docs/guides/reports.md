# Understand the report

```python
report = estimator.report(n_draws=100, primary_metric="accuracy")
report.ranking("accuracy")
report.table
report.save("results/evaluation")
```

## Metrics and rankings

| Key | Quantity | Better direction |
| --- | --- | --- |
| `accuracy` | Fraction of correct class predictions | Higher |
| `auc` | ROC area; macro one-vs-rest for multiclass | Higher |
| `auprc` | Average precision; macro average for multiclass | Higher |
| `ece` | Expected calibration error, 10 equal-width bins | Lower |

`auprc` is computed with sklearn's average precision, not trapezoidal integration of a precision–recall curve. Binary ECE uses positive-class probabilities; multiclass ECE uses top-label confidence. Ranking uses the point estimate and assigns the same minimum rank to ties. Choose the metric appropriate to your task before comparing candidates.

## Table columns

`model`, `metric`, and `estimate` identify each estimate. `interval_low` and `interval_high` are conditional latent-label quantiles at the requested confidence level. `label_std` is the standard deviation across valid label draws; `mc_standard_error` quantifies numerical Monte Carlo error. `valid_draws` records how many finite metric values were available.

Accuracy uses its exact conditional expectation, so its point estimate has zero Monte Carlo error. Other metric estimates average valid draws. Accuracy intervals are still obtained from sampled labels. AUC/AP draws missing a class are undefined and omitted; inspect `valid_draws` for small or imbalanced targets.

Intervals condition on a fitted posterior. They do not include density-fit, bandwidth, or population sampling uncertainty. Increasing `n_draws` improves numerical precision; it does not account for those additional sources.

## Files you can share

| File | Purpose |
| --- | --- |
| `report.html` | Interactive report, with embedded assets and data; opens offline |
| `metrics.csv` | Metric table for spreadsheets or further analysis |
| `metadata.json` | Fit settings, target, seed, and interpretation notes |
| `report_data.json` | Structured data powering the report |
| Metric PNGs | Static plots for presentations |
| `contribution.csv` | Produced when model contribution is enabled |

The report summarizes prediction and posterior data. Inspect its contents before sharing outputs from sensitive datasets.

## Model influence

```python
report = estimator.report(contribution=True)
report.save("results/with_contribution")
```

This refits once per removed model, holding visible labels, scalar bandwidth, iteration count, and full-fit initialization fixed. Total variation measures the change in the unlabeled posterior; the report also shows changes to the remaining models' Accuracy estimates. It describes conditional influence, not a causal effect or a guaranteed benefit from dropping a model. Runtime increases by roughly one fit per candidate.
