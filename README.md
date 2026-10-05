# SSME-Lite

**Evaluate existing classifiers with a few labels and many unlabeled predictions.**

SSME-Lite packages semi-supervised model evaluation into a small Python interface: provide model probabilities and partial labels, estimate performance, rank candidates, and export an interactive HTML report that opens offline.

- **One estimator:** `fit()` followed by `report()`.
- **Four metrics:** Accuracy, ROC AUC, average precision (`auprc`), and ECE.
- **Flexible inputs:** precomputed probabilities or already fitted sklearn-style models.
- **Portable outputs:** HTML, CSV, JSON, and metric plots.
- **Reproducible experiments:** explicit seeds, shared splits, and optional model-influence diagnostics.
- **Lightweight core:** no PyTorch dependency; transfer learning is optional.

## Installation

Python 3.10 or newer:

```bash
python -m pip install ssme-lite
```

To work from a checkout:

```bash
git clone https://github.com/JasonShen2002/ssme-lite.git
cd ssme-lite
python -m pip install -e '.[dev]'
```

## Minimal interface

```python
from ssme_lite import SSMEEstimator

# scores: (samples, models), positive-class probabilities for binary tasks.
# y_partial: one integer label per row; -1 means unknown.
estimator = SSMEEstimator(random_state=0)
estimator.fit(scores, y_partial, model_names=model_names)
report = estimator.report(primary_metric="accuracy")
print(report.ranking("accuracy"))
report.save("results/my_report")
```

Every class must have at least one observed label. For multiclass tasks, use `(samples, models, classes)` probabilities and class indices `0..K-1`.

For a complete example without external data, run `python examples/quickstart.py` from the checkout. It generates a small synthetic dataset, fits SSME, and saves `results/quickstart/report.html`. This example illustrates the interface, not an empirical performance claim.

## Documentation and tutorials

- [Documentation home](https://github.com/JasonShen2002/ssme-lite/blob/main/docs/index.md)
- [Five-minute quickstart](https://github.com/JasonShen2002/ssme-lite/blob/main/docs/quickstart.md)
- [Use your own models](https://github.com/JasonShen2002/ssme-lite/blob/main/docs/guides/inputs.md)
- [Understand the report](https://github.com/JasonShen2002/ssme-lite/blob/main/docs/guides/reports.md)
- [API reference](https://github.com/JasonShen2002/ssme-lite/blob/main/docs/api.md)

| Tutorial | What you will learn |
| --- | --- |
| [CivilComments](https://github.com/JasonShen2002/ssme-lite/blob/main/docs/tutorials/civilcomments.md) | Quickstart on the original paper's dataset and a qualitative reproduction check |
| [PneumoniaMNIST](https://github.com/JasonShen2002/ssme-lite/blob/main/docs/tutorials/pneumoniamnist.md) | Main end-to-end demonstration: prediction, evaluation, ranking, and reporting |
| [Transfer learning](https://github.com/JasonShen2002/ssme-lite/blob/main/docs/tutorials/transfer.md) | Frozen ResNet18 features, five classifier heads, and a shared evaluation interface |

[View saved experiment reports](https://jasonshen2002.github.io/ssme-lite/).

## Method and interpretation

SSME-Lite follows Shanmugam et al., *Evaluating Multiple Models Using Labeled and Unlabeled Data*, NeurIPS 2025. It combines classifier probabilities through additive log-ratio features, weighted class-conditional kernel densities, and iterative posterior updates with observed labels clamped.

Report intervals describe latent-label uncertainty conditional on the fitted posterior. Model contribution measures sensitivity to leaving out an input model. See the [method guide](https://github.com/JasonShen2002/ssme-lite/blob/main/docs/method.md) for definitions, implementation choices, and attribution. The current real-data demonstrations are binary classification tasks; the interface also accepts multiclass probabilities.

## Development

```bash
python -m pip install -e '.[dev,docs]'
python -m pytest -q
mkdocs serve
```

See [Contributing](https://github.com/JasonShen2002/ssme-lite/blob/main/CONTRIBUTING.md) and the [release guide](https://github.com/JasonShen2002/ssme-lite/blob/main/docs/releasing.md).

MIT licensed. Maintained by Junje Shen. Please cite the original SSME paper when using the method.
