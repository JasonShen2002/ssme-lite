# SSME-Lite

[![PyPI](https://img.shields.io/pypi/v/ssme-lite.svg)](https://pypi.org/project/ssme-lite/)
[![Tests](https://github.com/JasonShen2002/ssme-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/JasonShen2002/ssme-lite/actions/workflows/ci.yml)

**Evaluate existing classifiers with a few labels and many unlabeled predictions.**

SSME-Lite packages semi-supervised model evaluation into a small Python interface: provide model probabilities and partial labels, estimate performance, rank candidates, and export an interactive HTML report that opens offline.

- **One estimator:** `fit()` followed by `report()`.
- **Four metrics:** Accuracy, ROC AUC, average precision (`auprc`), and ECE.
- **Flexible inputs:** precomputed probabilities or already fitted sklearn-style models.
- **Portable outputs:** HTML, CSV, JSON, and metric plots.
- **Reproducible experiments:** explicit seeds, shared splits, and optional model-influence diagnostics.
- **Lightweight core:** no PyTorch dependency; transfer learning is optional.
- **Parallel execution:** configurable prediction, density-fit, and benchmark workers, with batched posterior evaluation.

## Published release

**SSME-Lite 0.1.0 is available on [PyPI](https://pypi.org/project/ssme-lite/0.1.0/).** The release was built and uploaded through GitHub Actions Trusted Publishing. Automated checks passed on Python 3.10–3.14; a fresh installation from public PyPI was verified with both runnable examples and the CLI report workflow. [Release run](https://github.com/JasonShen2002/ssme-lite/actions/runs/37307075613).

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

## Experimental evidence

PneumoniaMNIST is the main study: seven released checkpoints demonstrate the complete inference-to-report workflow, and five frozen-ResNet18 transfer heads demonstrate reuse with newly trained classifiers. With 20 visible labels, mean held-out Accuracy estimation error decreases from **5.72 to 2.26 percentage points** for checkpoints and from **7.11 to 4.16** for transfer heads, over ten shared splits.

CivilComments is the original-paper quickstart and qualitative reproduction check. These are measured results for the stated protocols, rather than guarantees for every dataset. See the [current experiment summary](https://github.com/JasonShen2002/ssme-lite/blob/main/experiments/REPORT.md) and [course-requirement evidence](https://github.com/JasonShen2002/ssme-lite/blob/main/docs/course-requirements.md).

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
