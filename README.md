# SSME-Lite

**Evaluate multiple classifiers with scarce labels—and explore the results in an interactive report.**

[![PyPI](https://img.shields.io/pypi/v/ssme-lite.svg)](https://pypi.org/project/ssme-lite/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3572A5)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-087F8C.svg)](LICENSE)

SSME-Lite is a scikit-learn-style Python package for semi-supervised model evaluation. Given predictions from multiple classifiers and a small set of known labels, it estimates Accuracy, ROC AUC, AP, and ECE, with model rankings, uncertainty intervals, and model-influence diagnostics. Reports export to interactive HTML, CSV, and JSON; the core package requires no PyTorch.

**Junje Shen · Student ID: 2026211240 · Course Project 8**

## Explore the interactive report

### [Open the PneumoniaMNIST report →](https://jasonshen2002.github.io/ssme-lite/experiments/pneumoniamnist/pneumoniamnist_report.html)

View the rendered HTML directly in your browser—no installation or download needed. The medical-image demonstration evaluates seven pretrained classifiers and presents metric estimates, model comparisons, uncertainty, and influence diagnostics in one report. Generated HTML reports are also self-contained and can be opened offline.

[Read the project paper](report/main.pdf) · [PneumoniaMNIST walkthrough](docs/tutorials/pneumoniamnist.md) · [CivilComments report](https://jasonshen2002.github.io/ssme-lite/experiments/civilcomments/civilcomments_report.html) · [Reproduction guide](docs/reproduction.md)

## Available on PyPI

The package is published on [PyPI](https://pypi.org/project/ssme-lite/) and can be installed with:

```bash
pip install ssme-lite
```

Python 3.10 or newer is required. PyPI provides the published 0.1.0 release; the model-file CLI and additional experiments in this submission are available from the current repository.

## Highlights

- **Predictions to reports:** class-aligned probabilities, semi-supervised estimation, rankings, and interactive diagnostics through a unified Python interface.
- **Evidence from real models:** pretrained PneumoniaMNIST checkpoints, five classifiers built on frozen ResNet18 features, and a CivilComments reproduction study.
- **Scarce-label evaluation:** with 20 visible labels and ten shared splits, mean Accuracy estimation error improves in both medical-model pipelines.
- **Reproducible delivery:** frozen predictions, saved configurations and splits, input hashes, and an independently audited 342-configuration extension study.

| PneumoniaMNIST pipeline | Labels-only Accuracy MAE | SSME-Lite Accuracy MAE |
| --- | ---: | ---: |
| Seven pretrained checkpoints | 5.70 pp | **2.22 pp** |
| Five transfer classifiers | 7.11 pp | **4.16 pp** |

Values are mean errors against held-out truth over ten splits, with 20 visible labels and 400 unlabeled examples. The linked interactive report demonstrates one fitted evaluation; the paper and archived experiment results provide the aggregate comparisons.

## Quickstart

```python
from ssme_lite import SSMEEstimator

# scores: (samples, models), positive-class probabilities for binary tasks
# y_partial: known class indices; -1 denotes an unknown label
estimator = SSMEEstimator(max_iter=100, random_state=0)
estimator.fit(scores, y_partial, model_names=model_names)
report = estimator.report(primary_metric="accuracy")
print(report.ranking("accuracy"))
report.save("results/my_report")
```

Open `results/my_report/report.html` to explore your results. Multiclass scores use shape `(samples, models, classes)`; each class needs at least one visible label. See the [input guide](docs/guides/inputs.md) and [report guide](docs/guides/reports.md).

### Run the repository examples

```bash
git clone https://openi.pcl.ac.cn/jasonshen/SSME-Lite-2026211240.git ssme-lite
cd ssme-lite
python -m pip install -e '.[dev]'
python examples/quickstart.py
```

The example writes `results/quickstart/report.html` without external datasets. To generate a report from fitted model files:

```bash
python examples/model_files.py
ssme-lite check-config results/model_files/config.json
ssme-lite results/model_files/config.json --trust-model-files
```

Use trusted joblib files and specify models, evaluation data, class order, and output location in JSON. See the [CLI guide](docs/guides/cli.md).

## Reproduce the results

Refit the main experiments from the included frozen predictions, without a GPU:

```bash
python reproduction/run.py --jobs 2
```

Audit the extension results or rebuild the paper figures:

```bash
python experiments/extensions/audit_results.py
python report/build_report.py
python report/build_extensions.py
```

[Reproduction guide](docs/reproduction.md) · [Extension commands](experiments/extensions/README.md) · [Validation record](experiments/extensions/validation.json)

## Repository guide

| Directory | Contents |
| --- | --- |
| `ssme_lite/` | Installable Python package |
| `report/` | Paper, LaTeX sources, vector figures, and Overleaf source archive |
| `examples/` | Quickstart and model-file examples |
| `docs/` | API, tutorials, CLI, methods, and reproduction instructions |
| `reproduction/` | Frozen inputs, hashes, reference results, and refitting scripts |
| `experiments/` | Medical-image, transfer, CivilComments, and extension experiments |
| `tests/` | Numerical, interface, reporting, and protocol tests |

## Documentation and citation

[Documentation](docs/index.md) · [API](docs/api.md) · [Method](docs/method.md) · [Contributing](CONTRIBUTING.md)

Based on Shanmugam et al., *Evaluating Multiple Models Using Labeled and Unlabeled Data* (NeurIPS 2025). Please cite the original method when using this project; bibliographic details are in [references.bib](report/references.bib).

Released under the [MIT License](LICENSE).
