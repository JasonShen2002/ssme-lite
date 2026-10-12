# SSME-Lite

**From scarce labels to model rankings and interactive evaluation reports.**

[![PyPI](https://img.shields.io/pypi/v/ssme-lite.svg)](https://pypi.org/project/ssme-lite/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3572A5)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-087F8C.svg)](LICENSE)

A Python implementation of Semi-Supervised Model Evaluation (SSME): combine predictions from multiple classifiers with a small labeled sample to estimate performance, compare models, and export an interactive report.

**Junje Shen · Student ID: 2026211240 · Course Project 8**

## Explore the submission

| Start here | What you will find |
| --- | --- |
| **[1. Open the interactive PneumoniaMNIST report →](https://jasonshen2002.github.io/ssme-lite/experiments/pneumoniamnist/pneumoniamnist_report.html)** | Seven pretrained models; metrics, rankings, uncertainty, and influence diagnostics. Opens directly in your browser. |
| **[2. Read the project paper →](report/main.pdf)** | Method, software design, experimental results, and figures. Four-page body, references, and reproduction appendix. |
| **[3. Follow the tutorials →](#tutorials)** | Runnable examples for original-paper reproduction, medical-image evaluation, and transfer learning. |
| **[4. Reproduce the results →](docs/reproduction.md)** | Included prediction artifacts, exact commands, saved reference results, and verification steps. |

## What the package provides

**Fitted models + data → prediction matrix → SSME estimation → model-selection report**

| Component | Implementation | Documentation |
| --- | --- | --- |
| **Prediction matrix generator** | Accepts fitted sklearn-style classifiers, aligns class columns, and collects binary or multiclass probabilities. | [Inputs and models](docs/guides/inputs.md) |
| **SSME estimator** | A compact `SSMEEstimator` class with weighted kernel-density updates and observed-label clamping. | [Method](docs/method.md) · [API reference](docs/api.md) |
| **Model-selection report** | Accuracy, ROC AUC, AP, and ECE estimates; rankings, conditional uncertainty intervals, visual comparisons, and HTML/CSV/JSON export. | [Report guide](docs/guides/reports.md) |
| **Model-influence analysis** | Controlled leave-one-model-out refits quantify changes in the posterior and remaining models' estimates. | [Influence interpretation](docs/method.md) |
| **Configurable execution** | Parallel prediction and fitting, batched evaluation, and measured CPU scaling across 500–5,000 samples. | [Parallelism](docs/guides/performance.md) |
| **Model files to report** | JSON configuration specifies data paths and named fitted models; preflight checks, input hashes, and a CLI complete the workflow. | [CLI tutorial](docs/guides/cli.md) |

The [course-requirement mapping](docs/course-requirements.md) links each assignment item to its implementation and experimental evidence.

## Install and try it

The package is **published on [PyPI](https://pypi.org/project/ssme-lite/)**:

```bash
pip install ssme-lite
```

Requires Python 3.10+. The core package runs without PyTorch. Version **0.2.0** includes the fitted-model CLI. Clone the repository for tutorials, frozen inputs, and experiment scripts:

```bash
git clone https://openi.pcl.ac.cn/jasonshen/SSME-Lite-2026211240.git ssme-lite
cd ssme-lite
python -m pip install -e '.[dev]'
python examples/quickstart.py
```

Open `results/quickstart/report.html`. This example generates its own data and needs no model or dataset downloads.

### Use the Python interface

```python
from ssme_lite import SSMEEstimator

# scores: (samples, models), binary positive-class probabilities
# y_partial: known class indices; -1 marks an unknown label
estimator = SSMEEstimator(max_iter=100, random_state=0)
estimator.fit(scores, y_partial, model_names=model_names)
report = estimator.report(primary_metric="accuracy")
print(report.ranking("accuracy"))
report.save("results/my_report")
```

Multiclass scores have shape `(samples, models, classes)`; each class needs at least one visible label. See the [complete quickstart](docs/quickstart.md) for runnable inputs and the [installation guide](docs/installation.md) for optional dependencies.

## Tutorials

| Tutorial | What it demonstrates | Open the materials |
| --- | --- | --- |
| **CivilComments: reproduce the original setting** | Seven released prediction vectors, 20 visible labels, 1,000 unlabeled samples, and comparisons with labeled-only and aggregation baselines. | **[Tutorial](docs/tutorials/civilcomments.md)** · [Notebook](experiments/civilcomments/quickstart.ipynb) · [Live report](https://jasonshen2002.github.io/ssme-lite/experiments/civilcomments/civilcomments_report.html) |
| **PneumoniaMNIST: images to evaluation** | Run seven pretrained checkpoints, align predictions, fit SSME, and inspect the report. | **[Tutorial](docs/tutorials/pneumoniamnist.md)** · [Notebook](experiments/pneumoniamnist/pneumoniamnist.ipynb) · [Live report](https://jasonshen2002.github.io/ssme-lite/experiments/pneumoniamnist/pneumoniamnist_report.html) |
| **Transfer learning: pretrained features to new classifiers** | Extract 512-dimensional features with a frozen ImageNet ResNet18, train five sklearn heads, and evaluate them with scarce labels. | **[Tutorial](docs/tutorials/transfer.md)** · [Runnable script](experiments/transfer/transfer_ssme.py) |
| **Your own fitted models: configuration to report** | Prepare trusted joblib models and evaluation data, validate a JSON configuration, and generate a report through the CLI. | **[Tutorial](docs/guides/cli.md)** · [Runnable example](examples/model_files.py) |

## Experimental evidence

The main study compares SSME-Lite with evaluation using only the observed labels. Both methods share the same splits and are scored against held-out ground truth.

| PneumoniaMNIST pipeline | Labels-only Accuracy MAE | SSME-Lite Accuracy MAE | Relative reduction |
| --- | ---: | ---: | ---: |
| Seven pretrained checkpoints | 5.70 pp | **2.22 pp** | **61.1%** |
| Five transfer classifiers | 7.11 pp | **4.16 pp** | **41.4%** |

Each row averages ten splits with **20 visible labels and 400 unlabeled examples**. All twelve candidate models improve in mean Accuracy estimation error. The live reports illustrate individual fits; the [paper](report/main.pdf) presents the aggregate comparisons.

Further evidence includes a **50-split CivilComments study** and **342 extension configurations** covering label budgets, CPU scaling, parallel consistency, multiclass inputs, and diverse or duplicated model views. The [extension guide](docs/guides/extensions.md) explains the protocols and connects them to the archived results.

## Reproduce and verify

Run these commands from the repository root:

| Purpose | Command |
| --- | --- |
| Refit the main studies from included frozen predictions; no GPU required | `python reproduction/run.py --jobs 2` |
| Independently audit the saved extension results | `python experiments/extensions/audit_results.py` |
| Rebuild the main paper tables and figures | `python report/build_report.py` |
| Rebuild the four-panel experimental figure | `python report/build_extensions.py` |
| Run the package tests | `python -m pytest -q` |

The current checkout passed **48 tests**, with one optional official-code audit skipped, and was also checked as an installed wheel. See the [validation record](experiments/extensions/validation.json). The [official-code execution record](docs/evidence/official-execution.json) documents runs of the original SSME implementation; the [reproduction guide](docs/reproduction.md) covers the final package experiments and input preparation.

## Project materials

| Location | Contents |
| --- | --- |
| [`ssme_lite/`](ssme_lite/) | Installable package: prediction, estimation, benchmarking, and reporting |
| [`docs/`](docs/index.md) | Tutorials, API, method, CLI, and troubleshooting |
| [`examples/`](examples/) | Small runnable examples |
| [`experiments/`](experiments/README.md) | Real-data workflows, saved outputs, and extension studies |
| [`reproduction/`](reproduction/) | Frozen inputs, hashes, reference results, and refitting scripts |
| [`report/`](report/) | Paper, LaTeX sources, vector figures, and [Overleaf source archive](report/ssme_lite_report_source.zip) |
| [`tests/`](tests/) | Numerical, interface, reporting, and experiment-protocol checks |

## Attribution and license

Based on Shanmugam et al., *Evaluating Multiple Models Using Labeled and Unlabeled Data* (NeurIPS 2025). Please cite the original method; full references are available in [references.bib](report/references.bib).

Released under the [MIT License](LICENSE). See [Contributing](CONTRIBUTING.md) for development guidance.
