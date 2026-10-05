# SSME-Lite

## From classifier probabilities to an evaluation report

SSME-Lite helps you evaluate several already trained classifiers when labels are scarce. It combines a small labeled sample with the models' predictions on unlabeled samples, then produces metric estimates, rankings, and an offline report.

**The workflow:** collect probabilities → reveal a few labels → fit `SSMEEstimator` → inspect and export results.

## Start here

1. [Install the package](installation.md).
2. [Run the five-minute quickstart](quickstart.md), with no downloads or model weights.
3. [Connect your own classifiers](guides/inputs.md).
4. [Read the report](guides/reports.md) and choose your target metric.

## Learn through real experiments

| Tutorial | Role | Inputs |
| --- | --- | --- |
| [CivilComments](tutorials/civilcomments.md) | Original-paper quickstart and reproduction check | Released classifier scores |
| [PneumoniaMNIST](tutorials/pneumoniamnist.md) | Main end-to-end demonstration | Images and seven released models, or saved probabilities |
| [Transfer learning](tutorials/transfer.md) | Adapt the workflow to newly trained heads | Frozen ImageNet backbone and five sklearn heads |

The core package consumes probabilities. Training models and downloading datasets belong to the tutorials, so you can use the estimator without a deep-learning framework.

## Explore further

- [Command-line evaluation](guides/cli.md): run from an NPZ file and JSON configuration.
- [Method and interpretation](method.md): algorithm, metrics, intervals, and model influence.
- [API reference](api.md): public classes, defaults, inputs, and outputs.
- [Troubleshooting](troubleshooting.md): common setup and data-alignment problems.
- [Saved interactive reports](https://jasonshen2002.github.io/ssme-lite/): inspect results before running experiments.
