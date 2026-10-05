# Experiment tutorials

Start with the English walkthroughs:

- [CivilComments: original-paper quickstart](../docs/tutorials/civilcomments.md)
- [PneumoniaMNIST: end-to-end evaluation](../docs/tutorials/pneumoniamnist.md)
- [Transfer learning with frozen ResNet18](../docs/tutorials/transfer.md)

For a first run without data downloads, use `python examples/quickstart.py` from the repository root. The experiment scripts below are source-checkout tools, not installed CLI commands.

These scripts are not installed with the package. Images, weights, and feature caches stay in each experiment's `data/` directory and are not committed.

## CivilComments

The walkthrough is [`civilcomments/quickstart.ipynb`](civilcomments/quickstart.ipynb). The saved interactive report is [`civilcomments/civilcomments_report.html`](https://jasonshen2002.github.io/ssme-lite/experiments/civilcomments/civilcomments_report.html). The notebook also writes `civilcomments/results/report/report.html`. The comparison figures next to that report use this package's `SSMEEstimator` on the paper's half-split: 50 seeds, 20 labels, 1,000 unlabeled comments, `bandwidth="official"`, and 20 EM epochs. Predictions come from `../official/SSME/inputs`.

```bash
python experiments/civilcomments/reproduce_official_split.py
```

## PneumoniaMNIST

The walkthrough is [`pneumoniamnist/pneumoniamnist.ipynb`](pneumoniamnist/pneumoniamnist.ipynb). The saved interactive report is [`pneumoniamnist/pneumoniamnist_report.html`](pneumoniamnist/pneumoniamnist_report.html).

```bash
python experiments/pneumoniamnist/pneumonia_ssme.py
```

`pneumonia_ssme.py` loads seven released checkpoints, runs them through `PredictionMatrixGenerator`, then `SemiSupervisedSplit` and `SSMEEstimator`. AutoML Vision weights need `pip install ai-edge-litert`. The 10-seed comparison is already in `experiments/pneumoniamnist/results/nl20_nu400_m7/`. Model audit notes are in `experiments/pneumoniamnist/AUDIT.md`. The write-up is `experiments/REPORT.md`.

## Transfer learning

```bash
pip install -e '.[transfer]'
python experiments/transfer/transfer_ssme.py
python experiments/transfer/transfer_models.py --suite
```

`transfer_ssme.py` freezes an ImageNet ResNet18, fits five sklearn heads on the official training split, and evaluates them on the test split with the same package API. `transfer_resnet.py` is the same idea for any ImageFolder. The first run downloads the backbone.
