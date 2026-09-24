# Experiments

These scripts are not installed with the package. Images, weights, and feature caches stay in each experiment's `data/` directory and are not committed.

## PneumoniaMNIST

The walkthrough is [`pneumoniamnist.ipynb`](../pneumoniamnist.ipynb). The saved interactive report is [`pneumoniamnist_report.html`](../pneumoniamnist_report.html).

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
