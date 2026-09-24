# Experiments

These scripts are not part of the installed `ssme-lite` package. They show the package on PneumoniaMNIST. Downloaded images, weights and feature caches stay in each experiment's `data/` directory and are not committed.

## Official checkpoints

```bash
python experiments/pneumoniamnist/pneumonia_ssme.py
ssme-lite experiments/pneumoniamnist/config.json
```

`pneumonia_ssme.py` loads seven released checkpoints, runs them through `PredictionMatrixGenerator`, then `SemiSupervisedSplit` and `SSMEEstimator`. AutoML Vision weights need `pip install ai-edge-litert`.

## Transfer learning

```bash
pip install -e '.[transfer]'
python experiments/transfer/transfer_ssme.py
python experiments/transfer/transfer_models.py --suite
```

`transfer_ssme.py` freezes an ImageNet ResNet18, fits five sklearn heads on the official training split, and evaluates them on the test split with the same package API. The first run downloads the backbone.

Saved comparisons are under each experiment's `results/`.
