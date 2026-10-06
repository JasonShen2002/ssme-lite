# Transfer learning with a frozen backbone

**Goal:** connect a pretrained feature extractor and newly trained classifiers to the same SSME interface.

This tutorial uses a frozen ImageNet ResNet18 to extract 512-dimensional features. Five sklearn heads are trained on the official PneumoniaMNIST training split: extra trees, histogram gradient boosting, logistic regression, MLP, and random forest. The test images supply the evaluation pool; they are not used to train the heads.

## Run the demonstration

From the source checkout:

```bash
python -m pip install -e '.[transfer]'
python experiments/transfer/transfer_ssme.py
```

The first run downloads the dataset and backbone and caches extracted features in `experiments/transfer/data/`. Subsequent runs reuse them. This is a heavier tutorial than the synthetic quickstart and requires network access for missing assets.

The script trains the heads, saves `data/transfer_scores.npz`, collects predictions through `PredictionMatrixGenerator`, reveals 20 labels among 420 estimation samples, and writes `experiments/transfer/results/demo_report/report.html`.

## Understand the integration

The package-facing steps remain the same regardless of feature extractor:

```python
generator = PredictionMatrixGenerator(fitted_heads)
scores = generator.fit_transform(test_features)
estimator.fit(scores[parts["estimation"]], partial_labels,
              model_names=generator.model_names_)
report = estimator.report(primary_metric="accuracy")
```

This fragment illustrates the interface inside the runnable script. Training, preprocessing, and feature extraction happen before SSME. Replacing the backbone does not require rewriting the estimator.

## Repeated evaluation

```bash
python experiments/transfer/transfer_models.py --suite
```

This performs the repository's repeated comparisons and label-budget study, which cost more than a single report. Saved results are under `experiments/transfer/results/`. With 20 visible labels, mean held-out Accuracy MAE across the saved ten-seed runs falls from 7.11 to 4.16 percentage points. A report's conditional intervals and variation across repeated splits describe different quantities.

## Your own image dataset

`experiments/transfer/transfer_resnet.py` accepts an ImageFolder directory with one subdirectory per class:

```bash
python experiments/transfer/transfer_resnet.py \
  --images /path/to/images --output results/custom_transfer --device cpu
```

This separate example uses ten classifier families, a 40% training split, and a remaining evaluation pool that must exceed 1,020 images (more than roughly 1,700 total). It is not the five-head PneumoniaMNIST protocol. For smaller datasets, use your own training split and the [core input interface](../guides/inputs.md) with an appropriate annotation budget.
