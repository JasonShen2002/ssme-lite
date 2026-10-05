# PneumoniaMNIST: end-to-end evaluation

**Goal:** follow the main demonstration from image predictions to an SSME report. PneumoniaMNIST is a binary chest-X-ray dataset in MedMNIST, distinct from the handwritten-digit MNIST dataset.

Seven released classifiers are used: ResNet18, three AutoKeras models, and three AutoML Vision models. The official test set contains 624 images. The demonstration uses 20 observed labels, 400 unlabeled images, and 204 held-out images.

## Choose your starting point

- **Inspect immediately:** [open the saved report](https://jasonshen2002.github.io/ssme-lite/experiments/pneumoniamnist/pneumoniamnist_report.html).
- **Use saved predictions:** run the core-only example below with a previously generated NPZ file.
- **Run image inference:** use the repository script and optional dependencies in the next section.

## Evaluate saved predictions

The inference script writes `experiments/pneumoniamnist/data/inferred_test_scores.npz`. This file contains `scores`, `y`, and `model_names`; it is generated locally and not included in the wheel. With that file available:

```python
import numpy as np
from ssme_lite import SSMEEstimator, SemiSupervisedSplit

with np.load("experiments/pneumoniamnist/data/inferred_test_scores.npz") as data:
    scores, y = data["scores"], data["y"]
    names = data["model_names"].tolist()
splitter = SemiSupervisedSplit(20, 400, random_state=0)
parts = splitter.indices(len(y))
estimator = SSMEEstimator(random_state=0).fit(
    scores[parts["estimation"]], splitter.partial_labels(y, parts),
    model_names=names,
)
report = estimator.report(primary_metric="accuracy", contribution=True)
report.save("results/pneumoniamnist")
```

This path does not need PyTorch or TFLite. `contribution=True` adds seven refits and can be omitted for a faster first run.

## Run inference from images

From the source checkout:

```bash
python -m pip install -e '.[transfer]'
python -m pip install ai-edge-litert
python experiments/pneumoniamnist/pneumonia_ssme.py
```

The first run downloads the dataset and released weights; allow network access and additional disk space. The script runs the seven models, constructs their probability matrix, fits the estimator, and writes `experiments/pneumoniamnist/results/demo_report/report.html` plus a copy beside the notebook.

Optional published-score CSVs belong in `data/pneumoniamnist/official_test_csv/`. If available, the script compares predictions with those files; otherwise it reports that the optional audit is skipped. TFLite predictions can depend on image resizing, so matching aggregate AUC is distinct from matching every probability.

## Read the result

The report estimates performance on the 420-sample estimation pool. The saved ten-seed benchmark in `experiments/pneumoniamnist/results/nl20_nu400_m7/` separately compares estimates with held-out ground truth. Across those runs, mean Accuracy MAE falls from 5.72 to 2.26 percentage points. Keep the target and aggregation explicit when quoting results.

The notebook `experiments/pneumoniamnist/pneumoniamnist.ipynb` and experiment audit provide additional context. Continue with [transfer learning](transfer.md) to evaluate heads you train yourself.
