# CivilComments: original-paper quickstart

**Goal:** use the original paper's released predictions to check the evaluation workflow. This is the reproduction entry point; PneumoniaMNIST is the main end-to-end demonstration.

## Prepare the data

Install the package and reference bandwidth dependency:

```bash
python -m pip install 'ssme-lite[official]'
```

Obtain the released inputs from the [official SSME repository](https://github.com/divyashan/SSME), following its data instructions. The loader expects `INPUTS/CivilComments/<model>/labels.npy` and `preds.npy`, for these seven models:

`alg_CORAL`, `alg_ERM`, `alg_IRM`, `alg_ERM_seed1`, `alg_ERM_seed2`, `alg_IRM_seed1`, `alg_IRM_seed2`.

`INPUTS` denotes your downloaded `inputs` directory; it is not included in the PyPI package. All label files must agree row by row.

## Fit and report

Run the following from any directory after updating the input path:

```python
from pathlib import Path
from ssme_lite import SSMEEstimator, SemiSupervisedSplit
from ssme_lite.benchmark import official_scores

inputs = Path("/path/to/SSME/inputs")
scores, y, names = official_scores(inputs, "CivilComments")
splitter = SemiSupervisedSplit(20, 1000, random_state=0)
parts = splitter.indices(len(y))
partial = splitter.partial_labels(y, parts)
estimator = SSMEEstimator(bandwidth="official", max_iter=20, random_state=0)
estimator.fit(scores[parts["estimation"]], partial, model_names=names)
report = estimator.report(n_draws=100, primary_metric="accuracy")
print(report.ranking("accuracy"))
report.save("results/civilcomments")
```

Expected output: seven models per metric, an offline report, and machine-readable exports. Twenty labels are visible and 1,000 samples are unlabeled. The remainder is held out and not fitted. As in every experiment, the observed draw must include both classes.

## Reproduce the saved comparison

The single-split example above demonstrates the API. The repository's multi-seed reproduction uses a separate half-split protocol: 66,891 held-out samples and 50 valid draws with 20 labeled and 1,000 unlabeled samples. Its script expects the original inputs at `../official/SSME/inputs` relative to the checkout root:

```bash
python experiments/civilcomments/reproduce_official_split.py
```

The saved comparison reports SSME mean absolute errors of 2.27, 2.12, 3.28, and 9.41 percentage points for Accuracy, ECE, AUC, and average precision respectively. These are aggregate held-out errors, not the point estimates in a single report. The experiment is a qualitative reproduction check; ECE binning differs from the paper.

[Open the saved report](https://jasonshen2002.github.io/ssme-lite/experiments/civilcomments/civilcomments_report.html). The original notebook is in `experiments/civilcomments/quickstart.ipynb`.
