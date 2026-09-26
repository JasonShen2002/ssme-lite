# SSME-Lite experiment report

Semi-supervised model evaluation on PneumoniaMNIST. The comparison is between a labeled-only estimate and SSME-Lite. Transfer learning is a separate chain: a frozen ImageNet ResNet18 extracts features, and sklearn heads classify them.

## 1. Question

Accuracy, AUC, AUPRC, and ECE are usually computed on a large labeled set. Pediatric chest X-rays are the opposite: labels are scarce and unlabeled images are plentiful. SSME (Shanmugam et al., NeurIPS 2025) uses predicted probabilities from several classifiers, together with a few true labels, to estimate how those classifiers perform on unlabeled data.

SSME-Lite turns that procedure into a callable pipeline rather than a new core algorithm:

`model probabilities → PredictionMatrixGenerator → SSMEEstimator → ranks, intervals, and a report`

This report answers three questions. Can the official PneumoniaMNIST models form a diverse classifier set? On a shared split, is the absolute error of SSME-Lite lower than the labeled-only estimate? Can features from a pretrained ResNet enter the same pipeline?

## 2. Method

The estimator follows the pipeline already implemented in the package. Class probabilities from several models are mapped with an additive log-ratio transform. A weighted Gaussian kernel density estimate is fit in that space, and EM updates class responsibilities for unlabeled samples. Observed labels stay one-hot at every E-step. The default run is a fixed 100 iterations, labeled samples have weight 10, and the bandwidth uses the Scott scale. Accuracy uses the posterior expectation. AUC, AUPRC, and ECE use latent-label draws. These choices match the other benchmarks in this repository. They are not a line-by-line reproduction of the paper's improved Sheather-Jones bandwidth.

The split is `SemiSupervisedSplit`. One uniform random permutation cuts the evaluation pool into three parts:

- Model training data: already used by the classifiers, and not used again by SSME
- Estimation set: `n_l` visible labels plus `n_u` samples that contribute predictions only
- Held-out: labels never enter SSME, and are used for ground truth

A seed depends only on the sample counts, so model-count ablations compare the same indices. The error is

\[
|\widehat{\mathrm{Metric}} - \mathrm{Metric}_{GT}|
\]

Means are taken over 10 seeds. A Student-t interval describes uncertainty in that mean. The interval covers the random split only, not uncertainty in the density estimate itself.

## 3. Setup

### 3.1 Data and official models

PneumoniaMNIST is a binary chest X-ray task: 4,708 train, 524 validation, and 624 test images. The test split has 234 normal and 390 pneumonia cases. Official classifiers are trained on train and use validation for early stopping, so the evaluation pool is test only. The test split has 624 images, which cannot hold the paper's 1,000 unlabeled samples. The protocol is

\[
n_l \in \{20, 50, 100\}, \quad n_u = 400
\]

Held-out size is 204 when `n_l = 20` and 124 when `n_l = 100`.

Models come from the MedMNIST v2 released test predictions (Zenodo `10.5281/zenodo.7782114`), not from five copies of one checkpoint. Each family keeps repeat 1 only. Recomputed AUC and Accuracy match the three decimals in the filenames, and scores lie in `[0, 1]`.

| Model | Architecture | Test AUC | Test ACC |
| --- | --- | ---: | ---: |
| `resnet18_28_1` | ResNet-18, 28×28 | 0.948 | 0.833 |
| `resnet50_224_1` | ResNet-50, 224×224 | 0.967 | 0.896 |
| `autokeras_1` | AutoKeras | 0.939 | 0.883 |
| `autosklearn_1` | auto-sklearn | 0.942 | 0.859 |
| `automl_vision_1` | Google AutoML Vision | 0.993 | 0.941 |

The Accuracy range is 0.107. The highest pairwise disagreement of thresholded predictions is 0.120, and positive-class probability correlations lie in 0.80–0.93, so the scores are not copies of one another. auto-sklearn probabilities sit in 0.38–0.63: ranking remains, and confidence is lower. Its model snapshot was not released, so SSME uses the published probabilities. The full audit is `experiments/pneumoniamnist/AUDIT.md`.

### 3.2 Transfer learning

This line does not load the checkpoints above. An ImageNet-pretrained ResNet18 drops its classification layer and is frozen. Each 28×28 grayscale image is copied to three channels, bilinearly resized to 224, and normalized with ImageNet mean and variance, producing a 512-dimensional feature. Logistic regression, random forest, MLP, extra trees, and histogram gradient boosting are fit only on the official train split. Hyperparameters are fixed in advance. The validation set is not used for selection. SSME still sees only the test split, with the same split protocol as the previous section.

### 3.3 Repeated runs

Each setting uses 10 seeds (0–9). The main comparison is 5 models, `n_l = 20`, and `n_u = 400`. Later runs change only the label count, or only the model count. Nested subsets of 2, 3, and 4 official models are added from lowest to highest prediction correlation. The rule uses scores only, not SSME error. Transfer-learning subsets are specified in advance by head type: logistic regression plus forest, then MLP, extra trees, and histogram gradient boosting.

The main experiment turns on leave-one-model-out sensitivity. Every seed finishes 100 iterations. No split is dropped for a missing class among the visible labels.

## 4. Results

### 4.1 Official models, 20 labels

Mean absolute error on the held-out set. Brackets are 95% Student-t intervals for the mean over 10 seeds.

| Metric | 20 labels only | SSME-Lite |
| --- | ---: | ---: |
| Accuracy | 0.053 [0.040, 0.065] | 0.030 [0.023, 0.037] |
| AUC | 0.032 [0.023, 0.041] | 0.030 [0.022, 0.038] |
| AUPRC | 0.033 [0.021, 0.045] | 0.030 [0.021, 0.038] |
| ECE | 0.038 [0.030, 0.045] | 0.033 [0.024, 0.043] |

The Accuracy intervals do not overlap, and SSME-Lite is closer to the held-out truth. Absolute errors for AUC, AUPRC, and ECE are close, and their intervals overlap.

Ranking is a separate question. Accuracy Spearman correlation moves from 0.71 to 0.78. AUC Spearman correlation falls from 0.40 to about 0, and Top-1 falls from 0.56 to 0: across ten splits, SSME never ranks the held-out AUC leader first. The estimates themselves explain it. True estimation-pool AUC for the five models runs from about 0.94 to 0.99, and SSME places all of them in 0.966–0.978. The strongest model, AutoML Vision (truth about 0.992), is estimated at 0.966, below ResNet-50. Absolute error is small, but the order is reversed. The same conclusion appears on estimation-pool ground truth, so it is not a mismatch between the held-out set and the estimation pool.

### 4.2 Label count and model count

Mean absolute error of held-out Accuracy:

| Setting | Labels only | SSME-Lite |
| --- | ---: | ---: |
| `n_l = 20`, 5 models | 0.053 | 0.030 |
| `n_l = 50`, 5 models | 0.029 | 0.027 |
| `n_l = 100`, 5 models | 0.028 | 0.032 |
| `n_l = 20`, 2 / 3 / 4 / 5 models | 0.056 / 0.055 / 0.052 / 0.053 | 0.028 / 0.029 / 0.028 / 0.030 |

Once the label count rises from 20 to 50, labeled-only Accuracy error falls to the same level as SSME. At 100 labels, labels only is slightly better, and the intervals overlap. SSME Accuracy error barely changes from 2 models to 5 models. AUC Top-1 stays at 0 for 3, 4, and 5 models. Adding another correlated classifier does not automatically improve the AUC ranking.

Leave-one-model-out (mean posterior total variation over 10 seeds) shows that the information is uneven: ResNet-50 is 0.041, AutoKeras and AutoML Vision are about 0.020, ResNet-18 is 0.013, and auto-sklearn is 0.0001. auto-sklearn probabilities sit near 0.5 and barely move the posterior. That is a redundant signal, not a claim that the model cannot predict: its test AUC is still 0.942.

### 4.3 Transfer learning

The five sklearn heads produce different test predictions. On one split, SSME AUC estimates run from 0.950 for extra trees to 0.982 for the MLP. In the repeated experiment, held-out absolute error with `n_l = 20` and 5 heads is:

| Metric | 20 labels only | SSME-Lite |
| --- | ---: | ---: |
| Accuracy | 0.071 [0.046, 0.096] | 0.042 [0.025, 0.058] |
| AUC | 0.048 [0.033, 0.062] | 0.039 [0.022, 0.056] |
| AUPRC | 0.043 [0.032, 0.053] | 0.040 [0.018, 0.063] |
| ECE | 0.064 [0.042, 0.086] | 0.040 [0.026, 0.053] |

Accuracy ranking separates here as well: Spearman 0.30 versus 0.87, and Top-1 0.25 versus 0.70. ECE Spearman correlation moves from −0.46 to 0.53. AUC ranking is weak on both sides because every head sits above 0.95. At `n_l = 50`, Accuracy error is still 0.056 versus 0.032. At `n_l = 100` it is 0.043 versus 0.034.

When one head is removed, the MLP and logistic regression move the posterior the most (total variation 0.047 and 0.043). The two tree models are only 0.004 and 0.003. On the same ResNet features, the tree models substitute for each other, while the linear model and the MLP contribute more information.

Resizing 28×28 images to 224 does not recover the resolution of the original radiograph. This example shows that the pipeline runs, and that with 20 labels the Accuracy and ECE estimates beat the labeled-only baseline. It is not a new pneumonia-detection result.

### 4.4 How to reproduce

```bash
conda activate ssme
python experiments/pneumoniamnist/pneumonia_ssme.py
python experiments/transfer/transfer_ssme.py
```

`experiments/pneumoniamnist/pneumoniamnist.ipynb` is the same official-model demo. The single-split transfer demo is `experiments/transfer/transfer_ssme.py`. Tables across seeds are in `experiments/pneumoniamnist/results/` and `experiments/transfer/results/`. The main config can also be run with `ssme-lite configs/pneumoniamnist.json`.

Feature extraction used the local PyTorch install and MPS. ImageNet ResNet18 weights are downloaded once. The official experiment reads prediction CSVs only and does not need the 853 MB MedMNIST weights.

### 4.5 Limits

Ten seeds are enough to see the direction. That is fewer than the 30–50 repeats common in the paper, so the intervals are wider. `n_u = 400` is set by the size of the test split, not by 1,000. On this data, SSME improves Accuracy estimates when labels are scarce, and Accuracy and ECE in the transfer setting. It does not stably rank models by AUC. When the models are all strong and their AUCs are close, a smaller absolute error is not enough to select the right model.
