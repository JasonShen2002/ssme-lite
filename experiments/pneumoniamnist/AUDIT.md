# PneumoniaMNIST data and model audit

**Historical five-family prediction-file audit.** This document records the earlier precomputed-score protocol. The final course report uses seven runnable checkpoints: ResNet18, three AutoKeras models, and three AutoML Vision models. See [the current experiment summary](../REPORT.md) for that study. The original audit below is retained as provenance.

Verdict for the historical protocol: **pass**. Five models from different families were chosen in advance and scored on the official test split.

## Data

PneumoniaMNIST is a binary classification of pediatric chest X-rays (0 = normal, 1 = pneumonia). The official split is train 4708 / val 524 / test 624. In this audit the test counts are normal 234 and pneumonia 390.

Classifiers are trained on train and use val for early stopping. The SSME evaluation pool is therefore **test** only. The test split has 624 images, which cannot hold the 1,000 unlabeled samples used in the paper. The protocol is `n_l ∈ {20, 50, 100}`, `n_u = 400`, with the remaining samples held out. Held-out size is 124 when `n_l = 100`. `n_u >> n_l` holds at 20 and 50 labels, and is a factor of four at 100 labels.

The validation set is not added to the evaluation pool. The transfer-learning example fits sklearn heads on train and also evaluates only on test. The ImageNet backbone was not trained on this dataset.

## What the official files contain

The source is `predictions.zip` on Zenodo `10.5281/zenodo.7782114`. The archive is about 1.9 GB. This audit keeps only the PneumoniaMNIST test CSVs. Each file is written by `medmnist.Evaluator`. Binary accuracy uses a 0.5 threshold on the score, and ROC-AUC uses that same score, so the values are probabilities (or scores on the same scale), not hard labels.

The auto-sklearn model snapshot was not released, but its prediction file was. SSME consumes predicted probabilities and does not need to rerun the 853 MB weights.

There are 21 test prediction files: ResNet-18 and ResNet-50, each at 28 and 224 input size, plus AutoKeras, auto-sklearn, and Google AutoML Vision, with three repeats per family.

## Selected models

The rule was fixed before any SSME run: each family keeps repeat `1` only. Checkpoints are not chosen by SSME error.

The four ResNet settings do not each take a slot. ResNet-18 at 28 pixels and ResNet-50 at 224 pixels keep both a depth difference and an input-size difference. The remaining slots go to different training systems.

| Model | Input (filename) | Output | File AUC | Recomputed AUC | File ACC | Recomputed ACC | Score range |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `resnet18_28_1` | ResNet-18 @ 28 | 2 class-probability columns, positive class kept | 0.948 | 0.948 | 0.833 | 0.833 | [0.000, 1.000] |
| `resnet50_224_1` | ResNet-50 @ 224 | 2 class-probability columns, positive class kept | 0.967 | 0.967 | 0.896 | 0.896 | [0.000, 1.000] |
| `autokeras_1` | AutoKeras | 1 positive-class probability column | 0.939 | 0.939 | 0.883 | 0.883 | [0.000, 1.000] |
| `autosklearn_1` | auto-sklearn | 2 class-probability columns, positive class kept | 0.942 | 0.942 | 0.859 | 0.859 | [0.377, 0.629] |
| `automl_vision_1` | Google AutoML Vision | 2 class-probability columns, positive class kept | 0.993 | 0.993 | 0.941 | 0.941 | [0.039, 0.969] |

The Accuracy range is 0.107. The maximum disagreement of thresholded predictions is 0.120. auto-sklearn positive-class probabilities sit in 0.377–0.629: ranking is still valid (AUC 0.942), and confidence is clearly lower. That is a calibration difference. The file and the recomputed metrics agree.

Pearson correlation of positive-class probabilities:

| | `resnet18_28_1` | `resnet50_224_1` | `autokeras_1` | `autosklearn_1` | `automl_vision_1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `resnet18_28_1` | 1.000 | 0.829 | 0.881 | 0.868 | 0.804 |
| `resnet50_224_1` | 0.829 | 1.000 | 0.912 | 0.892 | 0.894 |
| `autokeras_1` | 0.881 | 0.912 | 1.000 | 0.928 | 0.887 |
| `autosklearn_1` | 0.868 | 0.892 | 0.928 | 1.000 | 0.891 |
| `automl_vision_1` | 0.804 | 0.894 | 0.887 | 0.891 | 1.000 |

Disagreement rate of predictions thresholded at 0.5:

- `resnet18_28_1|resnet50_224_1`: 0.104
- `resnet18_28_1|autokeras_1`: 0.082
- `resnet18_28_1|autosklearn_1`: 0.067
- `resnet18_28_1|automl_vision_1`: 0.120
- `resnet50_224_1|autokeras_1`: 0.064
- `resnet50_224_1|autosklearn_1`: 0.085
- `resnet50_224_1|automl_vision_1`: 0.074
- `autokeras_1|autosklearn_1`: 0.056
- `autokeras_1|automl_vision_1`: 0.090
- `autosklearn_1|automl_vision_1`: 0.101

## Nested subsets for the model-count ablation

Subsets add models from lowest to highest prediction correlation. The rule uses scores only, not labels, and not SSME error. The 5-model order in the main experiment is still the table order above.

- 2 models: `resnet18_28_1`, `automl_vision_1`
- 3 models: `resnet18_28_1`, `automl_vision_1`, `resnet50_224_1`
- 4 models: `resnet18_28_1`, `automl_vision_1`, `resnet50_224_1`, `autosklearn_1`
- 5 models: `resnet18_28_1`, `automl_vision_1`, `resnet50_224_1`, `autosklearn_1`, `autokeras_1`

A split at a given seed depends only on the sample counts, so these subsets compare the same labeled, unlabeled, and held-out indices.
