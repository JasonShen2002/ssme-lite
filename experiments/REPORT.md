# Current experimental evidence

This summary matches the final JMLR-style course report. PneumoniaMNIST is the main dataset; CivilComments provides the original-paper quickstart and qualitative reproduction check. The software is published as [ssme-lite 0.1.0](https://pypi.org/project/ssme-lite/0.1.0/).

## Protocols

| Pipeline | Candidates | Visible / unlabeled / held-out | Repetitions | Bandwidth / iterations |
| --- | --- | --- | --- | --- |
| PneumoniaMNIST checkpoint inference | 7 | 20 / 400 / 204 | 10 splits | Scott / 100 |
| PneumoniaMNIST frozen-ResNet18 transfer | 5 | 20 / 400 / 204 | 10 splits | Scott / 100 |
| CivilComments reproduction | 7 | 20 / 1,000 / 66,891 | 50 valid splits | Reference-code / 20 |

The seven checkpoint candidates are `resnet18_28_1`, `autokeras_1`, `autokeras_2`, `autokeras_3`, `automl_vision_1`, `automl_vision_2`, and `automl_vision_3`. The five transfer heads are logistic regression, random forest, MLP, extra trees, and histogram gradient boosting, trained only on the 4,708 official training images. Validation has 524 images and test has 624.

The report evaluates a fitted estimation pool. Experimental errors compare those estimates with separately held-out truth. Both methods receive the same visible labels; paired reductions are summarized across splits, not treated as independent model-level observations.

## Main results

Mean held-out absolute estimation errors, in percentage points:

| Pipeline | Metric | Labels only | SSME-Lite |
| --- | --- | ---: | ---: |
| Seven checkpoints | Accuracy | 5.72 | 2.26 |
| Seven checkpoints | ROC AUC | 3.15 | 2.55 |
| Seven checkpoints | Average precision | 2.97 | 2.40 |
| Seven checkpoints | ECE | 3.94 | 2.40 |
| Five transfer heads | Accuracy | 7.11 | 4.16 |
| Five transfer heads | ROC AUC | 4.75 | 3.89 |
| Five transfer heads | Average precision | 4.27 | 4.05 |
| Five transfer heads | ECE | 6.43 | 3.96 |

Accuracy MAE decreases by 60.5% for checkpoints and 41.4% for transfer heads. All twelve candidates improve in mean Accuracy MAE. Accuracy Top-1 selection success increases from 23.3% to 50.0% for checkpoints and from 25.0% to 70.0% for transfer heads. The clearest benefits are Accuracy and ECE; other selection metrics do not show consistent improvement.

At label budgets 20, 50, and 100, transfer Accuracy MAE is respectively 7.11/4.16, 5.60/3.19, and 4.28/3.45 pp (labels only / SSME). Unlabeled count stays 400, while held-out counts are 204, 174, and 124.

Model-influence diagnostics hold initialization, scalar bandwidth, visible labels, and iteration count fixed. Across ten transfer splits, removing MLP or logistic regression produces mean posterior total variation 0.0465 or 0.0425; the two forests produce 0.0043 or 0.0030. This measures conditional sensitivity to each input model.

## CivilComments reproduction

Across 50 valid half-splits, Accuracy / AUC / average precision / ECE MAE is 2.27 / 3.28 / 9.41 / 2.12 pp. Relative to labeled-only errors, the ratios are 0.41 / 0.47 / 0.42 / 0.42. The comparison also includes weighted voting, pseudo-labeling, and Dawid–Skene aggregation.

This is a qualitative reproduction: the package's ECE uses equal-width bins, whereas the paper uses equal-frequency bins. Archived execution evidence for the official algorithm is linked from the [requirement audit](../docs/course-requirements.md).

## Reproduce and inspect

- Checkpoint inference: `python experiments/pneumoniamnist/pneumonia_ssme.py`.
- Transfer demo: `python experiments/transfer/transfer_ssme.py`.
- Repeated checkpoint evaluation, after predictions exist: `ssme-lite experiments/pneumoniamnist/config.json`.
- Repeated transfer evaluation, after predictions exist: `ssme-lite experiments/transfer/config.json`.
- CivilComments comparison: `python experiments/civilcomments/reproduce_official_split.py` after preparing the original inputs.

See the [PneumoniaMNIST](../docs/tutorials/pneumoniamnist.md), [transfer](../docs/tutorials/transfer.md), and [CivilComments](../docs/tutorials/civilcomments.md) tutorials for dependencies and input preparation. Saved benchmark tables are in `pneumoniamnist/results/nl20_nu400_m7/` and `transfer/results/nl20_nu400_m5/`.

## Historical protocol

An earlier study used five published prediction families, including ResNet50 and auto-sklearn. It is preserved in [the historical report](archive/five_model_score_study.md) and [prediction-file audit](pneumoniamnist/AUDIT.md). Its candidate set and numerical results differ from the current seven-checkpoint inference study.
