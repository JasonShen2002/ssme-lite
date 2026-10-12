# Course requirements and implementation evidence

This audit maps the assignment to the published 0.1.0 baseline and the current source checkout. Version 0.2.0 incorporates the 2026-10-11 extensions for fitted-model configuration and scaling/statistical experiments; see the [extension guide](guides/extensions.md). Production deployment and arbitrary-scale guarantees remain outside the evidence.

## Requirement mapping

| Assignment item | Status | Evidence |
| --- | --- | --- |
| Read SSME and execute the official example/code | Met with archived evidence | Original-code runs on CivilComments, five seeds and 20 epochs, with native outputs and source hashes in [the execution record](evidence/official-execution.json); method attribution in [the method guide](method.md) |
| CivilComments or CIFAR-10 with few labels and many unlabeled samples | Met | CivilComments: 20 labels, 1,000 unlabeled samples; 50 valid half-split repetitions |
| sklearn-style prediction matrix generator | Met | `PredictionMatrixGenerator` accepts mappings or lists of fitted models, aligns matching class sets, and returns binary or multiclass probabilities |
| SSME core algorithm in a simple Python class | Met | `SSMEEstimator.fit`, posterior inference, weighted KDE updates, observed-label clamping, explicit bandwidth and iteration controls |
| Rankings, uncertainty intervals, and visual comparisons | Met | `EvaluationReport.ranking`, conditional label intervals, interactive HTML, plots, CSV and JSON exports |
| Pretrained-feature transfer-learning example | Met | Frozen ImageNet ResNet18, 512-dimensional features, five sklearn heads trained on the official PneumoniaMNIST training split |
| User-friendly documentation and tutorials | Met | Installation, two runnable synthetic examples, three real-data tutorials, API reference, CLI, and troubleshooting |
| SSME versus labeled-only across different models | Met | Seven checkpoints and five transfer heads, ten paired splits each; four metrics; CivilComments comparison and annotation-budget study |
| Model-contribution analysis | Met as conditional influence | Controlled leave-one-model-out refits, posterior total variation and changes in remaining Accuracy estimates; no causal or guaranteed redundancy claim |
| Parallel computation | Met | Prediction workers, class-density workers, independent benchmark-seed workers; serial/parallel benchmark equivalence test |
| Large-scale / production-grade software | Partially evidenced | Published baseline, CI, validation and batch evaluation; current checkout adds isolated scaling/RSS experiments under `experiments/extensions`. No operational deployment or million-row fit claim |
| One-command configuration with data path and model list | Met for stored predictions and trusted fitted model files in the current checkout | JSON `source: models` reads features, partial labels, an explicit class axis and named joblib files; preflight, batch prediction, provenance and an end-to-end example are included |

“First” is not claimed: no literature-wide novelty search establishes that assertion. The demonstrated contribution is an installable, documented workflow with reproducible experiments and controlled influence diagnostics.

## Evidence that travels with the repository

The [official execution record](evidence/official-execution.json) retains native official-code outputs for seeds 0–4 and provenance hashes from the archived audit. These runs used the original `SSME_KDE` function. They are onboarding evidence from an earlier experiment, not a rerun of the published 0.1.0 package and not the 50-seed final reproduction protocol.

The final real-data protocols and numbers are in [the current experiment summary](https://github.com/JasonShen2002/ssme-lite/blob/main/experiments/REPORT.md). An earlier five-family score study remains explicitly archived; the seven-checkpoint inference experiment is the current main study.

## Release and validation

- [PyPI 0.1.0](https://pypi.org/project/ssme-lite/0.1.0/) is publicly installable.
- [CI](https://github.com/JasonShen2002/ssme-lite/actions/runs/37306787797) passed on Python 3.10–3.14.
- Local testing with the reference-bandwidth dependency installed: 38 passed, one optional official-code integration audit skipped. Minimal CI environments may also skip the statsmodels-specific check.
- The [publication workflow](https://github.com/JasonShen2002/ssme-lite/actions/runs/37307075613) passed build, tests, strict documentation build, distribution validation, and installed-wheel smoke checks.
- A fresh public-PyPI installation ran the quickstart, sklearn/contribution example, JSON CLI, and dependency check successfully.

## Scope of the course report

The final report uses four body pages plus one reference page in JMLR style, with three figures and two tables. It reports saved, reproducibly aggregated experiments. Conditional label intervals are identified as such; they are not full frequentist confidence intervals over all model-fitting uncertainty. Large source-dataset size is not presented as evidence that every row was fitted jointly.

## Remaining extension opportunities

The current checkout implements the model-file adapter and records measured scaling, fixed-target label-efficiency and multiclass/redundancy results. These package extensions are included in version 0.2.0. Larger-machine measurements may extend the tested scale; they do not establish production certification. Broader interval coverage and independent reference-algorithm equivalence remain separate research opportunities.


## Current checkout verification (2026-10-11)

48 tests passed with one optional official-code audit skipped, both in the
source checkout and against an installed wheel outside the checkout. The full
extension grid completed 342 worker conditions, including 27 recorded
missing-class SSME conditions in the stress study. See the
[extension findings](guides/extensions.md); earlier published CI results above
are retained as release history, not claimed as a CI run for this extension.
