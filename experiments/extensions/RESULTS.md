# SSME-Lite extension findings — 2026-10-11

All four selected extensions are implemented. The full study completed **342 independent worker configurations**: 42 performance, 180 label-efficiency, and 120 stress conditions. No worker crashed or timed out. In the stress study, 27 SSME fits were not performed because visible labels omitted a class; these are recorded method failures, not successful fits.

The fitted-model CLI example generated an [offline report](model_cli_demo/report/report.html). The experiments used the local CPU. A 10,000-row server profile is provided but has not been executed. Package functionality from this study is included in version 0.2.0; archived measurements and source snapshots retain their original identities.

## 1. Scaling and parallelism

Each setting uses three independent processes, 100 updates, and one numerical-library thread. Environment: macOS 26.6 arm64, 10 logical CPUs, Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1, pandas 2.3.3, scikit-learn 1.9.1, and joblib 1.6.0.

Controlled overlapping Gaussian scores use three models, two classes, one KDE worker, and posterior batches of 512 rows.

| Fitted rows | Mean fit time (s) | Mean whole-worker peak RSS (MiB) |
| ---: | ---: | ---: |
| 500 | 2.73 | 198.18 |
| 1,000 | 10.62 | 198.69 |
| 2,000 | 42.63 | 197.65 |
| 5,000 | 254.07 | 195.85 |

Runtime grows approximately quadratically over this range. Whole-worker RSS includes library and reporting overhead; small decreases reflect process variability, not lower memory requirements for larger inputs. Inputs, responsibilities, and KDE training points remain in memory; this is not an out-of-core implementation.

At 1,000 rows, one/two class-fit workers take 10.62/11.56 seconds. For seven real checkpoint models on 624 rows, they take 3.88/5.20 seconds. Parallel overhead exceeds the benefit in these settings. A batch size of 128 takes 10.74 seconds at 1,000 rows. Five classes take 27.10 seconds, while six models with two classes take 11.56 seconds.

Prediction timing uses three fitted sklearn classifiers, 20 features, and 5,000 evaluation rows: one/two workers take 0.0173/0.1305 seconds. These inexpensive predictors do not demonstrate a parallel speedup.

**Maximum absolute output difference is 0.0 across recorded worker, batch, and repetition comparisons.** Metrics, inference, and report generation are timed separately.

[Per-run measurements](results/performance/full/per_run.csv) · [Summary](results/performance/full/summary.csv) · [Equivalence checks](results/performance/full/equivalence.csv) · [Manifest](results/performance/full/manifest.json)

![Scaling, runtime, and peak memory](results/performance/full/scaling.png)

## 2. Fitted models to reports

The `source: "models"` configuration accepts named trusted joblib files, NPZ evaluation features and partial labels, and an explicit class axis. Preflight checks and batched prediction are included; `init-config` produces a template. Existing `ssme-lite config.json` calls remain supported. The evaluation CLI consumes already trained models.

The example trains on disjoint training rows and exposes only 20 evaluation labels. Outputs include HTML, CSV, JSON, frozen predictions, and SHA256 hashes of input data and models. Tests compare CLI predictions and estimates with the Python API, including execution from another working directory. Incorrect class order and untrusted model loading are rejected. Batched NumPy, pandas, and sparse predictions agree with whole-array prediction.

[Example configuration](model_cli_demo/config.json) · [Input provenance](model_cli_demo/report/input_provenance.json) · [CLI guide](../../docs/guides/cli.md)

## 3. Label efficiency with fixed evaluation targets

Each medical pipeline uses 30 new seeds (100–129), a fixed 420-row estimation pool, and a disjoint 204-row holdout. Visible-label budgets of 20/50/100 are nested. Neither the fitted pool nor the evaluation target changes with budget.

Accuracy MAE averages absolute estimation errors across candidates within each seed, then across seeds. All 180 configurations completed SSME fitting. Values below are percentage points.

| Pipeline | Labels only, 20 | SSME, 20 | Mean probability, 20 | Labels only, 50 | Labels only, 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seven checkpoints | 5.84 | 3.21 | 3.73 | 3.38 | 2.61 |
| Five transfer heads | 7.35 | 4.60 | 4.11 | 4.48 | 3.61 |

At the same 20-label budget, mean error decreases by **45.1% and 37.3%** relative to labeled-only estimation.

The primary cross-budget criterion compares SSME with 20 labels against labeled-only with 50. Bonferroni-adjusted one-sided upper bounds on their paired error difference are **+0.97 pp** for checkpoints and **+1.95 pp** for transfer, neither below zero. None of the six predeclared adjusted comparisons establishes lower error. These results do not establish 60% label savings or equivalence.

Mean probability is competitive, particularly for transfer heads. Checkpoint top-model selection rises from 38.4% to 63.3%, and selection regret falls from 1.68 to 0.60 pp. Transfer selection rises from 25.1% to 40.0%, while mean probability reaches 46.7%. Estimation error and selection quality should therefore be reported separately.

This new-seed, fixed-target protocol complements rather than replaces the original ten-split study.

[Summary](results/labels/full/summary.csv) · [Paired comparisons](results/labels/full/paired_comparisons.csv) · [Split audit](results/labels/full/audit.json) · [Per-seed results](results/labels/full/per_seed.csv)

![Accuracy estimation error across annotation budgets](results/labels/full/label_efficiency.png)

## 4. Multiclass and model composition

Six classifiers are trained on the sklearn digits training split. Frozen three-class and ten-class probabilities include training/evaluation row identifiers and model configurations. Every regime evaluates **the same three original candidates**; additional views only supply auxiliary information to SSME.

The fixed-pool Accuracy MAEs below are in percentage points. Duplicated views copy each original model once, producing six input views.

| Classes / labels | Labels only | SSME: 3 diverse | SSME: 6 diverse | SSME: 3 duplicated | Valid seeds per SSME regime |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 / 20 | 2.29 | 3.69 | 1.59 | 3.48 | 10/10 |
| 3 / 60 | 1.40 | 2.78 | 1.39 | 2.67 | 10/10 |
| 10 / 60 | 2.13 | 4.22 | 3.93 | 4.17 | 10/10 |

With ten classes and 20 randomly observed labels, nine of ten masks omit a class, affecting all three regimes and yielding 27 method failures. Only one seed remains estimable per regime, so its mean is not directly compared with ten-seed baselines. Missing-class statuses and undefined AUC values remain recorded without imputation or hidden-label repairs.

For three classes and 20 labels, six diverse views reduce MAE by 2.11 pp relative to three views; duplication reduces it by 0.22 pp. These are descriptive paired means over ten masks. Duplication adds no information but changes ALR geometry and Scott bandwidth, so estimates need not be invariant to duplicated inputs.

At ten classes and 60 labels, all SSME regimes have higher MAE than labeled-only estimation. This confirms functional multiclass support, not universal multiclass improvement. The primary target is the fixed 240-row pool; the three-class independent holdout has only 29 rows and provides limited supplementary evidence.

[Summary](results/stress/full/summary.csv) · [Method statuses](results/stress/full/status.csv) · [Paired regime effects](results/stress/full/regime_pairs.csv) · [Diagnostics](results/stress/full/diagnostics.csv) · [Input manifest](inputs/digits_manifest.json)

![Settings with ten valid seeds for every method](results/stress/full/stress.png)

## 5. Validation and scope

- Original extension validation: **48 passed, one optional official-code audit skipped**, in both source and installed-wheel tests outside the checkout.
- Strict MkDocs and twine checks passed.
- [Independent audit](results/audit_full.json): 342 configurations, 2,358 statistical rows, 8,478 Accuracy truth values, 72 snapshot files, and all 27 missing-class decisions.
- Checks cover input hashes, fixed targets, disjoint splits, nested labels, and worker/batch equivalence.
- The measured 5,000-row fit takes approximately 4.2 minutes. Quadratic extrapolation suggests roughly 17 minutes for 10,000 rows on the same machine; this is an estimate, not a measurement.

See the [reproduction commands](README.md) for experiments and resume rules. Analysis provenance is separate from immutable measurement snapshots. The displayed stress figure uses three classes / 20 labels and ten classes / 60 labels, both with ten valid seeds per method; all other settings remain in the tables.
