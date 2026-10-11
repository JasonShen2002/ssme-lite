# Completed extension results — 2026-10-11

All four requested additions are implemented in this `ssme-lite` submission
folder. The full measurement run completed **342 worker conditions**: 42
performance conditions, 180 label-efficiency conditions and 120 stress
conditions. No worker crashed or timed out. Within the stress study, 27 SSME
fits were deliberately not attempted because the visible labels lacked a class;
these are recorded method failures, not successful fits.

The model-file CLI example also completed and produced
[an offline report](model_cli_demo/report/report.html). No server was required
for the completed grid. The 10,000-row server profile is available but has not
been run.

## 1. Scale and parallelism

Three independent process repetitions per condition, 100 fixed updates, native
numerical threads limited to one. Recorded environment: macOS 26.6 arm64,
10 logical CPUs, Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1, pandas 2.3.3,
scikit-learn 1.9.1 and joblib 1.6.0. Controlled overlapping Gaussian score views,
three models, two classes, one KDE worker and 512-row posterior batches:

| Fitted rows | Mean fit time (s) | Mean whole-worker peak RSS (MiB) |
| ---: | ---: | ---: |
| 500 | 2.73 | 198.18 |
| 1,000 | 10.62 | 198.69 |
| 2,000 | 42.63 | 197.65 |
| 5,000 | 254.07 | 195.85 |

Time rises close to quadratically over this tested range. RSS is dominated by
library/report overhead at these sizes; the small decrease is process noise,
not evidence that larger fits need less memory. Inputs, responsibilities and
KDE training points remain resident. No out-of-core or arbitrary-scale claim.

At 1,000 rows, using two class-fit workers increases mean fit time from 10.62
to 11.56 seconds. At 624 real checkpoint rows with seven models, one/two workers
take 3.88/5.20 seconds. Thread overhead outweighs the benefit in these cases.
Changing the batch size from 512 to 128 takes 10.74 seconds at 1,000 rows.
Five classes at the same 1,000 rows and three score views take 27.10 seconds;
six score views with two classes take 11.56 seconds.

Prediction was measured separately using three fitted sklearn classifiers over
20 features. On 5,000 evaluation rows, one/two workers take 0.0173/0.1305 seconds.
These are inexpensive predictions; parallel execution is supported, but no
speedup should be advertised for them.

**All recorded worker/batch/repetition comparisons have maximum absolute output
difference 0.0.** Metrics, posterior inference and report construction are timed
separately. Full raw and aggregate records:
[per_run.csv](results/performance/full/per_run.csv),
[summary.csv](results/performance/full/summary.csv),
[equivalence.csv](results/performance/full/equivalence.csv),
[manifest](results/performance/full/manifest.json).

![Scaling](results/performance/full/scaling.png)

## 2. Model files to a report

The current source checkout supports `source: "models"`, named trusted joblib
files, NPZ evaluation features and partial labels, an explicit ordered class
axis, configuration preflight, and batch prediction. `init-config` creates a
template; existing `ssme-lite config.json` calls still work. The CLI evaluates
already trained models and never trains them.

The committed example trains on separate rows and exports only 20 visible
labels. The output includes HTML/CSV/JSON, frozen predictions and SHA256 hashes
for the input and models. Tests compare its probabilities and metric estimates
with direct Python calls, including calls from a different working directory.
Class-order mistakes and untrusted loading are rejected; NumPy, pandas and sparse
batch predictions match whole-array predictions.

See [the example configuration](model_cli_demo/config.json),
[input provenance](model_cli_demo/report/input_provenance.json), and
[the CLI guide](../../docs/guides/cli.md).

## 3. Label efficiency: improvement at equal budget, replacement not established

Both bundled medical-model pipelines use 30 new seeds (100–129). Each seed has
a fixed 420-row estimation pool, a disjoint 204-row held-out set, and nested
20/50/100 visible labels. Neither the truth set nor the fitted pool changes
with label budget. Accuracy MAE is averaged across candidate models per seed,
then across seeds. All 180 conditions have valid SSME fits.

| Pipeline | Labeled-only 20 | SSME 20 | Mean-probability 20 | Labeled-only 50 | Labeled-only 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seven checkpoints | 5.84 pp | 3.21 pp | 3.73 pp | 3.38 pp | 2.61 pp |
| Five transfer heads | 7.35 pp | 4.60 pp | 4.11 pp | 4.48 pp | 3.61 pp |

The same-budget mean error reductions are approximately 45.1% and 37.3%.
The checkpoint mean with 20 labels is close to labeled-only with 50, but that
alone does not establish replacement. For SSME20 minus labeled-only50, the
one-sided Bonferroni upper bounds are **+0.97 pp** (checkpoints) and **+1.95 pp**
(transfer); both fail the predeclared less-than-zero criterion. None of the six
predeclared comparisons establishes lower error after adjustment. Therefore
**this experiment does not justify claiming 60% label savings**. It also does
not prove equivalence or rule out benefits under other protocols.

The mean-probability baseline is competitive, especially for transfer heads.
On checkpoints, held-out top-1 selection rises from 38.4% (labeled-only20) to
63.3% (SSME20), and selection regret falls from 1.68 to 0.60 pp. On transfer,
top-1 rises from 25.1% to 40.0%, while the mean-probability baseline reaches
46.7%. Estimation error and selection quality should be reported separately.

These new results use different seeds and a fixed-target design. They neither
overwrite nor numerically reproduce the earlier manuscript's 10-split table.
Sources: [summary](results/labels/full/summary.csv),
[paired comparisons](results/labels/full/paired_comparisons.csv),
[split audit](results/labels/full/audit.json),
[per-seed results](results/labels/full/per_seed.csv).

![Label efficiency](results/labels/full/label_efficiency.png)

## 4. Multiclass and redundant score views

Six classifiers were trained only on the training split of sklearn digits.
Three-class and ten-class frozen probabilities are bundled with train/eval row
IDs and model specifications. Every regime evaluates the **same original three
candidates**; extra views change only the input to SSME.

Mean fixed-pool Accuracy MAE, in percentage points:

| Classes / visible labels | Labeled-only | SSME: 3 diverse | SSME: 6 diverse | SSME: 3 repeated twice | Valid seeds per SSME regime |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 / 20 | 2.29 | 3.69 | 1.59 | 3.48 | 10/10 |
| 3 / 60 | 1.40 | 2.78 | 1.39 | 2.67 | 10/10 |
| 10 / 60 | 2.13 | 4.22 | 3.93 | 4.17 | 10/10 |

With ten classes and only twenty randomly observed labels, **9 of 10 masks lack
at least one class**, affecting all three input regimes equally (27 method
failures). Only one seed is estimable per regime, so its SSME mean is not directly
compared with the ten-seed baseline. Raw values and missing-class status remain
in the files. AUC likewise retains undefined cases rather than filling them or
repairing labels.

The three-class result distinguishes adding diverse auxiliary predictions from
simply repeating existing views: six diverse views reduce MAE by 2.11 pp versus
three at budget 20, whereas exact duplication reduces it by only 0.22 pp. These
are descriptive paired means over ten masks, not a universal benefit guarantee.
Duplicates add no new information, but change the ALR geometry and Scott
bandwidth; consequently the estimate need not be invariant to replication.

The ten-class, 60-label results are a useful failure case: all SSME regimes have
larger Accuracy MAE than labeled-only. Multiclass input support is verified;
consistent multiclass accuracy gains are not established. The primary stress
target is the fixed 240-row pool. The separate three-class held-out set has only
29 rows and is treated as limited supplementary evidence.

Sources: [summary](results/stress/full/summary.csv),
[method status](results/stress/full/status.csv),
[paired regime effects](results/stress/full/regime_pairs.csv),
[diagnostics](results/stress/full/diagnostics.csv),
[frozen input manifest](inputs/digits_manifest.json).

![Stress results at budgets with full class coverage](results/stress/full/stress.png)

## Verification and practical limits

- Source-checkout tests: **48 passed, 1 optional official audit skipped**.
- Installed-wheel tests, run outside the repository: **48 passed, 1 skipped**.
- Strict MkDocs build and strict twine checks passed.
- Independent audit passed: 342 raw conditions, 2,358 statistic rows, 8,478 Accuracy truth values, 72 source snapshot files and all 27 missing-class decisions. See [audit_full.json](results/audit_full.json).
- Raw result hashes, fixed targets, disjoint splits, nested label sets and parallel/batch agreement were checked.
- This is unreleased source work, not a new PyPI publication or a revised manuscript claim.
- The measured 5,000-row fit takes about 4.2 minutes. A rough quadratic extrapolation suggests about 17 minutes per 10,000-row fit on the same machine; this is an estimate, not a measurement. Use the optional CPU-server profile if that scale is needed.

Reproduction commands and resume behavior are in [README.md](README.md).
Analysis provenance is saved separately from immutable measurement snapshots;
the final stress plot uses 3-class/20-label and 10-class/60-label settings so all
plotted method means have ten valid seeds. The failed 10-class/20-label setting
remains in the tabular results.
