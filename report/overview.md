# SSME-Lite: reproducible evaluation with scarce labels

**Junje Shen · Student ID: 2026211240**

This guide accompanies the [course paper](main.pdf), which presents the method, software, main experiments, and extension evidence in four body pages, followed by references and reproduction instructions.

## Problem and workflow

When several trained classifiers are available but labels are scarce, SSME-Lite combines their predictions with a small labeled sample to estimate Accuracy, ROC AUC, AP, and ECE. The workflow connects probability generation, class and sample alignment, semi-supervised estimation, model selection, and offline reporting.

![End-to-end workflow](figures/workflow.png)

The core uses a sklearn-style interface without a deep-learning dependency. Transfer learning is an optional tutorial. Model probabilities undergo an additive log-ratio transform; weighted kernel-density updates estimate class posteriors while observed labels remain clamped. Accuracy uses the posterior expectation; other metrics use shared latent-label draws for paired comparisons.

Python users call `fit()` and `report()`. CLI users provide named fitted models, evaluation data, class order, and output paths in JSON. Preflight checks, input hashes, frozen predictions, and experiment manifests support traceability. Parallel prediction, class-density fitting, and batched posterior evaluation are configurable. Leave-one-model-out refits measure conditional model influence.

## Main medical study

PneumoniaMNIST supplies seven released checkpoints and five classifiers trained on frozen ResNet18 features. Ten shared splits each contain 20 visible labels, 400 unlabeled examples, and a separate 204-row evaluation target.

| Pipeline | Labels-only Accuracy MAE | SSME-Lite MAE | Reduction |
| --- | ---: | ---: | ---: |
| Seven checkpoints | 5.70 pp | 2.22 pp | 61.1% |
| Five transfer heads | 7.11 pp | 4.16 pp | 41.4% |

All twelve candidates improve in mean Accuracy estimation error. Accuracy-based top-model selection increases from 23.3% to 50.0% and from 25.0% to 70.0%, respectively. Four-metric results, paired intervals, and model-level records are archived.

![Model-level Accuracy estimation errors](figures/diagnostics.png)

CivilComments provides an original-paper-data reproduction check. Across 50 valid splits, relative MAEs against labeled-only estimation are 0.41, 0.47, 0.42, and 0.42 for Accuracy, AUC, AP, and ECE. Voting, pseudo-label, and Dawid–Skene comparisons are also retained.

## Extension evidence

![Annotation budgets, CPU scaling, and model composition](figures/extensions.png)

The fixed-target study uses 30 new seeds per medical pipeline, with nested 20/50/100-label budgets, a fixed 420-row pool, and a disjoint 204-row holdout. At 20 labels, mean errors decrease from 5.84 to 3.21 pp and from 7.35 to 4.60 pp. Cross-budget replacement is assessed separately; see the [full findings](../experiments/extensions/RESULTS.md).

Figure 3a shows the CivilComments comparison. Figure 3b shows the original ten-split transfer study with 400 unlabeled rows and holdouts of 204/174/124 rows. These panels are distinct from the fixed-target extension.

The 42-condition performance study measures separate execution phases. With three models, two classes, and 100 updates, mean fit times at 500/1,000/2,000/5,000 rows are 2.73/10.62/42.63/254.07 seconds. Each setting has three independent process repetitions. Recorded worker and batch comparisons have maximum absolute output difference 0.0.

The 120-condition digits study compares three or ten classes and diverse or duplicated input views, always evaluating the same three candidates. For three classes and 20 labels, six diverse views achieve 1.59 pp MAE, versus 3.69 for three views and 3.48 for duplicated views. Full class-coverage and method-status records remain available.

## Review and reproduce

The submission includes the paper, editable LaTeX, vector figures, figure builders, package API and CLI, tutorials, frozen inputs, saved splits, source snapshots, and audit receipts. The extension validation recorded 48 passing tests in both source and installed-wheel checks. Independent auditing checked 342 configurations, 2,358 statistical rows, and 8,478 Accuracy truth values.

Run `python reproduction/run.py --jobs 2` from the repository root to refit the medical studies. See [extension commands](../experiments/extensions/README.md) for complete experiments and optional server configurations; run `python experiments/extensions/audit_results.py` to audit saved evidence.
