# Extension protocol — fixed before the new runs

Protocol date: 2026-10-11. Baseline Git commit: `ec4b72c`.
These additions live entirely in the final `ssme-lite` submission directory.
Existing manuscript results remain separate. No interval-coverage or new
estimator research is included in this extension.

## Performance

Use independent subprocesses, one condition per process, with native BLAS
threads limited to one. Record process peak RSS (including imported libraries),
phase times, actual iteration count, posterior checksum, and failures/timeouts.
Vary sample count, model count, class count, class-fit workers and posterior
batch size. Use controlled overlapping Gaussian score views, plus the bundled
real checkpoint predictions without duplicating rows to claim a larger dataset.
Time prediction and report generation separately from estimator fitting.
Fit runs use 100 fixed updates in the full profile; pilot runs use 5 updates
and are never presented as full-scale evidence. Compare identical input/seed
posteriors for worker/batch variants. Three repetitions per full condition.
The server profile extends the same conditions to 10,000 rows. Timeouts are
recorded, not silently replaced with smaller problems.

## Label efficiency

Use both hashed bundled PneumoniaMNIST prediction artifacts (624 rows each).
For each seed 100–129, select a fixed 420-row estimation pool and a disjoint
204-row held-out set. Randomly permute the pool once; expose nested prefixes
of 20, 50, 100 labels. The pool and held-out truth are identical at every budget.
No class-based repairs or favorable seed selection. Fit SSME for 100 updates
with default Scott bandwidth and weight 10. Evaluate Accuracy analytically;
this is the predeclared primary metric for label savings. Record all-model MAE,
selection regret and tie-aware top-1 rate. Include the mean-probability posterior
baseline, clamping only observed labels just as in SSME.

Primary comparison: per-seed mean-model Accuracy MAE of SSME20 minus
labeled-only50 on held-out truth, separately for checkpoints and transfer.
Secondary: SSME20 minus labeled-only100 and mean-probability20. One-sided
Bonferroni t upper bounds control a declared family of six comparisons
(2 pipelines × 3 comparators), alpha=0.05. A negative bound supports lower
mean error for this protocol; a positive bound is inconclusive, not equivalence.
Report both adjusted upper bounds and unadjusted two-sided 95% t intervals,
effective paired seeds and failure counts. A label-saving claim is restricted
to the tested budget/metric/data. Seeds share a finite dataset; these are
conditional mask/split results, not independent external populations.

## Multiclass and redundancy stress

Use sklearn digits, classes 0–2 and 0–9. Split train/evaluation once using
stratification and random_state=2718. Fit preprocessing and six distinct
classifiers only on train. Save the frozen probabilities, row IDs, model
specifications and input hash. For seeds 100–109 choose a 240-row pool;
expose nested 20/60 labels with uniform sampling, retaining missing-class
failures. Hold truth outside estimation until scoring.

Compare three input sets: the first three diverse models, all six diverse
models, and the first three each repeated twice (six score views). Duplicate
views are an intentional adversarial diagnostic, not six independently trained
models. Always score the same original three candidates to avoid changing the
selection task. Report Accuracy and macro OvR AUC, estimated with 30 latent-label
draws for AUC, with valid draw counts. Compare SSME, labeled-only and mean
probability; record ALR dimension, bandwidth, posterior entropy, max probability,
final update and missing-class failures. Use 100 fixed updates. No claim of
universal gains and no hyperparameter selection using hidden evaluation labels.

## Profiles and provenance

Pilot profiles exist only to validate execution and estimate cost. Full runs
write separate directories, immutable input/protocol hashes, per-condition JSON
and aggregate CSV/plots. Resume is allowed only for matching configuration,
protocol, input and implementation fingerprints. Preserve unsuccessful runs.
The delivered summary must identify exactly which profile actually finished.
