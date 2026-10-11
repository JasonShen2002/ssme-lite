# Current checkout extensions

These additions are unreleased source-checkout work dated 2026-10-11. Install
this repository with `python -m pip install -e .`; the previously published
0.1.0 wheel does not contain the new model-file CLI.

## Four deliverables

| Addition | Entry point | Evidence |
| --- | --- | --- |
| Performance and parallelism | `python experiments/extensions/run.py --study performance --profile full` | Independent processes, phase times, whole-worker peak RSS, numerical equivalence |
| Model files to report | `python examples/model_files.py`, then the [CLI workflow](cli.md) | Partial labels, explicit class axis, trusted files, batched predictions, input hashes |
| Label efficiency | `python experiments/extensions/run.py --study labels --profile full` | Two frozen medical-model pipelines, 30 seeds, fixed pool and held-out truth, nested budgets |
| Multiclass/redundancy stress | `python experiments/extensions/run.py --study stress --profile full` | Frozen digits probabilities; 3/10 classes; diverse and duplicated views; fixed candidate set |

The scripts, fixed protocol, frozen inputs and results are under
`experiments/extensions/` in the source repository. `RESULTS.md` there records
the completed results; `README.md` includes local and optional CPU-server
commands. Each full result directory retains a source snapshot and manifest.

## Interpretation

Label savings are tested for Accuracy estimation error on the same held-out
set at every label budget. The primary comparison is SSME with 20 labels versus
labeled-only with 50 labels. Six predeclared comparisons use one-sided Bonferroni
t upper bounds; failure to find lower error does not prove equivalence.
The repetitions share a finite dataset and concern conditional split/mask
variability, not independent external populations.

The stress experiment evaluates the same original three candidates in every
regime. Extra diverse models or exact duplicate score views only change the
information supplied to SSME. Missing-class runs remain failures rather than
being repaired using hidden labels. Inspect effective seed counts, AUC validity
and convergence diagnostics alongside the mean errors.

Performance numbers apply to the recorded hardware, workload and iterations.
Whole-worker RSS includes imported libraries and report generation. More workers
may be slower when only a few class fits are available. A server profile extends
to 10,000 rows; an available command is not evidence that it has been executed.


## Completed full-run findings

- 42 performance conditions completed; 5,000 controlled rows / 100 updates took
  254.07 seconds on average. Tested two-worker configurations were slower; outputs agreed exactly.
- 180 label-efficiency conditions completed. At 20 labels, mean Accuracy MAE
  decreased from 5.84 to 3.21 pp (checkpoints) and 7.35 to 4.60 pp (transfer),
  but the predeclared 20-versus-50-label replacement criterion did not pass.
- 120 stress conditions completed, including 27 explicit missing-class SSME
  failures. Diverse auxiliary views helped the three-class setting; duplicated
  views added no new information. Ten-class/60-label SSME errors exceeded the
  labeled-only baseline.

The results support a working, audited tool and clarify its limits. They do not
establish 60% label savings, general multiclass superiority, or parallel speedup.
