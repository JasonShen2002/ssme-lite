# Submission extensions

This directory delivers the four additions requested on 2026-10-11:
performance/parallel benchmarks, a model-file CLI example, label-efficiency
experiments, and multiclass/redundancy stress tests. All code, inputs, and
results are inside the final `ssme-lite` repository. The previous state is
saved by Git commit `ec4b72c`.

Read `PROTOCOL.md` for the hypotheses, targets and fixed choices established
before the new full runs. `RESULTS.md` records the completed measurements.
These results extend, rather than replace, `experiments/REPORT.md`.

## Install and run

From the `ssme-lite` repository root, Python 3.10+ on macOS or Linux:

```bash
python -m pip install -e '.[dev]'
python experiments/extensions/prepare_digits.py
python experiments/extensions/run.py --study all --profile pilot --jobs 2 --output experiments/extensions/recomputed_results
python experiments/extensions/analyze.py --profile pilot --results experiments/extensions/recomputed_results
python experiments/extensions/run.py --study all --profile full --jobs 2 --output experiments/extensions/recomputed_results
python experiments/extensions/analyze.py --profile full --results experiments/extensions/recomputed_results
```

The bundled frozen digits inputs permit skipping the preparation step. That
step retrains six classifiers on the training split and may produce different
bytes on different dependency versions; retain the new manifest and use a new
result directory when regenerating it. The two image prediction inputs come
from `reproduction/inputs/` and are verified against its hash manifest.

Full profile: 30 label-efficiency seeds per image pipeline, 10 stress seeds per
class-count setting, and three performance repetitions per condition. Each fit
uses 100 fixed updates. Pilot runs use five updates and only two statistical
seeds; they verify execution and are not the scientific results.

Each condition runs in a separate subprocess. `--jobs` controls statistical
experiment concurrency; performance conditions always run sequentially and
limit native numerical-library threads to one. Do not run other compute-heavy
workloads concurrently when interpreting performance numbers. Performance RSS
is the peak for the entire worker, including imports, prediction training or
report construction; it is not fit-only incremental memory.

The commands above use a new output root to preserve the bundled measurements. The final analyzer has a post-run presentation update; its receipt is separate from the original measurement source snapshot. To regenerate only delivered summaries, use `python experiments/extensions/analyze.py --profile full`.

Resume a new run with the same command. Completed conditions, including timeouts and
failures, remain recorded and are not silently retried. Configuration, source,
input or environment changes require a new `--output`. To retry with a larger
time budget, use a new output root and retain the original failures.

Audit the delivered full run without refitting:

```bash
python experiments/extensions/audit_results.py
```

## Optional server scale extension

No GPU is required. The server profile includes the full performance grid plus
10,000-row controlled fits, still with 100 fixed updates and three repetitions:

```bash
python experiments/extensions/run.py --study performance --profile server \
  --timeout 3600 --output experiments/extensions/server_results
python experiments/extensions/analyze.py --profile server \
  --results experiments/extensions/server_results
```

Start with the pilot to size memory/time for the target machine. Copy the
entire repository, including frozen NPZ inputs. Report hardware and timings
from that server separately from this laptop. The manifest stores Python,
dependency versions, OS, architecture, logical CPU count and all code/input
hashes. A source snapshot accompanies each full result directory.

## Model-file example

```bash
python examples/model_files.py --output experiments/extensions/model_cli_demo
ssme-lite check-config experiments/extensions/model_cli_demo/config.json
ssme-lite experiments/extensions/model_cli_demo/config.json --trust-model-files
```

This uses locally generated trusted joblib files. Recreate these files on your
own sklearn version if loading an older artifact is incompatible. Only partial
evaluation labels are passed to SSME. The resulting `report/report.html` opens
offline; the report is a single demonstration, not the statistical experiment.

## Artifacts

- `inputs/`: frozen digits probability tensors, original row IDs, train/eval split and model specifications.
- `results/<study>/<profile>/manifest.json`: expected/completed conditions, provenance and result hashes.
- `jobs/`, `runs/`: exact per-condition configuration, measurements, diagnostic state and logs.
- `source_snapshot/`: the source and protocol used by the corresponding run.
- `summary.csv`, `per_seed.csv` or `per_run.csv`: summaries and raw measurements.
- `paired_comparisons.csv`: predeclared one-sided label-saving comparisons.
- `regime_pairs.csv`: same-seed effects of diverse or duplicated auxiliary views.
- `equivalence.csv`: numerical agreement across performance repetitions, worker counts and batch sizes.
- `audit.json`: disjoint/nested/fixed-target checks or parallel agreement checks.

The stress test uses the same original three candidates when scoring every
input regime. Repeated views are intentionally duplicated evidence, never
reported as independently trained models. The fixed-pool target is the main
stress interpretation; the three-class held-out set has only 29 rows and is
retained as a limited supplementary result. Missing-class fits are recorded;
AUC is undefined when the observed sample lacks a class. Consult `status.csv`
and effective seed counts when comparing means.
