# Parallelism and workload sizing

SSME-Lite exposes parallelism where independent work already exists:

| Component | Setting | Work distributed |
| --- | --- | --- |
| `PredictionMatrixGenerator` | `n_jobs` | Candidate-model prediction calls |
| `SSMEEstimator` | `n_jobs` | Class-conditional KDE fits |
| `benchmark` | `n_jobs` | Independent random-seed runs |
| `SSMEEstimator` | `batch_size` | Number of rows in each posterior-evaluation batch |

```python
estimator = SSMEEstimator(n_jobs=2, batch_size=512, random_state=0)
```

In a JSON evaluation configuration, put these options in `estimator`. For a repeated benchmark, put `n_jobs` inside `benchmark`. The benchmark parallelizes seeds; it does not forward that setting as per-estimator worker count. Prefer one main level of parallelism to avoid CPU oversubscription.

For a binary task there are only two class-density fits, so more workers do not imply proportional speedup. Smaller posterior batches bound temporary query arrays but do not make fitting out-of-core: inputs, responsibilities, and KDE training data remain resident in memory. Kernel-density evaluation can become expensive as the fitted sample count and model dimension increase.

Start with the [small runnable examples](../quickstart.md), record wall time and peak memory on your own workload, and enable contribution diagnostics after validating the main fit. Contribution adds one refit per removed candidate.

The real-data experiments demonstrate fitting pools of 420 or 1,020 rows. CivilComments comes from a much larger source dataset, but its source size must not be quoted as a jointly fitted pool size. The published 0.1.0 evidence establishes parallel functionality and reproducibility. The current source checkout adds the measured extension below.


## Measured checkout extension (2026-10-11)

The full grid completed 42 independent worker conditions. Controlled binary fits
with three views and 100 updates took 2.73/10.62/42.63/254.07 seconds at
500/1,000/2,000/5,000 rows (means of three repetitions). Whole-worker peak RSS
was roughly 196–199 MiB, dominated by libraries/report construction at this
scale; this is not fit-only incremental memory.

At 1,000 rows, two class-fit workers took 11.56 seconds versus 10.62 with one.
Cheap sklearn prediction also slowed down with two workers. Enable parallelism
after measuring the relevant workload; the tested configurations do not justify
a speedup claim. All worker/batch comparisons had maximum absolute output
difference 0.0.

Raw measurements and plots are in `experiments/extensions/results/performance/full/`.
The [extension guide](extensions.md) describes the full experiment. A 10,000-row
CPU-server profile is provided but remains unexecuted; the measured limit here
is 5,000 controlled rows and 624 real checkpoint rows.
