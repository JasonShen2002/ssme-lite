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

The real-data experiments demonstrate fitting pools of 420 or 1,020 rows. CivilComments comes from a much larger source dataset, but its source size must not be quoted as a jointly fitted pool size. Current evidence establishes parallel functionality and reproducibility; it is not a measured large-scale performance claim.
