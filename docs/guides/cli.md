# Command-line evaluation

The CLI accepts a JSON configuration and can evaluate saved probabilities without model objects.

## 1. Save inputs

```python
import numpy as np

np.savez("predictions.npz", scores=scores, y=y_partial,
         model_names=np.asarray(model_names, dtype=str))
```

Use the [input contract](inputs.md). Store names as a Unicode array, not a pickled object array. Unknown labels are `-1`.

## 2. Write `evaluation.json`

```json
{
  "source": "npz",
  "mode": "evaluate",
  "data": "predictions.npz",
  "output": "results/evaluation",
  "estimator": {"max_iter": 100, "random_state": 0},
  "report": {"n_draws": 100, "primary_metric": "accuracy"}
}
```

## 3. Run

```bash
ssme-lite evaluation.json
```

Paths resolve relative to the configuration file, not the shell's current directory. The command prints the output directory and saves the report plus `input_config.json`.

For a downloadable example, run `python examples/quickstart.py` from the checkout: it also writes `results/quickstart/predictions.npz` and `evaluation.json`. Then run `ssme-lite results/quickstart/evaluation.json`.

## Benchmark configurations

The repository's `configs/` also includes synthetic and released-score benchmark configurations. Those run repeated experimental comparisons and have different data requirements and costs. The small NPZ example above is the recommended first CLI run. See `ssme-lite --help` for the command syntax.
