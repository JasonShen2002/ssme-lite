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

## Model names and model objects

The candidate list is the `model_names` array saved inside the NPZ file, aligned with its model axis. The published 0.1.0 CLI uses this probability-input contract. Version 0.2.0 also supports fitted model files, as described below. See [input adapters](inputs.md) for the Python API.

## Benchmark configurations

The repository's `configs/` also includes synthetic and released-score benchmark configurations. Those run repeated experimental comparisons and have different data requirements and costs. The small NPZ example above is the recommended first CLI run. See `ssme-lite --help` for the command syntax.

## Fitted-model workflow (0.2.0)

Version 0.2.0 supports `source: "models"`. Install with
`python -m pip install ssme-lite==0.2.0`; run the repository example below
from a source checkout.

```bash
python examples/model_files.py
ssme-lite check-config results/model_files/config.json
ssme-lite results/model_files/config.json --trust-model-files
```

The example trains logistic regression and a random forest on a disjoint
training split, saves the fitted models, and exports only 20 visible evaluation
labels. Model training happens in the preparation example; the evaluation CLI
never calls `fit` on a candidate classifier.

Create a template for your own files:

```bash
ssme-lite init-config my_evaluation/config.json
```

This refuses to overwrite an existing file. Fill its relative data/model paths:

```json
{
  "source": "models",
  "mode": "evaluate",
  "data": "evaluation.npz",
  "models": [
    {"name": "logistic", "path": "models/logistic.joblib"},
    {"name": "forest", "path": "models/forest.joblib"}
  ],
  "prediction": {"n_jobs": 2, "batch_size": 512},
  "estimator": {"max_iter": 100, "random_state": 0},
  "report": {"target": "unlabeled", "n_draws": 100},
  "output": "report"
}
```

The NPZ must contain `X` (numeric feature matrix), `y` (integer positions into
`classes`, with `-1` for unknown), and `classes` (ordered class values matching
the first model's `classes_`). Optional `sample_ids` identifies each evaluation
row. Arrays must load with `allow_pickle=False`. Every class needs at least one
visible label. Models may have different probability column orders; the
prediction generator aligns them. The explicit input `classes` prevents
silently interpreting label indices in the wrong class order.

`check-config` checks paths, model names and the NPZ input contract without
loading any model files. Model fittedness, class compatibility and prediction
outputs are checked during `run`. joblib can execute code when loading, so
`--trust-model-files` is required for files you created or otherwise trust.
The flag grants this file-loading capability, not permission to train or
retrieve models remotely.

Outputs include the usual report, `input_config.json`, a frozen
`predictions.npz`, and `input_provenance.json` with input/model SHA256 hashes.
`ssme-lite run config.json` is an explicit alias for the original positional
syntax. `source: "npz"` continues to work without the trust flag.
