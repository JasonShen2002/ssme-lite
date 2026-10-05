# Troubleshooting

| Symptom | Resolution |
| --- | --- |
| A class has no observed example | Obtain at least one observed label for every class. For retrospective benchmarks, record any skipped draws rather than silently repairing them. |
| Probability validation fails | Use finite probabilities in `[0, 1]`, with full class vectors summing to one. Convert logits before fitting. |
| A model predicts a different class set | Align class sets before evaluation; the generator only reorders matching sets. |
| `bandwidth="official"` cannot import statsmodels | Install `ssme-lite[official]`. |
| AUC/AP has few valid draws | Inspect target size, class balance, and `valid_draws`; these metrics need every class represented. |
| Contribution reporting is slow | Start with `contribution=False`; it adds one fit per model. |
| Imports work in the checkout but fail elsewhere | Install the package with the Python interpreter you are running; test from outside the repository. |
| HTML opens as source on GitHub | Download and open it locally, or use the published experiment report links. |
| Tutorial weights or input files are missing | Follow that tutorial's data setup. Dataset files and checkpoints are not included in the wheel. |
| Parallel runs oversubscribe the CPU | Start with `n_jobs=1`, especially when running multiple experiment seeds in parallel. |

For reproducibility, record the package and dependency versions, seed, input ordering, visible-label mask, bandwidth, iteration count, and report settings. A fixed seed alone does not guarantee identical results across all dependency versions.

Report issues at <https://github.com/JasonShen2002/ssme-lite/issues>, including a small synthetic reproducer and the full error message. Do not attach private data or credentials.
