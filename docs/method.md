# Method and interpretation

## Algorithm

SSME-Lite implements a semi-supervised evaluation workflow based on Shanmugam et al. (2025). Candidate classifiers are already trained. Their probability vectors form the inputs to the evaluator.

1. Clip and normalize probabilities, then apply an additive log-ratio transformation with the final class as reference.
2. Initialize latent classes by sampling from the mean candidate probabilities; clamp observed labels.
3. Fit weighted class-conditional Gaussian kernel densities and update class priors.
4. Update class posteriors and clamp observed labels again.
5. Repeat for the requested number of iterations, then evaluate candidates under the fitted posterior.

Observed samples have default weight 10; unlabeled samples have weight 1. The default is 100 fixed iterations (`early_stopping=False`). With early stopping enabled, the largest posterior change is compared with `tol=1e-3`. `history_`, `n_iter_`, `converged_`, and `stop_reason_` expose fit diagnostics. Fixed-bandwidth KDE updates are an EM-style procedure; the recorded objective is a diagnostic rather than a promised monotonic optimization trace.

## Bandwidth choices

- `"scott"` (default): mean feature standard deviation multiplied by the multivariate Scott sample-size factor, with a minimum scalar bandwidth of `1e-3`.
- `"official"`: the smallest per-feature statsmodels normal-reference bandwidth divided by four, following the public reference implementation; install the `official` extra.
- A positive number: use that fixed scalar bandwidth in ALR space.

`official` identifies a reference-code bandwidth rule, not bitwise equivalence of every experiment. For example, the package uses equal-width ECE bins, whereas the paper reports equal-frequency bins.

## What is evaluated?

By default, reports estimate performance over the fitted estimation pool, including observed and unobserved rows. `target="unlabeled"` restricts the target to unlabeled rows. Held-out benchmark metrics are computed separately and must be labeled as such. The real-data tutorials demonstrate binary tasks; multiclass input support alone is not evidence of benchmark performance on multiclass datasets.

The model-influence diagnostic is a controlled leave-one-model-out sensitivity analysis. See [report interpretation](guides/reports.md) for interval and influence definitions.

## Attribution

SSME-Lite is an independent packaging and workflow implementation, not the authors' official distribution.

Divya Shanmugam, Shuvom Sadhuka, Manish Raghavan, John Guttag, Bonnie Berger, and Emma Pierson. **Evaluating Multiple Models Using Labeled and Unlabeled Data.** Advances in Neural Information Processing Systems 38, 2025, pp. 30612–30648. [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2c0e78cc177dfeca260ef990a8c99209-Abstract-Conference.html) · [Official code](https://github.com/divyashan/SSME).

The software is MIT licensed. Dataset and checkpoint terms remain those of their original providers; these assets are not bundled in the Python distribution.
