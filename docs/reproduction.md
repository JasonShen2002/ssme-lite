# Reproduce the course report

The submission has two complementary routes: recompute the statistical results from frozen model predictions, or regenerate predictions from images and fitted classifiers. The first route is the recommended starting point for checking the paper. It needs only CPU computation and the published core package.

## 1. Install in a clean environment

Download or clone the submission repository, then open a terminal at its root. Use Python 3.11 to match the recorded verification environment.

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Install the recorded core environment:

```bash
python -m pip install -r reproduction/requirements.txt
```

The code uses platform-independent Python paths. The verification receipt records the platform actually tested; the commands are not a claim that every operating system was separately tested.

## 2. Recompute the main table and image-study figures

```bash
python reproduction/run.py --jobs 2
```

The two NPZ files in `reproduction/inputs/` contain 624 official PneumoniaMNIST test rows, full ground truth for benchmark scoring, and the ordered model names. They preserve the predictions used by the paper: seven released checkpoints and five frozen-ResNet18 transfer heads. They contain no images or model weights. See `reproduction/inputs/manifest.json` for hashes and provenance.

The runner recomputes **40 seed-level fits**: ten checkpoint runs with 20 labels, and ten transfer runs at each of 20, 50, and 100 labels. Each fit uses 400 unlabeled examples, 100 fixed EM iterations, and 100 posterior label draws. At 20 labels it also recomputes all leave-one-model-out fits used for influence analysis. Split sampling never uses hidden labels to repair an annotation draw.

Outputs go to `results/reproduction/`; existing paper results remain intact. Verification checks model-level estimates and errors, split indices, ranking statistics, and posterior-TV influence against the reference experiment files. Numeric checks use an absolute tolerance of `1e-10`; a mismatch raises an error rather than silently updating the reference. `verification.json` records input hashes, versions, timings, and maximum differences. Set `--jobs 1` for serial execution.

Expected held-out Accuracy MAE (percentage points):

| Pipeline | Visible labels | Labels only | SSME-Lite |
|:--|--:|--:|--:|
| Seven checkpoints | 20 | 5.70 | 2.22 |
| Five transfer heads | 20 | 7.11 | 4.16 |
| Five transfer heads | 50 | 5.60 | 3.19 |
| Five transfer heads | 100 | 4.28 | 3.45 |

This is a new SSME fit from fixed classifier outputs, not a rerun of classifier training or a redraw of plots from stored summary numbers.

## 3. Recompute the CivilComments comparison

Obtain the original prediction inputs using the [official SSME repository](https://github.com/divyashan/SSME). Preserve `inputs/CivilComments/<model>/labels.npy` and `preds.npy` for the seven models listed in the [CivilComments tutorial](tutorials/civilcomments.md). These external inputs are not bundled with the submission.

```bash
python -m pip install 'ssme-lite[official]==0.2.0'
python experiments/civilcomments/reproduce_official_split.py --inputs /path/to/SSME/inputs --output results/reproduction/civilcomments --jobs 2
```

Replace `/path/to/SSME/inputs` with the actual directory; quote paths containing spaces. The script collects 50 valid splits in increasing seed order, skipping visible-label draws missing a class. It uses 20 labels, 1,000 unlabeled comments, a 66,891-row holdout, reference-code bandwidth, and 20 EM iterations. It writes per-model errors, MAE tables, and figures. The seven-model SSME MAEs for Accuracy/AUC/AP/ECE are **2.27/3.28/9.41/2.12 pp**. ECE uses this package's equal-width bins.

## 4. Optional image-to-report rerun

For checkpoint inference:

```bash
python -m pip install 'ssme-lite[transfer]==0.2.0' ai-edge-litert
python experiments/pneumoniamnist/pneumonia_ssme.py
ssme-lite experiments/pneumoniamnist/config.json
```

For frozen-backbone features, classifier training, and evaluation:

```bash
python -m pip install 'ssme-lite[transfer]==0.2.0'
python experiments/transfer/transfer_ssme.py
ssme-lite experiments/transfer/config.json
```

These commands need additional downloads and dependencies. They write new predictions under each experiment's `data/` directory and regenerate the corresponding demonstration and main benchmark. Run them in a separate checkout if retaining the supplied reference outputs. The larger transfer `--suite` script skips result directories already present, so it is not the fresh-verification entry point.

Image resampling, neural-network runtimes, and classifier training can introduce numerical differences across platforms. The frozen-score route isolates SSME reproducibility from those upstream differences. The full image workflows and data sources are described in the [checkpoint](tutorials/pneumoniamnist.md) and [transfer](tutorials/transfer.md) tutorials.

## 5. Recorded verification

On 2026-10-06, the 40 image-study fits passed all numeric comparisons with zero observed differences. The 50 valid CivilComments splits also matched all saved model-level errors exactly. Receipts are in `reproduction/verification.json` and `reproduction/civilcomments-verification.json`. These runs verified macOS with the recorded dependencies.

The audit found that the previous checkpoint tables were not exactly reproducible from the retained predictions. Those tables are preserved under `experiments/archive/checkpoints_before_frozen_input_audit/`. The final checkpoint tables were refreshed from the bundled, hashed inputs and then confirmed by a second independent ten-seed refit. Updated Accuracy MAE is 5.70/2.22 pp; the main conclusion is unchanged.

To rebuild the paper figures and statistics in the submission archive, run `python report/build_report.py`. This step consumes saved benchmark tables; it does not perform model fits. Compile `report/main.tex` with a LaTeX installation (for example, `tectonic report/main.tex`).

## 6. Submission artifacts

The OpenI submission should contain source code, this guide, `reproduction/`, the reference experiment tables, and the final paper PDF and LaTeX source. Keep large downloadable weights and private environment files out of the repository. The course experiment project is [SSME-Lite-2026211240](https://openi.pcl.ac.cn/jasonshen/SSME-Lite-2026211240). Its address is included in the paper appendix. PyPI and GitHub provide the published package and development repository.
