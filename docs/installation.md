# Installation

## Core package

Use Python 3.10 or newer, preferably in a dedicated virtual environment:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ssme-lite
python -c "from ssme_lite import SSMEEstimator; print(SSMEEstimator())"
```

The core uses NumPy, SciPy, pandas, scikit-learn, Matplotlib, and joblib. It does not require PyTorch, model weights, or network access at evaluation time.

## Optional features

```bash
# Reference-code bandwidth rule
python -m pip install 'ssme-lite[official]'
# Frozen-backbone transfer-learning dependencies
python -m pip install 'ssme-lite[transfer]'
```

`official` adds statsmodels; it does not download the original research repository. `transfer` adds PyTorch, torchvision, and Pillow; experiment scripts remain in the source repository.

## Tutorials and development

```bash
git clone https://github.com/JasonShen2002/ssme-lite.git
cd ssme-lite
python -m pip install -e '.[dev,docs]'
python examples/quickstart.py
mkdocs serve
```

Run repository commands from the checkout root unless a tutorial says otherwise. Notebooks additionally require Jupyter: `python -m pip install jupyterlab`. The full PneumoniaMNIST inference path also requires `ai-edge-litert`; availability depends on your Python version and platform. Its saved-probability path uses only the core package.

## Conda environments

You can install the PyPI distribution inside an activated conda environment:

```bash
conda create -n ssme python=3.11 pip
conda activate ssme
python -m pip install ssme-lite
```

A draft recipe is included in `conda/`; this does not mean the package has been accepted into conda-forge.
