# Contributing

Use Python 3.10 or newer and run commands from the repository root.

```bash
python -m pip install -e '.[dev,docs]'
python -m pytest -q
python examples/quickstart.py
mkdocs build --strict
```

Keep the public imports in `ssme_lite` stable. Document new parameters and update runnable examples when behavior changes. Use small synthetic inputs for tests; keep heavyweight datasets and model downloads out of the default test suite. The optional official-reference audit skips when its external dependencies or reference tree are unavailable.

Describe the behavior change and validation in pull requests. Do not commit datasets, model weights, credentials, generated distribution files, or local environments. See `docs/releasing.md` for release verification.
