# Release guide

## Validate the distribution

From a clean checkout of the intended release:

```bash
python -m pip install -e '.[dev,docs]' build twine
python -m pytest -q
python examples/quickstart.py
mkdocs build --strict
python -m build
python -m twine check --strict dist/*
```

Install the wheel in a fresh environment, then run `examples/quickstart.py`, `ssme-lite --help`, and the NPZ CLI example from outside the checkout. Verify that the HTML assets, version, and license are included. Rebuild a wheel from the sdist as well. Exclude experiment datasets, model weights, caches, and credentials from distributions.

## PyPI account setup

The owner must register at <https://pypi.org/account/register/>, verify their email, and enable two-factor authentication. Never put an API token in source code, issues, chat, or a workflow file.

For the first release, open <https://pypi.org/manage/account/publishing/> and add a pending GitHub publisher:

| Field | Value |
| --- | --- |
| PyPI project name | `ssme-lite` |
| Owner | `JasonShen2002` |
| Repository | `ssme-lite` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

A pending publisher does not reserve the project name. If PyPI rejects the name, stop and resolve ownership or choose a new distribution name before uploading.

## Publish with GitHub Actions

Push the reviewed source and `.github/workflows/publish.yml` to the repository. The workflow is manually dispatched, builds and validates the package, and uploads with OIDC Trusted Publishing from the `pypi` environment. Configure that environment in GitHub to match the publisher.

```bash
gh workflow run publish.yml --ref main
```

After success, install the exact version from PyPI in a fresh environment and run the quickstart. Record the published version and commit in a GitHub release. PyPI does not permit replacing an already uploaded filename; increment the version for a changed release.

## Documentation

`mkdocs build --strict` writes the local site to `site/`. The source documentation also renders directly on GitHub. A dedicated docs-deployment workflow can publish it under a separate path without replacing the existing experiment-report site.

## Conda

`conda/meta.yaml` is a starting recipe, not a published package. After the PyPI sdist exists, update its exact URL and SHA256, add recipe maintainers, run conda-build validation, and submit through conda-forge staged-recipes if desired.
