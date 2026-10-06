"""Regression: parallel seeds must reproduce the serial benchmark exactly."""
import numpy as np
import json
import pytest
from scipy.special import softmax
from ssme_lite.benchmark import benchmark
from sklearn.datasets import make_classification


@pytest.fixture(scope="module")
def toy():
    y = make_classification(n_samples=900, n_features=8, random_state=3)[1]
    rng = np.random.default_rng(7)
    # 3 models x 2 classes logits with varying quality, mapped to probabilities
    logits = rng.normal(0, 0.4, (len(y), 3, 2)) + np.eye(2)[y, None, :] * rng.uniform(0.5, 1.5, (1, 3, 2))
    return softmax(logits, axis=-1), y


def test_parallel_matches_serial(tmp_path, toy):
    scores, y = toy
    common = dict(names=["m1", "m2", "m3"], seeds=(0, 1, 2), n_labeled=10, n_unlabeled=200,
                  n_draws=5, max_iter=30, dataset="toy")
    serial_dir, par_dir = tmp_path / "serial", tmp_path / "parallel"
    benchmark(scores, y, output=serial_dir, n_jobs=1, **common)
    benchmark(scores, y, output=par_dir, n_jobs=3, **common)
    for name in ["per_seed.csv", "details.csv", "summary.csv", "splits.json", "run_status.json"]:
        left = (serial_dir / name).read_text()
        right = (par_dir / name).read_text()
        assert left == right, f"{name} differs between serial and parallel runs"


def test_benchmark_tolerance_is_forwarded_and_recorded(tmp_path, toy):
    scores, y = toy
    benchmark(scores, y, ["m1", "m2", "m3"], tmp_path,
              seeds=(0,), n_labeled=20, n_unlabeled=30,
              n_draws=2, max_iter=5, tol=1.1, labeled_weight=3, early_stopping=True)
    assert json.loads((tmp_path / "config.json").read_text())["tol"] == 1.1
    assert json.loads((tmp_path / "config.json").read_text())["labeled_weight"] == 3
    status = json.loads((tmp_path / "run_status.json").read_text())[0]
    assert status["converged"] and status["iterations"] == 1
