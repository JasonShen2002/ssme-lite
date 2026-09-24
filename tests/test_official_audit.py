"""Optional official-dependency tests for the read-only comparison adapter."""
import sys
from pathlib import Path
import numpy as np
import pytest


def test_profile_capture_does_not_change_official_result():
    pytest.importorskip("statsmodels")
    pytest.importorskip("composition_stats")
    pytest.importorskip("tqdm")
    root = Path(__file__).resolve().parents[1]
    if not (root / "official" / "SSME").exists():
        pytest.skip("official SSME tree is not part of this repository")
    from examples.audit_official_same_split import capture_official
    sys.path.insert(0, str(root / "official/SSME"))
    import model
    rng = np.random.default_rng(2)
    positive = rng.uniform(.05, .95, (30, 2))
    p = np.stack([1-positive, positive], axis=-1)
    before = p.copy()
    partial = np.full(30, -1, dtype=int)
    partial[:10] = np.arange(10) % 2
    captured = capture_official(model, p, partial, 2, 1, 7)
    np.random.seed(7)
    groups = np.repeat("global", 30)
    plain = model.SSME_KDE(
        (p[:10, :, 1].copy(), [groups[:10]], partial[:10]),
        (p[10:, :, 1].copy(), [groups[10:]], partial[10:]),
        captured["config"])
    assert plain.to_dict(orient="records") == captured["native_metrics"]
    assert captured["cluster_preds"].shape == (30, 2)
    np.testing.assert_allclose(captured["cluster_preds"].sum(1), 1)
    np.testing.assert_array_equal(p, before)


def test_binary_alr_sign_preserves_gaussian_distances():
    from ssme_lite.prediction import alr
    rng = np.random.default_rng(4)
    p1 = rng.uniform(.01, .99, (20, 3))
    z = alr(np.stack([1-p1, p1], axis=-1))
    official_z = np.log(p1/(1-p1))
    np.testing.assert_allclose(z, -official_z)
    np.testing.assert_allclose(np.linalg.norm(z[:, None]-z, axis=-1),
                               np.linalg.norm(official_z[:, None]-official_z, axis=-1))
