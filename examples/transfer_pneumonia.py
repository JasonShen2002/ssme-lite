"""ImageNet ResNet18 features, sklearn heads, then SSME on PneumoniaMNIST.

This is the transfer-learning example. It does not reuse the official
PneumoniaMNIST checkpoints. The backbone stays frozen; only the sklearn heads
see PneumoniaMNIST labels, and only the official training split.

Requires ``pip install -e '.[transfer]'`` and the npz from
``examples/pneumonia_official.py``.
"""
from pathlib import Path
import argparse
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "pneumoniamnist"
FEATURE_FILE = DATA / "resnet18_features.npz"


def transfer_heads(seed=0):
    """Fixed heads. No validation search, so the val split stays unused."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=1),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=seed)),
        "extra_trees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=80, random_state=seed),
    }


# Nested subsets declared before any SSME fit. Each step adds a different head.
TRANSFER_SUBSETS = {
    "2": ["logistic", "random_forest"],
    "3": ["logistic", "random_forest", "mlp"],
    "4": ["logistic", "random_forest", "mlp", "extra_trees"],
    "5": ["logistic", "random_forest", "mlp", "extra_trees", "hist_gradient_boosting"],
}


def load_arrays():
    import sys
    path = DATA / "pneumoniamnist.npz"
    if not path.exists():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pneumonia_official import download_dataset
        download_dataset()
    with np.load(path) as archive:
        return (archive["train_images"], archive["train_labels"].astype(int).reshape(-1),
                archive["test_images"], archive["test_labels"].astype(int).reshape(-1))


def extract_features(images, batch_size=64, device="auto"):
    import torch
    import torch.nn.functional as functional
    from torchvision.models import resnet18, ResNet18_Weights
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    backbone.requires_grad_(False)
    backbone.to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    print(f"Feature extraction device: {device}", flush=True)
    features = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(images[start:start + batch_size])).float()
            if batch.ndim == 3:
                batch = batch.unsqueeze(1)
            batch = batch.repeat(1, 3, 1, 1).to(device) / 255.0
            batch = functional.interpolate(batch, size=224, mode="bilinear", align_corners=False)
            features.append(backbone((batch - mean) / std).cpu().numpy())
    return np.concatenate(features)


def cached_features(batch_size=64, device="auto"):
    if FEATURE_FILE.exists():
        with np.load(FEATURE_FILE, allow_pickle=False) as archive:
            return archive["X_train"], archive["y_train"], archive["X_test"], archive["y_test"]
    train_images, y_train, test_images, y_test = load_arrays()
    X_train = extract_features(train_images, batch_size=batch_size, device=device)
    X_test = extract_features(test_images, batch_size=batch_size, device=device)
    FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURE_FILE, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return X_train, y_train, X_test, y_test


def fit_heads(X_train, y_train, seed=0):
    models = transfer_heads(seed)
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def prediction_matrix(models, X_test):
    from ssme_lite import PredictionMatrixGenerator
    generator = PredictionMatrixGenerator(models)
    scores = generator.fit_transform(X_test)
    return scores, generator.model_names_


def demo_split(scores, y_test, names, n_labeled=20, n_unlabeled=400, seed=0):
    """One estimation/held-out split through SSMEEstimator and a report."""
    from ssme_lite import SSMEEstimator, SemiSupervisedSplit
    splitter = SemiSupervisedSplit(n_labeled, n_unlabeled, seed)
    parts = splitter.indices(len(y_test))
    if splitter.labeled_class_count(y_test, parts) != 2:
        raise RuntimeError("labeled draw missed a class; choose another seed")
    partial = splitter.partial_labels(y_test, parts)
    estimator = SSMEEstimator(max_iter=100, early_stopping=False, random_state=seed)
    estimator.fit(scores[parts["estimation"]], partial, model_names=names)
    report = estimator.report(
        n_draws=100, title="PneumoniaMNIST transfer · ResNet18 features",
        primary_metric="auc", sample_ids=parts["estimation"],
    )
    return estimator, report, parts


def repeated_benchmark(scores, y_test, names, output):
    from ssme_lite.experiments import benchmark
    protocol = json.loads((DATA / "protocol.json").read_text(encoding="utf-8"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "models.json").write_text(json.dumps({
        "backbone": "torchvision ResNet18_Weights.DEFAULT, frozen, fc removed",
        "feature_dim": 512,
        "input": "28x28 grayscale resized bilinear to 224 and ImageNet-normalized",
        "heads": TRANSFER_SUBSETS["5"],
        "train_split": "official train",
        "evaluation_pool": "official test",
        "subsets": TRANSFER_SUBSETS,
    }, indent=2), encoding="utf-8")
    for count, subset in TRANSFER_SUBSETS.items():
        columns = [names.index(name) for name in subset]
        destination = output / f"nl20_nu{protocol['n_unlabeled']}_m{count}"
        if (destination / "summary.csv").exists():
            print(f"skip existing {destination}", flush=True)
            continue
        benchmark(
            scores[:, columns], y_test, subset, destination, seeds=protocol["seeds"], n_labeled=20,
            n_unlabeled=protocol["n_unlabeled"], n_draws=100, max_iter=100,
            contribution=(int(count) == 5), dataset="PneumoniaMNIST_transfer",
            n_jobs=min(4, __import__("os").cpu_count() or 1),
        )
    for n_labeled in protocol["n_labeled"]:
        if n_labeled == 20:
            continue
        destination = output / f"nl{n_labeled}_nu{protocol['n_unlabeled']}_m5"
        if (destination / "summary.csv").exists():
            continue
        benchmark(
            scores, y_test, names, destination, seeds=protocol["seeds"],
            n_labeled=n_labeled, n_unlabeled=protocol["n_unlabeled"], n_draws=100,
            max_iter=100, dataset="PneumoniaMNIST_transfer",
            n_jobs=min(4, __import__("os").cpu_count() or 1),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "results" / "pneumonia_transfer"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    X_train, y_train, X_test, y_test = cached_features(device=args.device)
    models = fit_heads(X_train, y_train, seed=0)
    scores, names = prediction_matrix(models, X_test)
    np.savez_compressed(DATA / "transfer_scores.npz", scores=scores, y=y_test, model_names=np.asarray(names))
    _, report, _ = demo_split(scores, y_test, names)
    report.save(Path(args.output) / "demo_report")
    print(report.ranking("auc").to_string(index=False))
    if not args.skip_benchmark:
        repeated_benchmark(scores, y_test, names, args.output)


if __name__ == "__main__":
    main()
