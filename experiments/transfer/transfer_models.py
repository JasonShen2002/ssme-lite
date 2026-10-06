"""Fit sklearn heads on frozen ResNet18 features.

The short package demo, from features through the report, is
``transfer_ssme.py``. Heads see only the official training split.
Hyperparameters are fixed, so the validation split is unused.
"""
from pathlib import Path
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from transfer_features import cached_features

SCORES = HERE / "data" / "transfer_scores.npz"
HEADS = (
    "logistic",
    "random_forest",
    "mlp",
    "extra_trees",
    "hist_gradient_boosting",
)
# Nested subsets fixed before any SSME fit. Each step adds a different head.
TRANSFER_SUBSETS = {
    "2": ["logistic", "random_forest"],
    "3": ["logistic", "random_forest", "mlp"],
    "4": ["logistic", "random_forest", "mlp", "extra_trees"],
    "5": list(HEADS),
}


def transfer_heads(seed=0):
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


def fit_heads(X_train, y_train, seed=0):
    models = transfer_heads(seed)
    for name, model in models.items():
        print(f"fitting {name}", flush=True)
        model.fit(X_train, y_train)
    return models


def prediction_matrix(models, X_test):
    from ssme_lite import PredictionMatrixGenerator
    generator = PredictionMatrixGenerator({name: models[name] for name in HEADS})
    scores = generator.fit_transform(X_test)
    if list(generator.model_names_) != list(HEADS):
        raise RuntimeError("prediction matrix columns are not the predeclared head order")
    return scores, generator.model_names_


def save_scores(scores, y, names, destination=SCORES):
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, scores=scores, y=y, model_names=np.asarray(names))
    note = destination.with_name("transfer_models.json")
    note.write_text(json.dumps({
        "dataset": "PneumoniaMNIST",
        "backbone": "torchvision ResNet18_Weights.DEFAULT, frozen, fc removed",
        "feature_dim": 512,
        "input": "28x28 grayscale resized bilinear to 224 and ImageNet-normalized",
        "heads": list(names),
        "train_split": "official train",
        "evaluation_pool": "official test",
    }, indent=2), encoding="utf-8")
    return destination


def repeated_benchmark(scores=None, y=None, names=None):
    """10 seeds, n_l = 20/50/100, and the nested 2–5 head subsets."""
    import os
    from ssme_lite.benchmark import benchmark
    if scores is None:
        with np.load(SCORES) as archive:
            scores, y = archive["scores"], archive["y"]
            names = archive["model_names"].tolist()
    output = HERE / "results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "models.json").write_text(json.dumps({
        "backbone": "torchvision ResNet18 ImageNet-1k V1, frozen, fc removed",
        "feature_dim": 512,
        "input": "28x28 grayscale resized bilinear to 224 and ImageNet-normalized",
        "heads": TRANSFER_SUBSETS["5"],
        "train_split": "official train",
        "evaluation_pool": "official test",
        "subsets": TRANSFER_SUBSETS,
        "n_labeled": [20, 50, 100],
        "n_unlabeled": 400,
        "seeds": list(range(10)),
    }, indent=2), encoding="utf-8")
    n_jobs = min(4, os.cpu_count() or 1)
    for count, subset in TRANSFER_SUBSETS.items():
        columns = [names.index(name) for name in subset]
        destination = output / f"nl20_nu400_m{count}"
        if (destination / "summary.csv").exists():
            print(f"skip existing {destination}", flush=True)
            continue
        benchmark(
            scores[:, columns], y, subset, destination, seeds=list(range(10)), n_labeled=20,
            n_unlabeled=400, n_draws=100, max_iter=100,
            contribution=(int(count) == 5), dataset="PneumoniaMNIST_transfer", n_jobs=n_jobs,
        )
    for n_labeled in (50, 100):
        destination = output / f"nl{n_labeled}_nu400_m5"
        if (destination / "summary.csv").exists():
            print(f"skip existing {destination}", flush=True)
            continue
        benchmark(
            scores, y, names, destination, seeds=list(range(10)), n_labeled=n_labeled,
            n_unlabeled=400, n_draws=100, max_iter=100,
            dataset="PneumoniaMNIST_transfer", n_jobs=n_jobs,
        )
    print(output)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="store_true", help="run the saved scores through the 10-seed comparisons")
    args = parser.parse_args()
    if args.suite:
        repeated_benchmark()
        return
    X_train, y_train, X_test, y_test = cached_features()
    scores, names = prediction_matrix(fit_heads(X_train, y_train), X_test)
    path = save_scores(scores, y_test, names)
    print(path)
    print(f"scores {scores.shape}")


if __name__ == "__main__":
    main()
