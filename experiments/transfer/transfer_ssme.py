"""Frozen ImageNet ResNet18, sklearn heads, then one SSME-Lite report.

    python experiments/transfer/transfer_ssme.py

The repeated 10-seed comparison is already at
``experiments/transfer/results/``.
"""
from pathlib import Path
import sys
from ssme_lite import PredictionMatrixGenerator, SSMEEstimator, SemiSupervisedSplit

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from transfer_features import cached_features
from transfer_models import HEADS, fit_heads, save_scores

REPORT = HERE / "results" / "demo_report"


def main():
    X_train, y_train, X_test, y_test = cached_features()
    models = fit_heads(X_train, y_train)
    generator = PredictionMatrixGenerator({name: models[name] for name in HEADS})
    scores = generator.fit_transform(X_test)
    names = generator.model_names_
    save_scores(scores, y_test, names)

    splitter = SemiSupervisedSplit(20, 400, random_state=0)
    parts = splitter.indices(len(y_test))
    if splitter.labeled_class_count(y_test, parts) != 2:
        raise RuntimeError("labeled draw missed a class")
    estimator = SSMEEstimator(max_iter=100, early_stopping=False, random_state=0)
    estimator.fit(scores[parts["estimation"]], splitter.partial_labels(y_test, parts), model_names=names)
    report = estimator.report(
        n_draws=100,
        title="PneumoniaMNIST transfer",
        primary_metric="auc",
        sample_ids=parts["estimation"],
    )
    report.save(REPORT)
    print(report.ranking("auc").to_string(index=False))
    print(REPORT / "report.html")


if __name__ == "__main__":
    main()
