"""Train on one split, then evaluate separate samples with partial labels."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from ssme_lite import PredictionMatrixGenerator, SemiSupervisedSplit, SSMEEstimator


def main():
    X, y = make_classification(n_samples=600, n_features=8, n_informative=5,
                               random_state=11)
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=11)
    models = {
        "logistic": LogisticRegression(max_iter=500).fit(X_train, y_train),
        "forest": RandomForestClassifier(n_estimators=40, max_depth=5,
                                          random_state=11).fit(X_train, y_train),
    }
    generator = PredictionMatrixGenerator(models)
    scores = generator.fit_transform(X_eval)
    splitter = SemiSupervisedSplit(20, 180, random_state=11)
    parts = splitter.indices(len(y_eval))
    partial = splitter.partial_labels(y_eval, parts)
    if set(partial[partial >= 0]) != {0, 1}:
        raise RuntimeError("Observed sample must include both classes")
    estimator = SSMEEstimator(max_iter=20, random_state=11).fit(
        scores[parts["estimation"]], partial, model_names=generator.model_names_)
    report = estimator.report(n_draws=30, contribution=True)
    print(report.ranking("accuracy")[["model", "estimate", "rank"]].to_string(index=False))
    print(report.save("results/sklearn_models"))
    assert not np.intersect1d(parts["estimation"], parts["heldout"]).size


if __name__ == "__main__":
    main()
