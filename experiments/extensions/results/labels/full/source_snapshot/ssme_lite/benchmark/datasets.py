"""Example score matrices and loaders for released model outputs."""
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from ..prediction import PredictionMatrixGenerator, probabilities


def ten_models(seed=0):
    """Ten distinct classifier families; classifier fitting is example-only."""
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=1),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=80, random_state=seed),
        "adaboost": AdaBoostClassifier(n_estimators=60, random_state=seed),
        "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=seed),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=21)),
        "gaussian_nb": GaussianNB(),
        "lda": LinearDiscriminantAnalysis(),
        "qda": QuadraticDiscriminantAnalysis(reg_param=0.1),
    }


def synthetic_scores(seed=1729):
    X, y = make_classification(n_samples=7000, n_features=16, n_informative=8,
                               n_redundant=3, class_sep=1.3, flip_y=0.05, random_state=seed)
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, train_size=2000, random_state=seed, stratify=y)
    models = ten_models(seed)
    for model in models.values():
        model.fit(X_train, y_train)
    generator = PredictionMatrixGenerator(models)
    scores = generator.fit_transform(X_eval)
    return probabilities(scores), y_eval, generator.model_names_


def gaussian_scores(seed=1729):
    """Controlled binary task: independent Gaussian score views given the label.

    Predeclared signal strengths, no selection based on estimator performance.
    This diagnostic is distinct from the ten trained sklearn-family benchmark.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 5000)
    strength = np.linspace(0.5, 2.0, 10)
    X = rng.normal(size=(len(y), len(strength))) + (2*y[:, None]-1)*strength
    class ScoreModel:
        classes_ = np.array([0, 1])
        def __init__(self, column):
            self.column = column
        def predict_proba(self, features):
            positive = expit(features[:, self.column])
            return np.column_stack([1-positive, positive])
    models = {f"gaussian_view_{j}": ScoreModel(j) for j in range(len(strength))}
    generator = PredictionMatrixGenerator(models)
    return probabilities(generator.fit_transform(X)), y, generator.model_names_


def official_scores(root, dataset="CivilComments"):
    root = Path(root) / dataset
    if dataset == "CivilComments":
        names = ["alg_CORAL", "alg_ERM", "alg_IRM", "alg_ERM_seed1", "alg_ERM_seed2", "alg_IRM_seed1", "alg_IRM_seed2"]
    elif dataset == "MultiNLI":
        names = ["alg_ReWeight", "alg_SqrtReWeight", "alg_IRM", "alg_ReSample"]
    else:
        raise ValueError("official dataset must be CivilComments or MultiNLI")
    scores, labels = [], None
    for name in names:
        y = np.load(root / name / "labels.npy", allow_pickle=False).reshape(-1)
        if labels is not None and not np.array_equal(y, labels):
            raise ValueError("official model labels are not row-aligned")
        labels = y
        scores.append(np.load(root / name / "preds.npy", allow_pickle=False))
    return probabilities(np.stack(scores, axis=1)), labels, names


