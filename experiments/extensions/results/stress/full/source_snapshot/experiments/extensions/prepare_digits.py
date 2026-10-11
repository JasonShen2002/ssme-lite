"""Freeze multiclass probabilities from train-only sklearn classifiers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits
from ssme_lite import PredictionMatrixGenerator
from experiments.extensions.common import ROOT, environment, digest, write_json


def main():
    output = ROOT / 'experiments/extensions/inputs'
    output.mkdir(parents=True, exist_ok=True)
    X, y = load_digits(return_X_y=True)
    receipt = dict(environment=environment(), dataset='sklearn bundled digits', seed=2718, datasets=[])
    with threadpool_limits(limits=1):
        for k in (3, 10):
            rows = np.flatnonzero(y < k)
            train, test = train_test_split(rows, test_size=.5, stratify=y[rows], random_state=2718)
            models = {
                'logistic': make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000)),
                'forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=2718, n_jobs=1),
                'naive_bayes': GaussianNB(var_smoothing=1e-3),
                'extra_trees': ExtraTreesClassifier(n_estimators=100, max_depth=8, random_state=2718, n_jobs=1),
                'knn': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9)),
                'hist_gradient': HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, random_state=2718),
            }
            for model in models.values():
                model.fit(X[train], y[train])
            generator = PredictionMatrixGenerator(models)
            p = generator.fit_transform(X[test])
            destination = output / f'digits{k}.npz'
            np.savez_compressed(destination, scores=p, y=y[test], model_names=np.array(list(models)), sample_ids=test)
            receipt['datasets'].append(dict(classes=k, train_indices=train.tolist(), eval_indices=test.tolist(),
                                             training_rows=len(train), evaluation_rows=len(test),
                                             models={name: repr(m) for name, m in models.items()},
                                             file=destination.name, sha256=digest(destination)))
            print(f'Frozen digits{k}: {p.shape}', flush=True)
    write_json(output / 'digits_manifest.json', receipt)


if __name__ == '__main__':
    main()
