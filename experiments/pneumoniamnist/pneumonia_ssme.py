"""Seven PneumoniaMNIST checkpoints, then one SSME-Lite report.

    python experiments/pneumoniamnist/pneumonia_ssme.py

The repeated 10-seed comparison is already at
``experiments/pneumoniamnist/results/nl20_nu400_m7/``.
"""
from pathlib import Path
import sys
from ssme_lite import PredictionMatrixGenerator, SSMEEstimator, SemiSupervisedSplit

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from pneumonia_models import MODELS, build_models, check_against_official, download_weights, load_test_images, save_scores

REPORT = HERE / "results" / "demo_report"


def main():
    images, y = load_test_images()
    models = build_models(download_weights())
    generator = PredictionMatrixGenerator({name: models[name] for name in MODELS})
    scores = generator.fit_transform(images)
    names = generator.model_names_
    check_against_official(models, y)
    save_scores(scores, y, names)

    splitter = SemiSupervisedSplit(20, 400, random_state=0)
    parts = splitter.indices(len(y))
    if splitter.labeled_class_count(y, parts) != 2:
        raise RuntimeError("labeled draw missed a class")
    estimator = SSMEEstimator(max_iter=100, early_stopping=False, random_state=0)
    estimator.fit(scores[parts["estimation"]], splitter.partial_labels(y, parts), model_names=names)
    report = estimator.report(
        n_draws=100,
        title="PneumoniaMNIST",
        primary_metric="auc",
        sample_ids=parts["estimation"],
        contribution=True,
    )
    report.save(REPORT)
    public = HERE / "pneumoniamnist_report.html"
    public.write_bytes((REPORT / "report.html").read_bytes())
    print(report.ranking("auc").to_string(index=False))
    print(public)


if __name__ == "__main__":
    main()
