"""JSON-configured offline evaluation, fitted-model input and benchmarks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .benchmark import benchmark, gaussian_scores, official_scores, synthetic_scores
from .estimation import SSMEEstimator
from .prediction import PredictionMatrixGenerator

MODEL_TEMPLATE = {
    "source": "models", "mode": "evaluate", "data": "evaluation.npz",
    "models": [{"name": "logistic", "path": "models/logistic.joblib"},
               {"name": "forest", "path": "models/forest.joblib"}],
    "prediction": {"n_jobs": 1, "batch_size": 512},
    "estimator": {"random_state": 0, "max_iter": 100},
    "report": {"n_draws": 100, "target": "unlabeled"}, "output": "report",
}


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(filename):
    filename = Path(filename).resolve()
    config = json.loads(filename.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    def path(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("paths must be nonempty strings")
        return (filename.parent / value).resolve()
    source = config.get("source")
    if source not in ("models", "npz", "synthetic", "synthetic_gaussian", "CivilComments", "MultiNLI"):
        raise ValueError("unknown source; choose models, npz, synthetic, synthetic_gaussian, CivilComments or MultiNLI")
    path(config["output"])
    if config.get("mode", "evaluate") not in ("evaluate", "benchmark"):
        raise ValueError("mode must be evaluate or benchmark")
    if source in ("models", "npz") and not path(config["data"]).is_file():
        raise FileNotFoundError(path(config["data"]))
    if source == "models":
        if config.get("mode", "evaluate") != "evaluate":
            raise ValueError("models source evaluates partial labels; use frozen npz scores for benchmarks")
        entries = config.get("models")
        if not isinstance(entries, list) or not entries:
            raise ValueError("models must be a nonempty list of {name, path} objects")
        names = []
        for item in entries:
            if not isinstance(item, dict) or set(item) != {"name", "path"}:
                raise ValueError("each model needs exactly name and path")
            if not isinstance(item["name"], str) or not item["name"].strip():
                raise ValueError("model names must be nonempty strings")
            names.append(item["name"])
            if not path(item["path"]).is_file():
                raise FileNotFoundError(path(item["path"]))
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")
        options = config.get("prediction", {})
        if not isinstance(options, dict) or set(options) - {"n_jobs", "batch_size", "clip"}:
            raise ValueError("prediction supports n_jobs, batch_size and clip")
        with np.load(path(config["data"]), allow_pickle=False) as data:
            X, y, classes = data["X"], data["y"], data["classes"]
            if X.ndim != 2 or len(X) == 0 or y.shape != (len(X),):
                raise ValueError("evaluation.npz needs X (rows, features) and y (rows,)")
            if classes.ndim != 1 or len(classes) < 2 or len(set(classes.tolist())) != len(classes):
                raise ValueError("classes must be a unique ordered vector with at least two classes")
            if not np.issubdtype(y.dtype, np.number) or not np.isfinite(y).all() or not np.isin(y, np.arange(-1, len(classes))).all():
                raise ValueError("y must contain -1 or class indices into the supplied classes vector")
            if set(y[y >= 0]) != set(range(len(classes))):
                raise ValueError("at least one visible label per class is required")
            if "sample_ids" in data and data["sample_ids"].shape != y.shape:
                raise ValueError("sample_ids must have one entry per row")
    return config, path


def run_config(filename, *, trust_model_files=False):
    config, path = load_config(filename)
    output, source = path(config["output"]), config["source"]
    receipt, sample_ids = None, None
    if source == "models":
        if not trust_model_files:
            raise ValueError("joblib loading can execute code; use --trust-model-files only for your trusted fitted models")
        import joblib
        models = {item["name"]: joblib.load(path(item["path"])) for item in config["models"]}
        with np.load(path(config["data"]), allow_pickle=False) as data:
            X, y, classes = data["X"], data["y"], data["classes"]
            sample_ids = data["sample_ids"] if "sample_ids" in data else None
        generator = PredictionMatrixGenerator(models, **config.get("prediction", {})).fit()
        if not np.array_equal(generator.classes_, classes):
            raise ValueError("classes must exactly match the first model's classes_ order; encode y in that order")
        scores = generator.transform(X, y)
        names = generator.model_names_
        receipt = {"source": "trusted_fitted_models", "classes": classes.tolist(),
                   "data_sha256": _digest(path(config["data"])),
                   "models": [{"name": item["name"], "path": item["path"],
                               "sha256": _digest(path(item["path"]))} for item in config["models"]]}
    elif source == "synthetic":
        scores, y, names = synthetic_scores(config.get("data_seed", 1729))
    elif source == "synthetic_gaussian":
        scores, y, names = gaussian_scores(config.get("data_seed", 1729))
    elif source in ("CivilComments", "MultiNLI"):
        scores, y, names = official_scores(str(path(config["inputs"])), source)
    else:
        with np.load(path(config["data"]), allow_pickle=False) as data:
            scores, y = data["scores"], data["y"]
            names = data["model_names"].tolist() if "model_names" in data else None
    if source == "models" or (source == "npz" and config.get("mode", "evaluate") == "evaluate"):
        estimator = SSMEEstimator(**config.get("estimator", {})).fit(scores, y, model_names=names)
        report_options = dict(config.get("report", {}))
        report_options.setdefault("contribution", config.get("contribution", False))
        if sample_ids is not None:
            report_options.setdefault("sample_ids", sample_ids)
        estimator.report(**report_options).save(output)
        if receipt is not None:
            (output / "input_provenance.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            np.savez_compressed(output / "predictions.npz", scores=scores, y=y,
                                model_names=np.asarray(names), classes=classes)
    else:
        if names is None:
            names = [f"model_{j}" for j in range(scores.shape[1])]
        options = dict(config.get("benchmark", {}))
        dataset = options.pop("dataset", source)
        benchmark(scores, y, names, output, dataset=dataset, **options)
    (output / "input_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return output


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("run", "init-config", "check-config", "-h", "--help"):
        argv.insert(0, "run")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Evaluate JSON (also: ssme-lite config.json)")
    run.add_argument("config")
    run.add_argument("--trust-model-files", action="store_true", help="Allow loading your trusted joblib models")
    init = commands.add_parser("init-config", help="Create a fitted-model input template")
    init.add_argument("config")
    check = commands.add_parser("check-config", help="Check input files without loading models or fitting")
    check.add_argument("config")
    args = parser.parse_args(argv)
    try:
        if args.command == "init-config":
            destination = Path(args.config)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as f:
                f.write(json.dumps(MODEL_TEMPLATE, indent=2) + "\n")
            print(destination)
        elif args.command == "check-config":
            config, _ = load_config(args.config)
            print(f"Input preflight passed: {config['source']}. Models were not loaded; model compatibility is checked during run.")
        else:
            print(run_config(args.config, trust_model_files=args.trust_model_files))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
