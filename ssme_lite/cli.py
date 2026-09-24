"""JSON-configured offline evaluation and reproducible benchmarks."""
import argparse
import json
from pathlib import Path
import numpy as np
from .estimator import SSMEEstimator
from .experiments import benchmark, synthetic_scores, gaussian_scores, official_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="JSON configuration path")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    def path(value):
        return str((config_path.parent / value).resolve())
    output = path(config["output"])
    source = config["source"]
    if source == "synthetic":
        scores, y, names = synthetic_scores(config.get("data_seed", 1729))
    elif source == "synthetic_gaussian":
        scores, y, names = gaussian_scores(config.get("data_seed", 1729))
    elif source in ("CivilComments", "MultiNLI"):
        scores, y, names = official_scores(path(config["inputs"]), source)
    elif source == "npz":
        with np.load(path(config["data"]), allow_pickle=False) as data:
            scores, y = data["scores"], data["y"]
            names = data["model_names"].tolist() if "model_names" in data else None
        if config.get("mode", "evaluate") == "evaluate":
            estimator = SSMEEstimator(**config.get("estimator", {})).fit(scores, y, model_names=names)
            report_options = dict(config.get("report", {}))
            report_options.setdefault("contribution", config.get("contribution", False))
            estimator.report(**report_options).save(output)
            (Path(output) / "input_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            print(output)
            return
        if names is None:
            names = [f"model_{j}" for j in range(scores.shape[1])]
    else:
        raise ValueError("source must be synthetic, synthetic_gaussian, CivilComments, MultiNLI or npz")
    benchmark_config = dict(config.get("benchmark", {}))
    dataset = benchmark_config.pop("dataset", source)
    benchmark(scores, y, names, output, dataset=dataset, **benchmark_config)
    (Path(output) / "input_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
