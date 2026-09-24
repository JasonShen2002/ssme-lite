"""Frozen ImageNet ResNet18 features -> sklearn heads -> SSME evaluation.

Input: ImageFolder directory (one subdirectory per class), >1700 images.
Install optional dependencies: pip install -e '.[transfer]'
First invocation downloads public pretrained weights (~45MB); no images uploaded.
Example: python examples/transfer_resnet.py --images /path/to/images --output results/transfer
"""
import argparse
from pathlib import Path
import numpy as np


def extract_features(images, batch_size=32, device="auto"):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ImageFolder(images, transform=weights.transforms())
    backbone = resnet18(weights=weights)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    backbone.requires_grad_(False)
    backbone.to(device)
    print(f"Feature extraction device: {device}", flush=True)
    features, labels = [], []
    with torch.inference_mode():
        for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            features.append(backbone(x.to(device)).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels), dataset.classes


def main():
    from sklearn.model_selection import train_test_split
    from ssme_lite import PredictionMatrixGenerator
    from ssme_lite.experiments import ten_models, benchmark
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", default="results/transfer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    X, y, classes = extract_features(args.images, device=args.device)
    np.savez_compressed(output / "features.npz", X=X, y=y, classes=np.asarray(classes))
    train, evaluation = train_test_split(np.arange(len(y)), train_size=0.4, random_state=args.seed, stratify=y)
    if len(evaluation) <= 1020:
        raise ValueError("need more images: evaluation partition must exceed 1020 examples")
    models = ten_models(args.seed)
    for model in models.values():
        model.fit(X[train], y[train])
    generator = PredictionMatrixGenerator(models)
    scores = generator.fit_transform(X[evaluation])
    np.savez_compressed(output / "scores.npz", scores=scores, y=y[evaluation],
                        model_names=np.asarray(generator.model_names_), train_indices=train, evaluation_indices=evaluation)
    benchmark(scores, y[evaluation], generator.model_names_, output / "evaluation",
              dataset="resnet18_frozen_features", seeds=[args.seed])


if __name__ == "__main__":
    main()
